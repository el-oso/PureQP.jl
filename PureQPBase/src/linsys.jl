"""
    BackendInfo

What a [`LinearSystem`](@ref) backend is and how big the object it solves through is,
reported by [`backend_info`](@ref).

- `name` — the backend's name, the same symbol [`backend_name`](@ref) returns.
- `direct` — `true` when a factorization is stored, `false` for a matrix-free iterative
  backend, whose `factor_nnz` is `0` because there is nothing stored to count.
- `system` — `:reduced` for the `n×n` system that eliminates `ν`, `:kkt` for the full
  `(n+m)×(n+m)` quasi-definite system.
- `dim` — the side length of that system: `n` when `system` is `:reduced`, `n+m` when it is
  `:kkt`.
- `factor_nnz` — the size of the stored factorization as one triangle, in that factorization's
  own convention: `nnz(L)` for a sparse factor, `dim(dim+1)/2` for a dense one, and the
  inverse's triangle for a backend that stores an inverse instead.

It is a fill measure, not a memory total, and the conventions differ in ways that matter when
comparing two backends directly: an `LDLᵀ` factor's `L` is strictly lower with a unit diagonal
held elsewhere, so it counts `dim` fewer scalars than an `LLᵀ` factor of the same matrix, and a
backend may physically hold more than it reports — `SparseCholmod` keeps `L` and `Lᵀ` both,
`FullKKT` keeps a pivot vector beside its factor.

Comparing fills across backends needs the problem's `n` rather than `dim`, because `dim` is
`n+m` for a `:kkt` backend and `n` for a `:reduced` one. [`factor_fill`](@ref) takes a
workspace and does that normalization; it is the quantity the sparse selection thresholds are
stated in.

The sparse backends' factors are empty until `factorize!` has run. [`setup`](@ref) always
factorizes, so a backend reached through a workspace is populated.
"""
struct BackendInfo
    name::Symbol
    direct::Bool
    system::Symbol
    dim::Int
    factor_nnz::Int
end

"""
    backend_info(ls::LinearSystem) -> BackendInfo

Describe the backend a workspace holds: `backend_info(ws.linsys)`.
"""
function backend_info end

"""
    factor_fill(ws) -> Float64

The stored factorization's size as a fraction of `n²`, which is how the sparse selection
thresholds are stated and the only form in which two backends' fills compare.

`BackendInfo`'s own `dim` is `n+m` for a backend solving the full KKT system and `n` for one
solving the reduced system, so normalizing by `dim²` would divide the two families by
different denominators. This takes `n` from the workspace instead.
"""
factor_fill(ws) = backend_info(ws.linsys).factor_nnz / ws.prob.n^2

"""
    LinearSystem

Interface for the factorization that solves

    ⎡P̃ + σI        Ãᵀ     ⎤ ⎡x⎤   ⎡rhs_x⎤
    ⎣Ã       −diag(w_inv)⎦ ⎣ν⎦ = ⎣rhs_z⎦

for the [`Problem`](@ref) and the [`SystemWeights`](@ref) it is handed; ADMM solves it once
per iteration with `w = ρ`. A backend owns its own storage and factorization object; the
workspace holds one, chosen at [`setup`](@ref) and fixed for the workspace's life, so every
call dispatches statically.

Implementations must provide the three methods below; the contract is enforced at
precompilation. A method leaves the `Problem` and `SystemWeights` slots unannotated (or
annotates them with exactly those types) and annotates its return.
[`refactor_weights!`](@ref), [`solve_multiplier!`](@ref), [`check_update`](@ref),
[`set_tolerance_level!`](@ref), [`set_refresh_index!`](@ref), [`adopt_settings!`](@ref),
[`use_residual_stop!`](@ref), [`last_solve_converged`](@ref) and [`inner_iterations`](@ref)
are optional, each with a default for every backend. `TypeContracts.describe(LinearSystem)`
lists the whole interface.
"""
abstract type LinearSystem end

"""
    factorize!(ls, prob, wt) -> Bool

Rebuild the factorization of the system `prob` and `wt` define. `false` means this backend
cannot factor it (not positive definite for a reduced backend; a zero pivot for a KKT one).
Reads `prob.P A D E c n m` and `wt`; may use `prob.work_n` and `prob.work_m` as scratch.
"""
function factorize! end

"""
    solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z) -> Nothing

Solve for `x` and write `z = Ã x` as this backend computes it: a reduced backend forms the
product, a KKT backend recovers it from the eliminated multiplier as `rhs_z + w_inv ⊙ ν`.
None of `rhs_x rhs_z x z` may alias each other or `prob.work_n`, `prob.work_m`, `prob.tmp_n`,
`prob.tmp_m` ([`reduced_rhs!`](@ref) writes `work_n` and `work_m`; the products use `tmp_*`).
"""
function solve_system! end

# A StrictMode contract: the solves run once per iteration and are held to allocation-free,
# type-stable, `--trim` compatible code on top of the method surface TypeContracts checks.
@strict_contract LinearSystem begin
    factorize!(::Self, ::Problem, ::SystemWeights)::Bool
    solve_system!(::Self, ::Problem, ::SystemWeights, ::Any, ::Any, ::Any, ::Any)::Nothing
    backend_info(::Self)::BackendInfo
    # Every optional method has a default for `LinearSystem`, so a backend overrides only
    # what it needs.
    :optional
    refactor_weights!(::Self, ::Problem, ::SystemWeights)::Bool => "refresh after only the weights changed"
    solve_multiplier!(::Self, ::Problem, ::SystemWeights, ::Any, ::Any, ::Any, ::Any)::Nothing => "solve for `x` and the multiplier `ν`"
    check_update(::Self, ::Any, ::Any)::Nothing => "throw unless the backend can serve the replacement `P` and `A`"
    set_tolerance_level!(::Self, ::Any)::Nothing => "the residual level an inexact backend's next solves are relative to"
    set_refresh_index!(::Self, ::Int)::Nothing => "the refresh index the next preconditioner update receives"
    adopt_settings!(::Self, ::QPAlgorithm, ::Options)::Nothing => "copy in the algorithm parameters and options the backend reads"
    use_residual_stop!(::Self, ::Bool)::Nothing => "choose an iterative backend's inner stopping rule"
    last_solve_converged(::Self)::Bool => "whether the most recent solve met its stopping test"
    inner_iterations(::Self)::Int => "inner iterations spent over the backend's life"
end

"""
    refactor_weights!(ls, prob, wt) -> Bool

Refresh the factorization after only the contents of `wt.w` and `wt.w_inv` changed since the
last [`factorize!`](@ref), returning whether it succeeded.

Separate from `factorize!` because the two events are not the same: `ρ` changes on its own
every time the algorithm retunes it, while `P`, `A`, `D`, `E`, `c` and `σ` change only
through [`setup`](@ref), [`update!`](@ref) and [`update_settings!`](@ref). A backend whose
factorization is partly independent of the weights can keep that part.

Rebuilding everything is correct, and is what the default does. An override may assume the
weight-independent parts are current, since every path that invalidates them calls
`factorize!` instead.
"""
refactor_weights!(ls::LinearSystem, prob, wt) = factorize!(ls, prob, wt)

"""
    solve_multiplier!(ls, prob, wt, rhs_x, rhs_z, x, nu) -> Nothing

Solve the same system as [`solve_system!`](@ref) for `x` and the eliminated multiplier `ν`
directly, instead of for `x` and `z̃ = Ã x`. Row two of the system reads
`Ã x − w_inv ⊙ ν = rhs_z`, so `ν = w ⊙ (z̃ − rhs_z)`, which is what the default computes from
[`solve_system!`](@ref)'s own output. That subtraction cancels as `w_inv → 0` — `z̃` approaches
`rhs_z` in floating point before `ν` does — so a backend that already holds `ν` as part of an
augmented solve (a KKT backend) overrides this to return it directly, unaffected by `w_inv`'s
magnitude.

None of `rhs_x rhs_z x nu` may alias each other or `prob.work_n`, `prob.work_m`, `prob.tmp_n`,
`prob.tmp_m`, as for [`solve_system!`](@ref).
"""
function solve_multiplier!(ls::LinearSystem, prob, wt, rhs_x, rhs_z, x, nu)
    solve_system!(ls, prob, wt, rhs_x, rhs_z, x, nu)      # nu holds z̃ for a moment
    subtract!(nu, nu, rhs_z)
    scale_by!(nu, wt.w)
    return nothing
end

"""
    check_update(ls, P, A) -> Nothing

Throw unless the backend can go on serving the matrices `P` and `A` that [`update!`](@ref) is
about to adopt. Called with the matrices the workspace will hold afterwards, whenever either
is being replaced, before anything is adopted.

The default accepts: a backend that reads only what the representation implies needs nothing
beyond the type check `update!` already makes. A structured backend whose storage or
factorization depends on data-level invariants — a partition, a rank, a scalar `P` —
overrides this.
"""
check_update(ls::LinearSystem, P, A) = nothing

"""
    set_tolerance_level!(ls, level) -> Nothing

Hand an inexact backend the residual level its next solves are to be accurate relative to.
A direct backend solves exactly and ignores it, which is the default.
"""
set_tolerance_level!(ls::LinearSystem, level) = nothing

"""
    adopt_settings!(ls, alg, options) -> Nothing

Copy into the backend whatever of the workspace's algorithm parameters `alg` and
[`Options`](@ref) it reads while solving. Called once the workspace is built and whenever
either is replaced. The default backend reads none, and does nothing.
"""
adopt_settings!(ls::LinearSystem, alg, options) = nothing

"""
    ReducedInverse <: LinearSystem

Backends that eliminate `ν` and solve the reduced `n×n` system by multiplying against its
stored inverse.

They differ only in how the reduced matrix is *formed*, which is where the representation
of `A` matters; once formed, the factorization, the inversion and the per-iteration `symv`
are the same work regardless. `solve_system!` is therefore written once, here.
"""
abstract type ReducedInverse <: LinearSystem end

"""
    ReducedCholesky{T,M} <: ReducedInverse

Eliminates `ν` and solves the `n×n` symmetric positive definite reduced system

    (P̃ + σI + Ãᵀ diag(ρ) Ã) x̃ = rhs_x + Ãᵀ(ρ ⊙ rhs_z),   z̃ = Ã x̃

`W` holds the scaled `Ã` with `sqrt(ρ)` folded in, so the reduced matrix is one `syrk`.
`Rinv` holds that matrix's inverse in its upper triangle, so each iteration's solve is a
single `symv` rather than the two triangular solves of a Cholesky `ldiv!`. Both cost `2n²`
flops, but a triangular solve computes its entries in sequence while `symv` does not, which
makes it about seven times faster at `n = 200`. Inverting is sound here because the reduced
matrix carries the `σI` regularization: its conditioning is bounded, and the residuals of
the two forms agree to within a small factor.

This is the default: the reduced matrix is smaller than the full system for every `m`, and
measurement puts it faster in every dense regime.
"""
struct ReducedCholesky{T <: Real, M <: AbstractMatrix{T}} <: ReducedInverse
    W::M
    Rinv::M
end

"""
    ReducedCholesky(proto::AbstractVector, n, m)

Build the backend's storage as `similar(proto, ...)`, so it follows the array type of the
data it was given rather than always being a `Matrix`. `proto` is a dense vector, not one
of the problem matrices: both buffers are dense even when `P` and `A` are not.
"""
function ReducedCholesky(proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    W = similar(proto, T, m, n)
    return ReducedCholesky{T, typeof(W)}(W, similar(proto, T, n, n))
end

"""
    FullKKT{T,M,V,F} <: LinearSystem

Factors the full `(n+m)×(n+m)` quasi-definite matrix with `bunchkaufman!`, which is what
the reference implementation does. Slower than [`ReducedCholesky`](@ref), but it does not
square the conditioning of `Ã`, so it is the more accurate factorization at moderate
conditioning and the better choice when a result is in question.

`K0` caches the lower triangle of the scaled `P` and `A` blocks — the part of `K` that
depends on `P`, `A` and the equilibration `D`, `E`, `c` but not on the weights `wt`.
`k0_current` is `false` until [`assemble_kkt0!`](@ref) has filled it for the data currently
in `prob`, and is cleared by [`check_update`](@ref) whenever `P` or `A` is about to change.
`bunchkaufman!` reads only the lower triangle of `Symmetric(K, :L)`, so `K0`'s upper
triangle and its `m×m` diagonal block — entirely the weights' `-w_inv`, never `P` or `A` —
are never written past the zero the constructor sets once.

For a LAPACK element type the factorization is LAPACK's `sytrf` into `ipiv` and `work`,
which the backend holds and sized once, so a refactorization allocates nothing. It is the
call `bunchkaufman!` makes, with the same workspace size, so the factor is the same. `fact`
is then built once over `K` and `ipiv` and never replaced, and its `info` field is not
maintained: [`factorize!`](@ref)'s return value is what reports a failed factorization.
"""
mutable struct FullKKT{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F} <: LinearSystem
    const K::M
    const K0::M
    const rhs::V
    const ipiv::Vector{LinearAlgebra.BlasInt}
    const work::Vector{T}
    fact::F
    k0_current::Bool
end

"""
    FullKKT(proto::AbstractVector, n, m)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data
it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector. `K0` starts zeroed
and `k0_current = false`, so the first [`factorize!`](@ref) assembles it.
"""
function FullKKT(proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    K = similar(proto, T, n + m, n + m)
    K0 = fill!(similar(K), zero(T))
    rhs = similar(proto, T, n + m)
    ipiv = zeros(LinearAlgebra.BlasInt, n + m)
    work = Vector{T}(undef, sytrf_lwork(T, n + m))
    fact = kkt_factorization(K, ipiv)
    return FullKKT{T, typeof(K), typeof(rhs), typeof(fact)}(K, K0, rhs, ipiv, work, fact, false)
end

"""
    sytrf_lwork(T, dim) -> Int

The workspace length LAPACK's `sytrf` asks for on a `dim×dim` matrix of element type `T`;
zero for a type LAPACK has no method for.
"""
sytrf_lwork(::Type, dim::Integer) = 0

"""
    sytrf_lower!(A, ipiv, work) -> info

LAPACK's `sytrf` on the lower triangle of `A`, pivoting into `ipiv` and using `work`, which
must be at least [`sytrf_lwork`](@ref) long. Nothing is allocated.
"""
function sytrf_lower! end

for (fname, elty) in ((:dsytrf_, :Float64), (:ssytrf_, :Float32))
    @eval begin
        function sytrf_lwork(::Type{$elty}, dim::Integer)
            iszero(dim) && return 0
            work = Vector{$elty}(undef, 1)
            info = Ref{LinearAlgebra.BlasInt}()
            ccall(
                (LinearAlgebra.BLAS.@blasfunc($fname), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                (
                    Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ptr{$elty}, Ref{LinearAlgebra.BlasInt},
                    Ptr{LinearAlgebra.BlasInt}, Ptr{$elty}, Ref{LinearAlgebra.BlasInt},
                    Ref{LinearAlgebra.BlasInt}, Clong,
                ),
                'L', dim, C_NULL, max(1, dim), C_NULL, work, -1, info, 1
            )
            return Int(real(work[1]))
        end

        function sytrf_lower!(
                A::StridedMatrix{$elty}, ipiv::Vector{LinearAlgebra.BlasInt}, work::Vector{$elty}
            )
            LinearAlgebra.chkstride1(A)
            dim = LinearAlgebra.checksquare(A)
            length(ipiv) >= dim || throw(DimensionMismatch("ipiv is shorter than A"))
            iszero(dim) && return LinearAlgebra.BlasInt(0)
            info = Ref{LinearAlgebra.BlasInt}()
            ccall(
                (LinearAlgebra.BLAS.@blasfunc($fname), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                (
                    Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ptr{$elty}, Ref{LinearAlgebra.BlasInt},
                    Ptr{LinearAlgebra.BlasInt}, Ptr{$elty}, Ref{LinearAlgebra.BlasInt},
                    Ref{LinearAlgebra.BlasInt}, Clong,
                ),
                'L', dim, A, max(1, stride(A, 2)), ipiv, work, length(work), info, 1
            )
            return info[]
        end
    end
end

# `P` or `A` is about to change, so the cached scaled lower triangle no longer reflects the
# data it will be factored against.
check_update(ls::FullKKT, P, A) = (ls.k0_current = false; nothing)

"""
    DiagonalReduced{T,V} <: LinearSystem

The reduced system when it is diagonal, which it is when `P` and `A` both are:

    R = c D P D + σI + Ãᵀ diag(ρ) Ã

is a sum of diagonal terms, so there is nothing to factor and each solve is `n` divisions.
`dinv` holds `R`'s reciprocal diagonal.

`Ãᵀ diag(ρ) Ã` fills in for any other `A`, which is why this is keyed on `A`'s type and not
`P`'s: a `Diagonal` `P` with a general `A` still has a dense reduced matrix.
"""
struct DiagonalReduced{T <: Real, V <: AbstractVector{T}} <: LinearSystem
    dinv::V
end

"""
    DiagonalReduced(proto::AbstractVector, n)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data
it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function DiagonalReduced(proto::AbstractVector{T}, n::Integer) where {T <: Real}
    dinv = similar(proto, T, n)
    return DiagonalReduced{T, typeof(dinv)}(dinv)
end

"""
    TridiagonalReduced{T,V,F} <: LinearSystem

The reduced system when it is tridiagonal. Diagonal scaling preserves a bandwidth and
`Ãᵀ diag(ρ) Ã` doubles `A`'s, so

    bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))

which is 1 for a `SymTridiagonal` or `Tridiagonal` `P` with a `Diagonal` `A`, for a
`Diagonal` `P` with a `Bidiagonal` `A`, and for the two together. `ldlt` factors that in
`O(n)` and its `ldiv!` allocates nothing.

`dv` and `ev` hold `R`'s two bands. They are computed entry by entry rather than by forming
`c D P D + σI + Ãᵀ diag(ρ) Ã`: that product returns a dense `Array` for a `Bidiagonal` `A`
even though the result has bandwidth 1, so the structure has to be established here rather
than recovered from the arithmetic. `fdv` and `fev` are the copies `ldlt!` overwrites, which
keeps a refactorization from allocating.
"""
mutable struct TridiagonalReduced{T <: Real, V <: AbstractVector{T}, F} <: LinearSystem
    dv::V
    ev::V
    fdv::V
    fev::V
    fact::F
end

"""
    TridiagonalReduced(proto::AbstractVector, n)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data
it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function TridiagonalReduced(proto::AbstractVector{T}, n::Integer) where {T <: Real}
    dv, ev = similar(proto, T, n), similar(proto, T, max(n - 1, 0))
    fdv, fev = similar(dv), similar(ev)
    fact = ldlt!(SymTridiagonal(fill(one(T), 1), fill(one(T), 0)))
    return TridiagonalReduced{T, typeof(dv), typeof(fact)}(dv, ev, fdv, fev, fact)
end

"""
    band_columns(A, k) -> UnitRange

The columns row `k` of `A` can hold a nonzero in. Each structured `A` the tridiagonal
backend accepts answers this in `O(1)`, which is what keeps forming `Ãᵀ diag(ρ) Ã` linear.
"""
band_columns(A::Diagonal, k::Integer) = k:k
band_columns(A::Bidiagonal, k::Integer) =
    A.uplo == 'U' ? (k:min(k + 1, size(A, 2))) : (max(k - 1, 1):k)

"""
    is_convex(T, P, sigma) -> Bool

Whether `P + σI` is positive definite, which is what OSQP requires of `P` — not merely that
it be positive semidefinite. A positive semidefinite `P` always passes; only an indefinite
one fails. Without the check the reduced matrix `P + σI + Ãᵀ diag(ρ) Ã` can still factor and
an indefinite `P` would be accepted silently.

The generic method densifies, because a factorization it can rely on for an arbitrary
`AbstractMatrix` is the dense one. That is `O(n³)` and `O(n²)` in memory whatever `P` was,
so a representation with a cheaper test overrides this — `PureQPBase/ext/PureQPBaseSparseArraysExt.jl`
does, where the dense test measures 93× slower at `n = 2000`.
"""
function is_convex(::Type{T}, P::AbstractMatrix, sigma) where {T}
    isempty(P) && return true
    return issuccess(cholesky!(Symmetric(Matrix{T}(P) + sigma * I); check = false))
end

# A diagonal matrix is positive definite exactly when its diagonal is, so the test is a
# pass over `n` entries rather than a factorization of an `n×n` densification of them.
is_convex(::Type{T}, P::Diagonal, sigma) where {T} = all(d -> d + sigma > zero(T), P.diag)

# `ldlt` of a tridiagonal is `O(n)` and its pivots decide definiteness: all positive is
# positive definite, and a zero pivot throws rather than reporting.
function is_convex(::Type{T}, P::SymTridiagonal, sigma) where {T}
    isempty(P) && return true
    S = SymTridiagonal(P.dv .+ sigma, copy(P.ev))
    fact = try
        ldlt!(S)
    catch e
        e isa LinearAlgebra.ZeroPivotException || rethrow()
        return false
    end
    return all(>(zero(T)), fact.data.dv)
end

# `validate` has established that `P` is symmetric before this runs, so a `Tridiagonal`
# describes the same band as the `SymTridiagonal` built from its diagonal and superdiagonal,
# and gets the same `O(n)` test rather than the generic densification.
is_convex(::Type{T}, P::Tridiagonal, sigma) where {T} =
    is_convex(T, SymTridiagonal(diag(P), diag(P, 1)), sigma)

"""
    is_materializable(M) -> Bool

Whether `M`'s entries can be read one at a time. True unless the representation says
otherwise.

Forming the reduced matrix, polishing and the derivatives all read entries; an
operator that supplies only `mul!` declares `false` here and is refused by those paths by
name rather than by a `MethodError` from inside a factorization. Declining is a statement
by the operator's author about what it can answer, not a measured threshold.

The method body is a literal, so a call against a type with no override folds away and the
rungs that consult it stay concretely typed.
"""
is_materializable(M) = true

"""
    require_entries(P, A, what, remedy)

Throw unless both operators can be read entry by entry, naming what needs it.

Polishing and the derivatives copy `P` and `A` into a dense matrix one entry at a
time and factor it, which an operator supplying only products cannot serve. Without this
the caller gets a `MethodError` from inside the copy, which says nothing about what to do.

The message names no type. Interpolating one goes through `show(::IO, ::Type)`, which is a
runtime dispatch that `--trim` cannot resolve — and for an operator that declines, the
condition folds to `false`, so this branch is live code rather than the dead one it is for
every materializable pair.
"""
function require_entries(P, A, what::String, remedy::String)
    (is_materializable(P) && is_materializable(A)) || throw(
        ArgumentError(
            "$what reads the entries of P and A one at a time, and one of them declares " *
                "`PureQPBase.is_materializable` false: it supplies products only. $remedy"
        )
    )
    return nothing
end

"""
    SelectionFor

Which algorithm [`select_backend`](@ref) and the ladder rungs are choosing a backend for.
A rung whose choice does not depend on the algorithm defines one method, taking any
subtype; a rung whose choice differs adds a method for the specific subtype that needs the
different answer.

Four selection points have no algorithm-independent answer, so a new subtype must define
each of them before selection can serve it: [`select_backend`](@ref), which fixes the order
of the ladder; [`dense_rung`](@ref), its terminal; [`indirect_rung`](@ref), what sits below
the terminal; and, when the sparse rungs are in the ladder, the SparseArrays extension's
`sparse_form`, the rule that reads a sparsity pattern. Each throws through
[`refuse_selection`](@ref) until it is defined. Everything else — [`choose_backend`](@ref),
[`kkt_rung`](@ref), [`reduced_rung`](@ref), [`kronecker_rung`](@ref), [`block_rung`](@ref),
[`lowrank_rung`](@ref), [`formed_rung`](@ref) and the refusals the GPU extension raises —
takes any subtype already.
"""
abstract type SelectionFor end

"""
    refuse_selection(what, sel)

Throw, naming the selection method `sel`'s algorithm still has to define.

`what` is that method's qualified name. This is what a selection point whose answer differs
by algorithm does for a [`SelectionFor`](@ref) it has no method for, in place of the
`MethodError` from a call that looks unrelated to the algorithm being added.
"""
@noinline function refuse_selection(what::String, sel::SelectionFor)
    throw(
        ArgumentError(
            lazy"selecting a backend for $(nameof(typeof(sel))) needs a `$what` method and there is none. A rung whose choice does not depend on the algorithm already serves every SelectionFor by declining; this one's answer differs by algorithm, so define `$what` for $(nameof(typeof(sel)))."
        )
    )
end

"""
    ADMMSelection <: SelectionFor

Selecting a backend for the ADMM iteration, whose reduced system is
`P̃ + σI + Ãᵀ diag(ρ) Ã`. [`setup`](@ref) threads one instance through [`choose_backend`](@ref)
and every rung it reaches.
"""
struct ADMMSelection <: SelectionFor end

"""
    IPMSelection <: SelectionFor

Selecting a backend for the interior-point method's Newton system,
`P̃ + δ_p I + Ãᵀ diag(w) Ã` in reduced form, whose weights differ from row to row and reach
`1/δ_d`. Its ladder is its own [`select_backend`](@ref) method; the rungs whose choice does
not depend on the algorithm serve it unchanged.
"""
struct IPMSelection <: SelectionFor end

"""
    choose_backend(P, A, prob, wt, sel) -> (LinearSystem, Bool)

The backend `linsys = :auto` builds for these matrices, and whether it already carries a
factorization of the current data.

Dispatching on `typeof(P)` and `typeof(A)` is the point: a representation that admits a
cheaper way to form the reduced matrix is served by adding a method here rather than by
branching inside `factorize!`. The choice is made once, and the backend then becomes part
of the workspace's type, so the per-iteration solve still dispatches statically. `P` and `A`
stay explicit arguments, alongside `prob` that also holds them, because dispatch on them is
the point.

`prob` and `wt` are passed in because a method may build its factorization here, with the
values the solver will actually use, in which case that factorization is the setup
factorization: such a method returns `true` and [`setup`](@ref) does not factor again. A
method that only picks a representation returns `false`.

A `(P, A)` pair with no method of its own descends [`select_backend`](@ref)'s ladder, whose
named terminal rung is the dense reduced matrix.
"""
choose_backend(P, A, prob, wt, sel::SelectionFor) = select_backend(P, A, prob, wt, sel)

"""
    select_backend(P, A, prob, wt, sel::ADMMSelection) -> (LinearSystem, Bool)

Descend the selection ladder, returning the first rung that serves this `(P, A)` pair and
whether that rung already carries a factorization of the current data.

The rungs, in order:

1. [`kkt_rung`](@ref) — factor the full `(n+m)×(n+m)` quasi-definite matrix sparsely.
2. [`reduced_rung`](@ref) — factor the `n×n` reduced matrix sparsely.
3. [`block_rung`](@ref) — the reduced matrix decouples into independent blocks, solved one
   at a time and never assembled whole.
4. [`lowrank_rung`](@ref) — the reduced matrix is a structured core plus a low-rank
   correction, solved through the correction rather than formed.
5. [`formed_rung`](@ref) — assemble the reduced matrix from stored entries and invert it
   densely.
6. [`dense_rung`](@ref) — form the reduced matrix densely and invert it. Every pair that can
   be materialized at all stops here.
7. [`indirect_rung`](@ref) — matrix-free, for an operator the terminal cannot materialize.

Rungs 1 and 2 decide from the sparsity pattern and then build what they chose, so the
factorization they produce is the setup factorization and they return `true`. That is why a
rung returns the backend rather than a verdict: a query answering only "does this fit" would
throw that factorization away and pay for it twice.

This ladder is not the whole of selection, and reading it alone will mislead. Three things sit
outside it:

- `linsys = :kkt`, `:dense`, `:indirect` and the named kinds (`:sparse`, `:diagonal`,
  `:tridiagonal`, `:block`, `:kronecker`, `:lowrank`) are handled in [`setup`](@ref) before
  the ladder is reached, so a caller who names a backend never descends it. `:sparse`
  descends rungs 1, 2 and 5 only, and the named kinds reach their own rung or the
  [`choose_backend`](@ref) method for their pair rather than the whole ladder. `:indirect`
  in particular reaches [`indirect_backend`](@ref) directly and not through rung 5.
- A [`choose_backend`](@ref) method for a specific `(P, A)` pair wins over this ladder by
  dispatch, which is how the structured and banded backends are chosen. The ladder is the
  body of the *fallback* method.
- Any backend that arrives unfactored and whose `factorize!` then fails — an indefinite `P`
  that `σ` does not lift, whichever rung or `choose_backend` method produced it — is rebuilt
  on [`FullKKT`](@ref) by [`setup`](@ref). That is the last word on selection, and it is not
  a rung.

Each rung is a generic function whose default declines, so an extension adds itself to the
ladder by defining the method its representation needs. The order is fixed here, in one
place, rather than emerging from where each gate happens to sit.
"""
select_backend(P, A, prob, wt, sel::SelectionFor) =
    refuse_selection("PureQPBase.select_backend", sel)

function select_backend(P, A, prob, wt, sel::ADMMSelection)
    rung = kkt_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = reduced_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = kronecker_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = block_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = lowrank_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = formed_rung(P, A, prob, sel)
    isnothing(rung) || return rung
    rung = dense_rung(P, A, prob, sel)
    isnothing(rung) || return rung
    return indirect_rung(P, A, prob, sel)
end

"""
    kkt_rung(P, A, prob, wt, sel; gated = true) -> (LinearSystem, Bool) or nothing

Ladder rung 1: factor the full quasi-definite KKT matrix sparsely, when the sparsity pattern
says this is the form to use. What it returns is already factored.

`gated = false` skips that question and builds the backend for any pair whose representation
admits it, which is what a caller who names `linsys = :sparse` gets: the named kind is an
instruction rather than a hint, and only a representation mismatch or a genuine factorization
failure can still refuse the pair.
"""
kkt_rung(P, A, prob, wt, sel::SelectionFor; gated::Bool = true) = nothing

"""
    reduced_rung(P, A, prob, wt, sel; gated = true) -> (LinearSystem, Bool) or nothing

Ladder rung 2: factor the reduced matrix sparsely, when the sparsity pattern says this is the
form to use. What it returns is already factored. See [`kkt_rung`](@ref) on what
`gated = false` does for a named `linsys = :sparse`.
"""
reduced_rung(P, A, prob, wt, sel::SelectionFor; gated::Bool = true) = nothing

"""
    formed_rung(P, A, prob, sel) -> (LinearSystem, Bool) or nothing

Ladder rung 5: form the reduced matrix by accumulating over stored entries, then invert it
densely — the same dense arithmetic as [`dense_rung`](@ref) reached without the `m×n` buffer
its product needs. This is where a sparse `A` that no sparse factorization suits ends up, so
the buffer is never allocated for one.

Accumulating reads entries, so a method here declines an operand that answers
[`is_materializable`](@ref) with `false`, as rung 6 does.

The default declines for every algorithm. Whether the rung is worth having at all is a
property of the algorithm — an inverse rebuilt once serves ADMM's whole run, where an
interior-point method would rebuild it every outer iteration — so the method that serves a
pair is written for the [`SelectionFor`](@ref) that wants it, and the interior-point ladder
does not reach this rung at all.
"""
formed_rung(P, A, prob, sel::SelectionFor) = nothing

"""
    dense_rung(P, A, prob, sel) -> (LinearSystem, Bool) or nothing

The ladder's terminal, which serves any pair of materializable matrices and is why every
rung above it may decline freely. What it builds differs by algorithm, so there is no
generic method: for [`ADMMSelection`](@ref) it is rung 6, [`ReducedCholesky`](@ref), which
forms the reduced matrix with one dense product and inverts it.

Declines when either operand answers [`is_materializable`](@ref) with `false`, since forming
the product reads entries. The ladder then falls through to [`indirect_rung`](@ref).
"""
dense_rung(P, A, prob, sel::SelectionFor) = refuse_selection("PureQPBase.dense_rung", sel)

"""
    dense_rung(P, A, prob, sel::ADMMSelection) -> (LinearSystem, Bool) or nothing

Ladder rung 6, the ADMM terminal: [`ReducedCholesky`](@ref).
"""
function dense_rung(P::AbstractMatrix, A::AbstractMatrix, prob, sel::ADMMSelection)
    (is_materializable(P) && is_materializable(A)) || return nothing
    return (ReducedCholesky(prob.q0, prob.n, prob.m), false)
end

# Every rung declines rather than erroring on a pair it does not serve, so the ladder reaches
# its next rung instead of the caller reaching a `MethodError`. This is the terminal rung's
# share of that: an operand outside `AbstractMatrix` is served below, not here.
dense_rung(P, A, prob, sel::ADMMSelection) = nothing

"""
    indirect_rung(P, A, prob, sel) -> (LinearSystem, Bool)

What sits below the terminal, reached by an operator [`dense_rung`](@ref) cannot
materialize. It has no gate: reaching it means nothing above could serve. Whether a
matrix-free solve is acceptable at all is the algorithm's decision — ADMM takes it, the
interior-point method refuses — so there is no generic method.
"""
indirect_rung(P, A, prob, sel::SelectionFor) = refuse_selection("PureQPBase.indirect_rung", sel)

"""
    indirect_rung(P, A, prob, sel::ADMMSelection) -> (LinearSystem, Bool)

Ladder rung 7, below the ADMM terminal: conjugate gradients, which needs only products with
`P` and `A` and so serves an operator no other rung can materialize. Without Krylov loaded
there is no such backend and [`indirect_backend`](@ref) says so.
"""
indirect_rung(P, A, prob, sel::ADMMSelection) =
    (indirect_backend(prob.q0, prob.n, prob.m, nothing), false)

choose_backend(P::Diagonal, A::Diagonal, prob, wt, sel::SelectionFor) =
    (DiagonalReduced(prob.q0, prob.n), false)

# The pairs whose reduced matrix has bandwidth 1. `Bidiagonal` is as wide an `A` as this
# reaches: a `Tridiagonal` one squares to bandwidth 2, which no symmetric type in
# LinearAlgebra stores.
#
# A `Tridiagonal` `P` names the same band a `SymTridiagonal` one does, and `factorize!`
# reads it through `P[j, j]` and `P[j, j+1]` alone; `validate` has already established that
# `P` is symmetric, so the subdiagonal it also stores holds the same numbers.
choose_backend(
    P::Union{SymTridiagonal, Tridiagonal}, A::Diagonal, prob, wt, sel::SelectionFor
) = (TridiagonalReduced(prob.q0, prob.n), false)

choose_backend(
    P::Union{Diagonal, SymTridiagonal, Tridiagonal}, A::Bidiagonal, prob, wt, sel::SelectionFor
) = (TridiagonalReduced(prob.q0, prob.n), false)

"""
    sparse_refusal(sel) -> String

Why `linsys = :sparse` could not serve a pair, in the terms of `sel`'s algorithm.

The two algorithms ask for different things: ADMM's chain ends at [`formed_rung`](@ref),
which needs only `A` stored sparsely, where the interior-point chain is the two factored
rungs and both of them read `P`'s pattern as well.
"""
sparse_refusal(sel::SelectionFor) =
    "linsys = :sparse factors the reduced or KKT matrix sparsely and could not serve this " *
    "pair: it needs SparseArrays.jl loaded and a representation the sparse rungs accept, " *
    "with a system that actually factors at this regularization."

sparse_refusal(::ADMMSelection) =
    "linsys = :sparse factors the reduced or KKT matrix sparsely and could not " *
    "serve this pair: it needs SparseArrays.jl loaded and A a SparseMatrixCSC, " *
    "with a system that actually factors at this regularization."

sparse_refusal(::IPMSelection) =
    "linsys = :sparse factors the reduced or KKT matrix sparsely and could not " *
    "serve this pair: it needs a SparseMatrixCSC P and A, SparseArrays.jl " *
    "loaded, and a system that actually factors at this regularization."

"""
    named_backend(::Val{LS}, P, A, prob, wt, sel, preconditioner) -> (LinearSystem, Bool) or nothing

The backend `linsys = LS` names, and whether it already carries a factorization of the
current data. `nothing` for `LS === :auto`, the one name that descends
[`select_backend`](@ref)'s ladder instead.

`LS` is a type parameter rather than a value so that naming a backend leaves exactly one
branch live and the rest are gone by specialization — the same reason [`setup`](@ref)
carries it that way, and what keeps a named kind from dragging every other backend's code
onto the trimmed path.

A named kind is an instruction, not a hint. `:kkt`, `:dense` and `:indirect` build their
backend outright. The structured kinds reach their own rung, and `:sparse`, `:block` and
`:lowrank` do it in two stages: first with the ladder's own gate in force, so a pair the
gate accepts reaches exactly the backend `:auto` would, then with the gate skipped, so a
pair `:auto` sends to the dense terminal still reaches the named kind instead of throwing.
Only a representation mismatch or a factorization failure refuses after that, and the
refusal names the condition.

Both algorithms call this. The rungs it reaches answer for the `sel` they are handed, so the
kinds an algorithm cannot serve are refused where it builds its workspace, before the problem
is built, so the message can name the reason.
"""
function named_backend(::Val{LS}, P, A, prob, wt, sel::SelectionFor, preconditioner) where {LS}
    q0, n, m = prob.q0, prob.n, prob.m
    if LS === :kkt
        return (FullKKT(q0, n, m), false)
    elseif LS === :dense
        # Past `choose_backend` entirely. Its rule for a sparse `A` reads the pattern and not
        # the numbers, and this is the way to overrule one that misjudges a problem.
        return (ReducedCholesky(q0, n, m), false)
    elseif LS === :indirect
        check_preconditioner(preconditioner, typeof(q0))
        return (indirect_backend(q0, n, m, preconditioner), false)
    elseif LS === :sparse
        rung = kkt_rung(P, A, prob, wt, sel)
        isnothing(rung) && (rung = reduced_rung(P, A, prob, wt, sel))
        isnothing(rung) && (rung = formed_rung(P, A, prob, sel))
        isnothing(rung) && (rung = kkt_rung(P, A, prob, wt, sel; gated = false))
        isnothing(rung) && (rung = reduced_rung(P, A, prob, wt, sel; gated = false))
        isnothing(rung) && throw(ArgumentError(sparse_refusal(sel)))
        return rung
    elseif LS === :diagonal
        (P isa Diagonal && A isa Diagonal) || throw(
            ArgumentError("linsys = :diagonal needs P and A both diagonal")
        )
        return choose_backend(P, A, prob, wt, sel)
    elseif LS === :tridiagonal
        tridiag_pair =
            (P isa Union{SymTridiagonal, Tridiagonal} && A isa Diagonal) ||
            (P isa Union{Diagonal, SymTridiagonal, Tridiagonal} && A isa Bidiagonal)
        tridiag_pair || throw(
            ArgumentError(
                "linsys = :tridiagonal needs a diagonal, symmetric-tridiagonal or tridiagonal " *
                    "P with a diagonal A, or any of those P with a bidiagonal A"
            )
        )
        return choose_backend(P, A, prob, wt, sel)
    elseif LS === :kronecker
        rung = kronecker_rung(P, A, prob, wt, sel)
        isnothing(rung) && throw(
            ArgumentError(
                "linsys = :kronecker needs A a KroneckerOperator, P a scalar multiple of the " *
                    "identity, a uniform rho and scaling = 0, and declines this pair"
            )
        )
        return rung
    elseif LS === :block
        rung = block_rung(P, A, prob, wt, sel)
        isnothing(rung) && (rung = block_rung(P, A, prob, wt, sel; require_multiple = false))
        isnothing(rung) && throw(
            ArgumentError(
                "linsys = :block needs P and A both block diagonal over the same column " *
                    "partition, and declines this pair"
            )
        )
        return rung
    elseif LS === :lowrank
        rung = lowrank_rung(P, A, prob, wt, sel)
        isnothing(rung) && (rung = lowrank_rung(P, A, prob, wt, sel; require_crossover = false))
        isnothing(rung) && throw(
            ArgumentError(
                "linsys = :lowrank needs a diagonal P and a RowCoupled A with at least one " *
                    "coupling row, and declines this pair"
            )
        )
        return rung
    end
    return nothing
end

"Name of the backend, for reporting."
backend_name(::ReducedCholesky) = :cholesky
backend_name(::FullKKT) = :bunchkaufman
backend_name(::DiagonalReduced) = :diagonal
backend_name(::TridiagonalReduced) = :tridiagonal

# `Rinv` and `K` are dense, so their triangle is what the factorization occupies. The
# diagonal and tridiagonal backends store their bands and nothing else.
dense_triangle(dim::Integer) = dim * (dim + 1) ÷ 2

function backend_info(ls::ReducedCholesky)
    dim = size(ls.Rinv, 1)
    return BackendInfo(backend_name(ls), true, :reduced, dim, dense_triangle(dim))
end

function backend_info(ls::FullKKT)
    dim = size(ls.K, 1)
    return BackendInfo(backend_name(ls), true, :kkt, dim, dense_triangle(dim))
end

backend_info(ls::DiagonalReduced) =
    BackendInfo(backend_name(ls), true, :reduced, length(ls.dinv), length(ls.dinv))

function backend_info(ls::TridiagonalReduced)
    dim = length(ls.dv)
    return BackendInfo(backend_name(ls), true, :reduced, dim, dim + length(ls.ev))
end

# Overwrite the Cholesky factor occupying `R` with the inverse it factors. `potri!` does
# this in place and touches only the upper triangle; the fallback covers element types
# LAPACK has no method for.
invert_spd!(R::StridedMatrix{<:LinearAlgebra.BlasFloat}, F) = LAPACK.potri!('U', R)
invert_spd!(R::AbstractMatrix, F) = copyto!(R, inv(F))

function factorize!(ls::ReducedCholesky{T}, prob, wt)::Bool where {T}
    P, A, D, E, c, n, m = prob.P, prob.A, prob.D, prob.E, prob.c, prob.n, prob.m
    R = ls.Rinv
    # `scaled_col!` writes only the entries the matrix actually has, so W is zeroed first.
    fill!(ls.W, zero(T))
    rho = wt.w
    # `m` square roots instead of `m*n`: a per-entry closure would pay one for every entry
    # of `W`, which is most of a refactorization's setup at the sizes the dense backend
    # serves. They go in the `work_m` scratch, so a refactorization allocates nothing; a loop
    # rather than a broadcast, whose aliasing check leaves a copy AllocCheck reports.
    sr = prob.work_m
    for i in 1:m
        sr[i] = sqrt(rho[i]) * E[i]
    end
    for j in 1:n
        dj = D[j]
        scaled_col!(T, ls.W, A, j, (a, i) -> sr[i] * a * dj)
    end
    if m > 0
        mul!(R, ls.W', ls.W)
    else
        fill!(R, zero(T))
    end
    for j in 1:n
        dj = D[j]
        add_scaled_col!(T, R, P, j, (p, i) -> c * D[i] * p * dj)
    end
    for i in 1:n
        R[i, i] += wt.sigma
    end
    F = cholesky!(Symmetric(R); check = false)
    issuccess(F) || return false
    invert_spd!(R, F)
    return true
end

function factorize!(ls::DiagonalReduced{T}, prob, wt)::Bool where {T}
    P, A, D, E, c, n, m = prob.P, prob.A, prob.D, prob.E, prob.c, prob.n, prob.m
    rho, sigma = wt.w, wt.sigma
    for j in 1:n
        dj = D[j]
        r = c * dj * P[j, j] * dj + sigma
        if j <= m
            a = E[j] * A[j, j] * dj
            r += rho[j] * a * a
        end
        # The reduced matrix carries `σI`, so a non-positive entry means `P` was indefinite
        # by more than `σ` absorbs -- the same condition a Cholesky reports as a failure.
        r > zero(T) || return false
        ls.dinv[j] = inv(r)
    end
    return true
end

function factorize!(ls::TridiagonalReduced{T}, prob, wt)::Bool where {T}
    P, A, D, E, c, n, m = prob.P, prob.A, prob.D, prob.E, prob.c, prob.n, prob.m
    rho, sigma = wt.w, wt.sigma
    dv, ev = ls.dv, ls.ev
    for j in 1:n
        dv[j] = c * D[j] * P[j, j] * D[j] + sigma
    end
    for j in 1:(n - 1)
        ev[j] = c * D[j] * P[j, j + 1] * D[j + 1]
    end
    # `Ãᵀ diag(ρ) Ã` row by row: row `k` reaches only the columns in its band, so each row
    # contributes to at most two diagonal entries and one off-diagonal one.
    for k in 1:m
        w = rho[k] * E[k] * E[k]
        cols = band_columns(A, k)
        for j in cols
            akj = A[k, j] * D[j]
            dv[j] += w * akj * akj
            if j + 1 in cols
                ev[j] += w * akj * A[k, j + 1] * D[j + 1]
            end
        end
    end
    copyto!(ls.fdv, dv)
    copyto!(ls.fev, ev)
    # An `ldlt` reports neither indefiniteness nor a zero pivot the way a Cholesky does: it
    # returns a factorization with a negative pivot in the first case and throws in the
    # second. Both mean this backend cannot solve the system, so both are refused here.
    ls.fact = try
        ldlt!(SymTridiagonal(ls.fdv, ls.fev))
    catch e
        e isa LinearAlgebra.ZeroPivotException || rethrow()
        return false
    end
    return all(>(zero(T)), ls.fact.data.dv)
end

"""
    assemble_kkt0!(ls::FullKKT, prob) -> Nothing

Fill `ls.K0` with the lower triangle of the scaled `P` and `A` blocks — `c·D[i]·P[i,j]·D[j]`
for `i ∈ j:n` (the P block's own lower triangle, diagonal included but without `σ`) and
`E[i]·A[i,j]·D[j]` for the `A` block, which occupies rows `n+1:n+m` and so is entirely below
the diagonal already. Neither block's upper-triangle mirror is written: `bunchkaufman!` reads
only the lower triangle of `Symmetric(ls.K, :L)`.
"""
function assemble_kkt0!(ls::FullKKT{T}, prob) where {T}
    P, A, D, E, c, n, m = prob.P, prob.A, prob.D, prob.E, prob.c, prob.n, prob.m
    K0 = ls.K0
    for j in 1:n
        dj = D[j]
        for i in j:n
            K0[i, j] = c * D[i] * T(P[i, j]) * dj
        end
        for i in 1:m
            K0[n + i, j] = E[i] * T(A[i, j]) * dj
        end
    end
    ls.k0_current = true
    return nothing
end

function factorize!(ls::FullKKT{T}, prob, wt)::Bool where {T}
    ls.k0_current || assemble_kkt0!(ls, prob)
    n, m = prob.n, prob.m
    copyto!(ls.K, ls.K0)
    for j in 1:n
        ls.K[j, j] += wt.sigma
    end
    for i in 1:m
        ls.K[n + i, n + i] = -wt.w_inv[i]
    end
    return factor_kkt!(ls)
end

function factor_kkt!(ls::FullKKT)
    ls.fact = bunchkaufman!(Symmetric(ls.K, :L); check = false)
    return issuccess(ls.fact)
end

# `fact` was built over `K` and `ipiv` and a solve reads only those, so it stays current
# without being replaced; LAPACK's `info` alone decides whether the factorization succeeded.
factor_kkt!(ls::FullKKT{T, <:StridedMatrix{T}}) where {T <: Union{Float32, Float64}} =
    iszero(sytrf_lower!(ls.K, ls.ipiv, ls.work))

"""
    kkt_factorization(K, ipiv) -> BunchKaufman

The factorization object a `FullKKT` over `K` starts with. For a LAPACK element type it is a
view of `K` and `ipiv` themselves, which every later `sytrf` rewrites in place; otherwise a
placeholder of the type `bunchkaufman!` returns, replaced at every factorization.
"""
kkt_factorization(K::AbstractMatrix{T}, ipiv) where {T} = bunchkaufman!(Symmetric(fill(one(T), 1, 1)))
kkt_factorization(K::StridedMatrix{T}, ipiv) where {T <: Union{Float32, Float64}} =
    LinearAlgebra.BunchKaufman(K, ipiv, 'L', true, false, zero(LinearAlgebra.BlasInt))

"""
    reduced_rhs!(prob, wt, rhs_x, rhs_z) -> prob.work_n

Assemble `rhs_x + Ãᵀ(w ⊙ rhs_z)`, the right-hand side of the reduced system.

Written into `work_n` rather than over an argument because the solves that consume it may
not alias their input and output — `symv` in particular.
"""
function reduced_rhs!(prob, wt, rhs_x, rhs_z)
    if prob.m > 0
        multiply!(prob.work_m, wt.w, rhs_z)
        mul_At!(prob.work_n, prob, prob.work_m)
        increment!(prob.work_n, rhs_x)
    else
        copyto!(prob.work_n, rhs_x)
    end
    return prob.work_n
end

function solve_system!(ls::ReducedInverse, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    mul!(x, Symmetric(ls.Rinv, :U), prob.work_n)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end

function solve_system!(ls::DiagonalReduced, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    multiply!(x, ls.dinv, prob.work_n)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end

function solve_system!(ls::TridiagonalReduced, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    copyto!(x, prob.work_n)
    ldiv!(ls.fact, x)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end

function solve_system!(ls::FullKKT, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    n, m = prob.n, prob.m
    # Indexed rather than `copyto!(view(...), ...)`: the views leave allocation sites that
    # AllocCheck reports, and the loops make the no-allocation property provable.
    for i in 1:n
        ls.rhs[i] = rhs_x[i]
    end
    for i in 1:m
        ls.rhs[n + i] = rhs_z[i]
    end
    ldiv!(ls.fact, ls.rhs)
    for i in 1:n
        x[i] = ls.rhs[i]
    end
    w_inv = wt.w_inv
    for i in 1:m
        z[i] = rhs_z[i] + w_inv[i] * ls.rhs[n + i]
    end
    return nothing
end

"""
    solve_multiplier!(ls::FullKKT, prob, wt, rhs_x, rhs_z, x, nu) -> Nothing

`ls.rhs[n+i]` is the augmented solve's own eliminated multiplier, so `ν` is read off it
directly instead of recovered through `z̃`. This is [`solve_system!`](@ref) minus the loop
that would go on to form `z̃ = rhs_z + w_inv ⊙ ν`.
"""
function solve_multiplier!(ls::FullKKT, prob, wt, rhs_x, rhs_z, x, nu)::Nothing
    n, m = prob.n, prob.m
    for i in 1:n
        ls.rhs[i] = rhs_x[i]
    end
    for i in 1:m
        ls.rhs[n + i] = rhs_z[i]
    end
    ldiv!(ls.fact, ls.rhs)
    for i in 1:n
        x[i] = ls.rhs[i]
    end
    for i in 1:m
        nu[i] = ls.rhs[n + i]
    end
    return nothing
end


"""
    refactor!(ws)

Refresh the workspace's factorization after `ρ`, `σ` or the problem data changed.

The backend is fixed at [`setup`](@ref), so that every solve dispatches statically and the
workspace stays concretely typed. If the reduced Cholesky ever fails here — which no
measured problem has produced once equilibration is on — the remedy is to rebuild the
workspace with `linsys = :kkt` rather than to switch backend underneath the caller.
"""
function refactor!(ws)
    set_refresh_index!(ws.linsys, ws.refactor_count)
    return refactored!(ws, factorize!(ws.linsys, ws.prob, ws.weights))
end

"""
    refactor_rho!(ws)

Refresh the workspace's factorization after `ρ` alone changed, through the backend's
[`refactor_weights!`](@ref) rather than a full rebuild. Counted and reported like any other
refactorization.
"""
function refactor_rho!(ws)
    set_refresh_index!(ws.linsys, ws.refactor_count)
    return refactored!(ws, refactor_weights!(ws.linsys, ws.prob, ws.weights))
end

"""
    accelerator_reset!(accel) -> accel

Drop the history an accelerator has built up, called from [`refactored!`](@ref) whenever a
refactorization makes the iteration a fixed point of a different map — a window spanning
both would extrapolate across two maps rather than one. `nothing` with no accelerator; an
algorithm that accepts one defines further methods for its own accelerator types.
"""
accelerator_reset!(::Nothing) = nothing

# Kept out of line so the message it builds stays off the refactorization path, which runs
# inside the iteration and must not allocate.
@noinline function throw_unfactorized(ls)
    # The reduced form squares `cond(A)`, so the full quasi-definite system is the remedy —
    # but only for a backend that is not already solving it.
    throw(
        ArgumentError(
            "the linear system could not be factorized with the $(backend_name(ls)) backend. " *
                (
                backend_info(ls).system === :reduced ?
                    "Rebuild the workspace with linsys = :kkt, which does not square the conditioning of A." :
                    "This is already the full KKT system; the problem is singular at this ρ."
            )
        )
    )
end

function refactored!(ws, ok::Bool)
    ok || throw_unfactorized(ws.linsys)
    ws.refactor_count += 1
    # A refactorization means `ρ` or the data moved, and the iteration is a fixed point of a
    # different map afterwards. An accelerator extrapolating from both sides of that is
    # extrapolating across two maps, so its history is dropped here.
    accelerator_reset!(ws.accel)
    return ws
end

"""
    indirect_backend(proto, n, m, preconditioner) -> LinearSystem

Build the matrix-free backend selected by `linsys = :indirect`, preconditioned by
`preconditioner`, or by a [`JacobiPreconditioner`](@ref) when that is `nothing`. Supplied by
the Krylov extension; without Krylov loaded there is no such backend and this says so.
"""
function indirect_backend(proto::AbstractVector, n::Integer, m::Integer, preconditioner)
    throw(
        ArgumentError(
            "linsys = :indirect needs Krylov.jl, which is a weak dependency: run " *
                "`using Krylov` before `setup`. It is not a core dependency because the " *
                "backend is only worth reaching for when the reduced matrix cannot be formed."
        )
    )
end

"""
    ldl_backend(gram, proto, n) -> LinearSystem or nothing

A backend that factors the already-assembled reduced matrix `gram.R` with an `LDLᵀ` other
than the one SparseArrays supplies, or `nothing` if no such factorization is available or the
matrix does not factor.

The reduced matrix is passed in already built, so the extension answering this needs to know
nothing about how it was assembled — and the sparse extension, in turn, needs no dependency
on whatever does the factoring.

Only the factorization is delegated. The substitutions and the diagonal scaling on the
per-iteration path stay in this package, over whatever `L` and `D` the backend exposes,
because they are as fast as any library's and they carry the allocation guarantee.
"""
ldl_backend(gram, proto::AbstractVector, n::Integer) = nothing

"""
    ldl_kkt_backend(K, proto, n, m) -> LinearSystem or nothing

The same delegation for the full quasi-definite KKT matrix `K`, which is factored `LDLᵀ`
without pivoting because a quasi-definite matrix admits one under any symmetric permutation.

Separate from [`ldl_backend`](@ref) because the solve differs, not the factorization: the
full system yields `z̃` from the eliminated multiplier where the reduced one recovers it with
a product against `A`.
"""
ldl_kkt_backend(gram, proto::AbstractVector, n::Integer, m::Integer) = nothing

"""
    ldl_posdef(P, sigma) -> Bool or nothing

Whether `P + σI` is positive definite, answered by an `LDLᵀ` other than the one SparseArrays
supplies, or `nothing` if no such factorization is available.

[`is_convex`](@ref) asks this before reaching for a factorization of its own. An `LDLᵀ`
answers definiteness by the sign of `D` rather than by failing, so the question needs no
error path, and a pure-Julia one keeps the convexity test inside what `juliac --trim` can
resolve — which the SuiteSparse bindings are not.
"""
ldl_posdef(P, sigma) = nothing

"""
    require_host(v, what)

Throw unless `v` is host memory, naming what needs it.

`polish!` and the derivatives build a dense `(n+k)×(n+k)` matrix and factor it with
`bunchkaufman!`, which has no GPU implementation. Without this the caller gets
`GPUArraysCore`'s scalar-indexing error from somewhere inside the factorization, which says
nothing about what to do.
"""
function require_host(v::AbstractVector, what::String)
    v isa Vector || throw(
        ArgumentError(
            "$what runs on the host and this workspace holds $(typeof(v)): it factors a " *
                "dense matrix with `bunchkaufman!`, which has no GPU counterpart. Move the " *
                "problem to the host with `Array`, or leave `polishing = false` and take the " *
                "solver's own iterate."
        )
    )
    return nothing
end
