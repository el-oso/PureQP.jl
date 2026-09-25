"""
    PureQPBaseLDLFactorizationsExt

Factors the reduced matrix with LDLFactorizations.jl instead of CHOLMOD.

Loading LDLFactorizations changes nothing a caller can observe except speed: the same
problems select the same kind of backend, take the same iterations, and reach the same
answers. What changes is who computes the factorization, and how much of it has to be
rebuilt when `ρ` moves.

Two things make it worth the switch, both measured on the OSQP benchmark suite's own
problems, whose factors hold two to three nonzeros per column:

- The numeric factorization is 2.3–3.1× faster than CHOLMOD's and allocates nothing:
  9.1 µs against 21.4 (Lasso), 25.2 against 58.4 (Huber). A refactorization happens every
  time `ρ` is retuned, inside the solve loop, so this is not a setup-only saving.
- `L` and `D` are Julia arrays owned by the factorization. The CHOLMOD path has to extract
  them with `sparse(F.L)` and a transpose on every refactorization — 23 µs on Huber, and a
  fresh matrix each time for a pattern that never changes.

Only the factorization is delegated. The substitutions and the diagonal scaling stay in
this package: measured against LDLFactorizations' own solve they are as fast or faster
(2.74 µs against 3.02 on Lasso, 8.10 against 8.17 on Huber), and they are the code the
allocation guarantee is proved on.
"""
module PureQPBaseLDLFactorizationsExt

using PureQPBase: PureQPBase
using TypeContracts: TypeContracts, @verify
using LinearAlgebra: Symmetric, I
using SparseArrays: SparseMatrixCSC, nnz, nzrange, rowvals, nonzeros, triu
using LDLFactorizations: LDLFactorizations, ldl_analyze, ldl_factorize!, factorized

# Resolved once. `Base.get_extension` builds a `PkgId` on every call, and a `Module` returned
# at run time makes each call through it a dynamic dispatch with boxed arguments, which on the
# refactorization path is what `ρ` pays every time it moves. This extension loads after the
# SparseArrays one, which its `[extensions]` entry requires, so the lookup cannot be `nothing`.
const SparseExt = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)::Module

"""
    fact_L(F) -> SparseMatrixCSC
    fact_perm(F) -> Vector{Int}

The factor and the permutation, pulled out of an `LDLFactorization` once.

`LDLFactorization` reaches these through a `getproperty` that assembles them from its
internal arrays, so each access allocates and infers as `Any`. Reading them per iteration
would put both on the hot path — which is what the backends store them for, exactly as the
CHOLMOD backends store theirs rather than reaching into a foreign factor.

Asserted rather than converted: the assertion narrows an `Any` to something the caller can
infer through, while a concrete element type here would round every factor to it — a
`BigFloat` problem factored to `Float64` and returned as though it had not been.
"""
# Parametric rather than `SparseMatrixCSC{<:Real, Int}`: that `UnionAll` is constructed on
# every call, while an assertion naming the factorization's own element type folds away.
fact_L(F::LDLFactorizations.LDLFactorization{T}) where {T} = F.L::SparseMatrixCSC{T, Int}
fact_perm(F)::Vector{Int} = F.P

"""
    SparseLDL{T,V,G,F} <: PureQPBase.LinearSystem

The reduced backend factored by LDLFactorizations, solved by this package.

`R = Lᵤ D Lᵤᵀ` with `Lᵤ` unit lower triangular. LDLFactorizations stores `Lᵤ` strictly below
the diagonal and `D` separately, which is exactly the form the substitutions want — the unit
diagonal is never loaded and never divided by.

`gram` rebuilds `R`'s values in place when `ρ` moves; `fact` holds the ordering and the
symbolic analysis, so a refactorization is numeric only.
"""
mutable struct SparseLDL{T <: Real, V <: AbstractVector{T}, G, F} <: PureQPBase.LinearSystem
    gram::G
    fact::F
    L::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    dinv::Vector{T}
    permuted::V
end

PureQPBase.backend_name(::SparseLDL) = :ldlfactorizations

PureQPBase.backend_info(ls::SparseLDL) = PureQPBase.BackendInfo(
    PureQPBase.backend_name(ls), true, :reduced, size(ls.L, 1), nnz(ls.L)
)

"""
    PureQPBase.ldl_backend(gram, proto, n) -> SparseLDL or nothing

Analyse and factor `gram.R`, with `ldl_analyze`'s own fill-reducing ordering.

Returning `nothing` leaves the CHOLMOD path to answer instead, which is what happens when the
analysis fails or the reduced matrix turns out to be singular.
"""
function PureQPBase.ldl_backend(gram, proto::AbstractVector{T}, n::Integer) where {T <: Real}
    R = gram.R
    M = Symmetric(R, :U)
    fact = try
        ldl_analyze(M)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
    ldl_factorize!(M, fact)
    # `D` is singular exactly when the reduced matrix is, which for `P̃ + σI + Ãᵀ diag(ρ) Ã`
    # means the problem was not convex after all. Hand it back rather than divide by zero.
    any(iszero, fact.d) && return nothing
    SparseExt.check_factor(fact.L, n)
    return SparseLDL{T, typeof(proto), typeof(gram), typeof(fact)}(
        gram, fact, fact_L(fact), fact_perm(fact), inv.(fact.d), similar(proto, T, n)
    )
end

"""
    ldl_posdef(P::SparseMatrixCSC, sigma) -> Bool

Whether `P + σI` is positive definite, read off the `LDLᵀ` factorization's diagonal.

The return is `Bool` and not `Union{Bool, Nothing}` on purpose: [`PureQPBase.is_convex`](@ref)
keeps a factorization of its own for when this is unavailable, and a maybe-answer would leave
that fallback reachable for the trimmer even though dispatch has already settled the question.

`ldl_factorize!` stops at a zero pivot rather than raising, and `factorized` is how that is
detected. It is a verdict and not a refusal to answer: the factorization stops exactly when
`P + σI` is singular, which is a definite answer of "not positive definite". The diagonal must
not be read in that case — it is filled only as far as the column that stopped, and holds
uninitialized memory beyond it.
"""
function PureQPBase.ldl_posdef(P::SparseMatrixCSC, sigma)
    # No rows, nothing to violate definiteness, and `ldl_analyze` indexes unconditionally.
    isempty(P) && return true
    # Only the triangle: `ldl_factorize!` consumes every stored entry rather than reading the
    # half the `Symmetric` wrapper names, so a fully stored `P` — which is what this package
    # takes — would contribute each off-diagonal twice and report a definite matrix indefinite.
    M = Symmetric(triu(P) + sigma * I, :U)
    fact = ldl_analyze(M)
    ldl_factorize!(M, fact)
    factorized(fact) || return false
    d = fact.d
    for k in eachindex(d)
        d[k] > zero(eltype(d)) || return false
    end
    return true
end

function PureQPBase.factorize!(ls::SparseLDL{T}, prob, wt)::Bool where {T}
    P, A = prob.P, prob.A
    Ext = SparseExt
    if !Ext.describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries elsewhere, so both the slot map
        # and the analysis built on its pattern are stale.
        ls.gram = Ext.reduced_gram(T, P, A, prob.n)
        R = Ext.refill!(ls.gram, P, A, wt.w, prob.E, prob.D, prob.c, wt.sigma)
        ls.fact = ldl_analyze(Symmetric(R, :U))
    else
        Ext.refill!(ls.gram, P, A, wt.w, prob.E, prob.D, prob.c, wt.sigma)
    end
    ldl_factorize!(Symmetric(ls.gram.R, :U), ls.fact)
    d = ls.fact.d
    any(iszero, d) && return false
    ls.L = fact_L(ls.fact)
    ls.perm = fact_perm(ls.fact)
    Ext.check_factor(ls.L, prob.n)
    # In place: `D` has the same length every time, and a refactorization runs inside the
    # solve loop whenever `ρ` is retuned.
    length(ls.dinv) == length(d) || resize!(ls.dinv, length(d))
    ls.dinv .= inv.(d)
    return true
end


"""
    unit_forward!(x, L, N)
    unit_backward!(x, L, N)

Substitution against a unit lower triangular factor and its transpose, in place.

The diagonal is implied, so neither loop loads it and neither divides: forward scatters down
each column, backward gathers back up the same column. `L` holds only the strictly lower
entries, which is how LDLFactorizations stores it.

Both run unchecked, which [`check_factor`](@ref) is what makes safe — see there for why the
check is hoisted out of the loops and what it is worth. `@simd ivdep` is claimed only on the
forward loop: within a column the row indices are distinct, so its scattered writes carry no
dependency between iterations. The backward loop accumulates into a scalar, and vectorizing
a floating-point reduction reassociates it — which would move the iterates and cost the
property that this solver takes the same steps as the reference implementation.
"""
function unit_forward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in 1:N
        xj = x[j]
        @simd ivdep for p in colptr[j]:(colptr[j + 1] - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function unit_backward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in N:-1:1
        s = x[j]
        for p in colptr[j]:(colptr[j + 1] - 1)
            s -= vals[p] * x[rows[p]]
        end
        x[j] = s
    end
    return x
end

function PureQPBase.solve_system!(ls::SparseLDL{T}, prob, wt, rhs_x, rhs_z, x, z)::Nothing where {T}
    rhs = PureQPBase.reduced_rhs!(prob, wt, rhs_x, rhs_z)
    perm, work, n = ls.perm, ls.permuted, prob.n
    L, dinv = ls.L, ls.dinv
    # R[perm, perm] = Lᵤ D Lᵤᵀ, so the solve is a permutation, two substitutions and the
    # diagonal, all over buffers this backend owns.
    for i in 1:n
        work[i] = rhs[perm[i]]
    end
    unit_forward!(work, L, n)
    for i in 1:n
        work[i] *= dinv[i]
    end
    unit_backward!(work, L, n)
    for i in 1:n
        x[perm[i]] = work[i]
    end
    prob.m > 0 && PureQPBase.mul_A!(z, prob, x)
    return nothing
end

"""
    LDLKKT{T,V,F} <: PureQPBase.LinearSystem

The full quasi-definite KKT backend, factored by LDLFactorizations.

    K = ⎡P̃ + σI    Ãᵀ  ⎤
        ⎣Ã      −diag(ρ⁻¹)⎦

`K` is indefinite, which is not an obstacle: a quasi-definite matrix has an `LDLᵀ` under any
symmetric permutation, so the fill-reducing ordering can be chosen once and reused without
pivoting for stability. `D` carries the signs.

Unlike [`SparseLDL`](@ref) this recovers `z̃` from the eliminated multiplier, so the solve
needs no product against `A`.
"""
mutable struct LDLKKT{T <: Real, V <: AbstractVector{T}, G, F} <: PureQPBase.LinearSystem
    gram::G
    fact::F
    L::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    dinv::Vector{T}
    work::V
end

PureQPBase.backend_name(::LDLKKT) = :ldl_kkt

PureQPBase.backend_info(ls::LDLKKT) = PureQPBase.BackendInfo(
    PureQPBase.backend_name(ls), true, :kkt, size(ls.L, 1), nnz(ls.L)
)

function PureQPBase.ldl_kkt_backend(
        gram, proto::AbstractVector{T}, n::Integer, m::Integer
    ) where {T <: Real}
    M = Symmetric(gram.K, :U)
    fact = try
        ldl_analyze(M)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
    ldl_factorize!(M, fact)
    any(iszero, fact.d) && return nothing
    SparseExt.check_factor(fact.L, n + m)
    v = similar(proto, T, n + m)
    return LDLKKT{T, typeof(v), typeof(gram), typeof(fact)}(
        gram, fact, fact_L(fact), fact_perm(fact), inv.(fact.d), v
    )
end

function PureQPBase.factorize!(ls::LDLKKT{T}, prob, wt)::Bool where {T}
    Ext = SparseExt
    P, A = prob.P, prob.A
    if !Ext.describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries elsewhere, so the slot map and
        # the analysis built on its pattern are both stale.
        ls.gram = Ext.kkt_gram(T, P, A, prob.n, prob.m)
        K = Ext.refill_kkt!(ls.gram, P, A, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
        ls.fact = ldl_analyze(Symmetric(K, :U))
    else
        Ext.refill_kkt!(ls.gram, P, A, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
    end
    ldl_factorize!(Symmetric(ls.gram.K, :U), ls.fact)
    d = ls.fact.d
    any(iszero, d) && return false
    ls.L = fact_L(ls.fact)
    ls.perm = fact_perm(ls.fact)
    SparseExt.check_factor(ls.L, prob.n + prob.m)
    length(ls.dinv) == length(d) || resize!(ls.dinv, length(d))
    ls.dinv .= inv.(d)
    return true
end

function PureQPBase.solve_system!(ls::LDLKKT{T}, prob, wt, rhs_x, rhs_z, x, z)::Nothing where {T}
    n, m = prob.n, prob.m
    N = n + m
    perm, work = ls.perm, ls.work
    L, dinv = ls.L, ls.dinv
    # Permute straight out of the two right-hand sides: the assembled vector is never needed
    # in its own order.
    for i in 1:N
        p = perm[i]
        work[i] = p <= n ? rhs_x[p] : rhs_z[p - n]
    end
    unit_forward!(work, L, N)
    for i in 1:N
        work[i] *= dinv[i]
    end
    unit_backward!(work, L, N)
    # And scatter straight into the outputs.
    for i in 1:N
        p = perm[i]
        if p <= n
            x[p] = work[i]
        else
            z[p - n] = work[i]
        end
    end
    w_inv = wt.w_inv
    for i in 1:m
        z[i] = rhs_z[i] + w_inv[i] * z[i]
    end
    return nothing
end

"""
    PureQPBase.solve_multiplier!(ls::LDLKKT, prob, wt, rhs_x, rhs_z, x, nu) -> Nothing

The forward-backward solve already leaves `ν` in `work` before the eliminated multiplier
would be turned into `z̃ = rhs_z + w_inv ⊙ ν`, so this is [`PureQPBase.solve_system!`](@ref)
minus that last loop.
"""
function PureQPBase.solve_multiplier!(
        ls::LDLKKT{T}, prob, wt, rhs_x, rhs_z, x, nu
    )::Nothing where {T}
    n, m = prob.n, prob.m
    N = n + m
    perm, work = ls.perm, ls.work
    L, dinv = ls.L, ls.dinv
    for i in 1:N
        p = perm[i]
        work[i] = p <= n ? rhs_x[p] : rhs_z[p - n]
    end
    unit_forward!(work, L, N)
    for i in 1:N
        work[i] *= dinv[i]
    end
    unit_backward!(work, L, N)
    for i in 1:N
        p = perm[i]
        if p <= n
            x[p] = work[i]
        else
            nu[p - n] = work[i]
        end
    end
    return nothing
end

# No `trim_compat` claim: the factorization is a foreign package's, and the trim entry points
# cover the dense path.
@verify SparseLDL
@verify LDLKKT

end # module PureQPBaseLDLFactorizationsExt
