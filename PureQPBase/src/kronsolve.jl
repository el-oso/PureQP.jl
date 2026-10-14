"""
    KroneckerReduced{T,M,V} <: LinearSystem

The reduced system when `A` is `A₁ ⊗ A₂` and `P` is a multiple of the identity, which makes

    R = (cμ + σ)I + ρ (G₁ ⊗ G₂),   Gᵢ = AᵢᵀAᵢ

diagonal in the eigenbasis of the two factors. With `Gᵢ = QᵢΛᵢQᵢᵀ`,

    R⁻¹ = (Q₁ ⊗ Q₂) diag(1 / (cμ + σ + ρ λ₁ₐ λ₂ᵦ)) (Q₁ ⊗ Q₂)ᵀ

so a solve is four small matrix multiplications and a scale: `O(n₁n₂(n₁ + n₂))` against the
`O(n₁²n₂²)` of a dense `symv`, in `O(n₁² + n₂²)` storage against `O(n₁²n₂²)`. Factorizing is
two eigendecompositions of the factors, `O(n₁³ + n₂³)`.

`Q1` and `Q2` hold the eigenvectors and `dinv` the reciprocal of the diagonal, laid out so
that `dinv[b, a]` pairs `λ₂ᵦ` with `λ₁ₐ` — the second factor running fastest, matching `vec`.

**Every condition in the first sentence is load-bearing**, and the log records the measurement
for each: a Kronecker `P` that is not a scalar multiple of `I` breaks the diagonalization, a
non-uniform `ρ` breaks it, and so does equilibration, because `c·μ·D²` is diagonal but not
scalar. [`kronecker_rung`](@ref) declines all three rather than returning a wrong answer.
"""
mutable struct KroneckerReduced{
        T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T},
    } <: LinearSystem
    Q1::M
    Q2::M
    lambda1::V
    lambda2::V
    dinv::M          # `n₂×n₁`, the reciprocal diagonal in the eigenbasis
    X::M             # `n₂×n₁` scratch, the right-hand side reshaped
    Z::M             # `n₂×n₁` scratch, one product in
    work1::V         # `syev` workspace for the `n₁` factor, sized once at construction
    work2::V         # and for the `n₂` factor
    mu::T            # the `μ` of `P = μI`, read by the rung and by every `factorize!`
end

"""
    KroneckerReduced(proto::AbstractVector, n1, n2)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data it
was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function KroneckerReduced(
        proto::AbstractVector{T}, n1::Integer, n2::Integer, mu::T
    ) where {T <: Real}
    Q1 = similar(proto, T, n1, n1)
    Q2 = similar(proto, T, n2, n2)
    # Sized once here, from LAPACK's own query, so no factorization has to ask again.
    work1 = similar(proto, T, max(syev_lwork(Q1), 1))
    work2 = similar(proto, T, max(syev_lwork(Q2), 1))
    return KroneckerReduced{T, Matrix{T}, Vector{T}}(
        Q1, Q2,
        similar(proto, T, n1), similar(proto, T, n2),
        similar(proto, T, n2, n1), similar(proto, T, n2, n1), similar(proto, T, n2, n1),
        work1, work2,
        mu,
    )
end

backend_name(::KroneckerReduced) = :kronecker

# Two eigenbases and a diagonal is everything stored; neither factor's Gram is kept.
function backend_info(ls::KroneckerReduced)
    n1, n2 = size(ls.Q1, 1), size(ls.Q2, 1)
    stored = n1 * n1 + n2 * n2 + n1 * n2
    return BackendInfo(backend_name(ls), true, :reduced, n1 * n2, stored)
end

"""
    kronecker_rung(P, A, prob, wt, sel) -> (LinearSystem, Bool) or nothing

Ladder rung for `A = A₁ ⊗ A₂` with a scalar `P`. Declines unless every condition the
diagonalization needs holds: `P` a multiple of the identity, `ρ` uniform, and no equilibration
scaling in force.

The default declines for every algorithm. The uniform weight the diagonalization needs is a
property of the algorithm, so the method that serves a pair is written for the
[`SelectionFor`](@ref) whose weights are uniform, and the interior-point ladder — whose
weights differ from row to row — does not reach this rung at all.
"""
kronecker_rung(P, A, prob, wt, sel::SelectionFor) = nothing

function kronecker_rung(
        P, A::KroneckerOperator, prob, wt::SystemWeights{T}, sel::ADMMSelection
    ) where {T <: Real}
    # Predicate first, value second. A `Union{Nothing,T}` for the caller to narrow is a call
    # `--trim` refuses to resolve, however plainly the check narrows it.
    is_scalar_multiple(P) || return nothing
    rho_vec = wt.w
    # `ρ` enters as `ρ (G₁ ⊗ G₂)` only when it is one number. A single equality row or a free
    # row gives it two values and the eigenbasis stops diagonalizing.
    isempty(rho_vec) && return nothing
    all(==(first(rho_vec)), rho_vec) || return nothing
    D, E, c = prob.D, prob.E, prob.c
    # Equilibration puts `c·μ·D²` in the reduced matrix, which is diagonal but not scalar, so
    # only unscaled data keeps the structure. `scaling = 0` is what produces this.
    (all(isone, D) && all(isone, E) && isone(c)) || return nothing
    n1, n2 = size(A.A1, 2), size(A.A2, 2)
    return (KroneckerReduced(prob.q0, n1, n2, T(scalar_multiple(P))), false)
end

# The diagonalization has no form for a general `P`.
function check_update(ls::KroneckerReduced, P, A)
    is_scalar_multiple(P) || throw(
        ArgumentError(
            "P must stay a scalar multiple of the identity: the kronecker backend " *
                "diagonalizes cμ + σ + ρ(G₁⊗G₂) and has no form for a general P. " *
                "Rebuild the workspace with setup."
        )
    )
    return nothing
end

function factorize!(ls::KroneckerReduced{T}, prob, wt)::Bool where {T}
    A, P = prob.A, prob.P
    # `μ` is read from `P` on every factorization, so a `P` that `update!` replaced is the one
    # the diagonal is built from. The predicate comes first for the same reason as in the rung:
    # for a representation that is never a scalar multiple it folds to `false` and leaves no
    # call to `scalar_multiple` behind.
    is_scalar_multiple(P) || return false
    ls.mu = T(scalar_multiple(P))
    # `Gᵢ = AᵢᵀAᵢ` is formed straight into the eigenvector buffer, which `syev` then overwrites
    # with the basis: the Gram is never kept, and neither it nor the factorization allocates.
    mul!(ls.Q1, transpose(A.A1), A.A1)
    mul!(ls.Q2, transpose(A.A2), A.A2)
    info1 = syev_lower!(ls.Q1, ls.lambda1, ls.work1)
    info2 = syev_lower!(ls.Q2, ls.lambda2, ls.work2)
    # A Gram matrix is positive semidefinite, so `syev` converges; a failure here is the
    # factor being unusable rather than the problem being non-convex, and the rung that
    # selected this backend takes the next one.
    return iszero(info1) && iszero(info2) && refactor_weights!(ls, prob, wt)
end

# The eigenbases depend only on `A` and `μ` only on `P`, so new weights rebuild only the
# reciprocal diagonal, which allocates nothing.
function refactor_weights!(ls::KroneckerReduced{T}, prob, wt)::Bool where {T}
    rho = first(wt.w)
    shift = prob.c * ls.mu + wt.sigma
    for a in axes(ls.dinv, 2), b in axes(ls.dinv, 1)
        d = shift + rho * ls.lambda1[a] * ls.lambda2[b]
        d > zero(T) || return false
        ls.dinv[b, a] = inv(d)
    end
    return true
end

function solve_system!(ls::KroneckerReduced, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    # Copied into the backend's own `n₂×n₁` scratch rather than reshaped in place: `reshape`
    # of a vector allocates an array header, and this runs every iteration.
    copyto!(ls.X, prob.work_n)
    mul!(ls.Z, ls.Q2', ls.X)          # Q₂ᵀ X
    mul!(ls.X, ls.Z, ls.Q1)           # Q₂ᵀ X Q₁
    # A loop, not `ls.X .*= ls.dinv`: an in-place broadcast has `X` on both sides, which
    # leaves an `unaliascopy` branch AllocCheck reports as an allocation. See
    # `PureQPBase/src/elementwise.jl`, which does the same for the vector cases.
    for i in paired(ls.X, ls.dinv)
        ls.X[i] *= ls.dinv[i]
    end
    mul!(ls.Z, ls.Q2, ls.X)           # Q₂ (…)
    mul!(ls.X, ls.Z, ls.Q1')          # Q₂ (…) Q₁ᵀ
    copyto!(x, ls.X)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end

"""
    KroneckerPreconditioner{T} <: Preconditioner

Preconditions the interior-point reduced matrix of a Kronecker pair by solving the same matrix
with one weight for every row, exactly and without forming anything.

The reduced matrix is `P + σI + Aᵀ diag(w) A`. With `P = P₁ ⊗ P₂` and `A = A₁ ⊗ A₂` and a single
weight `ω` in place of `w`, that is `P₁ ⊗ P₂ + ω(G₁ ⊗ G₂)` for `Gᵢ = AᵢᵀAᵢ` — two Kronecker
terms, which one generalized eigendecomposition per factor diagonalizes together: with
`Pᵢ = LᵢLᵢᵀ`, `Qᵢ` the eigenvectors of `Lᵢ⁻¹GᵢLᵢ⁻ᵀ` and `Uᵢ = Lᵢ⁻ᵀQᵢ`,

    Uᵢᵀ Pᵢ Uᵢ = I        Uᵢᵀ Gᵢ Uᵢ = diag(λᵢ)

so `(U₁ ⊗ U₂)ᵀ(P₁ ⊗ P₂ + ω G₁ ⊗ G₂)(U₁ ⊗ U₂) = diag(1 + ω λ₁ᵢ λ₂ⱼ)` and the inverse is two
Kronecker applications around an elementwise division. The factors are `kᵢ × kᵢ`; nothing of
size `n` by `n` is formed, and a solve costs `O(n(k₁ + k₂))`.

`ω` is the mean of the current weights, refreshed every call. That is what makes it work: the
weights span the barrier's whole range by the time the method converges, and a fixed `ω` stops
tracking the matrix — measured on `kron_problem(11)`, `ω = 1` exceeds a 5000-iteration cap from
the 39th outer iteration, while the mean holds the conjugate-gradient count to 8, 44, 386 and
896 along the whole trajectory. The eigenvectors never change, so a refresh is `O(n)`.

`σ` is left out. `U` is `P`-orthogonal rather than orthogonal, so `σI` is not diagonal in this
basis and cannot be folded in exactly; it is small against `P`'s own scale and this is a
preconditioner, not a solve.

Two scratch matrices make one preconditioner unsafe to apply from two tasks at once.
"""
struct KroneckerPreconditioner{T <: Real} <: Preconditioner
    U1::Matrix{T}
    U2::Matrix{T}
    # Held rather than formed per solve: `transpose` of a matrix allocates its wrapper on
    # Julia 1.13, and this runs once per conjugate-gradient iteration.
    U1t::Transpose{T, Matrix{T}}
    U2t::Transpose{T, Matrix{T}}
    lambda1::Vector{T}
    lambda2::Vector{T}
    # `1 / (1 + ω λ₁ᵢ λ₂ⱼ)`, laid out `k₂ × k₁` so a solve scales the same matrix it reshapes.
    dinv::Matrix{T}
    X::Matrix{T}
    Z::Matrix{T}
end

"""
    KroneckerPreconditioner(P::KroneckerOperator, A::KroneckerOperator)

Build the preconditioner from the two pairs of factors. `P`'s factors must be positive definite,
which is what the simultaneous diagonalization needs; a pair that is not is refused by name.
"""
function KroneckerPreconditioner(P::KroneckerOperator, A::KroneckerOperator)
    U1, l1 = kron_precond_factor(P.A1, A.A1)
    U2, l2 = kron_precond_factor(P.A2, A.A2)
    T = promote_type(eltype(U1), eltype(U2))
    k1, k2 = size(U1, 1), size(U2, 1)
    u1, u2 = dense_copy(T, U1), dense_copy(T, U2)
    M = KroneckerPreconditioner{T}(
        u1, u2, transpose(u1), transpose(u2), Vector{T}(l1), Vector{T}(l2),
        Matrix{T}(undef, k2, k1), Matrix{T}(undef, k2, k1), Matrix{T}(undef, k2, k1),
    )
    kron_precond_diagonal!(M, one(T))
    return M
end

"`(U, λ)` with `UᵀPU = I` and `Uᵀ(AᵀA)U = diag(λ)`, from one generalized eigendecomposition."
function kron_precond_factor(Pi::AbstractMatrix, Ai::AbstractMatrix)
    T = float(promote_type(eltype(Pi), eltype(Ai)))
    L = cholesky(Symmetric(Matrix{T}(Pi)); check = false)
    issuccess(L) || throw(
        ArgumentError(
            "each factor of a Kronecker P must be positive definite to precondition the " *
                "interior-point system: the preconditioner diagonalizes P₁ ⊗ P₂ and " *
                "G₁ ⊗ G₂ together, which needs a Cholesky of each Pᵢ. Pass a caller " *
                "preconditioner of your own, or solve with OperatorSplitting()."
        )
    )
    Li = L.L
    G = Matrix{T}(transpose(Matrix{T}(Ai)) * Matrix{T}(Ai))
    # `L⁻¹ G L⁻ᵀ`, symmetric, so its eigenvectors are orthogonal.
    W = Matrix{T}(Li \ (G / transpose(Li)))
    E = eigen(Symmetric((W + transpose(W)) / 2))
    return (transpose(Li) \ E.vectors, E.values)
end

"Refresh `dinv` for one weight `omega`, which is `O(k₁k₂)` and allocates nothing."
function kron_precond_diagonal!(M::KroneckerPreconditioner{T}, omega::T) where {T}
    l1, l2, d = M.lambda1, M.lambda2, M.dinv
    for j in eachindex(l1)
        a = omega * l1[j]
        for i in eachindex(l2)
            d[i, j] = one(T) / (one(T) + a * l2[i])
        end
    end
    return M
end

function update_preconditioner!(M::KroneckerPreconditioner{T}, prob, wt, k::Int) where {T}
    w = wt.w
    # The mean, not a fixed value: see the type's docstring for what a fixed `ω` costs.
    omega = isempty(w) ? one(T) : convert(T, sum(w) / length(w))
    kron_precond_diagonal!(M, omega)
    return M
end

# `y = (U₁ ⊗ U₂) D⁻¹ (U₁ ⊗ U₂)ᵀ x`, with `x` read as `k₂ × k₁`: two products in, an elementwise
# scale, two products out, and no `n × n` anything.
function LinearAlgebra.ldiv!(
        y::AbstractVector, M::KroneckerPreconditioner, x::AbstractVector
    )
    copyto!(M.X, x)
    mul!(M.Z, M.U2t, M.X)
    mul!(M.X, M.Z, M.U1)
    # An explicit loop rather than `.*=`: the broadcast's machinery carries allocation sites the
    # scan counts even where the runtime takes none. The index runs over one of the two rather
    # than over both: the constructor builds `X` and `dinv` at one size, so they cannot disagree,
    # and the two-argument `eachindex` reports a disagreement through a message built by
    # `Base.join`, which `--trim=safe` rejects on Julia 1.12.
    X, dinv = M.X, M.dinv
    for i in eachindex(X)
        X[i] *= dinv[i]
    end
    mul!(M.Z, M.U2, M.X)
    mul!(M.X, M.Z, M.U1t)
    copyto!(y, M.X)
    return y
end

"""
    add_reduced_term!(R, T, A, weights, D, n, m, scratch) -> R

Add `D Aᵀ diag(weights) A D` into `R`.

The generic method is in `linsys.jl` and reaches it through `2n` products with `A`. This one
contracts the two factors of a `KroneckerOperator` instead.

With `A = A₁ ⊗ A₂`, the second factor running fastest, and the weights read as the `m₂ × m₁`
matrix `Ω`,

    (Aᵀ diag(w) A)[(i₁,i₂), (j₁,j₂)] = Σ_κ Ω[κ₂,κ₁] A₁[κ₁,i₁] A₁[κ₁,j₁] A₂[κ₂,i₂] A₂[κ₂,j₂]

so for one pair `(i₂, j₂)` the inner sum over `κ₂` gives a vector `t = Ωᵀ(A₂[:,i₂] ⊙ A₂[:,j₂])`
and the whole `k₁ × k₁` block over `(i₁, j₁)` is `A₁ᵀ diag(t) A₁` — one `gemm`. Running over the
`k₂(k₂+1)/2` pairs of the second factor fills the matrix: `O(k₂²(m₁m₂ + k₁²m₁))` against the
products' `O(n²(k₁+k₂))`, measured as 1.1 ms against 3.3 ms at `n = 625`, `m = 2208`.

`D` is applied at the scatter, not folded into a factor: it is an `n`-vector over the pairs
`(i₁,i₂)` and is not separable in general.
"""
function add_reduced_term!(
        R::AbstractMatrix{T}, ::Type{T}, A::KroneckerOperator, weights::AbstractVector,
        D::AbstractVector, n::Integer, m::Integer, scratch, ej, av, col
    ) where {T}
    A1, A2 = A.A1, A.A2
    m1, k1 = size(A1)
    m2, k2 = size(A2)
    # Which factor's column pairs to run over. Each pair costs one `gemm` against the *other*
    # factor, so running over the one with fewer columns leaves the larger `gemm` and fewer of
    # them. The two orders are the same arithmetic; `k₁ = 1` against `k₂ = 449` is 1 pair and a
    # `449 × 449` gemm one way, 101025 pairs and a `1 × 1` gemm the other.
    k1 <= k2 && return kron_reduced_over_first!(R, T, A, weights, D, scratch)
    # The transposes come from the operand this call was handed, not from `scratch`: a wrapper
    # held across calls would still name the operand the backend was built with, which
    # `update!` may since have replaced. Taking it here allocates nothing.
    Omega, t, S, blk = scratch[1], scratch[2], scratch[3], scratch[4]
    A1t = transpose(A.A1)
    # `Ω[κ₂, κ₁] = w[(κ₁-1)m₂ + κ₂]`, the layout `A`'s row index already has.
    for c in 1:m1, r in 1:m2
        Omega[r, c] = weights[(c - 1) * m2 + r]
    end
    for i2 in 1:k2, j2 in 1:i2
        # `t = Ωᵀ(A₂[:,i₂] ⊙ A₂[:,j₂])`, through the column pair rather than a formed product.
        for c in 1:m1
            acc = zero(T)
            for r in 1:m2
                acc += Omega[r, c] * A2[r, i2] * A2[r, j2]
            end
            t[c] = acc
        end
        # `blk = A₁ᵀ diag(t) A₁`, one scaling pass and one `gemm`.
        for c in 1:k1, r in 1:m1
            S[r, c] = t[r] * A1[r, c]
        end
        mul!(blk, A1t, S)
        for j1 in 1:k1
            q = (j1 - 1) * k2 + j2
            dq = D[q]
            for i1 in 1:k1
                p = (i1 - 1) * k2 + i2
                R[p, q] += D[p] * blk[i1, j1] * dq
            end
        end
        # The `(j₂, i₂)` pair is the transpose of this one and is not visited again.
        if j2 != i2
            for j1 in 1:k1
                q = (j1 - 1) * k2 + i2
                dq = D[q]
                for i1 in 1:k1
                    p = (i1 - 1) * k2 + j2
                    R[p, q] += D[p] * blk[j1, i1] * dq
                end
            end
        end
    end
    return R
end

"""
    kron_reduced_over_first!(R, T, A, weights, D, scratch) -> R

The contraction run over the column pairs of `A₁` rather than of `A₂`.

For one pair `(i₁, j₁)` the sum over `κ₁` gives `t = Ω(A₁[:,i₁] ⊙ A₁[:,j₁])`, a vector over the
rows of `A₂`, and the `k₂ × k₂` block over `(i₂, j₂)` is `A₂ᵀ diag(t) A₂` — one `gemm`. This is
the mirror of the other order and the same arithmetic; it is the one to run when `A₁` has the
fewer columns, which leaves `k₁(k₁+1)/2` pairs and a `gemm` over `A₂`.

`t` is not sign-definite off the diagonal, so it is folded into one side as a plain scale
rather than as a square root.
"""
function kron_reduced_over_first!(
        R::AbstractMatrix{T}, ::Type{T}, A::KroneckerOperator, weights::AbstractVector,
        D::AbstractVector, scratch
    ) where {T}
    A1, A2 = A.A1, A.A2
    m1, k1 = size(A1)
    m2, k2 = size(A2)
    Omega = scratch[1]
    t2, S2, blk2 = scratch[5], scratch[6], scratch[7]
    A2t = transpose(A2)
    for c in 1:m1, r in 1:m2
        Omega[r, c] = weights[(c - 1) * m2 + r]
    end
    for i1 in 1:k1, j1 in 1:i1
        for r in 1:m2
            acc = zero(T)
            for c in 1:m1
                acc += Omega[r, c] * A1[c, i1] * A1[c, j1]
            end
            t2[r] = acc
        end
        for c in 1:k2, r in 1:m2
            S2[r, c] = t2[r] * A2[r, c]
        end
        mul!(blk2, A2t, S2)
        for j2 in 1:k2
            q = (j1 - 1) * k2 + j2
            dq = D[q]
            for i2 in 1:k2
                p = (i1 - 1) * k2 + i2
                R[p, q] += D[p] * blk2[i2, j2] * dq
            end
        end
        # The `(j₁, i₁)` pair is the transpose of this one and is not visited again.
        if j1 != i1
            for j2 in 1:k2
                q = (i1 - 1) * k2 + j2
                dq = D[q]
                for i2 in 1:k2
                    p = (j1 - 1) * k2 + i2
                    R[p, q] += D[p] * blk2[j2, i2] * dq
                end
            end
        end
    end
    return R
end

"Scratch the Kronecker contraction needs, or `nothing` for the generic product path."
function reduced_term_scratch(::Type{T}, A::KroneckerOperator) where {T}
    m1, k1 = size(A.A1)
    m2, k2 = size(A.A2)
    # Buffers only. Nothing derived from `A` is held: `add_reduced_term!` takes each factor's
    # transpose from the operand it is handed, so a replaced operand cannot leave a wrapper
    # naming the old one behind.
    #
    # Only the order `add_reduced_term!` will take gets its buffers; the other arm's are empty,
    # so the tuple's type does not depend on the sizes while its storage does.
    over_first = k1 <= k2
    return (
        Matrix{T}(undef, m2, m1), Vector{T}(undef, over_first ? 0 : m1),
        Matrix{T}(undef, over_first ? 0 : m1, over_first ? 0 : k1),
        Matrix{T}(undef, over_first ? 0 : k1, over_first ? 0 : k1),
        Vector{T}(undef, over_first ? m2 : 0),
        Matrix{T}(undef, over_first ? m2 : 0, over_first ? k2 : 0),
        Matrix{T}(undef, over_first ? k2 : 0, over_first ? k2 : 0),
    )
end

"""
    kron_cross_pair(Ai, Aj) -> Bool

Whether the two blocks are Kronecker products laid over the same row structure, so that
[`kron_cross_term!`](@ref) can serve their cross term.

Both factors must agree in rows: the identity it rests on reads the weight vector as an
`m₁ × m₂` grid, which both blocks have to index the same way.
"""
kron_cross_pair(Ai, Aj) = false
kron_cross_pair(Ai::KroneckerOperator, Aj::KroneckerOperator) =
    size(Ai.A1, 1) == size(Aj.A1, 1) && size(Ai.A2, 1) == size(Aj.A2, 1)

"""
    kron_cross_term!(R, T, Ai, Aj, w, D, scaled, gram, v, offi, offj) -> R

Write `Dᵢ Aᵢᵀ diag(w) Aⱼ Dⱼ` and its transpose into `R`, as one matrix product per pair of
columns of the two first factors.

For `Aᵢ = B₁ ⊗ B₂` and `Aⱼ = F₁ ⊗ F₂`, column `(p-1)k + q` of a Kronecker product pairs column
`p` of the first factor with column `q` of the second, and row `(c-1)m₂ + d` pairs row `c` with
row `d`. Summing the entry definition over the first factor's rows leaves

    (B₁[:,p] ⊗ B₂)ᵀ diag(w) (F₁[:,r] ⊗ F₂)  =  B₂ᵀ diag(v) F₂,
        v[d] = Σ_c B₁[c,p] · F₁[c,r] · w[(c-1)m₂ + d]

so each pair `(p, r)` of first-factor columns costs one product of the second factors against a
weight vector of length `m₂`, in place of one product per column of the narrower block. `v` goes
negative whenever the two first factors disagree in sign, so the weight cannot be folded into a
row scaling of either operand as a square root.

`scaled` holds `diag(v) F₂ Dⱼ` and `gram` the product; `v` is borrowed from the caller's
`m`-length buffer.
"""
function kron_cross_term!(
        R, ::Type{T}, Ai::KroneckerOperator, Aj::KroneckerOperator, w, D,
        scaled, gram, v, offi, offj
    ) where {T}
    B1, B2 = Ai.A1, Ai.A2
    F1, F2 = Aj.A1, Aj.A2
    m1, p1 = size(B1)
    m2, q1 = size(B2)
    p2, q2 = size(F1, 2), size(F2, 2)
    Z = view(scaled, 1:m2, 1:q2)
    G = view(gram, 1:q1, 1:q2)
    vv = view(v, 1:m2)
    B2t = transpose(B2)
    for p in 1:p1, r in 1:p2
        fill!(vv, zero(T))
        for c in 1:m1
            a = B1[c, p] * F1[c, r]
            iszero(a) && continue
            base = (c - 1) * m2
            for d in 1:m2
                vv[d] += a * w[base + d]
            end
        end
        # `Dⱼ` scales the columns of the product, so it folds into `Z` here rather than costing
        # a second pass over `G`.
        for s in 1:q2
            dq = D[offj + (r - 1) * q2 + s - 1]
            for d in 1:m2
                Z[d, s] = vv[d] * F2[d, s] * dq
            end
        end
        mul!(G, B2t, Z)
        for s in 1:q2
            q = offj + (r - 1) * q2 + s - 1
            for i in 1:q1
                pp = offi + (p - 1) * q1 + i - 1
                val = D[pp] * G[i, s]
                R[pp, q] += val
                R[q, pp] += val
            end
        end
    end
    return R
end

"""
    kron_cross_scratch(T, blocks) -> (scaled, gram)

Buffers for [`kron_cross_term!`](@ref), sized to the largest qualifying pair in a join.

Both are empty when no pair qualifies, which is the case the column form serves.
"""
function kron_cross_scratch(::Type{T}, blocks::Tuple) where {T}
    rows = cols = inner = 0
    for i in 1:length(blocks), j in (i + 1):length(blocks)
        Ai, Aj = blocks[i], blocks[j]
        kron_cross_pair(Ai, Aj) || continue
        rows = max(rows, size(Aj.A2, 1))
        cols = max(cols, size(Aj.A2, 2))
        inner = max(inner, size(Ai.A2, 2))
    end
    return (Matrix{T}(undef, rows, cols), Matrix{T}(undef, inner, cols))
end
