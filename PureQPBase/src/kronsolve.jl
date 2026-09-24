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
    return KroneckerReduced{T, Matrix{T}, Vector{T}}(
        similar(proto, T, n1, n1), similar(proto, T, n2, n2),
        similar(proto, T, n1), similar(proto, T, n2),
        similar(proto, T, n2, n1), similar(proto, T, n2, n1), similar(proto, T, n2, n1),
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
    # `Gᵢ = AᵢᵀAᵢ` is formed at factor size and thrown away; only its eigenbasis is kept.
    F1 = eigen(Symmetric(A.A1' * A.A1))
    F2 = eigen(Symmetric(A.A2' * A.A2))
    copyto!(ls.Q1, F1.vectors)
    copyto!(ls.Q2, F2.vectors)
    copyto!(ls.lambda1, F1.values)
    copyto!(ls.lambda2, F2.values)
    return refactor_weights!(ls, prob, wt)
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
    for i in eachindex(ls.X, ls.dinv)
        ls.X[i] *= ls.dinv[i]
    end
    mul!(ls.Z, ls.Q2, ls.X)           # Q₂ (…)
    mul!(ls.X, ls.Z, ls.Q1')          # Q₂ (…) Q₁ᵀ
    copyto!(x, ls.X)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end
