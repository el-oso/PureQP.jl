using LinearAlgebra, Random

"""
    spd_factor(rng, n, kappa) -> Matrix

A symmetric positive definite `n × n` matrix whose condition number is `kappa`, its eigenvalues
log-spaced from one downwards.
"""
function spd_factor(rng::AbstractRNG, n::Integer, kappa::Real)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    lam = exp10.(range(0, -log10(kappa); length = n))
    return Matrix(Symmetric(Q * Diagonal(lam) * Q'))
end

"""
    rect_factor(rng, m, n, kappa) -> Matrix

An `m × n` matrix whose condition number is `kappa`, its singular values log-spaced from one
downwards.
"""
function rect_factor(rng::AbstractRNG, m::Integer, n::Integer, kappa::Real)
    U = Matrix(qr(randn(rng, m, m)).Q)[:, 1:n]
    V = Matrix(qr(randn(rng, n, n)).Q)
    return U * Diagonal(exp10.(range(0, -log10(kappa); length = n))) * V'
end

"""
    kron_problem(seed; n1, n2, m1, m2, condP, condA) -> (P1, P2, A1, A2, q, l, u)

The factors of `P = P1 ⊗ P2` and `A = A1 ⊗ A2`, and bounds a point satisfies with unit room on
every row.

`cond(X1 ⊗ X2) = cond(X1) cond(X2)`, so each factor is built with the square root of the
condition number asked of its product. The defaults are the shape of a two-dimensional
discretization on a product grid: `625` variables against `2208` rows, at the conditioning such
a problem reaches.
"""
function kron_problem(
        seed::Integer; n1::Integer = 25, n2::Integer = 25, m1::Integer = 46, m2::Integer = 48,
        condP::Real = 8.3e8, condA::Real = 2.4e11
    )
    rng = MersenneTwister(seed)
    P1 = spd_factor(rng, n1, sqrt(condP))
    P2 = spd_factor(rng, n2, sqrt(condP))
    A1 = rect_factor(rng, m1, n1, sqrt(condA))
    A2 = rect_factor(rng, m2, n2, sqrt(condA))
    q = randn(rng, n1 * n2)
    b = PureQPBase.KroneckerOperator(A1, A2) * randn(rng, n1 * n2)
    return (P1, P2, A1, A2, q, b .- 1, b .+ 1)
end

"""
    kron_operator_pair(P1, P2, A1, A2) -> (Pop, Aop)

The two factor pairs as [`PureQPBase.KroneckerOperator`](@ref)s.
"""
kron_operator_pair(P1, P2, A1, A2) =
    (PureQPBase.KroneckerOperator(P1, P2), PureQPBase.KroneckerOperator(A1, A2))
