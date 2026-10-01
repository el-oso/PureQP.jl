# The matrices `cholesky_factor` factors densely, into an `UpperTriangular{T, Matrix{T}}`. The
# two arms take separate methods: a bare matrix promises nothing about its two triangles and is
# symmetrized, while a wrapper names the triangle to read and the other is not read at all.
const DenseFactorable = Union{StridedMatrix, SymmetricFactorable}

"""
    has_cholesky_factor(P) -> Bool

Whether [`cholesky_factor`](@ref) can factor `P` in `P`'s own form, without forming `P`.

`true` for a strided matrix, a `Symmetric` of one, a `Diagonal`, a [`BlockDiagonal`](@ref) of
those, and a [`KroneckerOperator`](@ref) of strided factors. `false` for a
[`ProductOperator`](@ref), whose entries are unavailable and which a factorization cannot
reach through products, and `false` for every other matrix.

Split from the factorization because an `R` that is `nothing` when `P` cannot be factored is a
union that `--trim` refuses to resolve; a caller asks this first and then calls
[`cholesky_factor`](@ref), whose return type is concrete.
"""
has_cholesky_factor(::Any) = false
has_cholesky_factor(::DenseFactorable) = true
has_cholesky_factor(::Diagonal) = true
# By value: blocks of mixed types collect into a `Vector` of their common supertype.
has_cholesky_factor(P::BlockDiagonal) = all(B -> B isa DenseFactorable, P.blocks)
has_cholesky_factor(::KroneckerOperator{<:Any, <:StridedMatrix}) = true
has_cholesky_factor(::ProductOperator) = false

"""
    CholeskyFactor

The contract of the `R` that [`cholesky_factor`](@ref) returns, which is not a subtype of this
type: an `UpperTriangular` and a `Diagonal` are `R`s too. `TypeContracts.check_contract(typeof(R),
CholeskyFactor)` checks one against it.

An `R` is a square matrix with `RᵀR = P + shift*I` and answers two in-place solves on a vector:

  - `ldiv!(R, v)` overwrites `v` with `R⁻¹v`;
  - `ldiv!(Rt, v)`, with `Rt = transpose(R)`, overwrites `v` with `R⁻ᵀv`.

Both allocate nothing on a `Vector` of `R`'s element type. The wrapper `transpose(R)` is
itself a heap allocation on Julia 1.13 for some `R`, so a caller that solves in a loop forms
it once and holds it. Some `R` keep scratch that `ldiv!` overwrites, so one `R` is not safe to
solve with from two tasks at once.
"""
abstract type CholeskyFactor end

@strict_contract CholeskyFactor begin
    Base.size(::Self)::Tuple{Int, Int} => "the size of `P`"
    LinearAlgebra.ldiv!(::Self, ::AbstractVector) => "overwrite the vector with `R⁻¹v`, in place"
end

"""
    cholesky_factor(P, shift) -> R

The upper Cholesky factor of `P + shift*I`, in `P`'s own form, so that `RᵀR = P + shift*I`;
the contract `R` meets is [`CholeskyFactor`](@ref). Defined where [`has_cholesky_factor`](@ref)
holds. Nothing the size of `P` is formed unless `P` is dense.

| `P` | `R` |
|---|---|
| strided matrix | `UpperTriangular{T, Matrix{T}}`, from a symmetrized, shifted copy |
| `Symmetric` or `Hermitian` of one | `UpperTriangular{T, Matrix{T}}`, from the triangle `uplo` names; the other is not read |
| `Diagonal` | `Diagonal(sqrt.(d .+ shift))` |
| `BlockDiagonal` | a `BlockDiagonal` of the blocks' factors |
| `KroneckerOperator` | a [`KroneckerCholesky`](@ref), `R₁ ⊗ R₂`, when `shift = 0` and both factors are positive definite; otherwise a [`KroneckerSquareRoot`](@ref) |

A dense `P` equal to `μI` yields `√(μ + shift) I` without a factorization.

`R` is upper triangular for every `P` but a `KroneckerOperator` that needs the square root:
`P₁ ⊗ P₂ + εI` is not a Kronecker product, so it has no `R₁ ⊗ R₂`, and neither does a pair of
negative definite factors whose product is positive definite. Triangularity is not part of
[`CholeskyFactor`](@ref), which asks for `RᵀR = P + shift*I` and two solves.

Throws an `ArgumentError` naming the remedy when `P + shift*I` is not positive definite.
"""
function cholesky_factor end

function cholesky_factor(H::StridedMatrix, shift)
    T = eltype(H)
    n = LinearAlgebra.checksquare(H)
    eps = convert(T, shift)
    scalar = scalar_factor(H, n, eps)
    isnothing(scalar) || return scalar
    # Symmetrize into one buffer: a bare matrix says nothing about which of `H[i, j]` and
    # `H[j, i]` is meant, so both are read. `(H + H') / 2` reads pleasantly and allocates four
    # matrices to produce one.
    Hs = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        Hs[i, j] = (H[i, j] + H[j, i]) / 2
    end
    return shifted_cholesky!(Hs, n, eps)
end

"""
    cholesky_factor(H::SymmetricFactorable, shift) -> UpperTriangular

The factor of `H + shift*I` read from the triangle `H` names.

`H.uplo` says which triangle is the matrix, so the other is never read and no averaging is
needed: the entries are copied straight into the buffer the factorization overwrites, upper
from an `'U'` wrapper and transposed from an `'L'` one. That is half the reads and half the
writes of the method for a bare matrix, which has to consult both triangles because nothing
told it which one is meant.
"""
function cholesky_factor(H::SymmetricFactorable, shift)
    T = eltype(H)
    n = LinearAlgebra.checksquare(H)
    eps = convert(T, shift)
    scalar = scalar_factor(H, n, eps)
    isnothing(scalar) || return scalar
    # Only the upper triangle is filled, and `shifted_cholesky!` reads only that.
    Hs = copy_upper_triangle!(Matrix{T}(undef, n, n), parent(H), n, H.uplo == 'U')
    return shifted_cholesky!(Hs, n, eps)
end

"""
    scalar_factor(H, n, eps) -> UpperTriangular or nothing

The factor of `H + eps*I` when `H` is `μI`, and `nothing` when it is not.

`μI` has the factor `√μ I`, so a reduction `A R⁻¹` against it is a scaling rather than a
triangular solve. [`scalar_diagonal`](@ref) stops at the first entry that rules `μI` out, which
for a dense `H` is the first off-diagonal one.
"""
function scalar_factor(H::AbstractMatrix, n::Integer, eps)
    mu = scalar_diagonal(H)
    isnothing(mu) && return nothing
    mu + eps > 0 || throw(not_positive_definite(eps))
    return UpperTriangular(Matrix{eltype(H)}(sqrt(mu + eps) * I, n, n))
end

"""
    shifted_cholesky!(Hs, n, eps) -> UpperTriangular

Factor `Hs + eps*I` in place, reading `Hs`'s upper triangle and overwriting it with the factor.

`check = false` so an indefinite matrix is a value to test rather than an exception to catch.
"""
function shifted_cholesky!(Hs::Matrix, n::Integer, eps)
    if !iszero(eps)
        for i in 1:n
            Hs[i, i] += eps
        end
    end
    F = cholesky!(Symmetric(Hs, :U), NoPivot(); check = false)
    issuccess(F) || throw(not_positive_definite(eps))
    return UpperTriangular(F.factors)
end

function not_positive_definite(eps)
    return ArgumentError(
        iszero(eps) ?
            "P is not positive definite, which ActiveSet() needs when eps_prox = 0, " *
            "because the reduction factors it. Pass eps_prox > 0 to run proximal-point " *
            "iterations instead, which accept a positive semidefinite P." :
            "P + eps_prox*I is not positive definite, so P is not positive semidefinite " *
            "and the problem is not convex."
    )
end

"""
    scalar_diagonal(H) -> μ or nothing

`μ` when `H` is `μI`, and `nothing` otherwise. Stops at the first entry that rules it out, so
a dense `H` costs one comparison rather than a pass over the matrix.
"""
function scalar_diagonal(H::AbstractMatrix)
    n = size(H, 1)
    (n == size(H, 2) && n > 0) || return nothing
    mu = H[1, 1]
    for j in 1:n, i in 1:n
        if i == j
            H[i, j] == mu || return nothing
        else
            iszero(H[i, j]) || return nothing
        end
    end
    return mu
end

function cholesky_factor(P::Diagonal, shift)
    T = eltype(P)
    d = P.diag .+ convert(T, shift)
    all(>(zero(T)), d) || throw(not_positive_definite(shift))
    return Diagonal(sqrt.(d))
end

function cholesky_factor(P::BlockDiagonal, shift)
    return BlockDiagonal(map(B -> cholesky_factor(B, shift), P.blocks))
end

# Block by block, each against its own run of `v`. The views are not escaping, so they cost
# nothing.
function LinearAlgebra.ldiv!(R::BlockDiagonal{<:Any, <:UpperTriangular}, v::AbstractVector)
    check_solve_operand(R, v)
    for b in 1:nblocks(R)
        ldiv!(R.blocks[b], view(v, colrange(R, b)))
    end
    return v
end

function LinearAlgebra.ldiv!(
        Rt::Transpose{<:Any, <:BlockDiagonal{<:Any, <:UpperTriangular}}, v::AbstractVector
    )
    R = parent(Rt)
    check_solve_operand(R, v)
    for b in 1:nblocks(R)
        ldiv!(transpose(R.blocks[b]), view(v, colrange(R, b)))
    end
    return v
end

"""
    KroneckerCholesky{T} <: AbstractMatrix{T}

The Cholesky factor `R₁ ⊗ R₂` of `P₁ ⊗ P₂`, stored as the two factors `R₁ = chol(P₁)` and
`R₂ = chol(P₂)` (both `UpperTriangular`) and never as the `n₁n₂ × n₁n₂` product.
`(R₁ ⊗ R₂)ᵀ(R₁ ⊗ R₂) = R₁ᵀR₁ ⊗ R₂ᵀR₂ = P₁ ⊗ P₂`, and the Kronecker product of two upper
triangular matrices is upper triangular.

With `vec` column-major and `X` the `n₂×n₁` matrix holding a vector `y`,

    (R₁ ⊗ R₂)⁻¹ y = vec(R₂⁻¹ X R₁⁻ᵀ)        (R₁ ⊗ R₂)⁻ᵀ y = vec(R₂⁻ᵀ X R₁⁻¹)

so each solve is one triangular solve from each side, `O(n₁n₂(n₁ + n₂))`, through an `n₂×n₁`
scratch the factor owns. The scratch makes the factor unsafe to use from two tasks at once.
Both solves allocate nothing on a `Vector`; see [`CholeskyFactor`](@ref) for holding
`transpose(R)`.
"""
struct KroneckerCholesky{T <: Real} <: AbstractMatrix{T}
    R1::UpperTriangular{T, Matrix{T}}
    R2::UpperTriangular{T, Matrix{T}}
    # The transposes are held rather than formed per solve: `transpose` of a triangular matrix
    # allocates its wrappers on Julia 1.13. `transpose(U)` is a `LowerTriangular` over a
    # `Transpose` of the storage, which is what the solves below dispatch on.
    R1t::LowerTriangular{T, Transpose{T, Matrix{T}}}
    R2t::LowerTriangular{T, Transpose{T, Matrix{T}}}
    X::Matrix{T}

    function KroneckerCholesky{T}(R1, R2) where {T}
        U1 = convert(UpperTriangular{T, Matrix{T}}, R1)
        U2 = convert(UpperTriangular{T, Matrix{T}}, R2)
        return new{T}(
            U1, U2, transpose(U1), transpose(U2),
            Matrix{T}(undef, size(U2, 1), size(U1, 1)),
        )
    end
end

KroneckerCholesky(R1::AbstractMatrix{T}, R2::AbstractMatrix) where {T <: Real} =
    KroneckerCholesky{T}(R1, R2)

Base.size(R::KroneckerCholesky) = (n = size(R.R1, 1) * size(R.R2, 1); (n, n))

# Same ordering as `KroneckerOperator`: the second factor runs fastest.
function Base.getindex(R::KroneckerCholesky, i::Integer, j::Integer)
    @boundscheck checkbounds(R, i, j)
    n2 = size(R.R2, 1)
    i1, i2 = divrem(i - 1, n2)
    j1, j2 = divrem(j - 1, n2)
    return R.R1[i1 + 1, j1 + 1] * R.R2[i2 + 1, j2 + 1]
end

# Copied through the factor's own scratch rather than reshaped in place: `reshape` of a vector
# allocates an array header.
function LinearAlgebra.ldiv!(R::KroneckerCholesky, v::AbstractVector)
    check_solve_operand(R, v)
    copyto!(R.X, v)
    ldiv!(R.R2, R.X)
    rdiv!(R.X, R.R1t)
    copyto!(v, R.X)
    return v
end

function LinearAlgebra.ldiv!(Rt::Transpose{<:Any, <:KroneckerCholesky}, v::AbstractVector)
    R = parent(Rt)
    check_solve_operand(R, v)
    copyto!(R.X, v)
    ldiv!(R.R2t, R.X)
    rdiv!(R.X, R.R1)
    copyto!(v, R.X)
    return v
end

# `copyto!` between arrays of different length copies the shorter prefix without complaint.
function check_solve_operand(R::AbstractMatrix, v::AbstractVector)
    Base.require_one_based_indexing(v)
    length(v) == size(R, 1) ||
        throw(DimensionMismatch(lazy"v has length $(length(v)), R has $(size(R, 1)) rows"))
    return nothing
end

function cholesky_factor(K::KroneckerOperator, shift)
    # The triangular factor is the cheaper of the two and is exact only for an unshifted `P`
    # whose factors are each positive definite; `KroneckerSquareRoot` covers the rest.
    if iszero(shift)
        R1 = triangular_kronecker_factor(K.A1)
        R2 = triangular_kronecker_factor(K.A2)
        isnothing(R1) || isnothing(R2) || return KroneckerCholesky(R1, R2)
    end
    return KroneckerSquareRoot(K.A1, K.A2, shift)
end

"`cholesky_factor(A, 0)` when `A` is positive definite, and `nothing` when it is not."
function triangular_kronecker_factor(A)
    try
        return cholesky_factor(A, zero(eltype(A)))
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
end

"""
    KroneckerSquareRoot{T} <: AbstractMatrix{T}

The factor `R = D^{1/2}(U₁ ⊗ U₂)ᵀ` of `P₁ ⊗ P₂ + shift*I`, where `Pᵢ = UᵢΛᵢUᵢᵀ` and
`D = diag(λ₁ᵢλ₂ⱼ + shift)`. Stored as the two eigenvector matrices and `D`'s diagonal as a
vector of length `n₁n₂`, never as the `n₁n₂ × n₁n₂` product.

`RᵀR = (U₁ ⊗ U₂)D(U₁ ⊗ U₂)ᵀ = P₁ ⊗ P₂ + shift*I`, because `U₁ ⊗ U₂` is orthogonal and
diagonalizes `P₁ ⊗ P₂` with eigenvalues `λ₁ᵢλ₂ⱼ`, and `shift*I` is diagonal in every
orthogonal basis. The eigenvalue list is not the Kronecker product of two diagonals once
shifted, which is why it is held as a vector; that is `O(n)` and not the obstruction.

This is **not triangular**, and so not a Cholesky factor. [`CholeskyFactor`](@ref) asks only
for `RᵀR = P + shift*I` and two solves, which this meets: with `X` the `n₂×n₁` matrix holding
a vector,

    R⁻¹y = vec(U₂ X U₁ᵀ) where X holds y ./ d        R⁻ᵀy = vec(U₂ᵀ X U₁) ./ d

so each solve is two matrix products, `O(n₁n₂(n₁ + n₂))` — the same order as the triangular
factor's two triangular solves.

What this reaches that `R₁ ⊗ R₂` does not: a nonzero `shift`, since `P₁ ⊗ P₂ + εI` is not a
Kronecker product and so has no `R₁ ⊗ R₂`; and a pair of negative definite factors, whose
product is positive definite while neither factor has a Cholesky factor of its own. The
condition is on the shifted eigenvalues, `λ₁ᵢλ₂ⱼ + shift > 0` for every pair, not on the
factors' own definiteness.

Two scratch matrices the factor owns make it unsafe to solve with from two tasks at once.
"""
struct KroneckerSquareRoot{T <: Real} <: AbstractMatrix{T}
    U1::Matrix{T}
    U2::Matrix{T}
    # `D^{1/2}`'s diagonal, so a solve divides rather than taking a square root per element.
    d::Vector{T}
    X::Matrix{T}
    Y::Matrix{T}
end

function KroneckerSquareRoot(P1::AbstractMatrix, P2::AbstractMatrix, shift)
    T = promote_type(eltype(P1), eltype(P2), typeof(shift))
    E1 = eigen(Symmetric(Matrix{T}(P1)))
    E2 = eigen(Symmetric(Matrix{T}(P2)))
    eps = convert(T, shift)
    n1, n2 = length(E1.values), length(E2.values)
    d = Vector{T}(undef, n1 * n2)
    # Column-major, the second factor fastest, matching `KroneckerOperator`'s ordering.
    k = 0
    for i in 1:n1, j in 1:n2
        v = E1.values[i] * E2.values[j] + eps
        v > 0 || throw(indefinite_kronecker(E1.values[i], E2.values[j], eps))
        d[k += 1] = sqrt(v)
    end
    return KroneckerSquareRoot{T}(
        E1.vectors, E2.vectors, d, Matrix{T}(undef, n2, n1), Matrix{T}(undef, n2, n1)
    )
end

# States the condition the factor itself has, in the factors' own terms: the eigenvalues of
# `P₁ ⊗ P₂ + shift*I` are `λ₁ᵢλ₂ⱼ + shift`, so one non-positive pair is what rules the factor
# out, whatever either factor's own definiteness. The wording says "not positive definite" so a
# caller distinguishing this from an indefinite dense `P` does not have to.
function indefinite_kronecker(lambda1, lambda2, eps)
    return ArgumentError(
        lazy"P1 ⊗ P2 + $eps*I is not positive definite: its eigenvalues are the products of the factors' eigenvalues shifted, and $lambda1 * $lambda2 + $eps is not positive. A Kronecker P is positive definite when every such product is, which holds when both factors are positive definite and also when both are negative definite."
    )
end

Base.size(R::KroneckerSquareRoot) = (n = length(R.d); (n, n))

function Base.getindex(R::KroneckerSquareRoot, i::Integer, j::Integer)
    @boundscheck checkbounds(R, i, j)
    n2 = size(R.U2, 1)
    j1, j2 = divrem(j - 1, n2)
    # Row `i` of `D^{1/2}(U₁ ⊗ U₂)ᵀ` is `d[i]` times column `i` of `U₁ ⊗ U₂`.
    i1, i2 = divrem(i - 1, n2)
    return R.d[i] * R.U1[j1 + 1, i1 + 1] * R.U2[j2 + 1, i2 + 1]
end

function LinearAlgebra.ldiv!(R::KroneckerSquareRoot, v::AbstractVector)
    check_solve_operand(R, v)
    v ./= R.d
    copyto!(R.X, v)
    mul!(R.Y, R.U2, R.X)
    mul!(R.X, R.Y, transpose(R.U1))
    copyto!(v, R.X)
    return v
end

function LinearAlgebra.ldiv!(Rt::Transpose{<:Any, <:KroneckerSquareRoot}, v::AbstractVector)
    R = parent(Rt)
    check_solve_operand(R, v)
    copyto!(R.X, v)
    mul!(R.Y, transpose(R.U2), R.X)
    mul!(R.X, R.Y, R.U1)
    copyto!(v, R.X)
    v ./= R.d
    return v
end
