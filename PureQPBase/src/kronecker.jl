"""
    KroneckerOperator{T,M1,M2} <: AbstractMatrix{T}

The constraint matrix `A₁ ⊗ A₂`, stored as its two factors.

    A x = vec(A₂ X A₁ᵀ),   X = reshape(x, n₂, n₁)

so a product is two small matrix multiplications rather than one large one:
`O(n₁n₂(m₁ + m₂))` against the `O(m₁m₂n₁n₂)` a formed `A` would cost, out of `m₁n₁ + m₂n₂`
stored against `m₁m₂n₁n₂`.

The shape a separable operator takes — a two-dimensional transform applied along each axis in
turn, which is what a blur kernel, a tensor regression design, or a discretization on a
product grid gives.

`scratch1` and `scratch2` are the intermediates the two products need, so `mul!` allocates
nothing. They make an operator single-use at a time: one `KroneckerOperator` must not be
multiplied from two tasks at once.
"""
struct KroneckerOperator{T <: Real, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A1::M
    A2::M
    scratch1::M     #`m₂×n₁`, holds `A₂ X`
    scratch2::M     #`n₂×m₁`, holds `A₂ᵀ Y`
    xmat::M         # `n₂×n₁`, the operand of a product laid out as a matrix
    ymat::M         # `m₂×m₁`, the result laid out as a matrix
end

"""
    KroneckerOperator(A1, A2)

The operator `A1 ⊗ A2`, which is `size(A1, 1) * size(A2, 1)` by `size(A1, 2) * size(A2, 2)`.
"""
function KroneckerOperator(A1::M, A2::M) where {T <: Real, M <: AbstractMatrix{T}}
    Base.require_one_based_indexing(A1, A2)
    m1, n1 = size(A1)
    m2, n2 = size(A2)
    return KroneckerOperator{T, M}(
        A1, A2,
        similar(A2, T, m2, n1), similar(A2, T, n2, m1),
        similar(A2, T, n2, n1), similar(A2, T, m2, m1),
    )
end

"""
    KroneckerOperator(A1, A2)

Both factors must have the same matrix type. One type parameter rather than four keeps the
`OperatorSplittingWorkspace` type small enough for `--trim`'s inference to resolve the backend union rather
than widen it, which is what a fourth parameter cost.
"""
KroneckerOperator(A1::AbstractMatrix, A2::AbstractMatrix) =
    throw(ArgumentError("both Kronecker factors must have the same matrix type"))

"The factors, as `(A1, A2)`."
factors(K::KroneckerOperator) = (K.A1, K.A2)

# Two factors of `n₁²` and `n₂²` entries stand for `n₁²n₂²`, so a backend that formed the
# product would hold the square of what this type does.
holds_structure(::KroneckerOperator) = true

Base.size(K::KroneckerOperator) =
    (size(K.A1, 1) * size(K.A2, 1), size(K.A1, 2) * size(K.A2, 2))

# `(A₁ ⊗ A₂)[i, j] = A₁[i₁, j₁] · A₂[i₂, j₂]`, with the second factor running fastest, which
# is the ordering `vec` of a column-major `n₂×n₁` matrix gives.
function Base.getindex(K::KroneckerOperator{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(K, i, j)
    m2, n2 = size(K.A2)
    i1, i2 = divrem(i - 1, m2)
    j1, j2 = divrem(j - 1, n2)
    return K.A1[i1 + 1, j1 + 1] * K.A2[i2 + 1, j2 + 1]
end

# Copied through the operator's own matrix scratch rather than reshaped in place: `reshape` of
# a vector allocates an array header, and these run every iteration. The copies are `O(n)` and
# `O(m)` against two matrix products.
function LinearAlgebra.mul!(y::AbstractVector, K::KroneckerOperator, x::AbstractVector)
    Base.require_one_based_indexing(y, x)
    copyto!(K.xmat, x)
    mul!(K.scratch1, K.A2, K.xmat)    # m₂×n₁
    mul!(K.ymat, K.scratch1, K.A1')   # m₂×m₁
    copyto!(y, K.ymat)
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector, Kt::Adjoint{<:Any, <:KroneckerOperator}, x::AbstractVector
    )
    K = parent(Kt)
    Base.require_one_based_indexing(y, x)
    copyto!(K.ymat, x)
    mul!(K.scratch2, K.A2', K.ymat)   # n₂×m₁
    mul!(K.xmat, K.scratch2, K.A1)    # n₂×n₁
    copyto!(y, K.xmat)
    return y
end

# `setup` asks a `P` whether it is symmetric and convex before any backend exists. The generic
# answers walk `n²` entries and factor an `n×n` copy; a Kronecker product answers both from
# its two factors.

# `(P₁ ⊗ P₂)ᵀ = P₁ᵀ ⊗ P₂ᵀ`, so symmetric factors give a symmetric product. The converse fails
# only for a product of two antisymmetric factors, which this reports as not symmetric: the
# answer errs toward refusing, and such a `P` can be passed as its own matrix.
is_symmetric(K::KroneckerOperator) = issymmetric(K.A1) && issymmetric(K.A2)

# The eigenvalues of `P₁ ⊗ P₂` are the products `λᵢ(P₁) λⱼ(P₂)`. A product of two reals is
# bilinear, so its extremes over all pairs sit at the extremes of each factor's spectrum, and
# `P + σI` is positive definite exactly when the smallest of those four products exceeds `-σ`.
# Two eigenvalue problems of the factors' sizes replace a factorization of the `n×n` product.
function is_convex(::Type{T}, K::KroneckerOperator, sigma) where {T}
    isempty(K) && return true
    lo1, hi1 = extrema(eigvals(Symmetric(Matrix{T}(K.A1))))
    lo2, hi2 = extrema(eigvals(Symmetric(Matrix{T}(K.A2))))
    return min(lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2) + sigma > zero(T)
end

"""
    is_scalar_multiple(P) -> Bool

Whether `P` is `μI` for some `μ`, which is what the Kronecker backend needs of it.

Not merely a structured `P`: `c·D·P·D` and `ρ·Ãᵀ diag(ρ) Ã` are simultaneously diagonalizable
only when the first is a scalar multiple of the identity. A Kronecker `P` does not qualify,
which is measured rather than assumed — see the log.
"""
is_scalar_multiple(P) = false
is_scalar_multiple(P::Diagonal) = isempty(P.diag) || all(==(first(P.diag)), P.diag)
is_scalar_multiple(::UniformScaling) = true

"""
    scalar_multiple(P) -> Real

The `μ` for which `P == μI`. Defined only where [`is_scalar_multiple`](@ref) holds, and the
caller checks that first.

Split from the predicate rather than returning `Union{Nothing,T}` for it: a union-typed `μ`
flowing into the arithmetic below is a call `--trim` will not resolve, however obviously the
`isnothing` check narrows it.
"""
scalar_multiple(P::Diagonal{T}) where {T <: Real} =
    isempty(P.diag) ? zero(T) : first(P.diag)
scalar_multiple(P::UniformScaling{T}) where {T <: Real} = P.λ
