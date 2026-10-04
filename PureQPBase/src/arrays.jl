"""
    dense_copy(T, A) -> Matrix{T}

A dense `T` copy of `A`.

`Matrix{T}(A)` is the same value by a path `--trim` cannot resolve: its conversion reaches a
`string` call whose vararg stock inference widens to `Any`. This reaches the same result through
`copyto!`, which stays resolvable, and copies unconditionally as the constructor does — so a
caller may hand the result to anything that mutates it.
"""
dense_copy(::Type{T}, A::AbstractMatrix) where {T} =
    copyto!(Matrix{T}(undef, size(A, 1), size(A, 2)), A)

"""
    check_product_sizes(y, A, x)

Throw a `DimensionMismatch` unless `y` and `x` have the lengths `y = A * x` requires.

An operator built from blocks reaches each of them through a view of `x` or of `y`, and a vector
longer than the operator's own dimension satisfies every one of those views: the product then
answers for a prefix and the rest is read or left untouched without complaint. The lengths are
checked here instead, as `mul!` for a matrix checks them.
"""
@inline function check_product_sizes(y::AbstractVector, A, x::AbstractVector)
    (length(y) == size(A, 1) && length(x) == size(A, 2)) && return nothing
    throw(
        DimensionMismatch(
            lazy"A is $(size(A, 1))×$(size(A, 2)), so y must have length $(size(A, 1)) and x $(size(A, 2)), got $(length(y)) and $(length(x))"
        )
    )
end
