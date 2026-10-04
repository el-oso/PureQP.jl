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
