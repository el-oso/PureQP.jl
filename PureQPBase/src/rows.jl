"""
    dense_row!(dest, A, i) -> dest

Write row `i` of `A` into `dest`, which must have the axes of `A`'s columns.

The row-side counterpart of [`structural_rows`](@ref): a consumer that reads a matrix by rows
asks for them here, and each representation answers in its own cost.

  - any `AbstractMatrix`, dense or not: one read per column;
  - [`KroneckerOperator`](@ref): the outer product of the matching rows of the two factors,
    `n` multiplies and no `n×m` product formed;
  - [`BlockDiagonal`](@ref): the one block's row placed in its column range, zero elsewhere;
  - [`ProductOperator`](@ref): one adjoint product against a basis vector, which costs what the
    wrapped operator's adjoint product costs and allocates nothing when that does not.

A `ProductOperator` answers through its own scratch column, so one such operator must not be
asked for rows from two tasks at once.
"""
function dense_row!(dest::AbstractVector, A::AbstractMatrix, i::Integer)
    check_row_dest(dest, A)
    for j in axes(A, 2)
        dest[j] = A[i, j]
    end
    return dest
end

"""
    rows_operand(T, A) -> A, a row-readable form of it, or Matrix{T}

What a consumer that reads `A` by rows should hold.

The generic answer is a dense copy, because the generic [`dense_row!`](@ref) reads one entry per
column and a representation that makes that expensive is cheaper to densify once. A
representation with a row-readable form of its own overrides this and returns that form: the
SparseArrays extension returns one holding the transpose, where a row is a column.

A representation the consumer already reads directly never reaches this.
"""
rows_operand(::Type{T}, A::AbstractMatrix) where {T} = convert(Matrix{T}, A)

# `(A₁ ⊗ A₂)[i, (j₁, j₂)] = A₁[i₁, j₁] A₂[i₂, j₂]`, where `i = (i₁ - 1) m₂ + i₂` and the
# second factor's column index runs fastest, so `dest` laid out as `n₂×n₁` is the outer
# product of row `i₂` of `A₂` with row `i₁` of `A₁`. Loops rather than a reshaped view: the
# view's array header is a heap allocation.
function dense_row!(dest::AbstractVector, K::KroneckerOperator, i::Integer)
    check_row_dest(dest, K)
    checkbounds(K, i, :)
    A1, A2 = K.A1, K.A2
    i1, i2 = divrem(i - 1, size(A2, 1))
    n2 = size(A2, 2)
    for j1 in axes(A1, 2)
        a = A1[i1 + 1, j1]
        offset = (j1 - 1) * n2
        for j2 in axes(A2, 2)
            dest[offset + j2] = a * A2[i2 + 1, j2]
        end
    end
    return dest
end

function dense_row!(dest::AbstractVector, A::BlockDiagonal, i::Integer)
    check_row_dest(dest, A)
    checkbounds(A, i, :)
    b = block_of_row(A, i)
    fill!(dest, zero(eltype(dest)))
    dense_row!(view(dest, colrange(A, b)), A.blocks[b], i - A.rowstart[b] + 1)
    return dest
end

function dense_row!(dest::AbstractVector, M::ProductOperator{T}, i::Integer) where {T}
    check_row_dest(dest, M)
    checkbounds(M, i, :)
    fill!(M.column, zero(T))
    M.column[i] = one(T)
    mul!(dest, M.opt, M.column)
    return dest
end

# `lazy` defers the message's formatting past the call graph `--trim` verifies.
function check_row_dest(dest::AbstractVector, A::AbstractMatrix)
    axes(dest, 1) == axes(A, 2) || throw(
        DimensionMismatch(
            lazy"a row of a matrix with columns $(axes(A, 2)) cannot be written into a vector with axes $(axes(dest, 1))"
        )
    )
    return nothing
end
