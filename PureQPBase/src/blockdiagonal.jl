"""
    BlockDiagonal{T,M} <: AbstractMatrix{T}

A matrix that is a diagonal run of blocks, stored as the blocks and nothing else.

    A = ⎡ A₁          ⎤   block `i` occupies rows `rowrange(A, i)`
        ⎢    A₂       ⎥   and columns `colrange(A, i)`; everything
        ⎣       ⋱     ⎦   outside those runs is a structural zero.

The shape a decoupled problem takes: independent subsystems that share only the objective's
form, such as scenarios in a stochastic program or periods that no constraint links. Storage
is `Σ mᵢnᵢ` against `mn`. Nothing of size `m×n` or `n×n` is formed at any point, which takes
block-wise [`is_symmetric`](@ref) and [`is_convex`](@ref) methods as well as a backend: those
two run in `setup` before a backend is chosen, and their generic methods reach for entries.

The structure survives into the reduced matrix. `Ãᵀ diag(ρ) Ã` is block-diagonal on the same
column partition, so with a `P` partitioned the same way

    R = c D P D + σI + Ãᵀ diag(ρ) Ã

decouples into `K` independent systems that [`BlockReduced`](@ref) factors and solves one at a
time. That is `Σ nᵢ³` against `n³` to factor and `Σ nᵢ²` against `n²` to store — for `K` equal
blocks, a factor of `K²` and `K`.

Equilibration needs no probing here: a column's nonzeros are its own block's rows, which
[`structural_rows`](@ref) answers directly.
"""
struct BlockDiagonal{T <: Real, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{M}
    rowstart::Vector{Int}    # rowstart[i] is the first row of block i; rowstart[K+1] is m+1
    colstart::Vector{Int}    # likewise for columns

    function BlockDiagonal{T, M}(blocks::Vector{M}) where {T, M}
        isempty(blocks) && throw(ArgumentError("a BlockDiagonal needs at least one block"))
        rowstart = Vector{Int}(undef, length(blocks) + 1)
        colstart = Vector{Int}(undef, length(blocks) + 1)
        rowstart[1] = colstart[1] = 1
        for (i, B) in pairs(blocks)
            rowstart[i + 1] = rowstart[i] + size(B, 1)
            colstart[i + 1] = colstart[i] + size(B, 2)
        end
        return new{T, M}(blocks, rowstart, colstart)
    end
end

"""
    BlockDiagonal(blocks)

The block-diagonal matrix whose diagonal runs through `blocks`, in order.
"""
BlockDiagonal(blocks::Vector{M}) where {T <: Real, M <: AbstractMatrix{T}} =
    BlockDiagonal{T, M}(blocks)
BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix}) = BlockDiagonal(collect(blocks))

"The number of blocks on the diagonal."
nblocks(A::BlockDiagonal) = length(A.blocks)

# The off-diagonal blocks are zero and are not stored, so forming the matrix would hold them.
holds_structure(::BlockDiagonal) = true

"The rows block `i` occupies."
rowrange(A::BlockDiagonal, i::Integer) = A.rowstart[i]:(A.rowstart[i + 1] - 1)

"The columns block `i` occupies."
colrange(A::BlockDiagonal, i::Integer) = A.colstart[i]:(A.colstart[i + 1] - 1)

Base.size(A::BlockDiagonal) = (A.rowstart[end] - 1, A.colstart[end] - 1)

"""
    block_of_column(A, j) -> Int

The block holding column `j`, found by binary search over the block starts.
"""
function block_of_column(A::BlockDiagonal, j::Integer)
    return searchsortedlast(A.colstart, j)
end

"""
    block_of_row(A, i) -> Int

The block holding row `i`.
"""
function block_of_row(A::BlockDiagonal, i::Integer)
    return searchsortedlast(A.rowstart, i)
end

function Base.getindex(A::BlockDiagonal{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    bi = block_of_row(A, i)
    bi == block_of_column(A, j) || return zero(T)
    return A.blocks[bi][i - A.rowstart[bi] + 1, j - A.colstart[bi] + 1]
end

# Products run block by block over `Σ mᵢnᵢ` entries; a generic `mul!` would call the
# `getindex` above `mn` times, most of them landing on a structural zero.
function LinearAlgebra.mul!(y::AbstractVector, A::BlockDiagonal, x::AbstractVector)
    # The block boundaries are positions counted from one, so the operands are indexed from
    # one too. Declared rather than assumed, since the signature invites any `AbstractVector`.
    Base.require_one_based_indexing(y, x)
    for i in eachindex(A.blocks)
        mul!(view(y, rowrange(A, i)), A.blocks[i], view(x, colrange(A, i)))
    end
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector, At::Adjoint{<:Any, <:BlockDiagonal}, x::AbstractVector
    )
    A = parent(At)
    Base.require_one_based_indexing(y, x)
    for i in eachindex(A.blocks)
        mul!(view(y, colrange(A, i)), A.blocks[i]', view(x, rowrange(A, i)))
    end
    return y
end

# A column's nonzeros are exactly its own block's rows, so equilibration walks `Σ mᵢnᵢ`
# entries rather than the `mn` the generic answer would give.
@inline structural_rows(A::BlockDiagonal, j::Integer) = rowrange(A, block_of_column(A, j))

"""
    same_column_partition(P, A) -> Bool

Whether `P` and `A` split their columns at the same places, which is what makes the reduced
matrix decouple. `P` must also be square block by block, since each block of `R` is formed
from the corresponding blocks of both.
"""
function same_column_partition(P::BlockDiagonal, A::BlockDiagonal)
    P.colstart == A.colstart || return false
    return P.rowstart == P.colstart
end

# `setup` tests symmetry and convexity of `P` before any backend is chosen, and the generic
# methods reach for entries: `issymmetric` walks all `n²` positions, and `is_convex` builds
# `Matrix(P) + σI` and factors it densely. Both would form the `n×n` object this type exists
# to avoid, on every `setup` and on every `update!` that replaces `P`. A block-diagonal matrix
# is symmetric exactly when its blocks are square and each is symmetric, and `P + σI` is
# positive definite exactly when each `Pᵢ + σI` is, so both cost `Σ` over the blocks.
function is_symmetric(P::BlockDiagonal)
    P.rowstart == P.colstart || return false
    return all(issymmetric, P.blocks)
end

function is_convex(::Type{T}, P::BlockDiagonal, sigma) where {T}
    P.rowstart == P.colstart || return false
    return all(P.blocks) do B
        isempty(B) || issuccess(cholesky(Symmetric(Matrix{T}(B)) + sigma * I; check = false))
    end
end
