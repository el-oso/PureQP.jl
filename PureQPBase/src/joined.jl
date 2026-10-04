"""
    JoinedOperator{T, B, V} <: AbstractMatrix{T}

    A = [ A₁ A₂ A₃ ]   every block spans all `m` rows and its own columns,
                       `colrange(A, i)`; `hcat` of operators, held as the
                       blocks rather than as the join.

A join of operators over the same rows, which is what `hcat` of constraint blocks means: one
problem's variables arriving in groups, each group's columns with their own representation.

The horizontal counterpart of [`StackedOperator`](@ref), and it differs from the stack in what the
reduction can do with it. A stack shares no row between blocks, so its row-weighted product is a
sum of the blocks' own. A join shares every row, so

    Aᵀ diag(w) A = ⎡ A₁ᵀ W A₁  A₁ᵀ W A₂ ⎤
                   ⎣ A₂ᵀ W A₁  A₂ᵀ W A₂ ⎦

has a cross block for every pair, and no block's representation owns one. There is no
`add_reduced_term!` method here: the generic one reaches each column of the reduced matrix as
`Aᵀ(w ⊙ (A eⱼ))`, two products of the join, and that is what a `ProductReduced` backend over a
join costs. A method forming the blocks pairwise would cost `Σᵢ nᵢ(fᵢ + aᵢ) + Σᵢ<ⱼ min(nᵢ, nⱼ)(fⱼ + aᵢ)`
for blocks whose forward and adjoint products cost `fᵢ` and `aᵢ`, against the generic
`n · Σᵢ(fᵢ + aᵢ)`; it is worth writing only for a backend that reaches `add_reduced_term!` with a
join, which the dense terminals below do not (see `docs/design/horizontal-operators.md`).

[`holds_structure`](@ref) is `false`: the join's blocks are each a slice of the columns, and the
dense backends serve it directly, as they serve a matrix, with the blocks supplying the entries.
What the join keeps is each block's own products, rows and entries, so a Kronecker block still
contracts and a constant block still costs nothing.

The blocks are a `Tuple`, and every traversal of them recurses on that tuple rather than indexing
it, so each step sees one concrete block type: the traversal stays type-stable, allocation-free
and `--trim` compatible even where the blocks differ from one another. A `Vector` of mixed blocks
would be a `Vector{Any}` and hold none of the three.

The type names every block, so a join of hundreds costs compile time proportional to its length.

`work` is the intermediate the forward product needs, so `mul!` allocates nothing. It makes an
operator single-use at a time: one `JoinedOperator` must not be multiplied from two tasks at once.
"""
struct JoinedOperator{T <: Real, B <: Tuple, V <: AbstractVector{T}} <: AbstractMatrix{T}
    blocks::B
    colstart::Vector{Int}    # colstart[i] is the first column of block i; colstart[K+1] is n+1
    rows::Int
    work::V                  # `m`, holds one block's `Aᵢxᵢ` while it is added to the sum

    function JoinedOperator{T, B}(blocks::B) where {T, B <: Tuple}
        isempty(blocks) && throw(ArgumentError("a JoinedOperator needs at least one block"))
        rows = size(first(blocks), 1)
        colstart = Vector{Int}(undef, length(blocks) + 1)
        colstart[1] = 1
        for (i, Ai) in pairs(blocks)
            size(Ai, 1) == rows || throw(
                DimensionMismatch(
                    lazy"every block of a JoinedOperator spans the same rows: block $i has $(size(Ai, 1)), the first has $rows"
                )
            )
            colstart[i + 1] = colstart[i] + size(Ai, 2)
        end
        work = similar(first(blocks), T, rows)
        return new{T, B, typeof(work)}(blocks, colstart, rows, work)
    end
end

"""
    JoinedOperator(blocks...)

The blocks joined in order, as `hcat` would join them. Every block spans the same rows.
"""
function JoinedOperator(blocks::AbstractMatrix...)
    T = promote_type(map(eltype, blocks)...)
    T <: Real || throw(ArgumentError(lazy"a JoinedOperator needs a real element type, got $T"))
    return JoinedOperator{T, typeof(blocks)}(blocks)
end

"The number of blocks in the join."
nblocks(A::JoinedOperator) = length(A.blocks)

"The columns block `i` occupies."
colrange(A::JoinedOperator, i::Integer) = A.colstart[i]:(A.colstart[i + 1] - 1)

Base.size(A::JoinedOperator) = (A.rows, A.colstart[end] - 1)

# The block holding column `j`, and that column's index within it. A linear scan, as
# `StackedOperator`'s `block_of_row` is: a join has a handful of blocks.
@inline function block_of_column(A::JoinedOperator, j::Integer)
    for b in 1:nblocks(A)
        j < A.colstart[b + 1] && return (b, j - A.colstart[b] + 1)
    end
    return (nblocks(A), j - A.colstart[nblocks(A)] + 1)
end

function Base.getindex(A::JoinedOperator, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    b, jb = block_of_column(A, j)
    return joined_block_entry(A.blocks, b, i, jb)
end

# The blocks are a tuple, so the index is not a compile-time constant and `A.blocks[b]` would
# infer as their union. This recursion is unrolled and each arm reads one concrete block.
@inline joined_block_entry(blocks::Tuple{}, b, i, j) = error("block index out of range")
@inline joined_block_entry(blocks::Tuple, b, i, j) =
    isone(b) ? first(blocks)[i, j] : joined_block_entry(Base.tail(blocks), b - 1, i, j)

# `Ax = Σᵢ Aᵢ x[colrange(A, i)]`. The first block writes the sum and the rest add to it through
# `work`, so `y` needs no zeroing.
function LinearAlgebra.mul!(y::AbstractVector, A::JoinedOperator, x::AbstractVector)
    A1 = first(A.blocks)
    cols = size(A1, 2)
    mul!(y, A1, view(x, 1:cols))
    return join_mul!(y, Base.tail(A.blocks), x, 1 + cols, A.work)
end

@inline join_mul!(y, ::Tuple{}, x, off, work) = y

@inline function join_mul!(y, blocks::Tuple, x, off, work)
    A1 = first(blocks)
    cols = size(A1, 2)
    # Three-argument `mul!` into scratch, then added: the five-argument form that would accumulate
    # in place is not defined for every operator a block may be, and its generic fallback reads
    # entries one at a time — which an operator supplying only products refuses outright.
    mul!(work, A1, view(x, off:(off + cols - 1)))
    for i in eachindex(y)
        y[i] += work[i]
    end
    return join_mul!(y, Base.tail(blocks), x, off + cols, work)
end

# `Aᵀy` splits along the blocks with no cross terms: `(Aᵀy)[colrange(A, i)] = Aᵢᵀ y`.
function LinearAlgebra.mul!(
        y::AbstractVector, At::Union{Adjoint{<:Any, <:JoinedOperator}, Transpose{<:Any, <:JoinedOperator}},
        x::AbstractVector
    )
    return join_mul_adjoint!(y, parent(At).blocks, x, 1)
end

@inline join_mul_adjoint!(y, ::Tuple{}, x, off) = y

@inline function join_mul_adjoint!(y, blocks::Tuple, x, off)
    A1 = first(blocks)
    cols = size(A1, 2)
    # A block must be able to write into a `view`: its columns are a slice of the join's result.
    mul!(view(y, off:(off + cols - 1)), adjoint(A1), x)
    return join_mul_adjoint!(y, Base.tail(blocks), x, off + cols)
end

"Row `i` of the join is the blocks' rows `i`, side by side."
function dense_row!(dest::AbstractVector, A::JoinedOperator, i::Integer)
    check_row_dest(dest, A)
    @boundscheck checkbounds(A, i, :)
    return joined_row!(dest, A.blocks, i, 1)
end

@inline joined_row!(dest, ::Tuple{}, i, off) = dest

@inline function joined_row!(dest, blocks::Tuple, i, off)
    A1 = first(blocks)
    cols = size(A1, 2)
    dense_row!(view(dest, off:(off + cols - 1)), A1, i)
    return joined_row!(dest, Base.tail(blocks), i, off + cols)
end

# A column belongs to one block, and the block's rows are the join's rows, so the block's own
# answer is the join's. Equilibration and the reduced diagonal then cost each column what its block
# costs, rather than a walk over all `m` rows.
@inline function structural_rows(A::JoinedOperator, j::Integer)
    b, jb = block_of_column(A, j)
    return joined_structural_rows(A.blocks, b, jb)
end

@inline joined_structural_rows(blocks::Tuple{}, b, j) = error("block index out of range")
@inline joined_structural_rows(blocks::Tuple, b, j) =
    isone(b) ? structural_rows(first(blocks), j) :
    joined_structural_rows(Base.tail(blocks), b - 1, j)

# Readable exactly when every block is.
is_materializable(A::JoinedOperator) = all(is_materializable, A.blocks)

# Each block checks its own entries in its own way, so a Kronecker block checks its factors.
function check_finite(M::JoinedOperator, rows::Integer, cols::Integer, name::String)
    return joined_check_finite(M.blocks, name, 1)
end

@inline joined_check_finite(::Tuple{}, name, i) = nothing

@inline function joined_check_finite(blocks::Tuple, name, i)
    A1 = first(blocks)
    check_finite(A1, size(A1, 1), size(A1, 2), name * "'s block " * string(i))
    return joined_check_finite(Base.tail(blocks), name, i + 1)
end

"""
    reduced_diagonal!(dest, T, P, A::JoinedOperator, rho, E, D, sigma, c) -> dest

The inverted diagonal of the reduced matrix, each column read from the block that holds it.

`structural_rows` of a join is column-local, so the entry walk touches only the rows the owning
block has. A block that cannot be indexed at all leaves the diagonal unavailable, and conjugate
gradients then run unpreconditioned, which costs iterations rather than the answer.
"""
reduced_diagonal!(dest, ::Type{T}, P, A::JoinedOperator, rho, E, D, sigma, c) where {T} =
    is_materializable(A) ? indexed_reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c) :
    unpreconditioned!(dest)
