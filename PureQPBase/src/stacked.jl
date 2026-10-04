"""
    StackedOperator{T, B} <: AbstractMatrix{T}

    A = ⎡ A₁ ⎤   every block spans all `n` columns and its own rows,
        ⎢ A₂ ⎥   `rowrange(A, i)`; `vcat` of operators, held as the
        ⎣ A₃ ⎦   blocks rather than as the stack.

A stack of operators over the same variables, which is what `vcat` of constraint blocks means:
one problem's rows arriving in groups, each group with its own representation.

Holding the blocks rather than the stack is what lets a consumer reduce its work rather than only
its storage. A stack shares no row between two blocks, so a row-weighted product splits along the
blocks with no cross terms:

    Aᵀ diag(w) A = Σᵢ Aᵢᵀ diag(w[rowrange(A, i)]) Aᵢ

and each term is then whatever *its* block's representation makes it — a Kronecker block
contracts its factors, a sparse one walks its nonzeros. [`add_reduced_term!`](@ref) is where that
happens, and the saving compounds: a consumer that saw only the stack would reach every block
through `2n` products of the whole operator.

The blocks are a `Tuple`, and every traversal of them recurses on that tuple rather than indexing
it, so each step sees one concrete block type: the traversal stays type-stable, allocation-free
and `--trim` compatible even where the blocks differ from one another. A `Vector` of mixed blocks
would be a `Vector{Any}` and hold none of the three.

The type names every block, so a stack of hundreds costs compile time proportional to its length.

`work` is the intermediate the transposed product needs, so `mul!` allocates nothing. It makes an
operator single-use at a time: one `StackedOperator` must not be multiplied from two tasks at once.
"""
struct StackedOperator{T <: Real, B <: Tuple, V <: AbstractVector{T}} <: AbstractMatrix{T}
    blocks::B
    rowstart::Vector{Int}    # rowstart[i] is the first row of block i; rowstart[K+1] is m+1
    cols::Int
    work::V                  # `n`, holds one block's `Aᵢᵀxᵢ` while it is added to the sum

    function StackedOperator{T, B}(blocks::B) where {T, B <: Tuple}
        isempty(blocks) && throw(ArgumentError("a StackedOperator needs at least one block"))
        cols = size(first(blocks), 2)
        rowstart = Vector{Int}(undef, length(blocks) + 1)
        rowstart[1] = 1
        for (i, Ai) in pairs(blocks)
            size(Ai, 2) == cols || throw(
                DimensionMismatch(
                    lazy"every block of a StackedOperator spans the same columns: block $i has $(size(Ai, 2)), the first has $cols"
                )
            )
            rowstart[i + 1] = rowstart[i] + size(Ai, 1)
        end
        work = similar(first(blocks), T, cols)
        return new{T, B, typeof(work)}(blocks, rowstart, cols, work)
    end
end

"""
    StackedOperator(blocks...)

The blocks stacked in order, as `vcat` would stack them. Every block spans the same columns.
"""
function StackedOperator(blocks::AbstractMatrix...)
    T = promote_type(map(eltype, blocks)...)
    T <: Real || throw(ArgumentError(lazy"a StackedOperator needs a real element type, got $T"))
    return StackedOperator{T, typeof(blocks)}(blocks)
end

"The number of blocks in the stack."
nblocks(A::StackedOperator) = length(A.blocks)

"The rows block `i` occupies."
rowrange(A::StackedOperator, i::Integer) = A.rowstart[i]:(A.rowstart[i + 1] - 1)

Base.size(A::StackedOperator) = (A.rowstart[end] - 1, A.cols)

# A stack stores its blocks, so forming the matrix would hold what the blocks already hold, and
# in a representation that has lost which rows belong to which block.
holds_structure(::StackedOperator) = true

# The block holding row `i`, and that row's index within it. A linear scan: a stack has a handful
# of blocks, and this keeps the search allocation-free and free of the bookkeeping a search tree
# would add for a dozen entries.
@inline function block_of_row(A::StackedOperator, i::Integer)
    for b in 1:nblocks(A)
        i < A.rowstart[b + 1] && return (b, i - A.rowstart[b] + 1)
    end
    return (nblocks(A), i - A.rowstart[nblocks(A)] + 1)
end

function Base.getindex(A::StackedOperator, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    b, ib = block_of_row(A, i)
    return stacked_block_entry(A.blocks, b, ib, j)
end

# The blocks are a tuple, so the index is not a compile-time constant and `A.blocks[b]` would
# infer as their union. This recursion is unrolled and each arm reads one concrete block.
@inline stacked_block_entry(blocks::Tuple{}, b, i, j) = error("block index out of range")
@inline stacked_block_entry(blocks::Tuple, b, i, j) =
    isone(b) ? first(blocks)[i, j] : stacked_block_entry(Base.tail(blocks), b - 1, i, j)

# The blocks are walked by recursion on the tuple rather than by an index into it: `blocks[i]`
# for a runtime `i` infers as the union of the block types, which boxes and costs a dynamic
# dispatch, where this recursion is unrolled at compile time and each arm sees one concrete block.
# Measured on a stack whose blocks differ: 304 bytes per product through the index, 0 through this.
LinearAlgebra.mul!(y::AbstractVector, A::StackedOperator, x::AbstractVector) =
    stack_mul!(y, A.blocks, x, 1)

@inline stack_mul!(y, ::Tuple{}, x, off) = y
@inline function stack_mul!(y, blocks::Tuple, x, off)
    A1 = first(blocks)
    rows = size(A1, 1)
    # A block must be able to write into a `view`: its rows are a slice of the stack's result.
    mul!(view(y, off:(off + rows - 1)), A1, x)
    return stack_mul!(y, Base.tail(blocks), x, off + rows)
end

# `Aᵀy = Σᵢ Aᵢᵀ y[rowrange(A, i)]`. The first block writes the sum and the rest add to it, so `y`
# needs no zeroing.
function LinearAlgebra.mul!(
        y::AbstractVector, At::Union{Adjoint{<:Any, <:StackedOperator}, Transpose{<:Any, <:StackedOperator}},
        x::AbstractVector
    )
    A = parent(At)
    A1 = first(A.blocks)
    rows = size(A1, 1)
    mul!(y, adjoint(A1), view(x, 1:rows))
    return stack_mul_adjoint!(y, Base.tail(A.blocks), x, 1 + rows, A.work)
end

@inline stack_mul_adjoint!(y, ::Tuple{}, x, off, work) = y

@inline function stack_mul_adjoint!(y, blocks::Tuple, x, off, work)
    A1 = first(blocks)
    rows = size(A1, 1)
    # Three-argument `mul!` into scratch, then added: the five-argument form that would accumulate
    # in place is not defined for every operator a block may be, and its generic fallback reads
    # entries one at a time — which an operator supplying only products refuses outright.
    mul!(work, adjoint(A1), view(x, off:(off + rows - 1)))
    for j in eachindex(y)
        y[j] += work[j]
    end
    return stack_mul_adjoint!(y, Base.tail(blocks), x, off + rows, work)
end

"Row `i` of the stack is a row of the one block that holds it."
function dense_row!(dest::AbstractVector, A::StackedOperator, i::Integer)
    check_row_dest(dest, A)
    @boundscheck checkbounds(A, i, :)
    b, ib = block_of_row(A, i)
    return stacked_block_row!(dest, A.blocks, b, ib)
end

@inline stacked_block_row!(dest, blocks::Tuple{}, b, i) = error("block index out of range")
@inline stacked_block_row!(dest, blocks::Tuple, b, i) =
    isone(b) ? dense_row!(dest, first(blocks), i) :
    stacked_block_row!(dest, Base.tail(blocks), b - 1, i)

# Readable exactly when every block is.
is_materializable(A::StackedOperator) = all(is_materializable, A.blocks)

"""
    add_reduced_term!(R, T, A::StackedOperator, weights, D, n, m, scratch, ej, av, col) -> R

Add `D Aᵀ diag(weights) A D` into `R`, one block at a time.

No row is shared between two blocks, so the sum has no cross terms and each block contributes
`Aᵢᵀ diag(w[rowrange(A, i)]) Aᵢ` over the same `n` columns. Each contribution goes through
`add_reduced_term!` again, so a block keeps whatever reduction its own representation has.
"""
add_reduced_term!(
    R::AbstractMatrix{T}, ::Type{T}, A::StackedOperator, weights::AbstractVector,
    D::AbstractVector, n::Integer, m::Integer, scratch, ej, av, col
) where {T} = stack_reduced_term!(R, T, A.blocks, weights, D, n, scratch, ej, av, col, 1)

@inline stack_reduced_term!(R, ::Type{T}, ::Tuple{}, w, D, n, ::Tuple{}, ej, av, col, off) where {T} = R

# The blocks and their scratch are walked together, so each arm reads one concrete block and the
# scratch its own representation asked for.
@inline function stack_reduced_term!(
        R, ::Type{T}, blocks::Tuple, w, D, n, scratch::Tuple, ej, av, col, off
    ) where {T}
    A1 = first(blocks)
    rows = size(A1, 1)
    rng = off:(off + rows - 1)
    add_reduced_term!(
        R, T, A1, view(w, rng), D, n, rows, first(scratch), ej, view(av, rng), col
    )
    return stack_reduced_term!(
        R, T, Base.tail(blocks), w, D, n, Base.tail(scratch), ej, av, col, off + rows
    )
end

reduced_term_scratch(::Type{T}, A::StackedOperator) where {T} =
    map(Ai -> reduced_term_scratch(T, Ai), A.blocks)

"""
    reduced_diagonal!(dest, T, P, A::StackedOperator, rho, E, D, sigma, c) -> dest

The inverted diagonal of the reduced matrix, taking each block's contribution from the block.

`Aᵀ diag(ρ) A` is a sum over the blocks, so its diagonal is the sum of theirs, and each block
supplies its own through [`structural_rows`](@ref): a block that declares its structure touches
only the rows it has. A block that cannot be indexed at all leaves the diagonal unavailable, and
conjugate gradients then run unpreconditioned, which costs iterations rather than the answer.
"""
function reduced_diagonal!(dest, ::Type{T}, P, A::StackedOperator, rho, E, D, sigma, c) where {T}
    is_materializable(A) || return unpreconditioned!(dest)
    # `rho` and `E` are workspace vectors indexed by the same `i` that indexes the stack's rows,
    # so a block's rows are offset into them rather than read from one.
    Base.require_one_based_indexing(rho, E)
    for j in eachindex(dest)
        dj = D[j]
        dest[j] = c * dj * T(P[j, j]) * dj + sigma
    end
    stacked_diagonal!(dest, T, A.blocks, rho, E, D, 1)
    for j in eachindex(dest)
        dest[j] = inv(max(dest[j], sqrt(eps(T))))
    end
    return dest
end

@inline stacked_diagonal!(dest, ::Type{T}, ::Tuple{}, rho, E, D, off) where {T} = dest

@inline function stacked_diagonal!(dest, ::Type{T}, blocks::Tuple, rho, E, D, off) where {T}
    A1 = first(blocks)
    for j in eachindex(dest)
        dj = D[j]
        acc = zero(T)
        for ib in structural_rows(A1, j)
            i = off + ib - 1
            a = E[i] * T(A1[ib, j]) * dj
            acc += rho[i] * a * a
        end
        dest[j] += acc
    end
    return stacked_diagonal!(dest, T, Base.tail(blocks), rho, E, D, off + size(A1, 1))
end
