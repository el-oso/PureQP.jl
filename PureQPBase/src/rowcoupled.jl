"""
    RowCoupled{T} <: AbstractMatrix{T}

A constraint matrix that is a few dense rows above a block holding one entry per row.

    A = ⎡  C  ⎤   `C` is `k×n` and dense; row `i` of the lower block is
        ⎣ S ⎦   `weights[i]` in column `cols[i]` and zero elsewhere.

The shape a box-constrained problem takes when a handful of constraints couple the variables:
a budget row, a set of factor rows, a few linking rows, above bounds that each touch one
variable. The lower block's `Ãᵀ diag(ρ) Ã` is diagonal, so with a diagonal `P` the whole
reduced matrix is a diagonal plus a rank-`k` correction, which
[`DiagonalLowRank`](@ref) solves in `O(nk)` without forming it.

Nothing here requires `k` to be small — it is `size(C, 1)`, and the backend declines when it
is large enough that the correction stops paying.
"""
struct RowCoupled{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    coupling::M
    weights::V
    cols::Vector{Int}
    n::Int
    ptr::Vector{Int}     # column j holds base rows `ord[ptr[j]:ptr[j+1]-1]`
    ord::Vector{Int}

    function RowCoupled{T, M, V}(coupling, weights, cols, n) where {T, M, V}
        # `cols` indexes columns of `coupling`, and `weights[r]` pairs with the `r`th row
        # below it, so every index here is a position counted from one.
        Base.require_one_based_indexing(coupling, weights, cols)
        size(coupling, 2) == n || throw(
            DimensionMismatch(
                "coupling rows have $(size(coupling, 2)) columns, expected $n"
            )
        )
        length(weights) == length(cols) || throw(
            DimensionMismatch(
                "$(length(weights)) weights against $(length(cols)) column indices"
            )
        )
        all(j -> 1 <= j <= n, cols) || throw(
            ArgumentError("a column index falls outside 1:$n")
        )
        # Equilibration walks a column at a time, so the rows selecting each column are
        # gathered once here rather than searched for `n` times per sweep.
        counts = zeros(Int, n + 1)
        for j in cols
            counts[j + 1] += 1
        end
        ptr = cumsum(counts) .+ 1
        ord = Vector{Int}(undef, length(cols))
        fill = copy(ptr)
        k = size(coupling, 1)
        for (r, j) in pairs(cols)
            ord[fill[j]] = k + r
            fill[j] += 1
        end
        return new{T, M, V}(coupling, weights, cols, n, ptr, ord)
    end
end

"""
    RowCoupled(coupling, weights, cols)

`coupling` is the `k×n` dense block, and the `i`th remaining row is `weights[i]` in column
`cols[i]`. `n` is taken from `coupling`.
"""
function RowCoupled(coupling::AbstractMatrix{T}, weights::AbstractVector{T}, cols::AbstractVector{<:Integer}) where {T <: Real}
    idx = collect(Int, cols)
    return RowCoupled{T, typeof(coupling), typeof(weights)}(
        coupling, weights, idx, size(coupling, 2)
    )
end

"""
    RowCoupled(coupling, m₀)

The common case: `m₀` unit rows selecting the first `m₀` variables, which is how box bounds
on a prefix of the variables appear.
"""
function RowCoupled(coupling::AbstractMatrix{T}, m0::Integer) where {T <: Real}
    n = size(coupling, 2)
    m0 <= n || throw(ArgumentError("$m0 unit rows cannot select from $n columns"))
    return RowCoupled(coupling, ones(T, m0), collect(1:m0))
end

"The number of dense coupling rows, which is the rank of the correction they contribute."
coupling_rank(A::RowCoupled) = size(A.coupling, 1)

Base.size(A::RowCoupled) = (coupling_rank(A) + length(A.weights), A.n)

Base.@propagate_inbounds function Base.getindex(A::RowCoupled{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    k = coupling_rank(A)
    i <= k && return @inbounds A.coupling[i, j]
    r = i - k
    return @inbounds A.cols[r] == j ? A.weights[r] : zero(T)
end

# Products walk the two blocks rather than the `getindex` above, which a generic `mul!` would
# call `mn` times for a matrix holding `kn + m₀` entries.
function LinearAlgebra.mul!(y::AbstractVector, A::RowCoupled, x::AbstractVector)
    k = coupling_rank(A)
    mul!(view(y, 1:k), A.coupling, x)
    w, cols = A.weights, A.cols
    for r in paired(w, cols)
        y[k + r] = w[r] * x[cols[r]]
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, At::Adjoint{<:Any, <:RowCoupled}, x::AbstractVector)
    A = parent(At)
    k = coupling_rank(A)
    mul!(y, A.coupling', view(x, 1:k))
    w, cols = A.weights, A.cols
    for r in paired(w, cols)
        y[cols[r]] += w[r] * x[k + r]
    end
    return y
end

"""
    CoupledRows

The rows of one column of a [`RowCoupled`](@ref): the `k` coupling rows, then the entries of
`ord[lo:hi]`, which are the single-entry rows selecting that column.

A type of its own rather than `Iterators.flatten` over the two, whose state is a union across
the halves and so leaves a dynamic dispatch on the equilibration path that `--trim` rejects.
"""
struct CoupledRows
    k::Int
    ord::Vector{Int}
    lo::Int
    hi::Int
end

Base.eltype(::Type{CoupledRows}) = Int
Base.length(c::CoupledRows) = c.k + (c.hi - c.lo + 1)

@inline function Base.iterate(c::CoupledRows, i::Int = 1)
    i <= c.k && return (i, i + 1)
    i > length(c) && return nothing
    return (c.ord[c.lo + i - c.k - 1], i + 1)
end

# The dense rows sit above single-entry ones, so a column's nonzeros are `1:k` and whichever
# of the lower rows select it — the two together, not a range. The generic fallback would
# visit all `m` rows of every column, which is `O(mn)` against the `O(kn + m₀)` entries the
# matrix actually holds.
@inline function structural_rows(A::RowCoupled, j::Integer)
    return CoupledRows(coupling_rank(A), A.ord, A.ptr[j], A.ptr[j + 1] - 1)
end
