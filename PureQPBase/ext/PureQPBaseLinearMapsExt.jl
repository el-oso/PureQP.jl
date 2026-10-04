"""
Accepts a `LinearMaps.LinearMap` wherever [`PureQPBase.setup`](@ref) takes a matrix.

A `LinearMap` is not an `AbstractMatrix`, so it reaches the solver either as the base's own
representation of what it holds or through [`PureQPBase.ProductOperator`](@ref). A wrapped
matrix, a Kronecker product, a block-diagonal of wrapped matrices, a constant block, a uniform
scaling, a concatenation and a scalar multiple of any of these arrive as the base's own
representation, so the algorithms' structured paths see them. Every other map is wrapped. The protocol the
wrapper implements lives in `PureQPBase/src/operator.jl` and needs no dependency.

What loading LinearMaps buys over wrapping by hand is the two declarations the wrapper cannot
compute: LinearMaps tracks `issymmetric` and `isposdef` on its maps, so a map built from a
symmetric positive-definite factor arrives already saying so.
"""
module PureQPBaseLinearMapsExt

using PureQPBase
using LinearMaps
using LinearAlgebra
using FillArrays: Fill

"""
    PureQPBase.ProductOperator{T}(map::LinearMap; symmetric, posdef)

Wrap a `LinearMap`, taking `symmetric` and `posdef` from the map's own traits unless the
caller states otherwise.

`issymmetric` and `isposdef` are properties a `LinearMap` carries rather than computes, so
reading them costs nothing and is what the map's author already declared.
"""
function PureQPBase.ProductOperator{T}(
        map::LinearMap;
        symmetric::Bool = issymmetric(map), posdef::Bool = isposdef(map),
        probe::Bool = false
    ) where {T <: Real}
    rows, cols = size(map)
    basis = zeros(T, probe ? cols : 0)
    column = zeros(T, rows)
    mapt = adjoint(map)
    return PureQPBase.ProductOperator{T, typeof(map), typeof(mapt), typeof(basis)}(
        map, mapt, rows, cols, symmetric, posdef, probe, basis, column
    )
end

"""
    setup(P::LinearMap, q, A, l, u, alg = OperatorSplitting(); kwargs...)

Solve with an operator cost, an operator constraint, or both.

Each `LinearMap` goes through [`as_operator`](@ref): the structure the base can hold is
unwrapped and the rest is wrapped in a [`PureQPBase.ProductOperator`](@ref). A matrix argument
is passed through untouched, so mixing the two is ordinary. The element type is taken from `q`,
which is the vector the solve is carried out in.

Equilibration cannot read an operator's entries, so `scaling = 0` is required unless the
wrapped map has a `PureQPBase.structural_rows` method; without it, `setup` throws and names
both remedies.
Each of the three methods below takes at least one `LinearMap`, which is what stops this call
from reaching itself: a pair of plain matrices leaves [`as_operator`](@ref) with nothing to
wrap, so a method accepting that pair would forward the same arguments back until the stack
ends. That pair belongs to whichever package defines the default algorithm.
"""
PureQPBase.setup(P::LinearMap, q::AbstractVector, A::AbstractMatrix, l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...) =
    wrapped_setup(P, q, A, l, u, alg...; kwargs...)
PureQPBase.setup(P::AbstractMatrix, q::AbstractVector, A::LinearMap, l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...) =
    wrapped_setup(P, q, A, l, u, alg...; kwargs...)
PureQPBase.setup(P::LinearMap, q::AbstractVector, A::LinearMap, l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...) =
    wrapped_setup(P, q, A, l, u, alg...; kwargs...)

"Wrap whichever arguments are maps and hand the pair on."
function wrapped_setup(P, q, A, l, u, alg...; kwargs...)
    T = float(eltype(q))
    return PureQPBase.setup(as_operator(T, P), q, as_operator(T, A), l, u, alg...; kwargs...)
end

"""
    solve(P::LinearMap, q, A, l, u, alg = OperatorSplitting(); kwargs...)

Set up and solve in one call, wrapping each `LinearMap` as [`setup`](@ref) does.

`solve` takes `AbstractMatrix` arguments, so a `LinearMap` reaches neither it nor the
`warm_start!` it forwards to without this. It is split over three signatures for the reason
[`setup`](@ref) is.
"""
PureQPBase.solve(P::LinearMap, q::AbstractVector, A::AbstractMatrix, l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...) =
    wrapped_solve(P, q, A, l, u, alg...; kwargs...)
PureQPBase.solve(P::AbstractMatrix, q::AbstractVector, A::LinearMap, l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...) =
    wrapped_solve(P, q, A, l, u, alg...; kwargs...)
PureQPBase.solve(P::LinearMap, q::AbstractVector, A::LinearMap, l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...) =
    wrapped_solve(P, q, A, l, u, alg...; kwargs...)

"Wrap whichever arguments are maps and hand the pair on."
function wrapped_solve(P, q, A, l, u, alg...; kwargs...)
    T = float(eltype(q))
    return PureQPBase.solve(as_operator(T, P), q, as_operator(T, A), l, u, alg...; kwargs...)
end

"""
    as_operator(T, M)

What `M` is to the solver. A `LinearMap` that `unwrap` recognizes becomes the base's own
representation of it; any other `LinearMap` becomes a [`PureQPBase.ProductOperator`](@ref), and
a matrix is already what it needs to be.
"""
function as_operator(::Type{T}, M::LinearMap) where {T}
    unwrapped = unwrap(T, M, one(T))
    return isnothing(unwrapped) ? PureQPBase.ProductOperator{T}(M) : unwrapped
end
as_operator(::Type{T}, M::AbstractMatrix) where {T} = M

"""
    unwrap(T, M, λ)

`λ * M` in the base's own representation, or `nothing` when `M` is not one the base can hold.
Only the factors are handed over; the product they stand for is never formed.

| `M` | becomes |
|---|---|
| `LinearMap(B)` for a matrix `B` | `B` itself; `λ * B`, a copy, when `λ ≠ 1` |
| `FillMap(c, (m, n))` | `Fill(λ * c, (m, n))` |
| `LinearMaps.UniformScalingMap(c, n)` | `Diagonal(Fill(λ * c, n))` |
| `kron(LinearMap(B₁), LinearMap(B₂))` | `KroneckerOperator(λ * B₁, B₂)` |
| `kron(M₁, …, M_k)` | `KroneckerOperator` of two sides, the one that is not already a strided factor formed |
| `blockdiag(LinearMap(B₁), …)` | `BlockDiagonal([λ * B₁, …])` |
| `vcat(M₁, …)` | `StackedOperator` of what each `λ * Mᵢ` becomes |
| `hcat(M₁, …)` | `JoinedOperator` of what each `λ * Mᵢ` becomes |
| `hvcat(rows, M₁, …)` | `StackedOperator` of one `JoinedOperator` per row of blocks |
| `M₁ + M₂ + …` | `SumOperator` of what each `λ * Mᵢ` becomes |
| `M₁ * M₂` | `ComposedOperator(λ * M₁, M₂)` |
| `c * M′` | what `M′` becomes, with `λ * c` in place of `λ` |

Every other map, such as a `FunctionMap`, a chain of more than two composed maps, or a Kronecker
product whose cheapest split would form more than `KRON_FORM_ENTRIES` entries, is `nothing`, and
stays a [`PureQPBase.ProductOperator`](@ref).

The concatenations, the sum and the product never decline: their parts go through
[`as_operator`](@ref), so a part the table does not cover becomes a `ProductOperator` inside the
composition while the others keep their own representation.
"""
unwrap(::Type{T}, ::LinearMap, λ) where {T} = nothing
unwrap(::Type{T}, M::LinearMaps.WrappedMap, λ) where {T} = matrix_factor(T, M, λ)
unwrap(::Type{T}, M::LinearMaps.ScaledMap, λ) where {T} =
    isreal(M.λ) ? unwrap(T, M.lmap, λ * T(real(M.λ))) : nothing

# A constant block is an `AbstractMatrix` with `O(1)` entries and `O(m + n)` products, so it is
# materializable and costs no storage; the stack or join it sits in stays readable.
unwrap(::Type{T}, M::LinearMaps.FillMap, λ) where {T} =
    isreal(M.λ) ? Fill(T(λ * real(M.λ)), size(M)) : nothing

# `λI` of size `M.M`, as a `Diagonal` of a `Fill`: `O(1)` storage and readable entries.
unwrap(::Type{T}, M::LinearMaps.UniformScalingMap, λ) where {T} =
    isreal(M.λ) ? Diagonal(Fill(T(λ * real(M.λ)), M.M)) : nothing

# The most entries `unwrap` forms to hand a Kronecker product over as two factors. Above it the
# product stays a `ProductOperator`, supplying products only: slower, and never an allocation
# the caller did not ask for.
const KRON_FORM_ENTRIES = 1 << 22

"The entries one side of a Kronecker split holds once formed."
side_entries(side) = prod(m -> size(m, 1), side) * prod(m -> size(m, 2), side)

"""
    formable(M) -> Bool

Whether every entry of `M` is known from its parts, so forming it reads no function.

A `FunctionMap` is the case this excludes: its author supplies products and nothing else, and a
dense copy of it would be built by applying the function `n` times to recover what was
deliberately never stored. Such a map stays a [`PureQPBase.ProductOperator`](@ref).
"""
formable(::LinearMap) = false
formable(M::LinearMaps.WrappedMap) = M.lmap isa AbstractMatrix && eltype(M.lmap) <: Real
formable(M::LinearMaps.ScaledMap) = isreal(M.λ) && formable(M.lmap)
formable(M::LinearMaps.FillMap) = isreal(M.λ)
formable(M::LinearMaps.UniformScalingMap) = isreal(M.λ)
formable(
    M::Union{
        LinearMaps.KroneckerMap, LinearMaps.BlockMap, LinearMaps.BlockDiagonalMap,
        LinearMaps.LinearCombination, LinearMaps.CompositeMap,
    }
) = all(formable, M.maps)

"""
    side_cost(T, side, λ)

The entries forming `side` costs: `0` where it is already a factor `KroneckerOperator` accepts,
and `typemax(Int)` where it cannot be formed at all.
"""
function side_cost(::Type{T}, side, λ) where {T}
    all(formable, side) || return typemax(Int)
    length(side) == 1 && matrix_factor(T, side[1], λ) isa StridedMatrix && return 0
    return side_entries(side)
end

"`λ` times one side of a Kronecker split, as the strided factor it already is or formed."
function kron_factor(::Type{T}, side, λ) where {T}
    if length(side) == 1
        B = matrix_factor(T, side[1], λ)
        B isa StridedMatrix && return B
    end
    B = Matrix{T}(length(side) == 1 ? side[1] : kron(side...))
    return isone(λ) ? B : rmul!(B, λ)
end

# `KroneckerOperator` holds exactly two factors, both strided and of one type: it builds its
# scratch with `similar(A2, T, m, n)` at rectangular sizes, which only a strided factor
# survives. A product of more than two maps, or one holding a factor that is not a strided
# matrix, is split in two, and whichever side cannot be handed over as a factor is formed. The
# cheapest of the `k - 1` splits is the one taken, so a `FillMap` or `UniformScalingMap` factor
# is formed rather than costing the whole product its structure.
function unwrap(::Type{T}, M::LinearMaps.KroneckerMap, λ) where {T}
    maps = M.maps
    k = length(maps)
    k < 2 && return nothing
    split, cost = 0, typemax(Int)
    for p in 1:(k - 1)
        left = side_cost(T, maps[1:p], λ)
        left == typemax(Int) && continue
        right = side_cost(T, maps[(p + 1):k], one(T))
        right == typemax(Int) && continue
        left + right < cost && ((split, cost) = (p, left + right))
    end
    (iszero(split) || cost > KRON_FORM_ENTRIES) && return nothing
    A1 = kron_factor(T, maps[1:split], λ)
    A2 = kron_factor(T, maps[(split + 1):k], one(T))
    return PureQPBase.KroneckerOperator(uniform(T, (A1, A2))...)
end

function unwrap(::Type{T}, M::LinearMaps.BlockDiagonalMap, λ) where {T}
    blocks = map(m -> matrix_factor(T, m, λ), M.maps)
    any(isnothing, blocks) && return nothing
    return PureQPBase.BlockDiagonal(uniform(T, blocks))
end

# A `BlockMap` lays its blocks out by `rows`, the number of blocks in each row of blocks: all ones
# is a `vcat`, a single count is an `hcat`, and anything else is an `hvcat`, which is a stack of
# joins. Every block takes `λ`, since scaling a concatenation scales each of its blocks.
# `as_operator` rather than `unwrap`: a block it does not recognize becomes a `ProductOperator`
# instead of declining the whole composition, so the blocks it does recognize keep their own
# reduction — one opaque block does not cost the others their structure.
function unwrap(::Type{T}, M::LinearMaps.BlockMap, λ) where {T}
    rows = M.rows
    blocks = map(m -> as_operator(T, isone(λ) ? m : λ * m), M.maps)
    all(isone, rows) && return PureQPBase.StackedOperator(blocks...)
    length(rows) == 1 && return PureQPBase.JoinedOperator(blocks...)
    groups = Vector{AbstractMatrix{T}}(undef, length(rows))
    off = 0
    for (k, r) in pairs(rows)
        groups[k] = PureQPBase.JoinedOperator(blocks[(off + 1):(off + r)]...)
        off += r
    end
    return PureQPBase.StackedOperator(groups...)
end

# `λ(B + C) = λB + λC`, so every term takes `λ`.
function unwrap(::Type{T}, M::LinearMaps.LinearCombination, λ) where {T}
    terms = map(m -> as_operator(T, isone(λ) ? m : λ * m), M.maps)
    return PureQPBase.SumOperator(terms...)
end

# `maps[1]` is the one applied first, so the last is the outer operator. Two maps only: a longer
# chain has no one inner part to reduce against, and `ComposedOperator` holds exactly two.
function unwrap(::Type{T}, M::LinearMaps.CompositeMap, λ) where {T}
    length(M.maps) == 2 || return nothing
    inner, outer = M.maps
    # `λ(BE) = (λB)E`, so the scalar goes on the outer part.
    return PureQPBase.ComposedOperator(
        as_operator(T, isone(λ) ? outer : λ * outer), as_operator(T, inner)
    )
end

"`λ * B` for the matrix a wrapped map holds, or `nothing` when the map holds anything else."
matrix_factor(::Type{T}, ::LinearMap, λ) where {T} = nothing
function matrix_factor(::Type{T}, M::LinearMaps.WrappedMap, λ) where {T}
    B = M.lmap
    (B isa AbstractMatrix && eltype(B) <: Real) || return nothing
    return isone(λ) ? conform(T, B) : conform(T, λ * B)
end
matrix_factor(::Type{T}, M::LinearMaps.ScaledMap, λ) where {T} =
    isreal(M.λ) ? matrix_factor(T, M.lmap, λ * T(real(M.λ))) : nothing

conform(::Type{T}, B::AbstractMatrix) where {T} = eltype(B) === T ? B : AbstractMatrix{T}(B)

"""
    uniform(T, mats) -> Vector

`mats` as a vector of one matrix type, which `KroneckerOperator` and `BlockDiagonal` require.
Matrices of mixed types are all converted to `Matrix{T}`.
"""
function uniform(::Type{T}, mats) where {T}
    M = typeof(first(mats))
    all(B -> typeof(B) === M, mats) && return collect(mats)
    return Matrix{T}[Matrix{T}(B) for B in mats]
end

end
