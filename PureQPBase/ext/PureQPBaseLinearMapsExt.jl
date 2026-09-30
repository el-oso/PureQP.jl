"""
Accepts a `LinearMaps.LinearMap` wherever [`PureQPBase.setup`](@ref) takes a matrix.

A `LinearMap` is not an `AbstractMatrix`, so it reaches the solver either as the base's own
representation of what it holds or through [`PureQPBase.ProductOperator`](@ref). A wrapped
matrix, a Kronecker product of two wrapped matrices, a block-diagonal of wrapped matrices and a
scalar multiple of any of these arrive as a matrix, a `KroneckerOperator` and a `BlockDiagonal`,
so the algorithms' structured paths see them. Every other map is wrapped. The protocol the
wrapper implements lives in `PureQPBase/src/operator.jl` and needs no dependency.

What loading LinearMaps buys over wrapping by hand is the two declarations the wrapper cannot
compute: LinearMaps tracks `issymmetric` and `isposdef` on its maps, so a map built from a
symmetric positive-definite factor arrives already saying so.
"""
module PureQPBaseLinearMapsExt

using PureQPBase
using LinearMaps
using LinearAlgebra

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
| `kron(LinearMap(B₁), LinearMap(B₂))` | `KroneckerOperator(λ * B₁, B₂)` |
| `blockdiag(LinearMap(B₁), …)` | `BlockDiagonal([λ * B₁, …])` |
| `c * M′` | what `M′` becomes, with `λ * c` in place of `λ` |

Every other map, such as a `FunctionMap`, a sum, a general product, a Kronecker product of more
than two maps or a factor that is not itself a wrapped matrix, is `nothing`, and stays a
[`PureQPBase.ProductOperator`](@ref).
"""
unwrap(::Type{T}, ::LinearMap, λ) where {T} = nothing
unwrap(::Type{T}, M::LinearMaps.WrappedMap, λ) where {T} = matrix_factor(T, M, λ)
unwrap(::Type{T}, M::LinearMaps.ScaledMap, λ) where {T} =
    isreal(M.λ) ? unwrap(T, M.lmap, λ * T(real(M.λ))) : nothing

function unwrap(::Type{T}, M::LinearMaps.KroneckerMap, λ) where {T}
    length(M.maps) == 2 || return nothing
    factors = (matrix_factor(T, M.maps[1], λ), matrix_factor(T, M.maps[2], one(T)))
    any(isnothing, factors) && return nothing
    A1, A2 = uniform(T, factors)
    return PureQPBase.KroneckerOperator(A1, A2)
end

function unwrap(::Type{T}, M::LinearMaps.BlockDiagonalMap, λ) where {T}
    blocks = map(m -> matrix_factor(T, m, λ), M.maps)
    any(isnothing, blocks) && return nothing
    return PureQPBase.BlockDiagonal(uniform(T, blocks))
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
