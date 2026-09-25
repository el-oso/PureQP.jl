"""
    ProductOperator{T,L,LT,V} <: AbstractMatrix{T}

An operator that supplies products but no entries, presented as an `AbstractMatrix` so the
solver's own seams reach it.

`LinearMaps.LinearMap`, `SciMLOperators.AbstractSciMLOperator` and
`AbstractOperators.AbstractOperator` are each their own type hierarchy, none of them below
`AbstractMatrix`, so none can be handed to [`setup`](@ref) directly. This wraps one. The
solver never needed the entries of such an operator — [`is_materializable`](@ref) is `false`
here, so the rungs that would form a matrix decline and `linsys = :auto` reaches the
matrix-free backend — but it does need three answers no product can give:

  - `size`, taken from the wrapped operator;
  - whether the operator is symmetric, which [`setup`](@ref) requires of `P`;
  - whether `P + σI` is positive definite, which [`is_convex`](@ref) answers.

The last two are declared by the author rather than computed, since proving either from products
alone costs more than the solve. [`setup`](@ref) does spot-check a symmetric declaration with two
products ([`check_symmetric_products`](@ref)). `symmetric` and `posdef` are ignored for a
`ProductOperator` standing in for `A`, which is neither.

Equilibration needs column and row ∞-norms, which products do not give directly. Three ways to
supply them, in the order they cost:

  - a [`structural_rows`](@ref) method on the wrapped type, which a *structured* operator can
    answer for free and which skips probing entirely;
  - `probe = true`, which recovers each column as `op * eⱼ` — `(2·sweeps + 1)·n` products,
    exact where the wrapped `mul!` selects stored entries;
  - `scaling = 0`, which skips equilibration and costs whatever the problem's scaling costs.

An operator offering none of the three is refused by name rather than escaping as a
`CanonicalIndexError` from inside a column walk.
"""
struct ProductOperator{T <: Real, L, LT, V <: AbstractVector{T}} <: AbstractMatrix{T}
    op::L
    # `Aᵀy` runs every iteration. For an operator whose `adjoint` builds a new object -- a
    # SciMLOperator's does -- rebuilding it per product allocates, so it is formed once here.
    opt::LT
    rows::Int
    cols::Int
    symmetric::Bool
    posdef::Bool
    probe::Bool
    basis::V       # scratch of length `cols`, holds one basis vector at a time
    column::V      # scratch of length `rows`, receives `op * eⱼ`
end

"""
    ProductOperator{T}(op; symmetric = false, posdef = false)

Wrap `op`, which must answer `size`, `mul!` against a vector, and `mul!` against a vector for
`adjoint(op)` — so `op` has to be adjoint-able. `LinearMap`, `AbstractSciMLOperator` and
`AbstractOperator` all are.

`symmetric` and `posdef` are the author's declaration about the operator. They are required of
a `P` and irrelevant to an `A`. `setup` checks a symmetric declaration with two products and
throws if it is wrong; `posdef` is not checked.

`probe` decides what happens at equilibration, which needs column and row maxima that products
do not give. With it, [`probe_column!`](@ref) recovers column `j` as `op * eⱼ` and the maxima
are read off that; without it, an operator that overrides neither equilibration seam is
refused and told to pass `scaling = 0`. Probing costs `(2·sweeps + 1)·n` products — 21n at the
default `scaling = 10` — so it is worth it when the product is cheap and the problem is badly
scaled, and not otherwise. It is exact for an operator whose `mul!` selects stored entries and
agrees to that operator's own rounding for one that recomputes them.
"""
function ProductOperator{T}(
        op; symmetric::Bool = false, posdef::Bool = false, probe::Bool = false
    ) where {T <: Real}
    rows, cols = size(op)
    basis = zeros(T, probe ? cols : 0)
    column = zeros(T, probe ? rows : 0)
    opt = adjoint(op)
    return ProductOperator{T, typeof(op), typeof(opt), typeof(basis)}(
        op, opt, rows, cols, symmetric, posdef, probe, basis, column
    )
end

Base.size(M::ProductOperator) = (M.rows, M.cols)

# The wrapped operator is the thing that knows how to multiply; both directions forward
# straight to it so no copy of the operand is made.
LinearAlgebra.mul!(y::AbstractVector, M::ProductOperator, x::AbstractVector) = mul!(y, M.op, x)

function LinearAlgebra.mul!(
        y::AbstractVector, Mt::Adjoint{<:Any, <:ProductOperator}, x::AbstractVector
    )
    return mul!(y, parent(Mt).opt, x)
end

# The message names no type: interpolating one goes through `show(::IO, ::Type)`, a runtime
# dispatch `--trim` cannot resolve, and this branch is live code for an operator that has no
# entries to give.
"""
    no_entries()

Throw the refusal an operator with no readable entries owes its caller, naming every way out.

One function rather than a message per site, so the three remedies stay listed together. The
message interpolates nothing: `show(::IO, ::Type)` is a runtime dispatch `--trim` cannot
resolve, and this branch is live code for an operator that declines.
"""
function no_entries()
    throw(
        ArgumentError(
            "this operator supplies products only and has no entries to read. Equilibration " *
                "needs column and row norms: build it with `probe = true` to recover each " *
                "column as a product, pass `scaling = 0` to skip equilibration, or give the " *
                "wrapped type a `PureQPBase.structural_rows` method."
        )
    )
end

Base.getindex(::ProductOperator, ::Integer, ::Integer) = no_entries()

# `getindex` refusing is right for equilibration and baffling as the answer to printing the
# object, which is what the `AbstractArray` display would call it for. Showing the shape and
# the wrapped operator says everything there is to say without reading an entry.
function Base.show(io::IO, ::MIME"text/plain", M::ProductOperator{T}) where {T}
    print(io, M.rows, "×", M.cols, " ProductOperator{", T, "} over ", typeof(M.op))
    M.probe && print(io, ", probing")
    M.symmetric && print(io, ", symmetric")
    M.posdef && print(io, ", posdef")
    return nothing
end

Base.show(io::IO, M::ProductOperator) = show(io, MIME"text/plain"(), M)

is_materializable(::ProductOperator) = false
is_symmetric(M::ProductOperator) = M.symmetric
is_convex(::Type{T}, P::ProductOperator, sigma) where {T} = P.posdef

"""
    check_symmetric_products(P, proto)

Throw unless `dot(v, P w) ≈ dot(P v, w)` for two fixed vectors `v` and `w`.

An operator's symmetry is declared, not read from entries, so nothing else checks it. For a symmetric `P` the two products
agree to rounding; for one that is not, they differ by the size of the products themselves. The
vectors are fixed rather than random so that `setup` stays reproducible. They are built on the
host and copied into buffers `similar` to `proto`, so a device operator is checked on the device.
"""
function check_symmetric_products(P, proto::AbstractVector)
    T = float(eltype(proto))
    n = size(P, 2)
    v = copyto!(similar(proto, T, n), T[sin(T(0.7) * i + T(0.3)) for i in 1:n])
    w = copyto!(similar(proto, T, n), T[cos(T(1.3) * i + T(0.1)) for i in 1:n])
    Pv, Pw = similar(v), similar(w)
    mul!(Pv, P, v)
    mul!(Pw, P, w)
    a, b = dot(v, Pw), dot(Pv, w)
    abs(a - b) <= sqrt(eps(T)) * max(abs(a), abs(b), eps(T)) || throw(
        ArgumentError(
            lazy"P is declared symmetric but is not: dot(v, P*w) = $a and dot(P*v, w) = $b for the same v and w. Correct the declaration on the operator."
        )
    )
    return nothing
end

"""
    probe_column!(M::ProductOperator, j) -> AbstractVector

Column `j` of `M`, recovered as `M * eⱼ` and returned in `M`'s own scratch, which the next
call overwrites.

Exact for an operator whose `mul!` selects stored entries, since the product then copies the
column. An operator applying a factored or composed form recomputes each entry instead, so its
column agrees to that operator's rounding rather than bitwise — which is the case probing
exists for, the entries not being there to walk.
"""
function probe_column!(M::ProductOperator{T}, j::Integer) where {T}
    # The scratch is empty on an operator built without `probe`, which would otherwise fail
    # here as a `BoundsError` saying nothing about why.
    M.probe || no_entries()
    fill!(M.basis, zero(T))
    M.basis[j] = one(T)
    mul!(M.column, M.op, M.basis)
    return M.column
end

# Equilibration's per-column seam, answered by one product each. A structured operator
# overrides `structural_rows` instead and never reaches these.
function weighted_colmax(::Type{T}, M::ProductOperator, j::Integer, w::AbstractVector) where {T}
    M.probe || no_entries()
    col = probe_column!(M, j)
    r = zero(T)
    for i in paired(col, w)
        r = max(r, w[i] * abs(T(col[i])))
    end
    return r
end

function weighted_colmax_rowmax!(
        ::Type{T}, e::AbstractVector, M::ProductOperator, j::Integer, w::AbstractVector, s
    ) where {T}
    M.probe || no_entries()
    col = probe_column!(M, j)
    r = zero(T)
    for i in paired(col, w, e)
        v = abs(T(col[i]))
        r = max(r, w[i] * v)
        e[i] = max(e[i], s * v)
    end
    return r
end

"""
    unpreconditioned!(dest)

Fill `dest` with the identity preconditioner, which is what [`reduced_diagonal!`](@ref)
returns for an operator whose diagonal is unavailable.

Jacobi preconditioning needs `P[j,j]` and the `ρ`-weighted column norms of `A`; neither is a
product, so an operator that supplies only products has nothing to build it from. Conjugate
gradients converges without a preconditioner, so this costs iterations rather than the
answer. An operator that can produce the reduced diagonal gives its wrapped type a
[`reduced_diagonal!`](@ref) method, which is more specific than the three below.
"""
unpreconditioned!(dest) = fill!(dest, one(eltype(dest)))

reduced_diagonal!(dest, ::Type{T}, P::ProductOperator, A, rho, E, D, sigma, c) where {T} =
    unpreconditioned!(dest)
reduced_diagonal!(dest, ::Type{T}, P, A::ProductOperator, rho, E, D, sigma, c) where {T} =
    unpreconditioned!(dest)
reduced_diagonal!(
    dest, ::Type{T}, P::ProductOperator, A::ProductOperator, rho, E, D, sigma, c
) where {T} = unpreconditioned!(dest)
