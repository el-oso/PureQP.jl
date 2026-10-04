# `B + C` and `B * E` held as their parts, so the reduced matrix is built from the parts' own
# reductions rather than from `2n` products of the composition. `vcat` is `stacked.jl`; scaling by a
# constant needs no type, since a scalar folds into the factor it multiplies.

"""
    ComposedOperator{T, O, I, M, V} <: AbstractMatrix{T}

    A = B E      `B` is `m×k`, `E` is `k×n`, and the product is never formed.

A composition of two operators, which is what `B * E` means: the inner one maps the variables into
an intermediate space and the outer one out of it.

Holding the parts is what lets the reduction skip the outer operator entirely:

    Aᵀ diag(w) A = Eᵀ (Bᵀ diag(w) B) E

so `G = Bᵀ diag(w) B` is formed once, by whatever reduction `B`'s own representation has, and the
`n` columns then cost one product through `E`, one `k×k` product with `G`, and one through `Eᵀ`.
Reaching the same matrix through products of `A` would apply `B` and `Bᵀ` once per column — `2n`
times rather than once — so the saving is the whole of the outer operator's share, and it grows
with how much larger `m` is than `k`.

`work` and `gwork` are the intermediates the products and the reduction need, so neither allocates.
They make an operator single-use at a time: one `ComposedOperator` must not be multiplied from two
tasks at once.
"""
struct ComposedOperator{T <: Real, O <: AbstractMatrix{T}, I <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    outer::O        # `B`, `m×k`
    inner::I        # `E`, `k×n`
    work::V         # `k`, the intermediate of a product
    gwork::V        # `k`, the intermediate of the reduction

    function ComposedOperator{T, O, I}(outer::O, inner::I) where {T, O, I}
        size(outer, 2) == size(inner, 1) || throw(
            DimensionMismatch(
                lazy"a ComposedOperator needs its parts to meet: the outer takes $(size(outer, 2)) columns, the inner gives $(size(inner, 1)) rows"
            )
        )
        k = size(inner, 1)
        work = similar(inner, T, k)
        return new{T, O, I, typeof(work)}(outer, inner, work, similar(work))
    end
end

"""
    ComposedOperator(outer, inner)

The operator `outer * inner`, which is `size(outer, 1)` by `size(inner, 2)`.
"""
ComposedOperator(outer::AbstractMatrix{T}, inner::AbstractMatrix{T}) where {T <: Real} =
    ComposedOperator{T, typeof(outer), typeof(inner)}(outer, inner)

"The parts, as `(outer, inner)`."
parts(A::ComposedOperator) = (A.outer, A.inner)

Base.size(A::ComposedOperator) = (size(A.outer, 1), size(A.inner, 2))

# The product of two stored parts stands for `mn` entries that are each a length-`k` sum, so
# forming the matrix both holds more and discards which part each factor came from.
holds_structure(::ComposedOperator) = true

is_materializable(A::ComposedOperator) =
    is_materializable(A.outer) && is_materializable(A.inner)

# `(BE)[i,j] = Σₗ B[i,l] E[l,j]`, which both parts must be able to answer.
function Base.getindex(A::ComposedOperator{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    acc = zero(T)
    for l in axes(A.outer, 2)
        acc += T(A.outer[i, l]) * T(A.inner[l, j])
    end
    return acc
end

"Row `i` of `B E` is row `i` of `B` carried through `E`: `Eᵀ (Bᵀ eᵢ)`."
function dense_row!(dest::AbstractVector, A::ComposedOperator, i::Integer)
    check_row_dest(dest, A)
    @boundscheck checkbounds(A, i, :)
    dense_row!(A.work, A.outer, i)
    mul!(dest, adjoint(A.inner), A.work)
    return dest
end

function LinearAlgebra.mul!(y::AbstractVector, A::ComposedOperator, x::AbstractVector)
    mul!(A.work, A.inner, x)
    mul!(y, A.outer, A.work)
    return y
end

# `(BE)ᵀ = Eᵀ Bᵀ`, so the parts are applied in the other order.
function LinearAlgebra.mul!(
        y::AbstractVector, At::Union{Adjoint{<:Any, <:ComposedOperator}, Transpose{<:Any, <:ComposedOperator}},
        x::AbstractVector
    )
    A = parent(At)
    mul!(A.work, adjoint(A.outer), x)
    mul!(y, adjoint(A.inner), A.work)
    return y
end

"""
    add_reduced_term!(R, T, A::ComposedOperator, weights, D, n, m, scratch, ej, av, col) -> R

Add `D Aᵀ diag(weights) A D` into `R` as `D Eᵀ G E D`, with `G = Bᵀ diag(weights) B`.

`G` is `k×k` and is built by the outer operator's own reduction, so a structured outer operator
contributes its structure once rather than through `2n` products. Each column of the result then
costs one product through `E`, one with `G`, and one through `Eᵀ`.
"""
function add_reduced_term!(
        R::AbstractMatrix{T}, ::Type{T}, A::ComposedOperator, weights::AbstractVector,
        D::AbstractVector, n::Integer, m::Integer, scratch, ej, av, col
    ) where {T}
    G = composed_gram!(A, T, weights, m, scratch, av)
    return add_composed_term!(R, T, A.inner, G, D, n, scratch, ej, A.gwork)
end

"""
    composed_gram!(A, T, weights, m, scratch, av) -> G

`G = Bᵀ diag(weights) B` for the outer part, through whatever reduction that part's own
representation has.

The scaling is left out of `G`: it belongs to `A`'s columns, which are the inner part's, and
`add_composed_term!` applies it there.
"""
function composed_gram!(A::ComposedOperator, ::Type{T}, weights, m, scratch, av) where {T}
    G, ones_k, outer_scratch, ek, _, _ = scratch
    k = size(A.inner, 1)
    fill!(G, zero(T))
    add_reduced_term!(G, T, A.outer, weights, ones_k, k, m, outer_scratch, ek, av, A.gwork)
    return G
end

# `D Eᵀ G E D = (ED)ᵀ G (ED)`, so a stored inner part scales its columns once and the rest is two
# matrix products accumulated straight into `R`. Going column by column instead would do `n`
# matrix-vector products in place of these, which is the same arithmetic at a worse rate.
function add_composed_term!(
        R::AbstractMatrix{T}, ::Type{T}, inner::StridedMatrix{T}, G, D, n, scratch, ej, gwork
    ) where {T}
    _, _, _, _, Es, M = scratch
    for j in 1:n
        dj = D[j]
        for i in axes(inner, 1)
            Es[i, j] = inner[i, j] * dj
        end
    end
    mul!(M, G, Es)
    mul!(R, adjoint(Es), M, true, true)
    return R
end

# An inner part that is itself an operator has no matrix to multiply, so its columns are recovered
# one product at a time.
function add_composed_term!(
        R::AbstractMatrix{T}, ::Type{T}, inner, G, D, n, scratch, ej, gwork
    ) where {T}
    _, _, _, ek, _, M = scratch
    for j in 1:n
        fill!(ej, zero(T))
        ej[j] = one(T)
        mul!(ek, inner, ej)              # `E eⱼ`
        mul!(gwork, G, ek)               # `G E eⱼ`
        mul!(view(M, :, j), adjoint(inner), gwork)
    end
    for j in 1:n
        dj = D[j]
        for i in 1:n
            R[i, j] += D[i] * M[i, j] * dj
        end
    end
    return R
end

function reduced_term_scratch(::Type{T}, A::ComposedOperator) where {T}
    k, n = size(A.inner)
    G = similar(A.inner, T, k, k)
    ones_k = fill!(similar(A.work, T, k), one(T))
    ek = similar(A.work, T, k)
    Es = similar(A.inner, T, k, n)
    # `M` holds `G (ED)`, `k×n`, for a stored inner part; for an operator inner part it holds the
    # `n×n` result a column at a time, so it is sized for whichever is larger.
    M = similar(A.inner, T, A.inner isa StridedMatrix ? (k, n) : (n, n))
    return (G, ones_k, reduced_term_scratch(T, A.outer), ek, Es, M)
end

"""
    reduced_diagonal!(dest, T, P, A::ComposedOperator, rho, E, D, sigma, c) -> dest

The reduced diagonal, read from entries when both parts have them.

A column of `B E` is a combination of all `k` columns of `B`, so the diagonal of `Aᵀ diag(ρ) A`
has no shorter form than the entries give; what the composition saves is the reduced matrix, not
this. A part that supplies only products leaves the diagonal unavailable, and conjugate gradients
then run unpreconditioned.
"""
reduced_diagonal!(dest, ::Type{T}, P, A::ComposedOperator, rho, E, D, sigma, c) where {T} =
    is_materializable(A) ? indexed_reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c) :
    unpreconditioned!(dest)

"""
    SumOperator{T, B, V} <: AbstractMatrix{T}

    A = B + C    the terms share both dimensions, and the sum is never formed.

A sum of operators over the same variables and the same rows, which is what `B + C` means: one
problem's constraint matrix written as a correction to another, or a structured part plus a
coupling.

Unlike the other compositions, a sum has **no reduction of its own**, and the reason is worth
stating so it is not added again. Expanding gives

    Aᵀ W A = Σᵢ BᵢᵀW Bᵢ + Σᵢ≠ⱼ BᵢᵀW Bⱼ

whose `K` diagonal terms cost `2n` products of a term each and whose `K(K-1)/2` cross pairs belong
to no term's representation, so they cost `2n` more each: `nK(K+1)` products of a term in all.
Reaching the same matrix through products of the sum costs `2nK`, because `A eⱼ = Σₜ Bₜ eⱼ` already
sums the terms in a single pass. The expansion is therefore worse for every `K ≥ 2`, and measurably
so — 1.04x at two terms, 0.69x at three and 0.49x at four, against the products it replaced. The
generic reduction is the right one here, and this type exists for the other things it gives: the
sum is never formed, each term keeps its own representation, and products split across them.

`work` and `work_n` are the intermediates a product needs, so `mul!` allocates nothing. They make
an operator single-use at a time.
"""
struct SumOperator{T <: Real, B <: Tuple, V <: AbstractVector{T}} <: AbstractMatrix{T}
    terms::B
    work::V       # `m`, holds one term's product while it is added to the sum
    work_n::V     # `n`, the same for a transposed product and for a row

    function SumOperator{T, B}(terms::B) where {T, B <: Tuple}
        isempty(terms) && throw(ArgumentError("a SumOperator needs at least one term"))
        sz = size(first(terms))
        for (i, Ai) in pairs(terms)
            size(Ai) == sz || throw(
                DimensionMismatch(
                    lazy"every term of a SumOperator has the same size: term $i is $(size(Ai)), the first is $sz"
                )
            )
        end
        work = similar(first(terms), T, sz[1])
        return new{T, B, typeof(work)}(terms, work, similar(work, T, sz[2]))
    end
end

"""
    SumOperator(terms...)

The sum of `terms`, which all share both dimensions.
"""
function SumOperator(terms::AbstractMatrix...)
    T = promote_type(map(eltype, terms)...)
    T <: Real || throw(ArgumentError(lazy"a SumOperator needs a real element type, got $T"))
    return SumOperator{T, typeof(terms)}(terms)
end

"The number of terms in the sum."
nterms(A::SumOperator) = length(A.terms)

Base.size(A::SumOperator) = size(first(A.terms))

# The terms are stored, so forming the sum holds another `mn` entries and loses which term each
# contribution came from, which is what the reduction below needs.
holds_structure(::SumOperator) = true

is_materializable(A::SumOperator) = all(is_materializable, A.terms)

Base.getindex(A::SumOperator{T}, i::Integer, j::Integer) where {T} =
    (@boundscheck checkbounds(A, i, j); sum_entry(A.terms, T, i, j))

# Recursion over the tuple rather than an index into it, so each arm reads one concrete term.
@inline sum_entry(::Tuple{}, ::Type{T}, i, j) where {T} = zero(T)
@inline sum_entry(terms::Tuple, ::Type{T}, i, j) where {T} =
    T(first(terms)[i, j]) + sum_entry(Base.tail(terms), T, i, j)

function LinearAlgebra.mul!(y::AbstractVector, A::SumOperator, x::AbstractVector)
    mul!(y, first(A.terms), x)
    return sum_mul!(y, Base.tail(A.terms), x, A.work, identity)
end

function LinearAlgebra.mul!(
        y::AbstractVector, At::Union{Adjoint{<:Any, <:SumOperator}, Transpose{<:Any, <:SumOperator}},
        x::AbstractVector
    )
    A = parent(At)
    mul!(y, adjoint(first(A.terms)), x)
    return sum_mul!(y, Base.tail(A.terms), x, A.work_n, adjoint)
end

@inline sum_mul!(y, ::Tuple{}, x, work, orient) = y

@inline function sum_mul!(y, terms::Tuple, x, work, orient)
    # Three-argument `mul!` into scratch, then added: the five-argument form that would accumulate
    # in place is not defined for every operator a term may be, and its generic fallback reads
    # entries, which an operator supplying only products refuses.
    mul!(work, orient(first(terms)), x)
    for j in eachindex(y)
        y[j] += work[j]
    end
    return sum_mul!(y, Base.tail(terms), x, work, orient)
end

"Row `i` of a sum is the sum of the terms' rows."
function dense_row!(dest::AbstractVector, A::SumOperator, i::Integer)
    check_row_dest(dest, A)
    @boundscheck checkbounds(A, i, :)
    dense_row!(dest, first(A.terms), i)
    return sum_row!(dest, Base.tail(A.terms), i, A.work_n)
end

@inline sum_row!(dest, ::Tuple{}, i, work) = dest

@inline function sum_row!(dest, terms::Tuple, i, work)
    dense_row!(work, first(terms), i)
    for j in eachindex(dest)
        dest[j] += work[j]
    end
    return sum_row!(dest, Base.tail(terms), i, work)
end

# No `add_reduced_term!` method: the generic one reaches the reduced matrix through `2n` products
# of the sum, which is cheaper than expanding into the terms. The type's docstring has the count.

"""
    reduced_diagonal!(dest, T, P, A::SumOperator, rho, E, D, sigma, c) -> dest

The reduced diagonal, read from entries when every term has them.

An entry of the sum is the sum of the terms' entries, so the diagonal follows from them directly.
A term that supplies only products leaves it unavailable, and conjugate gradients then run
unpreconditioned.
"""
reduced_diagonal!(dest, ::Type{T}, P, A::SumOperator, rho, E, D, sigma, c) where {T} =
    is_materializable(A) ? indexed_reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c) :
    unpreconditioned!(dest)
