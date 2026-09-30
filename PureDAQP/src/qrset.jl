# The working set, factored as a QR of its rows rather than an LDLᵀ of their Gram matrix.
#
# Both represent the same set and answer the same three questions -- is this row dependent on
# the ones already held, what are the multipliers, what direction does a dependency admit --
# but not to the same accuracy. `Mₐ Mₐᵀ` has the condition number of `Mₐ` squared, and the
# pivot that decides dependence is formed by cancellation: for unit rows `d = 1 - (1 - σ²)`,
# whose absolute error is about `k·eps`, so a pivot cannot distinguish `σ` below roughly
# `sqrt(k·eps)` from zero however the tolerance is set. On a reduction where `σ` reaches 1e-7
# that decision is noise, and the method reports a problem infeasible that is not.
#
# The QR decides it on `|R_kk|`, the norm of the part of the entering row orthogonal to the
# ones held, which carries the condition number of `Mₐ` once rather than twice.
#
# The multipliers do not improve: `μ = R⁻¹R⁻ᵀt` solves the same system as before and inherits
# the same cond² sensitivity, which is a property of the map and not of the factorization.
# What changes is that a working set is no longer declared dependent by mistake.

"""
    WorkingSetQR{T}

The `QR` of `Mₐᵀ`, the active rows of the reduced constraint matrix as columns.

`ModifiableFactorizations` maintains it under the two edits the method makes: a row entering
appends a column, a row leaving deletes one, each `O(nk)`.

The factored matrix is `n × k` and the factorization requires more rows than columns, so the
working set holds at most `n` rows. That is the rank of the problem's own rows, and so the
most an active set can carry at a solution -- but not the most the method reaches on its way
there: the singular step wants the dependent row *in* the set, and from a full set that would
be row `n + 1`. A `QR` cannot hold it, since there is no `R[n+1, n+1]` for
[`first_dependent`](@ref) to read. Such a row is dependent by counting rather than by
measurement, so it needs no place in the factorization: `ldiv!` gives its coefficients against
the rows already held, and those are the null direction the step walks.
"""
struct WorkingSetQR{T <: Real}
    # Spelled out rather than left as a parameter: the reduction is dense by construction
    # (`M = A R⁻¹` is dense whatever `A` was), so this is the only factorization the method
    # ever holds, and naming it keeps the type of every workspace that carries one concrete.
    qr::ModifiableQR{T, Matrix{T}, ModifiableFactorizations.DenseQ{T, Matrix{T}}}
end

"An empty working set over `n` variables, sized for the largest it can reach."
function WorkingSetQR{T}(n::Integer, kmax::Integer) where {T <: Real}
    cap = min(Int(kmax), Int(n))
    # Built from one column and then emptied: the constructor takes a matrix, and a working
    # set starts with no rows in it. The capacity is what the insertions below grow into, so
    # none of them allocates.
    seed = zeros(T, Int(n), 1)
    seed[1] = one(T)
    f = ModifiableQR(seed; capacity = (Int(n), cap))
    delete_column!(f, 1)
    return WorkingSetQR{T}(f)
end

"The number of rows the working set holds."
@inline nactive(W::WorkingSetQR) = W.qr.n

"""
The most rows this representation can hold.

The factored matrix is `n × k` and the factorization requires more rows than columns, so this
is `n`. A working set that reaches it has one row per variable and spans the whole space;
[`full_set_step!`](@ref) is what the method does with a row that wants in after that.
"""
@inline maxrows(W::WorkingSetQR) = W.qr.m

"""
    conditioning(W) -> T

An estimate of `cond(Mₐ)`, or zero when the working set gives none.

The ratio of the largest and smallest diagonal entries of `R` bounds it, and those entries
are the norms of each row orthogonal to the ones before it.
"""
function conditioning(W::WorkingSetQR{T}) where {T}
    R = W.qr.R
    rmin = typemax(T)
    rmax = zero(T)
    @inbounds for i in 1:W.qr.n
        r = abs(R[i, i])
        r > zero(T) || continue
        rmin = min(rmin, r)
        rmax = max(rmax, r)
    end
    return (rmax > zero(T) && rmin < typemax(T)) ? rmax / rmin : zero(T)
end

"""
    add_row!(W, m_r) -> Bool

Append `m_r` to the working set, reporting whether it went in.

`rtol = 0` admits a row whatever its residual, because the caller decides what counts as
dependent: the method needs the dependent row *in* the set, since the direction it then walks
is how it either drops a blocking row or proves the problem infeasible. A row that duplicates
one already held goes in at `R_ii = 0`, which is the value [`first_dependent`](@ref) reads.

So `false` means the row is not finite, the one case `rtol = 0` still refuses. It is not a
state the method can continue from: the row would be recorded as active while the
factorization did not take it, leaving the two disagreeing about how many rows are held, and a
row so recorded is skipped by pricing and can never be reconsidered.
"""
@inline add_row!(W::WorkingSetQR, m_r::AbstractVector) =
    try_insert_column!(W.qr, W.qr.n + 1, m_r; rtol = 0)

"Drop the `i`th row of the working set."
@inline remove_row!(W::WorkingSetQR, i::Integer) = (delete_column!(W.qr, Int(i)); W)

"""
    first_dependent(W, rtol) -> Int

The first row whose part orthogonal to the rows before it is at or below `rtol`, or 0.

The rows of `M` are normalized, so `|R_ii|` lies in `[0, 1]` and the test is already
relative. This is the decision the Gram factorization cannot make: it holds `|R_ii|²`, and
below about `1e-8` that square is indistinguishable from zero in double precision.
"""
function first_dependent(W::WorkingSetQR{T}, rtol::T) where {T}
    R = W.qr.R
    @inbounds for i in 1:W.qr.n
        abs(R[i, i]) <= rtol && return i
    end
    return 0
end

"""
    solve_gram!(W, rhs) -> rhs

Overwrite `rhs` with the solution of `Mₐ Mₐᵀ x = rhs`, through `RᵀR = Mₐ Mₐᵀ`.
"""
function solve_gram!(W::WorkingSetQR{T}, rhs::AbstractVector{T}) where {T}
    k = W.qr.n
    R = W.qr.R
    b = view(rhs, 1:k)
    ldiv!(adjoint(R), b)
    ldiv!(R, b)
    return rhs
end

"""
    null_direction!(p, W, i) -> p

The combination of the first `i` rows that the `i`th is dependent on, with `+1` at `i`.

`Mₐᵀp` is then the part of row `i` the rows before it cannot reach, which is `R[i,i]` in
norm: zero exactly when the dependency is exact, and otherwise as small as the factorization
can tell.
"""
function null_direction!(p::AbstractVector{T}, W::WorkingSetQR{T}, i::Integer) where {T}
    k = W.qr.n
    fill!(view(p, 1:k), zero(T))
    if i > 1
        R = W.qr.R
        head = view(p, 1:(i - 1))
        @inbounds for j in 1:(i - 1)
            head[j] = -R[j, i]
        end
        ldiv!(UpperTriangular(view(parent(R), 1:(i - 1), 1:(i - 1))), head)
    end
    p[i] = one(T)
    return p
end

"""
    active_product!(u, W, mu, scratch) -> u

Overwrite `u` with `Mₐᵀ μ`, the point the working set's multipliers place.

`Mₐᵀ = QR`, so this is `Q(Rμ)`: the triangular product and one matrix-vector product against
an orthonormal `Q`, which is where the reduced iterate stops inheriting the squared condition
number the Gram form gave it. `Rμ` lands in `scratch` rather than in `μ`, which the caller
still holds.
"""
function active_product!(
        u::AbstractVector{T}, W::WorkingSetQR{T}, mu::AbstractVector{T},
        scratch::AbstractVector{T}
    ) where {T}
    k = W.qr.n
    w = view(scratch, 1:k)
    copyto!(w, view(mu, 1:k))
    lmul!(W.qr.R, w)
    mul!(u, W.qr.Q, w)
    return u
end
