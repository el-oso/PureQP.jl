"""
In-place `LDLᵀ` of the Gram matrix of an active row set, maintained under row
insertion and deletion.

`L` is unit lower triangular and `D` holds its diagonal. Both live in full-size
buffers with `k` marking how much is live, so a solve that adds and drops
thousands of constraints allocates only at setup.

The factored matrix is `Mₐ Mₐᵀ`, where `Mₐ` holds the rows of `M` currently
active, in the order they were added.
"""
# `const` on every field but `k`: only the live size is rebound, the buffers are written
# through. A fully immutable struct would be the wrong trade -- inlined into the workspace it
# loses the nonnull and alignment facts the vectorizer needs, and these loops go scalar.
mutable struct GramLDL{T <: Real}
    const L::Matrix{T}
    const D::Vector{T}
    k::Int
    const w::Vector{T}   # scratch for the rank-one sweep
    const b::Vector{T}   # scratch for the forward solve
end

function GramLDL{T}(kmax::Integer) where {T <: Real}
    return GramLDL{T}(zeros(T, kmax, kmax), zeros(T, kmax), 0, zeros(T, kmax), zeros(T, kmax))
end

Base.size(F::GramLDL) = F.k

"""
Working sets up to this size solve through plain loops rather than LAPACK.

`ldiv!` on a triangular view routes to `trtrs!`, whose call overhead dominates at this scale:
58 ns against 14 ns for a loop at `k = 8`. These solves run twice per iteration, so on a
small problem they are a third of it. LAPACK wins again once the triangle is big enough for
its blocking to pay, which is why this is a threshold and not a replacement.
"""
const SMALL_TRIANGLE = 32

"Solve `L y = b` in place over the leading `length(b)` block."
function forward_L!(F::GramLDL{T}, b::AbstractVector{T}) where {T}
    isempty(b) && return b
    k = length(b)
    if k > SMALL_TRIANGLE
        ldiv!(UnitLowerTriangular(view(F.L, 1:k, 1:k)), b)
        return b
    end
    # Column-oriented: each column of `L` is contiguous, and the update walks the tail of
    # `b`. `k <= F.k <= size(L, 1)` by construction, so the indices are in range.
    L = F.L
    @inbounds for j in 1:k
        bj = b[j]
        @simd for i in (j + 1):k
            b[i] -= L[i, j] * bj
        end
    end
    return b
end

"Solve `Lᵀ y = b` in place over the leading `length(b)` block."
function backward_L!(F::GramLDL{T}, b::AbstractVector{T}) where {T}
    isempty(b) && return b
    k = length(b)
    if k > SMALL_TRIANGLE
        ldiv!(UnitLowerTriangular(view(F.L, 1:k, 1:k))', b)
        return b
    end
    # `Lᵀ` is upper triangular, so this runs backwards, reading down column `i` of `L`. No
    # `@simd`: this is a reduction, and reassociating it moves the answer at the last digit,
    # which is where agreement with the reference solver is measured.
    L = F.L
    @inbounds for i in k:-1:1
        acc = b[i]
        for j in (i + 1):k
            acc -= L[j, i] * b[j]
        end
        b[i] = acc
    end
    return b
end

"""
    solve_gram!(F, rhs) -> rhs

Solve `Mₐ Mₐᵀ x = rhs` in place through the stored factors.
"""
function solve_gram!(F::GramLDL, rhs::AbstractVector)
    forward_L!(F, rhs)
    @inbounds @simd for i in eachindex(rhs)
        rhs[i] /= F.D[i]
    end
    backward_L!(F, rhs)
    return rhs
end

"""
    add_row!(F, g, beta; zero_tol) -> F

Extend the factorization by one active row `a`, given `g = Mₐ a` against the
rows already active and `beta = aᵀa`.

A new diagonal at or below `zero_tol` means the row is linearly dependent on
those already active. It is stored as an exact zero: that is what makes the
dependency detectable, since the caller scans `D` for a zero and takes the
singular branch instead of dividing by it.
"""
# Capacity is the caller's invariant, so exceeding it is a bug rather than a condition to handle.
@noinline function throw_gram_full(k::Int, kmax::Int)
    throw(
        ArgumentError(
            # One `lazy` string and no concatenation: `*` on strings calls `string`, which
            # builds the message eagerly and is not trim-safe.
            lazy"the Gram working set holds $kmax rows and a $(k + 1)th was added: the set carries one row more than there are variables and no path may exceed that"
        )
    )
end

function add_row!(F::GramLDL{T}, g::AbstractVector{T}, beta::T, zero_tol::T = sqrt(eps(T))) where {T}
    k = F.k
    # The writes below are `@inbounds`, so a `k` past the buffer would write foreign memory
    # rather than raise. The branch is never taken, and `@noinline` keeps its throw off the
    # hot path's allocation proof.
    k + 1 <= size(F.L, 1) || throw_gram_full(k, size(F.L, 1))
    if iszero(k)
        F.L[1, 1] = one(T)
        F.D[1] = beta
        F.k = 1
        return F
    end
    b = view(F.b, 1:k)
    # An explicit loop rather than `copyto!`: that checks lengths and throws, and building
    # the exception is an allocation site the hot-path guarantee sees whether or not the
    # branch can be reached. The caller always passes `k` entries.
    @inbounds @simd for i in 1:k
        b[i] = g[i]
    end
    forward_L!(F, b)

    d = beta
    # The `D[i] > zero_tol` test cannot leave this loop: a zero pivot must not be divided by,
    # and which pivots are zero is a property of the data. It predicts well, being false only
    # on the dependent rows, which are rare.
    @inbounds for i in 1:k
        lki = F.D[i] > zero_tol ? b[i] / F.D[i] : zero(T)
        F.L[k + 1, i] = lki
        d -= F.D[i] * lki^2
    end
    F.L[k + 1, k + 1] = one(T)
    F.D[k + 1] = d > zero_tol ? d : zero(T)
    F.k = k + 1
    return F
end

"""
    rank_one!(F, off, l, delta, n)

`L D Lᵀ ← L D Lᵀ + delta·l lᵀ` on the `n×n` block whose first index is `off+1`.
Algorithm C1 of Gill, Golub, Murray and Saunders, *Methods for modifying matrix
factorizations*, Math. Comp. 28(126):505-535, 1974. `l` is overwritten.
"""
function rank_one!(F::GramLDL{T}, off::Int, l::AbstractVector{T}, delta::T, n::Int) where {T}
    a = delta
    # `off + n <= F.k <= size(L, 1)`, so every index is in range. The inner loop is not
    # `@simd`: `l[r]` is written then read in the same iteration, a dependence that
    # reassociating would break.
    @inbounds for j in 1:n
        p = l[j]
        dold = F.D[off + j]
        dnew = dold + a * p^2
        F.D[off + j] = dnew > 0 ? dnew : zero(T)
        # A pivot reaching zero ends the sweep's contribution rather than dividing by it.
        if !(dnew > 0)
            a = zero(T)
            continue
        end
        b = p * a / dnew
        a = dold * a / dnew
        for r in (j + 1):n
            lrj = F.L[off + r, off + j]
            l[r] -= p * lrj
            F.L[off + r, off + j] = lrj + b * l[r]
        end
    end
    return F
end

"""
    remove_row!(F, i) -> F

Drop the `i`-th active row. The trailing block is repaired with a rank-one
update first, then everything below and right of `i` shifts up and left by one.
The repair is what a deletion costs.
"""
function remove_row!(F::GramLDL{T}, i::Integer) where {T}
    k = F.k
    ntail = k - i
    if ntail > 0
        l = view(F.w, 1:ntail)
        for r in 1:ntail
            l[r] = F.L[i + r, i]
        end
        rank_one!(F, i, l, F.D[i], ntail)

        # Shift the repaired trailing block, and the columns left of i, up-left.
        for c in 1:(i - 1), r in 1:ntail
            F.L[i + r - 1, c] = F.L[i + r, c]
        end
        for j in 1:ntail, r in (j + 1):ntail
            F.L[i + r - 1, i + j - 1] = F.L[i + r, i + j]
        end
        for r in 1:ntail
            F.D[i + r - 1] = F.D[i + r]
            F.L[i + r - 1, i + r - 1] = one(T)
        end
    end
    F.k = k - 1
    return F
end

"""
    singular_direction!(p, F, i) -> p

A direction in the space of active multipliers along which the Gram matrix is
singular, for the dependency marked by a zero at `D[i]`. It satisfies
`Mₐᵀ p = 0`, so moving along it changes the multipliers and not the primal point.
"""
function singular_direction!(p::AbstractVector{T}, F::GramLDL{T}, i::Integer) where {T}
    fill!(p, zero(T))
    if i > 1
        for j in 1:(i - 1)
            p[j] = -F.L[i, j]
        end
        backward_L!(F, view(p, 1:(i - 1)))
    end
    p[i] = one(T)
    return p
end

"""
    WorkingSetGram{T}

The working set held as the `LDLᵀ` of `Mₐ Mₐᵀ`, with the active rows gathered beside it.

This is the representation the method was published with, and it is the faster of the two:
a row entering costs one rank-one extension of the factorization against `k` values, where
[`WorkingSetQR`](@ref) orthogonalizes the entering row against every row held.

The Gram matrix has the condition number of `Mₐ` squared, so its pivots and the multipliers
that solve against it carry that conditioning. The pivot that decides dependence is formed by
cancellation, which [`refine_pivot!`](@ref) replaces with an explicit residual when the pivot is
small enough for the cancellation to matter. On a reduction whose conditioning is worse than
that repairs, the method can declare a working set singular that is not, walk a direction that
means nothing, and report a feasible problem infeasible. `qrset.jl` says what the other
representation does instead.

Choose it with `ActiveSet(; working_set = :gram)` when the problems are known to be well
conditioned and the speed matters.
"""
struct WorkingSetGram{T <: Real}
    F::GramLDL{T}
    # The active rows, gathered as columns in working-set order. A product against the
    # working set is then one matrix-vector call rather than one short call per row, and the
    # entering row's products against the set are one more.
    Ma::Matrix{T}
    g::Vector{T}    # scratch: the entering row against the rows held
    r::Vector{T}    # scratch: the part of the entering row outside the span of the rows held
    c::Vector{T}    # scratch: the coefficients of the entering row's projection on that span
end

"An empty working set over `n` variables, sized for the largest it can reach."
function WorkingSetGram{T}(n::Integer, kmax::Integer) where {T <: Real}
    k = Int(kmax)
    return WorkingSetGram{T}(
        GramLDL{T}(k), zeros(T, Int(n), k), zeros(T, k), zeros(T, Int(n)), zeros(T, k)
    )
end

@inline nactive(W::WorkingSetGram) = W.F.k

"""
The most rows this representation can hold.

The `LDLᵀ` is of a `k × k` Gram matrix, which stays square however dependent its rows are, so
the set has room for the dependent row the singular step needs. [`WorkingSetQR`](@ref) has
not, which is what [`full_set_step!`](@ref) exists for.
"""
@inline maxrows(W::WorkingSetGram) = size(W.Ma, 2)

"""
A new pivot at most this multiple of `eps(T)/min(D)` is recomputed from an explicit residual.

The pivot is `‖m‖² − gᵀc` for `c` solving the Gram system, a difference of two numbers near one
whose rounding error is about `eps(T)`, and `c` itself carries that error amplified by `1/min(D)`.
A pivot below this threshold may be dominated by that error. The rows of `M` have unit norm.
"""
const PIVOT_REFINE_GAIN = 100

"""
    suspect_pivot(F, k) -> Bool

Whether the pivot just appended after `k` rows is small enough for rounding to dominate it. A
pivot stored as zero counts, because [`add_row!`](@ref) stores zero for any pivot that came out
at or below `eps`, and that includes pivots whose exact value is above it. A set that holds a
dependent row has no accurate `min(D)` to test against and never counts. Inlined, because it
runs on every row that enters and is false for nearly all of them.
"""
@inline function suspect_pivot(F::GramLDL{T}, k::Int) where {T}
    k > 0 || return false
    dnew = F.D[k + 1]
    dmin = typemax(T)
    for i in 1:k
        d = F.D[i]
        dmin = d < dmin ? d : dmin
    end
    return dmin > zero(T) && dnew * dmin < PIVOT_REFINE_GAIN * eps(T)
end

"""
    refine_pivot!(W, m_r, k) -> W

Recompute the pivot of the row just appended as the squared norm of its residual against the
`k` rows held, for a pivot [`suspect_pivot`](@ref) flags, unless the row's own coefficients
show that the pivot the factorization wrote is accurate.

The coefficients `c` of the projection of `m_r` on the held rows come from the factorization.
The pivot it wrote is exact for a Gram matrix that is off by about `k·eps(T)` an entry, which
moves the pivot by at most `k·eps(T)·‖c‖₁²`. `suspect_pivot` flags on `1/min(D)`, which stands
for `‖c‖²` before `c` is known, and a row whose projection puts little weight on the nearly
dependent rows has a far smaller one. A pivot more than `PIVOT_REFINE_GAIN` times that bound
stands, with the row of `L` written beside it. A pivot stored as zero never does.

Otherwise `c` is refined once against the explicit residual `r = m_r − Mₐᵀc`. The pivot is then
`‖r‖²`, which is formed without the cancellation, and the new row of `L` is `Lᵀc`, the row that
reproduces `g` under the factorization. A pivot that is still at or below `eps(T)` is stored as
an exact zero, which is what [`first_dependent`](@ref) reads.
"""
@noinline function refine_pivot!(W::WorkingSetGram{T}, m_r::AbstractVector{T}, k::Int) where {T}
    F = W.F
    Mk = view(W.Ma, :, 1:k)
    c = view(W.c, 1:k)
    r = W.r
    for i in 1:k
        c[i] = F.L[k + 1, i]
    end
    backward_L!(F, c)
    c1 = zero(T)
    for i in 1:k
        c1 += abs(c[i])
    end
    PIVOT_REFINE_GAIN * k * eps(T) * c1^2 < F.D[k + 1] && return W
    # `r = m_r − Mₐᵀc`, then one correction of `c` against it, then `r` again.
    for pass in 1:2
        for i in eachindex(r)
            r[i] = m_r[i]
        end
        mul!(r, Mk, c, -one(T), one(T))
        pass == 2 && break
        rhs = view(W.g, 1:k)
        mul!(rhs, transpose(Mk), r)
        solve_gram!(F, rhs)
        for i in 1:k
            c[i] += rhs[i]
        end
    end
    pivot = dot(r, r)
    # `Lᵀc`, in place of the row the factorization wrote.
    for j in 1:k
        s = c[j]
        for i in (j + 1):k
            s += F.L[i, j] * c[i]
        end
        F.L[k + 1, j] = s
    end
    F.D[k + 1] = pivot > eps(T) ? pivot : zero(T)
    return W
end

"""
    add_row!(W, m_r) -> Bool

Append `m_r` to the working set. `false` when the row is not finite, which is the one case
the caller cannot continue from.

A dependent row goes in at `D[k] = 0`, which is the value [`first_dependent`](@ref) reads. A
pivot small enough for rounding to dominate it is first checked by [`refine_pivot!`](@ref),
which recomputes it unless the row's own coefficients show it accurate, and the row is dependent
only if the pivot that leaves is at or below `eps(T)`.

The tolerance is `eps(T)` rather than the `sqrt(eps(T))` the factorization defaults to,
because `D` holds the *square* of the distance `R`'s diagonal holds in the other
representation. Compared against `sqrt(eps(T))`, a row counts as dependent once that distance
falls below `eps(T)^(1/4)`, about `1e-4`, four orders above where [`WorkingSetQR`](@ref)
draws the line; rows well inside the span of the others are then dropped, and the run settles
on an active set that is not the solution's.
"""
function add_row!(W::WorkingSetGram{T}, m_r::AbstractVector{T}) where {T}
    k = W.F.k
    beta = dot(m_r, m_r)
    isfinite(beta) || return false
    g = view(W.g, 1:k)
    mul!(g, transpose(view(W.Ma, :, 1:k)), m_r)
    add_row!(W.F, g, beta, eps(T))
    # A row past the variable count is dependent on the rows held, whatever its pivot came out
    # as: `n + 1` rows in `n` variables always admit a combination. That pivot is formed by
    # cancellation, so it is noise of any size rather than the zero `first_dependent` reads, and
    # a set claiming full rank in fewer dimensions than it has rows answers a singular system as
    # though it were solvable.
    if k + 1 > size(W.Ma, 1)
        W.F.D[k + 1] = zero(T)
    elseif suspect_pivot(W.F, k)
        refine_pivot!(W, m_r, k)
    end
    # An explicit loop rather than `copyto!`, which checks whether two views of one matrix
    # alias and copies the source when it cannot tell -- an allocation site the hot-path
    # guarantee sees whether or not the branch can be reached.
    dest = view(W.Ma, :, k + 1)
    @inbounds @simd for i in eachindex(dest)
        dest[i] = m_r[i]
    end
    return true
end

"Drop the `i`th row of the working set."
function remove_row!(W::WorkingSetGram, i::Integer)
    k = W.F.k
    remove_row!(W.F, Int(i))
    # Close the gap in the gathered block. Columns `i+1:k` are one contiguous run, so the
    # tail moves as a single block rather than column by column.
    if i < k
        n = size(W.Ma, 1)
        copyto!(W.Ma, (i - 1) * n + 1, W.Ma, i * n + 1, (k - i) * n)
    end
    return W
end

"""
    first_dependent(W, zero_tol) -> Int

The first row whose distance from the span of the rows before it is at or below `zero_tol`,
or 0.

`D` holds that distance *squared* — the pivot `‖Mₐᵀp‖²` for the part of the row orthogonal to
those before it — so the comparison is against `zero_tol^2`: `zero_tol` names a distance, and
[`WorkingSetQR`](@ref) holds the distance itself. The square has to stay. Against `zero_tol`
itself the line falls at `sqrt(zero_tol)`, four orders coarser at the default, which declares
a working set singular where it is not, walks a direction that means nothing, and reports a
feasible problem infeasible. At the default `zero_tol` the square is `eps(T)`, the tolerance
[`add_row!`](@ref) admits a row at, so the two tests agree about which rows are dependent.

A pivot small enough for the cancellation in forming it to matter is recomputed from an explicit
residual when the row enters ([`refine_pivot!`](@ref)), so the entry this reads is accurate
there. The multipliers still solve the squared system, which is the limit
[`WorkingSetQR`](@ref) exists for.
"""
@inline function first_dependent(W::WorkingSetGram{T}, zero_tol::T) where {T}
    F = W.F
    tol = zero_tol * zero_tol
    @inbounds for i in 1:F.k
        F.D[i] <= tol && return i
    end
    return 0
end

solve_gram!(W::WorkingSetGram{T}, rhs::AbstractVector{T}) where {T} = solve_gram!(W.F, rhs)

null_direction!(p::AbstractVector{T}, W::WorkingSetGram{T}, i::Integer) where {T} =
    singular_direction!(p, W.F, Int(i))

"`u ← Mₐᵀ μ`, the primal point the working set's multipliers place."
function active_product!(
        u::AbstractVector{T}, W::WorkingSetGram{T}, mu::AbstractVector{T}, ::AbstractVector{T}
    ) where {T}
    k = W.F.k
    mul!(u, view(W.Ma, :, 1:k), view(mu, 1:k))
    return u
end

"""
    conditioning(W) -> T

An estimate of `cond(Mₐ)`, or zero when the working set gives none.

`D` holds the squares of what `R`'s diagonal holds in the other representation, so the ratio
is taken under a square root to estimate the same quantity. That the estimate has to come
through a square is the same reason this representation cannot decide rank as finely.
"""
function conditioning(W::WorkingSetGram{T}) where {T}
    D = W.F.D
    dmin = typemax(T)
    dmax = zero(T)
    @inbounds for i in 1:W.F.k
        d = D[i]
        d > zero(T) || continue
        dmin = min(dmin, d)
        dmax = max(dmax, d)
    end
    return (dmax > zero(T) && dmin < typemax(T)) ? sqrt(dmax / dmin) : zero(T)
end
