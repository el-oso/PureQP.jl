@enum LDPStatus LDP_OPTIMAL LDP_INFEASIBLE LDP_ITERATION_LIMIT LDP_CYCLED

const SIDE_INACTIVE = Int8(0)
const SIDE_UPPER = Int8(1)
const SIDE_LOWER = Int8(-1)

"""
    DenseRows(Mt)

The reduced constraint matrix `M = A R⁻¹`, row-normalized, stored transposed as `n × m`.

Every kernel of the loop reads one *row* of `M`, which as a column of `Mt` is contiguous. Held
the other way round each read is strided, BLAS drops to its scalar path, and every element
costs a cache line.
"""
struct DenseRows{T <: Real}
    Mt::Matrix{T}
end

"""
    ImplicitRows(A, R, Rt, scale, rowbuf, tmp)

The reduced constraint matrix `M = A R⁻¹`, row-normalized, as its two operands rather than as
an `m × n` array.

    M[r, :] = R⁻ᵀ(Aᵀ e_r) / scale[r]          M u = (A (R⁻¹u)) ./ scale

so a row costs one [`PureQPBase.dense_row!`](@ref) and one transposed solve, and pricing costs
one solve and one product of `A`. Storage is `A`'s own plus a handful of length-`n` and
length-`m` vectors, whatever `m` is.

`rowbuf` is the vector [`row`](@ref) hands back, so only one row is readable at a time; `tmp`
carries `R⁻¹u` into the product. Both are overwritten by every call, and `R` and `A` may keep
scratch of their own, so one of these must not be read from two tasks at once.
"""
struct ImplicitRows{T <: Real, MA <: AbstractMatrix{T}, F, FT}
    A::MA
    R::F
    # `transpose(R)`, held rather than formed per row: the wrapper is a heap allocation on
    # Julia 1.13 for a triangular `R`, and a row is read every iteration.
    Rt::FT
    scale::Vector{T}
    rowbuf::Vector{T}
    tmp::Vector{T}
end

"How many variables the reduced rows have."
nvars(M::DenseRows) = size(M.Mt, 1)
nvars(M::ImplicitRows) = size(M.A, 2)

"How many reduced rows there are."
nreduced(M::DenseRows) = size(M.Mt, 2)
nreduced(M::ImplicitRows) = size(M.A, 1)

"Row `r` of the reduced constraint matrix, as a dense length-`n` vector."
@inline row(M::DenseRows, r::Integer) = view(M.Mt, :, r)

function row(M::ImplicitRows{T}, r::Integer) where {T}
    buf = M.rowbuf
    dense_row!(buf, M.A, r)
    ldiv!(M.Rt, buf)
    s = M.scale[r]
    @simd for i in eachindex(buf)
        buf[i] /= s
    end
    return buf
end

"""
    price!(dest, M, u, rows) -> dest

Write `(M u)[rows]` into `dest[rows]`.

[`DenseRows`](@ref) prices the rows asked for and no others. [`ImplicitRows`](@ref) prices
every row whatever `rows` was: `M u = A(R⁻¹u) ./ scale` reaches all of them for the price of
one, so a window saves nothing, and pricing all of them leaves the row a caller selects inside
the window the same row it would have selected from a windowed product.
"""
function price!(
        dest::AbstractVector{T}, M::DenseRows{T}, u::AbstractVector{T}, rows::UnitRange{Int}
    ) where {T}
    if length(rows) == size(M.Mt, 2)
        mul!(dest, transpose(M.Mt), u)
    else
        mul!(view(dest, rows), transpose(view(M.Mt, :, rows)), u)
    end
    return dest
end

function price!(
        dest::AbstractVector{T}, M::ImplicitRows{T}, u::AbstractVector{T}, ::UnitRange{Int}
    ) where {T}
    tmp = M.tmp
    copyto!(tmp, u)
    ldiv!(M.R, tmp)
    mul!(dest, M.A, tmp)
    scale = M.scale
    @simd for j in eachindex(dest, scale)
        dest[j] /= scale[j]
    end
    return dest
end

"""
State for one two-sided least-distance problem, `lo ≤ Mu ≤ hi`.

A row is in the working set at one of its two bounds, and `side` records which. Its
multiplier is then signed: non-negative at the upper bound, non-positive at the lower one.
Working in those signed multipliers means the factored Gram matrix is over the *unsigned*
rows, so a row switching sides costs nothing — the factorization does not change.

That is also why there is one row here per problem row. Splitting each two-sided row into
two one-sided rows would double the length of the pricing loop, which is the dominant cost
of an iteration.

`active` lists the working set in the order it was built, matching the order of the `LDLᵀ`;
`slot[j]` is where row `j` sits in it, or 0.
"""
# Immutable: nothing here is ever rebound. Every field is a buffer written through, and the
# live size lives in `F.k`, which is why that one stays in a mutable struct of its own.
struct LDPWorkspace{T <: Real, WS, MR}
    # The reduced constraint matrix, read one row at a time through `row` and priced as a whole
    # through `price!`. `DenseRows` holds it; `ImplicitRows` holds `A` and `R` instead, so
    # nothing here scales with `m·n`. Concrete once instantiated, so neither costs a dispatch.
    M::MR
    hi::Vector{T}    # upper target, recomputed whenever `v` changes
    lo::Vector{T}
    iseq::Vector{Bool}
    side::Vector{Int8}
    # `active` and `mu` are preallocated to the largest working set and share their live
    # length with the factorization's `F.k`. Growing them with `push!` instead would allocate
    # on a cold solve — invisible to a measurement, because `deleteat!` keeps the capacity a
    # previous solve grew, but a real allocation in the hot path all the same.
    active::Vector{Int}
    slot::Vector{Int}
    mu::Vector{T}    # signed multipliers of the active rows, in `active` order
    mu_star::Vector{T}
    p::Vector{T}
    u::Vector{T}     # primal point of the least-distance problem, Mₐᵀμ
    g::Vector{T}     # scratch: products against the active rows
    Mv::Vector{T}    # scratch: M * v
    # The working set. Two representations answer the same questions at different cost and
    # accuracy -- `WorkingSetQR` factors the rows themselves, `WorkingSetGram` their Gram
    # matrix -- and the caller chooses between them. Concrete once instantiated, so neither
    # costs a dispatch in the loop. `qrset.jl` and `ldl.jl` say what the choice buys.
    W::WS
    price::Vector{T} # scratch: M * u over the rows priced
    # How many leading entries of `active` and `p` the infeasibility ray spans, or 0 for the
    # working set's own length. They differ only after [`full_set_step!`](@ref) proves
    # infeasibility, where the ray covers one row more than the set holds.
    certrow::Vector{Int}
    # The norm each row of `M` was divided by. Pricing multiplies it back, so a row is
    # compared against the tolerance in the units the caller stated its bounds in.
    scale::Vector{T}
    v::Vector{T}     # reduced linear term, rebuilt at the start of every pass
    xbuf::Vector{T}  # primal iterate, and the vector `run_daqp!` returns
    xold::Vector{T}  # proximal centre of the previous pass
end

function LDPWorkspace(M, iseq::AbstractVector{Bool}, scale::Vector{T}, W) where {T <: Real}
    n, m = nvars(M), nreduced(M)
    kmax = min(m, n) + 1
    return LDPWorkspace{T, typeof(W), typeof(M)}(
        M, zeros(T, m), zeros(T, m), convert(Vector{Bool}, iseq), zeros(Int8, m),
        zeros(Int, kmax), zeros(Int, m), zeros(T, kmax), zeros(T, kmax), zeros(T, kmax),
        zeros(T, n), zeros(T, kmax), zeros(T, m),
        W, zeros(T, m), zeros(Int, 1), scale,
        zeros(T, n), zeros(T, n), zeros(T, n)
    )
end

"""
    build_working_set(kind, T, n, kmax)

The working set `kind` names, over `n` variables and at most `kmax` rows.

`:rows` factors the active rows themselves and `:gram` their Gram matrix; the two answer the
same questions, at different cost and different accuracy.
"""
function build_working_set(kind::Symbol, ::Type{T}, n::Integer, kmax::Integer) where {T <: Real}
    kind === :rows && return WorkingSetQR{T}(n, kmax)
    kind === :gram && return WorkingSetGram{T}(n, kmax)
    return throw(ArgumentError(lazy"working_set must be :rows or :gram, got :$kind"))
end


"Row `r` of the reduced constraint matrix the workspace runs on."
@inline row(ws::LDPWorkspace, r::Integer) = row(ws.M, r)

"The working set, as the live prefix of the preallocated buffer."
@inline activeset(ws::LDPWorkspace) = view(ws.active, 1:nactive(ws.W))

"The bound row `j` is held at, given the side it entered on."
@inline target(ws::LDPWorkspace, j::Integer) = ws.side[j] == SIDE_LOWER ? ws.lo[j] : ws.hi[j]

"""
The sign row `j`'s multiplier must carry to be dual feasible.

`u = Mₐᵀμ`, and a row held at its *upper* target pushes `u` in the negative direction of
that row, so its multiplier is non-positive; one held at its lower target is non-negative.
Multiplying by this makes both cases read `musign * μ ≥ 0`.
"""
@inline musign(ws::LDPWorkspace{T}, j::Integer) where {T} =
    ws.side[j] == SIDE_UPPER ? -one(T) : one(T)

# `lazy` defers the interpolation past the call graph trim verifies, which an eagerly built
# message does not: it lowers to a `string` call whose argument tuple inference despecializes
# to an `Any` vararg.
@noinline _nonfinite_row(r::Integer) =
    throw(ArgumentError(lazy"row $r of the reduced constraint matrix is not finite"))

"""
Put row `r` into the working set at `side`, extending the factorization.

The row goes in whatever its residual against the rows already held: a dependent row is
exactly the one the method needs in the set, because the direction its dependency admits is
how the pass either drops a blocking row or proves the problem infeasible. Which rows are
dependent is then [`first_dependent`](@ref)'s question, asked of `|R_ii|`.
"""
function activate!(ws::LDPWorkspace{T}, r::Integer, side::Int8) where {T}
    k = nactive(ws.W)
    add_row!(ws.W, row(ws, r)) || _nonfinite_row(r)
    ws.active[k + 1] = r
    ws.mu[k + 1] = zero(T)
    ws.side[r] = side
    ws.slot[r] = k + 1
    return ws
end

"Take the `i`-th working-set row back out."
function deactivate!(ws::LDPWorkspace, i::Integer)
    k = nactive(ws.W)
    r = ws.active[i]
    # The factorization closes its own gap: deleting a column returns the columns after it to
    # triangular form with a sweep of rotations, so the order below matches it.
    remove_row!(ws.W, i)
    for j in i:(k - 1)
        ws.active[j] = ws.active[j + 1]
        ws.mu[j] = ws.mu[j + 1]
        ws.slot[ws.active[j]] = j
    end
    ws.slot[r] = 0
    ws.side[r] = SIDE_INACTIVE
    return ws
end

"""
    blocking_step(ws, p, zero_tol) -> (alpha, blocking)

How far the multipliers move along `p` before one reaches zero from its feasible side, and
which row of the working set it is. `blocking` is 0 when nothing blocks, and `alpha` is then
meaningless.

An equality row never blocks: its multiplier is free in sign.
"""
function blocking_step(ws::LDPWorkspace{T}, p::AbstractVector{T}, zero_tol::T) where {T}
    alpha = typemax(T)
    blocking = 0
    for i in 1:nactive(ws.W)
        r = ws.active[i]
        ws.iseq[r] && continue
        # Feasibility is `musign * mu >= 0`, so the step blocks when `musign * p < 0`.
        musign(ws, r) * p[i] < -zero_tol || continue
        cand = -ws.mu[i] / p[i]
        if cand < alpha
            alpha = cand
            blocking = i
        end
    end
    return alpha, blocking
end

"""
    step_and_drop!(ws, p, zero_tol) -> Bool

Move the multipliers along `p` until the first one reaches zero from its feasible side, then
drop that row. `false` when nothing blocks, which on the singular branch means the problem
is infeasible.
"""
function step_and_drop!(ws::LDPWorkspace{T}, p::AbstractVector{T}, zero_tol::T) where {T}
    alpha, blocking = blocking_step(ws, p, zero_tol)
    iszero(blocking) && return false
    for i in 1:nactive(ws.W)
        ws.mu[i] += alpha * p[i]
    end
    deactivate!(ws, blocking)
    return true
end

"""
    full_set_step!(ws, r, side, zero_tol) -> Bool

Step toward row `r` when the working set already spans every variable.

`r` cannot be added: the factored matrix is `n × k` and the factorization needs more rows
than columns. Nor should it be. With `k = n` the row is a combination of the rows held
whatever residual it would have shown, because `n + 1` rows in `n` variables always admit
one, so there is nothing for [`first_dependent`](@ref) to discover and no `R[n+1, n+1]` for
it to read. What the singular branch wants from such a row is the direction its dependency
opens, and that needs no place in the factorization: `Mₐᵀ` is square and of full rank here,
so `c = Mₐ⁻ᵀ m_r` is the combination and `[c; -1]` spans the null space of the rows held
together with `r`.

Only the first `k` entries of that direction move a stored multiplier. The last belongs to
`r`, whose multiplier would grow from zero, and the orientation below is what keeps it from
blocking the step immediately -- [`singular_step!`](@ref) applies the same one, for the same
reason.

`r` then takes the place the dropped row leaves. It has to: the direction is in the null
space of the rows held *together with* `r`, so it is only the pair of moves that holds
`Mₐᵀμ` still. Moving the stored multipliers while discarding `r`'s own would shift the dual
point by `alpha * m_r`, and the run stops converging.
"""
function full_set_step!(
        ws::LDPWorkspace{T, <:WorkingSetQR}, r::Integer, side::Int8, zero_tol::T
    ) where {T}
    k = nactive(ws.W)
    p = view(ws.p, 1:k)
    # `ldiv!` reads its right-hand side without writing to it, so the row of `M` is safe to
    # pass directly.
    ldiv!(p, ws.W.qr, row(ws, r))
    if side != SIDE_UPPER
        for i in eachindex(p)
            p[i] = -p[i]
        end
    end
    alpha, blocking = blocking_step(ws, p, zero_tol)
    if iszero(blocking)
        # Nothing blocks, so this direction proves the problem infeasible -- and the proof
        # includes `r`, which is not in the working set and so is not in `p[1:k]`. Recording
        # it past the live prefix is what lets `certifiable` rebuild the whole ray: without
        # its `∓1` the direction does not satisfy `Aᵀy = 0`, the check against the caller's
        # rows fails, and a sound proof is reported as a numerical breakdown instead.
        ws.active[k + 1] = r
        ws.p[k + 1] = side == SIDE_UPPER ? -one(T) : one(T)
        ws.certrow[1] = k + 1
        return false
    end
    for i in 1:k
        ws.mu[i] += alpha * p[i]
    end
    deactivate!(ws, blocking)
    activate!(ws, r, side)
    ws.mu[nactive(ws.W)] = side == SIDE_UPPER ? -alpha : alpha
    return true
end

# The Gram representation has room for the dependent row, so the case above does not arise
# for it: `maxrows` counts one more than there are variables, the row goes in at a zero
# pivot, and `first_dependent` picks it up on the next pass through the loop.
full_set_step!(
    ws::LDPWorkspace{T, <:WorkingSetGram}, r::Integer, side::Int8, ::T
) where {T} = (activate!(ws, r, side); true)

"""
    singular_step!(ws, singular, zero_tol) -> Bool

Walk the null direction of a singular working set and drop the first row that blocks.
`false` when nothing blocks, which is what makes the problem infeasible.
"""
function singular_step!(ws::LDPWorkspace{T}, singular::Int, zero_tol::T) where {T}
    p = view(ws.p, 1:nactive(ws.W))
    null_direction!(p, ws.W, singular)
    # `singular_direction!` puts `+1` at the dependent row. Signed multipliers make the two
    # sides asymmetric, so that `+1` is a blocking candidate for a row held at its upper
    # bound — and blocking on it would drop the row that just entered, which pricing would
    # then choose again, forever. Orienting the direction so the dependent row moves the
    # feasible way restores the invariant the one-sided form gets for free, and leaves the
    # dependency to be resolved by one of the rows it depends on.
    if musign(ws, ws.active[singular]) < 0
        for i in eachindex(p)
            p[i] = -p[i]
        end
    end
    return step_and_drop!(ws, p, zero_tol)
end

"""
    working_set_multipliers!(ws) -> Bool

Solve `Mₐ Mₐᵀ μ* = tₐ` for the bounds the working set is held at, into `ws.mu_star`.

`true` when every inequality multiplier has the sign its side requires, which is what makes
the point dual feasible. An equality row is held whatever the sign.
"""
function working_set_multipliers!(ws::LDPWorkspace{T}) where {T}
    k = nactive(ws.W)
    mus = view(ws.mu_star, 1:k)
    @inbounds for i in 1:k
        mus[i] = target(ws, ws.active[i])
    end
    solve_gram!(ws.W, mus)
    tol = multiplier_noise(ws, mus)
    @inbounds for i in 1:k
        r = ws.active[i]
        !ws.iseq[r] && musign(ws, r) * mus[i] < -tol && return false
    end
    return true
end

"""
    multiplier_noise(ws, mus) -> T

How far from zero a multiplier has to be for its sign to mean anything.

The multipliers come from `Mₐ Mₐᵀ μ = t`, whose conditioning is that of `Mₐ` squared, so
they carry a relative error of about `cond(Mₐ)² eps`. The factorization already holds an
estimate of that: the ratio of the largest and smallest diagonal entries of `R` bounds
`cond(Mₐ)`, and the multipliers carry its square. Tested against exact zero instead, a
multiplier that is indistinguishable from zero decides which row leaves the working set, and
the row it drops re-enters on the next pass.
"""
function multiplier_noise(ws::LDPWorkspace{T}, mus) where {T}
    k = nactive(ws.W)
    k > 0 || return zero(T)
    ratio = conditioning(ws.W)
    ratio > zero(T) || return zero(T)
    return 10 * k * eps(T) * ratio * ratio * maximum(abs, mus)
end

"""
    step_toward_multipliers!(ws, zero_tol) -> Bool

Step from `ws.mu` toward the multipliers just solved for, dropping the first row whose own
reaches zero. `false` when nothing blocks the step.
"""
function step_toward_multipliers!(ws::LDPWorkspace{T}, zero_tol::T) where {T}
    k = nactive(ws.W)
    p = view(ws.p, 1:k)
    mus, mu = ws.mu_star, ws.mu
    @inbounds @simd for i in 1:k
        p[i] = mus[i] - mu[i]
    end
    return step_and_drop!(ws, p, zero_tol)
end

"""
    primal_point!(ws) -> ws

Adopt the multipliers just solved for and form the primal point `u = Mₐᵀ μ` from them.
"""
function primal_point!(ws::LDPWorkspace{T}) where {T}
    k = nactive(ws.W)
    mu, mus = ws.mu, ws.mu_star
    # An explicit loop rather than `copyto!`: between two views of a vector that checks
    # whether they alias and copies the source if it cannot tell, which is an allocation
    # site the hot-path guarantee sees whether or not the branch can be reached.
    @inbounds @simd for i in 1:k
        mu[i] = mus[i]
    end
    active_product!(ws.u, ws.W, ws.mu, ws.g)
    return ws
end

"""
    entering_row(ws, primal_tol, bland, rows) -> (row, side)

The row of `rows` that violates its bound by the most, and the side it violates, or
`(0, SIDE_UPPER)` when none does.

Reads `price` only over `rows`, so a caller that priced a window may pass that window.
Deciding that no row violates — which is what ends the run — needs every row of `M`, so only
a caller that priced all of them may conclude it from a zero here.

Dantzig's rule takes the worst violation, which is the fast choice but can cycle: a row that
keeps swapping sides re-enters forever. `bland` switches to the lowest violated index,
Bland's rule, which terminates finitely at the cost of taking more steps.
"""
function entering_row(
        ws::LDPWorkspace{T}, primal_tol::T, bland::Bool, rows::UnitRange{Int}
    ) where {T}
    worst = primal_tol
    entering = 0
    entering_side = SIDE_UPPER
    # Not `@simd`: this is an argmax search, and the `slot` test skips the working set.
    @inbounds for r in rows
        iszero(ws.slot[r]) || continue
        # Priced in the caller's own units. The reduction divided each row by its norm,
        # and those norms span five orders here, so a violation that is at the tolerance in
        # the normalized rows is that much larger in the problem the caller posed: the row
        # that is worst after scaling is not the row that is worst to them.
        sr = ws.scale[r]
        rr = ws.price[r] * sr
        over = rr - ws.hi[r] * sr
        under = ws.lo[r] * sr - rr
        if over > worst
            worst = bland ? primal_tol : over
            entering = r
            entering_side = SIDE_UPPER
            bland && break
        end
        if under > worst
            worst = bland ? primal_tol : under
            entering = r
            entering_side = SIDE_LOWER
            bland && break
        end
    end
    return entering, entering_side
end

"""
    solve_ldp!(ws, alg, max_iter) -> (status, iterations)

Algorithm 1 of Arnström, Bemporad & Axehill, *A dual active-set solver for embedded
quadratic programming using recursive LDLᵀ updates*, IEEE TAC 2022, on the two-sided
problem `lo ≤ Mu ≤ hi`.

Each iteration solves `Mₐ Mₐᵀ μ = tₐ` for the bounds the working set is held at. Multipliers
that are feasible in sign make the point dual feasible, and the run stops once no inactive
row is outside its bounds; otherwise the worst violator enters at the side it violates.
Multipliers that are not lead to a step toward them, dropping the first row whose multiplier
reaches zero. A singular Gram matrix is handled by walking along its null direction.

Takes its tolerances from `alg` for the reason [`run_daqp!`](@ref) does, which also leaves it
a signature `test_signatures` can state.
"""
function solve_ldp!(ws::LDPWorkspace{T}, alg::ActiveSet{T}, max_iter::Int) where {T}
    zero_tol, primal_tol = alg.zero_tol, alg.primal_tol
    m = nreduced(ws.M)
    bland_after = 4 * (m + 1)
    # How many rows one iteration examines, and the row it starts at. `chunk = m` examines
    # every row, which is what `scan = :all` asks for and what leaves the entering row the
    # worst violator rather than the worst one near where the last search stopped.
    chunk = alg.scan === :window ? max(256, m >> 3) : m
    scan = 1
    # Any ray a previous pass left behind describes a working set this one no longer holds.
    ws.certrow[1] = 0
    for iter in 1:max_iter
        # Every step reads `nactive(ws.W)` for itself: dropping and adding a row both change it, so
        # the working set's size does not survive an iteration.
        singular = first_dependent(ws.W, zero_tol)
        if !iszero(singular)
            singular_step!(ws, singular, zero_tol) || return (LDP_INFEASIBLE, iter)
            continue
        end
        if !working_set_multipliers!(ws)
            step_toward_multipliers!(ws, zero_tol) || return (LDP_CYCLED, iter)
            continue
        end
        primal_point!(ws)

        # Price with one matrix-vector product rather than a dot product per inactive row. The
        # working set is priced too and its values ignored, which is `k` wasted products out of
        # however many are priced — cheaper than it sounds, because a dot product per row is
        # one BLAS call per row, and on a short row a call costs about what the arithmetic
        # does. One call for all of them is what makes the small sizes competitive.
        #
        # A window of `chunk` rows, resumed where the last one entered, in place of all `m`:
        # any violated row is a step the method can take, so it does not have to be the worst
        # one, and pricing is the bulk of an iteration. What the window cannot decide is that
        # *no* row violates, so a window that finds nothing prices the rest before the run ends
        # on it. Bland's rule prices everything too: its finite termination rests on scanning
        # indices in a fixed order, which a moving window does not.
        bland = iter > bland_after
        entering, entering_side = 0, SIDE_UPPER
        if !bland && chunk < m
            window = scan:min(scan + chunk - 1, m)
            price!(ws.price, ws.M, ws.u, window)
            entering, entering_side = entering_row(ws, primal_tol, false, window)
        end
        if iszero(entering)
            price!(ws.price, ws.M, ws.u, 1:m)
            entering, entering_side = entering_row(ws, primal_tol, bland, 1:m)
            iszero(entering) && return (LDP_OPTIMAL, iter)
        end
        scan = entering == m ? 1 : entering + 1
        if nactive(ws.W) == maxrows(ws.W)
            # The set already spans every variable, so the entering row is dependent by
            # counting and is stepped toward rather than added. A row drops, and pricing
            # reaches this row again with room to hold it.
            full_set_step!(ws, entering, entering_side, zero_tol) ||
                return (LDP_INFEASIBLE, iter)
        else
            activate!(ws, entering, entering_side)
        end
    end
    return (LDP_ITERATION_LIMIT, max_iter)
end

"""
The once-computed part of the reduction from a QP to a least-distance problem: the Cholesky
factor `R` of `P + εI`, the transformed and row-normalized constraint matrix `M = A R⁻¹`
with the caller's bounds in the same scaling, and the working state.

Only the targets change between proximal-point iterations, so `R`, `M` and the `LDLᵀ` of the
working set are all reused.
"""
struct DAQPReduction{T <: Real, F, FT, WS, MR}
    # `R` is in `P`'s own form, whatever [`PureQPBase.cholesky_factor`](@ref) handed back for
    # it: an `UpperTriangular` for a matrix, a `Diagonal`, a block-diagonal or Kronecker factor
    # for those. Only the two solves below are asked of it.
    R::F
    # `transpose(R)`, kept rather than formed per pass. It solves against the stored factor
    # without copying it, but the wrapper itself is a heap allocation on Julia 1.13, once per
    # solve in a loop that has none otherwise.
    Rt::FT
    bu::Vector{T}     # the caller's bounds, divided by the row norms
    bl::Vector{T}
    eps_prox::T
    ws::LDPWorkspace{T, WS, MR}
    scale::Vector{T}  # the row norm each row was divided by
end

"""
    reduce_qp(P, A, bupper, blower, iseq; eps_prox) -> DAQPReduction

Factor `P + εI` and transform the constraints into `lo ≤ Mu ≤ hi` with `M = A R⁻¹`.

With `eps_prox = 0` this needs `P ≻ 0`. Any `ε > 0` makes `P + εI` positive definite for
`P ⪰ 0`, which is what lets the proximal-point loop take a singular `P`, an LP being the
extreme case.

`R` comes back in `P`'s own form, so a `P` that is not a dense matrix is never formed. `M` is
formed only when both `A` and `R` are dense; otherwise the reduction holds the two of them and
answers rows and products from them ([`ImplicitRows`](@ref)).
"""
function reduce_qp(
        P::AbstractMatrix{T}, A::AbstractMatrix{T},
        bupper::AbstractVector{T}, blower::AbstractVector{T},
        iseq::AbstractVector{Bool}; eps_prox::T = zero(T), working_set::Symbol = :rows
    ) where {T <: Real}
    has_cholesky_factor(P) || refuse_unfactorable_P()
    # One factorization answers the convexity question as well as supplying `R`: factoring
    # twice, once to check and once to use, is most of what setup costs.
    R = cholesky_factor(P, eps_prox)
    return build_reduction(R, A, bupper, blower, iseq, eps_prox, working_set)
end

"Refuse a `P` the reduction has no factor of, naming every form that has one."
@noinline refuse_unfactorable_P() = throw(
    ArgumentError(
        "ActiveSet() needs a P it can factor: the reduction forms A R⁻¹ for the Cholesky " *
            "factor R of P, and an operator that supplies products only has no such factor. " *
            "Pass P as a matrix, a Diagonal, a BlockDiagonal, a KroneckerOperator, or a " *
            "LinearMaps kron or blockdiag of matrices; or solve with OperatorSplitting(), " *
            "which needs no factor of P."
    )
)

"""
Normalize the rows of `M = A R⁻¹` and pack the reduction around them.

A dense `A` against a dense triangular `R` forms `M` once and reads it from memory afterwards.
Every other pair keeps `A` and `R` and derives each row and each product from them, which is
what makes the storage independent of `m·n`.
"""
function build_reduction(
        R::UpperTriangular{T, <:StridedMatrix}, A::StridedMatrix{T},
        bupper::AbstractVector{T}, blower::AbstractVector{T},
        iseq::AbstractVector{Bool}, eps_prox::T, kind::Symbol
    ) where {T}
    m, n = size(A)
    Mr = Matrix{T}(undef, m, n)
    copyto!(Mr, A)
    s = scalar_diagonal(R)
    if isnothing(s)
        # `M = A R⁻¹`, solved with the triangle on the right of the constraint matrix. Each
        # step of that substitution scales and subtracts whole columns of `M`, which are
        # contiguous and carry no dependence within a column; solving `R⁻ᵀ Aᵀ` instead makes
        # every entry a short dot product against the entries above it, and runs well under
        # half the speed.
        rdiv!(Mr, R)
    else
        # `R = sI`, the factor of `P = s²I`, so the reduction is a scaling rather than a
        # triangular solve. Worth the test because it is the shape of every objective that is
        # a plain squared norm.
        rmul!(Mr, one(T) / s)
    end
    # The rows are written out to `Mt` in the layout the loop reads, so the transpose costs no
    # pass of its own.
    Mt = Matrix{T}(undef, n, m)
    scale = Vector{T}(undef, m)
    bu = Vector{T}(undef, m)
    bl = Vector{T}(undef, m)
    for j in 1:m
        sq = zero(T)
        # `ivdep` throughout this loop: `Mr`, `Mt` and the bound vectors are separate buffers
        # allocated here, so no iteration can reach another's memory.
        @simd ivdep for i in 1:n
            sq += Mr[j, i]^2
        end
        # `norm` computes the same value while scaling against overflow and underflow, which
        # is what the sum of squares cannot represent; it covers exactly those two cases.
        nrm = (isfinite(sq) && sq > 0) ? sqrt(sq) : norm(view(Mr, j, :))
        # A row of `A` in the kernel of `R⁻ᵀ` normalizes to nothing; it is taken unscaled,
        # which is what a scale of one means.
        sj = nrm > 0 ? nrm : one(T)
        scale[j] = sj
        @simd ivdep for i in 1:n
            Mt[i, j] = Mr[j, i] / sj
        end
        bu[j] = bupper[j] / sj
        bl[j] = blower[j] / sj
    end
    return pack_reduction(R, DenseRows(Mt), bu, bl, iseq, scale, eps_prox, n, m, kind)
end

function build_reduction(
        R, A::AbstractMatrix{T}, bupper::AbstractVector{T}, blower::AbstractVector{T},
        iseq::AbstractVector{Bool}, eps_prox::T, kind::Symbol
    ) where {T}
    m, n = size(A)
    # Every scale is one until the loop below writes it, so the row read back there is the
    # unnormalized one whose norm is that scale.
    scale = ones(T, m)
    M = ImplicitRows(A, R, transpose(R), scale, Vector{T}(undef, n), Vector{T}(undef, n))
    bu = Vector{T}(undef, m)
    bl = Vector{T}(undef, m)
    for j in 1:m
        nrm = norm(row(M, j))
        # A row of `A` in the kernel of `R⁻ᵀ` normalizes to nothing; it is taken unscaled,
        # which is what a scale of one means.
        s = nrm > 0 ? nrm : one(T)
        scale[j] = s
        bu[j] = bupper[j] / s
        bl[j] = blower[j] / s
    end
    return pack_reduction(R, M, bu, bl, iseq, scale, eps_prox, n, m, kind)
end

"Wrap the working state around a factor and a reduced constraint matrix."
function pack_reduction(
        R, M, bu::Vector{T}, bl::Vector{T}, iseq::AbstractVector{Bool}, scale::Vector{T},
        eps_prox::T, n::Int, m::Int, kind::Symbol
    ) where {T}
    ws = LDPWorkspace(M, iseq, scale, build_working_set(kind, T, n, min(m, n) + 1))
    Rt = transpose(R)
    return DAQPReduction{T, typeof(R), typeof(Rt), typeof(ws.W), typeof(M)}(
        R, Rt, bu, bl, eps_prox, ws, scale
    )
end

"""
    set_targets!(red, v) -> red

Rebuild the least-distance bounds for the current `v`.

With `x = R⁻¹(−u − v)`, the caller's `bl ≤ Ax ≤ bu` becomes
`−(bu + Mv) ≤ Mu ≤ −(bl + Mv)`: the two bounds swap, because `Ax` runs against `Mu`.
"""
function set_targets!(red::DAQPReduction{T}, v::AbstractVector{T}) where {T}
    ws = red.ws
    price!(ws.Mv, ws.M, v, 1:nreduced(ws.M))
    for j in eachindex(ws.hi)
        ws.hi[j] = -(red.bl[j] + ws.Mv[j])
        ws.lo[j] = -(red.bu[j] + ws.Mv[j])
    end
    return red
end

"""
    rebuild_bounds!(red, bupper, blower) -> red

Adopt new caller bounds, keeping the factorization and the transformed constraint matrix.
"""
function rebuild_bounds!(red::DAQPReduction{T}, bupper::AbstractVector{T}, blower::AbstractVector{T}) where {T}
    iseq = red.ws.iseq
    moved = false
    for j in eachindex(red.bu)
        red.bu[j] = bupper[j] / red.scale[j]
        red.bl[j] = blower[j] / red.scale[j]
        # Which rows are equalities is a property of the bounds, so it is rebuilt with them.
        # Left alone, a row whose bounds have just been separated keeps a multiplier free in
        # sign and never blocks a step, and the pass settles on a point that is not the
        # answer -- reported as one, because nothing about the run looks wrong.
        eq = bupper[j] == blower[j]
        moved |= eq != iseq[j]
        iseq[j] = eq
    end
    return moved
end

"""
    reset_working_set!(red) -> red

Empty the working set apart from the equality rows, which must always be in it. The next
solve then starts where a fresh setup would.
"""
function reset_working_set!(red::DAQPReduction)
    ws = red.ws
    while nactive(ws.W) > 0
        deactivate!(ws, nactive(ws.W))
    end
    for j in eachindex(ws.iseq)
        ws.iseq[j] && activate!(ws, j, SIDE_UPPER)
    end
    return red
end

"""
    multipliers!(y, red, red_ws) -> y

Map the least-distance multipliers back to one per problem row, undoing the row scaling. The
sign already distinguishes the two bounds, so nothing else is needed.
"""
function multipliers!(y::AbstractVector{T}, red::DAQPReduction{T}) where {T}
    fill!(y, zero(T))
    ws = red.ws
    for (slot, j) in enumerate(activeset(ws))
        # The signed multiplier already distinguishes the two bounds: non-negative for a row
        # held at the caller's upper bound, non-positive at the lower one.
        y[j] = ws.mu[slot] / red.scale[j]
    end
    return y
end

"""
    inner_solve!(red, f, x, alg, max_iter) -> (status, iters, v)

One pass of Algorithm 1 at the current proximal centre `x`: rebuild the targets from
`v = R⁻ᵀ(f − εx)`, then run the dual active-set loop, warm-started from whatever working set
is in place.

Takes its tolerances from `alg` for the reason [`run_daqp!`](@ref) does.
"""
function inner_solve!(
        red::DAQPReduction{T}, f::AbstractVector{T}, x::AbstractVector{T},
        alg::ActiveSet{T}, max_iter::Int
    ) where {T}
    v = red.ws.v
    if iszero(red.eps_prox)
        copyto!(v, f)
    else
        eps_prox = red.eps_prox
        @simd for i in paired(v, f, x)
            v[i] = f[i] - eps_prox * x[i]
        end
    end
    # The held transpose, rather than one formed here: the wrapper is a heap allocation on
    # Julia 1.13, and it solves against the same stored factor either way.
    ldiv!(red.Rt, v)
    set_targets!(red, v)
    status, iters = solve_ldp!(red.ws, alg, max_iter)
    return status, iters, v
end

"""
    run_daqp!(red, f, alg, max_iter) -> (x, status, iters)

Algorithm 1, or Algorithm 2 wrapped around it when `eps_prox > 0`, on a reduction that
already exists. The working set is whatever the reduction holds, so a repeated call warm
starts from the previous answer.

The tolerances are taken from `alg` rather than passed one by one: a keyword call here is
boxed on Julia 1.13, which is an allocation in the one function a solve is meant not to have
any in.
"""
function run_daqp!(
        red::DAQPReduction{T}, f::AbstractVector{T}, alg::ActiveSet{T}, max_iter::Int
    ) where {T}
    zero_tol, primal_tol = alg.zero_tol, alg.primal_tol
    eps_prox, eta_prox, max_prox = alg.eps_prox, alg.eta_prox, alg.max_prox
    ws = red.ws
    # The returned vector is a workspace buffer. Callers copy it out before starting the
    # next solve, which writes it again.
    x = ws.xbuf
    fill!(x, zero(T))
    if iszero(eps_prox)
        status, iters, v = inner_solve!(red, f, x, alg, max_iter)
        status == LDP_OPTIMAL || return (x, status, iters)
        primal!(x, red, v)
        return (x, status, iters)
    end
    xold = ws.xold
    total = 0
    for _ in 1:max_prox
        copyto!(xold, x)
        status, iters, v = inner_solve!(red, f, x, alg, max_iter)
        total += iters
        status == LDP_OPTIMAL || return (x, status, total)
        primal!(x, red, v)
        d = zero(T)
        for i in paired(x, xold)
            d = max(d, abs(x[i] - xold[i]))
        end
        d < eta_prox && return (x, status, total)
    end
    return (x, LDP_ITERATION_LIMIT, total)
end

"Recover the primal point `x = R⁻¹(−u − v)` of the original problem, in place."
function primal!(x::AbstractVector{T}, red::DAQPReduction{T}, v::AbstractVector{T}) where {T}
    u = red.ws.u
    # A loop rather than a broadcast: `Base.Broadcast` keeps an `unaliascopy` branch it
    # cannot rule out between three buffers of the same type, which is an allocation site
    # whether or not the branch can be reached.
    @simd for i in paired(x, u, v)
        x[i] = -u[i] - v[i]
    end
    # `transpose(red.Rt)` unwraps to `red.R` itself, which is what makes this the untransposed
    # solve against the same stored factor.
    ldiv!(transpose(red.Rt), x)
    return x
end
