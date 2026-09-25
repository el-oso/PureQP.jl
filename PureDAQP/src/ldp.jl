@enum LDPStatus LDP_OPTIMAL LDP_INFEASIBLE LDP_ITERATION_LIMIT LDP_CYCLED

const SIDE_INACTIVE = Int8(0)
const SIDE_UPPER = Int8(1)
const SIDE_LOWER = Int8(-1)

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
struct LDPWorkspace{T <: Real}
    # Stored transposed, `n × m`: every kernel here reads one *row* of `M`, which as a
    # column of `Mt` is contiguous. Held the other way round each read is strided, BLAS
    # drops to its scalar path, and every element costs a cache line.
    Mt::Matrix{T}
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
    # The active rows of `M`, packed contiguously in working-set order, `n × kmax`. The same
    # rows live scattered through the columns of `Mt`; gathered here, the two products the
    # loop takes against the working set are single matrix-vector products rather than one
    # short BLAS call per active row, which is where the small sizes are won.
    Ma::Matrix{T}
    price::Vector{T} # scratch: M * u, every row at once
    v::Vector{T}     # reduced linear term, rebuilt at the start of every pass
    xbuf::Vector{T}  # primal iterate, and the vector `run_daqp!` returns
    xold::Vector{T}  # proximal centre of the previous pass
    F::GramLDL{T}
end

function LDPWorkspace(Mt::Matrix{T}, iseq::AbstractVector{Bool}) where {T <: Real}
    n, m = size(Mt)
    kmax = min(m, n) + 1
    return LDPWorkspace{T}(
        Mt, zeros(T, m), zeros(T, m), convert(Vector{Bool}, iseq), zeros(Int8, m),
        zeros(Int, kmax), zeros(Int, m), zeros(T, kmax), zeros(T, kmax), zeros(T, kmax),
        zeros(T, n), zeros(T, kmax), zeros(T, m),
        Matrix{T}(undef, n, kmax >= PACKED_KMIN ? kmax : 0), zeros(T, m),
        zeros(T, n), zeros(T, n), zeros(T, n), GramLDL{T}(kmax)
    )
end

"""
Smallest working-set capacity for which the packed block is worth keeping.

Gathering the active rows turns the loop's two products against the working set into single
matrix-vector products, which from about eight active rows up runs two to three times faster
than one short call per row. Below that a matrix-vector call costs more than the products it
performs, and the copying that keeps the block current is not repaid. A solve holds about
half its capacity in the working set on average, so the capacity has to be twice the point
where the products themselves break even.
"""
const PACKED_KMIN = 16

"Row `r` of the constraint matrix, contiguous because the matrix is stored transposed."
@inline row(ws::LDPWorkspace, r::Integer) = view(ws.Mt, :, r)

"Whether this workspace keeps the active rows gathered; see [`PACKED_KMIN`](@ref)."
@inline ispacked(ws::LDPWorkspace) = !isempty(ws.Ma)

"The working set, as the live prefix of the preallocated buffer."
@inline activeset(ws::LDPWorkspace) = view(ws.active, 1:ws.F.k)

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

"Put row `r` into the working set at `side`, extending the factorization."
function activate!(ws::LDPWorkspace{T}, r::Integer, side::Int8) where {T}
    k = ws.F.k
    m_r = row(ws, r)
    g = view(ws.g, 1:k)
    if ispacked(ws)
        mul!(g, transpose(view(ws.Ma, :, 1:k)), m_r)
    else
        @inbounds for j in 1:k
            g[j] = dot(row(ws, ws.active[j]), m_r)
        end
    end
    add_row!(ws.F, g, dot(m_r, m_r))
    # The entering row joins the packed block at the position it takes in the working set.
    # An explicit loop rather than `copyto!`: between two views of a matrix that checks
    # whether they alias and copies the source if it cannot tell, which is an allocation site
    # the hot-path guarantee sees whether or not the branch can be reached.
    if ispacked(ws)
        dest = view(ws.Ma, :, k + 1)
        @simd for i in paired(dest, m_r)
            dest[i] = m_r[i]
        end
    end
    ws.active[k + 1] = r
    ws.mu[k + 1] = zero(T)
    ws.side[r] = side
    ws.slot[r] = k + 1
    return ws
end

"Take the `i`-th working-set row back out."
function deactivate!(ws::LDPWorkspace, i::Integer)
    k = ws.F.k
    r = ws.active[i]
    remove_row!(ws.F, i)
    # Close the gap in the packed block. Columns `i+1:k` sit in one contiguous run of
    # storage, so the whole tail moves as a single block rather than column by column.
    if ispacked(ws) && i < k
        n = size(ws.Ma, 1)
        copyto!(ws.Ma, (i - 1) * n + 1, ws.Ma, i * n + 1, (k - i) * n)
    end
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
    step_and_drop!(ws, p, zero_tol) -> Bool

Move the multipliers along `p` until the first one reaches zero from its feasible side, then
drop that row. `false` when nothing blocks, which on the singular branch means the problem
is infeasible.

An equality row never blocks: its multiplier is free in sign.
"""
function step_and_drop!(ws::LDPWorkspace{T}, p::AbstractVector{T}, zero_tol::T) where {T}
    alpha = typemax(T)
    blocking = 0
    for i in 1:ws.F.k
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
    iszero(blocking) && return false
    for i in 1:ws.F.k
        ws.mu[i] += alpha * p[i]
    end
    deactivate!(ws, blocking)
    return true
end

"Index of the first zero pivot in the working set's factorization, or 0."
@inline function first_singular_pivot(F::GramLDL{T}, zero_tol::T) where {T}
    # `@inbounds` throughout the steps below: they index `1:F.k`, and the factorization's
    # buffers are allocated to the largest working set the problem admits, so `F.k` never
    # reaches their end. `eachindex` does not stand in for it here -- measured on the
    # multiplier step, it is slower than the bare loop, because the views carry their checks
    # into the loop body and block the vectorization the annotation buys.
    @inbounds for i in 1:F.k
        F.D[i] <= zero_tol && return i
    end
    return 0
end

"""
    singular_step!(ws, singular, zero_tol) -> Bool

Walk the null direction of a singular working set and drop the first row that blocks.
`false` when nothing blocks, which is what makes the problem infeasible.
"""
function singular_step!(ws::LDPWorkspace{T}, singular::Int, zero_tol::T) where {T}
    p = view(ws.p, 1:ws.F.k)
    singular_direction!(p, ws.F, singular)
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
    k = ws.F.k
    mus = view(ws.mu_star, 1:k)
    @inbounds for i in 1:k
        mus[i] = target(ws, ws.active[i])
    end
    solve_gram!(ws.F, mus)
    @inbounds for i in 1:k
        r = ws.active[i]
        !ws.iseq[r] && musign(ws, r) * mus[i] < 0 && return false
    end
    return true
end

"""
    step_toward_multipliers!(ws, zero_tol) -> Bool

Step from `ws.mu` toward the multipliers just solved for, dropping the first row whose own
reaches zero. `false` when nothing blocks the step.
"""
function step_toward_multipliers!(ws::LDPWorkspace{T}, zero_tol::T) where {T}
    k = ws.F.k
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
    k = ws.F.k
    mu, mus = ws.mu, ws.mu_star
    # An explicit loop rather than `copyto!`: between two views of a vector that checks
    # whether they alias and copies the source if it cannot tell, which is an allocation
    # site the hot-path guarantee sees whether or not the branch can be reached.
    @inbounds @simd for i in 1:k
        mu[i] = mus[i]
    end
    if ispacked(ws)
        mul!(ws.u, view(ws.Ma, :, 1:k), view(ws.mu, 1:k))
    else
        uu = ws.u
        fill!(uu, zero(T))
        @inbounds for i in 1:k
            mui = mu[i]
            col = row(ws, ws.active[i])
            @simd for j in paired(uu, col)
                uu[j] += mui * col[j]
            end
        end
    end
    return ws
end

"""
    entering_row(ws, primal_tol, bland) -> (row, side)

The inactive row that violates its bound by the most, and the side it violates, or
`(0, SIDE_UPPER)` when none does, which is what ends the run.

Dantzig's rule takes the worst violation, which is the fast choice but can cycle: a row that
keeps swapping sides re-enters forever. `bland` switches to the lowest violated index,
Bland's rule, which terminates finitely at the cost of taking more steps.
"""
function entering_row(ws::LDPWorkspace{T}, primal_tol::T, bland::Bool) where {T}
    m = size(ws.Mt, 2)
    worst = primal_tol
    entering = 0
    entering_side = SIDE_UPPER
    # Not `@simd`: this is an argmax search, and the `slot` test skips the working set.
    @inbounds for r in 1:m
        iszero(ws.slot[r]) || continue
        rr = ws.price[r]
        over = rr - ws.hi[r]
        under = ws.lo[r] - rr
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
    bland_after = 4 * (size(ws.Mt, 2) + 1)
    for iter in 1:max_iter
        # Every step reads `ws.F.k` for itself: dropping and adding a row both change it, so
        # the working set's size does not survive an iteration.
        singular = first_singular_pivot(ws.F, zero_tol)
        if !iszero(singular)
            singular_step!(ws, singular, zero_tol) || return (LDP_INFEASIBLE, iter)
            continue
        end
        if !working_set_multipliers!(ws)
            step_toward_multipliers!(ws, zero_tol) || return (LDP_CYCLED, iter)
            continue
        end
        primal_point!(ws)

        # Price every row with one matrix-vector product rather than a dot product per
        # inactive row. The working set is priced too and its values ignored, which is `k`
        # wasted products out of `m` — cheaper than it sounds, because a dot product per row
        # is `m` BLAS calls per iteration, and on a short row a call costs about what the
        # arithmetic does. One call for all of them is what makes the small sizes competitive.
        mul!(ws.price, transpose(ws.Mt), ws.u)

        entering, entering_side = entering_row(ws, primal_tol, iter > bland_after)
        iszero(entering) && return (LDP_OPTIMAL, iter)
        activate!(ws, entering, entering_side)
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
struct DAQPReduction{T <: Real, F <: Cholesky, FT}
    R::F
    # `transpose(R.U)`, kept rather than formed per pass. It solves against the stored
    # triangle without copying it, but the wrapper itself is a heap allocation on Julia 1.13,
    # once per solve in a loop that has none otherwise.
    Rt::FT
    bu::Vector{T}     # the caller's bounds, divided by the row norms
    bl::Vector{T}
    eps_prox::T
    ws::LDPWorkspace{T}
    scale::Vector{T}  # the row norm each row was divided by
end

"""
    reduce_qp(H, A, bupper, blower, iseq; eps_prox) -> DAQPReduction

Factor `H + εI` and transform the constraints into `lo ≤ Mu ≤ hi` with `M = A R⁻¹`.

With `eps_prox = 0` this needs `H ≻ 0`. Any `ε > 0` makes `H + εI` positive definite for
`H ⪰ 0`, which is what lets the proximal-point loop take a singular `H`, an LP being the
extreme case.
"""
function reduce_qp(
        H::AbstractMatrix{T}, A::AbstractMatrix{T},
        bupper::AbstractVector{T}, blower::AbstractVector{T},
        iseq::AbstractVector{Bool}; eps_prox::T = zero(T)
    ) where {T <: Real}
    n = size(H, 1)
    m = size(A, 1)
    # `H = μI` has the Cholesky factor `√μ I`, so the reduction `A R⁻¹` is a scaling rather
    # than a triangular solve. Worth testing for because it is the shape of every problem
    # whose objective is a plain squared norm. The test leaves the general case alone: it
    # stops at the first entry that disqualifies `H`, which for a dense `H` is the first
    # off-diagonal one.
    mu = scalar_diagonal(H)
    if !isnothing(mu)
        s = sqrt(mu + eps_prox)
        s > 0 || throw(
            ArgumentError(
                "P is not positive definite, which ActiveSet() needs when eps_prox = 0, " *
                    "because the reduction factors it. Pass eps_prox > 0 to run proximal-point " *
                    "iterations instead, which accept a positive semidefinite P."
            )
        )
        Mr = Matrix{T}(undef, m, n)
        copyto!(Mr, A)
        rmul!(Mr, one(T) / s)
        R = Cholesky(Matrix{T}(s * I, n, n), 'U', 0)
        return finish_reduction(R, Mr, bupper, blower, iseq, eps_prox, n, m)
    end
    # Symmetrize into one buffer. `(H + H') / 2` reads pleasantly and allocates four
    # matrices to produce one, which is most of what a small solve costs.
    Hs = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        Hs[i, j] = (H[i, j] + H[j, i]) / 2
    end
    if !iszero(eps_prox)
        for i in 1:n
            Hs[i, i] += eps_prox
        end
    end
    # `check = false` so an indefinite `H` is a value to test rather than an exception to
    # catch, and so this one factorization also answers the convexity question: factoring it
    # twice, once to check and once to use, is most of what setup costs. `cholesky!` works in
    # the buffer just filled, which is not needed afterwards.
    R = cholesky!(Symmetric(Hs), NoPivot(); check = false)
    issuccess(R) || throw(
        ArgumentError(
            iszero(eps_prox) ?
                "P is not positive definite, which ActiveSet() needs when eps_prox = 0, " *
                "because the reduction factors it. Pass eps_prox > 0 to run proximal-point " *
                "iterations instead, which accept a positive semidefinite P." :
                "P + eps_prox*I is not positive definite, so P is not positive semidefinite " *
                "and the problem is not convex."
        )
    )
    # `M = A R⁻¹`, solved with the triangle on the right of the constraint matrix. Each step
    # of that substitution scales and subtracts whole columns of `M`, which are contiguous
    # and carry no dependence within a column; solving `R⁻ᵀ Aᵀ` instead makes every entry a
    # short dot product against the entries above it, and runs well under half the speed.
    Mr = Matrix{T}(undef, m, n)
    copyto!(Mr, A)
    rdiv!(Mr, R.U)
    return finish_reduction(R, Mr, bupper, blower, iseq, eps_prox, n, m)
end

"""
    scalar_diagonal(H) -> μ or nothing

`μ` when `H` is `μI`, and `nothing` otherwise.

Stops at the first entry that rules it out, so a dense `H` costs one comparison rather than a
pass over the matrix.
"""
function scalar_diagonal(H::AbstractMatrix{T}) where {T}
    n = size(H, 1)
    n == size(H, 2) || return nothing
    mu = H[1, 1]
    for j in 1:n, i in 1:n
        if i == j
            H[i, j] == mu || return nothing
        else
            iszero(H[i, j]) || return nothing
        end
    end
    return mu
end

"""
Normalize the rows of `M = A R⁻¹` and pack the reduction around them.

Shared by the two ways of forming `M`, because normalization is what makes `primal_tol` mean
the same thing on every row however that row happened to be scaled.
"""
function finish_reduction(
        R, Mr::Matrix{T}, bupper::AbstractVector{T}, blower::AbstractVector{T},
        iseq::AbstractVector{Bool}, eps_prox::T, n::Int, m::Int
    ) where {T}
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
        s = nrm > 0 ? nrm : one(T)
        scale[j] = s
        @simd ivdep for i in 1:n
            Mt[i, j] = Mr[j, i] / s
        end
        bu[j] = bupper[j] / s
        bl[j] = blower[j] / s
    end

    ws = LDPWorkspace(Mt, iseq)
    Rt = transpose(R.U)
    return DAQPReduction{T, typeof(R), typeof(Rt)}(R, Rt, bu, bl, eps_prox, ws, scale)
end

"""
    set_targets!(red, v) -> red

Rebuild the least-distance bounds for the current `v`.

With `x = R⁻¹(−u − v)`, the caller's `bl ≤ Ax ≤ bu` becomes
`−(bu + Mv) ≤ Mu ≤ −(bl + Mv)`: the two bounds swap, because `Ax` runs against `Mu`.
"""
function set_targets!(red::DAQPReduction{T}, v::AbstractVector{T}) where {T}
    ws = red.ws
    mul!(ws.Mv, transpose(ws.Mt), v)
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
    for j in eachindex(red.bu)
        red.bu[j] = bupper[j] / red.scale[j]
        red.bl[j] = blower[j] / red.scale[j]
    end
    return red
end

"""
    reset_working_set!(red) -> red

Empty the working set apart from the equality rows, which must always be in it. The next
solve then starts where a fresh setup would.
"""
function reset_working_set!(red::DAQPReduction)
    ws = red.ws
    while ws.F.k > 0
        deactivate!(ws, ws.F.k)
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
    # `R.L` on an upper-stored Cholesky materializes the transpose, copying the whole factor
    # on every pass; `Rt` solves against the same triangle without copying it.
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
    # `transpose(red.Rt)` is `R.U` recovered from the handle the reduction already holds;
    # `red.R.U` re-derives it, and on a lower-stored factorization that branch copies the
    # whole factor.
    ldiv!(transpose(red.Rt), x)
    return x
end
