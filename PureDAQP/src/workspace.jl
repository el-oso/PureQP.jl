"""
    ActiveSetWorkspace

The state a dual active-set solve runs on: the [`QPData`](@ref), the resolved
[`ActiveSet`](@ref), the [`Options`](@ref), the reduction to a least-distance problem, and
the iterates in problem space.

The reduction holds the Cholesky factor of `P` (or `P + εI`), in `P`'s own form, and the
transformed constraint matrix `A R⁻¹`, both built once at [`setup`](@ref). When `A` and the
factor are not both dense it holds `A` and the factor instead, and derives rows of `A R⁻¹` from
them. A re-solve through [`update!`](@ref) keeps them whenever `P` and `A` are unchanged, and
keeps the working set too, which is what makes a warm start cheap here.
"""
mutable struct ActiveSetWorkspace{
        T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T},
        RD <: DAQPReduction{T}, ALG <: ActiveSet{T, T, T, T, T},
    } <: QPWorkspace{T}
    # Not `const`: `update!` replaces `P` or `A` by handing back another `QPData` around the
    # same vectors, which is what an immutable problem costs and all it costs.
    prob::QPData{T, MP, MA, V}
    # Concretely typed for the same reason `red` is: `ActiveSet{T, T, T, T, T}` leaves the
    # working-set parameter free, and a field of that type is abstract.
    algorithm::ALG
    options::Options{T}
    # Concretely typed: `DAQPReduction{T}` alone leaves the factorization parameter abstract,
    # which costs a dynamic dispatch on every solve. Rebound by `update!` when `P` or `A`
    # changes, which is the one thing that forces a fresh reduction.
    red::RD
    const x::V
    const y::V
    const z::V
    status::Status
    const polished::Bool
    const status_polish::PolishStatus
    iter::Int
    warm::Bool          # carry the working set into the next solve
    const setup_time::Float64
    update_time::Float64
    solve_time::Float64
    # `Px`, which the objective, the dual residual and the gap all share. Owned here: the
    # method has no linear-system backend, so there is no problem scratch to borrow.
    const px::V
    # One entry per constraint row, where a candidate infeasibility certificate is written
    # back into the caller's own indexing before it is checked.
    const ycert::V
    # The 2-norm of each row of `A`, which sets the rounding of the product `Ax`. Read once when
    # the problem is set up and again when `A` is replaced.
    const row_norm::V
    # The dual infeasibility certificate, a direction in the caller's variables, and the empty
    # vector `Solution.dual_inf_cert` holds when there is none. Both are allocated once so that
    # handing back either is a rebinding.
    const cert_dual::V
    const cert_none::V
    # Refilled and handed back by every solve, so a solve allocates nothing at all. Its `x`
    # and `y` are this workspace's own arrays. `Solution` says what that means for a caller
    # holding one across a solve.
    const sol::Solution{T}
end

"""
The workspace `solve(P, q, A, l, u, ActiveSet())` builds for dense data.

The guarantees on the entry points are stated against this instantiation: a `where` clause
binds its type variables to the method rather than the module, so a parametric declaration
has no concrete signature to check and needs the instantiations named. Every parameter is
spelled out for that reason — `DenseWorkspace{Float64}` must be a concrete type, or the
check that names it is skipped instead of run.
"""
const DenseWorkspace{T} = ActiveSetWorkspace{
    T, Matrix{T}, Matrix{T}, Vector{T},
    DAQPReduction{
        T, UpperTriangular{T, Matrix{T}},
        LowerTriangular{T, Transpose{T, Matrix{T}}}, WorkingSetQR{T}, DenseRows{T},
    },
    ActiveSet{T, T, T, T, T, WorkingSetQR},
}

"""
The workspace `solve(P, q, A, l, u, ActiveSet())` builds for a `KroneckerOperator` `P` and a
`KroneckerOperator` `A`, which reduces to [`ImplicitRows`](@ref) over a
[`PureQPBase.KroneckerCholesky`](@ref).

The guarantees are stated against this instantiation alongside [`DenseWorkspace`](@ref),
because holding the operands rather than `A R⁻¹` puts different code in the iteration.
"""
const KroneckerWorkspace{T} = ActiveSetWorkspace{
    T, KroneckerOperator{T, Matrix{T}}, KroneckerOperator{T, Matrix{T}}, Vector{T},
    DAQPReduction{
        T, KroneckerCholesky{T}, Transpose{T, KroneckerCholesky{T}}, WorkingSetQR{T},
        ImplicitRows{
            T, KroneckerOperator{T, Matrix{T}}, KroneckerCholesky{T},
            Transpose{T, KroneckerCholesky{T}},
        },
    },
    ActiveSet{T, T, T, T, T, WorkingSetQR},
}

function Base.show(io::IO, ws::ActiveSetWorkspace{T}) where {T}
    n, m = dimensions(ws)
    print(io, "ActiveSetWorkspace{", T, "}: ", n, " variables, ", m, " rows, ")
    print(io, nactive(ws.red.ws.W), " rows in the working set")
    return nothing
end

"""
Stands in for the linear-system backend this method does not have.

`validate_update!` asks the backend to invalidate whatever it caches about `P` and `A`.
There is nothing to invalidate here, because [`update!`](@ref) rebuilds the whole reduction
when either changes. Passing a type of this package's own keeps that a method we are
entitled to define, rather than one attached to `Nothing`.
"""
struct NoBackend end

check_update(::NoBackend, P, A) = nothing

"Refuse what the reduction cannot represent, naming the way out."
function refuse_activeset(LS::Symbol, options::Options)
    LS in (:auto, :dense) || throw(
        ArgumentError(
            "linsys = :$LS is not available with ActiveSet(): the method reduces the problem " *
                "to a least-distance problem and maintains its own factorization, so it has no " *
                "backend to choose. Pass linsys = :auto."
        )
    )
    iszero(options.scaling) || throw(
        ArgumentError(
            "ActiveSet() needs scaling = 0: the reduction normalizes the rows of the " *
                "transformed constraint matrix itself, and equilibration on top of that would " *
                "rescale the rows the working set is priced against."
        )
    )
    options.polishing && throw(
        ArgumentError(
            "polishing = true has no effect with ActiveSet(): the method already ends on an " *
                "exact solution of the equality-constrained QP over its working set, which is " *
                "what polishing computes. Pass polishing = false."
        )
    )
    return nothing
end

"""
The representations the reduction reads as they are: a dense matrix, a `Symmetric` of one, and
the structured and unmaterialized forms the base owns.

A composition belongs here for the same reason a `ProductOperator` does: a working-set change
needs one row, and each of these answers that from its parts — a stack from the block holding the
row, a composition by carrying the outer part's row through the inner one — without entries.
"""
const ReadDirectly{T} = Union{
    StridedMatrix{T}, Symmetric{T, <:StridedMatrix{T}}, Diagonal{T}, BlockDiagonal{T},
    KroneckerOperator{T}, ProductOperator{T},
    StackedOperator{T}, ComposedOperator{T}, SumOperator{T},
}

"""
    factored_operand(T, P) -> P or Matrix{T}

What the reduction hands to `cholesky_factor`.

A representation the reduction reads directly is passed through whether or not it has a factor,
so a `P` with no factor is refused by `reduce_qp` naming the forms that have one rather than
here. Everything else is handed to [`PureQPBase.factorable_operand`](@ref), which keeps what the
base factors in its own form — a sparse `P` among them — and densifies the rest. A sparse `A` is
not kept; see [`row_operand`](@ref).
"""
factored_operand(::Type{T}, P::ReadDirectly{T}) where {T} = P

# A stack is read directly as an `A`, where the reduction wants its rows. As a `P` it has no
# Cholesky factor of its own, and its blocks are slices of the rows, which a dense `n×n` factor
# does not lose — the same reason a join is densified. So a square stack is factored rather than
# refused for want of a factor.
function factored_operand(::Type{T}, P::StackedOperator{T}) where {T}
    is_materializable(P) || refuse_unreadable_operand(T)
    return factorable_operand(T, P)
end

function factored_operand(::Type{T}, P::AbstractMatrix) where {T}
    is_materializable(P) || refuse_unreadable_operand(T)
    return factorable_operand(T, P)
end

"""
    row_operand(T, A) -> A or Matrix{T}

`A` itself when the reduction reads its rows and products directly, and a dense copy otherwise.

The base decides what a row-reading consumer should hold, so a sparse `A` is kept as a sparse
pair with its transpose rather than densified; this adds only the refusal.
"""
row_operand(::Type{T}, A::ReadDirectly{T}) where {T} = A

# A join answers a row from its blocks and a product through them, so it is read directly as an
# `A`. It is not in `ReadDirectly`, because as a `P` it has no Cholesky factor of its own and is
# densified by [`factored_operand`](@ref) instead — a join holds no structure a dense `n×n`
# factor would lose.
row_operand(::Type{T}, A::JoinedOperator{T}) where {T} = A

function row_operand(::Type{T}, A::AbstractMatrix) where {T}
    is_materializable(A) || refuse_unreadable_operand(T)
    return rows_operand(T, A)
end

# Two conditions reach this, and the message names the one state both leave the operand in
# rather than guessing which: an operand with no readable entries, where the reduction needs
# either entries or a products-only operator of its own element type.
@noinline refuse_unreadable_operand(::Type{T}) where {T} = throw(
    ArgumentError(
        "the reduction cannot read this operand. It takes entries, or an operator that " *
            "supplies products only when that operator carries the solve's own element type: " *
            "build it as a ProductOperator{$T}, pass P, q, A, l and u in one element type, or " *
            "make every block of a composition readable."
    )
)

function setup_backend(
        alg::ActiveSet, ::Val{LS}, ::Type{T}, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, options::Options,
        preconditioner, accelerator
    ) where {LS, T <: Real}
    t0 = time_ns()
    isnothing(accelerator) || throw(
        ArgumentError("accelerator is used only by OperatorSplitting: this method has no fixed-point iteration to accelerate.")
    )
    isnothing(preconditioner) || throw(
        ArgumentError("preconditioner is used only by the matrix-free backend, which ActiveSet() does not have.")
    )
    refuse_activeset(LS, options)

    n, m = validate(P, q, A, l, u)
    resolved = element_typed(alg, T, options)
    # Convexity is not checked here: `reduce_qp` factors `P + eps_prox*I` and reports a
    # failure, which is the same question asked once instead of twice.
    # The caller's data and nothing else: this method never forms the scaled products and has
    # no linear-system backend, so it asks for neither the equilibrated copy nor their scratch.
    # `refuse_activeset` has already required `options.scaling` to be zero.
    prob = validated_data(T, n, m, P, q, A, l, u)
    # `convert` rather than `Vector{T}`: that copies even when the argument already has the
    # type asked for, and the reduction only reads this data.
    iseq = [is_equality(prob.l0[i], prob.u0[i]) for i in 1:m]
    # The working set comes from the algorithm's type, not its `working_set` field: that leaves
    # one type for the reduction to settle rather than two, which is what keeps this
    # construction statically resolvable for a `P` whose factor has two forms.
    red = reduce_qp(
        factored_operand(T, P), row_operand(T, A),
        convert(Vector{T}, prob.u0), convert(Vector{T}, prob.l0),
        iseq, working_set_param(resolved); eps_prox = resolved.eps_prox
    )

    # The reported point is these arrays, not copies of them, so the solution the workspace
    # hands back is built here and refilled rather than rebuilt.
    x, y, z = zeros(T, n), zeros(T, m), zeros(T, m)
    ws = ActiveSetWorkspace{
        T, typeof(prob.P), typeof(prob.A), typeof(prob.q0), typeof(red), typeof(resolved),
    }(
        prob, resolved, options, red,
        x, y, z,
        UNSOLVED, false, POLISH_NOT_PERFORMED, 0, false,
        (time_ns() - t0) / 1.0e9, 0.0, 0.0,
        zeros(T, n), zeros(T, m), zeros(T, m),
        zeros(T, n), T[],
        # The Farkas vector of a primal infeasibility proof is checked in `ycert` and not
        # reported, so `prim_inf_cert` stays empty. A dual certificate is a direction in the
        # caller's variables, reported from `cert_dual`.
        empty_solution(x, y, T[], T[]),
    )
    row_norms!(ws.row_norm, prob.A, ws.px)
    return ws
end

"""
    row_norms!(dest, A, buf) -> dest

The 2-norm of each row of `A`, read one row at a time through `buf`, a vector of length `n`.
"""
function row_norms!(dest::AbstractVector, A, buf::AbstractVector)
    for i in eachindex(dest)
        dense_row!(buf, A, i)
        dest[i] = norm(buf)
    end
    return dest
end

"""
    solve!(ws) -> Solution

Run the dual active-set method from the workspace's state.

The working set carries over from the previous solve when one has run and nothing has
invalidated it, so a re-solve after [`update!`](@ref) starts from the previous answer's
active rows. [`cold_start!`](@ref) drops it.

How much that saves depends on how far the data moved, because the carried set is a good guess
exactly to the extent the active set did not change. Measured on an ill-conditioned problem,
a re-solve after a 1% change in `q` costs a fiftieth of a cold one; after a tenth it can cost
more than one, and on the hardest draws it reaches no answer at all and reports
`NUMERICAL_ERROR` or `MAX_ITER_REACHED`.

Such a run is recoverable and never a wrong answer: [`cold_start!`](@ref) and a second
`solve!` solve the same data. This is left to the caller rather than done here, because a
solve that sometimes silently runs twice is worse than one that fails predictably for a caller
working to a deadline. A control loop that has the time to spare can retry; one that does not
can take the failure and keep the previous input.

The result is the workspace's own `Solution`, refilled by each solve rather than rebuilt. To
keep one across a later solve, copy it -- [`copyto!`](@ref) into a `Solution` you already hold
allocates nothing.
"""
function solve!(ws::ActiveSetWorkspace{T}) where {T}
    t0 = time_ns()
    prob, alg = ws.prob, ws.algorithm
    ws.warm || reset_working_set!(ws.red)

    x, status, iters = run_daqp!(ws.red, prob.q0, alg, ws.options.max_iter)
    ws.iter = iters
    ws.warm = true
    return finish_solve!(ws, prob, x, status, t0)
end

"""
    certifiable(ws) -> Bool

Whether the direction the loop stopped on really proves the problem infeasible.

The loop finds a direction it cannot step along and reports infeasibility. That direction
lives in the reduced space of `M = A R⁻¹`, and the factorization it comes from is of
`Mₐ Mₐᵀ`, which carries the conditioning of `A R⁻¹` squared — so the pivot that declared it
singular can have come from rounding rather than from a dependency among the rows.

The claim is settled where it is stated: against the caller's own data. Written back to one
entry per row as `y`, the direction is a Farkas certificate when `Aᵀy = 0` and the support
function of `y` over `[l, u]` is negative. In floating point `Aᵀy` is not zero, and the
direction then proves infeasibility only within a radius of `x`; [`separates`](@ref) accepts it
when that radius is at least `infeasibility_radius`, with the rounding error of both sums
counted against it.

It runs once, on the branch that has already decided to report infeasibility, so no solve
that reaches an answer pays for it.
"""
function certifiable(ws::ActiveSetWorkspace{T}) where {T}
    lw = ws.red.ws
    prob = ws.prob
    # The ray spans the working set, except where `full_set_step!` proved infeasibility with a
    # row it could not hold; there it reaches one further, and `certrow` says so.
    k = iszero(lw.certrow[1]) ? nactive(lw.W) : lw.certrow[1]
    k > 0 || return false
    y = ws.ycert
    fill!(y, zero(T))
    # Back to the caller's rows: the reduction normalized each row of `M` by `scale`, so the
    # multiplier of row `r` carries that factor back out.
    for i in 1:k
        r = lw.active[i]
        y[r] = lw.p[i] / ws.red.scale[r]
    end
    # A certificate's sign is whichever orientation separates; the loop orients its direction
    # for its own stepping rule, not for this.
    separates(ws, y) && return true
    for i in eachindex(y)
        y[i] = -y[i]
    end
    return separates(ws, y)
end

"""
    separates(ws, y) -> Bool

Whether `y` proves the caller's own rows infeasible within `infeasibility_radius`. After the
projection onto the polar of the recession cone of `[l, u]`, with `s` the support function of
`[l, u]` at `y`:

- `Aᵀy` is at rounding level against the terms that cancel in it, `‖Aᵀy‖∞ ≤ primal_tol·max|yᵢ|‖aᵢ‖₂`;
- no `x` with `‖x‖∞ < ρ` satisfies the rows, for `ρ = −s/‖Aᵀy‖₁`, and `ρ` is at least
  `infeasibility_radius`.

Both `s` and `Aᵀy` are computed in floating point, so the second condition is evaluated with each
side moved against the claim by the rounding error of its sum, `(m + 2)·eps` times the sum of the
absolute terms. `y` is overwritten by the projection.
"""
function separates(ws::ActiveSetWorkspace{T}, y) where {T}
    prob = ws.prob
    ny = norm_inf(y)
    ny > DIVISION_TOL(T) || return false
    # Outside the polar of the recession cone a direction cannot separate at all, and the
    # support function below would read a bound the row does not have. The projection reads a
    # bound as absent only at the clamp, as the rest of this solver does.
    absent = prevfloat(INFTY(T))
    for i in eachindex(y)
        y[i] = polar_reccone(y[i], prob.l0[i], prob.u0[i], absent)
    end
    norm_inf(y) > DIVISION_TOL(T) || return false
    # `adjoint` rather than `transpose`: the element type is real, so the two products are the
    # same one, and it is the adjoint that an operator supplying products only answers. Asked
    # for the transpose it falls back to reading entries, which such an operator refuses.
    mul!(ws.px, adjoint(prob.A), y)
    # Only the rows of the ray are read, each once: for the rounding scale of `Aᵀy` and of the
    # support function.
    row = ws.red.ws.xold
    support = zero(T)
    support_abs = zero(T)
    row_scale = zero(T)
    row_sum = zero(T)
    for i in eachindex(y)
        yi = y[i]
        iszero(yi) && continue
        bound = yi > zero(T) ? prob.u0[i] : prob.l0[i]
        support += yi * bound
        support_abs += abs(yi * bound)
        dense_row!(row, prob.A, i)
        row_scale = max(row_scale, abs(yi) * norm(row))
        row_sum += abs(yi) * norm(row, 1)
    end
    support < zero(T) || return false
    norm_inf(ws.px) <= ws.algorithm.primal_tol * row_scale || return false
    gamma = (length(y) + 2) * eps(T)
    proved = -support - gamma * support_abs
    needed = ws.algorithm.infeasibility_radius * (norm(ws.px, 1) + gamma * row_sum)
    return proved >= needed
end

"""
    unbounded_direction(ws, x) -> Bool

Whether the last step of the proximal-point loop, `x − xold`, certifies that the problem is
unbounded below.

A step certifies when it descends, `qᵀd < 0`, leaves the cost exactly flat, `Pd = 0`, and moves
no row toward a bound that is present: `Ad` has no positive entry at a row with a finite upper
bound and no negative entry at one with a finite lower bound. The components of the step below
`sqrt(eps)/16` of its largest are first set to zero. An unbounded problem makes every pass of the
loop move the iterate along such a direction, and the components that curvature resists shrink
geometrically, so the last step is the candidate.

The conditions hold exactly or not at all. A tolerance cannot tell an unbounded problem from a
bounded one whose minimizer lies beyond the passes the loop takes, whatever its size: curvature
of `1e-9` against a tolerance of `1e-8` is the whole difference. A step that satisfies the
conditions only to a tolerance leaves the status at `MAX_ITER_REACHED`, with the point reached.

Only the proximal-point loop produces one: with `eps_prox = 0` the problem is bounded below, as
the method then needs `P ≻ 0`. The step is written over `xold`, and `Pd` and `Ad` into `px` and
`z`, which a run that ended on its iteration limit has no other use for.
"""
function unbounded_direction(ws::ActiveSetWorkspace{T}, x::AbstractVector{T}) where {T}
    iszero(ws.algorithm.eps_prox) && return false
    prob, d = ws.prob, ws.red.ws.xold
    for j in eachindex(d)
        d[j] = x[j] - d[j]
    end
    nd = norm_inf(d)
    nd > DIVISION_TOL(T) || return false
    # A component that the curvature still resists is a fraction of the step that shrinks each
    # pass, and below `sqrt(eps)/16` of it, `6e-10` in `Float64`, it is rounding. It is set to
    # zero, and the direction that remains is what the conditions below test.
    snap = sqrt(eps(T)) / 16 * nd
    for j in eachindex(d)
        abs(d[j]) < snap && (d[j] = zero(T))
    end
    dot(prob.q0, d) < zero(T) || return false
    mul!(ws.px, prob.P, d)
    iszero(norm_inf(ws.px)) || return false
    mul!(ws.z, prob.A, d)
    for i in eachindex(ws.z)
        zi = ws.z[i]
        (zi > zero(T) && is_upper_bound(prob.u0[i])) && return false
        (zi < zero(T) && is_lower_bound(prob.l0[i])) && return false
    end
    unit_certificate!(ws.cert_dual, ws.px, d, d, false)
    return true
end

"""
    violates_rows(ws) -> Bool

Whether a row of `ws.z = Ax` leaves a bound that is present by more than
`primal_tol·(max(1, ‖aᵢ‖₂‖x‖∞) + |zᵢ|)`.

The product `Ax` is formed with an error of about `eps‖aᵢ‖‖x‖`, so a row of large norm at a
point of size one is held to the rounding of its own product and not to `primal_tol`. The norm
is that of the caller's row, so scaling the cost does not move it. The floor of one is that of pricing's absolute tolerance, and `|zᵢ|` lets a
row of value `1e8` be off by a part in `1e8`. A bound that is absent is not a bound, whatever
the row's value. The comparison is a product rather than a quotient and the loop ends at the
first row that fails: it runs on every solve, and a row that violates is the exception.
"""
function violates_rows(ws::ActiveSetWorkspace{T}) where {T}
    prob = ws.prob
    tol = ws.algorithm.primal_tol
    row_norm = ws.row_norm
    xn = max(one(T), norm_inf(ws.x))
    for i in eachindex(ws.z)
        zi = ws.z[i]
        slack = tol * (max(one(T), row_norm[i] * xn) + abs(zi))
        (zi > prob.u0[i] && is_upper_bound(prob.u0[i]) && zi - prob.u0[i] > slack) && return true
        (zi < prob.l0[i] && is_lower_bound(prob.l0[i]) && prob.l0[i] - zi > slack) && return true
    end
    return false
end

"Record the outcome of a pass, whatever it was, and build the result."
function finish_solve!(ws::ActiveSetWorkspace{T}, prob, x, status, t0) where {T}
    if status == LDP_OPTIMAL
        copyto!(ws.x, x)
        multipliers!(ws.y, ws.red)
        mul!(ws.z, prob.A, ws.x)
        # The refinement reads `ws.z` for the active rows' residual, and moves the point it
        # was computed from, so `z` is formed again from the corrected `x`.
        refine_primal!(ws) && mul!(ws.z, prob.A, ws.x)
        if !violates_rows(ws)
            ws.status = SOLVED
        else
            # The loop stopped because no row prices in, and a row is violated all the same: the
            # multipliers solved a system whose pivots were rounding, so the point is not the
            # working set's. A point that violates the rows is not reported as an answer.
            fill!(ws.x, T(NaN))
            fill!(ws.y, T(NaN))
            fill!(ws.z, T(NaN))
            reset_working_set!(ws.red)
            ws.warm = false
            ws.status = NUMERICAL_ERROR
        end
    elseif status == LDP_INFEASIBLE
        fill!(ws.x, T(NaN))
        fill!(ws.y, T(NaN))
        fill!(ws.z, T(NaN))
        # The loop proves infeasibility by finding a direction the working set cannot move
        # along. That proof is only worth as much as the factorization it came from, and the
        # factorization is of `Mₐ Mₐᵀ`, whose conditioning is that of `A R⁻¹` squared. The
        # claim is checked against the caller's own rows before it is made.
        if certifiable(ws)
            ws.status = PRIMAL_INFEASIBLE
        else
            ws.status = NUMERICAL_ERROR
        end
        # The working set the loop stopped on holds the dependent rows of the failed pass. A
        # later solve with other bounds, from `update!` or `warm_start!`, would read that
        # dependency again, so the set is emptied to the equality rows.
        reset_working_set!(ws.red)
        ws.warm = false
    elseif status == LDP_ITERATION_LIMIT && unbounded_direction(ws, x)
        # Passes that keep moving the iterate along a descent direction of the cost that the
        # constraints allow are an unbounded problem, which the dual active-set method has no
        # other way to see: its dual is infeasible, and nothing in the working set says so.
        fill!(ws.x, T(NaN))
        fill!(ws.y, T(NaN))
        fill!(ws.z, T(NaN))
        ws.status = DUAL_INFEASIBLE
    elseif status == LDP_ITERATION_LIMIT
        # `MAX_ITER_REACHED` reports a point, so the iterate the loop stopped at is adopted
        # here. Left alone, the workspace keeps whatever it held before -- the zeros `setup`
        # filled, or the answer of the previous solve, either of which reads as this solve's.
        copyto!(ws.x, x)
        multipliers!(ws.y, ws.red)
        mul!(ws.z, prob.A, ws.x)
        ws.status = MAX_ITER_REACHED
    else
        # The pass cycled. The iterate it reached is not a point worth reading: it is
        # wherever the loop was when it stopped making progress.
        fill!(ws.x, T(NaN))
        fill!(ws.y, T(NaN))
        fill!(ws.z, T(NaN))
        ws.status = NUMERICAL_ERROR
        # The working set it cycled on would cycle again if carried into the next solve, so one
        # bad pass would end every pass after it; the set is emptied to the equality rows.
        reset_working_set!(ws.red)
        ws.warm = false
    end
    ws.solve_time = (time_ns() - t0) / 1.0e9
    return build_solution(ws)
end

@strict_function signatures = [(DenseWorkspace{Float64},), (KroneckerWorkspace{Float64},)] function warm_start!(ws::ActiveSetWorkspace{T}; x = nothing, y = nothing) where {T}
    prob = ws.prob
    if !isnothing(x)
        length(x) == prob.n || throw(ArgumentError(lazy"length(x) must be $(prob.n)"))
        all(isfinite, x) || throw(ArgumentError("x must be finite, found NaN or Inf"))
        ws.x .= T.(x)
    end
    if !isnothing(y)
        length(y) == prob.m || throw(ArgumentError(lazy"length(y) must be $(prob.m)"))
        all(isfinite, y) || throw(ArgumentError("y must be finite, found NaN or Inf"))
        ws.y .= T.(y)
    end
    # The working set, not the point, is what this method restarts from, and the caller has
    # given a point. Keep whatever working set is there: it is the best guess available.
    ws.warm = true
    return ws
end

@strict_function signatures = [(DenseWorkspace{Float64},), (KroneckerWorkspace{Float64},)] function cold_start!(ws::ActiveSetWorkspace{T}) where {T}
    fill!(ws.x, zero(T))
    fill!(ws.y, zero(T))
    fill!(ws.z, zero(T))
    ws.warm = false
    ws.status = UNSOLVED
    return ws
end

"""
    update!(ws; q, l, u, P, A) -> ws

Replace problem data. Changing `q`, `l` or `u` keeps the Cholesky factor and the transformed
constraint matrix, so only the right-hand side is rebuilt. Changing `P` or `A` rebuilds the
reduction, which is the expensive path.
"""
function update!(
        ws::ActiveSetWorkspace{T}; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing
    ) where {T}
    t0 = time_ns()
    validate_update!(ws.prob, NoBackend(); P, A, q, l, u)
    ws.prob = adopt_update!(ws.prob; P, A, q, l, u)
    # Bound once. The comprehension below closes over this name, and a captured variable
    # that is also reassigned is boxed -- an allocation on every call, including the ones
    # that never reach the branch holding the comprehension.
    data = ws.prob
    if !isnothing(P) || !isnothing(A)
        iseq = [is_equality(data.l0[i], data.u0[i]) for i in 1:data.m]
        red = reduce_qp(
            factored_operand(T, data.P), row_operand(T, data.A),
            convert(Vector{T}, data.u0), convert(Vector{T}, data.l0),
            iseq, working_set_param(ws.algorithm); eps_prox = ws.algorithm.eps_prox
        )
        # `validate_update!` has already required `P` and `A` to keep their types, which fixes
        # the reduction's type, so this assignment cannot change what the workspace holds.
        ws.red = red
        row_norms!(ws.row_norm, data.A, ws.px)
        ws.warm = false
    elseif !isnothing(l) || !isnothing(u)
        # A row that has just become an equality, or stopped being one, is held on different
        # terms: an equality's multiplier is free in sign and never blocks a step. The
        # working set carried from the previous solve holds it on the old terms, so it is no
        # longer a starting point and the next solve builds its own.
        rebuild_bounds!(ws.red, data.u0, data.l0) && (ws.warm = false)
    end
    ws.update_time += (time_ns() - t0) / 1.0e9
    return ws
end

"""
    update_settings!(ws; kwargs...) -> ws

Merge the keywords into the workspace's options.

A method of its own because the shared one ends by handing the new options to the
linear-system backend, and this method has none to hand them to.
"""
function update_settings!(ws::ActiveSetWorkspace{T}; kwargs...) where {T}
    check_option_names(kwargs, ws.algorithm)
    old = ws.options
    new = Options{T}(; settings_tuple(old)..., kwargs...)
    refuse_activeset(new.linsys, new)
    ws.options = new
    return ws
end

function update_settings!(ws::ActiveSetWorkspace{T}, alg::ActiveSet) where {T}
    new = element_typed(alg, T, ws.options)
    new.eps_prox == ws.algorithm.eps_prox || throw(
        ArgumentError(
            "eps_prox is built into the factorization of P + eps_prox*I, so it cannot be " *
                "changed on an existing workspace. Build a new one with setup."
        )
    )
    ws.algorithm = new
    return ws
end

# An active-set solution puts every inactive row's multiplier at exactly zero, which is what
# the active-set test reads, so no polishing is needed before a derivative.
derivative_ready(::ActiveSetWorkspace) = nothing
