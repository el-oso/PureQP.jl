"""
    admm_step!(ws)

One ADMM iteration:

    (x̃, z̃) ← solve of the subproblem for  (σx − q,  z − ρ⁻¹ ⊙ y)
    x ← α x̃ + (1−α) x
    z ← Π_[l,u](α z̃ + (1−α) z + ρ⁻¹ ⊙ y)
    y ← y + ρ ⊙ (α z̃ + (1−α) z_prev − z)

`x_prev` and `z_prev` are swapped rather than copied.
"""
function admm_step!(ws::OperatorSplittingWorkspace{T}) where {T}
    prob, wt = ws.prob, ws.weights
    ws.x, ws.x_prev = ws.x_prev, ws.x
    ws.z, ws.z_prev = ws.z_prev, ws.z
    scale_subtract!(ws.rhs_x, wt.sigma, ws.x_prev, prob.q)
    subtract_scaled!(ws.rhs_z, ws.z_prev, wt.w_inv, ws.y)
    set_tolerance_level!(ws.linsys, max(ws.scaled_prim_res, ws.scaled_dual_res))
    solve_system!(ws.linsys, prob, wt, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde)
    update_x!(ws.x, ws.delta_x, ws.xtilde, ws.x_prev, ws.algorithm.alpha)
    prob.m > 0 && update_zy!(
        ws.z, ws.y, ws.delta_y, ws.ztilde, ws.z_prev,
        wt.w, wt.w_inv, prob.l, prob.u, ws.algorithm.alpha, prob.work_m
    )
    return ws
end

# The `verbose` output.
#
# Everything here writes to `Core.stdout` and formats by hand. That is not a style choice:
# `--trim` analyses this code whether or not `verbose` is ever set, and it rejects both
# Printf (its format specifications carry type parameters that do not infer) and bare
# `println(x)` (`Base.stdout` is an abstractly typed global). `Core.stdout` is a concrete
# singleton, so calls through it resolve statically; `redirect_stdout` still captures it,
# since that redirects the file descriptor.
#
# Each algorithm keeps its own copy of these two. Shared, `print_padded`'s value argument is
# inferred over both callers at once, and the `string` it reaches then takes an argument
# `--trim` cannot resolve.
const VERBOSE_RULE = "------------------------------------------------------------------"

"Right-align `s` in `width` columns."
function print_padded(s::String, width::Int)
    for _ in (ncodeunits(s) + 1):width
        print(Core.stdout, " ")
    end
    print(Core.stdout, s)
    return nothing
end

print_padded(v, width::Int, digits::Int) = print_padded(string(round(v; sigdigits = digits)), width)

function print_header(ws::OperatorSplittingWorkspace)
    println(Core.stdout, VERBOSE_RULE)
    println(Core.stdout, "            PureOSQP - operator splitting QP solver")
    print(Core.stdout, "     n = ")
    print(Core.stdout, ws.prob.n)
    print(Core.stdout, ", m = ")
    print(Core.stdout, ws.prob.m)
    print(Core.stdout, ", backend = ")
    println(Core.stdout, backend_name(ws.linsys))
    print(Core.stdout, "     eps_abs = ")
    print(Core.stdout, ws.options.eps_abs)
    print(Core.stdout, ", eps_rel = ")
    print(Core.stdout, ws.options.eps_rel)
    print(Core.stdout, ", max_iter = ")
    print(Core.stdout, ws.options.max_iter)
    print(Core.stdout, ", polishing = ")
    println(Core.stdout, ws.options.polishing ? "on" : "off")
    println(Core.stdout, VERBOSE_RULE)
    println(Core.stdout, " iter      objective      prim res      dual res           rho")
    return nothing
end

function print_row(ws::OperatorSplittingWorkspace)
    print_padded(string(ws.iter), 5)
    print_padded(ws.obj_val, 15, 6)
    print_padded(ws.prim_res, 14, 3)
    print_padded(ws.dual_res, 14, 3)
    print_padded(ws.rho, 14, 3)
    print(Core.stdout, "\n")
    return nothing
end

function print_footer(ws::OperatorSplittingWorkspace)
    println(Core.stdout, VERBOSE_RULE)
    print(Core.stdout, "status:               ")
    println(Core.stdout, status_name(ws.status))
    if ws.options.polishing
        print(Core.stdout, "polish:               ")
        println(Core.stdout, ws.polished ? "successful" : "unsuccessful")
    end
    print(Core.stdout, "number of iterations: ")
    println(Core.stdout, ws.iter)
    if has_solution(ws.status)
        print(Core.stdout, "optimal objective:    ")
        println(Core.stdout, round(ws.obj_val; sigdigits = 6))
        print(Core.stdout, "primal residual:      ")
        println(Core.stdout, round(ws.prim_res; sigdigits = 3))
        print(Core.stdout, "dual residual:        ")
        println(Core.stdout, round(ws.dual_res; sigdigits = 3))
    end
    println(Core.stdout, VERBOSE_RULE)
    return nothing
end

"""
    solve!(ws) -> Solution

Run the ADMM loop on an existing workspace. Safe to call repeatedly; with
`warm_starting = true` the previous iterates are the starting point.

If the iteration limit is reached, the termination tests are retried once at ten times the
requested tolerances before the run is declared unconverged, which is where the
`*_INACCURATE` statuses come from.

`time_limit` bounds this loop and returns `TIME_LIMIT_REACHED`. It measures the loop only:
equilibration and the first factorization happen in [`setup`](@ref) and are not counted,
so on a fresh workspace the wall-clock cost of `solve` exceeds the limit by however long
setup took. The status is returned as soon as the budget is spent, without re-checking the
tolerances, so a run that stops this way reports `TIME_LIMIT_REACHED` even if its last
point would have passed. The iterates are still meaningful and
[`has_solution`](@ref) accepts it, as it does `MAX_ITER_REACHED`.

An `InterruptException` raised during the loop — a `Ctrl-C` — returns `INTERRUPTED` with
the point reached rather than losing the run; its residuals are recomputed first, since an
interrupt lands wherever it lands and not on a scheduled check. Every other exception
propagates.
"""
function solve!(ws::OperatorSplittingWorkspace{T}) where {T}
    s, alg = ws.options, ws.algorithm
    s.warm_starting || cold_start!(ws)
    ws.status = UNSOLVED
    ws.polished = false
    ws.status_polish = POLISH_NOT_PERFORMED
    ws.iter = 0
    # Per-run counters, so a second `solve!` on this workspace reports its own numbers
    # rather than the sum of both. `refactor_count` is deliberately not reset: it is a
    # property of the workspace's whole life.
    ws.rho_updates = 0
    # The accelerator counts over its whole life; this solve reports only its own share.
    declined_before = accelerator_declined(ws.accel)
    inner_before = inner_iterations(ws.linsys)
    ws.last_rel_kkt = INFTY(T)
    ws.solve_time = 0.0
    ws.polish_time = 0.0
    s.verbose && print_header(ws)
    # `time_ns` is monotonic and costs tens of nanoseconds against a per-iteration cost of
    # microseconds, but the whole check is skipped when no limit is set, so the default
    # path is exactly what it was. A limit makes the iteration count machine-dependent,
    # which is why it is off unless asked for.
    limited = isfinite(s.time_limit)
    started = time_ns()
    budget = limited ? round(UInt64, Float64(s.time_limit) * 1.0e9) : typemax(UInt64)
    # The integral is per solve, so a re-solve on the same workspace starts from zero rather
    # than continuing the previous one's curve.
    profiling = alg.profile_primdual
    ws.loop_start = started
    ws.primdual_int = 0.0
    ws.primdual_int_log = 0.0
    ws.last_gap_time = 0.0
    ws.last_gap = zero(T)
    try
        for iter in 1:s.max_iter
            ws.iter = iter
            accelerate_pre!(ws.accel, ws, iter)
            admm_step!(ws)
            accelerate_post!(ws.accel, ws, iter)
            if limited && time_ns() - started >= budget
                # Report the residuals of the point actually reached, not the stale ones
                # from the last scheduled check.
                update_residuals!(ws)
                profiling && accumulate_primdual!(ws)
                ws.status = TIME_LIMIT_REACHED
                s.verbose && print_row(ws)
                break
            end
            adapting = alg.adaptive_rho !== :disabled && alg.adaptive_rho_interval > 0 &&
                iszero(iter % alg.adaptive_rho_interval)
            checking = s.check_termination > 0 && iszero(iter % s.check_termination)
            (adapting || checking || isone(iter)) || continue
            update_residuals!(ws)
            # The integral is sampled wherever the gap is refreshed, which is here and at the
            # exits below. Its clock lives in this loop rather than in `update_residuals!`
            # because `time_ns` costs that function its allocation-free guarantee.
            profiling && accumulate_primdual!(ws)
            # Only on a termination check: the residuals and objective a row reports are
            # the ones that check just used, so a printed row always explains the decision
            # made alongside it.
            s.verbose && checking && print_row(ws)
            if checking
                st = check_termination(ws, false)
                if st != UNSOLVED
                    ws.status = st
                    break
                end
            end
            # The interval decides when the test is made. Under `:kkt_error` the test
            # itself is whether the error has fallen to `adaptive_rho_fraction` of what it
            # was when `ρ` last moved, so a run whose error stops falling stops retuning
            # `ρ` instead of paying for refactorizations that are not helping.
            if adapting
                allowed = alg.adaptive_rho !== :kkt_error ||
                    ws.rel_kkt_error <= alg.adaptive_rho_fraction * ws.last_rel_kkt
                allowed && adapt_rho!(ws) && (ws.last_rel_kkt = ws.rel_kkt_error)
            end
        end
    catch e
        e isa InterruptException || rethrow()
        # The iterates reached are a valid, if unconverged, point, so hand them back rather
        # than lose the run. The residuals are refreshed because an interrupt lands
        # wherever it lands, not on a scheduled check.
        update_residuals!(ws)
        profiling && accumulate_primdual!(ws)
        ws.status = INTERRUPTED
    end
    if ws.status == UNSOLVED
        update_residuals!(ws)
        profiling && accumulate_primdual!(ws)
        st = check_termination(ws, false)
        if st == UNSOLVED
            st = check_termination(ws, true)
        end
        ws.status = st == UNSOLVED ? MAX_ITER_REACHED : st
    end
    ws.solve_time = (time_ns() - started) / 1.0e9
    ws.accel_declined = accelerator_declined(ws.accel) - declined_before
    ws.cg_iters = inner_iterations(ws.linsys) - inner_before
    if (ws.status == SOLVED || ws.status == SOLVED_INACCURATE) && s.polishing
        t_polish = time_ns()
        ws.status_polish = polish!(ws)
        ws.polished = ws.status_polish === POLISH_SUCCESS
        ws.polish_time = (time_ns() - t_polish) / 1.0e9
    end
    s.verbose && print_footer(ws)
    sol = build_solution(ws)
    ws.first_run = false
    # The updates belonged to this run and are now reported; the next solve counts only the
    # ones made after it.
    ws.update_time = 0.0
    # An infeasible run leaves the iterates on a diverging ray; a later solve on this
    # workspace must not resume from there.
    has_solution(ws.status) || cold_start!(ws)
    return sol
end

"""
    solution_from(ws, obj, dual_obj, gap) -> Solution

Refill everything in the workspace's [`Solution`](@ref) that does not depend on the
outcome. The objectives and the gap are passed in because a run without a meaningful point
must not report them; `x`, `y` and the certificates are written by the caller.
"""
function solution_from(
        ws::OperatorSplittingWorkspace{T}, obj::T, dual_obj::T, gap::T
    ) where {T}
    sol = ws.sol
    sol.status = ws.status
    sol.obj_val = obj
    sol.dual_obj_val = dual_obj
    sol.duality_gap = gap
    sol.prim_res = ws.prim_res
    sol.dual_res = ws.dual_res
    sol.rel_kkt_error = ws.rel_kkt_error
    sol.iter = ws.iter
    sol.primdual_int = ws.primdual_int
    sol.primdual_int_log = ws.primdual_int_log
    sol.rho_estimate = ws.rho_estimate
    sol.rho_updates = ws.rho_updates
    sol.accel_declined = ws.accel_declined
    sol.cg_iters = ws.cg_iters
    sol.polished = ws.polished
    sol.status_polish = ws.status_polish
    sol.setup_time = ws.setup_time
    sol.update_time = ws.update_time
    sol.solve_time = ws.solve_time
    sol.polish_time = ws.polish_time
    # Setup is charged to the first run only; a re-solve did not pay it again. The updates
    # since the previous solve are charged here, because they are what this run cost the
    # caller.
    sol.run_time = (ws.first_run ? ws.setup_time : 0.0) +
        ws.update_time + ws.solve_time + ws.polish_time
    return sol
end

function build_solution(ws::OperatorSplittingWorkspace{T}) where {T}
    prob = ws.prob
    sol = ws.sol
    nan = T(NaN)
    scaled = prob.scaling > 0
    # The certificates carry the outcome in their length: a run reports the one its status
    # names and empties the other, and the memory for both is reserved at setup.
    if ws.status == PRIMAL_INFEASIBLE || ws.status == PRIMAL_INFEASIBLE_INACCURATE
        fill!(sol.x, nan)
        fill!(sol.y, nan)
        unit_certificate!(sol.prim_inf_cert, ws.yout, prob.E, ws.delta_y, scaled)
        resize!(sol.dual_inf_cert, 0)
        return solution_from(ws, T(Inf), nan, nan)
    elseif ws.status == DUAL_INFEASIBLE || ws.status == DUAL_INFEASIBLE_INACCURATE
        fill!(sol.x, nan)
        fill!(sol.y, nan)
        resize!(sol.prim_inf_cert, 0)
        unit_certificate!(sol.dual_inf_cert, ws.xout, prob.D, ws.delta_x, scaled)
        return solution_from(ws, T(-Inf), nan, nan)
    end
    resize!(sol.prim_inf_cert, 0)
    resize!(sol.dual_inf_cert, 0)
    if !has_solution(ws.status)
        # NON_CONVEX and anything else without a meaningful point: no number here would
        # mean anything, so do not hand back one that looks like a solution.
        fill!(sol.x, nan)
        fill!(sol.y, nan)
        return solution_from(ws, nan, nan, nan)
    end
    # Unscaled where the iterates live, then copied across once: `sol` holds `Vector`s and
    # the workspace's arrays need not support scalar indexing.
    unscale!(ws.xout, prob.D, ws.x, one(T))
    unscale!(ws.yout, prob.E, ws.y, prob.c)
    copyto!(sol.x, ws.xout)
    copyto!(sol.y, ws.yout)
    return solution_from(ws, ws.obj_val, ws.dual_obj_val, ws.duality_gap)
end
