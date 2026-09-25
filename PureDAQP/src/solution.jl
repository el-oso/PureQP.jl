"""
    build_solution(ws) -> Solution

Package the workspace's point as a [`Solution`](@ref), with the residuals recomputed from the
caller's own data.

The workspace's own solution is refilled and returned, and its `x` and `y` are the
workspace's own arrays, so a solve allocates nothing. Building a fresh one instead costs an
allocation for the object whatever its arrays are. What this means for a caller holding the
result across a solve is on [`Solution`](@ref).

Several fields are structurally zero here and stay that way: there is no `ρ` to report, no
accelerator, no conjugate-gradient count and no primal-dual integral, because none of those
exist in an active-set method.
"""
function build_solution(ws::ActiveSetWorkspace{T}) where {T}
    run_time = ws.setup_time + ws.update_time + ws.solve_time
    if has_solution(ws.status)
        # `Px` serves the objective, the dual residual and the gap, and `ws.z` already holds
        # `Ax` from the solve. Asking for either again is the same matrix product repeated.
        obj, prim, dual, gap = report(ws)
    else
        obj = ws.status == PRIMAL_INFEASIBLE ? T(Inf) : T(-Inf)
        prim = dual = gap = T(NaN)
    end
    sol = ws.sol
    sol.status = ws.status
    sol.obj_val = obj
    sol.dual_obj_val = obj - gap
    sol.duality_gap = gap
    sol.prim_res = prim
    sol.dual_res = dual
    sol.rel_kkt_error = max(prim, dual, abs(gap))
    sol.iter = ws.iter
    sol.polished = ws.polished
    sol.status_polish = ws.status_polish
    sol.setup_time = ws.setup_time
    sol.update_time = ws.update_time
    sol.solve_time = ws.solve_time
    sol.run_time = run_time
    return sol
end

"""
    report(ws) -> (objective, primal_residual, dual_residual, duality_gap)

Everything a [`Solution`](@ref) reports about the point, from the caller's own data, in one
pass.

They share their terms: `Px` appears in the objective, the dual residual and the gap, and
`Ax` is already in `ws.z` from the solve. Computed separately, a three-line report costs four
matrix products where two will do.

- objective `½ xᵀPx + qᵀx`
- primal `‖max(Ax−u, 0) + max(l−Ax, 0)‖∞`
- dual `‖Px + q + Aᵀy‖∞`
- gap `xᵀPx + qᵀx + uᵀmax(y,0) + lᵀmin(y,0)`, which an exact answer drives to rounding
"""
function report(ws::ActiveSetWorkspace{T}) where {T}
    prob = ws.prob
    Px = ws.px
    mul!(Px, prob.P, ws.x)
    quad = dot(ws.x, Px)
    linear = dot(prob.q0, ws.x)

    prim = zero(T)
    for i in eachindex(ws.z)
        zi = ws.z[i]
        prim = max(prim, max(zi - prob.u0[i], zero(T)), max(prob.l0[i] - zi, zero(T)))
    end

    r = Px
    mul!(r, transpose(prob.A), ws.y, one(T), one(T))
    dual = zero(T)
    for j in eachindex(r)
        dual = max(dual, abs(r[j] + prob.q0[j]))
    end

    support = zero(T)
    for i in eachindex(ws.y)
        yi = ws.y[i]
        if yi > 0
            isfinite(prob.u0[i]) && (support += prob.u0[i] * yi)
        elseif yi < 0
            isfinite(prob.l0[i]) && (support += prob.l0[i] * yi)
        end
    end

    return T(0.5) * quad + linear, prim, dual, quad + linear + support
end
