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
    elseif ws.status == PRIMAL_INFEASIBLE
        # The conventional objective of an empty feasible set.
        obj = T(Inf)
        prim = dual = gap = T(NaN)
    elseif ws.status == DUAL_INFEASIBLE
        obj = T(-Inf)
        prim = dual = gap = T(NaN)
    else
        # A run that reached no conclusion: hitting the iteration limit, or a factorization
        # that stopped being trustworthy. An infinity here would read as a verdict on the
        # problem -- `-Inf` is what an unbounded one reports -- and no verdict was reached.
        obj = prim = dual = gap = T(NaN)
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
    refine_primal!(ws) -> Bool

Correct `x` on the rows the working set holds, and say whether it moved.

`primal!` forms `x` from `R⁻¹(−u − v)`, and those two terms are far larger than their
difference when `P` is badly conditioned -- a factor of 20 at `cond(P) = 1e8`. What cancels
is accuracy in `x`, so the active rows of the point sit further from their bounds than the
factorization itself is wrong by.

The step that puts them back is the smallest one that zeroes their residual: `Mₐ R dx = −ρ`
has the minimum-norm solution `R dx = −Mₐᵀ(Mₐ Mₐᵀ)⁻¹ρ`, and the working set already holds a
factorization of `Mₐ Mₐᵀ`. `ws.z` holds `A x`, so the residual costs no product of its own,
and the correction costs one triangular solve against `R`.

One step, not a loop: the residual it leaves is at the rounding level of the data, which a
second step cannot improve.
"""
function refine_primal!(ws::ActiveSetWorkspace{T}) where {T}
    red = ws.red
    lw = red.ws
    k = nactive(lw.W)
    k > 0 || return false
    prob = ws.prob
    # `mu_star` holds the multipliers the last pass solved for, which `multipliers!` does not
    # read -- it reads `mu` -- so it is free to carry the residual.
    rho = view(lw.mu_star, 1:k)
    @inbounds for i in 1:k
        r = lw.active[i]
        # The reduction swaps the two bounds -- `set_targets!` builds the lower target from
        # `bu` -- so a row held at its lower target is at the caller's upper bound.
        bound = lw.side[r] == SIDE_LOWER ? prob.u0[r] : prob.l0[r]
        # In the units the rows of `M` were normalized to, which is what the factorization
        # below is a factorization of.
        rho[i] = (ws.z[r] - bound) / red.scale[r]
    end
    solve_gram!(lw.W, rho)
    # `xold` is the proximal centre of the pass in flight, read only at the top of a pass, so
    # once the run has returned it is free for the correction to land in.
    dx = lw.xold
    active_product!(dx, lw.W, rho, lw.g)
    # `transpose(red.Rt)` unwraps to `red.R`, which is what makes this the untransposed solve.
    ldiv!(transpose(red.Rt), dx)
    @inbounds @simd for j in paired(ws.x, dx)
        ws.x[j] -= dx[j]
    end
    return true
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
    # `xold` is the proximal centre of the pass in flight, copied from the iterate before
    # every read, so once a solve has returned it is free for an `Aᵀy` to land in.
    add_adjoint_product!(r, prob.A, ws.y, ws.red.ws.xold)
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

# `r += Aᵀy`. A strided `A` does it in one `gemv`. Any other `A` forms `Aᵀy` in `scratch` first
# and adds it: the adjoint product an operator supplies has no accumulating form, and the
# generic one reads `A` entry by entry.
function add_adjoint_product!(r, A::StridedMatrix, y, scratch)
    mul!(r, adjoint(A), y, one(eltype(r)), one(eltype(r)))
    return r
end

function add_adjoint_product!(r, A::AbstractMatrix, y, scratch)
    mul!(scratch, adjoint(A), y)
    @simd for j in paired(r, scratch)
        r[j] += scratch[j]
    end
    return r
end
