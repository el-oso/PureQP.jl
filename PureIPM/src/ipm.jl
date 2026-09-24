# The Mehrotra predictor–corrector interior-point method on the equilibrated problem.
#
# Each inequality side `i` has the regularized slack row `±Ã_iΔx − Δs + δ_d Δz = −r` and the
# complementarity row `z Δs + s Δz = −r_c`. Eliminating `Δs` and `Δz` side by side leaves
#
#     ⎡P̃ + δ_p I      Ãᵀ     ⎤ ⎡Δx⎤   ⎡   −r_d     ⎤
#     ⎣Ã         −diag(w_inv)⎦ ⎣Δy⎦ = ⎣−w_inv ⊙ g ⎦
#
# with `d_l = s_l + δ_d z_l`, `d_u = s_u + δ_d z_u`, `w = z_l/d_l + z_u/d_u` and
# `g = (r_cl + z_l r_l)/d_l − (r_cu + z_u r_u)/d_u`, which is exactly the system every
# `LinearSystem` backend solves. The slack and multiplier steps are then recovered side by side
# from `ÃΔx`, so the recovered `Δz_u − Δz_l` equals the solved `Δy`: a harmonic weight
# `1/(1/W + δ_d)` with `Δs = ÃΔx + r` would not, and the step would solve no single Newton
# system. An equality row is `Ã_iΔx − δ_d Δy_i = −r_e`; a free row has `w_inv = 1/δ_d` and its
# `Δy` is zeroed after the solve.

"`max|v|` with a start value, for the Newton residual norm."
@inline absmax(acc, v) = max(acc, abs(v))

"Steps shorter than this, `STALL_STEPS` times in a row, stall the iteration."
@inline STALL_STEP(::Type{T}) where {T} = ipm_floor(T)
const STALL_STEPS = 3
"Iterations in a row whose merit does not fall before the certificate tests run every iteration."
const STALL_MERIT = 10
"""
Multiple of [`iterate_bound`](@ref) past which an iterate cannot be recovered by further
steps: three orders of magnitude past the point where certificate testing already runs every
iteration is past any margin a genuine convergent or infeasible run needs.
"""
@inline DIVERGENCE_CEILING(::Type{T}) where {T} = T(1000)

"""
    set_regularization!(ws, reg_primal, reg_dual) -> ws

Put `reg_primal` and `reg_dual` in force. A new `reg_primal` becomes the weights' `sigma`, set
in place so a bump inside the iteration allocates nothing, and marks the backend's
factorization as needing a full rebuild.
"""
function set_regularization!(ws::InteriorPointWorkspace{T}, reg_primal::T, reg_dual::T) where {T}
    ws.reg_dual = reg_dual
    if reg_primal != ws.reg_primal
        ws.reg_primal = reg_primal
        ws.weights.sigma = reg_primal
        ws.sigma_changed = true
    end
    return ws
end

"""
    factorize_newton!(ws, starting) -> Bool

Factorize the Newton system at the current weights, raising both regularizations tenfold and
retrying whenever the backend reports failure, at most `max_reg_bumps` times per solve.
`starting` says the weights are the unit weights of the starting point, which do not depend
on `reg_dual`; otherwise they are recomputed after a bump. `false` means the bumps ran out.
"""
function factorize_newton!(ws::InteriorPointWorkspace{T}, starting::Bool) where {T}
    ls = ws.linsys
    while true
        full = starting || ws.sigma_changed
        ok = full ? factorize!(ls, ws.prob, ws.weights) : refactor_weights!(ls, ws.prob, ws.weights)
        if ok
            ws.sigma_changed = false
            return true
        end
        ws.reg_bumps < ws.algorithm.max_reg_bumps || return false
        ws.reg_bumps += 1
        set_regularization!(ws, T(10) * ws.reg_primal, T(10) * ws.reg_dual)
        starting || weights!(ws)
    end
    return
end

"""
    starting_point!(ws) -> Bool

Build the starting slacks and multipliers. Unseeded, `x` solves
`(P̃ + δ_p I + ÃᵀÃ) x = −q̃ + Ãᵀt` with `t` the midpoint of a two-sided row, the finite bound of
a one-sided one, `l̃` of an equality and `0` of a free row, and `y` starts at zero; seeded, `x`
and `y` are the workspace's. The slacks are `Ãx − l̃` and `ũ − Ãx`, shifted up by
`1.5·max(0, −min s)`; the multipliers are one, or `max(∓y, 0) + 1` when seeded; both then take
Mehrotra's balancing shift `½ sᵀz / eᵀz` and `½ sᵀz / eᵀs`. With no inequality side only `x`
and `y` are set. `false` means the unseeded system could not be factorized within
`max_reg_bumps`, and nothing was changed.
"""
function starting_point!(ws::InteriorPointWorkspace{T}) where {T}
    prob, wt, ls = ws.prob, ws.weights, ws.linsys
    l, u, rclass, has_l, has_u = prob.l, prob.u, ws.rclass, ws.has_l, ws.has_u
    x, y, s_l, s_u, z_l, z_u = ws.x, ws.y, ws.s_l, ws.s_u, ws.z_l, ws.z_u
    o, zr = one(T), zero(T)
    if !ws.seeded
        fill!(wt.w, o)
        fill!(wt.w_inv, o)
        set_refresh_index!(ls, -1)
        factorize_newton!(ws, true) || return false
        wt = ws.weights
        for j in eachindex(ws.rhs_x)
            ws.rhs_x[j] = -prob.q[j]
        end
        for i in eachindex(ws.rhs_z)
            c = rclass[i]
            ws.rhs_z[i] = c == ROW_EQUALITY ? l[i] : c == ROW_FREE ? zr :
                has_l[i] && has_u[i] ? (l[i] + u[i]) / 2 : has_l[i] ? l[i] : u[i]
        end
        # An iterative backend solves this system to its relative floor and starts from zero.
        set_tolerance_level!(ls, zr)
        fill!(x, zr)
        solve_system!(ls, prob, wt, ws.rhs_x, ws.rhs_z, x, ws.Adx)
        count_miss!(ws)
        fill!(y, zr)
    end
    prob.m > 0 && mul_A!(ws.Ax, prob, x)
    Ax = ws.Ax
    smin = zr
    for i in eachindex(s_l)
        s_l[i] = has_l[i] ? Ax[i] - l[i] : o
        s_u[i] = has_u[i] ? u[i] - Ax[i] : o
        z_l[i] = has_l[i] ? (ws.seeded ? max(-y[i], zr) + o : o) : zr
        z_u[i] = has_u[i] ? (ws.seeded ? max(y[i], zr) + o : o) : zr
        has_l[i] && (smin = min(smin, s_l[i]))
        has_u[i] && (smin = min(smin, s_u[i]))
        rclass[i] == ROW_FREE && (y[i] = zr)
    end
    ws.n_sides > 0 || return true
    theta = max(zr, -T(1.5) * smin)
    sz, sums, sumz = zr, zr, zr
    for i in eachindex(s_l)
        has_l[i] && (s_l[i] += theta; sums += s_l[i])
        has_u[i] && (s_u[i] += theta; sums += s_u[i])
        sz += s_l[i] * z_l[i] + s_u[i] * z_u[i]
        sumz += z_l[i] + z_u[i]
    end
    shift_s = sz / (2 * sumz)
    shift_z = sz / (2 * sums)
    for i in eachindex(s_l)
        if has_l[i]
            s_l[i] += shift_s
            z_l[i] += shift_z
        end
        if has_u[i]
            s_u[i] += shift_s
            z_u[i] += shift_z
        end
        rclass[i] == ROW_INEQUALITY && (y[i] = z_u[i] - z_l[i])
    end
    return true
end

"""
    ipm_residuals!(ws) -> ws

At the current `(x, y, s)`: the primal and dual residuals, the duality gap and the objective,
exactly as ADMM reports them with `z = clamp(Ãx, l̃, ũ)`, and the Newton residuals `r_d`,
`r_l`, `r_u` and their largest magnitude `rnorm`, in scaled space.
"""
function ipm_residuals!(ws::InteriorPointWorkspace{T}) where {T}
    prob = ws.prob
    m = prob.m
    scaled = prob.scaling > 0
    l, u, Ax, z = prob.l, prob.u, ws.Ax, ws.z
    if m > 0
        mul_A!(Ax, prob, ws.x)
        for i in eachindex(z)
            z[i] = clamp(Ax[i], l[i], u[i])
        end
        subtract!(prob.work_m, Ax, z)
        ws.scaled_prim_res = norm_inf(prob.work_m)
        ws.prim_res = scaled ? invscaled_norm_inf(prob.E, prob.work_m) : ws.scaled_prim_res
    else
        ws.scaled_prim_res = zero(T)
        ws.prim_res = zero(T)
    end
    mul_P!(ws.Px, prob, ws.x)
    add!(ws.r_d, prob.q, ws.Px)
    if m > 0
        mul_At!(ws.Aty, prob, ws.y)
        increment!(ws.r_d, ws.Aty)
    else
        fill!(ws.Aty, zero(T))
    end
    ws.scaled_dual_res = norm_inf(ws.r_d)
    ws.dual_res = scaled ? invscaled_norm_inf(prob.D, ws.r_d) / prob.c : ws.scaled_dual_res
    quad, lin, sup = gap_terms(prob, ws.y, ws.Px, ws.x)
    ws.xtPx = quad
    ws.qtx = lin
    ws.SCy = sup
    ws.scaled_duality_gap = quad + lin + sup
    cinv = inv(prob.c)
    ws.obj_val = (quad / 2 + lin) * cinv
    ws.dual_obj_val = (-quad / 2 - sup) * cinv
    ws.duality_gap = ws.scaled_duality_gap * cinv
    ws.rel_kkt_error = max(ws.prim_res, ws.dual_res, abs(ws.duality_gap))
    rclass, has_l, has_u, r_l, r_u = ws.rclass, ws.has_l, ws.has_u, ws.r_l, ws.r_u
    rn = ws.scaled_dual_res
    for i in eachindex(r_l)
        c = rclass[i]
        rl = c == ROW_EQUALITY ? Ax[i] - l[i] : has_l[i] ? Ax[i] - l[i] - ws.s_l[i] : zero(T)
        ru = has_u[i] ? u[i] - Ax[i] - ws.s_u[i] : zero(T)
        r_l[i] = rl
        r_u[i] = ru
        rn = absmax(absmax(rn, rl), ru)
    end
    ws.rnorm = rn
    return ws
end

"""
    weights!(ws) -> ws

`μ = (s_lᵀz_l + s_uᵀz_u)/N_s` (zero without an inequality side) and the system weights of the
current point: `w = z_l/(s_l + δ_d z_l) + z_u/(s_u + δ_d z_u)` and `w_inv = 1/w` on inequality
rows, `w_inv = δ_d` on equality rows, `w_inv = 1/δ_d` on free rows.
"""
function weights!(ws::InteriorPointWorkspace{T}) where {T}
    delta = ws.reg_dual
    w, w_inv = ws.weights.w, ws.weights.w_inv
    s_l, s_u, z_l, z_u, rclass = ws.s_l, ws.s_u, ws.z_l, ws.z_u, ws.rclass
    sz = zero(T)
    for i in eachindex(w)
        sz += s_l[i] * z_l[i] + s_u[i] * z_u[i]
        c = rclass[i]
        if c == ROW_EQUALITY
            w[i] = inv(delta)
            w_inv[i] = delta
        elseif c == ROW_FREE
            w[i] = delta
            w_inv[i] = inv(delta)
        else
            wi = z_l[i] / (s_l[i] + delta * z_l[i]) + z_u[i] / (s_u[i] + delta * z_u[i])
            w[i] = wi
            w_inv[i] = inv(wi)
        end
    end
    ws.mu = ws.n_sides > 0 ? sz / ws.n_sides : zero(T)
    return ws
end

"""
    refine!(ws) -> ws

One refinement step of `(dx, dy)` against the regularized system the backend factored.
"""
function refine!(ws::InteriorPointWorkspace{T}) where {T}
    prob, wt = ws.prob, ws.weights
    sigma, w_inv = wt.sigma, wt.w_inv
    dx, dy, res_x, res_z = ws.dx, ws.dy, ws.res_x, ws.res_z
    mul_P!(res_x, prob, dx)
    if prob.m > 0
        mul_At!(ws.corr_x, prob, dy)
    else
        fill!(ws.corr_x, zero(T))
    end
    for j in eachindex(res_x)
        res_x[j] = ws.rhs_x[j] - (res_x[j] + sigma * dx[j] + ws.corr_x[j])
    end
    if prob.m > 0
        mul_A!(res_z, prob, dx)
        for i in eachindex(res_z)
            res_z[i] = ws.rhs_z[i] - (res_z[i] - w_inv[i] * dy[i])
        end
    end
    solve_multiplier!(ws.linsys, prob, wt, res_x, res_z, ws.corr_x, ws.corr_y)
    increment!(dx, ws.corr_x)
    increment!(dy, ws.corr_y)
    return ws
end

"""
Count the backend's last solve toward the run of consecutive missed solves, or end that run;
`cg_total_misses` counts every miss over the whole solve, for the verbose footer.
"""
function count_miss!(ws::InteriorPointWorkspace)
    if last_solve_converged(ws.linsys)
        ws.cg_misses = 0
    else
        ws.cg_misses += 1
        ws.cg_total_misses += 1
    end
    return ws
end

"""
    direction!(ws) -> ws

Solve the Newton system for the complementarity terms in `rc_l`, `rc_u`, and recover the slack
and multiplier steps side by side.
"""
function direction!(ws::InteriorPointWorkspace{T}) where {T}
    prob, wt, s = ws.prob, ws.weights, ws.algorithm
    delta = ws.reg_dual
    rclass, has_l, has_u = ws.rclass, ws.has_l, ws.has_u
    s_l, s_u, z_l, z_u, r_l, r_u, rc_l, rc_u = ws.s_l, ws.s_u, ws.z_l, ws.z_u, ws.r_l, ws.r_u, ws.rc_l, ws.rc_u
    w_inv, rhs_z, dy = wt.w_inv, ws.rhs_z, ws.dy
    zr = zero(T)
    for i in eachindex(rhs_z)
        c = rclass[i]
        if c == ROW_EQUALITY
            rhs_z[i] = -r_l[i]
        elseif c == ROW_FREE
            rhs_z[i] = zr
        else
            gl = has_l[i] ? (rc_l[i] + z_l[i] * r_l[i]) / (s_l[i] + delta * z_l[i]) : zr
            gu = has_u[i] ? (rc_u[i] + z_u[i] * r_u[i]) / (s_u[i] + delta * z_u[i]) : zr
            rhs_z[i] = -w_inv[i] * (gl - gu)
        end
    end
    fill!(ws.dx, zr)
    solve_multiplier!(ws.linsys, prob, wt, ws.rhs_x, rhs_z, ws.dx, dy)
    count_miss!(ws)
    for _ in 1:s.refine_iter
        refine!(ws)
    end
    prob.m > 0 && mul_A!(ws.Adx, prob, ws.dx)
    Adx = ws.Adx
    for i in eachindex(dy)
        rclass[i] == ROW_FREE && (dy[i] = zr)
        if has_l[i]
            dzl = -(rc_l[i] + z_l[i] * (Adx[i] + r_l[i])) / (s_l[i] + delta * z_l[i])
            ws.dz_l[i] = dzl
            ws.ds_l[i] = Adx[i] + r_l[i] + delta * dzl
        else
            ws.dz_l[i] = zr
            ws.ds_l[i] = zr
        end
        if has_u[i]
            dzu = -(rc_u[i] + z_u[i] * (-Adx[i] + r_u[i])) / (s_u[i] + delta * z_u[i])
            ws.dz_u[i] = dzu
            ws.ds_u[i] = -Adx[i] + r_u[i] + delta * dzu
        else
            ws.dz_u[i] = zr
            ws.ds_u[i] = zr
        end
    end
    return ws
end

"The largest step in `(0, cap]` along the current direction that keeps every slack and multiplier nonnegative."
function max_step(ws::InteriorPointWorkspace{T}, cap::T) where {T}
    a = cap
    s_l, s_u, z_l, z_u, ds_l, ds_u, dz_l, dz_u = ws.s_l, ws.s_u, ws.z_l, ws.z_u, ws.ds_l, ws.ds_u, ws.dz_l, ws.dz_u
    for i in eachindex(s_l)
        ds_l[i] < zero(T) && (a = min(a, -s_l[i] / ds_l[i]))
        ds_u[i] < zero(T) && (a = min(a, -s_u[i] / ds_u[i]))
        dz_l[i] < zero(T) && (a = min(a, -z_l[i] / dz_l[i]))
        dz_u[i] < zero(T) && (a = min(a, -z_u[i] / dz_u[i]))
    end
    return a
end

"""
    ipm_step!(ws) -> ws

One predictor–corrector step from the residuals of [`ipm_residuals!`](@ref) and the
factorization of the weights of [`weights!`](@ref). The predictor's step length `α_a` gives
`μ_a` and the centering `σ = max((μ_a/μ)³, min(1, 0.1‖r‖∞/μ))`, whose floor keeps `μ` from
being driven to zero while the Newton residuals are still large. The corrector adds
`Δs_a ∘ Δz_a − σμ` to the complementarity terms, and the step taken is `step_fraction` of the
way to the boundary, capped at one, for primal and dual alike.

With no inequality side there is no complementarity: one solve, and the full step.
"""
function ipm_step!(ws::InteriorPointWorkspace{T}) where {T}
    s = ws.algorithm
    zr, o = zero(T), one(T)
    for j in eachindex(ws.rhs_x)
        ws.rhs_x[j] = -ws.r_d[j]
    end
    s_l, s_u, z_l, z_u, rc_l, rc_u = ws.s_l, ws.s_u, ws.z_l, ws.z_u, ws.rc_l, ws.rc_u
    mu = ws.mu
    if ws.n_sides > 0
        set_tolerance_level!(ws.linsys, min(mu, ws.rnorm))
        # Absent sides hold `z = 0`, so their products vanish without a mask.
        multiply!(rc_l, s_l, z_l)
        multiply!(rc_u, s_u, z_u)
        direction!(ws)
        alpha_a = max_step(ws, o)
        sz = zr
        for i in eachindex(s_l)
            sz += (s_l[i] + alpha_a * ws.ds_l[i]) * (z_l[i] + alpha_a * ws.dz_l[i]) +
                (s_u[i] + alpha_a * ws.ds_u[i]) * (z_u[i] + alpha_a * ws.dz_u[i])
        end
        sigma = max((sz / ws.n_sides / mu)^3, min(o, T(0.1) * ws.rnorm / mu))
        target = sigma * mu
        has_l, has_u = ws.has_l, ws.has_u
        for i in eachindex(rc_l)
            rc_l[i] = has_l[i] ? s_l[i] * z_l[i] + ws.ds_l[i] * ws.dz_l[i] - target : zr
            rc_u[i] = has_u[i] ? s_u[i] * z_u[i] + ws.ds_u[i] * ws.dz_u[i] - target : zr
        end
        direction!(ws)
        alpha = min(o, s.step_fraction * max_step(ws, INFTY(T)))
    else
        set_tolerance_level!(ws.linsys, ws.rnorm)
        fill!(rc_l, zr)
        fill!(rc_u, zr)
        direction!(ws)
        alpha = o
    end
    ws.alpha = alpha
    x, y, rclass = ws.x, ws.y, ws.rclass
    for j in eachindex(x)
        x[j] += alpha * ws.dx[j]
    end
    for i in eachindex(y)
        c = rclass[i]
        if c == ROW_EQUALITY
            y[i] += alpha * ws.dy[i]
        elseif c == ROW_INEQUALITY
            s_l[i] += alpha * ws.ds_l[i]
            s_u[i] += alpha * ws.ds_u[i]
            z_l[i] += alpha * ws.dz_l[i]
            z_u[i] += alpha * ws.dz_u[i]
            y[i] = z_u[i] - z_l[i]
        end
    end
    return ws
end

"""
    primal_certificate!(ws, eps, test_direction) -> Bool

Run the primal infeasibility test on `y/‖y‖∞`, copied into `cert_y`, and, when
`test_direction`, first on the last step `Δy`. `true` leaves the passing candidate, projected,
in `cert_y`.
"""
function primal_certificate!(ws::InteriorPointWorkspace{T}, eps::T, test_direction::Bool) where {T}
    prob, cert = ws.prob, ws.cert_y
    if test_direction
        copyto!(cert, ws.dy)
        is_primal_infeasible(prob, cert, eps) && return true
    end
    ny = norm_inf(ws.y)
    ny > zero(T) || return false
    for i in eachindex(cert)
        cert[i] = ws.y[i] / ny
    end
    return is_primal_infeasible(prob, cert, eps)
end

"""
    dual_certificate!(ws, eps, test_direction) -> Bool

Run the dual infeasibility test on `x/‖x‖∞`, copied into `cert_x`, and, when `test_direction`,
first on the last step `Δx`. `true` leaves the passing candidate in `cert_x`.
"""
function dual_certificate!(ws::InteriorPointWorkspace{T}, eps::T, test_direction::Bool) where {T}
    prob, cert = ws.prob, ws.cert_x
    if test_direction
        copyto!(cert, ws.dx)
        is_dual_infeasible(prob, cert, eps) && return true
    end
    nx = norm_inf(ws.x)
    nx > zero(T) || return false
    for j in eachindex(cert)
        cert[j] = ws.x[j] / nx
    end
    return is_dual_infeasible(prob, cert, eps)
end

"""
    check_termination(ws::InteriorPointWorkspace, approximate = false, test_direction = false) -> Status

`SOLVED` when the primal residual, the dual residual and, with `check_dualgap`, the duality
gap pass the tolerances [`eps_prim`](@ref), [`eps_dual`](@ref) and
[`eps_duality_gap`](@ref), exactly as ADMM tests them. A primal residual that fails runs the
primal infeasibility test at `eps_prim_inf`, and a dual residual that fails runs the dual one
at `eps_dual_inf`, each on the normalized iterate and, when `test_direction`, first on the last
step (see [`is_primal_infeasible`](@ref), [`is_dual_infeasible`](@ref)); a passing test gives
`PRIMAL_INFEASIBLE` or `DUAL_INFEASIBLE`. `UNSOLVED` otherwise. With `approximate = true`
every tolerance is ten times larger and the statuses are the `*_INACCURATE` variants.
`test_direction` is reserved for a termination check the stall or divergence guard has
triggered; a plain periodic check tests the normalized iterate only.
"""
function check_termination(
        ws::InteriorPointWorkspace{T}, approximate::Bool = false, test_direction::Bool = false
    ) where {T}
    s, prob = ws.options, ws.prob
    f = approximate ? T(10) : one(T)
    scaled_term = s.scaled_termination && prob.scaling > 0
    pres = scaled_term ? ws.scaled_prim_res : ws.prim_res
    dres = scaled_term ? ws.scaled_dual_res : ws.dual_res
    prim_ok = iszero(prob.m) || pres < f * eps_prim(prob, s, ws.z, ws.Ax)
    if !prim_ok && primal_certificate!(ws, f * s.eps_prim_inf, test_direction)
        return approximate ? PRIMAL_INFEASIBLE_INACCURATE : PRIMAL_INFEASIBLE
    end
    dual_ok = dres < f * eps_dual(prob, s, ws.Aty, ws.Px)
    if !dual_ok && dual_certificate!(ws, f * s.eps_dual_inf, test_direction)
        return approximate ? DUAL_INFEASIBLE_INACCURATE : DUAL_INFEASIBLE
    end
    (prim_ok && dual_ok) || return UNSOLVED
    if s.check_dualgap
        gap = scaled_term ? ws.scaled_duality_gap : ws.duality_gap
        abs(gap) < f * eps_duality_gap(prob, s, ws.xtPx, ws.qtx, ws.SCy) || return UNSOLVED
    end
    return approximate ? SOLVED_INACCURATE : SOLVED
end

"""
    iterate_bound(ws) -> T

`1/sqrt(eps)` times the size of the data, `max(1, ‖q̃‖∞, finite |l̃|, finite |ũ|)`. An iterate
larger than this is diverging.
"""
function iterate_bound(ws::InteriorPointWorkspace{T}) where {T}
    prob = ws.prob
    loose = INFTY(T) * MIN_SCALING(T)
    b = max(one(T), norm_inf(prob.q))
    for i in eachindex(prob.l)
        li, ui = prob.l[i], prob.u[i]
        li > -loose && (b = max(b, abs(li)))
        ui < loose && (b = max(b, abs(ui)))
    end
    return b / T(sqrt(precision_eps(T)))
end

"""
    stalled!(ws, bound) -> Bool

Advance the guards by the step just taken. The iteration stalls after `STALL_STEPS`
consecutive steps shorter than `STALL_STEP`. `alert` is raised, and stays raised for the
solve, after `STALL_MERIT` consecutive iterations whose merit, `μ` (`‖r‖∞` without an
inequality side), is not below the previous iteration's, or once `‖x‖∞` or `‖y‖∞` exceeds
`bound`. A rising `μ` is what an infeasible problem's diverging multipliers produce, so it
calls for the certificate tests rather than ending the run. `diverged` is raised once `‖x‖∞`
or `‖y‖∞` exceeds `DIVERGENCE_CEILING(T)` times `bound`, past which the iterate is beyond
recovery regardless of what the certificate tests find.
"""
function stalled!(ws::InteriorPointWorkspace{T}, bound::T) where {T}
    ws.short_steps = ws.alpha < STALL_STEP(T) ? ws.short_steps + 1 : 0
    merit = ws.n_sides > 0 ? ws.mu : ws.rnorm
    ws.flat_merit = merit < ws.last_merit ? 0 : ws.flat_merit + 1
    ws.last_merit = merit
    nx, ny = norm_inf(ws.x), norm_inf(ws.y)
    (ws.flat_merit >= STALL_MERIT || nx > bound || ny > bound) && (ws.alert = true)
    ws.diverged = nx > DIVERGENCE_CEILING(T) * bound || ny > DIVERGENCE_CEILING(T) * bound
    return ws.short_steps >= STALL_STEPS
end

finite_residuals(ws::InteriorPointWorkspace) = isfinite(ws.rnorm) && isfinite(ws.prim_res) && isfinite(ws.dual_res)

# The `verbose` output.
#
# Everything here writes to `Core.stdout` and formats by hand. That is not a style choice:
# `--trim` analyses this code whether or not `verbose` is ever set, and it rejects both
# Printf (its format specifications carry type parameters that do not infer) and bare
# `println(x)` (`Base.stdout` is an abstractly typed global). `Core.stdout` is a concrete
# singleton, so calls through it resolve statically; `redirect_stdout` still captures it,
# since that redirects the file descriptor.
#
# These two are an algorithm's own rather than shared. Shared, `print_padded`'s value argument
# is inferred over every caller at once, and the `string` it reaches then takes an argument
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

function print_header(ws::InteriorPointWorkspace)
    println(Core.stdout, VERBOSE_RULE)
    println(Core.stdout, "            PureIPM - interior-point QP solver")
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
    if backend_name(ws.linsys) === :indirect
        println(Core.stdout, " iter      objective      prim res      dual res            mu         alpha      cg iters")
    else
        println(Core.stdout, " iter      objective      prim res      dual res            mu         alpha")
    end
    return nothing
end

"One row: `cg_this_iter` is this outer iteration's conjugate-gradient count, printed only on
the matrix-free backend."
function print_row(ws::InteriorPointWorkspace, cg_this_iter::Int)
    print_padded(string(ws.iter), 5)
    print_padded(ws.obj_val, 15, 6)
    print_padded(ws.prim_res, 14, 3)
    print_padded(ws.dual_res, 14, 3)
    print_padded(ws.mu, 14, 3)
    print_padded(ws.alpha, 14, 3)
    backend_name(ws.linsys) === :indirect && print_padded(string(cg_this_iter), 14)
    print(Core.stdout, "\n")
    return nothing
end

function print_footer(ws::InteriorPointWorkspace)
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
    print(Core.stdout, "run time:             ")
    println(Core.stdout, (ws.first_run ? ws.setup_time : 0.0) + ws.update_time + ws.solve_time + ws.polish_time)
    if backend_name(ws.linsys) === :indirect
        print(Core.stdout, "total CG iterations:  ")
        println(Core.stdout, ws.cg_iters)
        print(Core.stdout, "missed CG solves:     ")
        println(Core.stdout, ws.cg_total_misses)
    end
    println(Core.stdout, VERBOSE_RULE)
    return nothing
end

"""
    solve!(ws::InteriorPointWorkspace) -> Solution

Run the interior-point method. Each outer iteration refactorizes the Newton system at the
current weights, takes one predictor–corrector step and recomputes the residuals, testing
for termination and infeasibility (see [`check_termination`](@ref)) every
`check_termination` iterations. At `max_iter` the tests are retried at ten times the
tolerances, and otherwise the status is `MAX_ITER_REACHED`.

Safeguards, checked every iteration:

- A factorization failure raises both regularizations tenfold and retries, at most
  `max_reg_bumps` times in the solve; past that the run ends `NUMERICAL_ERROR`.
- A non-finite residual ends the run `NUMERICAL_ERROR`.
- `cg_fail_limit` consecutive missed solves of an iterative backend (see
  [`last_solve_converged`](@ref)) end the run `NUMERICAL_ERROR`. With `linsys = :indirect` a
  miss is a solve that spent `cg_max_iter` iterations or that conjugate gradients abandoned,
  which a preconditioner that is not symmetric positive definite causes.
- A stalled iteration (see [`stalled!`](@ref)) runs the tests at once, on the step direction
  and the normalized iterate, then at ten times the tolerances, and ends `NUMERICAL_ERROR` if
  neither passes.
- Once `μ` has stopped falling for `STALL_MERIT` iterations, or an iterate exceeds
  [`iterate_bound`](@ref), the tests run every iteration, on the step direction and the
  normalized iterate, whatever `check_termination` says, and the run continues.
- An iterate past `DIVERGENCE_CEILING(T)` times [`iterate_bound`](@ref) ends the run
  `NUMERICAL_ERROR` once that iteration's own termination check (already running, per the
  previous point) finds no certificate.
- `time_limit` and an `InterruptException` end the run `TIME_LIMIT_REACHED` and
  `INTERRUPTED` with the point reached, as for ADMM; the clock includes the starting point.

The point is seeded from the previous solve with `warm_starting = true`, from
[`warm_start!`](@ref), and otherwise computed (see [`InteriorPointWorkspace`](@ref)). A solve that ends
without a point ([`has_solution`](@ref) false) clears the seed, and its `Solution` carries
`NaN` in `x` and `y`.
"""
function solve!(ws::InteriorPointWorkspace{T}) where {T}
    s, alg = ws.options, ws.algorithm
    s.warm_starting || (ws.seeded = false)
    ws.status = UNSOLVED
    ws.polished = false
    ws.status_polish = POLISH_NOT_PERFORMED
    ws.iter = 0
    ws.reg_bumps = 0
    set_regularization!(ws, alg.reg_primal, alg.reg_dual)
    ws.short_steps = 0
    ws.flat_merit = 0
    ws.last_merit = INFTY(T)
    ws.alert = false
    ws.diverged = false
    ws.cg_misses = 0
    ws.cg_total_misses = 0
    ws.polish_time = 0.0
    bound = iterate_bound(ws)
    inner_before = inner_iterations(ws.linsys)
    # As in ADMM's loop: the clock is read only when a limit is set.
    limited = isfinite(s.time_limit)
    started = time_ns()
    budget = limited ? round(UInt64, Float64(s.time_limit) * 1.0e9) : typemax(UInt64)
    s.verbose && print_header(ws)
    try
        if starting_point!(ws)
            ipm_residuals!(ws)
            (finite_residuals(ws) && ws.cg_misses < alg.cg_fail_limit) || (ws.status = NUMERICAL_ERROR)
        else
            ws.status = NUMERICAL_ERROR
        end
        for iter in 1:s.max_iter
            ws.status == UNSOLVED || break
            ws.iter = iter
            weights!(ws)
            set_refresh_index!(ws.linsys, iter - 1)
            if !factorize_newton!(ws, false)
                ws.status = NUMERICAL_ERROR
                break
            end
            cg_before = inner_iterations(ws.linsys)
            ipm_step!(ws)
            cg_this_iter = inner_iterations(ws.linsys) - cg_before
            ipm_residuals!(ws)
            if !finite_residuals(ws) || ws.cg_misses >= alg.cg_fail_limit
                ws.status = NUMERICAL_ERROR
                break
            end
            if limited && time_ns() - started >= budget
                ws.status = TIME_LIMIT_REACHED
                break
            end
            stall = stalled!(ws, bound)
            checking = s.check_termination > 0 && iszero(iter % s.check_termination)
            triggered = stall || ws.alert
            s.verbose && (checking || triggered) && print_row(ws, cg_this_iter)
            if checking || triggered
                st = check_termination(ws, false, triggered)
                if st != UNSOLVED
                    ws.status = st
                    break
                end
            end
            if ws.diverged
                ws.status = NUMERICAL_ERROR
                break
            end
            if stall
                st = check_termination(ws, true, true)
                ws.status = st == UNSOLVED ? NUMERICAL_ERROR : st
                break
            end
        end
    catch e
        e isa InterruptException || rethrow()
        # An interrupt lands wherever it lands, so the residuals are those of the point reached.
        ipm_residuals!(ws)
        ws.status = INTERRUPTED
    end
    if ws.status == UNSOLVED
        st = check_termination(ws, false, ws.alert)
        st == UNSOLVED && (st = check_termination(ws, true, ws.alert))
        ws.status = st == UNSOLVED ? MAX_ITER_REACHED : st
    end
    ws.solve_time = (time_ns() - started) / 1.0e9
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
    ws.seeded = has_solution(ws.status)
    return sol
end

"""
    polish!(ws::InteriorPointWorkspace) -> PolishStatus

Guess the active set from the interior-point iterates `(ws.x, ws.y, ws.z)` — `ws.z` is
already `clamp(Ãx, l̃, ũ)` — solve the resulting equality-constrained QP exactly, and adopt
the result only if both residuals improve. Delegates to [`polish_kernel!`](@ref). On
`POLISH_SUCCESS` it copies the polished point into `ws.x` and `ws.y` and recomputes the
residuals with [`ipm_residuals!`](@ref), which rebuilds `ws.z` from the polished `x` rather
than adopting the kernel's own candidate, keeping `z` the same `clamp(Ãx, l̃, ũ)` it is
everywhere else in the interior-point method.
"""
function polish!(ws::InteriorPointWorkspace{T}) where {T}
    prob = ws.prob
    status, xpol, ypol, _ = polish_kernel!(
        prob, ws.x, ws.y, ws.z, ws.prim_res, ws.dual_res, ws.Ax, ws.Px, ws.Aty;
        delta = ws.options.delta, refine_iter = ws.options.polish_refine_iter
    )
    status === POLISH_SUCCESS || return status
    copyto!(ws.x, xpol)
    copyto!(ws.y, ypol)
    ipm_residuals!(ws)
    return POLISH_SUCCESS
end

"""
    ipm_solution(ws, x, y, obj, dual_obj, gap, prim_cert, dual_cert) -> Solution

Assemble a [`Solution`](@ref) from the workspace's counters and the values that depend on
how the run ended.
"""
function ipm_solution(
        ws::InteriorPointWorkspace{T}, x, y, obj::T, dual_obj::T, gap::T, prim_cert, dual_cert
    ) where {T}
    return Solution{T}(
        Vector{T}(x), Vector{T}(y), ws.status, obj, dual_obj, gap,
        ws.prim_res, ws.dual_res, ws.rel_kkt_error, ws.iter,
        0.0, 0.0, zero(T), 0, 0, ws.cg_iters, ws.polished, ws.status_polish,
        ws.setup_time, ws.update_time, ws.solve_time, ws.polish_time,
        (ws.first_run ? ws.setup_time : 0.0) + ws.update_time + ws.solve_time + ws.polish_time,
        Vector{T}(prim_cert), Vector{T}(dual_cert),
    )
end

function build_solution(ws::InteriorPointWorkspace{T}) where {T}
    prob = ws.prob
    n, m = prob.n, prob.m
    nan = T(NaN)
    if ws.status == PRIMAL_INFEASIBLE || ws.status == PRIMAL_INFEASIBLE_INACCURATE
        cert = prob.E .* ws.cert_y
        nc = norm_inf(cert)
        nc > zero(T) && (cert ./= nc)
        return ipm_solution(ws, fill(nan, n), fill(nan, m), T(Inf), nan, nan, cert, T[])
    elseif ws.status == DUAL_INFEASIBLE || ws.status == DUAL_INFEASIBLE_INACCURATE
        cert = prob.D .* ws.cert_x
        nc = norm_inf(cert)
        nc > zero(T) && (cert ./= nc)
        return ipm_solution(ws, fill(nan, n), fill(nan, m), T(-Inf), nan, nan, T[], cert)
    elseif !has_solution(ws.status)
        return ipm_solution(ws, fill(nan, n), fill(nan, m), nan, nan, nan, T[], T[])
    end
    x = prob.D .* ws.x
    y = (prob.E .* ws.y) ./ prob.c
    return ipm_solution(ws, x, y, ws.obj_val, ws.dual_obj_val, ws.duality_gap, T[], T[])
end
