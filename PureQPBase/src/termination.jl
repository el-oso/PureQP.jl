@inline norm_inf(v::AbstractVector{T}) where {T} = maximum(abs, v; init = zero(T))

"`max|s[i] v[i]|`. See `PureQPBase/src/elementwise.jl` on why there are two schedules."
@inline function scaled_norm_inf(s::Array{T}, v::Array{T}) where {T}
    r = zero(T)
    for i in paired(s, v)
        r = max(r, abs(s[i] * v[i]))
    end
    return r
end

@inline function scaled_norm_inf(s::AbstractVector{T}, v::AbstractVector{T}) where {T}
    return mapreduce((a, b) -> abs(a * b), max, s, v; init = zero(T))
end

"`max|v[i] / s[i]|`."
@inline function invscaled_norm_inf(s::Array{T}, v::Array{T}) where {T}
    r = zero(T)
    for i in paired(s, v)
        r = max(r, abs(v[i] / s[i]))
    end
    return r
end

@inline function invscaled_norm_inf(s::AbstractVector{T}, v::AbstractVector{T}) where {T}
    return mapreduce((a, b) -> abs(b / a), max, s, v; init = zero(T))
end

"""
    gap_terms(prob, y, Px, x) -> (xtPx, qtx, SCy)

The three terms of the duality gap at `(x, y)`: `xᵀPx`, `qᵀx`, and `uᵀmax(ŷ,0) + lᵀmin(ŷ,0)`
where `ŷ` is `y` projected onto the polar of the recession cone of `[l, u]`. Uses
`prob.tmp_m` as scratch for the projected `y`.
"""
function gap_terms(prob::Problem{T}, y::AbstractVector{T}, Px::AbstractVector{T}, x::AbstractVector{T}) where {T}
    xtPx = dot(Px, x)
    qtx = dot(prob.q, x)
    SCy = zero(T)
    if prob.m > 0
        copyto!(prob.tmp_m, y)
        project_polar_reccone!(prob.tmp_m, prob.l, prob.u)
        SCy = support_sum(prob.tmp_m, prob.l, prob.u)
    end
    return (xtPx, qtx, SCy)
end

"""
    support_sum(y, l, u) -> T

`uᵀ max(y, 0) + lᵀ min(y, 0)`, the support function of `[l, u]` at `y`.

Multipliers below the deadzone contribute nothing: a `1e-20` multiplier against a bound of
`1e20` is noise that would otherwise dominate the sum. `y` must already be projected onto
the polar recession cone, which is what keeps an infinite bound from meeting a nonzero
multiplier here.
"""
@inline function support_sum(y::Array{T}, l::Array{T}, u::Array{T}) where {T}
    dead = ZERO_DEADZONE(T)
    s = zero(T)
    for i in eachindex(y)
        v = y[i]
        abs(v) < dead && continue
        s += (v > zero(T) ? u[i] : l[i]) * v
    end
    return s
end

@inline function support_sum(y::AbstractVector{T}, l::AbstractVector{T}, u::AbstractVector{T}) where {T}
    dead = ZERO_DEADZONE(T)
    return mapreduce(
        (v, lo, hi) -> abs(v) < dead ? zero(T) : (v > zero(T) ? hi : lo) * v,
        +, y, l, u; init = zero(T)
    )
end

"""
    eps_prim(prob, s, z, Ax)

Tolerance for the primal residual test, relative to the larger of `‖z‖∞` and `‖Ax‖∞`. `s` is
the workspace's [`Options`](@ref); the three tolerances read `eps_abs`,
`eps_rel` and `scaled_termination` from it.
"""
function eps_prim(prob::Problem{T}, s, z::AbstractVector{T}, Ax::AbstractVector{T}) where {T}
    mx = if prob.scaling > 0
        max(invscaled_norm_inf(prob.E, z), invscaled_norm_inf(prob.E, Ax))
    else
        max(norm_inf(z), norm_inf(Ax))
    end
    return s.eps_abs + s.eps_rel * mx
end

"""
    eps_dual(prob, s, Aty, Px)

Tolerance for the dual residual test, relative to the largest of `‖q‖∞`, `‖Aᵀy‖∞` and
`‖Px‖∞`.
"""
function eps_dual(prob::Problem{T}, s, Aty::AbstractVector{T}, Px::AbstractVector{T}) where {T}
    mx = if prob.scaling > 0
        max(invscaled_norm_inf(prob.D, prob.q), invscaled_norm_inf(prob.D, Aty), invscaled_norm_inf(prob.D, Px)) / prob.c
    else
        max(norm_inf(prob.q), norm_inf(Aty), norm_inf(Px))
    end
    return s.eps_abs + s.eps_rel * mx
end

"""
    eps_duality_gap(prob, s, xtPx, qtx, SCy)

Tolerance for the duality-gap test, relative to the size of the terms that make up the
gap. Without the relative part a problem whose objective is `1e8` could never pass.
"""
function eps_duality_gap(prob::Problem{T}, s, xtPx::T, qtx::T, SCy::T) where {T}
    mx = max(abs(xtPx), abs(qtx), abs(SCy))
    # The stored terms are scaled; unscale unless termination is being judged scaled.
    (prob.scaling > 0 && !s.scaled_termination) && (mx /= prob.c)
    return s.eps_abs + s.eps_rel * mx
end

"The polar recession cone projection, elementwise."
@inline function polar_reccone(v::T, lo::T, hi::T, loose::T) where {T}
    if hi > loose
        return lo < -loose ? zero(T) : min(v, zero(T))
    elseif lo < -loose
        return max(v, zero(T))
    end
    return v
end

"""
    project_polar_reccone!(v, l, u)

Project `v` onto the polar of the recession cone of `[l, u]`, in place.
"""
function project_polar_reccone!(v::Array{T}, l::Array{T}, u::Array{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    for i in eachindex(v)
        v[i] = polar_reccone(v[i], l[i], u[i], loose)
    end
    return v
end

function project_polar_reccone!(v::AbstractVector{T}, l::AbstractVector{T}, u::AbstractVector{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    v .= polar_reccone.(v, l, u, loose)
    return v
end

"`uᵀ max(v, 0) + lᵀ min(v, 0)`, the support function evaluated without a deadzone."
@inline function support_plain(v::Vector{T}, l, u) where {T}
    s = zero(T)
    for i in eachindex(v)
        dy = v[i]
        s += u[i] * max(dy, zero(T)) + l[i] * min(dy, zero(T))
    end
    return s
end

@inline function support_plain(v::AbstractVector{T}, l, u) where {T}
    return mapreduce(
        (dy, lo, hi) -> hi * max(dy, zero(T)) + lo * min(dy, zero(T)),
        +, v, l, u; init = zero(T)
    )
end

"""
    is_primal_infeasible(prob, dy, eps) -> Bool

Certificate test on the caller-owned buffer `dy`: after projecting `dy` onto the polar of
the recession cone of `[l, u]`, the problem is primal infeasible when
`uᵀ max(dy,0) + lᵀ min(dy,0) < 0` and `‖Aᵀdy‖ < ε‖dy‖`. Projects `dy` in place, which then
becomes the certificate; never touches the workspace's actual `x` or `y`.

The support function is tested against zero rather than against `ε‖dy‖`. A tolerance there
admits directions that do not separate, and a certificate is a proof or it is nothing.
"""
function is_primal_infeasible(prob::Problem{T}, dy::AbstractVector{T}, eps::T) where {T}
    iszero(prob.m) && return false
    project_polar_reccone!(dy, prob.l, prob.u)
    ndy = prob.scaling > 0 ? scaled_norm_inf(prob.E, dy) : norm_inf(dy)
    ndy > DIVISION_TOL(T) || return false
    # Strict: the support function of the direction must be negative, not merely under a
    # tolerance that scales with the direction's own norm. A certificate is a proof or it is
    # nothing, and a tolerance here admits directions that do not separate.
    support_plain(dy, prob.l, prob.u) < zero(T) || return false
    mul_At!(prob.work_n, prob, dy)
    if prob.scaling > 0
        # mul_At! applies D; the unscaled test is Aᵀ(E ⊙ dy), so divide it back out once.
        divide!(prob.work_n, prob.work_n, prob.D)
    end
    return norm_inf(prob.work_n) < eps * ndy
end

"Whether `Aδx` leaves the recession cone of `[l, u]` at any row, to within `tol`."
@inline function leaves_reccone(v::Vector{T}, l, u, loose, tol) where {T}
    for i in eachindex(v)
        vi = v[i]
        u[i] < loose && vi > tol && return true
        l[i] > -loose && vi < -tol && return true
    end
    return false
end

@inline function leaves_reccone(v::AbstractVector{T}, l, u, loose, tol) where {T}
    return mapreduce(
        (vi, lo, hi) -> (hi < loose && vi > tol) || (lo > -loose && vi < -tol),
        |, v, l, u; init = false
    )
end

"""
    is_dual_infeasible(prob, dx, eps) -> Bool

Certificate test on the caller-owned buffer `dx`: `qᵀdx < 0`, `‖Pdx‖ < ε‖dx‖`, and `Adx` in
the recession cone of `[l, u]` to within `ε‖dx‖`. Only reads `dx`; never touches the
workspace's actual `x` or `y`.
"""
function is_dual_infeasible(prob::Problem{T}, dx::AbstractVector{T}, eps::T) where {T}
    scaled = prob.scaling > 0
    ndx = scaled ? scaled_norm_inf(prob.D, dx) : norm_inf(dx)
    cost = scaled ? prob.c : one(T)
    ndx > DIVISION_TOL(T) || return false
    # A strict sign test. Allowing `qᵀdx` up to `+ε‖dx‖` certifies a
    # direction that does not descend, and on an ill-conditioned `A` the near-null directions
    # clear the two remaining tests, so a bounded problem is declared unbounded.
    dot(prob.q, dx) < zero(T) || return false
    mul_P!(prob.work_n, prob, dx)
    scaled && divide!(prob.work_n, prob.work_n, prob.D)
    norm_inf(prob.work_n) < cost * eps * ndx || return false
    iszero(prob.m) && return true
    mul_A!(prob.work_m, prob, dx)
    scaled && divide!(prob.work_m, prob.work_m, prob.E)
    return !leaves_reccone(prob.work_m, prob.l, prob.u, INFTY(T) * MIN_SCALING(T), eps * ndx)
end

"""
    check_termination(ws, force = false) -> Status

The status a solve should stop with, or `UNSOLVED` to keep going. Each algorithm defines it
for its own workspace: the tolerances and the residuals are shared, the schedule of when to
test them is not. Declared here so both reach the same function.
"""
function check_termination end
