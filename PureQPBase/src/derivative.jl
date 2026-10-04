"""
Derivatives of the solution with respect to the problem data, by implicit
differentiation of the KKT conditions.

The solution is not given by a formula, so what is differentiated is the conditions that
define it. With the active set `𝒜` frozen, the inequalities on it become equalities and
the rest drop out, leaving

    G(z, θ) = [ P x + q + Ā' ν ;  Ā x - b ] = 0,   z = (x, ν),  θ = (P, q, A, l, u)

with `Ā = A[𝒜, :]`, `ν = y[𝒜]` and `b` the active bound of each row. Differentiating,

    dz = -M \\ (∂G/∂θ) dθ,     M = [ P  Ā' ; Ā  0 ]

so one solve with `M` gives the derivative, whatever the solver did to arrive there.
`M` carries no `σ`, no `ρ` and no `δ`: it is the exact equality-QP KKT matrix in problem
space, not the subproblem either algorithm iterates on, and not the regularized system
`polish!` factors.
"""

"""
    active_kkt(prob, x, y, z) -> (F, M, act, lower, x, y)

Factor `M = [P Ā'; Ā 0]` for the active set of the problem-space point `(x, y, z)`, and
return that point alongside the factorization.

Throws rather than returning anything when the derivative does not exist. See
[`adjoint_derivative`](@ref) for why there is no fallback.
"""
function active_kkt(
        prob::AnyProblem{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}
    ) where {T}
    require_entries(
        prob.P, prob.A, "differentiating the solution",
        "`adjoint_derivative` and `forward_derivative` need the active-set KKT matrix, which " *
            "has no matrix-free form: re-express the problem with a matrix `P` and `A` to " *
            "differentiate it."
    )
    n, m = prob.n, prob.m
    τ = sqrt(eps(T)) * max(norm_inf(y), one(T))
    act = Int[]
    lower = Bool[]
    for i in 1:m
        li, ui = prob.l0[i], prob.u0[i]
        if li == ui
            # An equality row is always active, but which bound the derivative belongs to
            # is still decided by the sign of the multiplier, exactly as for an inequality.
            # Widening the bound the row pushes against moves the solution; widening the
            # other one does not, since the row is not resting on it.
            push!(act, i)
            push!(lower, y[i] < zero(T))
        elseif abs(y[i]) > τ
            push!(act, i)
            push!(lower, y[i] < zero(T))
        else
            # A row sitting on a bound with a vanishing multiplier is weakly active: the
            # solution map is only directionally differentiable there, and which side the
            # active set falls on is decided by rounding error.
            # A one-sided row's absent bound is stored as `±INFTY`, a large finite number
            # that `isfinite` accepts. Left in, it both sets the scale of `gap` — making it
            # around `1e22`, which every row clears — and is compared against as if the row
            # could rest on it. Only the bounds the row actually has take part.
            inf = INFTY(T)
            has_l, has_u = li > -inf, ui < inf
            gap = sqrt(eps(T)) *
                max(one(T), has_l ? abs(li) : zero(T), has_u ? abs(ui) : zero(T))
            if (has_l && abs(z[i] - li) <= gap) || (has_u && abs(z[i] - ui) <= gap)
                throw(
                    ArgumentError(
                        "constraint row $i sits on its bound with multiplier $(y[i]): the QP " *
                            "is degenerate at this solution and the derivative does not exist. " *
                            "Perturb the data, or drop the redundant constraint."
                    )
                )
            end
        end
    end
    k = length(act)
    k <= n || throw(
        ArgumentError(
            "the active-set KKT matrix is singular: $k active rows for $n variables, so the " *
                "active constraint gradients cannot be independent."
        )
    )

    M = zeros(T, n + k, n + k)
    for j in 1:n, i in 1:n
        M[i, j] = T(prob.P[i, j])
    end
    for (r, i) in enumerate(act), j in 1:n
        a = T(prob.A[i, j])
        M[n + r, j] = a
        M[j, n + r] = a
    end
    F = bunchkaufman!(Symmetric(copy(M), :L); check = false)
    issuccess(F) || throw(
        ArgumentError(
            "the active-set KKT matrix is singular, so the derivative does not exist. The " *
                "active constraint gradients are dependent."
        )
    )
    return (F, M, act, lower, x, y)
end

"""
    derivative_ready(ws) -> Nothing

Throw unless `ws`'s iterate is one the active-set test can read.

The test calls a row active when its multiplier is far from zero. An algorithm whose
inactive-row multipliers settle somewhere else owes the caller a refusal here, because the
derivative that follows is wrong rather than imprecise. `OperatorSplittingWorkspace`
projects its multipliers, so it has nothing to check; `InteriorPointWorkspace` holds
inactive rows at the barrier parameter and requires polishing.
"""
derivative_ready(ws::QPWorkspace) = nothing

"""
    active_kkt(ws) -> (F, M, act, lower, x, y)

Check that `ws` holds a converged, host-resident solution, unscale it into problem space,
and delegate to the `active_kkt(prob, x, y, z)` method above. One method serves every
algorithm's workspace, since each holds its iterate the same way, scaled by the same `D`,
`E`, `c`. An algorithm whose iterate the active-set test cannot read says so through
[`derivative_ready`](@ref).
"""
function active_kkt(ws::QPWorkspace{T}) where {T}
    # The derivative is of the solution map at a solution. An unconverged point is not one,
    # and an infeasible run has already been cold started, so differentiating either returns
    # a number for a question that was not asked.
    ws.status === SOLVED || throw(
        ArgumentError(
            "the derivative is taken at a solution, and this workspace's last solve ended " *
                "as $(status_name(ws.status)). Solve it first, or tighten the tolerances " *
                "until it converges."
        )
    )
    require_host(ws.x, "differentiating the solution")
    prob = ws.prob
    # An operator that cannot be materialized was never a candidate for polishing either, so
    # that refusal takes priority: it names the actual obstacle, where the check below would
    # otherwise blame a workspace that had no way to satisfy it.
    require_entries(
        prob.P, prob.A, "differentiating the solution",
        "`adjoint_derivative` and `forward_derivative` need the active-set KKT matrix, which " *
            "has no matrix-free form: re-express the problem with a matrix `P` and `A` to " *
            "differentiate it."
    )
    derivative_ready(ws)
    # An algorithm that works on the caller's problem directly holds a `QPData` and has no
    # equilibration to undo, which is what the second method of this is for.
    x, y, z = unscaled_iterates(prob, ws.x, ws.y, ws.z)
    return active_kkt(prob, x, y, z)
end

"""
    kkt_solve(F, M, r) -> w

Solve `M w = r` and refuse the answer if it is not one. A backward-stable factorization
reports success on a matrix that is merely very ill-conditioned, and then `‖w‖` blows up
while `‖r‖` does not, so the relative residual is a condition estimate costing one
matrix-vector product. Refusing above `sqrt(eps)` rejects `cond(M) ≳ 1e8`.
"""
function kkt_solve(F, M::AbstractMatrix{T}, r::AbstractVector{T}) where {T}
    w = F \ r
    res = norm_inf(M * w - r)
    res <= sqrt(eps(T)) * max(norm_inf(r), one(T)) || throw(
        ArgumentError(
            "the active-set KKT matrix is too ill-conditioned to differentiate through " *
                "(relative residual $res). The active constraints are near-dependent, and a " *
                "regularized answer here would be a plausible wrong number, so it is refused."
        )
    )
    return w
end

"""
    adjoint_derivative(ws, dx, dy) -> (; dP, dq, dA, dl, du)

Gradients of a scalar loss with respect to the problem data, given its gradients
`dx = ∂L/∂x` and `dy = ∂L/∂y` at the solution the workspace currently holds.

Solving `M w = -[dx; dy[𝒜]]` once gives all five:

    ∇P = ½(wₓ xᵀ + x wₓᵀ),   ∇q = wₓ,   ∇A = y wₓᵀ + ŵ xᵀ
    ∇l = -ŵ on rows active at their lower bound,  ∇u = -ŵ on rows active at their upper

where `ŵ` is the tail of `w` scattered back into the active rows. `∇A` has two terms
because `A` enters both blocks of `G`; dropping the cross term is the usual error.

There is no fallback when the derivative does not exist — a degenerate active set, or a
singular or near-singular `M`. A least-squares answer would carry the right shape and
units while being a different quantity, and nothing downstream could tell the difference.
`polish!` is allowed a regularized guess because it measures whether the guess helped; a
gradient consumer has no such test.
"""
function adjoint_derivative(
        ws::QPWorkspace{T}, dx::AbstractVector, dy::AbstractVector
    ) where {T}
    n, m = ws.prob.n, ws.prob.m
    length(dx) == n || throw(ArgumentError(lazy"length(dx) = $(length(dx)) must equal n = $n"))
    length(dy) == m || throw(ArgumentError(lazy"length(dy) = $(length(dy)) must equal m = $m"))

    F, M, act, lower, x, y = active_kkt(ws)
    k = length(act)
    r = zeros(T, n + k)
    for i in 1:n
        r[i] = -T(dx[i])
    end
    for (rix, i) in enumerate(act)
        r[n + rix] = -T(dy[i])
    end
    w = kkt_solve(F, M, r)
    wx = w[1:n]

    scattered = zeros(T, m)
    for (rix, i) in enumerate(act)
        scattered[i] = w[n + rix]
    end

    dP = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        dP[i, j] = (wx[i] * x[j] + x[i] * wx[j]) / 2
    end
    dA = y * wx' + scattered * x'
    dl = zeros(T, m)
    du = zeros(T, m)
    for (rix, i) in enumerate(act)
        if lower[rix]
            dl[i] = -scattered[i]
        else
            du[i] = -scattered[i]
        end
    end
    return (; dP, dq = wx, dA, dl, du)
end

"""
    forward_derivative(ws; dP, dq, dA, dl, du) -> (dx, dy)

Directional derivative of the solution along the given perturbation of the data. Omitted
arguments are zero. Solves the same `M` as [`adjoint_derivative`](@ref), against

    M [Δx; Δν] = -[ΔP x + Δq + ΔĀᵀ ν ;  ΔĀ x - Δb]

and scatters `Δν` back into the active rows to give `dy`.
"""
function forward_derivative(
        ws::QPWorkspace{T}; dP = nothing, dq = nothing, dA = nothing,
        dl = nothing, du = nothing
    ) where {T}
    n, m = ws.prob.n, ws.prob.m
    F, M, act, lower, x, y = active_kkt(ws)
    k = length(act)

    r = zeros(T, n + k)
    if !isnothing(dP)
        for j in 1:n, i in 1:n
            r[i] -= T(dP[i, j]) * x[j]
        end
    end
    if !isnothing(dq)
        for i in 1:n
            r[i] -= T(dq[i])
        end
    end
    if !isnothing(dA)
        for (rix, i) in enumerate(act)
            yi = y[i]
            s = zero(T)
            for j in 1:n
                aij = T(dA[i, j])
                r[j] -= aij * yi
                s += aij * x[j]
            end
            r[n + rix] -= s
        end
    end
    for (rix, i) in enumerate(act)
        db = if lower[rix]
            isnothing(dl) ? zero(T) : T(dl[i])
        else
            isnothing(du) ? zero(T) : T(du[i])
        end
        r[n + rix] += db
    end

    w = kkt_solve(F, M, r)
    dxs = w[1:n]
    dys = zeros(T, m)
    for (rix, i) in enumerate(act)
        dys[i] = w[n + rix]
    end
    return (dxs, dys)
end
