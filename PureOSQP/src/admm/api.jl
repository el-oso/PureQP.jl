function update_settings!(ws::OperatorSplittingWorkspace{T}, alg::OperatorSplitting) where {T}
    old = ws.algorithm
    new = element_typed(alg, T, ws.options)
    # Compared field by field rather than by looping over a tuple of symbols: `getfield`
    # with a symbol the compiler cannot see is a dynamic call, and `--trim` rejects it.
    refactor_needed = new.rho != old.rho || new.sigma != old.sigma ||
        new.rho_is_vec != old.rho_is_vec
    ws.algorithm = new
    adopt_settings!(ws.linsys, new, ws.options)
    if refactor_needed
        # `σ` is held by value in the weights, so a new one needs a new weights object; the
        # vectors are shared and refilled in place by `set_rho_vec!`.
        wt = ws.weights
        ws.weights = SystemWeights(wt.w, wt.w_inv, new.sigma)
        set_rho_vec!(ws, new.rho)
        refactor!(ws)
    end
    return ws
end

"""
    update_rho!(ws, rho) -> ws

Set the workspace's `ρ` and refactorize. `rho` is clamped to `[1e-6, 1e6]` and then split
across the constraint classes exactly as adaptive `ρ` does, so this is the same operation
the solver performs on itself, made available to the caller.

`ws.algorithm.rho` keeps the value [`setup`](@ref) was given; the live value is `ws.rho`.
"""
function update_rho!(ws::OperatorSplittingWorkspace{T}, rho::Real) where {T}
    rho > 0 || throw(ArgumentError("rho must be positive, got $rho"))
    set_rho_vec!(ws, T(rho))
    refactor!(ws)
    return ws
end

"""
    constraint_violation!(out, ws) -> out

Write how far each row of `l ≤ Ax ≤ u` is from being satisfied, in the caller's units.

`out[i]` is `max(l[i] - (Ax)[i], (Ax)[i] - u[i], 0)`: zero where the row holds, and the
distance to the nearer bound where it does not. `‖out‖∞` is the primal residual
[`solve!`](@ref) reports as `prim_res`, so this is that number broken out by row.

The iterate `z` is projected into `[l, u]` every iteration and is feasible by construction;
`Ax` is what can miss, and is what this measures. A row that was one-sided on input has a
bound of `±1e30` here, far enough that it never reports a violation of its own.

`out` must have one entry per constraint row. Nothing is allocated.
"""
# `out`'s element type is not tied to the workspace's: the violations are computed in the
# workspace's own type and converted on assignment, so a caller may collect them in whatever
# vector it already holds. Tying the two also put this method outside the contract's slot, which
# is stated as `(::AbstractVector, ::Self)` and cannot match a signature whose two arguments
# share a parameter.
function constraint_violation!(out::AbstractVector, ws::OperatorSplittingWorkspace{T}) where {T}
    prob = ws.prob
    length(out) == prob.m || throw(
        DimensionMismatch("out must have one entry per constraint row")
    )
    iszero(prob.m) && return out
    mul_A!(ws.Ax, prob, ws.x)
    scaled = prob.scaling > 0
    for i in eachindex(out)
        # `l`, `u` and `Ax` are all equilibrated by the same row factor, so the violation
        # comes back to the caller's units by dividing it out once.
        gap = max(prob.l[i] - ws.Ax[i], ws.Ax[i] - prob.u[i], zero(T))
        out[i] = scaled ? gap / prob.E[i] : gap
    end
    return out
end

"""
    constraint_violation(ws) -> Vector

How far each row of `l ≤ Ax ≤ u` is from being satisfied, in the caller's units.

Allocates the result; [`constraint_violation!`](@ref) writes into a vector you supply and
carries the description of what the entries mean.
"""
function constraint_violation(ws::OperatorSplittingWorkspace{T}) where {T}
    return constraint_violation!(similar(ws.x, T, ws.prob.m), ws)
end
