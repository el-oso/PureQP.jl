"Row classes of the interior-point method: a free row has no bound, an equality row `l == u`."
const ROW_FREE = Int8(-1)
const ROW_INEQUALITY = Int8(0)
const ROW_EQUALITY = Int8(1)

"""
    classify_rows!(rclass, has_l, has_u, prob) -> n_sides

Set every row's class and side masks from `prob`'s current bounds, and return the number of
inequality sides across all rows. Used at workspace construction and by [`update!`](@ref)
whenever `l` or `u` changes.
"""
function classify_rows!(rclass::AbstractVector{Int8}, has_l::AbstractVector{Bool}, has_u::AbstractVector{Bool}, prob::Problem{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    sides = 0
    for i in eachindex(rclass)
        if prob.l0[i] == prob.u0[i]
            rclass[i] = ROW_EQUALITY
            has_l[i] = false
            has_u[i] = false
        elseif prob.l[i] < -loose && prob.u[i] > loose
            rclass[i] = ROW_FREE
            has_l[i] = false
            has_u[i] = false
        else
            rclass[i] = ROW_INEQUALITY
            has_l[i] = prob.l[i] > -loose
            has_u[i] = prob.u[i] < loose
            sides += has_l[i] + has_u[i]
        end
    end
    return sides
end

"""
    InteriorPointWorkspace{T,MP,MA,V,VI,VB,LS} <: QPWorkspace{T}

Solver state of [`InteriorPoint`](@ref), built by [`setup`](@ref).
The problem is the equilibrated one, and every iterate is in scaled space.

An inequality row `i` carries a slack and a multiplier for each finite side:
`s_l = Ãx − l̃` and `z_l` when `has_l[i]`, `s_u = ũ − Ãx` and `z_u` when `has_u[i]`, and its
multiplier is `y = z_u − z_l`. An absent side holds `s = 1`, `z = 0`, so the elementwise
loops need no branch beyond the mask. An equality row carries a free multiplier `y`; a free
row has `y = 0`.

`seeded` says whether `x` and `y` are a starting point: a solve that ends with a point
([`has_solution`](@ref)) and [`warm_start!`](@ref) set it; a solve that ends without one,
[`cold_start!`](@ref) and `warm_starting = false` clear it. After a solve without a point, `x`
and `y` still hold the last iterate.
"""
mutable struct InteriorPointWorkspace{
        T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T},
        VI <: AbstractVector{Int8}, VB <: AbstractVector{Bool}, LS <: LinearSystem,
    } <: QPWorkspace{T}
    # Not `const`: `update!` replaces `P` or `A` by handing back another problem around the
    # same vectors, which is what an immutable problem costs and all it costs.
    prob::Problem{T, MP, MA, V}
    const linsys::LS
    # `sigma` is the current `reg_primal`, so a regularization bump replaces the object; `w`
    # and `w_inv` are rewritten in place every outer iteration.
    weights::SystemWeights{T, V}
    const rclass::VI
    const has_l::VB
    const has_u::VB
    # Inequality sides across every row; recomputed by `update!` whenever `l` or `u` changes
    # a row's class.
    n_sides::Int
    const x::V
    const y::V
    const s_l::V
    const s_u::V
    const z_l::V
    const z_u::V
    const Ax::V
    const Px::V
    const Aty::V
    # `clamp(Ãx, l̃, ũ)`, the feasible point the primal residual is measured against.
    const z::V
    # Newton residuals: `r_d = P̃x + q̃ + Ãᵀy`; `r_l = Ãx − l̃ − s_l` on a lower side and
    # `Ãx − l̃` on an equality row; `r_u = ũ − Ãx − s_u` on an upper side; zero elsewhere.
    const r_d::V
    const r_l::V
    const r_u::V
    # The complementarity terms of the right-hand side, zero on absent sides.
    const rc_l::V
    const rc_u::V
    const rhs_x::V
    const rhs_z::V
    const dx::V
    const dy::V
    const ds_l::V
    const ds_u::V
    const dz_l::V
    const dz_u::V
    const Adx::V
    # Refinement residual and correction.
    const res_x::V
    const res_z::V
    const corr_x::V
    const corr_y::V
    # Infeasibility certificate candidates. The tests project them in place, and the one that
    # passes is what the solution reports.
    const cert_x::V
    const cert_y::V
    # The regularization in force: the algorithm's values times ten per bump in this solve.
    reg_primal::T
    reg_dual::T
    reg_bumps::Int
    # The backend's factorization was built with a `sigma` other than the current one, so the
    # next factorization must be a full one.
    sigma_changed::Bool
    # Guards: consecutive steps shorter than `STALL_STEP`, consecutive iterations whose merit
    # (`mu`, or `rnorm` without an inequality side) did not fall, the previous merit, whether
    # the certificate tests now run every iteration, and whether the iterate has passed the
    # divergence ceiling.
    short_steps::Int
    flat_merit::Int
    last_merit::T
    alert::Bool
    diverged::Bool
    mu::T
    rnorm::T
    alpha::T
    prim_res::T
    dual_res::T
    scaled_prim_res::T
    scaled_dual_res::T
    obj_val::T
    dual_obj_val::T
    duality_gap::T
    scaled_duality_gap::T
    xtPx::T
    qtx::T
    SCy::T
    rel_kkt_error::T
    cg_iters::Int
    # Consecutive Newton solves the backend reported as missed (see `last_solve_converged`).
    cg_misses::Int
    # Newton solves reported as missed anywhere in this run, for the verbose footer.
    cg_total_misses::Int
    iter::Int
    status::Status
    seeded::Bool
    first_run::Bool
    polished::Bool
    status_polish::PolishStatus
    setup_time::Float64
    # Accumulated across the `update!` calls made since the previous solve, and charged to
    # the next one; reset once reported, as in `OperatorSplittingWorkspace`.
    update_time::Float64
    solve_time::Float64
    polish_time::Float64
    algorithm::InteriorPoint{T, T, T, Int}
    options::Options{T}
    # Where the result is unscaled, in the workspace's own array type. The `Solution` holds
    # `Vector`s, so an array that forbids scalar indexing is scaled here and copied across
    # once rather than indexed element by element.
    xout::V
    yout::V
    # Refilled and handed back by every solve, so a solve allocates nothing at all. Its
    # arrays are plain `Vector`s whatever the workspace was built from: the result is what a
    # caller reads, not a buffer the solver iterates on. `Solution` says what reuse means
    # for a caller holding one across a solve.
    sol::Solution{T}
end

function Base.show(io::IO, ws::InteriorPointWorkspace)
    print(
        io, "PureIPM InteriorPointWorkspace: ", ws.prob.n, "×", ws.prob.m,
        ", backend ", backend_name(ws.linsys),
        ", status ", status_name(ws.status),
    )
    return nothing
end

"""
    ipm_workspace(ls, prob, wt, algorithm, options) -> InteriorPointWorkspace

Classify the rows of `prob` and allocate the interior-point state around the backend `ls`,
which solves through the weights object `wt`.
"""
function ipm_workspace(
        ls::LinearSystem, prob::Problem{T}, wt::SystemWeights{T}, algorithm::InteriorPoint{T, T, T, Int},
        options::Options{T}
    ) where {T}
    n, m, q0 = prob.n, prob.m, prob.q0
    buf(k) = fill!(similar(q0, T, k), zero(T))
    rclass = fill!(similar(q0, Int8, m), ROW_INEQUALITY)
    has_l = fill!(similar(q0, Bool, m), false)
    has_u = fill!(similar(q0, Bool, m), false)
    sides = classify_rows!(rclass, has_l, has_u, prob)
    ws = InteriorPointWorkspace{T, typeof(prob.P), typeof(prob.A), typeof(q0), typeof(rclass), typeof(has_l), typeof(ls)}(
        prob, ls, wt, rclass, has_l, has_u, sides,
        buf(n), buf(m), buf(m), buf(m), buf(m), buf(m),
        buf(m), buf(n), buf(n), buf(m),
        buf(n), buf(m), buf(m), buf(m), buf(m),
        buf(n), buf(m), buf(n), buf(m), buf(m), buf(m), buf(m), buf(m), buf(m),
        buf(n), buf(m), buf(n), buf(m),
        buf(n), buf(m),
        algorithm.reg_primal, algorithm.reg_dual, 0, false,
        0, 0, zero(T), false, false,
        zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T),
        0, 0, 0, 0, UNSOLVED, false, true, false, POLISH_NOT_PERFORMED, 0.0, 0.0, 0.0, 0.0,
        algorithm, options,
        buf(n), buf(m),
        # The certificates are reserved rather than sized: only one of them is reported, and
        # only by a run that ends infeasible, so their lengths are what a solve sets.
        empty_solution(
            Vector{T}(undef, n), Vector{T}(undef, m), reserved(T, m), reserved(T, n)
        ),
    )
    return ws
end

function refuse_ipm_operators()
    throw(
        ArgumentError(
            "InteriorPoint() factors a matrix built from the entries of P and A, and one of " *
                "them declares `PureQPBase.is_materializable` false: it supplies products only. " *
                "Pass matrices, pass linsys = :indirect with a caller-supplied preconditioner " *
                "and scaling = 0, or use OperatorSplitting()."
        )
    )
end

"Whether `M` is a preconditioner the caller built, rather than `nothing` or a built-in one."
caller_preconditioner(M) = !(M isa Union{Nothing, IdentityPreconditioner, JacobiPreconditioner})

probing(M) = M isa ProductOperator && M.probe

function setup_backend(
        alg::InteriorPoint, ::Val{LS}, ::Type{T}, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, options::Options,
        preconditioner, accelerator
    ) where {LS, T <: Real}
    t0 = time_ns()
    isnothing(accelerator) || throw(
        ArgumentError(
            "accelerator is used only by OperatorSplitting: the interior-point method has no " *
                "fixed-point iteration to accelerate."
        )
    )
    nv, mv = validate(P, q, A, l, u)
    algorithm = element_typed(alg, T, options)
    isnothing(preconditioner) || LS === :indirect || throw(
        ArgumentError(
            "preconditioner is used only by the matrix-free backend: pass linsys = :indirect with it."
        )
    )
    LS === :indirect && !caller_preconditioner(preconditioner) && throw(
        ArgumentError(
            "the interior-point method uses conjugate gradients only with a caller-supplied " *
                "preconditioner; measured without one, or with the Jacobi diagonal, it does not " *
                "reach the tolerance on most problems. Pass one, choose a direct linsys, or use " *
                "OperatorSplitting()."
        )
    )
    LS === :indirect || (is_materializable(P) && is_materializable(A)) || refuse_ipm_operators()
    if LS === :indirect
        iszero(options.scaling) || throw(
            ArgumentError(
                "a caller-supplied preconditioner approximates P + reg_primal*I + A' * Diagonal(w) * A " *
                    "for the P and A passed to setup, which equilibration would change: pass scaling = 0 with it."
            )
        )
        (probing(P) || probing(A)) && throw(
            ArgumentError(
                "a caller-supplied preconditioner approximates the operators passed to setup, " *
                    "and probe = true exists only to equilibrate them: build them without probe."
            )
        )
    end
    LS === :kronecker && throw(
        ArgumentError(
            "linsys = :kronecker is not available with InteriorPoint(): the Kronecker " *
                "backend needs the same weight on every row, and the interior-point weights " *
                "differ from row to row. Choose another linsys, or use OperatorSplitting()."
        )
    )
    LS === :lowrank && throw(
        ArgumentError(
            "linsys = :lowrank is not available with InteriorPoint(): it solves the reduced " *
                "matrix, and an active row's weight reaches 1/reg_dual, so forming that matrix " *
                "loses the accuracy the method needs. Leave linsys = :auto, which serves the " *
                "pair with the full KKT system, or use OperatorSplitting()."
        )
    )
    is_convex(T, P, algorithm.reg_primal) || throw(
        ArgumentError(
            "P + reg_primal*I is not positive definite: P is indefinite, so the problem is not convex."
        )
    )
    prob = validated_problem(T, nv, mv, P, q, A, l, u, options.scaling)
    n, m, q0 = prob.n, prob.m, prob.q0
    # Unit weights are the starting-point system, so a rung that decides by factoring leaves
    # the first factorization of a solve in place.
    wt = SystemWeights(fill!(similar(q0, T, m), one(T)), fill!(similar(q0, T, m), one(T)), algorithm.reg_primal)
    sel = IPMSelection()
    # `named_backend` holds the branch per named kind for both algorithms; `:kronecker` and
    # `:lowrank` are refused above, before the problem is built, so its branches for them are
    # unreachable from here.
    named = named_backend(Val(LS), P, A, prob, wt, sel, preconditioner)
    ls = isnothing(named) ? first(choose_backend(P, A, prob, wt, sel)) : first(named)
    ws = ipm_workspace(ls, prob, wt, algorithm, options)
    if LS === :indirect
        adopt_settings!(ws.linsys, algorithm, options)
        use_residual_stop!(ws.linsys, true)
    end
    ws.setup_time = (time_ns() - t0) / 1.0e9
    return ws
end

"""
    select_backend(P, A, prob, wt, sel::IPMSelection) -> (LinearSystem, Bool)

The interior-point ladder: the sparse KKT factorization first, then the sparse reduced one
and the structured reduced backends, and [`FullKKT`](@ref) as the terminal for any
materializable pair. Which of the two sparse forms serves a `SparseMatrixCSC` pair, and
whether either does, is one question asked of the sparsity pattern; both rungs consult that
one answer. The Kronecker rung is absent, since it needs uniform weights; the low-rank rung
declines (see its [`IPMSelection`](@ref) method); and [`formed_rung`](@ref) is absent, since
its inverse would be rebuilt every outer iteration for a handful of solves.
"""
function select_backend(P, A, prob, wt, sel::IPMSelection)
    rung = kkt_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = reduced_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = block_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = lowrank_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = dense_rung(P, A, prob, sel)
    isnothing(rung) || return rung
    return indirect_rung(P, A, prob, sel)
end

"""
    dense_rung(P, A, prob, sel::IPMSelection) -> (LinearSystem, Bool) or nothing

The interior-point terminal: [`FullKKT`](@ref) when LAPACK serves the element type. The
reduced matrix it avoids is inverted once per outer iteration for a handful of solves, and
its weights reach `1/δ_d`. Other element types get [`ReducedCholesky`](@ref), since
`bunchkaufman!` has no generic method.
"""
function dense_rung(P::AbstractMatrix, A::AbstractMatrix, prob::Problem{T}, sel::IPMSelection) where {T}
    (is_materializable(P) && is_materializable(A)) || return nothing
    T <: LinearAlgebra.BlasFloat && return (FullKKT(prob.q0, prob.n, prob.m), false)
    return (ReducedCholesky(prob.q0, prob.n, prob.m), false)
end

dense_rung(P, A, prob, sel::IPMSelection) = nothing

"""
    lowrank_rung(P::Diagonal, A::RowCoupled, prob, wt, sel::IPMSelection) -> nothing

Declines, so the pair reaches [`FullKKT`](@ref).

[`DiagonalLowRank`](@ref) solves the reduced matrix `P̃ + δ_p I + Ãᵀ diag(w) Ã`, and forming
that product is what this method declines, not the Woodbury identity that solves it. An
active row's weight reaches `1/δ_d`, so the rank-`k` correction arrives orders of magnitude
above the diagonal core it corrects, and the small directions are rounded away as the matrix
is built. Measured on the low-rank families of `bench/structured_problems.jl` at their
converged weights, against a reference in extended precision: the reduced matrix solved
through Woodbury reaches `9e-9` and the same matrix factored densely `9e-8`, where the
augmented factorization this decline routes to reaches `9e-16`
(`PureIPM/bench/ipm_lowrank_terminal.jl`). The loss is the reduced form's, and a structured
backend that also reduces cannot avoid it.

`linsys = :lowrank` is refused outright for `InteriorPoint()` before this is ever reached
(see `setup_backend`), so `require_crossover` is accepted only for signature parity with the
generic method.
"""
lowrank_rung(P::Diagonal, A::RowCoupled, prob, wt, sel::IPMSelection; require_crossover::Bool = true) = nothing

"""
    indirect_rung(P, A, prob, sel::IPMSelection)

Refuses: the interior-point method runs the matrix-free backend only when it is named, with
a caller-supplied preconditioner.
"""
indirect_rung(P, A, prob, sel::IPMSelection) = refuse_ipm_operators()

"""
    warm_start!(ws::InteriorPointWorkspace; x = nothing, y = nothing)

Seed the next solve's starting point in problem space. Slacks and multipliers are rebuilt
from `x` and `y` when the solve starts.
"""
function warm_start!(ws::InteriorPointWorkspace{T}; x = nothing, y = nothing) where {T}
    prob = ws.prob
    if !isnothing(x)
        length(x) == prob.n || throw(ArgumentError("length(x) must be $(prob.n)"))
        all(isfinite, x) || throw(ArgumentError("x must be finite, found NaN or Inf"))
        ws.x .= T.(x) ./ prob.D
    end
    if !isnothing(y)
        length(y) == prob.m || throw(ArgumentError("length(y) must be $(prob.m)"))
        all(isfinite, y) || throw(ArgumentError("y must be finite, found NaN or Inf"))
        ws.y .= prob.c .* T.(y) ./ prob.E
    end
    ws.seeded = true
    return ws
end

"""
    cold_start!(ws::InteriorPointWorkspace) -> ws

Zero `x` and `y` and clear `seeded`, so the next solve computes its own starting point.
"""
function cold_start!(ws::InteriorPointWorkspace{T}) where {T}
    fill!(ws.x, zero(T))
    fill!(ws.y, zero(T))
    ws.seeded = false
    return ws
end

"""
    update!(ws::InteriorPointWorkspace; q, l, u, P, A) -> ws

Replace problem data in an existing interior-point workspace: the validation and adoption
[`update!`](@ref) does for any workspace, checked against convexity at `reg_primal`. A row whose bounds move into or out of equality or freeness is
reclassified; `s_l`, `s_u`, `z_l` and `z_u` are left as they are, since `starting_point!`
rebuilds them from `x`, `y` and the current classes at the next solve.

No factorization happens here: every outer iteration factorizes the Newton system at its own
weights regardless, and a solve resets the regularization from the algorithm parameters
before its first one, so a later solve picks up the new data whether or not `P` or `A`
changed.
"""
function update!(
        ws::InteriorPointWorkspace{T}; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing
    ) where {T}
    t0 = time_ns()
    prob = ws.prob
    validate_update!(prob, ws.linsys; P, A, q, l, u)
    !isnothing(P) && !is_convex(T, P, ws.algorithm.reg_primal) &&
        throw(
        ArgumentError(
            "P + reg_primal*I is not positive definite: P is indefinite, so the problem is not convex."
        )
    )
    ws.prob = adopt_update!(prob; P, A, q, l, u)
    prob = ws.prob
    if !isnothing(l) || !isnothing(u)
        ws.n_sides = classify_rows!(ws.rclass, ws.has_l, ws.has_u, prob)
    end
    ws.update_time += (time_ns() - t0) / 1.0e9
    return ws
end

# No refactorization: a solve resets the regularization from the algorithm parameters before
# its first iteration and refactorizes every iteration after.
function update_settings!(ws::InteriorPointWorkspace{T}, alg::InteriorPoint) where {T}
    new = element_typed(alg, T, ws.options)
    ws.algorithm = new
    adopt_settings!(ws.linsys, new, ws.options)
    return ws
end

# Inactive rows carry a multiplier of the size of the final barrier parameter rather than
# zero, which the active-set test cannot tell apart from a row that is genuinely active.
# Polishing recomputes the point from the guessed active set, which restores the distinction.
function derivative_ready(ws::InteriorPointWorkspace)
    ws.polished || throw(
        ArgumentError(
            "the derivative of an interior-point solution needs a polished workspace: its " *
                "inactive-row multipliers sit at the barrier parameter rather than at zero, " *
                "which the active-set test cannot tell apart from a genuinely active row. " *
                "Polishing ended as " * polish_status_name(ws.status_polish) *
                ": solve with polishing = true, and if it is already on, tighten the " *
                "tolerances so polishing has a clean active set to work from."
        )
    )
    return nothing
end
