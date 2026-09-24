"""
    PureOSQP

Pure-Julia implementation of the OSQP operator-splitting solver for

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

`P` and `A` may be any `AbstractMatrix`; there is no sparse-matrix dependency and the
caller's matrices are never modified. The algorithm follows Stellato, Banjac, Goulart,
Bemporad and Boyd, *OSQP: an operator splitting solver for quadratic programs*,
Mathematical Programming Computation 12(4):637–672, 2020.

The problem representation, the linear-system backends and the `QPAlgorithm`/`QPWorkspace`
contracts live in PureQPBase.jl, re-exported here in full; this package adds
[`OperatorSplitting`](@ref), which the five-argument `setup` and `solve` run by default.
PureIPM.jl supplies an interior-point method for the same problem, and the two can be loaded
together.
"""
module PureOSQP

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify
using StrictMode: @assert_noalloc, @assert_trim_compatible
using PureQPBase

import PureQPBase:
    setup, solve, solve!, warm_start!, cold_start!, update!, update_settings!, update_rho!,
    constraint_violation, derivative_ready, setup_backend, algorithm_defaults, element_typed,
    recommend_linsys, OPTION_NAMES,
    ADMMSelection, IPMSelection, SelectionFor, BlockDiagonal, BlockReduced, DiagonalLowRank,
    DiagonalReduced, KroneckerOperator, KroneckerReduced, TridiagonalReduced, Problem,
    ProductOperator, RowCoupled, SystemWeights, LINSYS_OPTIONS,
    INFTY, MIN_SCALING, RHO_MIN, RHO_MAX, RHO_TOL, RHO_EQ_OVER_INEQ, DIVISION_TOL,
    active_kkt, accelerator_reset!, add!, adopt_settings!, adopt_update!, block_rung,
    check_finite, check_storage, choose_backend, dense_rung,
    eps_prim, eps_dual, eps_duality_gap, factorize!, factors, formed_rung, gap_terms,
    increment!, indirect_backend, indirect_rung, inner_iterations, invscaled_norm_inf,
    is_convex, is_dual_infeasible, is_materializable, is_primal_infeasible, is_scalar_multiple,
    is_symmetric, kkt_rung, kronecker_rung, last_solve_converged, lowrank_rung, mul_A!,
    mul_At!, mul_P!, multiply!, named_backend, norm_inf, polish_kernel!, reduced_diagonal!,
    reduced_rung, refactor!, refactor_rho!, refactored!, refactor_weights!, scalar_multiple,
    scale_subtract!, select_backend, set_refresh_index!, set_tolerance_level!,
    solve_multiplier!, solve_system!, structural_rows, subtract!, subtract_scaled!,
    update_x!, update_zy!, use_residual_stop!, validate, validate_update!, validated_problem,
    polish_status_name, check_termination, polish!

export setup, solve, solve!, update!, update_settings!, update_rho!, warm_start!, cold_start!
export dimensions, capabilities, constraint_violation, constraint_violation!
export Optimizer, Solution, Status, Options, default_options
export QPAlgorithm, OperatorSplitting
export QPWorkspace, OperatorSplittingWorkspace
export has_solution, status_name
export recommend_linsys, LinsysAdvice
export backend_info, backend_name, factor_fill, BackendInfo
export PolishStatus
export adjoint_derivative, forward_derivative
export LinearSystem, ReducedCholesky, FullKKT
export Preconditioner, IdentityPreconditioner, JacobiPreconditioner, update_preconditioner!
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED, NUMERICAL_ERROR
export PolishStatus, POLISH_SUCCESS, POLISH_FAILED, POLISH_NOT_PERFORMED
export POLISH_NO_ACTIVE_SET_FOUND, POLISH_LINSYS_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

include("admm/algorithm.jl")
include("admm/accelerate.jl")
include("admm/rho.jl")
include("admm/termination.jl")
include("admm/admm.jl")
include("admm/polish.jl")
include("admm/update.jl")
include("admm/api.jl")

"""
    setup(P, q, A, l, u; kwargs...) -> QPWorkspace
    solve(P, q, A, l, u; x0 = nothing, y0 = nothing, kwargs...) -> Solution
    recommend_linsys(P, q, A, l, u; max_iter = 25, repeats = 3, kwargs...) -> LinsysAdvice

The five-argument forms of [`setup`](@ref), [`solve`](@ref) and [`recommend_linsys`](@ref),
running [`OperatorSplitting`](@ref) with its defaults. Pass an algorithm object as the sixth
positional argument — an `OperatorSplitting(; kwargs...)` with non-default parameters, or
another algorithm such as PureIPM.jl's `InteriorPoint()` — to choose otherwise.
"""
# `@constprop :aggressive` so a keyword given here still reaches `setup_backend`'s `Val` as a
# constant: the six-argument methods carry the same annotation, but propagation through a
# call chain needs every frame in it annotated or inlined, not just the frame that uses the
# constant.
Base.@constprop :aggressive setup(
    P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
    l::AbstractVector, u::AbstractVector; kwargs...
) = setup(P, q, A, l, u, OperatorSplitting(); kwargs...)

Base.@constprop :aggressive setup(
    ::Type{T}, P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
    l::AbstractVector, u::AbstractVector; kwargs...
) where {T <: Real} = setup(T, P, q, A, l, u, OperatorSplitting(); kwargs...)

Base.@constprop :aggressive solve(
    P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
    l::AbstractVector, u::AbstractVector; kwargs...
) = solve(P, q, A, l, u, OperatorSplitting(); kwargs...)

recommend_linsys(
    P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
    l::AbstractVector, u::AbstractVector; kwargs...
) = recommend_linsys(P, q, A, l, u, OperatorSplitting(); kwargs...)

"""
    Optimizer(; kwargs...)

MathOptInterface optimizer for [`OperatorSplitting`](@ref), available once MathOptInterface
is loaded. Keyword arguments are the fields of [`Options`](@ref) and the parameters of
[`OperatorSplitting`](@ref); each raw attribute is checked by name against those when it is
set. PureIPM.jl's `Optimizer` is the interior-point counterpart.

The wrapper lives in a package extension, so it costs nothing to a caller who does not use
it; this name is the only part of it this package owns.
"""
function Optimizer end

# The calls ADMM makes every iteration, checked on a problem small enough to solve here over
# every backend PureQPBase defines. `adapt_rho!` refactorizes whenever `ρ` moves. `solve!`
# itself carries no allocation claim: it returns a `Solution` holding unscaled copies of `x`
# and `y`. These checks report rather than throw; `test/strictmode_tests.jl` proves the same
# signatures with StrictModeTest.
let
    function check(ws)
        solve!(ws)
        @assert_noalloc accelerate_pre!(ws.accel, ws, 1)
        @assert_trim_compatible accelerate_pre!(ws.accel, ws, 1)
        @assert_noalloc admm_step!(ws)
        @assert_trim_compatible admm_step!(ws)
        @assert_noalloc accelerate_post!(ws.accel, ws, 1)
        @assert_trim_compatible accelerate_post!(ws.accel, ws, 1)
        @assert_noalloc update_residuals!(ws)
        @assert_trim_compatible update_residuals!(ws)
        @assert_noalloc check_termination(ws, false)
        @assert_trim_compatible check_termination(ws, false)
        @assert_noalloc adapt_rho!(ws)
        @assert_trim_compatible adapt_rho!(ws)
        return nothing
    end
    function run(linsys, P, A; scaling = 10)
        n, m = size(A, 2), size(A, 1)
        q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, m), fill(1.0, m)
        return check(setup(P, q, A, l, u; linsys, scaling))
    end
    P = [4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0]
    A = [1.0 1.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 -1.0 0.0]
    D = Diagonal([4.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    Ad = Diagonal([1.0, 0.5, 2.0, 1.0, 1.5, 0.5])
    run(:dense, P, A)
    run(:kkt, P, A)
    run(:diagonal, D, Ad)
    run(:tridiagonal, SymTridiagonal(diag(D), fill(0.25, 5)), Ad)
    run(
        :block, BlockDiagonal([[3.0 1.0; 1.0 2.0], [2.0 0.5; 0.5 3.0]]),
        BlockDiagonal([[1.0 0.5], [0.5 1.0]])
    )
    run(:lowrank, D, RowCoupled(fill(0.25, 1, 6), 5))
    run(
        :kronecker, Diagonal(fill(2.0, 4)),
        KroneckerOperator([2.0 1.0; 0.0 1.0], [1.0 0.5; 0.5 2.0]); scaling = 0
    )
end

# Every workspace and algorithm this package defines must satisfy its contract and be
# `--trim` compatible, asserted here rather than type by type: a per-type `@verify` is
# opt-in, so a new type acquires the guarantee only if whoever wrote it remembered to ask.
# This sees every subtype defined by the time the module finishes, so forgetting is not
# possible. `LinearSystem` and `Preconditioner` have no concrete subtype here — every
# backend lives in PureQPBase or one of its extensions, verified there — so re-running
# `subtypes = true` for either here would re-seal types PureQPBase's own precompilation
# already sealed, which `TypeContracts` refuses as a duplicate method definition.
@verify QPWorkspace subtypes = true trim_compat = true
@verify QPAlgorithm subtypes = true trim_compat = true

end # module PureOSQP
