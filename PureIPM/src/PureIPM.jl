"""
    PureIPM

Mehrotra predictor-corrector interior-point method for

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

`P` and `A` may be any `AbstractMatrix`. The problem representation, the linear-system
backends and the generic `setup`/`solve`/`solve!` come from PureQPBase.jl and are re-exported
here, so `using PureIPM` is enough to solve a problem:

    solve(P, q, A, l, u, InteriorPoint())

The algorithm follows Mehrotra, *On the implementation of a primal-dual interior point
method*, SIAM Journal on Optimization 2(4):575–601, 1992.
"""
module PureIPM

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify
using StrictMode: @assert_noalloc, @assert_trim_compatible
using PureQPBase

import PureQPBase:
    setup, solve, solve!, warm_start!, cold_start!, update!, update_settings!,
    constraint_violation, derivative_ready, setup_backend, algorithm_defaults, element_typed,
    recommend_linsys, OPTION_NAMES,
    IPMSelection, SelectionFor, BlockDiagonal, BlockReduced, DiagonalLowRank,
    DiagonalReduced, KroneckerOperator, KroneckerReduced, TridiagonalReduced, Problem,
    ProductOperator, RowCoupled, SystemWeights, LINSYS_OPTIONS,
    INFTY, MIN_SCALING, RHO_MIN, RHO_MAX, RHO_TOL, RHO_EQ_OVER_INEQ, DIVISION_TOL,
    ZERO_DEADZONE,
    active_kkt, add!, adopt_settings!, adopt_update!, block_rung,
    check_finite, check_storage, choose_backend, dense_rung,
    eps_prim, eps_dual, eps_duality_gap, factorize!, factors, formed_rung, gap_terms,
    increment!, indirect_backend, indirect_rung, inner_iterations, invscaled_norm_inf,
    is_convex, is_dual_infeasible, is_materializable, is_primal_infeasible, is_scalar_multiple,
    is_symmetric, kkt_rung, kronecker_rung, last_solve_converged, lowrank_rung, mul_A!,
    mul_At!, mul_P!, multiply!, named_backend, norm_inf, polish_kernel!, reduced_diagonal!,
    reduced_rung, refactor!, refactor_rho!, refactored!, refactor_weights!, scalar_multiple,
    scale_subtract!, select_backend, set_refresh_index!, set_tolerance_level!,
    solve_multiplier!, solve_system!, structural_rows, subtract!, subtract_scaled!,
    use_residual_stop!, validate, validate_update!, validated_problem,
    residuals_at!, status_name, polish_status_name, check_termination, polish!

export setup, solve, solve!, update!, update_settings!, warm_start!, cold_start!
export dimensions, capabilities, constraint_violation
export Solution, Status, Options, default_options
export QPAlgorithm, InteriorPoint
export QPWorkspace, InteriorPointWorkspace
export has_solution, status_name
export recommend_linsys, LinsysAdvice
export backend_info, backend_name, factor_fill, BackendInfo
export PolishStatus
export adjoint_derivative, forward_derivative
export LinearSystem, ReducedCholesky, FullKKT
export Preconditioner, IdentityPreconditioner, JacobiPreconditioner, update_preconditioner!
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED, NUMERICAL_ERROR
export POLISH_SUCCESS, POLISH_FAILED, POLISH_NOT_PERFORMED
export POLISH_NO_ACTIVE_SET_FOUND, POLISH_LINSYS_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

include("settings.jl")
include("workspace.jl")
include("ipm.jl")

"""
    Optimizer(; kwargs...)

MathOptInterface optimizer, available once MathOptInterface is loaded. Keyword arguments are
the fields of [`Options`](@ref) and the parameters of [`InteriorPoint`](@ref), each checked
by name when it is set.

The wrapper lives in a package extension, so it costs nothing to a caller who does not use
it; this name is the only part of it this package owns.
"""
function Optimizer end

# The calls the interior-point method makes every iteration, checked on a problem small
# enough to solve here. `factorize_newton!` is checked for `--trim` only: its dense backend
# `FullKKT` allocates LAPACK's pivot and work arrays in every `bunchkaufman!`, and a
# regularization bump builds new `SystemWeights`. `solve!` returns a `Solution` holding
# unscaled copies of `x` and `y`, so it carries no allocation claim either. These checks
# report rather than throw; `test/strictmode_tests.jl` proves the same signatures with
# StrictModeTest.
let
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    ws = setup(P, q, A, l, u, InteriorPoint())
    solve!(ws)
    @assert_noalloc weights!(ws)
    @assert_trim_compatible weights!(ws)
    @assert_trim_compatible factorize_newton!(ws, false)
    @assert_noalloc ipm_step!(ws)
    @assert_trim_compatible ipm_step!(ws)
    @assert_noalloc direction!(ws)
    @assert_trim_compatible direction!(ws)
    @assert_noalloc ipm_residuals!(ws)
    @assert_trim_compatible ipm_residuals!(ws)
    @assert_noalloc check_termination(ws, false, false)
    @assert_trim_compatible check_termination(ws, false, false)
end

# Every workspace and algorithm this package defines must satisfy its contract and be
# `--trim` compatible, asserted here rather than type by type: a per-type `@verify` is
# opt-in, so a new type acquires the guarantee only if whoever wrote it remembered to ask.
# This sees every subtype defined by the time the module finishes, so forgetting is not
# possible.
@verify QPWorkspace subtypes = true trim_compat = true
@verify QPAlgorithm subtypes = true trim_compat = true

end # module PureIPM
