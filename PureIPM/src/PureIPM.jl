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
using Reexport
# The base's own exports reach a caller through here, so `using PureIPM` gives the whole API:
# the verbs, the `Solution`, the statuses and the options. They stay PureQPBase's — this says
# where they come from in one line, where a list repeated per solver would make one generic
# function look like it had four owners.
@reexport using PureQPBase

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
    empty_solution, eps_prim, eps_dual, eps_duality_gap, factorize!, factors, formed_rung,
    gap_terms, no_certificate, unit_certificate!, unscale!, holds_structure,
    increment!, indirect_backend, indirect_rung, inner_iterations, invscaled_norm_inf,
    ProductReduced,
    is_convex, is_dual_infeasible, is_materializable, is_primal_infeasible, is_scalar_multiple,
    is_symmetric, kkt_rung, kronecker_rung, last_solve_converged, lowrank_rung, mul_A!,
    mul_At!, mul_P!, multiply!, named_backend, norm_inf, polish_kernel!, reduced_diagonal!,
    reduced_rung, refactor!, refactor_rho!, refactored!, refactor_weights!, scalar_multiple,
    scale_subtract!, select_backend, set_refresh_index!, set_tolerance_level!,
    solve_multiplier!, solve_system!, structural_rows, subtract!, subtract_scaled!,
    use_residual_stop!, validate, validate_update!, validated_problem,
    residuals_at!, status_name, polish_status_name, check_termination, polish!


# What this package owns. Everything else a caller needs is PureQPBase's and arrives through the
# `@reexport` above.
export InteriorPoint, InteriorPointWorkspace

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
# enough to solve here over every backend its ladder reaches from PureQPBase's own source.
# `factorize_newton!` refactorizes every iteration and, on a regularization bump, rebuilds
# the factorization after `set_regularization!` changes `σ`. A solve allocates nothing: the
# `Solution` it returns is the workspace's own, refilled. That claim is measured rather than
# asserted here — `build_solution` resizes the certificates into capacity reserved at setup
# and copies into a `Vector` the caller reads, and `solve!` reads the clock; the scan reads a
# `resize!` and a `copyto!` that cannot be proved free of aliasing as allocation whatever
# they do at run time. These checks report rather than throw; `test/strictmode_tests.jl`
# measures both at zero bytes and proves the same signatures with StrictModeTest.
let
    function check(ws)
        solve!(ws)
        bound = iterate_bound(ws)
        @assert_noalloc weights!(ws)
        @assert_trim_compatible weights!(ws)
        @assert_noalloc factorize_newton!(ws, false)
        @assert_trim_compatible factorize_newton!(ws, false)
        @assert_noalloc set_regularization!(ws, ws.reg_primal, ws.reg_dual)
        @assert_trim_compatible set_regularization!(ws, ws.reg_primal, ws.reg_dual)
        @assert_noalloc ipm_step!(ws)
        @assert_trim_compatible ipm_step!(ws)
        @assert_noalloc direction!(ws)
        @assert_trim_compatible direction!(ws)
        @assert_noalloc max_step(ws, one(bound))
        @assert_trim_compatible max_step(ws, one(bound))
        @assert_noalloc ipm_residuals!(ws)
        @assert_trim_compatible ipm_residuals!(ws)
        @assert_noalloc finite_residuals(ws)
        @assert_trim_compatible finite_residuals(ws)
        @assert_noalloc stalled!(ws, bound)
        @assert_trim_compatible stalled!(ws, bound)
        @assert_noalloc check_termination(ws, false, false)
        @assert_trim_compatible check_termination(ws, false, false)
        @assert_noalloc build_solution(ws)
        @assert_trim_compatible build_solution(ws)
        @assert_trim_compatible solve!(ws)
        return nothing
    end
    function run(P, A)
        n, m = size(A, 2), size(A, 1)
        q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, m), fill(1.0, m)
        return check(setup(P, q, A, l, u, InteriorPoint()))
    end
    D = Diagonal([4.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    Ad = Diagonal([1.0, 0.5, 2.0, 1.0, 1.5, 0.5])
    run([4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0], [1.0 1.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 -1.0 0.0])
    run(D, Ad)
    run(SymTridiagonal(diag(D), fill(0.25, 5)), Ad)
    run(BlockDiagonal([[3.0 1.0; 1.0 2.0], [2.0 0.5; 0.5 3.0]]), BlockDiagonal([[1.0 0.5], [0.5 1.0]]))
end

# Every workspace and algorithm this package defines must satisfy its contract and be
# `--trim` compatible, asserted here rather than type by type: a per-type `@verify` is
# opt-in, so a new type acquires the guarantee only if whoever wrote it remembered to ask.
# This sees every subtype defined by the time the module finishes, so forgetting is not
# possible.
@verify QPWorkspace subtypes = true trim_compat = true
@verify QPAlgorithm subtypes = true trim_compat = true

end # module PureIPM
