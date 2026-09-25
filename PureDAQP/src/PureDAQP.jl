"""
    PureDAQP

A dual active-set method for

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

The problem representation and the generic `setup`/`solve`/`solve!` come from PureQPBase.jl
and are re-exported here, so `using PureDAQP` is enough to solve a problem:

    solve(P, q, A, l, u, ActiveSet())

The method reduces the problem to a least-distance problem and maintains an `LDLᵀ` of the
working set's Gram matrix under rank-one updates, so each iteration costs `O(k²)` in the size
of the working set rather than a fresh factorization. It follows Algorithm 1 of Arnström,
Bemporad and Axehill, *A dual active-set solver for embedded quadratic programming using
recursive LDLᵀ updates*, IEEE Transactions on Automatic Control 67(8):4362-4369, 2022, with
that paper's Algorithm 2 as the proximal-point outer loop. The reference implementation is
MIT-licensed and was read alongside the paper.

Unlike the other algorithms in this repository it reads `P` and `A` as dense matrices and
exploits neither sparsity nor declared structure: the reduction forms `A R⁻¹`, which is dense
whatever `A` was. It is at its best where an active-set method always is, with few rows
active at the solution.
"""
module PureDAQP

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify
using StrictMode: @strict_function, @strict, @assert_noalloc, @assert_trim_compatible
using PureQPBase

import PureQPBase:
    setup, solve, solve!, warm_start!, cold_start!, update!, update_settings!,
    derivative_ready, setup_backend, algorithm_defaults, element_typed, dimensions,
    QPData, Options, Solution, Status, QPAlgorithm, QPWorkspace, PolishStatus,
    adopt_update!, check_update, has_solution, is_convex, is_materializable, validate,
    validated_data, validate_update!, check_option_names, settings_tuple, paired

export setup, solve, solve!, update!, update_settings!, warm_start!, cold_start!
export dimensions, capabilities
export Solution, Status, Options, default_options
export QPAlgorithm, ActiveSet
export QPWorkspace, ActiveSetWorkspace
export has_solution, status_name
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED, NUMERICAL_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

# `settings.jl` first: the loop takes its tolerances as an `ActiveSet`, so the type has to
# exist before the methods that name it.
include("settings.jl")
include("ldl.jl")
include("ldp.jl")
include("workspace.jl")
include("solution.jl")

# The entry points that reach forward, held to the same guarantee as the ones declared at
# their definitions. `solve!` calls `build_solution`, which `solution.jl` defines after it,
# so inference has the whole call graph only once every file is in. Running them on a problem
# small enough to solve here also compiles the path a caller arrives on.
let
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    # `setup` builds the workspace, so it allocates by contract and carries no such claim.
    ws = setup(P, q, A, l, u, ActiveSet())
    @strict solve!(ws)
    @strict update!(ws; q = q)
    @strict update!(ws; l = l, u = u)
    @strict update_settings!(ws, ActiveSet())

    # The dual active-set loop and the pieces a solve runs around it, on a workspace a solve
    # has already brought to a state each call is legal in. `solve!` is asserted trim-
    # compatible but not allocation-free: it reads the clock, and AllocCheck counts the
    # `jl_hrtime` foreign call as an allocation it cannot see through. Everything under the
    # clock carries both claims, and `run_daqp!` is the whole iteration. These report rather
    # than throw; `test/strictmode_tests.jl` proves every kernel with StrictModeTest.
    red = ws.red
    @assert_trim_compatible solve!(ws)
    @assert_noalloc run_daqp!(red, ws.prob.q0, ws.algorithm, ws.options.max_iter)
    @assert_trim_compatible run_daqp!(red, ws.prob.q0, ws.algorithm, ws.options.max_iter)
    @assert_noalloc multipliers!(ws.y, red)
    @assert_trim_compatible multipliers!(ws.y, red)
    @assert_noalloc build_solution(ws)
    @assert_trim_compatible build_solution(ws)
    @assert_noalloc reset_working_set!(red)
    @assert_trim_compatible reset_working_set!(red)
end

# Every workspace and algorithm this package defines must satisfy its contract, asserted for
# each subtype defined by the time the module finishes rather than type by type, so a new
# type cannot acquire the guarantee only by someone remembering to ask for it.
@verify QPWorkspace subtypes = true trim_compat = true
@verify QPAlgorithm subtypes = true trim_compat = true

end # module PureDAQP
