"""
    PureQPBase

The algorithm-independent core of PureOSQP: the problem representation, the linear-system
backends, equilibration, termination, polishing kernels, and the `QPAlgorithm`/`QPWorkspace`
contracts an algorithm implements.

This package solves nothing by itself — [`setup`](@ref) and [`solve`](@ref) both take a
mandatory `alg::QPAlgorithm`, and no concrete algorithm is defined here. PureOSQP.jl supplies
the algorithms (`OperatorSplitting` and `InteriorPoint`) and is the package most callers want.
"""
module PureQPBase

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify
using StrictMode: @strict_contract, @assert_noalloc, @assert_trim_compatible

include("blockdiagonal.jl")
include("kronecker.jl")
include("rowcoupled.jl")
include("problem.jl")
include("options.jl")
include("weights.jl")
include("linsys.jl")
include("preconditioner.jl")
include("operator.jl")
include("lowrank.jl")
include("block.jl")
include("kronsolve.jl")
include("types.jl")
include("elementwise.jl")
include("scaling.jl")
include("termination.jl")
include("polish.jl")
include("derivative.jl")
include("recommend.jl")
include("api.jl")
include("conformance.jl")

export setup, solve, solve!, update!, update_settings!, update_rho!, warm_start!, cold_start!
export dimensions, capabilities, constraint_violation
export Solution, Status, Options, default_options
export QPAlgorithm
export QPWorkspace
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

# The per-iteration calls into the backends this module defines, checked on a small problem.
# `solve_system!` and `solve_multiplier!` run every iteration and must not allocate.
# `factorize!` and `refactor_weights!` run when the weights move, and are held to that only
# on the diagonal and tridiagonal backends: `ReducedCholesky` allocates an `m`-vector of
# scaled weight roots per factorization, and `FullKKT`'s `bunchkaufman!` allocates LAPACK's
# pivot and work arrays. These checks report rather than throw; `test/strictmode_tests.jl`
# proves the same signatures with StrictModeTest.
let
    n, m = 3, 4
    P = [4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0]
    A = [1.0 1.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 -1.0 0.0]
    q = [1.0, -1.0, 0.5]
    wt = SystemWeights(fill(0.1, m), fill(10.0, m), 1.0e-6)
    dense = validated_problem(Float64, n, m, P, q, A, fill(-1.0, m), fill(1.0, m), 10)
    bx, bz, x, z = ones(n), ones(m), zeros(n), zeros(m)
    for ls in (ReducedCholesky(q, n, m), FullKKT(q, n, m))
        @assert_trim_compatible factorize!(ls, dense, wt)
        @assert_trim_compatible refactor_weights!(ls, dense, wt)
        @assert_noalloc solve_system!(ls, dense, wt, bx, bz, x, z)
        @assert_trim_compatible solve_system!(ls, dense, wt, bx, bz, x, z)
        @assert_noalloc solve_multiplier!(ls, dense, wt, bx, bz, x, z)
        @assert_trim_compatible solve_multiplier!(ls, dense, wt, bx, bz, x, z)
    end

    wt = SystemWeights(fill(0.1, n), fill(10.0, n), 1.0e-6)
    bz, z = ones(n), zeros(n)
    Ad = Diagonal([1.0, 0.5, 2.0])
    diagonal = validated_problem(Float64, n, n, Diagonal([4.0, 3.0, 2.0]), q, Ad, fill(-1.0, n), fill(1.0, n), 10)
    tridiagonal = validated_problem(Float64, n, n, SymTridiagonal([4.0, 3.0, 2.0], [1.0, 0.5]), q, Ad, fill(-1.0, n), fill(1.0, n), 10)
    for (ls, prob) in ((DiagonalReduced(q, n), diagonal), (TridiagonalReduced(q, n), tridiagonal))
        @assert_noalloc factorize!(ls, prob, wt)
        @assert_trim_compatible factorize!(ls, prob, wt)
        @assert_noalloc refactor_weights!(ls, prob, wt)
        @assert_trim_compatible refactor_weights!(ls, prob, wt)
        @assert_noalloc solve_system!(ls, prob, wt, bx, bz, x, z)
        @assert_trim_compatible solve_system!(ls, prob, wt, bx, bz, x, z)
        @assert_noalloc solve_multiplier!(ls, prob, wt, bx, bz, x, z)
        @assert_trim_compatible solve_multiplier!(ls, prob, wt, bx, bz, x, z)
    end
    # Applied once per conjugate-gradient iteration.
    for M in (IdentityPreconditioner(), JacobiPreconditioner(ones(n)))
        @assert_noalloc ldiv!(x, M, bx)
        @assert_trim_compatible ldiv!(x, M, bx)
    end
end

# Every `LinearSystem` and built-in preconditioner defined by the time this module finishes
# must satisfy its contract and be `--trim` compatible, asserted here rather than type by
# type: a per-type `@verify` is opt-in, so a new type acquires the guarantee only if whoever
# wrote it remembered to ask. This sees every subtype defined here, so forgetting is not
# possible. An extension's backends load later and carry the same declaration at the end of
# the extension. The `QPAlgorithm` and `QPWorkspace` contracts are declared here and asserted
# by the packages that implement them, since this one defines no concrete subtype of either.
@verify LinearSystem subtypes = true trim_compat = true
@verify Preconditioner subtypes = true trim_compat = true

end # module PureQPBase
