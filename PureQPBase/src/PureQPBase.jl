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

# The calls an algorithm makes into a backend while it iterates, checked on a small problem
# for every backend this module defines. `solve_system!` and `solve_multiplier!` run every
# iteration, `refactor_weights!` every time the weights move, and `factorize!` every time the
# interior-point method bumps its regularization. `KroneckerReduced`'s `factorize!` is the one
# checked for `--trim` only: it eigendecomposes `AᵢᵀAᵢ`, which allocates, and it runs only
# when `P`, `A` or `σ` change, since no interior-point rung selects that backend. The
# preconditioners' `ldiv!` runs once per conjugate-gradient iteration. These checks report
# rather than throw; `test/strictmode_tests.jl` proves the same signatures with
# StrictModeTest.
let
    function check(ls, prob, wt)
        bx, bz, x, z = ones(prob.n), ones(prob.m), zeros(prob.n), zeros(prob.m)
        ls isa KroneckerReduced || @assert_noalloc factorize!(ls, prob, wt)
        @assert_trim_compatible factorize!(ls, prob, wt)
        @assert_noalloc refactor_weights!(ls, prob, wt)
        @assert_trim_compatible refactor_weights!(ls, prob, wt)
        @assert_noalloc solve_system!(ls, prob, wt, bx, bz, x, z)
        @assert_trim_compatible solve_system!(ls, prob, wt, bx, bz, x, z)
        @assert_noalloc solve_multiplier!(ls, prob, wt, bx, bz, x, z)
        @assert_trim_compatible solve_multiplier!(ls, prob, wt, bx, bz, x, z)
        return nothing
    end
    function backend(linsys, P, A; scaling = 10)
        n, m = size(A, 2), size(A, 1)
        q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, m), fill(1.0, m)
        prob = validated_problem(Float64, n, m, P, q, A, l, u, scaling)
        wt = SystemWeights(fill(0.1, m), fill(10.0, m), 1.0e-6)
        ls, factored = named_backend(Val(linsys), P, A, prob, wt, ADMMSelection(), nothing)
        factored || factorize!(ls, prob, wt)
        return ls, prob, wt
    end
    P = [4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0]
    A = [1.0 1.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 -1.0 0.0]
    D = Diagonal([4.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    Ad = Diagonal([1.0, 0.5, 2.0, 1.0, 1.5, 0.5])
    for (linsys, Pc, Ac, scaling) in (
            (:dense, P, A, 10),
            (:kkt, P, A, 10),
            (:diagonal, D, Ad, 10),
            (:tridiagonal, SymTridiagonal(diag(D), fill(0.25, 5)), Ad, 10),
            (
                :block,
                BlockDiagonal([[3.0 1.0; 1.0 2.0], [2.0 0.5; 0.5 3.0]]),
                BlockDiagonal([[1.0 0.5], [0.5 1.0]]), 10,
            ),
            (:lowrank, D, RowCoupled(fill(0.25, 1, 6), 5), 10),
            (:kronecker, Diagonal(fill(2.0, 4)), KroneckerOperator([2.0 1.0; 0.0 1.0], [1.0 0.5; 0.5 2.0]), 0),
        )
        check(backend(linsys, Pc, Ac; scaling)...)
    end

    ls, prob, wt = backend(:dense, P, A)
    y, x = zeros(prob.n), ones(prob.n)
    for M in (IdentityPreconditioner(), JacobiPreconditioner(ones(prob.n)))
        @assert_noalloc update_preconditioner!(M, prob, wt, 0)
        @assert_trim_compatible update_preconditioner!(M, prob, wt, 0)
        @assert_noalloc ldiv!(y, M, x)
        @assert_trim_compatible ldiv!(y, M, x)
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
