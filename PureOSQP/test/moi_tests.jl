# `MOI.Test` is the point of these eight items: it is a far more thorough suite than anything
# written here would be, and it is the same one every registered solver runs. It is split by
# test-name prefix because compiling its several hundred test functions against this model
# costs minutes, and six items compile their share of them in parallel.
@testitem "the MathOptInterface wrapper passes MOI.Test: linear and quadratic" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[1])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: conic" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[2])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: model, solve and modification" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[3])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: basic scalar constraints" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[4])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: basic VectorOfVariables constraints" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[5])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: basic VectorAffineFunction constraints" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[6])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: basic vector nonlinear constraints" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); include = MOI_GROUPS[7])
end

@testitem "the MathOptInterface wrapper passes MOI.Test: the rest" tags = [:moi] begin
    include(joinpath(@__DIR__, "moi_helpers.jl"))
    MOI.Test.runtests(moi_model(), moi_config(); exclude = reduce(vcat, MOI_GROUPS))
end

@testitem "the wrapper reports the solver's own numbers" begin
    using MathOptInterface, LinearAlgebra, SparseArrays
    const MOI = MathOptInterface

    # minimize (x-1)^2 + (y-2)^2  s.t.  x + y <= 2,  x >= 0,  y >= 0
    o = PureOSQP.Optimizer()
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(src, 2)
    MOI.add_constraint.(src, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(
        src,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
        MOI.LessThan(2.0),
    )
    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarQuadraticTerm.([2.0, 2.0], x, x),
        MOI.ScalarAffineTerm.([-2.0, -4.0], x),
        5.0,
    )
    MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.copy_to(o, src)
    MOI.optimize!(o)

    @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(o, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(o, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(o, MOI.ResultCount()) == 1
    @test MOI.get(o, MOI.SolverName()) == "PureOSQP"

    # The constrained optimum of this problem: x + y = 2 is active.
    xv = MOI.get.(o, MOI.VariablePrimal(), x)
    @test sum(xv) ≈ 2.0 atol = 1.0e-4
    @test MOI.get(o, MOI.ObjectiveValue()) ≈ 0.5 atol = 1.0e-4

    # These come straight from `Solution`, so they double as a check that the fields added
    # for reporting are wired through rather than recomputed.
    @test MOI.get(o, MOI.SolveTimeSec()) > 0
    @test MOI.get(o, MOI.BarrierIterations()) > 0
    @test MOI.get(o, MOI.DualObjectiveValue()) ≈ MOI.get(o, MOI.ObjectiveValue()) atol = 1.0e-3

    # Settings reach the solver by their own names.
    @test MOI.supports(o, MOI.RawOptimizerAttribute("eps_abs"))
    @test !MOI.supports(o, MOI.RawOptimizerAttribute("not_a_setting"))
    MOI.set(o, MOI.RawOptimizerAttribute("eps_abs"), 1.0e-10)
    @test MOI.get(o, MOI.RawOptimizerAttribute("eps_abs")) == 1.0e-10
    MOI.set(o, MOI.TimeLimitSec(), 5.0)
    @test MOI.get(o, MOI.TimeLimitSec()) == 5.0
end

@testitem "raw settings are checked when set and read back their defaults" begin
    using MathOptInterface
    const MOI = MathOptInterface

    o = PureOSQP.Optimizer()
    @test MOI.get(o, MOI.RawOptimizerAttribute("max_iter")) == 4000
    @test MOI.get(o, MOI.RawOptimizerAttribute("linsys")) === :auto
    @test_throws MOI.UnsupportedAttribute MOI.get(o, MOI.RawOptimizerAttribute("not_a_setting"))

    @test_throws "linsys must be one of" MOI.set(o, MOI.RawOptimizerAttribute("linsys"), :nope)
    @test_throws "alpha must lie in (0, 2)" MOI.set(o, MOI.RawOptimizerAttribute("alpha"), 3.0)
    # A refused value leaves the setting as it was.
    @test MOI.get(o, MOI.RawOptimizerAttribute("alpha")) == 1.6

    MOI.set(o, MOI.RawOptimizerAttribute("linsys"), "kkt")
    @test MOI.get(o, MOI.RawOptimizerAttribute("linsys")) === :kkt
end

@testitem "each optimizer takes its own algorithm's parameters and no other's" begin
    using PureIPM
    using MathOptInterface
    const MOI = MathOptInterface

    admm, ipm = PureOSQP.Optimizer(), PureIPM.Optimizer()
    @test MOI.get(admm, MOI.SolverName()) == "PureOSQP"
    @test MOI.get(ipm, MOI.SolverName()) == "PureIPM"

    # Each takes the options, and only its own algorithm's parameters.
    @test MOI.supports(admm, MOI.RawOptimizerAttribute("rho"))
    @test !MOI.supports(admm, MOI.RawOptimizerAttribute("reg_primal"))
    @test MOI.supports(ipm, MOI.RawOptimizerAttribute("reg_primal"))
    @test !MOI.supports(ipm, MOI.RawOptimizerAttribute("rho"))
    @test MOI.supports(admm, MOI.RawOptimizerAttribute("eps_abs"))
    @test MOI.supports(ipm, MOI.RawOptimizerAttribute("eps_abs"))

    # A value is validated when it is set, not at `optimize!`.
    MOI.set(admm, MOI.RawOptimizerAttribute("rho"), 0.2)
    @test MOI.get(admm, MOI.RawOptimizerAttribute("rho")) == 0.2
    MOI.set(ipm, MOI.RawOptimizerAttribute("max_reg_bumps"), 3)
    @test MOI.get(ipm, MOI.RawOptimizerAttribute("max_reg_bumps")) == 3
    @test_throws "must be non-negative" MOI.set(ipm, MOI.RawOptimizerAttribute("max_reg_bumps"), -1)

    # The defaults read back are the algorithm's own, and an option set explicitly survives.
    MOI.set(ipm, MOI.RawOptimizerAttribute("eps_rel"), 1.0e-7)
    @test MOI.get(ipm, MOI.RawOptimizerAttribute("eps_rel")) == 1.0e-7
    @test MOI.get(ipm, MOI.RawOptimizerAttribute("max_iter")) == 100
    @test MOI.get(ipm, MOI.RawOptimizerAttribute("eps_abs")) == 1.0e-8
    @test MOI.get(admm, MOI.RawOptimizerAttribute("max_iter")) == 4000
    @test MOI.get(admm, MOI.RawOptimizerAttribute("eps_abs")) == 1.0e-3

    # A setting that depends on another is resolved from both, the same way `setup` does.
    @test MOI.get(ipm, MOI.RawOptimizerAttribute("refine_iter")) == 1
    MOI.set(ipm, MOI.RawOptimizerAttribute("linsys"), "indirect")
    @test iszero(MOI.get(ipm, MOI.RawOptimizerAttribute("refine_iter")))
end

@testitem "NUMERICAL_ERROR maps to MOI.NUMERICAL_ERROR with no result" begin
    using MathOptInterface
    const MOI = MathOptInterface

    # ADMM never ends this way, so the status is placed on a solution directly.
    o = PureOSQP.Optimizer()
    s = PureOSQP.solve([2.0;;], [1.0], [1.0;;], [-1.0], [1.0])
    fields = ntuple(i -> getfield(s, i), fieldcount(typeof(s)))
    o.sol = typeof(s)(fields[1:2]..., PureOSQP.NUMERICAL_ERROR, fields[4:end]...)
    @test MOI.get(o, MOI.TerminationStatus()) == MOI.NUMERICAL_ERROR
    @test MOI.get(o, MOI.ResultCount()) == 0
    @test MOI.get(o, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(o, MOI.DualStatus()) == MOI.NO_SOLUTION
    @test MOI.get(o, MOI.RawStatusString()) == "NUMERICAL_ERROR"
end
