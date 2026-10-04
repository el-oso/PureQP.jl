@testitem "the MathOptInterface wrapper passes MOI.Test" begin
    using PureDAQP
    using MathOptInterface, LinearAlgebra, SparseArrays
    const MOI = MathOptInterface

    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(PureDAQP.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    # `eps_prox > 0` is what lets this suite run at all: a linear program has `P = 0`, which is
    # positive semidefinite and not positive definite, and the reduction factors `P` directly
    # unless the proximal-point iterations are on.
    #
    # The value is bounded below by conditioning rather than by accuracy. With `P = 0` the factor
    # is `√eps_prox * I`, so the reduced rows carry `1 / √eps_prox`; against this suite's largest
    # data, `1e-6` leaves `test_linear_add_constraints` short of its optimum at any iteration
    # count, and `1e-4` reaches every answer here to the tolerance checked below.
    MOI.set(model, MOI.RawOptimizerAttribute("eps_prox"), 1.0e-4)
    # The basis attributes: this method tracks a working set rather than a basis. `ObjectiveBound`:
    # it has none to report. `test_linear_DUAL_INFEASIBLE`: an unbounded problem reaches
    # `max_iter` and is reported as `ITERATION_LIMIT`, because a dual active-set method issues no
    # dual-infeasibility certificate — the status is honest rather than wrong, and not one of the
    # three MOI accepts here.
    MOI.Test.runtests(
        model,
        MOI.Test.Config(;
            atol = 1.0e-4, rtol = 1.0e-4,
            exclude = Any[MOI.ConstraintBasisStatus, MOI.VariableBasisStatus, MOI.ObjectiveBound],
        ),
        include = ["test_linear_", "test_quadratic_"],
        exclude = ["test_linear_DUAL_INFEASIBLE"],
    )
end

@testitem "the optimizer solves a strictly convex QP exactly" begin
    using PureDAQP
    using MathOptInterface, LinearAlgebra
    const MOI = MathOptInterface

    # `min x'x s.t. x₁ + x₂ ≥ 1`, whose solution is the projection of the origin onto the
    # halfspace. Asserted against the closed form rather than against another solve, so a
    # wrapper that reached a different algorithm could not agree with it by construction.
    o = PureDAQP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(src, 2)
    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarQuadraticTerm.([2.0, 2.0], x, x), MOI.ScalarAffineTerm{Float64}[], 0.0,
    )
    MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(
        src, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.GreaterThan(1.0),
    )
    MOI.copy_to(o, src)
    MOI.optimize!(o)
    @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(o, MOI.VariablePrimal(), x) ≈ [0.5, 0.5] atol = 1.0e-8
    @test MOI.get(o, MOI.ObjectiveValue()) ≈ 0.5 atol = 1.0e-8
    @test MOI.get(o, MOI.SolverName()) == "PureDAQP"
end

@testitem "a semidefinite objective needs eps_prox, and says so" begin
    using PureDAQP
    using MathOptInterface, LinearAlgebra
    const MOI = MathOptInterface

    # A linear objective is the `P = 0` case every LP presents. The reduction factors `P`, so at
    # `eps_prox = 0` it is refused by name; the proximal-point iterations accept it. This is the
    # one thing a caller coming from another QP solver has to know about this wrapper.
    function lp(eps_prox)
        o = PureDAQP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        iszero(eps_prox) || MOI.set(o, MOI.RawOptimizerAttribute("eps_prox"), eps_prox)
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(src, 2)
        MOI.add_constraint.(src, x, MOI.Interval(0.0, 1.0))
        MOI.set(
            src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0, -1.0], x), 0.0),
        )
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.add_constraint(
            src, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], x), 0.0),
            MOI.LessThan(3.0),
        )
        MOI.copy_to(o, src)
        MOI.optimize!(o)
        return o, x
    end

    @test_throws "P is not positive definite" lp(0.0)
    o, x = lp(1.0e-6)
    @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    # `-x₁ - x₂` over `x ≤ 1` and `x₁ + 2x₂ ≤ 3`: the corner the second row does not cut off.
    @test MOI.get(o, MOI.VariablePrimal(), x) ≈ [1.0, 1.0] atol = 1.0e-4
end

@testitem "what ActiveSet refuses is refused through the wrapper too" begin
    using PureDAQP
    using MathOptInterface
    const MOI = MathOptInterface

    # Settings belonging to another algorithm, or to a stage this one does not have, are refused
    # rather than ignored: a caller who sets them has a wrong expectation of the solver, and
    # dropping them silently would leave it in place. Which refusal arrives depends on where the
    # name lives. An operator-splitting parameter is not an `ActiveSet` field at all, so the
    # wrapper refuses the attribute itself; a shared `Options` field is a name this solver knows
    # and rejects a value for, which it can only do once it has the options.
    function model!(o)
        src = MOI.Utilities.Model{Float64}()
        y = MOI.add_variables(src, 2)
        obj = MOI.ScalarQuadraticFunction(
            MOI.ScalarQuadraticTerm.([2.0, 2.0], y, y), MOI.ScalarAffineTerm{Float64}[], 0.0,
        )
        MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.add_constraint(
            src, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], y), 0.0),
            MOI.GreaterThan(1.0),
        )
        MOI.copy_to(o, src)
        return o
    end

    # Refused when set, because the name is no parameter of this algorithm.
    for (name, value) in (("rho", 0.1), ("sigma", 1.0e-6), ("alpha", 1.6))
        o = PureDAQP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        @test_throws MOI.UnsupportedAttribute MOI.set(o, MOI.RawOptimizerAttribute(name), value)
    end

    # Refused when solved, because the name is an option this solver reads and these values are
    # ones it cannot honour: equilibration would rescale the rows the working set is priced
    # against, it ends on the exact solution polishing would compute, and it has no backend to
    # choose.
    for (name, value) in (("scaling", 10), ("polishing", true), ("linsys", :kkt))
        o = PureDAQP.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        MOI.set(o, MOI.RawOptimizerAttribute(name), value)
        @test_throws ArgumentError MOI.optimize!(model!(o))
    end
end
