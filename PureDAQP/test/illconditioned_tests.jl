@testitem "an ill-conditioned working set does not settle on the wrong active set" begin
    using PureDAQP, LinearAlgebra
    include(joinpath(@__DIR__, "data", "illconditioned_wrong_set.jl"))

    for i in 1:WRONGSET_COUNT
        P, q, A, l, u, obj_ref, x_ref = illconditioned_wrong_set(i)
        sol = solve(P, q, A, l, u, ActiveSet())
        @test has_solution(sol.status)
        # The objective, not the residual: the point these problems admit carries a residual
        # near `1e-6` from the cancellation in `x = R⁻¹(−u − v)`, while the objective is
        # reached to ten digits. A multiplier noise floor that silences the dual sign test
        # instead converges on an active set larger than the solution's, which lands here as
        # a relative objective error of 13 and 4000.
        @test sol.obj_val ≈ obj_ref rtol = 1.0e-7
        # At most `n` rows can be active at once, and a set that overshoots that is what the
        # silenced sign test produces.
        @test count(>(sqrt(eps(Float64))), abs.(sol.y)) <= length(q)
    end
end

@testitem "the active rows are satisfied to rounding level under an ill-conditioned P" begin
    using PureDAQP, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(11)
    for _ in 1:40
        n, m = 20, 20
        P = spd_factor(rng, n, 1.0e8)
        A = rect_factor(rng, m, n, 1.0e4)
        q = randn(rng, n)
        x0 = randn(rng, n)
        b = A * x0
        l, u = b .- 1.0, b .+ 1.0
        sol = solve(P, q, A, l, u, ActiveSet())
        @test sol.status == SOLVED
        # `x = R⁻¹(−u − v)` subtracts two vectors 20 times larger than their difference at
        # this conditioning, so without the correction `refine_primal!` applies the active
        # rows sit about `1e-6` from their bounds. Measured against the size of the point,
        # which is where the cancellation shows up.
        @test sol.prim_res < 1.0e-11 * (1 + norm(A * sol.x, Inf))
    end
end

@testitem "the Gram working set agrees with the rows it factors the Gram matrix of" begin
    using PureDAQP, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(11)
    converged = Bool[]
    for _ in 1:40
        n, m = 20, 20
        P = spd_factor(rng, n, 1.0e8)
        A = rect_factor(rng, m, n, 1.0e4)
        q = randn(rng, n)
        b = A * randn(rng, n)
        l, u = b .- 1.0, b .+ 1.0
        rows = solve(P, q, A, l, u, ActiveSet(working_set = :rows))
        gram = solve(P, q, A, l, u, ActiveSet(working_set = :gram))
        @test rows.status == SOLVED
        # `:gram` may refuse here, or spend its iterations without converging -- it decides
        # rank through a squared quantity and that is what it costs. What it may not do is
        # report a converged answer that differs from the one the same rows give.
        push!(converged, gram.status == SOLVED)
        if gram.status == SOLVED
            @test gram.obj_val ≈ rows.obj_val rtol = 1.0e-6
        end
    end
    # Most of the family does converge, so the comparison above is not vacuous.
    @test count(converged) >= 30
end

@testitem "the iteration limit reports the point the loop reached" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(3)
    n, m = 20, 60
    P = spd_factor(rng, n, 1.0e6)
    A = rect_factor(rng, m, n, 1.0e3)
    q = randn(rng, n)
    b = A * randn(rng, n)
    l, u = b .- 1.0, b .+ 1.0

    full = solve(P, q, A, l, u, ActiveSet())
    @test full.status == SOLVED

    # Two iterations do not reach it; the run took 33. `MAX_ITER_REACHED` carries a point by
    # contract, so what comes back is where the loop stopped rather than the zeros `setup`
    # left in the workspace.
    short = solve(P, q, A, l, u, ActiveSet(); max_iter = 2)
    @test short.status == MAX_ITER_REACHED
    @test has_solution(short.status)
    @test !iszero(short.x)
    @test isfinite(short.obj_val)

    # The same on a re-solve, where what the workspace held before is the previous answer.
    ws = PureDAQP.setup(P, q, A, l, u, ActiveSet())
    first_answer = copy(PureQPBase.solve!(ws))
    @test first_answer.status == SOLVED
    PureQPBase.update!(ws; q = q .+ 50.0)
    PureDAQP.update_settings!(ws; max_iter = 1)
    limited = PureQPBase.solve!(ws)
    @test limited.status == MAX_ITER_REACHED
    @test limited.x != first_answer.x
end
