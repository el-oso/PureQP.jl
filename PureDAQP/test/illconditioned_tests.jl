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
