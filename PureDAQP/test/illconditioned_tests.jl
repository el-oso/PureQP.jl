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
