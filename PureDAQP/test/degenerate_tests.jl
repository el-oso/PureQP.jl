@testitem "a degenerate contact QP reaches the reference objective" begin
    using PureDAQP, LinearAlgebra
    include(joinpath(@__DIR__, "data", "quadruped_contact.jl"))

    P, q, A, l, u = quadruped_contact()
    sol = solve(P, q, A, l, u, ActiveSet())

    # The objective is what this problem pins, and `x` is not: the working set holds far more
    # rows than the solution has dimensions, so several sets span the solution and which one
    # the loop settles on follows the rounding. Under one BLAS thread this instance converges
    # in 53 iterations and under two or more in 57, to residuals differing by a factor of 15,
    # and both objectives agree with the reference to `3e-12`.
    @test has_solution(sol.status)
    @test sol.obj_val ≈ OBJ_REF rtol = 1.0e-8
    @test sol.dual_res < sqrt(eps(Float64)) * (1 + norm(P * sol.x + q, Inf))
end

@testitem "every instance of the degenerate family reaches the reference objective" begin
    using PureDAQP, LinearAlgebra
    include(joinpath(@__DIR__, "data", "quadruped_contact.jl"))

    _, _, _, l, u = quadruped_contact()
    for (P, q, A) in perturbations(300)
        sol = solve(P, q, A, l, u, ActiveSet())
        # A perturbation of `1e-13` changes which pyramid faces the working set picks up, and
        # with it the iteration count, which ranges from 53 to 335 across the family. What may
        # not change is that the run converges and finds the same objective.
        @test has_solution(sol.status)
        @test sol.obj_val ≈ OBJ_REF rtol = 1.0e-8
    end
end
