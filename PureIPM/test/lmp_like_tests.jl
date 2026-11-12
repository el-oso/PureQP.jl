@testitem "a LinearMaps problem shaped like the field case solves on the structured and dense backends" begin
    using PureIPM, PureQPBase, LinearAlgebra, LinearMaps
    # An epigraph problem `y_k ≥ |g_ρ·z − β_ρ|` over 76 groups of 13 nearly dependent rows, with an
    # ill-conditioned `M`; defined once in `bench/`. Handed over as `LinearMaps` it reaches the
    # backend that assembles the reduced matrix from the structure, and as matrices it reaches the
    # dense backends.
    include(joinpath(@__DIR__, "..", "..", "bench", "lmp_like_problem.jl"))

    p = lmp_like_problem(seed = 1)
    m = size(p.A, 1)
    u = fill(Inf, m)
    Hd, Ad = Matrix(p.H), Matrix(p.A)
    objective(x) = dot(x, Hd * x) / 2 + dot(p.f, x)

    cases = (
        ("LinearMaps", p.H, p.A, (;), :product_reduced),
        ("dense", Hd, Ad, (;), :cholesky),
        ("dense, linsys = :kkt", Hd, Ad, (; linsys = :kkt), :bunchkaufman),
    )
    solutions = map(cases) do (name, P, A, kw, backend)
        ws = setup(P, p.f, A, p.b, u, InteriorPoint(); scaling = 0, max_iter = 200, kw...)
        @test PureQPBase.backend_name(ws.linsys) === backend
        sol = solve!(ws)
        @test sol.status === SOLVED
        @test minimum(Ad * sol.x .- p.b) >= -1.0e-6
        sol
    end
    # The dense reduced form meets the nearly dependent rows with a bump of the regularization once
    # `μ` is below the tolerance and has stopped decreasing, so it finishes well inside the limit.
    @test solutions[2].iter < 150
    # Refinement that stops helping is undone rather than followed into `NaN`, so ten steps solve
    # these seeds too.
    for seed in 1:3
        q = lmp_like_problem(; seed)
        ws = setup(Matrix(q.H), q.f, Matrix(q.A), q.b, u, InteriorPoint(; refine_iter = 10); scaling = 0, max_iter = 200, linsys = :dense)
        sol = solve!(ws)
        @test sol.status === SOLVED
        @test isfinite(sol.obj_val)
    end

    # Whatever factorization solves the Newton system, the optimum is one point.
    objectives = [objective(sol.x) for sol in solutions]
    @test all(isapprox.(objectives, objectives[1]; rtol = 1.0e-6))
    @test isapprox(solutions[1].x, solutions[2].x; atol = 1.0e-3)
end
