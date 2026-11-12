@testitem "a LinearMaps problem shaped like the field case solves unmaterialized and dense" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps
    # An epigraph problem `y_k ≥ |g_ρ·z − β_ρ|` over 76 groups of 13 nearly dependent rows, with an
    # ill-conditioned `M`; defined once in `bench/`. Its constraint rows are the case that stresses
    # the working set, and it reaches the solver as `LinearMaps`.
    include(joinpath(@__DIR__, "..", "..", "bench", "lmp_like_problem.jl"))

    p = lmp_like_problem(seed = 1)
    n, m = size(p.A, 2), size(p.A, 1)
    @test (n, m) == (629, 2396)
    u = fill(Inf, m)
    Hd, Ad = Matrix(p.H), Matrix(p.A)
    @test issymmetric(Hd)
    objective(x) = dot(x, Hd * x) / 2 + dot(p.f, x)
    slack(x) = Ad * x .- p.b

    # The 13 rows of a group are one dominant curve and almost nothing else, which is what makes
    # their working set nearly singular.
    sv = svdvals(Ad[421 .+ 2 .* (0:12), (76 + 1):end])
    @test sv[2] / sv[1] ≈ 0.137 rtol = 0.2
    @test sv[3] / sv[1] < 0.01

    # Each working set on each representation reaches one answer: the same optimum, with no row
    # violated, and with the rows that bind almost all in the second block.
    objectives = Float64[]
    for wsk in (:rows, :gram), (P, A) in ((p.H, p.A), (Hd, Ad))
        ws = setup(P, p.f, A, p.b, u, ActiveSet(; working_set = wsk); max_iter = 5000)
        sol = solve!(ws)
        @test sol.status === SOLVED
        s = slack(sol.x)
        @test minimum(s) >= -1.0e-8
        @test 100 <= count(<(1.0e-7), s) <= 400
        @test count(<(1.0e-7), s[1:420]) <= 10
        push!(objectives, objective(sol.x))
    end
    @test all(isapprox.(objectives, objectives[1]; rtol = 1.0e-8))

    # A small change in the linear term is a warm re-solve of a few dozen iterations, not a new run.
    # The solution a workspace returns is its own, so the cold count is read before the re-solve.
    ws = setup(p.H, p.f, p.A, p.b, u, ActiveSet(); max_iter = 5000)
    cold_iterations = solve!(ws).iter
    update!(ws; q = p.f .* 1.01)
    warm = solve!(ws)
    @test warm.status === SOLVED
    @test warm.iter < cold_iterations ÷ 4

    # The same seed is the same problem and another seed is another one.
    again = lmp_like_problem(seed = 1)
    @test again.b == p.b && again.f == p.f && Matrix(again.H) == Hd
    @test lmp_like_problem(seed = 2).f != p.f
end
