@testitem "the documented unmaterialized paths reach the representations the table reports" begin
    using PureDAQP, PureQPBase
    using LinearAlgebra, Random
    # The problems behind `docs/src/benchmarks.md`'s `Unmaterialized operators` table and the
    # examples of the same name, defined once in `bench/`. Reading them needs none of the three
    # algorithm packages: each case builds its own algorithm when asked.
    include(joinpath(@__DIR__, "..", "..", "bench", "unmaterialized_problems.jl"))

    recorded = recorded_fingerprints(results_path())
    cases = filter(c -> c.algorithm == "PureDAQP", unmaterialized_cases())
    @test length(cases) == 2
    @test Set(c.path for c in cases) == Set(["direct, QR", "direct, LDLᵀ"])

    solutions = map(cases) do case
        # The dual active-set method has no linear-system backend, so the choice to pin is the
        # algorithm's own working-set parameter.
        @test case.make_alg().working_set === case.pin
        ws = case_workspace(case)
        sol = solve!(ws)
        @test sol.status === SOLVED
        @test haskey(recorded, case.name)
        @test string(case_fingerprint(case), base = 16) == recorded[case.name]
        sol
    end

    # Both representations describe the same working set, so they must agree on the answer; the
    # table reports their iteration counts separately, which only means something if they do.
    @test isapprox(solutions[1].x, solutions[2].x; atol = 1.0e-6)
    @test isapprox(solutions[1].obj_val, solutions[2].obj_val; rtol = 1.0e-8)
end

@testitem "ImplicitRows applies an explicit inverse of a well-conditioned factor and solves otherwise" begin
    using PureDAQP, LinearAlgebra, Random
    Random.seed!(61)

    n, m = 12, 30
    A = randn(m, n)
    scale = rand(m) .+ 0.5

    function rows_over(R)
        return PureDAQP.ImplicitRows(A, R, transpose(R), copy(scale), zeros(n), zeros(n))
    end
    # The same operands with the solve forced, as the reference for what the product must give.
    forced(M) = PureDAQP.ImplicitRows(
        M.A, M.R, M.Rt, M.Rinv, M.Rinvt, false, M.scale, zeros(n), zeros(n)
    )

    G = randn(n, n)
    R = UpperTriangular(Matrix(cholesky(Symmetric(G' * G + n * I)).U))
    M = rows_over(R)
    @test M.use_inverse
    Mref = forced(M)
    u = randn(n)
    dest, destref = zeros(m), zeros(m)
    PureDAQP.price!(dest, M, u, 1:m)
    PureDAQP.price!(destref, Mref, u, 1:m)
    @test dest ≈ destref rtol = 1.0e-12
    for r in (1, 7, m)
        @test PureDAQP.row(M, r) ≈ PureDAQP.row(Mref, r) rtol = 1.0e-12
    end

    # A factor with a reciprocal condition number below the gate keeps the solve.
    Rbad = UpperTriangular(Matrix(Diagonal([1.0; fill(1.0e-9, n - 1)])) + triu(0.1 * randn(n, n), 1))
    Mbad = rows_over(Rbad)
    @test !Mbad.use_inverse
    PureDAQP.price!(dest, Mbad, u, 1:m)
    @test dest ≈ (A * (Rbad \ u)) ./ scale rtol = 1.0e-10
end
