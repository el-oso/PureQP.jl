@testitem "the documented unmaterialized paths reach the backends the table reports" begin
    using PureIPM, PureQPBase
    using LinearMaps, LinearAlgebra, Krylov, Random
    # The problems behind `docs/src/benchmarks.md`'s `Unmaterialized operators` table and the
    # examples of the same name, defined once in `bench/`. Reading them needs none of the three
    # algorithm packages: each case builds its own algorithm when asked.
    include(joinpath(@__DIR__, "..", "..", "bench", "unmaterialized_problems.jl"))

    recorded = recorded_fingerprints(results_path())
    cases = filter(c -> c.algorithm == "PureIPM", unmaterialized_cases())
    @test length(cases) == 2
    @test Set(c.path for c in cases) == Set(["CG", "direct"])

    for case in cases
        ws = case_workspace(case)
        sol = solve!(ws)
        @test sol.status === SOLVED
        @test backend_name(ws.linsys) === case.pin
        # The table reports times for these problems. A change to one of them leaves the recorded
        # numbers describing a problem that is no longer solved here, so the fingerprint is pinned
        # and the benchmark has to be re-run with the table.
        @test haskey(recorded, case.name)
        @test string(case_fingerprint(case), base = 16) == recorded[case.name]
    end
end

@testitem "the interior-point method refuses conjugate gradients without a preconditioner" begin
    using PureIPM, PureQPBase, LinearAlgebra, Krylov, Random
    # The CG case in the table carries a `KroneckerPreconditioner` because the method will not run
    # that path without one, which is the documented reason the example passes it.
    Random.seed!(7203)
    S = randn(12, 12)
    P = PureQPBase.KroneckerOperator(Matrix(Symmetric(S'S ./ 12 + 2I)), Matrix(Symmetric(S'S ./ 12 + 2I)))
    A = PureQPBase.KroneckerOperator(randn(12, 12) ./ 4, randn(12, 12) ./ 4)
    n, m = size(A, 2), size(A, 1)
    args = (P, randn(n), A, fill(-1.0, m), fill(1.0, m), InteriorPoint())
    @test_throws "preconditioner" setup(args...; linsys = :indirect, scaling = 0, verbose = false)
    M = PureQPBase.KroneckerPreconditioner(P, A)
    ws = setup(args...; linsys = :indirect, scaling = 0, preconditioner = M, verbose = false)
    @test solve!(ws).status === SOLVED
end
