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
