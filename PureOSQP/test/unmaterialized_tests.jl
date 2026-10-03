@testitem "the documented unmaterialized paths reach the backends the table reports" begin
    using PureQPBase, LinearMaps, LinearAlgebra, Krylov, Random
    # The problems behind `docs/src/benchmarks.md`'s `Unmaterialized operators` table and the
    # examples of the same name, defined once in `bench/`. Reading them needs none of the three
    # algorithm packages: each case builds its own algorithm when asked.
    include(joinpath(@__DIR__, "..", "..", "bench", "unmaterialized_problems.jl"))

    recorded = recorded_fingerprints(results_path())
    cases = filter(c -> c.algorithm == "PureOSQP", unmaterialized_cases())
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
