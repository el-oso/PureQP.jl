@testitem "a dual active-set iteration allocates nothing and trims, proved" begin
    using PureDAQP, StrictMode, StrictModeTest, LinearAlgebra, Random

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)

    # A problem `eps_prox > 0` reaches, so the proximal-point outer loop is compiled too.
    Psing = Matrix(Diagonal([ones(n - 2); zeros(2)]))

    for (Pk, alg) in ((P, ActiveSet()), (Psing, ActiveSet(; eps_prox = 1.0e-5)))
        ws = setup(Pk, q, A, l, u, alg)
        # A full solve compiles every specialization the loop reaches on this data.
        solve!(ws)
        W = typeof(ws)
        RD = typeof(ws.red)          # the reduction, carrying its Cholesky concretely
        LW = typeof(ws.red.ws)       # the least-distance workspace the iteration runs on
        ALG = typeof(ws.algorithm)
        V = Vector{Float64}
        both = (:noalloc, :trim_compatible)

        # `solve!` reads the clock, and AllocCheck counts `time_ns`'s `jl_hrtime` foreign
        # call as an allocation. Everything the clock brackets is proved on both counts.
        @test test_signatures([(PureDAQP.solve!, (W,))]; guarantees = (:trim_compatible,)) isa Vector

        @test test_signatures(
            [
                (PureDAQP.run_daqp!, (RD, V, ALG, Int)),
                (PureDAQP.inner_solve!, (RD, V, V, ALG, Int)),
                (PureDAQP.solve_ldp!, (LW, ALG, Int)),
                (PureDAQP.primal_point!, (LW,)),
                (PureDAQP.working_set_multipliers!, (LW,)),
                (PureDAQP.entering_row, (LW, Float64, Bool)),
                (PureDAQP.activate!, (LW, Int, Int8)),
                (PureDAQP.deactivate!, (LW, Int)),
                (PureDAQP.step_and_drop!, (LW, V, Float64)),
                (PureDAQP.singular_step!, (LW, Int, Float64)),
                (PureDAQP.step_toward_multipliers!, (LW, Float64)),
                (PureDAQP.multipliers!, (V, RD)),
                (PureDAQP.primal!, (V, RD, V)),
                (PureDAQP.set_targets!, (RD, V)),
                (PureDAQP.reset_working_set!, (RD,)),
                (PureDAQP.build_solution, (W,)),
            ];
            guarantees = both
        ) isa Vector
    end
end

@testitem "a warm re-solve allocates nothing at run time" begin
    using PureDAQP, LinearAlgebra, Random

    # Measured inside a function so the workspace is a local with a known type; read as a
    # global of the test module, the call itself would allocate. The static proofs above
    # cover every path; this covers the one a caller actually takes, clock included.
    function resolve_bytes(ws)
        solve!(ws)
        return @allocated solve!(ws)
    end

    Random.seed!(2)
    n, m = 10, 24
    X = randn(n, n)
    A = randn(m, n)
    b = A * randn(n)
    ws = setup(Matrix(X'X / n + I), randn(n), A, b .- rand(m), b .+ rand(m), ActiveSet())
    @test iszero(resolve_bytes(ws))
end

@testitem "the dual active-set proofs fail on code that allocates or cannot be trimmed" begin
    using StrictMode, StrictModeTest

    # A gate that passes everything proves nothing; each of these must be refused.
    StrictMode.assert_enabled()
    grow(n) = zeros(n)
    @test_throws StrictMode.StrictViolation test_signatures([(grow, (Int,))]; guarantees = (:noalloc,))
    dynamic(r) = r[] + 1
    @test_throws StrictMode.StrictViolation test_signatures(
        [(dynamic, (Base.RefValue{Any},))]; guarantees = (:trim_compatible,)
    )
end
