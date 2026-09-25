@testitem "an interior-point iteration allocates nothing and trims on every backend, proved" begin
    using PureIPM, PureQPBase, StrictMode, StrictModeTest, LinearAlgebra, Random

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    # Measured inside a function so the workspace is a local with a known type; read as a
    # global of the test module, the call itself would allocate. Each call moves `σ`, which
    # is the regularization bump's full refactorization.
    function bump_bytes(ws)
        sigma = ws.reg_primal
        PureIPM.set_regularization!(ws, 2sigma, ws.reg_dual)
        PureIPM.factorize_newton!(ws, false)
        return @allocated begin
            PureIPM.set_regularization!(ws, sigma, ws.reg_dual)
            PureIPM.factorize_newton!(ws, false)
        end
    end

    # The rule the package holds itself to: everything a solve needs is allocated by
    # `setup`, so a second solve on the same workspace takes nothing at all.
    function resolve_bytes(ws)
        solve!(ws)
        return @allocated solve!(ws)
    end

    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    b = A * randn(n)
    dense = (P, randn(n), A, b .- rand(m), b .+ rand(m))
    nd = 50
    diagonal = (Diagonal(rand(nd) .+ 0.5), randn(nd), Diagonal(rand(nd) .+ 0.5), -rand(nd), rand(nd))
    tridiagonal = (
        SymTridiagonal(rand(nd) .+ 3, rand(nd - 1) ./ 8), randn(nd),
        Diagonal(rand(nd) .+ 0.5), -rand(nd), rand(nd),
    )
    spd(k) = (S = randn(k, k); Matrix(Symmetric(S'S ./ k + 3I)))
    Pb = PureQPBase.BlockDiagonal([spd(10) for _ in 1:5])
    Ab = PureQPBase.BlockDiagonal([randn(10, 10) ./ sqrt(10) for _ in 1:5])
    block = (Pb, randn(50), Ab, -rand(50), rand(50))
    # The interior-point ladder reaches these four of PureQPBase's own backends.
    cases = [
        (dense, PureQPBase.FullKKT),
        (diagonal, PureQPBase.DiagonalReduced),
        (tridiagonal, PureQPBase.TridiagonalReduced),
        (block, PureQPBase.BlockReduced),
    ]
    for (data, LS) in cases
        ws = setup(data..., InteriorPoint())
        @test ws.linsys isa LS
        # A full solve compiles every specialization the loop reaches on this data.
        @test solve!(ws).status == SOLVED
        W = typeof(ws)
        T = Float64
        @test test_signatures(
            [
                (PureIPM.weights!, (W,)),
                (PureIPM.factorize_newton!, (W, Bool)),
                (PureIPM.set_regularization!, (W, T, T)),
                (PureIPM.ipm_step!, (W,)),
                (PureIPM.direction!, (W,)),
                (PureIPM.max_step, (W, T)),
                (PureIPM.ipm_residuals!, (W,)),
                (PureIPM.finite_residuals, (W,)),
                (PureIPM.stalled!, (W, T)),
                (PureIPM.check_termination, (W, Bool, Bool)),
            ];
            guarantees = (:noalloc, :trim_compatible)
        ) isa Vector
        @test iszero(bump_bytes(ws))
        # `build_solution` and `solve!` are trim-safe but cannot be proved
        # allocation-free: the first resizes the certificates into capacity reserved at
        # setup, the second reads the clock, and AllocCheck sees through neither. What the
        # rule asks of them is measured on a warm re-solve instead.
        @test test_signatures(
            [(PureIPM.build_solution, (W,)), (PureIPM.solve!, (W,))];
            guarantees = (:trim_compatible,)
        ) isa Vector
        @test iszero(resolve_bytes(ws))
        # The result is the workspace's own, refilled rather than rebuilt.
        @test solve!(ws) === ws.sol
    end
end

@testitem "a regularization bump changes sigma in place" begin
    using PureIPM, LinearAlgebra

    P = [4.0 1.0; 1.0 2.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    ws = setup(P, [1.0, 1.0], A, [1.0, 0.0, 0.0], [1.0, 0.7, 0.7], InteriorPoint())
    solve!(ws)
    wt = ws.weights
    PureIPM.set_regularization!(ws, 10 * ws.reg_primal, ws.reg_dual)
    @test ws.weights === wt
    @test wt.sigma == ws.reg_primal
    @test ws.sigma_changed
end

@testitem "the interior-point proofs fail on code that allocates or cannot be trimmed" begin
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
