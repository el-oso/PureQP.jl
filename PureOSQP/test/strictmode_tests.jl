@testitem "an ADMM iteration allocates nothing and trims on every backend, proved" begin
    using PureOSQP, StrictMode, StrictModeTest, LinearAlgebra, Random

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    # Measured inside a function so the workspace is a local with a known type; read as a
    # global of the test module, the call itself would allocate. `refactor_rho!` is what
    # `adapt_rho!` runs when `ρ` moves, which a converged solve need not reach.
    function refactor_bytes(ws)
        PureOSQP.refactor_rho!(ws)
        return @allocated PureOSQP.refactor_rho!(ws)
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
    Pb = PureOSQP.BlockDiagonal([spd(10) for _ in 1:5])
    Ab = PureOSQP.BlockDiagonal([randn(10, 10) ./ sqrt(10) for _ in 1:5])
    block = (Pb, randn(50), Ab, -rand(50), rand(50))
    Al = PureOSQP.RowCoupled(randn(3, nd) ./ 4, ones(nd - 3), collect(1:(nd - 3)))
    lowrank = (Diagonal(rand(nd) .+ 0.5), randn(nd), Al, -rand(nd), rand(nd))
    Ak = PureOSQP.KroneckerOperator(randn(4, 4), randn(5, 5))
    bk = Matrix(Ak) * randn(20)
    kronecker = (Diagonal(fill(2.0, 20)), randn(20), Ak, bk .- rand(20), bk .+ rand(20))
    cases = [
        (dense, (; linsys = :dense), PureOSQP.ReducedCholesky),
        (dense, (; linsys = :kkt), PureOSQP.FullKKT),
        (diagonal, (;), PureOSQP.DiagonalReduced),
        (tridiagonal, (;), PureOSQP.TridiagonalReduced),
        (block, (; linsys = :block), PureOSQP.BlockReduced),
        (lowrank, (; linsys = :lowrank), PureOSQP.DiagonalLowRank),
        (kronecker, (; linsys = :kronecker, scaling = 0), PureOSQP.KroneckerReduced),
    ]
    for (data, kw, LS) in cases
        ws = setup(data...; kw...)
        @test ws.linsys isa LS
        # A full solve compiles every specialization the loop reaches on this data.
        solve!(ws)
        W = typeof(ws)
        AC = typeof(ws.accel)
        @test test_signatures(
            [
                (PureOSQP.accelerate_pre!, (AC, W, Int)),
                (PureOSQP.admm_step!, (W,)),
                (PureOSQP.accelerate_post!, (AC, W, Int)),
                (PureOSQP.update_residuals!, (W,)),
                (PureOSQP.check_termination, (W, Bool)),
                (PureOSQP.adapt_rho!, (W,)),
            ];
            guarantees = (:noalloc, :trim_compatible)
        ) isa Vector
        @test iszero(refactor_bytes(ws))
        # `build_solution` and `solve!` are trim-safe but cannot be proved
        # allocation-free: the first resizes the certificates into capacity reserved at
        # setup, the second reads the clock, and AllocCheck sees through neither. What the
        # rule asks of them is measured on a warm re-solve instead.
        @test test_signatures(
            [(PureOSQP.build_solution, (W,)), (PureOSQP.solve!, (W,))];
            guarantees = (:trim_compatible,)
        ) isa Vector
        @test iszero(resolve_bytes(ws))
        # The result is the workspace's own, refilled rather than rebuilt.
        @test solve!(ws) === ws.sol
    end
end

@testitem "the ADMM proofs fail on code that allocates or cannot be trimmed" begin
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
