@testitem "an interior-point iteration allocates nothing and trims, proved" begin
    using PureIPM, PureQPBase, StrictMode, StrictModeTest, LinearAlgebra, Random

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    # Measured inside a function so the arguments are locals with known types; read as
    # globals of the test module, the call itself would allocate.
    function newton_bytes(ws)
        PureIPM.factorize_newton!(ws, false)
        return @allocated PureIPM.factorize_newton!(ws, false)
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
    for (data, LS) in ((dense, PureQPBase.FullKKT), (diagonal, PureQPBase.DiagonalReduced))
        ws = setup(data..., InteriorPoint())
        @test ws.linsys isa LS
        # A full solve compiles every specialization the loop reaches on this data.
        @test solve!(ws).status == SOLVED
        W = typeof(ws)
        @test test_signatures(
            [
                (PureIPM.weights!, (W,)),
                (PureIPM.ipm_step!, (W,)),
                (PureIPM.direction!, (W,)),
                (PureIPM.ipm_residuals!, (W,)),
                (PureIPM.check_termination, (W, Bool, Bool)),
            ];
            guarantees = (:noalloc, :trim_compatible)
        ) isa Vector
        # The refactorization every iteration makes is `--trim` compatible but not
        # allocation-free: a regularization bump builds new `SystemWeights`, and the dense
        # backend's `bunchkaufman!` allocates LAPACK's pivot and work arrays on every call.
        @test test_signatures(
            [(PureIPM.factorize_newton!, (W, Bool))]; guarantees = (:trim_compatible,)
        ) isa Vector
        LS === PureQPBase.FullKKT && @test !iszero(newton_bytes(ws))
    end
end
