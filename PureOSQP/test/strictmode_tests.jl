@testitem "an ADMM iteration allocates nothing and trims, proved" begin
    using PureOSQP, StrictMode, StrictModeTest, LinearAlgebra, Random

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

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
    for (data, linsys) in ((dense, :auto), (dense, :kkt), (diagonal, :auto), (tridiagonal, :auto))
        ws = setup(data...; linsys)
        # A full solve compiles every specialization the loop reaches on this data.
        @test solve!(ws).status == SOLVED
        W = typeof(ws)
        @test test_signatures(
            [
                (PureOSQP.admm_step!, (W,)),
                (PureOSQP.update_residuals!, (W,)),
                (PureOSQP.check_termination, (W, Bool)),
            ];
            guarantees = (:noalloc, :trim_compatible)
        ) isa Vector
    end
end
