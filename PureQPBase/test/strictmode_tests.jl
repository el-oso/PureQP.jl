@testitem "the backends' per-iteration solves allocate nothing and trim, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    # Measured inside a function so the arguments are locals with known types; read as
    # globals of the test module, the call itself would allocate.
    function refactor_bytes(ls, prob, wt)
        PureQPBase.refactor_weights!(ls, prob, wt)
        return @allocated PureQPBase.refactor_weights!(ls, prob, wt)
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
    cases = (
        (dense, :auto, PureQPBase.ReducedCholesky),
        (dense, :kkt, PureQPBase.FullKKT),
        (diagonal, :auto, PureQPBase.DiagonalReduced),
        (tridiagonal, :auto, PureQPBase.TridiagonalReduced),
    )
    for (data, linsys, LS) in cases
        prob, wt, ls = backend_for(data...; linsys)
        @test ls isa LS
        bx, bz = randn(prob.n), randn(prob.m)
        x, z = zeros(prob.n), zeros(prob.m)
        # Warm every signature on the real data before proving it.
        PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
        PureQPBase.solve_multiplier!(ls, prob, wt, bx, bz, x, z)
        PureQPBase.refactor_weights!(ls, prob, wt)
        types = map(typeof, (ls, prob, wt, bx, bz, x, z))
        hot = [
            (PureQPBase.solve_system!, types),
            (PureQPBase.solve_multiplier!, types),
        ]
        refactor = [
            (PureQPBase.factorize!, types[1:3]),
            (PureQPBase.refactor_weights!, types[1:3]),
        ]
        @test test_signatures(hot; guarantees = (:noalloc, :trim_compatible)) isa Vector
        if ls isa Union{PureQPBase.ReducedCholesky, PureQPBase.FullKKT}
            # The dense factorizations allocate: `ReducedCholesky` an `m`-vector of scaled
            # weight roots, `FullKKT` LAPACK's pivot and work arrays in `bunchkaufman!`.
            @test test_signatures(refactor; guarantees = (:trim_compatible,)) isa Vector
            @test !iszero(refactor_bytes(ls, prob, wt))
        else
            @test test_signatures(refactor; guarantees = (:noalloc, :trim_compatible)) isa Vector
        end
    end
end

@testitem "the built-in preconditioners apply and refresh without allocating, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    StrictMode.assert_enabled()

    Random.seed!(2)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    prob, wt, _ = backend_for(P, randn(n), A, -rand(m), rand(m))
    y, x = zeros(n), randn(n)
    for M in (PureQPBase.IdentityPreconditioner(), PureQPBase.JacobiPreconditioner(zeros(n)))
        PureQPBase.update_preconditioner!(M, prob, wt, 0)
        ldiv!(y, M, x)
        @test test_signatures(
            [
                (ldiv!, map(typeof, (y, M, x))),
                (PureQPBase.update_preconditioner!, map(typeof, (M, prob, wt, 0))),
            ];
            guarantees = (:noalloc, :trim_compatible)
        ) isa Vector
    end
end
