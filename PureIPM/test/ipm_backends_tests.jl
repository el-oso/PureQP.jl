@testitem "interior point: each structured backend matches the full KKT factorization" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random, BandedMatrices, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(101)
    n = 60
    # Every pair is posed as given and with `P` zeroed. `x = 0` is feasible for each, and `A` is
    # square with full column rank, so the linear programs are bounded.
    zero_like(P::Diagonal) = Diagonal(zeros(n))
    zero_like(P::SymTridiagonal) = SymTridiagonal(zeros(n), zeros(n - 1))
    zero_like(P::PureQPBase.BlockDiagonal) = PureQPBase.BlockDiagonal([zeros(size(b)) for b in P.blocks])
    zero_like(P::SparseMatrixCSC) = spzeros(size(P)...)
    band = BandedMatrix(0 => rand(n) .+ 1, 1 => rand(n - 1) ./ 8, -1 => rand(n - 1) ./ 8, 2 => rand(n - 2) ./ 8)
    Pb, qb, Ab, lb, ub = banded_qp(200, 300)
    cases = [
        (:diagonal, Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5), randn(n), -rand(n), rand(n)),
        (:tridiagonal, SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5), randn(n), -rand(n), rand(n)),
        (:tridiagonal, SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L), randn(n), -rand(n), rand(n)),
        (:banded, SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), band, randn(n), -rand(n), rand(n)),
        (
            :block,
            PureQPBase.BlockDiagonal([Matrix(Diagonal(rand(12) .+ 1)) for _ in 1:5]),
            PureQPBase.BlockDiagonal([randn(12, 12) ./ sqrt(12) + 2I for _ in 1:5]),
            randn(n), -rand(n), rand(n),
        ),
        (:sparse, Pb, Ab, qb, lb, ub),
    ]
    for (backend, P, A, q, l, u) in cases, Pc in (P, zero_like(P))
        ws = setup(Pc, q, A, l, u, InteriorPoint())
        name = PureQPBase.backend_name(ws.linsys)
        # The sparse pair's KKT factor fails the fill gate, so the reduced factorization serves it.
        @test backend === :sparse ? name in SPARSE_FACTOR_BACKENDS : name === backend
        s = solve!(ws)
        kkt = PureQPBase.solve(Pc, q, A, l, u, InteriorPoint(); linsys = :kkt)
        @test s.status == kkt.status == SOLVED
        @test maximum(kkt_residuals(Matrix(Pc), q, Matrix(A), l, u, s.x, s.y)) < 1.0e-5
        @test s.iter <= 2 * kkt.iter
    end
end

@testitem "interior point: a low-rank pair is refused rather than formed" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(102)
    n = 60
    A = PureQPBase.RowCoupled(randn(3, n) ./ 4, ones(n - 3), collect(1:(n - 3)))
    q, l, u = randn(n), -rand(n), rand(n)
    P = Diagonal(rand(n) .+ 0.5)
    # A diagonal `P` with a `RowCoupled` `A`: the pair the low-rank backend exists for. That
    # backend solves the reduced matrix, and forming it at weights reaching `1/reg_dual` loses
    # the accuracy the method needs, so its rung declines -- for a positive definite `P` as much
    # as for the linear program. The rung below forms the KKT matrix from `A`'s entries, and
    # `RowCoupled` holds `k` dense rows and a diagonal in place of those entries, so no rung
    # serves the pair and it is refused rather than formed.
    for Pc in (P, Diagonal(zeros(n)))
        @test_throws "no interior-point backend serves this pair" setup(
            Pc, q, A, l, u, InteriorPoint()
        )
    end
    # The same numbers as a matrix are formed because the caller asked for that, and solve.
    for Pc in (P, Diagonal(zeros(n)))
        ws = setup(Pc, q, Matrix(A), l, u, InteriorPoint())
        @test PureQPBase.backend_name(ws.linsys) === :bunchkaufman
        s = solve!(ws)
        @test s.status == SOLVED
        @test maximum(kkt_residuals(Matrix(Pc), q, Matrix(A), l, u, s.x, s.y)) < 1.0e-5
    end
    @test_throws "linsys = :lowrank is not available with InteriorPoint()" setup(
        P, q, A, l, u, InteriorPoint(); linsys = :lowrank
    )
end

@testitem "interior point: dual numbers run on the reduced Cholesky" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random, ForwardDiff
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(8, 12; seed = 1)
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    ws = setup(D.(P), D.(q), D.(A), D.(l), D.(u), InteriorPoint())
    @test PureQPBase.backend_name(ws.linsys) === :cholesky
    @test solve!(ws).status == SOLVED
    # The objective's derivative in `q[1]` is `x[1]` at the solution.
    e1 = [1.0; zeros(7)]
    g = ForwardDiff.derivative(t -> PureQPBase.solve(P, q .+ t .* e1, A, l, u, InteriorPoint()).obj_val, 0.0)
    @test g ≈ PureQPBase.solve(P, q, A, l, u, InteriorPoint()).x[1] atol = 1.0e-5
end
