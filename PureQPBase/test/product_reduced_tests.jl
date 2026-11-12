@testitem "ProductReduced builds the P term from entries when they are cheap and from products otherwise" begin
    using PureQPBase, LinearAlgebra, SparseArrays, FillArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(53)

    n, m = 12, 20
    spd(k) = (S = randn(k, k); Matrix(Symmetric(S'S / k + I)))
    Pm = spd(n)
    Pd = Diagonal(rand(n) .+ 0.5)
    Pb = PureQPBase.BlockDiagonal([spd(5), spd(4), spd(3)])
    Pk = PureQPBase.KroneckerOperator(spd(3), spd(4))
    cases = [
        ("dense", Pm), ("Symmetric", Symmetric(Pm)), ("Diagonal", Pd),
        ("BlockDiagonal", Pb), ("Kronecker", Pk),
    ]
    A = randn(m, n)
    D = rand(n) .+ 0.5
    c, sigma = 1.5, 0.3

    function p_term(P)
        prob = raw_problem(P, A, n, m; D, c)
        ls = PureQPBase.ProductReduced(prob.q0, n, m, A, Val(true))
        @test PureQPBase.factorize!(ls, prob, raw_weights(fill(0.75, m), sigma))
        return copy(ls.R0)
    end

    for (name, P) in cases
        @test PureQPBase.entries_cheap(P)
        want = c .* D .* Matrix(P) .* D' + sigma * I
        @test p_term(P) ≈ want rtol = 1.0e-13
        # The route through products reaches the same matrix: a wrapper that answers products
        # only is not materializable, so it takes the column form.
        wrapped = PureQPBase.ProductOperator{Float64}(Matrix(P))
        @test !PureQPBase.entries_cheap(wrapped)
        @test p_term(wrapped) ≈ p_term(P) rtol = 1.0e-13
    end

    # Storage whose entries are searches, and operators that answer products only, keep the
    # column form.
    @test !PureQPBase.entries_cheap(sparse(Pm))
    @test !PureQPBase.entries_cheap(Symmetric(sparse(Pm)))
    @test p_term(sparse(Pm)) ≈ c .* D .* Pm .* D' + sigma * I rtol = 1.0e-13

    # A block that is not cheap makes its container not cheap.
    @test PureQPBase.entries_cheap(PureQPBase.BlockDiagonal([spd(3), Diagonal(ones(2))]))
    @test !PureQPBase.entries_cheap(PureQPBase.BlockDiagonal([spd(3), sparse(spd(2))]))
    @test PureQPBase.entries_cheap(Fill(0.5, 4, 4))
    @test PureQPBase.entries_cheap(PureQPBase.JoinedOperator(Fill(0.5, 6, 2), randn(6, 3)))
    @test !PureQPBase.entries_cheap(PureQPBase.JoinedOperator(Fill(0.5, 6, 2), sparse(randn(6, 3))))
    @test PureQPBase.entries_cheap(PureQPBase.StackedOperator(randn(2, 4), Fill(0.5, 3, 4)))
end
