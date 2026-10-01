@testitem "a KroneckerOperator agrees with the matrix it stands for" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(61)
    A1, A2 = randn(5, 4), randn(3, 6)
    K = PureQPBase.KroneckerOperator(A1, A2)
    dense = kron(A1, A2)

    @test size(K) == size(dense)
    @test Matrix(K) ≈ dense
    x = randn(size(K, 2))
    y = randn(size(K, 1))
    @test K * x ≈ dense * x
    @test K' * y ≈ dense' * y

    # The products run off the factors, so they must not depend on the scratch's contents.
    fill!(K.scratch1, NaN)
    fill!(K.scratch2, NaN)
    @test K * x ≈ dense * x
    @test K' * y ≈ dense' * y
end

@testitem "the Kronecker backend solves the system the dense one does" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(62)
    n1, n2 = 12, 10
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureQPBase.KroneckerOperator(A1, A2)
    n = n1 * n2
    P = Diagonal(fill(2.0, n))
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)

    prob, wt, ls = backend_for(P, q, K, l, u; scaling = 0)
    @test PureQPBase.backend_name(ls) === :kronecker

    bx, bz = randn(n), randn(n)
    x, z = zeros(n), zeros(n)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, kron(A1, A2), wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-7
    @test z ≈ K * x rtol = 1.0e-7

    # Two eigenbases and a diagonal, against a dense inverse's triangle.
    info = PureQPBase.backend_info(ls)
    @test info.factor_nnz == n1^2 + n2^2 + n1 * n2
    @test info.factor_nnz < n * (n + 1) ÷ 2
end

@testitem "the Kronecker rung declines what it cannot diagonalize" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(63)
    n1, n2 = 8, 6
    n = n1 * n2
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureQPBase.KroneckerOperator(A1, A2)
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    scalar = Diagonal(fill(2.0, n))
    name(P; kwargs...) = PureQPBase.backend_name(last(backend_for(P, q, K, l, u; kwargs...)))

    # Each of these breaks the diagonalization. A rung that accepted any of them would return
    # a wrong answer, not a slow one. What serves them instead is the matrix-free backend: a
    # `KroneckerOperator` answers `holds_structure` true, so the rungs that form the reduced
    # matrix decline rather than hold `n₁²n₂²` entries for a pair that stores `n₁² + n₂²`.
    # `factorize = false` because only the choice is under test, and the matrix-free backend
    # takes its conjugate-gradient settings from an algorithm, which this helper has none of.
    @test name(scalar; scaling = 0) === :kronecker
    # Equilibration puts `c·μ·D²` in the reduced matrix: diagonal, but not scalar.
    @test name(scalar; factorize = false) === :indirect
    # A `P` that is not a multiple of the identity, including a Kronecker one.
    @test name(Diagonal(rand(n) .+ 1); scaling = 0, factorize = false) === :indirect
    # A second weight among the rows, which an equality row is one way to produce.
    @test name(scalar; scaling = 0, rho = vcat(1.0e3, fill(0.1, n - 1)), factorize = false) ===
        :indirect

    # Predicate and value are separate so neither returns a union; the rung checks the first
    # before reading the second.
    @test PureQPBase.is_scalar_multiple(Diagonal(fill(3.0, 4)))
    @test PureQPBase.scalar_multiple(Diagonal(fill(3.0, 4))) == 3.0
    @test !PureQPBase.is_scalar_multiple(Diagonal([1.0, 2.0]))
    @test !PureQPBase.is_scalar_multiple(randn(3, 3))
end

@testitem "a non-finite Kronecker factor is refused by naming the factor" begin
    using PureQPBase, LinearAlgebra
    include(joinpath(@__DIR__, "helpers.jl"))
    # The check reads the two factors, not the product, so the entry it reports is the
    # factor's own.
    A1 = [1.0 2.0; 3.0 NaN]
    A2 = [1.0 0.0; 0.0 1.0]
    @test_throws "A's first Kronecker factor is not finite at entry (2, 2)" backend_for(
        Diagonal(ones(4)), zeros(4), PureQPBase.KroneckerOperator(A1, A2), fill(-1.0, 4), fill(1.0, 4)
    )
    @test_throws "A's second Kronecker factor" backend_for(
        Diagonal(ones(4)), zeros(4), PureQPBase.KroneckerOperator(A2, A1), fill(-1.0, 4), fill(1.0, 4)
    )
end

@testitem "a Kronecker P answers is_symmetric and is_convex from its factors" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(64)
    # A symmetric factor with a prescribed spectrum.
    function with_spectrum(d)
        Q = Matrix(qr(randn(length(d), length(d))).Q)
        return Symmetric(Q * Diagonal(d) * Q')
    end
    P1 = Matrix(with_spectrum([0.5, 1.0, 2.0, 4.0]))
    P2 = Matrix(with_spectrum([-1.0, 0.5, 3.0]))
    K = PureQPBase.KroneckerOperator(P1, P2)

    @test PureQPBase.is_symmetric(K)
    @test !PureQPBase.is_symmetric(PureQPBase.KroneckerOperator(P1, [1.0 2.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]))
    @test !PureQPBase.is_symmetric(PureQPBase.KroneckerOperator(randn(2, 3), randn(3, 2)))

    # The smallest eigenvalue of the product is `4 * (-1) = -4`, so `P + σI` is positive
    # definite exactly when σ > 4.
    threshold = minimum(kron(eigvals(Symmetric(P1)), eigvals(Symmetric(P2))))
    @test threshold ≈ -4.0
    for sigma in (threshold - 0.1, -threshold - 0.1, -threshold + 0.1, -threshold + 1.0, 100.0)
        @test PureQPBase.is_convex(Float64, K, sigma) == PureQPBase.is_convex(Float64, Matrix(K), sigma)
    end
    @test !PureQPBase.is_convex(Float64, K, 3.9)
    @test PureQPBase.is_convex(Float64, K, 4.1)

    # A positive semidefinite product passes at any positive shift, and a negative definite
    # one (both factors of opposite sign) fails.
    psd = PureQPBase.KroneckerOperator(P1, Matrix(with_spectrum([0.0, 1.0, 2.0])))
    @test PureQPBase.is_convex(Float64, psd, 1.0e-6)
    negative = PureQPBase.KroneckerOperator(P1, Matrix(with_spectrum([-3.0, -1.0, -0.5])))
    @test !PureQPBase.is_convex(Float64, negative, 1.0e-6)

    # `T` decides the arithmetic, not the factors' element type.
    K32 = PureQPBase.KroneckerOperator(Float32.(P1), Float32.(P2))
    @test PureQPBase.is_convex(Float32, K32, 4.1f0)
    @test !PureQPBase.is_convex(Float32, K32, 3.9f0)
end

@testitem "the Kronecker traits never form the product" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(65)
    n1, n2 = 40, 30
    n = n1 * n2
    sym(k) = (S = randn(k, k); Matrix(Symmetric(S + S') + 2k * I))
    K = PureQPBase.KroneckerOperator(sym(n1), sym(n2))
    dense_bytes = sizeof(Float64) * n^2

    # Warm, so compilation is not counted.
    PureQPBase.is_symmetric(K)
    PureQPBase.is_convex(Float64, K, 1.0e-6)
    PureQPBase.check_finite(K, n, n, "P")
    @test (@allocated PureQPBase.is_convex(Float64, K, 1.0e-6)) < dense_bytes ÷ 10
    @test (@allocated PureQPBase.check_finite(K, n, n, "P")) < dense_bytes ÷ 100

    # The same holds through `validate`, which `setup` runs first.
    A = randn(5, n)
    q, l, u = randn(n), fill(-1.0, 5), fill(1.0, 5)
    PureQPBase.validate(K, q, A, l, u)
    @test (@allocated PureQPBase.validate(K, q, A, l, u)) < dense_bytes ÷ 10
end
