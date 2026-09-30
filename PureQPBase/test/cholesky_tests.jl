@testitem "cholesky_factor solves against P + shift*I for every representation and element type" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    Random.seed!(81)
    spd(T, k) = (S = randn(T, k, k); Matrix(Symmetric(S'S / k + I)))

    for T in (Float64, Float32)
        tol = 200 * eps(T)
        P = spd(T, 6)
        big = randn(T, 9, 9)
        big[2:7, 3:8] .= P
        cases = Any[
            (P, T(0)),
            (P, T(0.5)),
            (Symmetric(P), T(0.5)),
            (view(big, 2:7, 3:8), T(0)),               # strided, not contiguous
            (Diagonal(rand(T, 6) .+ T(0.5)), T(0)),
            (Diagonal(rand(T, 6) .+ T(0.5)), T(0.3)),
            (PureQPBase.BlockDiagonal([spd(T, 3), spd(T, 1), spd(T, 4)]), T(0)),
            (PureQPBase.BlockDiagonal([spd(T, 3), spd(T, 2)]), T(0.2)),
            (PureQPBase.BlockDiagonal([Symmetric(spd(T, 3)), spd(T, 2)]), T(0)),
            (PureQPBase.KroneckerOperator(spd(T, 4), spd(T, 3)), T(0)),
            (PureQPBase.KroneckerOperator(spd(T, 1), spd(T, 5)), T(0)),
        ]
        for (P, shift) in cases
            @test PureQPBase.has_cholesky_factor(P)
            R = PureQPBase.cholesky_factor(P, shift)
            n = size(P, 1)
            @test size(R) == size(P)
            @test eltype(R) == T
            Pd = Matrix(P) + shift * I
            v = randn(T, n)
            ref = Symmetric(Pd) \ v

            # `R⁻¹ R⁻ᵀ v = P⁻¹ v`, and the two solves are each other's transpose.
            w = copy(v)
            @test ldiv!(transpose(R), w) === w
            @test ldiv!(R, w) === w
            @test norm(w - ref) <= tol * norm(ref) * cond(Pd)
            @test norm(Matrix(R)' * Matrix(R) - Pd) <= tol * norm(Pd)
            @test istriu(Matrix(R))

            u = copy(v)
            ldiv!(R, u)
            @test norm(Matrix(R) * u - v) <= tol * norm(v) * cond(Matrix(R))
        end
    end
end

@testitem "a dense scalar multiple of the identity yields its square root without a factorization" begin
    using PureQPBase, LinearAlgebra
    for T in (Float64, Float32)
        P = Matrix{T}(3I, 4, 4)
        R = PureQPBase.cholesky_factor(P, T(1))
        @test R isa UpperTriangular{T, Matrix{T}}
        @test Matrix(R) == Matrix{T}(2I, 4, 4)
        # Same type as the general path, so a caller holds one reduction type.
        S = PureQPBase.cholesky_factor([T(2) T(1); T(1) T(3)], T(0))
        @test typeof(S) == typeof(R)
    end
    @test PureQPBase.scalar_diagonal([2.0 0.0; 0.0 2.0]) == 2.0
    @test isnothing(PureQPBase.scalar_diagonal([2.0 0.1; 0.1 2.0]))
    @test isnothing(PureQPBase.scalar_diagonal([2.0 0.0; 0.0 3.0]))
    @test isnothing(PureQPBase.scalar_diagonal(zeros(0, 0)))
end

@testitem "a Kronecker factor stores two small factors and a scratch, never the product" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(82)
    spd(k) = (S = randn(k, k); Matrix(Symmetric(S'S / k + I)))
    n1, n2 = 40, 30
    P1, P2 = spd(n1), spd(n2)
    R = PureQPBase.cholesky_factor(PureQPBase.KroneckerOperator(P1, P2), 0.0)
    @test R isa PureQPBase.KroneckerCholesky{Float64}
    @test R.R1 ≈ cholesky(Symmetric(P1)).U
    @test R.R2 ≈ cholesky(Symmetric(P2)).U
    # `chol(P₁ ⊗ P₂) = chol(P₁) ⊗ chol(P₂)`
    @test Matrix(R) ≈ cholesky(Symmetric(kron(P1, P2))).U
    @test Base.summarysize(R) < 8 * 4 * (n1^2 + n2^2 + n1 * n2)
    @test Base.summarysize(R) < 8 * (n1 * n2)^2 ÷ 100

    v = randn(n1 * n2)
    @test_throws DimensionMismatch ldiv!(R, zeros(3))
    @test_throws DimensionMismatch ldiv!(transpose(R), zeros(n1 * n2 + 1))
    # The entry lookup agrees with the triangular structure it claims.
    @test R[1, 1] == R.R1[1, 1] * R.R2[1, 1]
    @test iszero(R[n2 + 1, 1])
    @test_throws BoundsError R[n1 * n2 + 1, 1]
end

@testitem "cholesky_factor refuses what has no factor, naming the remedy" begin
    using PureQPBase, LinearAlgebra
    indefinite = [1.0 2.0; 2.0 1.0]
    singular = [1.0 1.0; 1.0 1.0]

    @test_throws "P is not positive definite, which ActiveSet() needs when eps_prox = 0" PureQPBase.cholesky_factor(indefinite, 0.0)
    @test_throws "Pass eps_prox > 0" PureQPBase.cholesky_factor(singular, 0.0)
    @test_throws "P + eps_prox*I is not positive definite" PureQPBase.cholesky_factor(indefinite, 0.5)
    # A positive shift rescues a semidefinite `P`.
    R = PureQPBase.cholesky_factor(singular, 0.5)
    @test Matrix(R)' * Matrix(R) ≈ singular + 0.5I

    @test_throws "P is not positive definite" PureQPBase.cholesky_factor(Matrix(-1.0I, 3, 3), 0.0)
    @test_throws "P is not positive definite" PureQPBase.cholesky_factor(Diagonal([1.0, 0.0]), 0.0)
    @test_throws "P + eps_prox*I is not positive definite" PureQPBase.cholesky_factor(Diagonal([1.0, -1.0]), 0.5)
    @test_throws "P is not positive definite" PureQPBase.cholesky_factor(
        PureQPBase.BlockDiagonal([[2.0 0.0; 0.0 2.0], indefinite]), 0.0
    )
    @test_throws DimensionMismatch PureQPBase.cholesky_factor(ones(2, 3), 0.0)

    K = PureQPBase.KroneckerOperator([2.0 0.5; 0.5 3.0], [1.0 0.2; 0.2 2.0])
    @test_throws "a Kronecker P has no Kronecker Cholesky factor once a shift is added" PureQPBase.cholesky_factor(K, 1.0e-8)
    @test_throws "Pass eps_prox = 0" PureQPBase.cholesky_factor(K, 1.0e-8)
    Kbad = PureQPBase.KroneckerOperator(indefinite, [1.0 0.2; 0.2 2.0])
    @test_throws "each factor of a Kronecker P must be positive definite" PureQPBase.cholesky_factor(Kbad, 0.0)
    # Two negative definite factors give a positive definite product that no factor-wise
    # Cholesky reaches, and the refusal says so.
    Kneg = PureQPBase.KroneckerOperator(-[2.0 0.5; 0.5 3.0], -[1.0 0.2; 0.2 2.0])
    @test isposdef(Matrix(Kneg))
    @test_throws "also positive definite when both factors are negative definite" PureQPBase.cholesky_factor(Kneg, 0.0)
end

@testitem "has_cholesky_factor is true for the representations it factors and false for the rest" begin
    using PureQPBase, LinearAlgebra, SparseArrays
    P = [2.0 0.5; 0.5 3.0]
    @test PureQPBase.has_cholesky_factor(P)
    @test PureQPBase.has_cholesky_factor(Symmetric(P))
    @test PureQPBase.has_cholesky_factor(view(zeros(4, 4), 1:2, 1:2))
    @test PureQPBase.has_cholesky_factor(Diagonal([1.0, 2.0]))
    @test PureQPBase.has_cholesky_factor(PureQPBase.BlockDiagonal([P, P]))
    @test PureQPBase.has_cholesky_factor(PureQPBase.KroneckerOperator(P, P))

    # No entries, only products: a factorization cannot reach it.
    struct ProductsOnlyForCholesky{T}
        M::Matrix{T}
    end
    struct ProductsOnlyForCholeskyAdjoint{T}
        parent::ProductsOnlyForCholesky{T}
    end
    Base.size(o::ProductsOnlyForCholesky) = size(o.M)
    Base.size(o::ProductsOnlyForCholeskyAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::ProductsOnlyForCholesky) = ProductsOnlyForCholeskyAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::ProductsOnlyForCholesky, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::ProductsOnlyForCholeskyAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)
    op = PureQPBase.ProductOperator{Float64}(ProductsOnlyForCholesky(P); symmetric = true, posdef = true)
    @test !PureQPBase.has_cholesky_factor(op)

    # Everything else stays unfactored until something defines how.
    @test !PureQPBase.has_cholesky_factor(sparse(P))
    @test !PureQPBase.has_cholesky_factor(SymTridiagonal([2.0, 3.0], [0.5]))
    @test !PureQPBase.has_cholesky_factor(PureQPBase.BlockDiagonal([Diagonal([1.0, 2.0])]))
    @test !PureQPBase.has_cholesky_factor(1.0I)
end

@testitem "a Kronecker factor solves at a size where the dense product would be 50 MB" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(83)
    spd(k) = (S = randn(k, k); Matrix(Symmetric(S'S / k + I)))
    # `n = 2500`: the dense product is 50 MB, the factor is 40 KB.
    n1 = n2 = 50
    P1, P2 = spd(n1), spd(n2)
    K = PureQPBase.KroneckerOperator(P1, P2)
    R = PureQPBase.cholesky_factor(K, 0.0)
    Rt = transpose(R)
    b = randn(n1 * n2)
    x = copy(b)
    ldiv!(Rt, x)    # R⁻ᵀ b
    ldiv!(R, x)     # R⁻¹ R⁻ᵀ b = P⁻¹ b
    y = similar(x)
    mul!(y, K, x)
    @test norm(y - b) <= 1.0e-9 * norm(b) * max(cond(P1), cond(P2))
    @test Base.summarysize(R) < 100_000
end
