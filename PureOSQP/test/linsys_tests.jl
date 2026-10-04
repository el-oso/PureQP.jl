@testitem "an ill-conditioned A is refused at setup, and solved with linsys = :kkt" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(6)
    n, m = 40, 120
    U = Matrix(qr(randn(m, n)).Q)[:, 1:n]
    V = Matrix(qr(randn(n, n)).Q)
    A = U * Diagonal(exp10.(range(0, 11; length = n))) * V'
    P = zeros(n, n)
    q = randn(n)
    # The reduced matrix squares `cond(A)` and its Cholesky fails. Setup names the backend
    # that does not square it instead of switching to it.
    @test_throws "Rebuild the workspace with linsys = :kkt" setup(P, q, A, -ones(m), ones(m); scaling = 0)
    ws = setup(P, q, A, -ones(m), ones(m); scaling = 0, linsys = :kkt)
    @test ws.linsys isa PureOSQP.FullKKT
end

@testitem "both backends reach the same solution" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(15, 40; seed = 7)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    chol = PureOSQP.solve(P, q, A, l, u; linsys = :auto, opts...)
    kkt = PureOSQP.solve(P, q, A, l, u; linsys = :kkt, opts...)
    @test chol.status == SOLVED
    @test kkt.status == SOLVED
    @test chol.x ≈ kkt.x rtol = 1.0e-5
    @test abs(chol.obj_val - kkt.obj_val) <= 1.0e-6 * max(1, abs(kkt.obj_val))
end

@testitem "linsys rejects an unknown backend" begin
    @test_throws "linsys must be one of" setup([1.0;;], [0.0], [1.0;;], [0.0], [1.0]; linsys = :magic)
end

@testitem "a named linsys option solves what :auto solves" begin
    using LinearAlgebra, Random
    Random.seed!(74)
    n = 20
    P = Diagonal(rand(n) .+ 1)
    A = Diagonal(rand(n))
    q, l, u = randn(n), -rand(n), rand(n)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    named = solve(P, q, A, l, u; opts..., linsys = :diagonal)
    auto = solve(P, q, A, l, u; opts...)
    @test named.status == SOLVED
    @test named.iter == auto.iter
    @test named.x ≈ auto.x rtol = 1.0e-6
end

@testitem "a named linsys = :sparse is an instruction, not a hint" begin
    using PureIPM
    using LinearAlgebra, SparseArrays, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))
    sparse_names = (SPARSE_FACTOR_BACKENDS..., SPARSE_KKT_BACKENDS..., :sparse_formed)

    # Random QP (n=50, m=500) and Control (n=320, m=540) both have a sparse A whose reduced
    # or KKT form falls to the dense terminal under `:auto`'s IPM thresholds: naming the
    # backend must build it regardless, for both algorithms.
    for (P, q, A, l, u) in (random_qp(50), control(20)), alg in (OperatorSplitting(), InteriorPoint())
        ws = setup(P, q, A, l, u, alg; linsys = :sparse)
        @test PureOSQP.backend_name(ws.linsys) in sparse_names
        @test solve!(ws).status == SOLVED
    end

    # A dense A is converted rather than refused: the named kind is the instruction, and the
    # caller pays for meeting it with a factorization that stores an index per entry. The answer
    # is the one the dense backend gives for the same pair.
    P, q, A, l, u = random_qp(50)
    for alg in (OperatorSplitting(), InteriorPoint())
        ws = setup(P, q, Matrix(A), l, u, alg; linsys = :sparse)
        @test PureOSQP.backend_name(ws.linsys) in sparse_names
        sol = solve!(ws)
        @test sol.status == SOLVED
        @test sol.x ≈ solve(P, q, Matrix(A), l, u, alg; linsys = :dense).x atol = 1.0e-6
    end

    # An operand with no entries to read has nothing to convert, and the refusal names the
    # representation the sparse rungs need.
    @test_throws "SparseMatrixCSC" setup(
        P, q, PureOSQP.ProductOperator{Float64}(Matrix(A)), l, u, OperatorSplitting();
        linsys = :sparse, scaling = 0
    )
end

@testitem "a sparse A solves what the dense one solves, step for step" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(4)
    n, m = 60, 120
    A = sprandn(m, n, 0.05)
    S = sprandn(n, n, 0.05)
    P = sparse(Symmetric(S'S)) + (n * 0.05 + 1) * I
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    # The sparse backend forms the same matrix the dense product forms, so the whole run must
    # agree with the dense backend step for step -- not merely land on the same answer.
    sp = solve(P, q, A, l, u; opts...)
    dn = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test sp.status == SOLVED
    @test sp.iter == dn.iter
    @test sp.x ≈ dn.x atol = 1.0e-10
    @test sp.obj_val ≈ dn.obj_val atol = 1.0e-10
    @test maximum(kkt_residuals(P, q, A, l, u, sp.x, sp.y)) < 1.0e-7
end

@testitem "the sparse backend survives a data update" begin
    using LinearAlgebra, SparseArrays, Random
    # The transpose the accumulation walks is rebuilt per factorization rather than cached,
    # because `update!` may replace A with a different matrix entirely.
    Random.seed!(6)
    n, m = 30, 60
    A = sprandn(m, n, 0.08)
    P = sparse(2.0I, n, n)
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)
    A2 = sprandn(m, n, 0.08)
    b2 = A2 * randn(n)
    l2, u2 = b2 .- rand(m), b2 .+ rand(m)

    # Both workspaces run the identical sequence, because `update!` keeps the iterate and a
    # warm-started solve takes a different path from a cold one. Comparing against a fresh
    # solve would compare warm against cold rather than sparse against dense.
    sparse_ws = setup(P, q, A, l, u; opts...)
    dense_ws = setup(Matrix(P), q, Matrix(A), l, u; opts...)
    @test PureOSQP.backend_name(sparse_ws.linsys) == :sparse_formed
    for w in (sparse_ws, dense_ws)
        solve!(w)
    end
    update!(sparse_ws; A = A2, l = l2, u = u2)
    update!(dense_ws; A = Matrix(A2), l = l2, u = u2)
    s = solve!(sparse_ws)
    ref = solve!(dense_ws)

    @test s.status == SOLVED
    @test s.iter == ref.iter
    @test s.x ≈ ref.x atol = 1.0e-10
end

@testitem "a banded problem solves as the dense one does" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = banded_qp(200, 400; band = 1)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) in SPARSE_FACTOR_BACKENDS
    sp = PureOSQP.solve!(ws)
    dn = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test sp.status == SOLVED
    # A different factorization of the same matrix, so the path must be identical, not
    # merely the answer.
    @test sp.iter == dn.iter
    @test sp.x ≈ dn.x atol = 1.0e-10
    @test sp.obj_val ≈ dn.obj_val atol = 1.0e-10
    @test maximum(kkt_residuals(P, q, A, l, u, sp.x, sp.y)) < 1.0e-7

    # Overruling the choice with `:dense` changes only the route, not the answer.
    forced = solve(P, q, A, l, u; opts..., linsys = :dense)
    @test forced.status == SOLVED
    @test forced.iter == sp.iter
    @test forced.x ≈ sp.x atol = 1.0e-9
end

@testitem "the sparse factorization survives a change of sparsity pattern" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Refactorization reuses the symbolic analysis, which is only valid while the pattern
    # holds. `update!` may replace A with one shaped differently, and then it must not.
    P, q, A, l, u = banded_qp(200, 400; band = 2)
    P2, _, A2, l2, u2 = banded_qp(200, 400; band = 3, seed = 99)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    sparse_ws = setup(P, q, A, l, u; opts...)
    dense_ws = setup(Matrix(P), q, Matrix(A), l, u; opts...)
    @test PureOSQP.backend_name(sparse_ws.linsys) in SPARSE_FACTOR_BACKENDS
    for w in (sparse_ws, dense_ws)
        solve!(w)
    end
    update!(sparse_ws; A = A2, l = l2, u = u2)
    update!(dense_ws; A = Matrix(A2), l = l2, u = u2)
    s = solve!(sparse_ws)
    ref = solve!(dense_ws)

    @test s.status == SOLVED
    @test s.iter == ref.iter
    @test s.x ≈ ref.x atol = 1.0e-10
end

@testitem "a dense row in A solves through the full KKT" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # This is the OSQP suite's Portfolio shape: a budget constraint over every variable.
    Random.seed!(12)
    n, k = 150, 3
    F = sprandn(n, k, 0.5)
    P = blockdiag(spdiagm(0 => rand(n) .+ 1), sparse(2.0I, k, k))
    q = vcat(randn(n), zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n)), spzeros(1, k)),
        hcat(sparse(F'), sparse(-1.0I, k, k)),
        hcat(sparse(1.0I, n, n), spzeros(n, k)),
    )
    l = vcat(1.0, zeros(k), zeros(n))
    u = vcat(1.0, zeros(k), ones(n))
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) in SPARSE_KKT_BACKENDS
    s = PureOSQP.solve!(ws)
    @test s.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-6

    # A different factorization of the same system, so it must agree with the dense one.
    ref = solve(P, q, A, l, u; opts..., linsys = :dense)
    @test ref.status == SOLVED
    @test s.obj_val ≈ ref.obj_val rtol = 1.0e-7
    @test s.x ≈ ref.x atol = 1.0e-6
end

@testitem "the full-KKT backend survives a pattern change" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Refactorization reuses the symbolic analysis, which holds only while the pattern does.
    # `n` is large enough that the full KKT is the better form: below about n = 100 a dense
    # `symv` on the reduced system beats a sparse triangular solve on one of size n + m,
    # and the fill gate says so rather than taking the sparse route regardless.
    Random.seed!(13)
    n = 250
    P = spdiagm(0 => rand(n) .+ 1)
    q = randn(n)
    dense_row = sparse(ones(1, n))
    A = vcat(dense_row, sparse(1.0I, n, n))
    l = vcat(1.0, zeros(n))
    u = vcat(1.0, ones(n))
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) in SPARSE_KKT_BACKENDS
    @test solve!(ws).status == SOLVED

    A2 = vcat(dense_row, sparse(2.0I, n, n))
    update!(ws; A = A2)
    s = solve!(ws)
    @test s.status == SOLVED
    ref = solve(P, q, A2, l, u; opts..., linsys = :dense)
    @test s.obj_val ≈ ref.obj_val rtol = 1.0e-7
end

@testitem "the diagonal backend agrees with the dense one end to end" begin
    using LinearAlgebra, Random
    Random.seed!(12)
    n = 20
    d, a = rand(n) .+ 0.5, rand(n) .+ 0.5
    q, l, u = randn(n), -rand(n), rand(n)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    structured = solve(Diagonal(d), q, Diagonal(a), l, u; opts...)
    dense = solve(Matrix(Diagonal(d)), q, Matrix(Diagonal(a)), l, u; opts...)
    @test structured.status == PureOSQP.SOLVED
    @test dense.status == PureOSQP.SOLVED
    @test structured.x ≈ dense.x rtol = 1.0e-6
    @test structured.iter == dense.iter
end

@testitem "an indefinite P is refused, whatever its storage" begin
    using LinearAlgebra
    n = 5
    diagonal = Diagonal([1.0, 1.0, -3.0, 1.0, 1.0])
    @test_throws "convex" setup(diagonal, randn(n), Diagonal(ones(n)), -ones(n), ones(n))
    tri = SymTridiagonal([1.0, -6.0, 1.0, 1.0], fill(0.05, 3))
    @test_throws "convex" setup(tri, randn(4), Diagonal(ones(4)), -ones(4), ones(4))
end

@testitem "the tridiagonal backend agrees with the dense one end to end" begin
    using LinearAlgebra, Random
    Random.seed!(22)
    n = 25
    P = SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8)
    A = Diagonal(rand(n) .+ 0.5)
    q, l, u = randn(n), -rand(n), rand(n)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    structured = solve(P, q, A, l, u; opts...)
    dense = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test structured.status == PureOSQP.SOLVED
    @test structured.x ≈ dense.x rtol = 1.0e-6
    @test structured.iter == dense.iter
end

@testitem "the low-rank backend agrees with the dense one end to end" begin
    using LinearAlgebra, Random
    Random.seed!(42)
    n, k, m0 = 60, 4, 60
    C = randn(k, n)
    A = PureOSQP.RowCoupled(C, m0)
    dense_A = [C; Matrix(1.0I, m0, n)]
    P = Diagonal(rand(n) .+ 1)
    q, l, u = randn(n), -rand(k + m0), rand(k + m0)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    low = solve(P, q, A, l, u; opts...)
    dense = solve(Matrix(P), q, dense_A, l, u; opts...)
    @test low.status == PureOSQP.SOLVED
    @test low.iter == dense.iter
    @test low.x ≈ dense.x rtol = 1.0e-6
end

@testitem "setup runs is_convex rather than leaving it to the factorization" begin
    using LinearAlgebra, Random
    # The reduced matrix is `c D P D + σI + Ãᵀ diag(ρ) Ã`, so a large enough `ρ` makes it
    # positive definite over an indefinite `P`. `factorize!` therefore cannot stand in for
    # the convexity test.
    sigma = 1.0e-6
    rng = MersenneTwister(77)
    n = 40
    ev = rand(rng, n - 1) ./ 8
    dv = rand(rng, n) .+ 2.0
    bad = SymTridiagonal(dv .- 5.0, copy(ev))
    ws = setup(
        SymTridiagonal(copy(dv), copy(ev)), randn(rng, n), Diagonal(ones(n)),
        -rand(rng, n), rand(rng, n), OperatorSplitting(; sigma); scaling = 0,
    )
    @test PureOSQP.backend_name(ws.linsys) === :tridiagonal
    @test !PureOSQP.is_convex(Float64, bad, sigma)
    # The problem is immutable, so a replaced `P` comes back as another one to hold.
    ws.prob = PureOSQP.adopt_update!(ws.prob; P = bad)
    PureOSQP.set_rho_vec!(ws, 50.0)
    @test PureOSQP.factorize!(ws.linsys, ws.prob, ws.weights)
end

@testitem "a products-only operator solves what its matrix solves" begin
    using LinearAlgebra, Random, Krylov

    # An operator that supplies products and nothing else. It is an `AbstractMatrix` so that
    # `setup` accepts it, but it defines no `getindex`; `is_materializable` is how it says
    # so, and the reference matrix is kept beside it only so the test has something to
    # compare against.
    struct ProductsOnly{T} <: AbstractMatrix{T}
        m::Matrix{T}
    end
    Base.size(op::ProductsOnly) = size(op.m)
    LinearAlgebra.mul!(y::AbstractVector, op::ProductsOnly, x::AbstractVector) = mul!(y, op.m, x)
    LinearAlgebra.mul!(
        y::AbstractVector, op::Adjoint{<:Any, <:ProductsOnly}, x::AbstractVector
    ) = mul!(y, parent(op).m', x)
    PureOSQP.is_materializable(::ProductsOnly) = false
    LinearAlgebra.issymmetric(op::ProductsOnly) = issymmetric(op.m)
    PureOSQP.is_convex(::Type{T}, op::ProductsOnly, sigma) where {T} =
        PureOSQP.is_convex(T, op.m, sigma)
    PureOSQP.reduced_diagonal!(
        dest, ::Type{T}, P::ProductsOnly, A::ProductsOnly, rho, E, D, sigma, c
    ) where {T} = PureOSQP.reduced_diagonal!(dest, T, P.m, A.m, rho, E, D, sigma, c)

    Random.seed!(31)
    n, m = 20, 40
    X = randn(n, n)
    Pm = Matrix(X'X / n + I)
    Am = randn(m, n)
    q = randn(n)
    b = Am * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    ws = setup(ProductsOnly(Pm), q, ProductsOnly(Am), l, u; opts..., linsys = :auto)
    @test PureOSQP.backend_name(ws.linsys) === :indirect
    lazy = PureOSQP.solve!(ws)
    ref = PureOSQP.solve(Pm, q, Am, l, u; opts...)
    @test lazy.status == SOLVED
    @test ref.status == SOLVED
    # The inner solve is inexact, so the two take different iterates to the same answer.
    @test lazy.x ≈ ref.x atol = 1.0e-5
    @test lazy.obj_val ≈ ref.obj_val atol = 1.0e-5
end

@testitem "a failed factorization suggests the full KKT system only when it is not already in use" begin
    P = [4.0 1.0; 1.0 2.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l, u = [1.0, 0.0, 0.0], [1.0, 0.7, 0.7]
    @test_throws "Rebuild the workspace with linsys = :kkt" PureOSQP.refactored!(setup(P, [1.0, 1.0], A, l, u), false)
    @test_throws "already the full KKT system" PureOSQP.refactored!(
        setup(P, [1.0, 1.0], A, l, u; linsys = :kkt), false
    )
end
