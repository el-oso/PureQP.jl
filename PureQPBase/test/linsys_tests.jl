@testitem "the reduced solve reproduces the full KKT system" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(5)
    n, m = 9, 14
    P = (X = randn(n, n); Matrix(X'X))
    A = randn(m, n)
    prob, wt, ls = backend_for(P, randn(n), A, -rand(m), rand(m); scaling = 0)
    @test ls isa PureQPBase.ReducedCholesky
    bx, bz = randn(n), randn(m)
    x, z = zeros(n), zeros(m)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, A, wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-9
    @test z ≈ A * x rtol = 1.0e-9
end

@testitem "the full KKT backend solves what squaring A cannot" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(6)
    n, m = 40, 120
    U = Matrix(qr(randn(m, n)).Q)[:, 1:n]
    V = Matrix(qr(randn(n, n)).Q)
    A = U * Diagonal(exp10.(range(0, 11; length = n))) * V'
    P = zeros(n, n)
    q = randn(n)
    # The reduced matrix squares `cond(A)` and its Cholesky fails; the full KKT system does
    # not square it.
    prob, wt, ls = backend_for(P, q, A, -ones(m), ones(m); scaling = 0, linsys = :kkt)
    @test ls isa PureQPBase.FullKKT
    bx, bz = randn(n), randn(m)
    x, z = zeros(n), zeros(m)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    # At this conditioning a Float64 `K \ b` is no more trustworthy than the backend, so the
    # reference is computed in extended precision.
    K = kkt_matrix(P, A, wt)
    ref = Float64.(big.(K) \ big.([bx; bz]))[1:n]
    @test norm(x .- ref, Inf) < 1.0e-4 * norm(ref, Inf)
end

@testitem "FullKKT refuses to solve before it has factorized" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(7)
    n, m = 8, 10
    P, q, A, l, u = random_qp(n, m; seed = 7)
    prob, wt, ls = backend_for(P, q, A, l, u; linsys = :kkt, factorize = false)
    @test ls isa PureQPBase.FullKKT
    bx, bz = randn(n), randn(m)
    x, z = zeros(n), zeros(m)
    @test_throws ArgumentError PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    @test_throws ArgumentError PureQPBase.solve_multiplier!(ls, prob, wt, bx, bz, x, z)
    # Once factorized, the same calls succeed.
    @test PureQPBase.factorize!(ls, prob, wt)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
end

@testitem "named linsys options reach their backend, and decline loudly" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    using LDLFactorizations, BandedMatrices, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(74)
    sparse_names = (SPARSE_FACTOR_BACKENDS..., SPARSE_KKT_BACKENDS..., :sparse_formed)
    name(P, q, A, l, u; kwargs...) = PureQPBase.backend_name(last(backend_for(P, q, A, l, u; kwargs...)))

    # `:sparse` serves a CSC A through a sparse backend, and declines a dense pair.
    P, q, A, l, u = random_qp(40, 60; seed = 74)
    @test name(P, q, sparse(A), l, u; linsys = :sparse) in sparse_names
    @test_throws "linsys = :sparse" backend_for(P, q, A, l, u; linsys = :sparse)

    # `:diagonal` and `:tridiagonal` name their backends outright.
    n = 20
    Pd = Diagonal(rand(n) .+ 1)
    Ad = Diagonal(rand(n))
    qd = randn(n)
    ld, ud = -rand(n), rand(n)
    @test name(Pd, qd, Ad, ld, ud; linsys = :diagonal) === :diagonal
    @test_throws "linsys = :diagonal" backend_for(Matrix(Pd), qd, Ad, ld, ud; linsys = :diagonal)
    Pt = SymTridiagonal(rand(n) .+ 1, rand(n - 1) ./ 2)
    @test name(Pt, qd, Ad, ld, ud; linsys = :tridiagonal) === :tridiagonal
    @test_throws "linsys = :tridiagonal" backend_for(Pd, qd, Ad, ld, ud; linsys = :tridiagonal)

    # `:block`, `:kronecker` and `:lowrank` reach their rung and state the condition when the
    # pair does not admit it.
    Kc, nb, mb = 3, 8, 5
    Pb = PureQPBase.BlockDiagonal(
        [
            let S = randn(nb, nb)
                Matrix(Symmetric(S'S ./ nb + 2I))
            end for _ in 1:Kc
        ]
    )
    Ab = PureQPBase.BlockDiagonal([randn(mb, nb) ./ sqrt(nb) for _ in 1:Kc])
    qb = randn(Kc * nb)
    bb = Ab * randn(Kc * nb)
    lb, ub = bb .- rand(Kc * mb), bb .+ rand(Kc * mb)
    @test name(Pb, qb, Ab, lb, ub; linsys = :block) === :block
    @test_throws "linsys = :block" backend_for(Matrix(Pb), qb, Matrix(Ab), lb, ub; linsys = :block)

    A1, A2 = randn(6, 6), randn(5, 5)
    K = PureQPBase.KroneckerOperator(A1, A2)
    nk = 30
    Pk = Diagonal(fill(2.0, nk))
    qk = randn(nk)
    bk = kron(A1, A2) * randn(nk)
    lk, uk = bk .- rand(nk), bk .+ rand(nk)
    @test name(Pk, qk, K, lk, uk; scaling = 0, linsys = :kronecker) === :kronecker
    # Equilibration in force is one of the conditions the diagonalization needs.
    @test_throws "linsys = :kronecker" backend_for(Pk, qk, K, lk, uk; linsys = :kronecker)

    nn, kk = 40, 3
    Pn = Diagonal(rand(nn) .+ 1)
    An = PureQPBase.RowCoupled(randn(kk, nn), nn - kk)
    qn = randn(nn)
    bn = An * randn(nn)
    ln, un = bn .- rand(nn), bn .+ rand(nn)
    @test name(Pn, qn, An, ln, un; linsys = :lowrank) === :lowrank
    @test_throws "linsys = :lowrank" backend_for(Matrix(Pn), qn, Matrix(An), ln, un; linsys = :lowrank)
end

@testitem "the LinearSystem contract is enforced, not decorative" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random, Krylov, BandedMatrices, TypeContracts
    include(joinpath(@__DIR__, "helpers.jl"))
    LS = PureQPBase.LinearSystem
    spec = TypeContracts.list_contract(LS)
    @test [nameof(s.f) for s in spec if !s.optional] == [:factorize!, :solve_system!, :backend_info]
    @test [nameof(s.f) for s in spec if s.optional] == [
        :refactor_weights!, :solve_multiplier!, :check_update, :set_tolerance_level!,
        :set_refresh_index!, :adopt_settings!, :use_residual_stop!, :last_solve_converged,
        :inner_iterations,
    ]

    # Both backends this package ships satisfy it.
    for B in (
            PureQPBase.ReducedCholesky{Float64, Matrix{Float64}},
            PureQPBase.FullKKT{
                Float64, Matrix{Float64}, Vector{Float64},
                BunchKaufman{Float64, Matrix{Float64}, Vector{Int}},
            },
        )
        @test TypeContracts.satisfies(B, LS).satisfied
    end

    # The two backends declared in weak-dependency extensions satisfy the same contract:
    # `IndirectCG` (Krylov.jl, the matrix-free path) and `BandedReduced` (BandedMatrices.jl).
    P, q, A, l, u = random_qp(10, 15; seed = 5)
    _, _, cg = backend_for(P, q, A, l, u; linsys = :indirect, scaling = 0, factorize = false)
    @test TypeContracts.satisfies(typeof(cg), LS).satisfied

    Pt = SymTridiagonal(rand(20) .+ 4, rand(19) ./ 8)
    At = Tridiagonal(rand(19) ./ 4, rand(20) .+ 1, rand(19) ./ 4)
    _, _, banded = backend_for(Pt, randn(20), At, -rand(20), rand(20))
    @test PureQPBase.backend_name(banded) === :banded
    @test TypeContracts.satisfies(typeof(banded), LS).satisfied

    # And a type that declares the supertype without implementing it is rejected. Without
    # this the contract could be satisfied vacuously and nobody would notice.
    @eval struct IncompleteBackend <: PureQPBase.LinearSystem end
    @test !TypeContracts.satisfies(IncompleteBackend, LS).satisfied
    @test length(TypeContracts.satisfies(IncompleteBackend, LS).missing_methods) == 3
    @test_throws TypeContracts.InterfaceError TypeContracts.check_contract(IncompleteBackend, LS)
end

@testitem "the representation alone decides the sparse route" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(4)
    n, m = 60, 120
    A = sprandn(m, n, 0.05)
    S = sprandn(n, n, 0.05)
    P = sparse(Symmetric(S'S)) + (n * 0.05 + 1) * I
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)

    _, _, ls = backend_for(P, q, A, l, u)
    @test PureQPBase.backend_name(ls) == :sparse_formed
    # The point of the backend: no m×n buffer exists to hold a densified A.
    @test !hasproperty(ls, :W)

    # A reduced matrix this full is inverted densely either way, and accumulating over the
    # stored entries gets there without that buffer — so the representation alone decides,
    # at any density.
    Random.seed!(5)
    nd, md = 40, 80
    Af = sprandn(md, nd, 0.6)
    Pf = sparse(1.0I, nd, nd)
    bf = Af * randn(nd)
    qf, lf, uf = randn(nd), bf .- rand(md), bf .+ rand(md)
    @test PureQPBase.backend_name(last(backend_for(Pf, qf, Af, lf, uf))) == :sparse_formed
    @test PureQPBase.backend_name(
        last(backend_for(Matrix(Pf), qf, Matrix(Af), lf, uf))
    ) == :cholesky

    # Random sparsity has no separators, so the Cholesky factor fills in almost completely
    # and the dense inverse wins the per-iteration solve. The backend is chosen by asking the
    # factorization what the fill actually is, so this is decided rather than assumed.
    Random.seed!(11)
    nb, mb = 150, 300
    Ab = sprandn(mb, nb, 0.05)
    Sb = sprandn(nb, nb, 0.05)
    Pb = sparse(Symmetric(Sb'Sb)) + (nb * 0.05 + 1) * I
    bb = Ab * randn(nb)
    @test PureQPBase.backend_name(
        last(backend_for(Pb, randn(nb), Ab, bb .- rand(mb), bb .+ rand(mb)))
    ) == :sparse_formed
end

@testitem "a banded problem is factored sparsely, and its solve allocates nothing" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    # Banded is what MPC and other structured QPs look like, and a narrow band is where a
    # sparse factorization earns its place: the reduced matrix stays banded, so its Cholesky
    # factor does too. The dense backend would invert an n×n matrix instead.
    P, q, A, l, u = banded_qp(200, 400; band = 2)
    prob, wt, ls = backend_for(P, q, A, l, u)
    @test PureQPBase.backend_name(ls) in SPARSE_FACTOR_BACKENDS

    # CHOLMOD's own `ldiv!` allocates a result and workspace on every call, which the hot
    # path may not do, so the backend applies the permutation and the two triangular solves
    # itself over buffers it owns.
    n, m = prob.n, prob.m
    bx, bz, x, z = randn(n), randn(m), zeros(n), zeros(m)
    # Measured inside a function: from top-level scope the call is a dynamic dispatch on an
    # untyped global, and that dispatch allocates 32 bytes of its own.
    solve_bytes() = @allocated PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    solve_bytes()      # warm up
    @test all(iszero, [solve_bytes() for _ in 1:4])
end

@testitem "linsys = :dense overrules the representation gates" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    # The rule for a sparse A is fitted to a benchmark suite, so there has to be a way to
    # overrule it on a problem it misjudges. `:dense` goes past `choose_backend` entirely.
    P, q, A, l, u = banded_qp(200, 400; band = 1)
    @test PureQPBase.backend_name(last(backend_for(P, q, A, l, u))) in SPARSE_FACTOR_BACKENDS
    @test PureQPBase.backend_name(last(backend_for(P, q, A, l, u; linsys = :dense))) == :cholesky
end

@testitem "a dense row in A routes to the full KKT" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    # Eliminating to the reduced system squares A, so one dense row makes the reduced matrix
    # dense however sparse the rest of it is. The full KKT keeps that row as one sparse row.
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
    @test PureQPBase.backend_name(last(backend_for(P, q, A, l, u))) in SPARSE_KKT_BACKENDS
end

@testitem "the unchecked substitutions run behind a guard that fires" begin
    using PureQPBase
    using LinearAlgebra, SparseArrays, LDLFactorizations
    # `unit_forward!`/`unit_backward!` index `x` by a row read out of the factor, which no
    # compiler can prove is in range, so they drop the check and `check_factor` establishes
    # the property once per factorization instead. That trade is only sound while the guard
    # actually rejects a factor the loops would read out of bounds.
    # The guard lives with the substitutions it protects, which both factorization engines
    # share, so it sits in the SparseArrays extension rather than either engine's.
    Ext = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
    @test !isnothing(Ext)

    ok = SparseMatrixCSC(4, 4, [1, 2, 2, 2, 2], [3], [1.0])
    @test isnothing(Ext.check_factor(ok, 4))

    # A row index past the end of the system: the very access the loops no longer check.
    @test_throws "outside 1:4" Ext.check_factor(SparseMatrixCSC(4, 4, [1, 2, 2, 2, 2], [9], [1.0]), 4)
    # And one before its start.
    @test_throws "outside 1:4" Ext.check_factor(SparseMatrixCSC(4, 4, [1, 2, 2, 2, 2], [0], [1.0]), 4)
    # A factor whose order does not match the system would walk `colptr` off its end. The
    # other malformation, a column pointer inconsistent with the stored entries, cannot be
    # reached: `SparseMatrixCSC` rejects it in its own constructor.
    @test_throws "malformed column pointer" Ext.check_factor(ok, 8)
end

@testitem "a diagonal P and A solve through the diagonal backend" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(11)
    n = 12
    P = Diagonal(rand(n) .+ 0.5)
    A = Diagonal(rand(n) .+ 0.5)
    prob, wt, ls = backend_for(P, randn(n), A, -rand(n), rand(n); scaling = 0)
    @test ls isa PureQPBase.DiagonalReduced
    @test PureQPBase.backend_name(ls) == :diagonal

    # Against the same system the dense backend would have built and factored.
    bx, bz = randn(n), randn(n)
    x, z = zeros(n), zeros(n)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, A, wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-9
    @test z ≈ A * x rtol = 1.0e-9
end

@testitem "a bandwidth-one reduced system solves through the tridiagonal backend" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(21)
    n = 14
    cases = (
        (
            "SymTridiagonal P, Diagonal A",
            SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5),
        ),
        (
            "Diagonal P, Bidiagonal A",
            Diagonal(rand(n) .+ 2), Bidiagonal(rand(n) .+ 1, rand(n - 1), :U),
        ),
        (
            "SymTridiagonal P, Bidiagonal A",
            SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Bidiagonal(rand(n) .+ 1, rand(n - 1), :L),
        ),
    )
    for (name, P, A) in cases
        prob, wt, ls = backend_for(P, randn(n), A, -rand(n), rand(n); scaling = 0)
        @test ls isa PureQPBase.TridiagonalReduced
        @test PureQPBase.backend_name(ls) == :tridiagonal
        # The bands must equal the reduced matrix the dense backend would have formed.
        R = reduced_matrix(P, A, wt)
        @test ls.dv ≈ diag(R) rtol = 1.0e-12
        @test ls.ev ≈ diag(R, 1) rtol = 1.0e-12
        bx, bz = randn(n), randn(n)
        x, z = zeros(n), zeros(n)
        PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
        ref = kkt_matrix(P, A, wt) \ [bx; bz]
        @test x ≈ ref[1:n] rtol = 1.0e-9
        @test z ≈ A * x rtol = 1.0e-9
    end
end

@testitem "a diagonal core with coupling rows solves through the low-rank backend" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(41)
    n, k, m0 = 40, 3, 40
    P = Diagonal(rand(n) .+ 1)
    A = PureQPBase.RowCoupled(randn(k, n), m0)
    @test size(A) == (k + m0, n)
    # The type must agree with the matrix it stands for, entry by entry.
    dense_A = [A.coupling; Matrix(1.0I, m0, n)]
    @test Matrix(A) == dense_A

    prob, wt, ls = backend_for(P, randn(n), A, -rand(k + m0), rand(k + m0); scaling = 0)
    @test ls isa PureQPBase.DiagonalLowRank
    @test PureQPBase.backend_name(ls) == :lowrank

    # Against the full KKT system the backend stands for.
    bx, bz = randn(n), randn(k + m0)
    x, z = zeros(n), zeros(k + m0)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, dense_A, wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-9
    @test z ≈ dense_A * x rtol = 1.0e-9
end

@testitem "the low-rank rung declines a correction too wide to pay" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(43)
    n = 20

    "The rung's verdict for a coupling of rank `k`."
    function rung(k)
        A = PureQPBase.RowCoupled(randn(k, n), n)
        m = k + n
        prob = raw_problem(Diagonal(ones(n)), A, n, m)
        wt = raw_weights(ones(m), 1.0e-6)
        return PureQPBase.lowrank_rung(Diagonal(ones(n)), A, prob, wt, PureQPBase.ADMMSelection())
    end

    # The limit is `10k <= n`, which is `k <= 2` here.
    @test isnothing(rung(3))
    @test isnothing(rung(n ÷ 2))
    ls, factored = rung(2)
    @test ls isa PureQPBase.DiagonalLowRank
    @test !factored
end

@testitem "a weights update leaves the same factorization as a full rebuild" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # `refactor_weights!` exists so a changed weight does not cost a full rebuild. It is
    # only worth having while the two produce the same factorization.
    Random.seed!(44)
    n, k = 60, 3
    P = Diagonal(rand(n) .+ 0.5)
    A = PureQPBase.RowCoupled(randn(k, n) ./ 4, ones(n - k), collect(1:(n - k)))
    prob, wt, ls = backend_for(P, randn(n), A, -rand(n), rand(n))
    m = prob.m
    bx, bz = randn(n), randn(m)

    "The solve of a fixed right-hand side, which is what the factorization is for."
    function applied()
        x, z = zeros(n), zeros(m)
        PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
        return x
    end

    fill!(wt.w, 3.7)
    wt.w_inv .= inv.(wt.w)
    @test PureQPBase.refactor_weights!(ls, prob, wt)
    cheap = applied()
    @test PureQPBase.factorize!(ls, prob, wt)
    @test cheap ≈ applied() rtol = 1.0e-12
end

@testitem "is_convex answers without densifying for Tridiagonal and banded-symmetric P" begin
    using PureQPBase, LinearAlgebra, Random, BandedMatrices

    "The generic method's answer: densify, shift and factor."
    reference(M, sigma) = issuccess(cholesky!(Symmetric(Matrix{Float64}(M) + sigma * I); check = false))

    sigma = 1.0e-6
    rng = MersenneTwister(77)
    n, b = 40, 3

    "A symmetric band of half-width `b`, shifted on the diagonal."
    function band_matrix(shift)
        B = BandedMatrix{Float64}(undef, (n, n), (b, b))
        fill!(B.data, 0.0)
        B[band(0)] .= rand(MersenneTwister(78), n) .+ 2b .+ shift
        for k in 1:b
            B[band(k)] .= rand(MersenneTwister(78 + k), n - k) ./ (4b)
            B[band(-k)] .= B[band(k)]
        end
        return Symmetric(B)
    end

    ev = rand(rng, n - 1) ./ 8
    dv = rand(rng, n) .+ 2.0
    definite = (Tridiagonal(copy(ev), copy(dv), copy(ev)), band_matrix(0.0))
    indefinite = (Tridiagonal(copy(ev), dv .- 5.0, copy(ev)), band_matrix(-3.0 * 2b))

    for P in definite
        @test PureQPBase.is_convex(Float64, P, sigma)
        @test reference(P, sigma)
    end
    for P in indefinite
        @test !PureQPBase.is_convex(Float64, P, sigma)
        @test !reference(P, sigma)
    end

    # A method of its own, not the densifying `AbstractMatrix` fallback.
    generic = which(PureQPBase.is_convex, Tuple{Type{Float64}, AbstractMatrix, Float64})
    for P in definite
        @test which(PureQPBase.is_convex, Tuple{Type{Float64}, typeof(P), Float64}) !== generic
    end
end

@testitem "a products-only operator routes to the indirect rung" begin
    using PureQPBase, LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))

    # An operator that supplies products and nothing else. It is an `AbstractMatrix` so that
    # the problem accepts it, but it defines no `getindex`; `is_materializable` is how it
    # says so, and the reference matrix is kept beside it only so the test has something to
    # compare against.
    struct ProductsOnly{T} <: AbstractMatrix{T}
        m::Matrix{T}
    end
    Base.size(op::ProductsOnly) = size(op.m)
    LinearAlgebra.mul!(y::AbstractVector, op::ProductsOnly, x::AbstractVector) = mul!(y, op.m, x)
    LinearAlgebra.mul!(
        y::AbstractVector, op::Adjoint{<:Any, <:ProductsOnly}, x::AbstractVector
    ) = mul!(y, parent(op).m', x)
    PureQPBase.is_materializable(::ProductsOnly) = false
    LinearAlgebra.issymmetric(op::ProductsOnly) = issymmetric(op.m)

    Random.seed!(31)
    n, m = 20, 40
    X = randn(n, n)
    Pm = Matrix(X'X / n + I)
    Am = randn(m, n)

    # Falling through past the dense terminal without Krylov is a named refusal rather than
    # a `MethodError` from inside a factorization. `invoke` reaches the core method whether
    # or not the extension has added its own, so this holds in either load state.
    @test_throws "needs Krylov.jl" invoke(
        PureQPBase.indirect_backend, Tuple{AbstractVector, Integer, Integer, Any}, randn(n), n, m, nothing
    )

    # The trait is what declines rung 6, and it declines on either operand alone.
    sel = PureQPBase.ADMMSelection()
    @test isnothing(PureQPBase.dense_rung(ProductsOnly(Pm), Am, raw_problem(ProductsOnly(Pm), Am, n, m), sel))
    @test isnothing(PureQPBase.dense_rung(Pm, ProductsOnly(Am), raw_problem(Pm, ProductsOnly(Am), n, m), sel))
    @test PureQPBase.dense_rung(Pm, Am, raw_problem(Pm, Am, n, m), sel)[1] isa PureQPBase.ReducedCholesky
end

@testitem "solve_multiplier! recovers ν directly on FullKKT as w_inv shrinks" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 16; seed = 30)
    prob, wt, ls = backend_for(P, q, A, l, u; linsys = :kkt, scaling = 0)
    @test ls isa PureQPBase.FullKKT
    # This is the w_inv an interior-point method's dual regularization reaches on an active
    # inequality row, where the default path's cancellation below is worst.
    for i in (1, 2)
        wt.w_inv[i] = 1.0e-12
        wt.w[i] = inv(wt.w_inv[i])
    end
    @test PureQPBase.factorize!(ls, prob, wt)
    n, m = prob.n, prob.m
    bx, bz = randn(n), randn(m)
    x, nu = similar(bx), similar(bz)
    PureQPBase.solve_multiplier!(ls, prob, wt, bx, bz, x, nu)
    @test prob.A * x .- wt.w_inv .* nu ≈ bz atol = 1.0e-9

    # The default recovers ν from z̃ = rhs_z + w_inv ⊙ ν: at this w_inv the addition rounds
    # z̃ back to rhs_z before the subtraction sees ν, so on the rows w_inv was shrunk on it
    # comes back measurably (here ~1e-5 to 1e-4 relative) less accurate than the direct
    # extraction above, which stays at the factorization's own precision.
    x_def, nu_def = similar(bx), similar(bz)
    invoke(
        PureQPBase.solve_multiplier!,
        Tuple{PureQPBase.LinearSystem, Any, Any, Any, Any, Any, Any},
        ls, prob, wt, bx, bz, x_def, nu_def,
    )
    @test x_def ≈ x
    @test abs(nu_def[1] - nu[1]) > 1.0e-6 * abs(nu[1])
    @test abs(nu_def[2] - nu[2]) > 1.0e-6 * abs(nu[2])
end

@testitem "solve_multiplier! recovers ν directly on the sparse KKT backend as w_inv shrinks" begin
    using PureQPBase, LinearAlgebra, SparseArrays, LDLFactorizations, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(8)
    n, m = 200, 100
    # A dense row is what makes the KKT form win over the reduced one, as in the OSQP suite's
    # Portfolio class; the fill gate needs this many columns to accept it.
    A = vcat(sprandn(m - 1, n, 0.02), sparse(ones(1, n)))
    P = sparse(1.0I, n, n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    prob, wt, ls = backend_for(P, q, A, l, u; linsys = :sparse, scaling = 0)
    @test PureQPBase.backend_name(ls) in SPARSE_KKT_BACKENDS
    for i in (1, 2)
        wt.w_inv[i] = 1.0e-12
        wt.w[i] = inv(wt.w_inv[i])
    end
    @test PureQPBase.factorize!(ls, prob, wt)
    bx, bz = randn(n), randn(m)
    x, nu = similar(bx), similar(bz)
    PureQPBase.solve_multiplier!(ls, prob, wt, bx, bz, x, nu)
    @test prob.A * x .- wt.w_inv .* nu ≈ bz atol = 1.0e-9

    x_def, nu_def = similar(bx), similar(bz)
    invoke(
        PureQPBase.solve_multiplier!,
        Tuple{PureQPBase.LinearSystem, Any, Any, Any, Any, Any, Any},
        ls, prob, wt, bx, bz, x_def, nu_def,
    )
    @test x_def ≈ x
    @test abs(nu_def[1] - nu[1]) > 1.0e-6 * abs(nu[1])
    @test abs(nu_def[2] - nu[2]) > 1.0e-6 * abs(nu[2])
end
