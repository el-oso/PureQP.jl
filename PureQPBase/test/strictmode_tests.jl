@testitem "every in-package LinearSystem meets its strict contract, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, TypeContracts, LinearAlgebra, FillArrays, Random
    using InteractiveUtils: subtypes
    include(joinpath(@__DIR__, "helpers.jl"))

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    # `@strict_contract` records `LinearSystem` in a registry that a loaded package image does
    # not carry, so the contract is checked here, implementer by implementer. Every concrete
    # subtype this package defines must appear below; a new backend without a case fails the
    # coverage test rather than going unchecked.
    concrete(T) = isabstracttype(T) ? reduce(vcat, map(concrete, subtypes(T)); init = Type[]) : Type[T]
    implementers = filter(T -> parentmodule(T) === PureQPBase, concrete(PureQPBase.LinearSystem))

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
    Pb = PureQPBase.BlockDiagonal([spd(10) for _ in 1:5])
    Ab = PureQPBase.BlockDiagonal([randn(10, 10) ./ sqrt(10) for _ in 1:5])
    block = (Pb, randn(50), Ab, -rand(50), rand(50))
    Al = PureQPBase.RowCoupled(randn(3, nd) ./ 4, ones(nd - 3), collect(1:(nd - 3)))
    lowrank = (Diagonal(rand(nd) .+ 0.5), randn(nd), Al, -rand(nd), rand(nd))
    Ak = PureQPBase.KroneckerOperator(randn(4, 4), randn(5, 5))
    bk = Matrix(Ak) * randn(20)
    kronecker = (Diagonal(fill(2.0, 20)), randn(20), Ak, bk .- rand(20), bk .+ rand(20))
    cases = [
        (dense, (; linsys = :dense), PureQPBase.ReducedCholesky),
        (dense, (; linsys = :kkt), PureQPBase.FullKKT),
        (diagonal, (; linsys = :diagonal), PureQPBase.DiagonalReduced),
        (tridiagonal, (; linsys = :tridiagonal), PureQPBase.TridiagonalReduced),
        (block, (; linsys = :block), PureQPBase.BlockReduced),
        (lowrank, (; linsys = :lowrank), PureQPBase.DiagonalLowRank),
        (kronecker, (; linsys = :kronecker, scaling = 0), PureQPBase.KroneckerReduced),
    ]
    # `ProductReduced` is reached through the interior-point ladder rather than by naming a
    # `linsys`, so `backend_for` cannot build it and it is proved in its own block below.
    @test Set(map(Base.typename, implementers)) ==
        Set([map(c -> Base.typename(c[3]), cases); Base.typename(PureQPBase.ProductReduced)])

    for (data, kw, LS) in cases
        prob, wt, ls = backend_for(data...; kw...)
        @test ls isa LS
        @test TypeContracts.check_contract(typeof(ls), PureQPBase.LinearSystem).passed
        bx, bz = randn(prob.n), randn(prob.m)
        x, z = zeros(prob.n), zeros(prob.m)
        # Warm every signature on the real data before proving it.
        PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
        PureQPBase.solve_multiplier!(ls, prob, wt, bx, bz, x, z)
        PureQPBase.refactor_weights!(ls, prob, wt)
        types = map(typeof, (ls, prob, wt, bx, bz, x, z))
        per_iteration = [
            (PureQPBase.solve_system!, types),
            (PureQPBase.solve_multiplier!, types),
            (PureQPBase.refactor_weights!, types[1:3]),
        ]
        guarantees = (:typestable, :noalloc, :trim_compatible)
        @test test_signatures(per_iteration; guarantees) isa Vector
        @test test_signatures([(PureQPBase.factorize!, types[1:3])]; guarantees) isa Vector
    end

    # `ProductReduced` assembles the reduced matrix from products, so it is held to the same
    # per-iteration guarantees on both of its assembly paths: the Kronecker contraction, and the
    # products that serve anything else.
    # Built with `raw_problem` rather than `backend_for`: this backend is reached through the
    # interior-point ladder, so no `linsys` name builds it, and naming one that declines these
    # pairs would throw before the proof runs.
    for (label, Pin, Ain) in (
            ("contraction", kronecker[1], Ak),
            ("stacked", kronecker[1], PureQPBase.StackedOperator(Matrix(Ak), Diagonal(fill(1.5, 20)))),
            (
                "joined", kronecker[1],
                PureQPBase.JoinedOperator(PureQPBase.KroneckerOperator(randn(4, 3), randn(5, 4)), Fill(0.5, 20, 8)),
            ),
            ("products", Diagonal(fill(2.0, 20)), PureQPBase.ProductOperator{Float64}(Matrix(Ak))),
        )
        prob = raw_problem(Pin, Ain, 20, size(Ain, 1))
        wt = raw_weights(fill(0.75, size(Ain, 1)), 1.0e-6)
        ls = PureQPBase.ProductReduced(prob.q0, prob.n, prob.m, prob.A)
        @test TypeContracts.check_contract(typeof(ls), PureQPBase.LinearSystem).passed
        @test PureQPBase.factorize!(ls, prob, wt)
        bx, bz = randn(prob.n), randn(prob.m)
        x, z = zeros(prob.n), zeros(prob.m)
        PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
        PureQPBase.solve_multiplier!(ls, prob, wt, bx, bz, x, z)
        types = map(typeof, (ls, prob, wt, bx, bz, x, z))
        @test test_signatures(
            [
                (PureQPBase.factorize!, types[1:3]),
                (PureQPBase.solve_system!, types),
                (PureQPBase.solve_multiplier!, types),
            ];
            guarantees = (:typestable, :noalloc, :trim_compatible)
        ) isa Vector
    end
end

@testitem "the built-in preconditioners meet their strict contract, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, TypeContracts, LinearAlgebra, Random
    using InteractiveUtils: subtypes
    include(joinpath(@__DIR__, "helpers.jl"))
    StrictMode.assert_enabled()

    Random.seed!(2)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    prob, wt, _ = backend_for(P, randn(n), A, -rand(m), rand(m))
    y, x = zeros(n), randn(n)
    # The Kronecker preconditioner is built from two factor pairs, so it needs a Kronecker pair
    # of its own rather than the dense `P` and `A` above.
    k = 4
    g1, g2 = randn(k, k), randn(k, k)
    f1 = Matrix(Symmetric(g1'g1 / k + I))
    f2 = Matrix(Symmetric(g2'g2 / k + I))
    kronP = PureQPBase.KroneckerOperator(f1, f2)
    kronA = PureQPBase.KroneckerOperator(randn(k + 1, k), randn(k + 2, k))
    preconditioners = [
        PureQPBase.IdentityPreconditioner(), PureQPBase.JacobiPreconditioner(zeros(n)),
        PureQPBase.KroneckerPreconditioner(kronP, kronA),
    ]
    defined = filter(T -> parentmodule(T) === PureQPBase, subtypes(PureQPBase.Preconditioner))
    @test Set(map(Base.typename, defined)) == Set(map(M -> Base.typename(typeof(M)), preconditioners))
    for M in preconditioners
        @test TypeContracts.check_contract(typeof(M), PureQPBase.Preconditioner).passed
        # The Kronecker preconditioner acts on `k₁k₂` variables, not on this problem's `n`.
        dim = M isa PureQPBase.KroneckerPreconditioner ? k * k : n
        yv, xv = zeros(dim), randn(dim)
        PureQPBase.update_preconditioner!(M, prob, wt, 0)
        ldiv!(yv, M, xv)
        @test test_signatures(
            [
                (ldiv!, map(typeof, (yv, M, xv))),
                (PureQPBase.update_preconditioner!, map(typeof, (M, prob, wt, 0))),
            ];
            guarantees = (:typestable, :noalloc, :trim_compatible)
        ) isa Vector
    end
end

@testitem "the allocation-free dense factorizations match the library ones exactly" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    # `FullKKT` factors through `sytrf` into arrays it holds; `bunchkaufman!` on the same
    # matrix is the reference, bit for bit, for both LAPACK element types it specializes on.
    for T in (Float64, Float32)
        Random.seed!(3)
        n, m = 12, 30
        X = randn(T, n, n)
        P = Matrix(X'X / n + I)
        A = randn(T, m, n)
        b = A * randn(T, n)
        q, l, u = randn(T, n), b .- rand(T, m), b .+ rand(T, m)

        prob, wt, ls = if T === Float64
            backend_for(P, q, A, l, u; linsys = :kkt)
        else
            probT = PureQPBase.validated_problem(T, n, m, P, q, A, l, u, 10)
            wtT = PureQPBase.SystemWeights(fill(T(0.1), m), fill(T(10), m), T(1.0e-6))
            (probT, wtT, PureQPBase.FullKKT(probT.q, n, m))
        end
        PureQPBase.assemble_kkt0!(ls, prob)
        K = copy(ls.K0)
        for j in 1:n
            K[j, j] += wt.sigma
        end
        for i in 1:m
            K[n + i, n + i] = -wt.w_inv[i]
        end
        ref = bunchkaufman!(Symmetric(K, :L); check = false)
        for _ in 1:2    # the factorization and a refactorization over the same arrays
            @test PureQPBase.refactor_weights!(ls, prob, wt)
            @test ls.fact.LD == ref.LD
            @test ls.fact.ipiv == ref.ipiv
            bx, bz = randn(T, n), randn(T, m)
            x, z = zeros(T, n), zeros(T, m)
            PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
            r = ldiv!(ref, [bx; bz])
            @test x == r[1:n]
        end
    end

    # A zero pivot is reported as a failure, exactly as `bunchkaufman!` reports it.
    n, m = 2, 1
    prob, wt, ls = backend_for(
        zeros(n, n), [1.0, 1.0], [1.0 0.0], [0.0], [1.0];
        linsys = :kkt, scaling = 0, sigma = 0.0, factorize = false
    )
    PureQPBase.assemble_kkt0!(ls, prob)
    K = copy(ls.K0)
    K[3, 3] = -wt.w_inv[1]
    ok = PureQPBase.factorize!(ls, prob, wt)
    @test !ok
    @test ok == issuccess(bunchkaufman!(Symmetric(K, :L); check = false))
end

@testitem "the proofs fail on code that allocates or cannot be trimmed" begin
    using StrictMode, StrictModeTest

    # A gate that passes everything proves nothing; each of these must be refused.
    StrictMode.assert_enabled()
    grow(n) = zeros(n)
    @test_throws StrictMode.StrictViolation test_signatures([(grow, (Int,))]; guarantees = (:noalloc,))
    # A broadcast into a scratch vector keeps an aliasing check whose copy AllocCheck finds,
    # although it is never taken, which is why the dense factorization writes its scratch in a loop.
    scaled!(dst, a, b) = (dst .= sqrt.(a) .* b; nothing)
    V = Vector{Float64}
    @test_throws StrictMode.StrictViolation test_signatures([(scaled!, (V, V, V))]; guarantees = (:noalloc,))
    dynamic(r) = r[] + 1
    @test_throws StrictMode.StrictViolation test_signatures(
        [(dynamic, (Base.RefValue{Any},))]; guarantees = (:trim_compatible,)
    )
end

@testitem "dense_row! allocates nothing and trims on every representation, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, LinearAlgebra

    StrictMode.assert_enabled()
    struct RowsOnlyMap{T}
        M::Matrix{T}
    end
    struct RowsOnlyMapAdjoint{T}
        parent::RowsOnlyMap{T}
    end
    Base.size(o::RowsOnlyMap) = size(o.M)
    Base.size(o::RowsOnlyMapAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::RowsOnlyMap) = RowsOnlyMapAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::RowsOnlyMap, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::RowsOnlyMapAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)

    for T in (Float64, Float32)
        V = Vector{T}
        Kr = typeof(PureQPBase.KroneckerOperator(randn(T, 3, 2), randn(T, 2, 3)))
        Bd = typeof(PureQPBase.BlockDiagonal([randn(T, 2, 2), randn(T, 3, 1)]))
        Po = typeof(PureQPBase.ProductOperator{T}(RowsOnlyMap(randn(T, 4, 3))))
        signatures = [
            (PureQPBase.dense_row!, (V, Matrix{T}, Int)),
            (PureQPBase.dense_row!, (V, Diagonal{T, V}, Int)),
            (PureQPBase.dense_row!, (V, Kr, Int)),
            (PureQPBase.dense_row!, (V, Bd, Int)),
            (PureQPBase.dense_row!, (V, Po, Int)),
        ]
        @test test_signatures(signatures; guarantees = (:noalloc, :trim_compatible)) isa Vector
    end
end

@testitem "both triangular solves of every Cholesky factor allocate nothing and trim, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, TypeContracts, LinearAlgebra, Random
    StrictMode.assert_enabled()
    Random.seed!(5)
    spd(T, k) = (S = randn(T, k, k); Matrix(Symmetric(S'S / k + I)))

    for T in (Float64, Float32)
        cases = Any[
            spd(T, 6),
            Matrix{T}(2I, 6, 6),
            Diagonal(rand(T, 6) .+ T(0.5)),
            PureQPBase.BlockDiagonal([spd(T, 3), spd(T, 2), spd(T, 1)]),
            PureQPBase.KroneckerOperator(spd(T, 4), spd(T, 3)),
        ]
        for P in cases
            R = PureQPBase.cholesky_factor(P, zero(T))
            # `transpose(R)` is formed once and held, as a consumer that solves in a loop does.
            Rt = transpose(R)
            v = randn(T, size(P, 1))
            @test TypeContracts.check_contract(typeof(R), PureQPBase.CholeskyFactor).passed
            ldiv!(R, v)
            ldiv!(Rt, v)
            @test (@allocated ldiv!(R, v)) == 0
            @test (@allocated ldiv!(Rt, v)) == 0
            V = typeof(v)
            @test test_signatures(
                [(ldiv!, (typeof(R), V)), (ldiv!, (typeof(Rt), V))];
                guarantees = (:typestable, :noalloc, :trim_compatible)
            ) isa Vector
        end
    end
end
