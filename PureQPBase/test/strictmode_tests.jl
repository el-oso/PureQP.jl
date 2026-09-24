@testitem "every in-package LinearSystem meets its strict contract, proved" begin
    using PureQPBase, StrictMode, StrictModeTest, TypeContracts, LinearAlgebra, Random
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
    @test Set(map(Base.typename, implementers)) == Set(map(c -> Base.typename(c[3]), cases))

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
        if ls isa PureQPBase.KroneckerReduced
            # Its `factorize!` eigendecomposes `AᵢᵀAᵢ`, which allocates. It runs only when
            # `P`, `A` or `σ` change, and no interior-point rung selects this backend.
            @test test_signatures(
                [(PureQPBase.factorize!, types[1:3])]; guarantees = (:typestable, :trim_compatible)
            ) isa Vector
        else
            @test test_signatures([(PureQPBase.factorize!, types[1:3])]; guarantees) isa Vector
        end
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
    preconditioners = [PureQPBase.IdentityPreconditioner(), PureQPBase.JacobiPreconditioner(zeros(n))]
    defined = filter(T -> parentmodule(T) === PureQPBase, subtypes(PureQPBase.Preconditioner))
    @test Set(map(Base.typename, defined)) == Set(map(M -> Base.typename(typeof(M)), preconditioners))
    for M in preconditioners
        @test TypeContracts.check_contract(typeof(M), PureQPBase.Preconditioner).passed
        PureQPBase.update_preconditioner!(M, prob, wt, 0)
        ldiv!(y, M, x)
        @test test_signatures(
            [
                (ldiv!, map(typeof, (y, M, x))),
                (PureQPBase.update_preconditioner!, map(typeof, (M, prob, wt, 0))),
            ];
            guarantees = (:typestable, :noalloc, :trim_compatible)
        ) isa Vector
    end
end

@testitem "the allocation-free dense factorizations match the library ones exactly" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    Random.seed!(3)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    b = A * randn(n)

    # `FullKKT` factors through `sytrf` into arrays it holds; `bunchkaufman!` on the same
    # matrix is the reference, bit for bit.
    prob, wt, ls = backend_for(P, randn(n), A, b .- rand(m), b .+ rand(m); linsys = :kkt)
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
        bx, bz = randn(n), randn(m)
        x, z = zeros(n), zeros(m)
        PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
        r = ldiv!(ref, [bx; bz])
        @test x == r[1:n]
    end

    # `DiagonalLowRank` factors its capacitance through `potrf`; `cholesky!` is the reference.
    nd = 50
    Al = PureQPBase.RowCoupled(randn(3, nd) ./ 4, ones(nd - 3), collect(1:(nd - 3)))
    prob, wt, ls = backend_for(Diagonal(rand(nd) .+ 0.5), randn(nd), Al, -rand(nd), rand(nd); linsys = :lowrank)
    @test PureQPBase.refactor_weights!(ls, prob, wt)
    cap = ls.Y * ls.V'
    for i in axes(cap, 1)
        cap[i, i] += inv(wt.w[i])
    end
    @test triu(ls.cap) == triu(cholesky!(Symmetric(cap)).factors)
end

@testitem "the proofs fail on code that allocates or cannot be trimmed" begin
    using StrictMode, StrictModeTest

    # A gate that passes everything proves nothing; each of these must be refused.
    StrictMode.assert_enabled()
    grow(n) = zeros(n)
    @test_throws StrictMode.StrictViolation test_signatures([(grow, (Int,))]; guarantees = (:noalloc,))
    # A broadcast into a scratch vector keeps an aliasing check whose copy AllocCheck finds,
    # although it is never taken: the pattern the dense factorization no longer uses.
    scaled!(dst, a, b) = (dst .= sqrt.(a) .* b; nothing)
    V = Vector{Float64}
    @test_throws StrictMode.StrictViolation test_signatures([(scaled!, (V, V, V))]; guarantees = (:noalloc,))
    dynamic(r) = r[] + 1
    @test_throws StrictMode.StrictViolation test_signatures(
        [(dynamic, (Base.RefValue{Any},))]; guarantees = (:trim_compatible,)
    )
end
