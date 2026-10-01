@testitem "every benchmark suite class reaches its recorded backend" begin
    using PureQPBase, LinearAlgebra, SparseArrays
    using LDLFactorizations, BandedMatrices, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    # The benchmark problem classes are the reference selection is asserted against: they
    # carry the block and band structure real problems have, where the other generators
    # here are uniformly random and land every sparse pattern on the same rung.
    include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

    expected = Dict(
        "Random QP" => :sparse_formed,
        "Eq QP" => :sparse_formed,
        "Portfolio" => :ldl_kkt,
        "Lasso" => :ldlfactorizations,
        "SVM" => :ldlfactorizations,
        "Huber" => :ldlfactorizations,
        "Control" => :sparse_formed,
    )
    @test sort(first.(CASES)) == sort(collect(keys(expected)))
    for (name, make) in CASES
        P, q, A, l, u = make()
        @test PureQPBase.backend_name(last(backend_for(P, q, A, l, u))) === expected[name]
    end
end

@testitem "the sparse rule reads the pattern and nothing else" begin
    using PureQPBase
    using LinearAlgebra, SparseArrays, Random
    using LDLFactorizations
    Ext = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
    Random.seed!(74)

    # A pattern's answer must not move when the stored values do: that is what "decides from
    # the pattern" means, and it is what lets a solver factor once.
    n, m = 400, 200
    A = sprandn(m, n, 0.01)
    P = sparse(1.0I, n, n)
    for sel in (PureQPBase.ADMMSelection(), PureQPBase.IPMSelection())
        want = Ext.sparse_form(P, A, n, m, sel)
        scaled = SparseMatrixCSC(m, n, copy(A.colptr), copy(A.rowval), nonzeros(A) .* 1.0e6)
        @test Ext.sparse_form(P, scaled, n, m, sel) === want
    end

    # One row spanning the variables fills the reduced matrix by itself, so both algorithms
    # take the KKT form. It is the only route to that form under ADMM, so dropping the row
    # leaves the reduced one.
    budget = sparse([fill(1.0, 1, n); Matrix(sprandn(m - 1, n, 0.005))])
    for sel in (PureQPBase.ADMMSelection(), PureQPBase.IPMSelection())
        @test Ext.sparse_form(P, budget, n, m, sel) === :kkt
    end
    @test Ext.sparse_form(P, budget[2:end, :], n, m - 1, PureQPBase.ADMMSelection()) === :reduced

    # A pattern with no sparsity left to exploit is served by neither sparse form under the
    # interior-point method, whose terminal is the dense KKT factorization.
    full = sparse(randn(m, n))
    @test Ext.sparse_form(sparse(randn(n, n)), full, n, m, PureQPBase.IPMSelection()) === :none
    @test Ext.sparse_form(P, full, n, m, PureQPBase.ADMMSelection()) === :none

    # `row_pattern` is the one pass over `A` the rule needs.
    densest, sumsq = Ext.row_pattern(A)
    counts = [count(==(i), rowvals(A)) for i in 1:m]
    @test densest == maximum(counts)
    @test sumsq == sum(abs2, counts)
end

@testitem "every structured family reaches its recorded backend" begin
    using PureQPBase, LinearAlgebra, BandedMatrices, Random
    using LDLFactorizations, SparseArrays, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(71)

    "An `n×n` band of half-width `b`, diagonally dominant so the reduced matrix is definite."
    function wide_band(n, b)
        A = BandedMatrix{Float64}(undef, (n, n), (b, b))
        fill!(A.data, 0.0)
        for j in 1:n, i in max(1, j - b):min(n, j + b)
            A[i, j] = i == j ? 1.0 : 0.01
        end
        return A
    end

    # Each pair is named with the backend it selects. Handing the same numbers over as dense
    # `Matrix`es always reaches the terminal rung, which is what the second assertion checks —
    # for the pairs whose structured form selects something else, that is the structure being
    # worth something; for the one whose structured form is already `:cholesky`, it is not a
    # change.
    families(n) = [
        (
            :diagonal,
            Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5),
        ),
        (
            :tridiagonal,
            SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5),
        ),
        (
            :banded,
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4),
        ),
        (
            :tridiagonal,
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L),
        ),
        # `Symmetric` over a dense parent is not one of the types the banded rung accepts, so
        # this never reaches it and the terminal takes it however narrow the band is.
        (
            :cholesky,
            Symmetric(Matrix(SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8))),
            BandedMatrix(0 => rand(n) .+ 1, 1 => rand(n - 1) ./ 4, -1 => rand(n - 1) ./ 4),
        ),
        # Either side of the rung's limit, which accepts while `4b <= n`. A `BandedMatrix` `A`
        # of half-width `bA` gives a reduced bandwidth of `2bA`, so `bA = 12` is inside at
        # `n = 100` and `bA = 13` is outside.
        (:banded, SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), wide_band(n, 12)),
        (:cholesky, SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), wide_band(n, 13)),
        # Three dense rows above bounds on the remaining variables: a diagonal core plus a
        # rank-3 correction, which is not banded at any width.
        (
            :lowrank,
            Diagonal(rand(n) .+ 0.5),
            PureQPBase.RowCoupled(randn(3, n) ./ 4, ones(n - 3), collect(1:(n - 3))),
        ),
        # Either side of the rung's limit, which accepts while `10k <= n`.
        (
            :lowrank,
            Diagonal(rand(n) .+ 0.5),
            PureQPBase.RowCoupled(randn(10, n) ./ 4, ones(n - 10), collect(1:(n - 10))),
        ),
        # Past the limit the rung declines, and `RowCoupled` answers `holds_structure` true, so
        # the rungs that form the reduced matrix decline too: `k` dense rows and a diagonal
        # stand for `n²` entries, and the matrix-free backend works from them as they are.
        (
            :indirect,
            Diagonal(rand(n) .+ 0.5),
            PureQPBase.RowCoupled(randn(11, n) ./ 4, ones(n - 11), collect(1:(n - 11))),
        ),
    ]

    n = 100
    for (backend, P, A) in families(n)
        q, l, u = randn(n), -rand(n), rand(n)
        # `factorize = false`: the selection is what is under test, and the matrix-free backend
        # takes its conjugate-gradient settings from an algorithm, which this helper has none of.
        @test PureQPBase.backend_name(last(backend_for(P, q, A, l, u; factorize = false))) ===
            backend
        @test PureQPBase.backend_name(
            last(backend_for(Matrix(P), q, Matrix(A), l, u))
        ) === :cholesky
    end
end

@testitem "the ladder's terminal and indirect rungs" begin
    using PureQPBase, LinearAlgebra, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    n, m = 5, 4
    P, A = Matrix(1.0I, n, n), randn(m, n)
    prob = raw_problem(P, A, n, m)
    wt = raw_weights(ones(m), 1.0e-6)
    sel = PureQPBase.ADMMSelection()

    # A materializable pair stops at the dense terminal, and the rungs above it decline.
    @test isnothing(PureQPBase.formed_rung(P, A, prob, sel))
    ls, factored = PureQPBase.dense_rung(P, A, prob, sel)
    @test ls isa PureQPBase.ReducedCholesky
    @test !factored

    # Below the terminal: an operator that supplies only products reaches the matrix-free
    # rung instead of falling out of the ladder. `is_materializable` is what declines the
    # terminal, so the decline is reachable from a type a solver accepts.
    struct Opaque <: AbstractMatrix{Float64} end
    PureQPBase.is_materializable(::Opaque) = false
    opaque_prob = raw_problem(Opaque(), Opaque(), n, m)
    @test isnothing(PureQPBase.dense_rung(Opaque(), Opaque(), opaque_prob, sel))
    ls, factored = PureQPBase.indirect_rung(Opaque(), Opaque(), opaque_prob, sel)
    @test PureQPBase.backend_name(ls) === :indirect
    @test !factored

    # The descent itself, not just its rungs: a pair no rung above the terminal serves stops
    # at the terminal, and one no rung serves at all reaches the bottom.
    ls, factored = PureQPBase.select_backend(P, A, prob, wt, sel)
    @test ls isa PureQPBase.ReducedCholesky
    @test !factored
    ls, factored = PureQPBase.select_backend(Opaque(), Opaque(), opaque_prob, wt, sel)
    @test PureQPBase.backend_name(ls) === :indirect
    @test !factored
end

@testitem "backend_info describes each backend it is asked about" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    using LDLFactorizations, BandedMatrices, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(72)

    # `factor_nnz` counts one triangle of whatever the backend stores, in that
    # factorization's own convention.
    function check(ls, name, direct, system, dim)
        info = PureQPBase.backend_info(ls)
        @test info isa PureQPBase.BackendInfo
        @test info.name === name === PureQPBase.backend_name(ls)
        @test info.direct == direct
        @test info.system === system
        @test info.dim == dim
        @test 0 <= info.factor_nnz <= dim * (dim + 1) ÷ 2
        return info
    end

    n, m = 30, 20
    q, l, u = randn(n), -rand(m), rand(m)
    P, A = Matrix(1.0I, n, n), randn(m, n)
    pick(; kwargs...) = last(backend_for(P, q, A, l, u; kwargs...))
    red, kkt = pick(), pick(linsys = :kkt)
    # The matrix-free backend describes itself before it is configured; what it cannot do
    # without an algorithm is run, which is why it is asked for unfactorized.
    cg = pick(linsys = :indirect, factorize = false)
    check(red, :cholesky, true, :reduced, n)
    check(kkt, :bunchkaufman, true, :kkt, n + m)
    check(cg, :indirect, false, :reduced, n)
    @test iszero(PureQPBase.backend_info(cg).factor_nnz)

    ld = Diagonal(rand(n) .+ 0.5)
    dq, dl, du = randn(n), -rand(n), rand(n)
    square(P, A) = last(backend_for(P, dq, A, dl, du))
    check(square(ld, Diagonal(rand(n) .+ 0.5)), :diagonal, true, :reduced, n)
    tri = SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8)
    check(square(tri, ld), :tridiagonal, true, :reduced, n)
    band = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    check(square(tri, band), :banded, true, :reduced, n)

    # A banded factor of bandwidth `b` holds `b + 1` entries per column, less the corner.
    @test PureQPBase.backend_info(square(tri, band)).factor_nnz == n * 3 - 3

    # The sparse rungs: an identity `A` keeps the KKT factor sparse.
    info = PureQPBase.backend_info(square(sparse(1.0I, n, n), sparse(1.0I, n, n)))
    @test info.factor_nnz < info.dim^2
    @test info.system in (:reduced, :kkt)
    @test info.direct
end

@testitem "two spellings of one matrix select the same backend" begin
    using PureQPBase, LinearAlgebra, SparseArrays, BandedMatrices, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(73)

    # `Tridiagonal` and `SymTridiagonal` name the same band, so a problem written either way
    # is one problem and has to reach one backend. Calling `choose_backend` directly beside
    # the built problem is what makes a method ambiguity between `src` and the banded
    # extension fail here rather than at some caller's first use.
    n = 60
    dv, ev = rand(n) .+ 3, rand(n - 1) ./ 8
    sym = SymTridiagonal(copy(dv), copy(ev))
    tri = Tridiagonal(copy(ev), copy(dv), copy(ev))
    diag_A = Diagonal(rand(n) .+ 0.5)
    bidi_A = Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L)
    tri_A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)

    q, l, u = randn(n), -rand(n), rand(n)
    named(P, A) = PureQPBase.backend_name(last(backend_for(P, q, A, l, u)))

    @test named(tri, diag_A) === named(sym, diag_A) === :tridiagonal
    @test named(tri, bidi_A) === named(sym, bidi_A) === :tridiagonal
    @test named(sym, tri_A) === :banded

    wt = raw_weights(ones(n), 1.0e-6)
    picked(P, A) = PureQPBase.backend_name(
        first(PureQPBase.choose_backend(P, A, raw_problem(P, A, n, n), wt, PureQPBase.ADMMSelection()))
    )
    @test picked(tri, diag_A) === picked(sym, diag_A) === :tridiagonal
    @test picked(tri, bidi_A) === picked(sym, bidi_A) === :tridiagonal
    @test picked(sym, tri_A) === :banded
end

@testitem "no ladder forms a matrix for a pair that holds structure" begin
    using PureQPBase, LinearAlgebra, Krylov, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(14)

    # The three representations that answer `holds_structure` true, each against an `A` that
    # sends the pair past the rung built for it: equilibration defeats the Kronecker rung, and
    # the low-rank rung declines above `10k > n`.
    k1, k2 = 6, 5
    n = k1 * k2
    f1 = Matrix(Symmetric(rand(k1, k1) + k1 * I))
    f2 = Matrix(Symmetric(rand(k2, k2) + k2 * I))
    kron_P = PureQPBase.KroneckerOperator(f1, f2)
    kron_A = PureQPBase.KroneckerOperator(Matrix(1.0I, k1, k1), Matrix(1.0I, k2, k2))
    block_P = PureQPBase.BlockDiagonal([f1, Matrix(Symmetric(rand(k2, k2) + k2 * I))])
    nb = size(block_P, 1)
    coupled = PureQPBase.RowCoupled(randn(11, n) ./ 4, ones(n - 11), collect(1:(n - 11)))

    @test PureQPBase.holds_structure(kron_P)
    @test PureQPBase.holds_structure(block_P)
    @test PureQPBase.holds_structure(coupled)
    # A representation that stands for its own entries is formed as before.
    @test !PureQPBase.holds_structure(randn(4, 4))
    @test !PureQPBase.holds_structure(Diagonal(rand(4)))
    @test !PureQPBase.holds_structure(sparse(1.0I, 4, 4))
    @test !PureQPBase.holds_structure(Symmetric(randn(4, 4)))

    pairs = (
        (kron_P, kron_A, n),
        (kron_P, Matrix(1.0I, n, n), n),
        (Diagonal(rand(n) .+ 0.5), coupled, n),
        (block_P, Matrix(1.0I, nb, nb), nb),
    )
    for (P, A, dim) in pairs
        q, l, u = randn(dim), -rand(dim), rand(dim)
        # `scaling = 10`, the default, is what puts the Kronecker pair past its own rung.
        name = PureQPBase.backend_name(last(backend_for(P, q, A, l, u; factorize = false)))
        @test name === :indirect
        # The two rungs that form the reduced matrix decline by name, not by falling off the
        # end of a method table.
        prob = PureQPBase.validated_problem(Float64, dim, dim, P, q, A, l, u, 10)
        wt = raw_weights(fill(0.1, dim), 1.0e-6)
        @test isnothing(PureQPBase.dense_rung(P, A, prob, PureQPBase.ADMMSelection()))
        @test isnothing(PureQPBase.formed_rung(P, A, prob, PureQPBase.ADMMSelection()))
    end
end
