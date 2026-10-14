@testitem "a JoinedOperator answers as the hcat of its blocks" begin
    using PureQPBase, LinearAlgebra, FillArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(41)

    m = 6
    B = randn(m, 4)
    K = PureQPBase.KroneckerOperator(randn(3, 2), randn(2, 3))
    F = Fill(0.5, m, 5)
    S = PureQPBase.StackedOperator(randn(2, 3), Fill(-1.0, 4, 3))
    J = PureQPBase.JoinedOperator(B, K, F, S)
    ref = hcat(B, Matrix(K), Matrix(F), Matrix(S))
    n = size(ref, 2)

    @test size(J) == (m, n)
    @test PureQPBase.nblocks(J) == 4
    @test PureQPBase.colrange(J, 2) == 5:10
    @test !PureQPBase.holds_structure(J)
    @test PureQPBase.is_materializable(J)
    @test Matrix(J) == ref
    @test all(J[i, j] == ref[i, j] for i in 1:m, j in 1:n)

    # An entry has the eltype the operator promises, whatever its block holds. A `Fill` of an
    # integer is the natural way to write a constant block, and a caller given an
    # `AbstractMatrix{Float64}` must not read an `Int` out of it.
    mixed = PureQPBase.JoinedOperator(randn(3, 2), Fill(1, 3, 2))
    stacked_mixed = PureQPBase.StackedOperator(randn(2, 3), Fill(1, 2, 3))
    @test eltype(mixed) === Float64
    @test all(mixed[i, j] isa Float64 for i in 1:3, j in 1:4)
    @test all(stacked_mixed[i, j] isa Float64 for i in 1:4, j in 1:3)
    @test Base.infer_return_type(getindex, (typeof(mixed), Int, Int)) === Float64
    @test Base.infer_return_type(getindex, (typeof(stacked_mixed), Int, Int)) === Float64

    x, y = randn(n), randn(m)
    @test mul!(zeros(m), J, x) ≈ ref * x
    @test mul!(zeros(n), adjoint(J), y) ≈ ref' * y
    @test mul!(zeros(n), transpose(J), y) ≈ ref' * y
    # A view operand and a view destination, which a join inside a stack sees.
    @test mul!(view(zeros(m + 2), 2:(m + 1)), J, view([0.0; x; 0.0], 2:(n + 1))) ≈ ref * x

    for i in 1:m
        @test PureQPBase.dense_row!(zeros(n), J, i) ≈ ref[i, :]
    end
    @test_throws DimensionMismatch PureQPBase.dense_row!(zeros(n - 1), J, 1)

    # A column's rows are its block's: the stack's blocks answer with every row, as the
    # Kronecker and constant blocks do here; a diagonal block answers with one.
    @test PureQPBase.structural_rows(J, 3) == axes(B, 1)
    @test PureQPBase.structural_rows(J, 12) == axes(F, 1)
    D = Diagonal(rand(m))
    JD = PureQPBase.JoinedOperator(D, F)
    @test PureQPBase.structural_rows(JD, 4) == 4:4
    @test PureQPBase.structural_rows(JD, m + 2) == axes(F, 1)

    # The reduced term through the generic reduction: `D Aᵀ diag(w) A D` from products.
    w = rand(m) .+ 0.5
    Dv = rand(n) .+ 0.5
    R = zeros(n, n)
    scratch = PureQPBase.reduced_term_scratch(Float64, J)
    PureQPBase.add_reduced_term!(R, Float64, J, w, Dv, n, m, scratch, zeros(n), zeros(m), zeros(n))
    @test R ≈ Diagonal(Dv) * ref' * Diagonal(w) * ref * Diagonal(Dv)

    # The reduced diagonal is the diagonal of that matrix with `P` and `σ` added, inverted.
    P = Diagonal(fill(2.0, n))
    rho, E, sigma, c = rand(m) .+ 0.1, rand(m) .+ 0.5, 1.0e-6, 1.0
    dest = zeros(n)
    PureQPBase.reduced_diagonal!(dest, Float64, P, J, rho, E, Dv, sigma, c)
    full = c * Diagonal(Dv) * P * Diagonal(Dv) + sigma * I +
        Diagonal(Dv) * ref' * Diagonal(E .^ 2 .* rho) * ref * Diagonal(Dv)
    @test dest ≈ inv.(diag(full))

    # A block with no entries leaves the join unreadable, and the diagonal unavailable.
    Jp = PureQPBase.JoinedOperator(PureQPBase.ProductOperator{Float64}(B), F)
    @test !PureQPBase.is_materializable(Jp)
    @test mul!(zeros(m), Jp, x[1:9]) ≈ [B Matrix(F)] * x[1:9]
    @test_throws "no entries to read" Jp[1, 1]
    @test PureQPBase.reduced_diagonal!(zeros(9), Float64, Diagonal(ones(9)), Jp, rho, E, ones(9), sigma, c) == ones(9)

    # Every block spans the same rows, and the element type is real.
    @test_throws DimensionMismatch PureQPBase.JoinedOperator(B, randn(m + 1, 2))
    @test_throws ArgumentError PureQPBase.JoinedOperator(complex.(B))

    # A non-finite entry is reported by the block holding it.
    Bn = copy(B)
    Bn[2, 3] = NaN
    err = try
        PureQPBase.check_finite(PureQPBase.JoinedOperator(F, Bn), m, n, "A")
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("A's block 2 is not finite at entry (2, 3)", err.msg)
end

@testitem "a join serves the dense terminals, a stack of joins the rungs below them" begin
    using PureQPBase, LinearAlgebra, FillArrays, Krylov, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(42)

    n, m = 15, 20
    K = PureQPBase.KroneckerOperator(randn(4, 3), randn(5, 3))   # 20×9
    J = PureQPBase.JoinedOperator(K, Fill(0.25, m, n - 9))
    P = Diagonal(rand(n) .+ 1)
    q, l, u = randn(n), -rand(m), rand(m)

    # `holds_structure` is false, so the dense terminals take the join and form the reduced
    # matrix from its entries; here the ADMM one, whose ladder this package holds.
    prob, wt, ls = backend_for(P, q, J, l, u)
    @test ls isa PureQPBase.ReducedCholesky
    # Equilibration ran on the join through its blocks, and the scaled problem matches the
    # dense one scaled the same way.
    @test prob.D != ones(n)
    probd, = backend_for(P, q, Matrix(J), l, u)
    @test prob.D ≈ probd.D
    @test prob.E ≈ probd.E

    # A stack of joins holds structure, as any stack does, and reaches the rung below the ADMM
    # terminal: conjugate gradients.
    S = PureQPBase.StackedOperator(J, PureQPBase.JoinedOperator(Fill(1.0, 5, 9), randn(5, 6)))
    @test PureQPBase.holds_structure(S)
    @test Matrix(S) == [Matrix(J); ones(5, 9) S.blocks[2].blocks[2]]
    ls25, us25 = -rand(m + 5), rand(m + 5)
    _, _, lss = backend_for(P, q, S, ls25, us25; factorize = false)
    @test backend_name(lss) === :indirect
end

@testitem "a stack checks its blocks' entries, never the rows they stand for" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(11)

    # Two Kronecker blocks, each standing for 1.6e9 entries from four 200x200 factors. Walking
    # the stack's own entries through `getindex` would read every one of them, so a stack that
    # does not delegate to its blocks cannot finish this at all.
    k = 200
    factors = [Matrix(I * 1.0, k, k) + randn(k, k) / k for _ in 1:4]
    b1 = PureQPBase.KroneckerOperator(factors[1], factors[2])
    b2 = PureQPBase.KroneckerOperator(factors[3], factors[4])
    S = PureQPBase.StackedOperator(b1, b2)
    rows, cols = size(S)
    @test rows == 2k^2
    @test cols == k^2
    @test (@elapsed PureQPBase.check_finite(S, rows, cols, "A")) < 1.0

    # A non-finite entry is reported by the block holding it, and by that block's own factor.
    factors[4][3, 5] = Inf
    bad = PureQPBase.StackedOperator(b1, PureQPBase.KroneckerOperator(factors[3], factors[4]))
    err = try
        PureQPBase.check_finite(bad, rows, cols, "A")
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("A's block 2's second Kronecker factor is not finite at entry (3, 5)", err.msg)
end

@testitem "a join's reduced term is assembled from its blocks" begin
    using PureQPBase, LinearAlgebra, FillArrays, Random
    Random.seed!(29)

    # Two Kronecker blocks over the same rows plus a constant block, which is the shape an
    # interior-point reduced matrix meets when the constraint operator is a stack of joins.
    # Without the join's own method this falls to the generic product form and never reaches
    # either block's contraction, so the check is that the assembled matrix is the dense one.
    rows = 12
    b1 = PureQPBase.KroneckerOperator(randn(2, 1), randn(6, 5))      # 12 x 5
    b2 = PureQPBase.KroneckerOperator(randn(2, 1), randn(6, 3))      # 12 x 3
    b3 = Fill(0.25, rows, 4)
    J = PureQPBase.JoinedOperator(b1, b2, b3)
    n = size(J, 2)
    @test n == 12

    w = rand(rows) .+ 0.1
    D = rand(n) .+ 0.5
    R = zeros(n, n)
    PureQPBase.add_reduced_term!(
        R, Float64, J, w, D, n, rows, PureQPBase.reduced_term_scratch(Float64, J),
        zeros(n), zeros(rows), zeros(n)
    )
    Jm = Matrix(J)
    @test R ≈ Diagonal(D) * Jm' * Diagonal(w) * Jm * Diagonal(D) rtol = 1.0e-12
    # The cross blocks are written on both sides of the diagonal, not just one.
    @test R ≈ R' rtol = 1.0e-12

    # A zero constant block contributes nothing, diagonally or in its cross terms.
    Jz = PureQPBase.JoinedOperator(b1, b2, Fill(0.0, rows, 4))
    Rz = zeros(n, n)
    PureQPBase.add_reduced_term!(
        Rz, Float64, Jz, w, D, n, rows, PureQPBase.reduced_term_scratch(Float64, Jz),
        zeros(n), zeros(rows), zeros(n)
    )
    @test all(iszero, Rz[9:12, :])
    @test all(iszero, Rz[:, 9:12])
end

@testitem "a join contracts a Kronecker pair's cross term to matrix products" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(24)

    # Two Kronecker blocks over the same row structure take the contracted form; the shapes
    # below cover one column pair and several, and a first factor whose signs disagree with its
    # partner's, which is what stops the weight folding into a row scaling.
    function reduced(Ai, Aj)
        J = PureQPBase.JoinedOperator(Ai, Aj)
        n, m = size(J, 2), size(J, 1)
        w = rand(m) .+ 0.1
        D = rand(n) .+ 0.5
        R = zeros(n, n)
        PureQPBase.add_reduced_term!(
            R, Float64, J, w, D, n, m, PureQPBase.reduced_term_scratch(Float64, J),
            zeros(n), zeros(m), zeros(n)
        )
        Jm = Matrix(J)
        return R, Diagonal(D) * Jm' * Diagonal(w) * Jm * Diagonal(D)
    end
    kron_of(m1, p, m2, q) = PureQPBase.KroneckerOperator(randn(m1, p), randn(m2, q))

    for (a, b) in (
            (kron_of(2, 1, 6, 5), kron_of(2, 1, 6, 3)),
            (kron_of(3, 2, 5, 4), kron_of(3, 2, 5, 3)),
            (kron_of(4, 3, 7, 2), kron_of(4, 1, 7, 6)),
            (
                PureQPBase.KroneckerOperator(reshape([1.0, -1.0], 2, 1), randn(6, 5)),
                PureQPBase.KroneckerOperator(reshape([1.0, 1.0], 2, 1), randn(6, 4)),
            ),
        )
        @test PureQPBase.kron_cross_pair(a, b)
        R, Rtrue = reduced(a, b)
        @test R ≈ Rtrue rtol = 1.0e-12
        @test R ≈ R' rtol = 1.0e-12
    end

    # Rows that do not line up leave the pair on the column form, which must still be right.
    mismatched = (kron_of(2, 1, 6, 5), kron_of(3, 1, 4, 3))
    @test !PureQPBase.kron_cross_pair(mismatched...)
    @test isempty(first(PureQPBase.kron_cross_scratch(Float64, mismatched)))
    J = PureQPBase.JoinedOperator(kron_of(2, 1, 6, 5), PureQPBase.KroneckerOperator(randn(2, 1), randn(6, 3)))
    # A qualifying pair sizes the buffers from the wider of the two second factors.
    scratch = PureQPBase.kron_cross_scratch(Float64, J.blocks)
    @test size(first(scratch)) == (6, 3)
    @test size(last(scratch)) == (5, 3)
end
