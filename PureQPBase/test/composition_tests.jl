@testitem "a composition reduces through its parts and agrees with the matrix it stands for" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(21)

    # A Kronecker outer part, so the reduction reaches that part's own contraction.
    Bk = PureQPBase.KroneckerOperator(randn(4, 3), randn(5, 2))     # 20×6
    inner = randn(6, 5)
    composed = PureQPBase.ComposedOperator(Bk, inner)
    # A Kronecker term plus a dense one.
    sum_op = PureQPBase.SumOperator(
        PureQPBase.KroneckerOperator(randn(4, 3), randn(5, 2)), randn(20, 6)
    )

    for (A, dense) in (
            (composed, Matrix(Bk) * inner),
            (sum_op, Matrix(sum_op.terms[1]) + sum_op.terms[2]),
        )
        m, n = size(dense)
        @test size(A) == (m, n)
        @test PureQPBase.holds_structure(A)
        @test [A[i, j] for i in 1:m, j in 1:n] ≈ dense

        x, yy = randn(n), randn(m)
        @test mul!(zeros(m), A, x) ≈ dense * x
        @test mul!(zeros(n), adjoint(A), yy) ≈ dense'yy
        # `transpose` must reach the same product as `adjoint` for a real operator rather than
        # falling back to reading entries.
        @test mul!(zeros(n), transpose(A), yy) ≈ dense'yy

        row = zeros(n)
        @test all(i -> (PureQPBase.dense_row!(row, A, i); row ≈ dense[i, :]), 1:m)

        w, D = rand(m) .+ 0.5, rand(n) .+ 0.5
        R = zeros(n, n)
        scratch = PureQPBase.reduced_term_scratch(Float64, A)
        PureQPBase.add_reduced_term!(
            R, Float64, A, w, D, n, m, scratch, zeros(n), zeros(m), zeros(n)
        )
        @test R ≈ Diagonal(D) * dense' * Diagonal(w) * dense * Diagonal(D)
    end
end

@testitem "a composition is built from its parts and checks that they fit" begin
    using PureQPBase, LinearAlgebra
    @test_throws "parts to meet" PureQPBase.ComposedOperator(randn(5, 4), randn(3, 2))
    @test_throws "the same size" PureQPBase.SumOperator(randn(5, 4), randn(5, 3))
    @test_throws "at least one term" PureQPBase.SumOperator{Float64, Tuple{}}(())

    composed = PureQPBase.ComposedOperator(randn(5, 4), randn(4, 3))
    @test PureQPBase.parts(composed) === (composed.outer, composed.inner)
    @test PureQPBase.nterms(PureQPBase.SumOperator(randn(5, 4), randn(5, 4), randn(5, 4))) == 3

    # A part with no entries leaves the whole composition unreadable, and the diagonal with it.
    opaque = PureQPBase.ProductOperator{Float64}(randn(4, 3))
    mixed = PureQPBase.ComposedOperator(randn(5, 4), opaque)
    @test !PureQPBase.is_materializable(mixed)
    dest = zeros(3)
    PureQPBase.reduced_diagonal!(
        dest, Float64, Diagonal(ones(3)), mixed, ones(5), ones(5), ones(3), 1.0e-6, 1.0
    )
    @test all(isone, dest)
end

@testitem "a composition's products and reduction are proved" begin
    using PureQPBase, StrictMode, StrictModeTest, LinearAlgebra, Random
    StrictMode.assert_enabled()
    Random.seed!(22)

    composed = PureQPBase.ComposedOperator(
        PureQPBase.KroneckerOperator(randn(4, 3), randn(5, 2)), randn(6, 5)
    )
    sum_op = PureQPBase.SumOperator(
        PureQPBase.KroneckerOperator(randn(4, 3), randn(5, 2)), randn(20, 6)
    )

    for A in (composed, sum_op)
        m, n = size(A)
        x, yy = randn(n), randn(m)
        y, z, row = zeros(m), zeros(n), zeros(n)
        w, D = rand(m) .+ 0.5, rand(n) .+ 0.5
        R = zeros(n, n)
        scratch = PureQPBase.reduced_term_scratch(Float64, A)
        ej, av, col = zeros(n), zeros(m), zeros(n)
        # Warmed on the real data before the proof reads the compiled code.
        mul!(y, A, x)
        mul!(z, adjoint(A), yy)
        PureQPBase.dense_row!(row, A, 1)
        PureQPBase.add_reduced_term!(R, Float64, A, w, D, n, m, scratch, ej, av, col)
        @test test_signatures(
            [
                (mul!, (typeof(y), typeof(A), typeof(x))),
                (mul!, (typeof(z), typeof(adjoint(A)), typeof(yy))),
                (PureQPBase.dense_row!, (typeof(row), typeof(A), Int)),
                (
                    PureQPBase.add_reduced_term!,
                    (
                        typeof(R), Type{Float64}, typeof(A), typeof(w), typeof(D), Int, Int,
                        typeof(scratch), typeof(ej), typeof(av), typeof(col),
                    ),
                ),
            ];
            guarantees = (:typestable, :noalloc, :trim_compatible)
        ) isa Vector
    end
end
