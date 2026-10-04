@testitem "dense_row! equals the row of the matrix, for every representation and element type" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(71)

    # No `getindex` on this type or its adjoint: a row can only come from an adjoint product.
    struct RowsProductsOnly{T}
        M::Matrix{T}
    end
    struct RowsProductsOnlyAdjoint{T}
        parent::RowsProductsOnly{T}
    end
    Base.size(o::RowsProductsOnly) = size(o.M)
    Base.size(o::RowsProductsOnlyAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::RowsProductsOnly) = RowsProductsOnlyAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::RowsProductsOnly, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::RowsProductsOnlyAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)

    for T in (Float64, Float32)
        big = randn(T, 9, 11)
        dense = randn(T, 7, 5)
        symm = Symmetric(randn(T, 6, 6))
        cases = Any[
            dense,
            view(big, 2:8, 3:7),                      # strided, not contiguous
            symm,                                     # the entrywise fallback
            Diagonal(randn(T, 6)),
            PureQPBase.KroneckerOperator(randn(T, 4, 3), randn(T, 2, 5)),
            PureQPBase.KroneckerOperator(randn(T, 1, 3), randn(T, 4, 1)),
            PureQPBase.BlockDiagonal([randn(T, 3, 2), randn(T, 1, 4), randn(T, 2, 3)]),
            PureQPBase.ProductOperator{T}(RowsProductsOnly(dense)),
        ]
        for A in cases
            ref = A isa PureQPBase.ProductOperator ? A.op.M : Matrix(A)
            dest = zeros(T, size(A, 2))
            for i in axes(A, 1)
                fill!(dest, T(NaN))    # every entry, structural zeros included, must be written
                @test PureQPBase.dense_row!(dest, A, i) === dest
                @test dest == ref[i, :]
            end
        end

        A = cases[5]
        @test_throws BoundsError PureQPBase.dense_row!(zeros(T, size(A, 2)), A, size(A, 1) + 1)
        @test_throws "cannot be written into a vector" PureQPBase.dense_row!(zeros(T, 3), A, 1)
    end
end

@testitem "a ProductOperator holds its row scratch whether or not it probes" begin
    using PureQPBase, LinearAlgebra, LinearMaps, SciMLOperators, Random
    Random.seed!(72)
    B = randn(6, 4)

    wrapped = PureQPBase.ProductOperator{Float64}(LinearMap(B))
    @test length(wrapped.column) == 6
    @test isempty(wrapped.basis)
    dest = zeros(4)
    @test PureQPBase.dense_row!(dest, wrapped, 3) ≈ B[3, :]

    scimlop = PureQPBase.ProductOperator{Float64}(MatrixOperator(B))
    @test length(scimlop.column) == 6
    @test PureQPBase.dense_row!(dest, scimlop, 5) ≈ B[5, :]

    # A Kronecker product carried by a LinearMap: the row comes from the map's adjoint.
    A1, A2 = randn(3, 2), randn(2, 3)
    kmap = PureQPBase.ProductOperator{Float64}(LinearMap(kron(A1, A2)))
    ref = kron(A1, A2)
    kdest = zeros(6)
    for i in 1:6
        @test PureQPBase.dense_row!(kdest, kmap, i) ≈ ref[i, :]
    end

    # Probing still gets its own length-`cols` basis.
    probing = PureQPBase.ProductOperator{Float64}(LinearMap(B); probe = true)
    @test length(probing.basis) == 4
    @test length(probing.column) == 6
    @test PureQPBase.dense_row!(dest, probing, 2) ≈ B[2, :]
    @test PureQPBase.probe_column!(probing, 1) ≈ B[:, 1]
end
