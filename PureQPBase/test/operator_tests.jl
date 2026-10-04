@testitem "an operator with no entries declines the paths that need them" begin
    using PureQPBase, LinearAlgebra, Krylov, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(32)

    # No `getindex` anywhere on this type or its adjoint, so anything that reaches for an
    # entry fails rather than quietly working through a fallback.
    struct ProductsOnly{T}
        M::Matrix{T}
    end
    struct ProductsOnlyAdjoint{T}
        parent::ProductsOnly{T}
    end
    Base.size(o::ProductsOnly) = size(o.M)
    Base.size(o::ProductsOnlyAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::ProductsOnly) = ProductsOnlyAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::ProductsOnly, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::ProductsOnlyAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)

    n, m = 40, 25
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    Pop = PureQPBase.ProductOperator{Float64}(ProductsOnly(Matrix(P)); symmetric = true, posdef = true)
    Aop = PureQPBase.ProductOperator{Float64}(ProductsOnly(A))

    @test size(Pop) == (n, n)
    @test size(Aop) == (m, n)
    @test !PureQPBase.is_materializable(Pop)
    @test PureQPBase.is_symmetric(Pop)
    @test PureQPBase.is_convex(Float64, Pop, 1.0e-6)

    # Equilibration reads columns, which products cannot answer. The message names the two
    # ways out rather than escaping as an index error from inside a column walk.
    @test_throws "supplies products only" backend_for(Pop, q, Aop, l, u)
    @test_throws "scaling = 0" backend_for(Pop, q, Aop, l, u)

    # Every rung that would form a matrix declines, so `:auto` reaches the matrix-free one
    # instead of failing inside a factorization.
    _, _, ls = backend_for(Pop, q, Aop, l, u; scaling = 0, factorize = false)
    @test PureQPBase.backend_name(ls) === :indirect
end

@testitem "probing equilibrates an operator with no entries" begin
    using PureQPBase, LinearAlgebra, Krylov, Random
    Random.seed!(34)

    n, m = 40, 25
    # Badly scaled on purpose: equilibration is what this test is about, so the factors must
    # be far from one.
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    A[1, :] .*= 1.0e4
    A[:, 1] .*= 1.0e3
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    built(P, A) = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, 10)

    probed = built(
        PureQPBase.ProductOperator{Float64}(Matrix(P); symmetric = true, posdef = true, probe = true),
        PureQPBase.ProductOperator{Float64}(A; probe = true),
    )
    walked = built(P, A)

    # `A * eⱼ` copies column `j` when the wrapped operator selects stored entries, so the
    # factors are the same numbers by the same arithmetic, not merely close.
    @test probed.D == walked.D
    @test probed.E == walked.E
    @test probed.c == walked.c

    # Without `probe`, the same operator refuses and names every way out.
    bare = PureQPBase.ProductOperator{Float64}(Matrix(P); symmetric = true, posdef = true)
    @test_throws "probe = true" built(bare, PureQPBase.ProductOperator{Float64}(A))
end

@testitem "a LinearMap is wrapped with its own traits" begin
    using PureQPBase, LinearAlgebra, LinearMaps, Krylov, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(33)

    n, m = 50, 30
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    # A map that holds no entries is what keeps its traits: a wrapped matrix is handed on as
    # the matrix instead.
    Pd = Matrix(P)
    Pmap = LinearMap{Float64}(
        (y, x) -> mul!(y, Pd, x), n, n; issymmetric = true, isposdef = true, ismutating = true
    )
    Amap = LinearMap{Float64}(
        (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x), m, n; ismutating = true
    )
    # `as_operator` is the door: the extension's `setup` method sends every argument through
    # it before the problem is built, so a test of the wrapping calls exactly that.
    Ext = Base.get_extension(PureQPBase, :PureQPBaseLinearMapsExt)
    wrap(M) = Ext.as_operator(Float64, M)
    free = (scaling = 0, linsys = :indirect, factorize = false)

    # `symmetric` and `posdef` come from the map's own traits rather than from the caller.
    prob, = backend_for(wrap(Pmap), q, wrap(Amap), l, u; free...)
    @test prob.P isa PureQPBase.ProductOperator
    @test PureQPBase.is_symmetric(prob.P)
    @test PureQPBase.is_convex(Float64, prob.P, 1.0e-6)

    # A map and a matrix mix: only the map is wrapped.
    mixed, = backend_for(wrap(Pmap), q, wrap(A), l, u; free...)
    @test mixed.P isa PureQPBase.ProductOperator
    @test mixed.A isa Matrix
end

@testitem "a wrapped matrix reaches the base as the matrix" begin
    using PureQPBase, LinearAlgebra, LinearMaps, Random
    Random.seed!(35)
    Ext = Base.get_extension(PureQPBase, :PureQPBaseLinearMapsExt)
    wrap(M) = Ext.as_operator(Float64, M)
    B = randn(6, 4)

    # The matrix itself, not a copy and not a wrapper.
    @test wrap(LinearMap(B)) === B
    D = Diagonal(rand(5))
    @test wrap(LinearMap(D)) === D
    # LinearMaps stores an adjoint as a wrapped adjoint matrix, which is handed on lazily.
    @test wrap(LinearMap(B)') == B'

    # A scalar is folded into a copy; the caller's matrix is left alone.
    B0 = copy(B)
    scaled = wrap(3 * LinearMap(B))
    @test scaled isa Matrix{Float64}
    @test scaled == 3 * B
    @test scaled !== B
    @test B == B0
    @test wrap(2 * (3 * LinearMap(B))) == 6 * B

    # The element type the solve runs in is the one the matrix arrives in.
    @test wrap(LinearMap(Float32.(B))) isa Matrix{Float64}
    @test wrap(LinearMap(Float32.(B))) == Float32.(B)
    @test wrap(LinearMap(rand(1:3, 3, 3))) isa Matrix{Float64}
end

@testitem "a kron of wrapped matrices reaches the base as a KroneckerOperator" begin
    using PureQPBase, LinearAlgebra, LinearMaps, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(36)
    Ext = Base.get_extension(PureQPBase, :PureQPBaseLinearMapsExt)
    wrap(M) = Ext.as_operator(Float64, M)
    n1, n2, m1, m2 = 7, 5, 6, 4
    B1, B2 = randn(m1, n1), randn(m2, n2)

    K = wrap(kron(LinearMap(B1), LinearMap(B2)))
    @test K isa PureQPBase.KroneckerOperator
    @test K.A1 === B1
    @test K.A2 === B2
    x = randn(n1 * n2)
    @test K * x ≈ kron(B1, B2) * x

    # The scalar `kron` pulls out of its factors lands in the first factor, as a copy.
    Ks = wrap(kron(2 * LinearMap(B1), LinearMap(B2)))
    @test Ks isa PureQPBase.KroneckerOperator
    @test Ks.A1 == 2 * B1
    @test Ks.A2 === B2
    @test Ks * x ≈ 2 * kron(B1, B2) * x
    @test wrap(3 * kron(LinearMap(B1), LinearMap(B2))).A1 == 3 * B1

    # Factors of different types are brought to one, which `KroneckerOperator` requires.
    mixed = wrap(kron(LinearMap(Symmetric(B1[1:5, :]' * B1[1:5, :])), LinearMap(B2)))
    @test mixed isa PureQPBase.KroneckerOperator{Float64, Matrix{Float64}}

    # A problem's `A` built this way reaches the backend that needs the structure.
    n = n1 * n2
    Kp = wrap(kron(LinearMap(randn(n1, n1)), LinearMap(randn(n2, n2))))
    b = Kp * randn(n)
    prob, wt, ls = backend_for(
        Diagonal(fill(2.0, n)), randn(n), Kp, b .- rand(n), b .+ rand(n); scaling = 0
    )
    @test prob.A isa PureQPBase.KroneckerOperator
    @test PureQPBase.backend_name(ls) === :kronecker

    # Nothing forms the product: these factors stand for a 90000 × 90000 matrix.
    big = wrap(kron(LinearMap(randn(300, 300)), LinearMap(randn(300, 300))))
    @test big isa PureQPBase.KroneckerOperator
    @test size(big) == (90_000, 90_000)
end

@testitem "a blockdiag of wrapped matrices reaches the base as a BlockDiagonal" begin
    using PureQPBase, LinearAlgebra, LinearMaps, SparseArrays, Random
    Random.seed!(37)
    Ext = Base.get_extension(PureQPBase, :PureQPBaseLinearMapsExt)
    wrap(M) = Ext.as_operator(Float64, M)
    blocks = [randn(3, 3), randn(4, 2), randn(2, 5)]

    D = wrap(blockdiag(map(LinearMap, blocks)...))
    @test D isa PureQPBase.BlockDiagonal
    @test all(D.blocks .=== blocks)
    @test size(D) == (9, 10)
    x = randn(10)
    @test D * x ≈ Matrix(blockdiag(map(LinearMap, blocks)...)) * x

    # A scalar multiplies every block, as a copy.
    Ds = wrap(2 * blockdiag(map(LinearMap, blocks)...))
    @test Ds isa PureQPBase.BlockDiagonal
    @test Ds.blocks == [2 * B for B in blocks]
    @test blocks[1] !== Ds.blocks[1]

    # Blocks of different types are brought to one.
    S = Symmetric(blocks[1]' * blocks[1])
    mixed = wrap(blockdiag(LinearMap(S), LinearMap(blocks[2])))
    @test mixed isa PureQPBase.BlockDiagonal{Float64, Matrix{Float64}}

    # A vector of maps is unwrapped as a tuple of them is.
    vec_map = LinearMaps.BlockDiagonalMap{Float64}(LinearMap.(blocks))
    @test vec_map.maps isa Vector
    @test wrap(vec_map) isa PureQPBase.BlockDiagonal
end

@testitem "a map the base cannot hold stays a ProductOperator" begin
    using PureQPBase, LinearAlgebra, LinearMaps, SparseArrays, Random
    Random.seed!(38)
    Ext = Base.get_extension(PureQPBase, :PureQPBaseLinearMapsExt)
    wrap(M) = Ext.as_operator(Float64, M)
    n = 6
    B, C = randn(n, n), randn(n, n)
    fn = LinearMap{Float64}((y, x) -> mul!(y, B, x), (y, x) -> mul!(y, B', x), n, n; ismutating = true)
    opaque(M) = wrap(M) isa PureQPBase.ProductOperator

    @test opaque(fn)
    @test opaque(2 * fn)
    # A scalar multiple of a structured map is structured; of an opaque one, opaque.
    @test !opaque(2 * LinearMap(B))
    # The factors are what make a Kronecker product or a block-diagonal structured.
    @test opaque(kron(fn, LinearMap(C)))
    @test opaque(kron(LinearMap(B), fn))
    @test opaque(blockdiag(LinearMap(B), fn))
    # The base's Kronecker operator has two factors.
    @test opaque(kron(LinearMap(B), LinearMap(C), LinearMap(B)))
    # A sum and a two-map product each have a representation that keeps their parts.
    @test wrap(LinearMap(B) + LinearMap(C)) isa PureQPBase.SumOperator
    @test wrap(LinearMap(B)' * LinearMap(B)) isa PureQPBase.ComposedOperator
    # A chain of three has no single inner part to reduce against.
    @test opaque(LinearMap(B) * LinearMap(B)' * LinearMap(B))
    # A scalar reaches the parts: it multiplies every term of a sum and one part of a product.
    let x = randn(n)
        @test mul!(zeros(n), wrap(2 * (LinearMap(B) + LinearMap(C))), x) ≈ 2 * (B + C) * x
        @test mul!(zeros(n), wrap(3 * (LinearMap(B)' * LinearMap(B))), x) ≈ 3 * (B'B) * x
    end
    # The concatenations each have a representation that keeps the blocks, so each one's own
    # structure survives: a stack for `vcat`, a join for `hcat`, a stack of joins for `hvcat`.
    @test !opaque(vcat(LinearMap(B), LinearMap(C)))
    @test wrap(vcat(LinearMap(B), LinearMap(C))) isa PureQPBase.StackedOperator
    @test wrap(hcat(LinearMap(B), LinearMap(C))) isa PureQPBase.JoinedOperator
    let joined = wrap(hcat(LinearMap(B), 2 * LinearMap(C)))
        @test joined.blocks[1] === B
        @test joined.blocks[2] == 2C
        @test Matrix(joined) ≈ [B 2C]
    end
    let grid = wrap(hvcat((2, 2), LinearMap(B), LinearMap(C), LinearMap(C), LinearMap(B)))
        @test grid isa PureQPBase.StackedOperator
        @test map(b -> b isa PureQPBase.JoinedOperator, grid.blocks) == (true, true)
        @test Matrix(grid) ≈ [B C; C B]
        # A scalar reaches every block of the grid.
        @test Matrix(wrap(3 * hvcat((2, 2), LinearMap(B), LinearMap(C), LinearMap(C), LinearMap(B)))) ≈ 3 * [B C; C B]
    end
    # A constant block is a `Fill`, which is readable and costs no storage.
    @test wrap(LinearMaps.FillMap(2.0, (3, n))) == fill(2.0, 3, n)
    @test wrap(LinearMaps.FillMap(2.0, (3, n))) isa PureQPBase.Fill
    @test wrap(2 * LinearMaps.FillMap(1.5, (3, n))) == fill(3.0, 3, n)
    let joined = wrap(hcat(LinearMap(B), LinearMaps.FillMap(0.0, (n, 2))))
        @test joined isa PureQPBase.JoinedOperator
        @test PureQPBase.is_materializable(joined)
        @test Matrix(joined) == [B zeros(n, 2)]
    end
    # A stack whose block is opaque keeps the stack: the opaque block becomes a `ProductOperator`
    # inside it rather than costing its sibling its structure.
    let stacked = wrap(vcat(LinearMap(B), fn))
        @test stacked isa PureQPBase.StackedOperator
        @test map(b -> b isa PureQPBase.ProductOperator, stacked.blocks) == (false, true)
        # Compared by its products: the opaque block has no entries, so the stack has none to read
        # over those rows either, and asking for them is what it refuses.
        xs, ys = randn(n), randn(2n)
        @test mul!(zeros(2n), stacked, xs) ≈ [B; B] * xs
        @test mul!(zeros(n), adjoint(stacked), ys) ≈ [B; B]' * ys
        @test_throws "no entries to read" stacked[n + 1, 1]
    end
    @test opaque(LinearMap(I, n))
    # Only real entries have a place in the base.
    @test opaque(LinearMap(complex.(B)))
    @test opaque(LinearMap(B) * im)

    # The wrapped map keeps the traits it declared, which is all an opaque map can offer.
    sym = LinearMap{Float64}(
        (y, x) -> mul!(y, B + B', x), n, n; issymmetric = true, ismutating = true
    )
    @test PureQPBase.is_symmetric(wrap(sym))
end

@testitem "a SciMLOperator is wrapped, or unwrapped, by what it carries" begin
    using PureQPBase, LinearAlgebra, SciMLOperators, Krylov, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(34)

    n, m = 50, 30
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    free = (scaling = 0, linsys = :indirect, factorize = false)
    # `as_operator` is the door: the extension's `setup` method sends every argument through
    # it before the problem is built, so a test of the wrapping calls exactly that.
    Ext = Base.get_extension(PureQPBase, :PureQPBaseSciMLOperatorsExt)
    wrap(M) = Ext.as_operator(Float64, M)

    # The in-place signature is `op(w, v, u, p, t)`: `w` receives the result, `v` is the
    # vector being multiplied. A four-argument function is the out-of-place form instead,
    # and writing into its first argument overwrites the caller's vector.
    Pd, Ad = Matrix(P), A
    Pop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Pd, v), zeros(n), zeros(n);
        op_adjoint = (w, v, u, p, t) -> mul!(w, Pd', v),
        islinear = true, issymmetric = true, isposdef = true,
    )
    Aop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Ad, v), zeros(n), zeros(m);
        op_adjoint = (w, v, u, p, t) -> mul!(w, Ad', v), islinear = true,
    )

    # `symmetric` and `posdef` come from the operator's own traits rather than the caller.
    prob, = backend_for(wrap(Pop), q, wrap(Aop), l, u; free...)
    @test prob.P isa PureQPBase.ProductOperator
    @test PureQPBase.is_symmetric(prob.P)
    @test PureQPBase.is_convex(Float64, prob.P, 1.0e-6)

    # An operator and a matrix mix: only the operator is wrapped.
    mixed, = backend_for(wrap(Pop), q, wrap(A), l, u; free...)
    @test mixed.P isa PureQPBase.ProductOperator
    @test mixed.A isa Matrix

    # An operator holding entries is unwrapped, not wrapped: its entries are what the
    # equilibration and the factoring backends need, and the matrix itself is what the
    # problem holds.
    Pmat = Matrix(P)
    mat, = backend_for(wrap(MatrixOperator(Pmat)), q, A, l, u)
    @test mat.P === Pmat

    # Unwrapping keeps the type, so a diagonal operator still reaches the diagonal backend
    # rather than a dense one.
    d = 2.0 .+ rand(n)
    dg, = backend_for(wrap(DiagonalOperator(d)), q, Diagonal(ones(n)), fill(-1.0, n), fill(1.0, n))
    @test dg.P isa Diagonal

    # `Aᵀ` runs every iteration, so an operator that cannot supply one is refused when the
    # problem is built rather than failing inside the first product.
    noadj = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Ad, v), zeros(n), zeros(m); islinear = true
    )
    @test_throws "op_adjoint" backend_for(wrap(Pop), q, wrap(noadj), l, u; free...)

    # A composed operator carries no scratch until `cache_operator` gives it some, so it is
    # refused here rather than failing partway into the first iteration.
    @test_throws "cache_operator" backend_for(
        wrap(Pop), q, wrap(Aop * DiagonalOperator(ones(n))), l, u; free...
    )
end

@testitem "an operator declared symmetric is checked against its products" begin
    using PureQPBase, LinearAlgebra, Krylov, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(32)
    n, m = 20, 10
    S = randn(n, n)
    A = PureQPBase.ProductOperator{Float64}(randn(m, n))
    q, l, u = randn(n), fill(-1.0, m), fill(1.0, m)

    # Entries are never read, so the declaration is the only symmetry a solver has; a false
    # one would make CG iterate on a matrix that is not the problem's.
    lie = PureQPBase.ProductOperator{Float64}(S'S + triu(S); symmetric = true, posdef = true)
    @test_throws "P is declared symmetric but is not" backend_for(
        lie, q, A, l, u; scaling = 0, linsys = :indirect, factorize = false
    )
end
