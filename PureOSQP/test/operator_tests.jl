@testitem "a products-only operator solves what its matrix solves" begin
    using LinearAlgebra, Krylov, Random
    Random.seed!(31)

    # No `getindex` anywhere on this type or its adjoint, so anything that reaches for an
    # entry fails rather than quietly working through a fallback.
    struct MulOnly{T}
        M::Matrix{T}
    end
    struct MulOnlyAdjoint{T}
        parent::MulOnly{T}
    end
    Base.size(o::MulOnly) = size(o.M)
    Base.size(o::MulOnlyAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::MulOnly) = MulOnlyAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::MulOnly, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::MulOnlyAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)

    n, m = 50, 30
    # Well conditioned: the operator runs unpreconditioned, and an ill-conditioned reduced
    # matrix would separate the two paths by conditioning rather than by representation.
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    Pop = PureOSQP.ProductOperator{Float64}(MulOnly(Matrix(P)); symmetric = true, posdef = true)
    Aop = PureOSQP.ProductOperator{Float64}(MulOnly(A))

    dense = solve(P, q, A, l, u; opts...)
    operator = solve(Pop, q, Aop, l, u; opts...)
    @test operator.status === PureOSQP.SOLVED
    @test operator.x ≈ dense.x rtol = 1.0e-6

    # Polishing copies entries into a dense factorization, which products cannot answer.
    @test_throws "is_materializable" solve(Pop, q, Aop, l, u; opts..., polishing = true)
end

@testitem "a LinearMap reaches the solver" begin
    using LinearAlgebra, LinearMaps, Krylov, Random
    Random.seed!(33)

    n, m = 50, 30
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    Pmap = LinearMap(Matrix(P); issymmetric = true, isposdef = true)
    Amap = LinearMap(A)

    dense = solve(P, q, A, l, u; opts...)
    mapped = solve(Pmap, q, Amap, l, u; opts...)
    @test mapped.status === PureOSQP.SOLVED
    @test mapped.x ≈ dense.x rtol = 1.0e-6

    # A map and a matrix mix.
    @test solve(Pmap, q, A, l, u; opts...).status === PureOSQP.SOLVED
end

@testitem "a SciMLOperator reaches the solver" begin
    using LinearAlgebra, SciMLOperators, Krylov, Random
    Random.seed!(34)

    n, m = 50, 30
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

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

    dense = solve(P, q, A, l, u; opts...)
    opped = solve(Pop, q, Aop, l, u; opts...)
    @test opped.status === PureOSQP.SOLVED
    @test opped.x ≈ dense.x rtol = 1.0e-6

    cached = cache_operator(Aop * DiagonalOperator(ones(n)), zeros(n))
    @test solve(Pop, q, cached, l, u; opts...).status === PureOSQP.SOLVED
end

@testitem "an operator solve does not allocate per iteration" begin
    using LinearAlgebra, SciMLOperators, Krylov, Random
    Random.seed!(35)

    # `Aᵀy` runs every iteration, and a SciMLOperator's `adjoint` builds a new object, so a
    # wrapper that rebuilt it per product would allocate in proportion to the iteration
    # count. Holding the total flat across two iteration counts is what proves it does not.
    # The matrix travels as the operator's `p` rather than as a captured variable: a
    # `@testitem` body is module top level, so a captured matrix is a non-constant global and
    # every product would allocate through a dynamic dispatch, hiding what is being measured.
    n = 40
    Ad = Matrix(0.6I, n, n) + diagm(-1 => fill(0.3, n - 1))
    Aop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, p, v), zeros(n), zeros(n);
        op_adjoint = (w, v, u, p, t) -> mul!(w, p', v), islinear = true, p = Ad,
    )
    Pop = FunctionOperator(
        (w, v, u, p, t) -> (w .= 2 .* v), zeros(n), zeros(n);
        op_adjoint = (w, v, u, p, t) -> (w .= 2 .* v),
        islinear = true, issymmetric = true, isposdef = true,
    )
    q = randn(n)
    l, u = fill(-1.0, n), fill(1.0, n)

    results = map((25, 100)) do iters
        # A tolerance no run can reach, so both stop at `max_iter` while the termination
        # check still runs every iteration.
        ws = setup(
            Pop, q, Aop, l, u; scaling = 0, max_iter = iters,
            eps_abs = 1.0e-30, eps_rel = 1.0e-30, check_termination = 1,
        )
        solve!(ws)
        # Each measured run starts from the same warm start as the one that reports `iter`,
        # so the two describe the same work. A solve resumed from a converged workspace stops
        # in a handful of iterations and would measure something else entirely.
        warm_start!(ws; x = zeros(n), y = zeros(n))
        iters_taken = solve!(ws).iter
        warm_start!(ws; x = zeros(n), y = zeros(n))
        (@allocated(solve!(ws)), iters_taken)
    end
    # Both runs must stop at `max_iter`, or equal totals would say nothing about the
    # per-iteration cost.
    @test first(results)[2] == 25
    @test last(results)[2] == 100
    @test first(results)[1] == last(results)[1]
end

@testitem "an update may not replace P with an operator that is not symmetric" begin
    using LinearAlgebra, Krylov, Random
    Random.seed!(32)
    n, m = 20, 10
    S = randn(n, n)
    A = PureOSQP.ProductOperator{Float64}(randn(m, n))
    q, l, u = randn(n), fill(-1.0, m), fill(1.0, m)
    opts = (scaling = 0, linsys = :indirect)

    truth = PureOSQP.ProductOperator{Float64}(S'S + I; symmetric = true, posdef = true)
    ws = setup(truth, q, A, l, u; opts...)
    @test solve!(ws).status === PureOSQP.SOLVED
    lie = PureOSQP.ProductOperator{Float64}(S'S + triu(S); symmetric = true, posdef = true)
    @test_throws "P is declared symmetric but is not" update!(ws; P = lie)
end

@testitem "every LinearMaps composition solves, and equilibration says why it cannot" begin
    using PureOSQP, PureQPBase, LinearAlgebra, LinearMaps, Krylov, Random

    Random.seed!(5)
    k = 6
    n = k * k
    f1 = Matrix(Symmetric(rand(k, k) + k * I))
    f2 = Matrix(Symmetric(rand(k, k) + k * I))
    P = kron(f1, f2)
    a1, a2 = randn(k + 2, k), randn(k + 1, k)
    A = kron(a1, a2)
    q = randn(n)

    # Every way LinearMaps composes a map of the right shape. `vcat`, a sum, a product and a
    # bare `FunctionMap` reach the solver as a `ProductOperator`; the others unwrap to a type
    # the base holds. All of them solve, which is what "unmaterialized is supported" means.
    maps = (
        ("wrapped matrix", LinearMap(A)),
        ("kron", kron(LinearMap(a1), LinearMap(a2))),
        ("vcat", [LinearMap(A); LinearMap(A)]),
        ("sum", LinearMap(A) + LinearMap(A)),
        ("product", LinearMap(A) * LinearMap(Matrix(1.0I, n, n))),
        ("scaled", 2.0 * LinearMap(A)),
        ("function map", LinearMap(x -> A * x, y -> A' * y, size(A)...)),
    )

    for (label, M) in maps
        @test size(M, 2) == n
        b = M * randn(n)
        l, u = b .- 1, b .+ 1
        # `scaling = 0`: equilibration needs column and row norms, which a map supplying only
        # products has no entries to give. The tolerance is tightened past the default so the
        # comparison means something: an operator reaches the matrix-free backend and a matrix
        # a factorization, so at the default `1e-3` the two converge to different points inside
        # it and agreeing there would say nothing about the representation.
        tol = (; scaling = 0, eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
        s = solve(P, q, M, l, u, OperatorSplitting(); tol...)
        @test s.status == SOLVED
        dense = solve(P, q, Matrix(M), l, u, OperatorSplitting(); tol...)
        @test dense.status == SOLVED
        @test isapprox(s.obj_val, dense.obj_val; rtol = 1.0e-7)
    end

    # At the default `scaling` an operator with no entries is refused, and the refusal names
    # all three ways forward rather than leaving the caller to guess.
    fm = LinearMap(x -> A * x, y -> A' * y, size(A)...)
    b = fm * randn(n)
    err = try
        solve(P, q, fm, b .- 1, b .+ 1, OperatorSplitting())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("no entries to read", err.msg)
    @test occursin("probe = true", err.msg)
    @test occursin("scaling = 0", err.msg)
    @test occursin("structural_rows", err.msg)
end
