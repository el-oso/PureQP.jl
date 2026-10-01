@testitem "interior point: the structural corpus passes the referee on both backends" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(30)
    n = 8
    Xn = randn(n, n)
    Pfull = Matrix(Xn'Xn + I)
    Prank = (Y = randn(n, 3); Matrix(Y * Y'))
    Abase = randn(12, n)
    xf = randn(n)
    b = Abase * xf
    cases = Any[
        ("dense PSD", Pfull, randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("rank-deficient P", Prank, randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("LP, P = 0", zeros(n, n), randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("equalities", Pfull, randn(n), Abase[1:4, :], Abase[1:4, :] * xf, Abase[1:4, :] * xf),
        (
            "fixed variable", Pfull, randn(n), Matrix(1.0I, n, n),
            [xf[1]; fill(-5.0, n - 1)], [xf[1]; fill(5.0, n - 1)],
        ),
        ("upper bounds only", Pfull, randn(n), Abase, fill(-Inf, 12), b .+ rand(12)),
        ("lower bounds only", Pfull, randn(n), Abase, b .- rand(12), fill(Inf, 12)),
        ("free rows", Pfull, randn(n), Abase, fill(-Inf, 12), fill(Inf, 12)),
        (
            "mixed infinite bounds", Pfull, randn(n), Abase,
            [iszero(i % 2) ? -Inf : b[i] - rand() for i in 1:12],
            [iszero(i % 3) ? Inf : b[i] + rand() for i in 1:12],
        ),
        (
            "m < n", Pfull, randn(n), Abase[1:3, :],
            Abase[1:3, :] * xf .- 0.5, Abase[1:3, :] * xf .+ 0.5,
        ),
        ("m = 0", Pfull, randn(n), zeros(0, n), Float64[], Float64[]),
        ("n = 1", reshape([2.0], 1, 1), [1.0], reshape([1.0], 1, 1), [-1.0], [1.0]),
        ("diagonal P", Matrix(Diagonal(rand(n) .+ 1)), randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("Symmetric P", Symmetric(Pfull), randn(n), Abase, b .- rand(12), b .+ rand(12)),
        (
            "A::SubArray", Pfull, randn(n), view(Abase, 1:6, 1:n),
            b[1:6] .- rand(6), b[1:6] .+ rand(6),
        ),
        ("A::SparseMatrixCSC", Pfull, randn(n), sparse(Abase), b .- rand(12), b .+ rand(12)),
    ]
    for backend in (:auto, :kkt)
        for (name, P, q, A, l, u) in cases
            ws = setup(P, q, A, l, u, InteriorPoint(); linsys = backend)
            @test ws isa InteriorPointWorkspace
            # Every pair here is dense or has a dense P, which the interior-point ladder
            # serves with the full KKT factorization.
            @test PureQPBase.backend_name(ws.linsys) === :bunchkaufman
            s = solve!(ws)
            @test s.status == SOLVED
            @test s.iter <= 20
            r = maximum(kkt_residuals(Matrix(P), q, Matrix(A), l, u, s.x, s.y))
            @test r < 1.0e-5
        end
    end
end

@testitem "interior point: objective agrees with the C library" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    include(joinpath(@__DIR__, "osqp_oracle.jl"))
    for (n, m, seed) in ((8, 12, 41), (12, 5, 42), (6, 40, 43), (20, 20, 44), (40, 15, 55), (100, 150, 250))
        P, q, A, l, u = random_qp(n, m; seed)
        c = osqp_ref(
            P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9,
            max_iter = 100_000, polish = true
        )
        s = PureQPBase.solve(P, q, A, l, u, InteriorPoint())
        @test s.status == SOLVED
        @test s.iter <= 20
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
        @test abs(s.obj_val - c.info.obj_val) <= 1.0e-6 * max(1, abs(c.info.obj_val))
    end
end

@testitem "interior point: a sparse pair is factored sparsely" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    n = 400
    P, q, A0, _, _ = banded_qp(n, n ÷ 2; band = 2)
    # Box rows on every variable keep the LPs bounded, and the rows of `A0` are built around a
    # point inside the box so the problem stays feasible.
    Random.seed!(82)
    b0 = A0 * (rand(n) .- 0.5)
    A = [A0; sparse(1.0I, n, n)]
    l, u = [b0 .- rand(n ÷ 2); fill(-1.0, n)], [b0 .+ rand(n ÷ 2); fill(1.0, n)]
    cases = [
        # An LP on banded data: the reduced pattern stays banded, which is the form the
        # rule takes when `A` is tall and no row of it spans the variables.
        ("sparse LP", spzeros(n, n), A, SPARSE_FACTOR_BACKENDS),
        # The sparse rungs need a sparse P, so these reach the dense terminal.
        ("dense P, sparse A", Matrix(P), A, (:bunchkaufman,)),
        ("dense LP", zeros(n, n), Matrix(A), (:bunchkaufman,)),
    ]
    for (name, Pc, Ac, backends) in cases
        ws = setup(Pc, q, Ac, l, u, InteriorPoint())
        @test PureQPBase.backend_name(ws.linsys) in backends
        s = solve!(ws)
        @test s.status == SOLVED
        @test s.iter <= 20
        @test maximum(kkt_residuals(Matrix(Pc), q, Matrix(Ac), l, u, s.x, s.y)) < 1.0e-5
    end
end

@testitem "interior point: equality, one-sided and free rows" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(81)
    n, m = 40, 60
    X = randn(n, n)
    P = X'X / n + I
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    l[1:12] .= b[1:12]
    u[1:12] .= b[1:12]
    l[13:24] .= -Inf
    u[25:36] .= Inf
    l[37:42] .= -Inf
    u[37:42] .= Inf
    q = randn(n)
    for sc in (0, 10)
        s = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); scaling = sc)
        @test s.status == SOLVED
        @test s.iter <= 20
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
        # A free row carries no multiplier.
        @test all(iszero, s.y[37:42])
    end

    # Equality rows only: no complementarity, one solve and the full step per iteration,
    # which converges within two outer iterations.
    Ae = A[1:20, :]
    be = Ae * randn(n)
    for sc in (0, 10)
        s = PureQPBase.solve(P, q, Ae, be, be, InteriorPoint(); scaling = sc)
        @test s.status == SOLVED
        @test s.iter <= 2
        @test maximum(kkt_residuals(P, q, Ae, be, be, s.x, s.y)) < 1.0e-5
    end

    # Equality and free rows together, still with no inequality side.
    Af = A[1:30, :]
    lf, uf = [be; fill(-Inf, 10)], [be; fill(Inf, 10)]
    for sc in (0, 10)
        s = PureQPBase.solve(P, q, Af, lf, uf, InteriorPoint(); scaling = sc)
        @test s.status == SOLVED
        @test maximum(kkt_residuals(P, q, Af, lf, uf, s.x, s.y)) < 1.0e-5
        @test all(iszero, s.y[21:30])
    end
end

@testitem "interior point: outer iterations of the reference prototype" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    # The dense generator of `PureIPM/bench/ipm_matrixfree_spike.jl` (`make_instance`), every row
    # two-sided, and with the row mix of `PureIPM/bench/ipm_rowtypes_spike.jl` (`mixed`: 20% equality,
    # 20% lower-only, 20% upper-only, 10% free). Seeds are explicit so the instances do not
    # depend on hashing.
    function spike_problem(n, κ, frac, seed; mixed = false)
        rng = Xoshiro(seed)
        m = n
        U = Matrix(qr(randn(rng, m, m)).Q)
        V = Matrix(qr(randn(rng, n, n)).Q)
        Q = Matrix(qr(randn(rng, n, n)).Q)
        A = U * Diagonal(exp10.(range(0, -log10(κ); length = n))) * V'
        P = Q * Diagonal(exp10.(range(0, -2; length = n))) * Q'
        P = (P + P') / 2
        xstar = randn(rng, n)
        a = A * xstar
        l = a .- (0.5 .+ rand(rng, m))
        u = a .+ (0.5 .+ rand(rng, m))
        ystar = zeros(m)
        if !mixed
            for i in randperm(rng, m)[1:round(Int, frac * m)]
                mag = 0.5 + rand(rng)
                if rand(rng, Bool)
                    ystar[i] = mag
                    u[i] = a[i]
                else
                    ystar[i] = -mag
                    l[i] = a[i]
                end
            end
            return P, -(P * xstar + A' * ystar), A, l, u
        end
        kind = fill(:two, m)
        p = randperm(rng, m)
        j = 0
        for (k, f) in ((:eq, 0.2), (:lo, 0.2), (:up, 0.2), (:free, 0.1)), _ in 1:round(Int, f * m)
            kind[p[j += 1]] = k
        end
        fill!(l, -Inf)
        fill!(u, Inf)
        for i in 1:m
            gap = 0.5 + rand(rng)
            mag = 0.5 + rand(rng)
            active = rand(rng) < frac
            k = kind[i]
            if k === :eq
                l[i] = u[i] = a[i]
                ystar[i] = rand(rng, Bool) ? mag : -mag
            elseif k === :lo
                active ? (l[i] = a[i]; ystar[i] = -mag) : (l[i] = a[i] - gap)
            elseif k === :up
                active ? (u[i] = a[i]; ystar[i] = mag) : (u[i] = a[i] + gap)
            elseif k === :two
                l[i] = a[i] - gap
                u[i] = a[i] + 0.5 + rand(rng)
                if active
                    rand(rng, Bool) ? (u[i] = a[i]; ystar[i] = mag) : (l[i] = a[i]; ystar[i] = -mag)
                end
            end
        end
        return P, -(P * xstar + A' * ystar), A, l, u
    end

    # The sparse KKT backend on dense data, which its fill gate would never select.
    Ext = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
    function sparse_kkt_workspace(P, q, A, l, u; kwargs...)
        Ps, As = sparse(P), sparse(A)
        m, n = size(A)
        prob = PureQPBase.Problem(Float64, Ps, q, As, l, u; scaling = 0)
        wt = PureQPBase.SystemWeights(ones(m), ones(m), 1.0e-8)
        gram = Ext.kkt_gram(Float64, Ps, As, n, m)
        K = Ext.refill_kkt!(gram, Ps, As, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
        F = ldlt(Symmetric(K, :U))
        LD = sparse(F.LD)
        ls = Ext.SparseKKT{Float64, Vector{Float64}, typeof(F)}(
            gram, F, LD, inv.(diag(LD)), F.p, zeros(n + m), zeros(n + m)
        )
        options = Options{Float64}(; PureQPBase.algorithm_defaults(InteriorPoint(), Float64)..., scaling = 0)
        return PureIPM.ipm_workspace(ls, prob, wt, InteriorPoint{Float64}(InteriorPoint(; kwargs...), :auto), options)
    end

    # Outer iterations to `eps = 1e-8` of the prototype `ipm3` in `PureIPM/bench/ipm_rowtypes_spike.jl`
    # on these instances at `δ = 1e-8`: `:exact` (Bunch–Kaufman, no refinement) and
    # `:exact_cholmod_ref1` (CHOLMOD `ldlt`, one refinement step) agree on every one.
    # Keyed by (κ, active fraction), then (two-sided, mixed).
    expected = Dict(
        (1.0, 0.1) => (7, 7), (1.0, 0.5) => (7, 7), (1.0, 0.9) => (7, 7),
        (1.0e3, 0.1) => (6, 6), (1.0e3, 0.5) => (8, 8), (1.0e3, 0.9) => (9, 8),
        (1.0e6, 0.1) => (7, 7), (1.0e6, 0.5) => (9, 9), (1.0e6, 0.9) => (11, 10),
    )
    grid = [(κ, frac) for κ in (1.0, 1.0e3, 1.0e6) for frac in (0.1, 0.5, 0.9)]
    for (index, (κ, frac)) in enumerate(grid)
        seed = 700 + index
        for (k, mixed) in enumerate((false, true))
            data = spike_problem(200, κ, frac, seed; mixed)
            ref = expected[(κ, frac)][k]
            dense = PureQPBase.solve(data..., InteriorPoint(refine_iter = 0); linsys = :kkt, scaling = 0)
            @test dense.status == SOLVED
            @test abs(dense.iter - ref) <= 2
            sparse_ws = sparse_kkt_workspace(data...; refine_iter = 1)
            @test PureQPBase.backend_name(sparse_ws.linsys) === :sparse_kkt
            s = solve!(sparse_ws)
            @test s.status == SOLVED
            @test abs(s.iter - ref) <= 2
        end
    end
end

@testitem "interior point: unsupported inputs are refused by name" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random, Krylov
    Random.seed!(83)
    n, m = 6, 9
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    q, l, u = randn(n), -rand(m), rand(m)

    Pop = PureQPBase.ProductOperator{Float64}(P; symmetric = true, posdef = true)
    Aop = PureQPBase.ProductOperator{Float64}(A)
    @test_throws "supplies products only" setup(Pop, q, Aop, l, u, InteriorPoint())
    @test_throws "supplies products only" setup(Pop, q, Aop, l, u, InteriorPoint(); linsys = :kkt)
    F = cholesky(Symmetric(P))
    for (PP, AA) in ((P, A), (Pop, Aop)), M in (nothing, IdentityPreconditioner(), JacobiPreconditioner(ones(n)))
        @test_throws "uses conjugate gradients only with a caller-supplied preconditioner" setup(
            PP, q, AA, l, u, InteriorPoint(); linsys = :indirect, scaling = 0, preconditioner = M
        )
    end
    @test_throws "pass scaling = 0 with it" setup(
        Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, preconditioner = F
    )
    @test_throws "pass linsys = :indirect with it" setup(
        P, q, A, l, u, InteriorPoint(); linsys = :kkt, scaling = 0, preconditioner = F
    )
    Aprobe = PureQPBase.ProductOperator{Float64}(A; probe = true)
    @test_throws "build them without probe" setup(
        Pop, q, Aprobe, l, u, InteriorPoint(); linsys = :indirect, scaling = 0, preconditioner = F
    )
    # The algorithm is the sixth positional argument, not a keyword.
    @test_throws "algorithm is not an option" setup(P, q, A, l, u, InteriorPoint(); algorithm = :ipm)

    n1, n2 = 3, 4
    K = PureQPBase.KroneckerOperator(randn(n1, n1), randn(n2, n2))
    Pk = Diagonal(fill(2.0, n1 * n2))
    qk, lk, uk = randn(n1 * n2), -rand(n1 * n2), rand(n1 * n2)
    @test_throws "linsys = :kronecker is not available with InteriorPoint()" setup(
        Pk, qk, K, lk, uk, InteriorPoint(); linsys = :kronecker, scaling = 0
    )
    # On `:auto` the Kronecker rung declines too, and every rung below it either needs uniform
    # weights or would form the matrix from `K`'s entries, which two factors stand in place of.
    @test_throws "no interior-point backend serves this pair" setup(
        Pk, qk, K, lk, uk, InteriorPoint(); scaling = 0
    )
    # The refusal names both ways forward, so a caller is not left guessing.
    @test_throws "linsys = :indirect with a caller-supplied preconditioner" setup(
        Pk, qk, K, lk, uk, InteriorPoint(); scaling = 0
    )
    @test_throws "or use OperatorSplitting()" setup(
        Pk, qk, K, lk, uk, InteriorPoint(); scaling = 0
    )
    # The same numbers as a matrix are formed because the caller asked for that, and solve.
    ws = setup(Pk, qk, Matrix(K), lk, uk, InteriorPoint(); scaling = 0)
    @test PureQPBase.backend_name(ws.linsys) !== :kronecker
    @test solve!(ws).status == SOLVED
end

@testitem "interior point: a GPU array is refused by name" tags = [:gpu] begin
    using PureIPM, PureQPBase
    using LinearAlgebra, Random, JLArrays, GPUArraysCore
    JLArrays.allowscalar(false)
    Random.seed!(84)
    n, m = 8, 16
    X = randn(n, n)
    P = jl(Matrix(X'X / n + I))
    A = jl(randn(m, n))
    q, l, u = jl(randn(n)), jl(-ones(m)), jl(ones(m))
    @test_throws "InteriorPoint() runs on the host" setup(P, q, A, l, u, InteriorPoint())
end

@testitem "interior point: the workspace and its step infer concretely" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 15; seed = 5)

    ws = setup(P, q, A, l, u, InteriorPoint())
    W = typeof(ws)
    @test isconcretetype(W)
    @test only(Base.return_types(solve!, (W,))) === Solution{Float64}
    @test only(Base.return_types(PureIPM.ipm_step!, (W,))) === W

    # A re-solve starts from the previous point and a cold start forgets it.
    s1 = solve!(ws)
    @test s1.status == SOLVED
    @test ws.seeded
    s2 = solve!(ws)
    @test s2.status == SOLVED
    @test s2.x ≈ s1.x atol = 1.0e-6
    cold_start!(ws)
    @test !ws.seeded
    s3 = solve!(ws)
    @test s3.iter == s1.iter
    @test s3.x == s1.x
end

@testitem "warm_start! seeds an InteriorPointWorkspace's next solve" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 96)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)
    cold = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); opts...)
    ws = setup(P, q, A, l, u, InteriorPoint(); opts...)

    warm_start!(ws; x = cold.x, y = cold.y)
    @test ws.seeded
    warm = PureQPBase.solve!(ws)
    @test warm.status == SOLVED
    @test warm.iter <= cold.iter
    @test warm.x ≈ cold.x rtol = 1.0e-5

    cold_start!(ws)
    @test !ws.seeded
    @test all(iszero, ws.x)
    @test all(iszero, ws.y)

    x = zeros(12)
    x[1] = NaN
    @test_throws "x must be finite" warm_start!(ws; x = x)
    y = zeros(30)
    y[2] = Inf
    @test_throws "y must be finite" warm_start!(ws; y = y)
    @test_throws "length(x) must be 12" warm_start!(ws; x = zeros(11))
    @test_throws "length(y) must be 30" warm_start!(ws; y = zeros(29))
    # A refused call leaves the seed alone.
    @test !ws.seeded
end

@testitem "interior point: Float32 on the full KKT and the reduced Cholesky" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    rng = Xoshiro(84)
    n, m = 10, 15
    X = randn(rng, n, n)
    P = X'X / n + I
    A = randn(rng, m, n)
    b = A * randn(rng, n)
    q, l, u = randn(rng, n), b .- rand(rng, m), b .+ rand(rng, m)
    # A square `A` with full rank and finite bounds keeps the linear program bounded.
    Al = randn(rng, n, n) + 3I
    le, ue = copy(l), copy(u)
    le[1:4] .= b[1:4]
    ue[1:4] .= b[1:4]
    le[5:7] .= -Inf
    ue[8:9] .= Inf
    cases = (
        qp = (P, q, A, l, u),
        lp = (zeros(n, n), randn(rng, n), Al, -rand(rng, n), rand(rng, n)),
        equality = (P, q, A, le, ue),
    )
    tol = sqrt(eps(Float32))
    for data in cases, (linsys, backend) in ((:kkt, :bunchkaufman), (:dense, :cholesky))
        ws = setup(map(v -> Float32.(v), data)..., InteriorPoint(); linsys)
        @test PureQPBase.backend_name(ws.linsys) === backend
        s = solve!(ws)
        @test s isa Solution{Float32}
        @test s.status == SOLVED
        @test maximum(kkt_residuals(data..., Float64.(s.x), Float64.(s.y))) < tol
    end

    # The defaults follow the element type; a value passed explicitly is kept.
    o32 = default_options(InteriorPoint(), Float32)
    @test o32.eps_abs == o32.eps_rel == o32.eps_prim_inf == o32.eps_dual_inf == tol
    a32 = InteriorPoint{Float32}(InteriorPoint(), :auto)
    @test a32.reg_primal == a32.reg_dual == tol
    @test default_options(InteriorPoint(), Float64).eps_abs == 1.0e-8
    @test InteriorPoint{Float64}(InteriorPoint(), :auto).reg_primal == 1.0e-8
    @test InteriorPoint{BigFloat}(InteriorPoint(), :auto).reg_dual == BigFloat(1.0e-8)
    ws = setup(map(v -> Float32.(v), cases.qp)..., InteriorPoint(reg_primal = 1.0e-6); eps_abs = 1.0e-3)
    @test ws.options.eps_abs == 1.0f-3
    @test ws.algorithm.reg_primal == 1.0f-6
end

@testitem "interior point: BigFloat and dual numbers" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random, ForwardDiff
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(8, 12; seed = 1)

    B = BigFloat
    s = PureQPBase.solve(
        B.(P), B.(q), B.(A), B.(l), B.(u), InteriorPoint(); linsys = :kkt, eps_abs = 1.0e-20, eps_rel = 1.0e-20
    )
    @test s isa Solution{BigFloat}
    @test s.status == SOLVED
    @test maximum(kkt_residuals(B.(P), B.(q), B.(A), B.(l), B.(u), s.x, s.y)) < 1.0e-18

    # `bunchkaufman!` has no method for dual numbers, so they run on the reduced Cholesky.
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    qd = D.(q)
    qd[1] = ForwardDiff.Dual{Nothing}(q[1], 1.0)
    ws = setup(D.(P), qd, D.(A), D.(l), D.(u), InteriorPoint())
    @test PureQPBase.backend_name(ws.linsys) === :cholesky
    sd = solve!(ws)
    @test sd.status == SOLVED
    # The objective is quadratic in `q[1]` while the active set holds, so a central difference
    # of step 1e-3 is exact up to the solve's tolerance divided by the step: 1e-7 at 1e-10.
    h = 1.0e-3
    obj(t) = PureQPBase.solve(
        P, q .+ t .* [1.0; zeros(7)], A, l, u, InteriorPoint(); eps_abs = 1.0e-10, eps_rel = 1.0e-10
    ).obj_val
    fd = (obj(h) - obj(-h)) / (2h)
    @test ForwardDiff.partials(sd.obj_val)[1] ≈ fd atol = 1.0e-6
end

@testitem "interior point: random infeasible problems return checkable certificates" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    for seed in 1:5, sc in (0, 10)
        P, q, A, l, u = primal_infeasible_qp(20, 40, seed)
        s = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); scaling = sc)
        @test s.status == PRIMAL_INFEASIBLE
        @test !has_solution(s.status)
        @test all(isnan, s.x)
        @test is_primal_certificate(A, l, u, s.prim_inf_cert)

        P, q, A, l, u = dual_infeasible_qp(20, 40, seed)
        s = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); scaling = sc)
        @test s.status == DUAL_INFEASIBLE
        @test all(isnan, s.y)
        @test is_dual_certificate(P, q, A, l, u, s.dual_inf_cert)
    end
end

@testitem "interior point: a rising μ runs the certificate tests without ending the run" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # On this instance the diverging multipliers raise `μ` for more than ten iterations in a
    # row before the certificate passes.
    P, q, A, l, u = primal_infeasible_qp(20, 40, 4)
    ws = setup(P, q, A, l, u, InteriorPoint(); scaling = 0)
    s = solve!(ws)
    @test s.status == PRIMAL_INFEASIBLE
    @test ws.alert
    @test ws.flat_merit >= PureIPM.STALL_MERIT
    @test is_primal_certificate(A, l, u, s.prim_inf_cert)
    # A run without a point leaves no seed behind.
    @test !ws.seeded
end

@testitem "interior point: a diverging iterate ends the run without a point" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, Random
    # A backend that solves the Newton system correctly and then scales the recovered `dx` by
    # `factor`, driving `x` away from the data without ever failing to factorize or leaving a
    # certificate to detect: an equality-only problem has no inequality side, so `max_step`
    # never caps the step and every iteration takes the full (blown-up) step.
    mutable struct Blowup{L <: PureQPBase.LinearSystem} <: PureQPBase.LinearSystem
        inner::L
        factor::Float64
    end
    PureQPBase.factorize!(g::Blowup, prob, wt)::Bool = PureQPBase.factorize!(g.inner, prob, wt)
    PureQPBase.refactor_weights!(g::Blowup, prob, wt)::Bool = PureQPBase.refactor_weights!(g.inner, prob, wt)
    PureQPBase.solve_system!(g::Blowup, prob, wt, rx, rz, x, z)::Nothing =
        PureQPBase.solve_system!(g.inner, prob, wt, rx, rz, x, z)
    function PureQPBase.solve_multiplier!(g::Blowup, prob, wt, rx, rz, x, nu)::Nothing
        PureQPBase.solve_multiplier!(g.inner, prob, wt, rx, rz, x, nu)
        x .*= g.factor
        return nothing
    end
    PureQPBase.backend_info(g::Blowup) = PureQPBase.backend_info(g.inner)

    n, m = 8, 5
    Random.seed!(200)
    X = randn(n, n)
    P = X'X / n + I
    A = randn(m, n)
    xstar = randn(n)
    b = A * xstar
    q = randn(n)
    prob = PureQPBase.Problem(Float64, P, q, A, b, b; scaling = 0)
    wt = PureQPBase.SystemWeights(ones(m), ones(m), 1.0e-8)
    options = Options{Float64}(; PureQPBase.algorithm_defaults(InteriorPoint(), Float64)..., scaling = 0, max_iter = 60)
    ls = Blowup(FullKKT(zeros(n), n, m), 3.0)
    ws = PureIPM.ipm_workspace(ls, prob, wt, InteriorPoint{Float64}(InteriorPoint(), :auto), options)
    s = solve!(ws)
    @test s.status == NUMERICAL_ERROR
    @test !has_solution(s.status)
    @test all(isnan, s.x)
    @test all(isnan, s.y)
    # The iterate blew up while every residual stayed finite: this is the divergence ceiling,
    # not the non-finite-residual guard.
    @test ws.diverged
    @test norm(ws.x, Inf) > PureIPM.DIVERGENCE_CEILING(Float64) * PureIPM.iterate_bound(ws)
    @test isfinite(norm(ws.x, Inf))
    @test !ws.seeded
end

@testitem "interior point: the C suite infeasibility cases" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # `primal_dual_infeasibility` from `c_suite_tests.jl`, under the interior-point method.
    P = [1.0 0.0; 0.0 0.0]
    q = [1.0, -1.0]
    A12 = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    A34 = [1.0 0.0; 1.0 0.0; 0.0 1.0]
    l = [0.0, 1.0, 1.0]
    for sc in (0, 10)
        s1 = PureQPBase.solve(P, q, A12, l, [5.0, 3.0, 3.0], InteriorPoint(); scaling = sc)
        @test s1.status == SOLVED
        @test norm(s1.x .- [1.0, 3.0], Inf) < 1.0e-4
        @test norm(s1.y .- [0.0, -2.0, 1.0], Inf) < 1.0e-4
        @test abs(s1.obj_val - (-1.5)) < 1.0e-4

        u2 = [0.0, 3.0, 3.0]
        s2 = PureQPBase.solve(P, q, A12, l, u2, InteriorPoint(); scaling = sc)
        @test s2.status == PRIMAL_INFEASIBLE
        @test is_primal_certificate(A12, l, u2, s2.prim_inf_cert)

        u3 = [2.0, 3.0, Inf]
        s3 = PureQPBase.solve(P, q, A34, l, u3, InteriorPoint(); scaling = sc)
        @test s3.status == DUAL_INFEASIBLE
        @test is_dual_certificate(P, q, A34, l, u3, s3.dual_inf_cert)

        # Both infeasible at once: `x₁ ≤ 0` and `x₁ ≥ 1` conflict, and `x₂ → ∞` descends
        # without leaving the rows. Either certificate proves a true statement.
        u4 = [0.0, 3.0, Inf]
        s4 = PureQPBase.solve(P, q, A34, l, u4, InteriorPoint(); scaling = sc)
        @test s4.status in (PRIMAL_INFEASIBLE, DUAL_INFEASIBLE)
        if s4.status == PRIMAL_INFEASIBLE
            @test is_primal_certificate(A34, l, u4, s4.prim_inf_cert)
        else
            @test is_dual_certificate(P, q, A34, l, u4, s4.dual_inf_cert)
        end
    end
end

@testitem "interior point: factorization failure bumps the regularization" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # A backend that refuses to factorize while `sigma` is below a threshold, and once more
    # at a chosen call, and otherwise defers to the full KKT factorization.
    mutable struct Gate{L <: PureQPBase.LinearSystem} <: PureQPBase.LinearSystem
        inner::L
        threshold::Float64
        fail_at::Int
        calls::Int
    end
    function PureQPBase.factorize!(g::Gate, prob, wt)::Bool
        g.calls += 1
        (wt.sigma < g.threshold || g.calls == g.fail_at) && return false
        return PureQPBase.factorize!(g.inner, prob, wt)
    end
    PureQPBase.solve_system!(g::Gate, prob, wt, rx, rz, x, z)::Nothing =
        PureQPBase.solve_system!(g.inner, prob, wt, rx, rz, x, z)
    PureQPBase.solve_multiplier!(g::Gate, prob, wt, rx, rz, x, nu)::Nothing =
        PureQPBase.solve_multiplier!(g.inner, prob, wt, rx, rz, x, nu)
    PureQPBase.backend_info(g::Gate) = PureQPBase.backend_info(g.inner)
    function gated(P, q, A, l, u; threshold = 0.0, fail_at = 0, kwargs...)
        m, n = size(A)
        prob = PureQPBase.Problem(Float64, P, q, A, l, u; scaling = 0)
        wt = PureQPBase.SystemWeights(ones(m), ones(m), 1.0e-8)
        ls = Gate(FullKKT(zeros(n), n, m), threshold, fail_at, 0)
        options = Options{Float64}(; PureQPBase.algorithm_defaults(InteriorPoint(), Float64)..., scaling = 0)
        return PureIPM.ipm_workspace(ls, prob, wt, InteriorPoint{Float64}(InteriorPoint(; kwargs...), :auto), options)
    end

    P, q, A, l, u = random_qp(20, 30; seed = 91)
    ref = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); scaling = 0)
    @test ref.status == SOLVED

    # The starting point's factorization fails at 1e-8 and succeeds after one bump.
    ws = gated(P, q, A, l, u; threshold = 5.0e-8)
    s = solve!(ws)
    @test s.status == SOLVED
    @test ws.reg_bumps == 1
    @test ws.reg_primal ≈ 1.0e-7
    @test ws.reg_dual ≈ 1.0e-7
    @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
    # Every solve starts from the settings' regularization.
    s = solve!(ws)
    @test s.status == SOLVED
    @test ws.reg_bumps == 1

    # A failure inside the loop, rescued by one bump.
    ws = gated(P, q, A, l, u; fail_at = 4)
    s = solve!(ws)
    @test s.status == SOLVED
    @test ws.reg_bumps == 1
    @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5

    # No threshold the bumps can reach: the run ends without a point.
    for (threshold, bumps) in ((Inf, 5), (5.0e-8, 0))
        local ws, s
        ws = gated(P, q, A, l, u; threshold, max_reg_bumps = bumps)
        @test ws.algorithm.max_reg_bumps == bumps
        s = solve!(ws)
        @test s.status == NUMERICAL_ERROR
        @test !has_solution(s.status)
        @test ws.reg_bumps == bumps
        @test all(isnan, s.x)
        @test all(isnan, s.y)
        @test isnan(s.obj_val)
        @test !ws.seeded
    end
    @test PureQPBase.status_name(NUMERICAL_ERROR) == "numerical error"

    @test_throws "max_reg_bumps must be non-negative" InteriorPoint(max_reg_bumps = -1)
    @test_throws "time_limit must be positive" setup(P, q, A, l, u, InteriorPoint(); time_limit = 0.0)
    @test_throws "eps_prim_inf and eps_dual_inf must be positive" setup(
        P, q, A, l, u, InteriorPoint(); eps_dual_inf = 0.0
    )
end

@testitem "update_settings! on an InteriorPointWorkspace validates and never refactorizes" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 24; seed = 95)
    ws = setup(P, q, A, l, u, InteriorPoint(); eps_abs = 1.0e-6, eps_rel = 1.0e-6)

    # A tolerance is not read by any factorization, so changing it is free, exactly as ADMM's
    # `update_settings!` treats it.
    update_settings!(ws; eps_abs = 1.0e-9, max_iter = 200)
    @test ws.options.eps_abs == 1.0e-9
    @test ws.options.max_iter == 200
    @test ws.options.eps_rel == 1.0e-6          # untouched options survive

    # `reg_primal`/`reg_dual` are free too: every solve resets its regularization from the
    # algorithm parameters before the first iteration, so nothing here needs to trigger a
    # refactorization the way ADMM's `rho`/`sigma` do.
    update_settings!(ws, InteriorPoint(reg_primal = 1.0e-6, reg_dual = 1.0e-6))
    @test ws.algorithm isa InteriorPoint{Float64, Float64, Float64, Int}
    @test ws.algorithm.reg_primal == 1.0e-6
    @test ws.algorithm.reg_dual == 1.0e-6
    @test ws.options.eps_abs == 1.0e-9           # the options are untouched
    got = PureQPBase.solve!(ws)
    @test got.status == SOLVED
    @test ws.reg_primal == 1.0e-6
    @test ws.reg_dual == 1.0e-6
    # The object replaces every parameter: one left out takes its default.
    update_settings!(ws, InteriorPoint(reg_dual = 1.0e-7))
    @test ws.algorithm.reg_primal == 1.0e-8
    @test ws.algorithm.reg_dual == 1.0e-7

    # Rejected, not silently ignored: the backend is part of the workspace's type, and the
    # equilibration factors were computed once from the data setup saw.
    @test_throws "linsys is fixed" update_settings!(ws; linsys = :kkt)
    @test_throws "scaling is fixed" update_settings!(ws; scaling = 0)
    # A rejected call leaves the workspace alone.
    @test ws.options.linsys === :auto
    @test_throws "eps_abs and eps_rel must be non-negative" update_settings!(ws; eps_abs = -1)
    @test_throws "reg_primal must be positive" update_settings!(ws, InteriorPoint(reg_primal = -1.0))
    @test_throws "reg_primal is a parameter of InteriorPoint, not an option" update_settings!(ws; reg_primal = 1.0e-6)
    # `verbose` is a shared option: InteriorPoint() accepts it, unlike a genuine parameter mismatch above.
    update_settings!(ws; verbose = true)
    @test ws.options.verbose
    update_settings!(ws; verbose = false)
    # Any algorithm object that is not this workspace's own is refused, whichever package it
    # came from; a bare subtype stands in for one here.
    @eval struct AnotherAlgorithm <: PureQPBase.QPAlgorithm end
    @test_throws "the algorithm is fixed once the workspace is built" update_settings!(ws, AnotherAlgorithm())

    @test PureQPBase.solve!(ws).status == SOLVED
end

@testitem "update_settings! refreshes the matrix-free IPM backend's CG settings" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 20; seed = 94)
    ws = setup(
        P, q, A, l, u, InteriorPoint(); linsys = :indirect, scaling = 0,
        preconditioner = Diagonal(ones(10)), cg_max_iter = 7, cg_tol_fraction = 0.3
    )
    @test (ws.linsys.max_iter, ws.linsys.tol_fraction) == (7, 0.3)
    update_settings!(ws; cg_max_iter = 3, cg_tol_fraction = 0.05)
    @test (ws.linsys.max_iter, ws.linsys.tol_fraction) == (3, 0.05)
    @test ws.options.cg_max_iter == 3
end

@testitem "the accelerator is refused under InteriorPoint()" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(6, 12; seed = 97)
    # Anything other than `nothing` is refused, and what is refused is never inspected, so the
    # accelerator an operator-splitting solve would pass here need not be built.
    @test_throws "accelerator is used only by OperatorSplitting" setup(
        P, q, A, l, u, InteriorPoint(); accelerator = :an_accelerator
    )
end

@testitem "verbose prints a progress report, and is silent when off (InteriorPoint)" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 21)

    # `Core.stdout` writes to the file descriptor, so capture at that level rather than by
    # rebinding `Base.stdout`, as the ADMM verbose test does.
    function capture(f)
        (path, io) = mktemp()
        try
            redirect_stdout(f, io)
            close(io)
            return read(path, String)
        finally
            rm(path; force = true)
        end
    end

    loud = capture() do
        PureQPBase.solve(P, q, A, l, u, InteriorPoint(); verbose = true)
    end
    quiet = capture() do
        PureQPBase.solve(P, q, A, l, u, InteriorPoint(); verbose = false)
    end

    @test isempty(quiet)
    @test !isempty(loud)
    @test occursin("PureIPM", loud)
    @test occursin("iter", loud)
    @test occursin("status:", loud)
    @test occursin("solved", loud)
    @test occursin("number of iterations:", loud)
    @test occursin("run time:", loud)
    # One row per termination check (`check_termination = 1` by default for InteriorPoint()),
    # plus the header and footer blocks.
    sol = PureQPBase.solve(P, q, A, l, u, InteriorPoint())
    @test count(==('\n'), loud) >= sol.iter

    polished = capture() do
        PureQPBase.solve(P, q, A, l, u, InteriorPoint(); verbose = true, polishing = true)
    end
    @test occursin("polish:", polished)
    @test !occursin("polish:", loud)
end

@testitem "verbose shows the CG column on the matrix-free IPM backend" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 20; seed = 94)

    function capture(f)
        (path, io) = mktemp()
        try
            redirect_stdout(f, io)
            close(io)
            return read(path, String)
        finally
            rm(path; force = true)
        end
    end

    loud = capture() do
        PureQPBase.solve(
            P, q, A, l, u, InteriorPoint(); verbose = true, linsys = :indirect, scaling = 0,
            preconditioner = Diagonal(ones(10))
        )
    end
    @test occursin("cg iters", loud)
    @test occursin("total CG iterations:", loud)
    @test occursin("missed CG solves:", loud)

    # A direct backend prints neither column nor footer line.
    direct = capture() do
        PureQPBase.solve(P, q, A, l, u, InteriorPoint(); verbose = true)
    end
    @test !occursin("cg iters", direct)
    @test !occursin("total CG iterations:", direct)
end

@testitem "interior point: time_limit and an interrupt return the point reached" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(40, 60; seed = 92)
    unlimited = PureQPBase.solve(P, q, A, l, u, InteriorPoint())
    @test unlimited.status == SOLVED
    # A nanosecond is spent by the first iteration, so the run stops there.
    limited = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); time_limit = 1.0e-9)
    @test limited.status == TIME_LIMIT_REACHED
    @test limited.iter == 1 < unlimited.iter
    @test has_solution(limited.status)
    @test all(isfinite, limited.x)
    @test all(isfinite, limited.y)
    @test isfinite(limited.obj_val)

    # A matrix that throws once it has been read a set number of times, as in the ADMM test:
    # the factorization and the products read `A` every iteration, so the throw lands inside
    # the loop.
    mutable struct Fuse{T} <: AbstractMatrix{T}
        A::Matrix{T}
        n::Int
        interrupt::Bool
    end
    Base.size(F::Fuse) = size(F.A)
    function Base.getindex(F::Fuse, i::Int, j::Int)
        F.n -= 1
        iszero(F.n) && throw(F.interrupt ? InterruptException() : ErrorException("boom"))
        return F.A[i, j]
    end

    P, q, A, l, u = random_qp(8, 16; seed = 3)
    F = Fuse(A, typemax(Int), true)
    ws = setup(P, q, F, l, u, InteriorPoint(); check_termination = 0)
    F.n = 20 * length(A)
    sol = solve!(ws)
    @test sol.status == INTERRUPTED
    @test has_solution(sol.status)
    @test 0 < sol.iter < 100
    @test all(isfinite, sol.x)
    @test all(isfinite, sol.y)
    @test isfinite(sol.prim_res)

    F = Fuse(A, typemax(Int), false)
    ws = setup(P, q, F, l, u, InteriorPoint(); check_termination = 0)
    F.n = 20 * length(A)
    @test_throws "boom" solve!(ws)
end

@testitem "interior point: conjugate gradients with a caller-supplied preconditioner" begin
    using PureIPM, PureQPBase
    using LinearAlgebra, SparseArrays, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))

    # The Cholesky factor of `P + σI + Aᵀ diag(w) A`, rebuilt at the starting point, every
    # `every` outer iterations and whenever `σ` changes. `every = 1` is the exact inverse.
    mutable struct LaggedCholesky{T}
        const P::Matrix{T}
        const A::Matrix{T}
        const every::Int
        F::Cholesky{T, Matrix{T}}
        sigma::T
        const ks::Vector{Int}
    end
    LaggedCholesky(P, A; every = 3) =
        LaggedCholesky(Matrix(P), Matrix(A), every, cholesky(Matrix(1.0I, size(P)...)), NaN, Int[])
    function PureQPBase.update_preconditioner!(M::LaggedCholesky, prob, wt, k::Int)
        push!(M.ks, k)
        (k < 0 || iszero(k % M.every) || wt.sigma != M.sigma) || return M
        M.F = cholesky(Symmetric(M.P + wt.sigma * I + M.A' * Diagonal(wt.w) * M.A))
        M.sigma = wt.sigma
        return M
    end
    LinearAlgebra.ldiv!(y::AbstractVector, M::LaggedCholesky, x::AbstractVector) = ldiv!(y, M.F, x)

    P, q, A, l, u = random_qp(20, 30; seed = 93)
    l[1:4] .= u[1:4]
    l[5:7] .= -Inf
    ref = PureQPBase.solve(P, q, A, l, u, InteriorPoint(); linsys = :kkt, scaling = 0)
    @test ref.status == SOLVED
    Pop = PureQPBase.ProductOperator{Float64}(P; symmetric = true, posdef = true)
    Aop = PureQPBase.ProductOperator{Float64}(A)
    for (PP, AA) in ((P, A), (Pop, Aop)), every in (1, 3)
        M = LaggedCholesky(P, A; every)
        ws = setup(PP, q, AA, l, u, InteriorPoint(); linsys = :indirect, scaling = 0, preconditioner = M)
        @test PureQPBase.backend_name(ws.linsys) === :indirect
        @test iszero(ws.algorithm.refine_iter)
        s = solve!(ws)
        @test s.status == SOLVED
        @test s.cg_iters > 0
        @test s.cg_iters == PureQPBase.inner_iterations(ws.linsys)
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
        @test s.x ≈ ref.x atol = 1.0e-5
        # One refresh per factorization: the starting point, then every outer iteration.
        @test M.ks == -1:(s.iter - 1)
        every == 1 && @test abs(s.iter - ref.iter) <= 1
    end

    # A preconditioner so poor that one iteration never reaches the tolerance: every solve is
    # missed, and the third in a row ends the run.
    s = PureQPBase.solve(
        Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, scaling = 0,
        preconditioner = Diagonal(fill(1.0e3, 20)), cg_max_iter = 1
    )
    @test s.status == NUMERICAL_ERROR
    @test s.iter == 1
    @test 0 < s.cg_iters <= 3
    s = PureQPBase.solve(
        Pop, q, Aop, l, u, InteriorPoint(cg_fail_limit = 1); linsys = :indirect, scaling = 0,
        preconditioner = Diagonal(fill(1.0e3, 20)), cg_max_iter = 1
    )
    @test s.status == NUMERICAL_ERROR
    @test iszero(s.iter)

    # A preconditioner that is not positive definite: Krylov abandons each solve, which counts
    # as a miss rather than escaping as an exception.
    d = ones(20)
    d[1] = -1.0
    s = PureQPBase.solve(
        Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, scaling = 0, preconditioner = Diagonal(d)
    )
    @test s.status == NUMERICAL_ERROR
    @test !has_solution(s.status)

    # A refresh must hand back the type the backend was built with.
    struct Retyping end
    PureQPBase.update_preconditioner!(::Retyping, prob, wt, k::Int) = I
    LinearAlgebra.ldiv!(y::AbstractVector, ::Retyping, x::AbstractVector) = copyto!(y, x)
    @test_throws "update_preconditioner! must return a preconditioner of the type" PureQPBase.solve(
        P, q, A, l, u, InteriorPoint(); linsys = :indirect, scaling = 0, preconditioner = Retyping()
    )
    @test_throws "cg_fail_limit must be positive" InteriorPoint(cg_fail_limit = 0)
end
