@testitem "a dual active-set iteration allocates nothing and trims, proved" begin
    using PureDAQP, PureQPBase, StrictMode, StrictModeTest, LinearAlgebra, Random

    # A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    # Verify against the Base `juliac --trim=safe` compiles, not against stock Base. The working
    # set inserts and deletes columns through ModifiableFactorizations, whose multi-argument
    # `eachindex` and reductions stock inference leaves unresolved — `Base.join` over a tuple of
    # axes, and a `MappingRF` whose function parameters widen to `Function`. juliac patches both
    # before trim inference, so a real trimmed build accepts them and only this scan does not.
    # Checked both ways on 1.12: blocked on stock, clean patched.
    #
    # The patched verifier runs in a child process and falls back to stock, with a warning, for a
    # function that child cannot load. A fallback here would reinstate the stock verdict, so a
    # failure rather than a pass is what it produces — the signatures below are the ones stock
    # rejects.
    StrictModeTest.set_juliac_patches!(true)

    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)

    # A problem `eps_prox > 0` reaches, so the proximal-point outer loop is compiled too.
    Psing = Matrix(Diagonal([ones(n - 2); zeros(2)]))

    # A Kronecker pair, which reduces to `ImplicitRows` over a `KroneckerCholesky`: a row is a
    # product and a solve rather than a view into memory, and pricing is a solve and a product
    # rather than one `gemv`, so the iteration runs different code under the same guarantee.
    Pkron = PureQPBase.KroneckerOperator([2.0 0.5; 0.5 3.0], [4.0 1.0; 1.0 2.0])
    Akron = PureQPBase.KroneckerOperator([1.0 0.5; 0.0 1.0; 1.0 1.0], [1.0 0.0; 0.5 1.0])
    qkron, lkron, ukron = [1.0, 1.0, 0.5, -0.5], fill(-1.0, 6), fill(1.0, 6)

    for (Pk, Ak, qk, lk, uk, alg) in (
            (P, A, q, l, u, ActiveSet()),
            (Psing, A, q, l, u, ActiveSet(; eps_prox = 1.0e-5)),
            (Pkron, Akron, qkron, lkron, ukron, ActiveSet()),
        )
        ws = setup(Pk, qk, Ak, lk, uk, alg)
        # A full solve compiles every specialization the loop reaches on this data.
        solve!(ws)
        W = typeof(ws)
        RD = typeof(ws.red)          # the reduction, carrying its Cholesky concretely
        LW = typeof(ws.red.ws)       # the least-distance workspace the iteration runs on
        ALG = typeof(ws.algorithm)
        V = Vector{Float64}
        both = (:noalloc, :trim_compatible)

        # `solve!` reads the clock, and AllocCheck counts `time_ns`'s `jl_hrtime` foreign
        # call as an allocation. Everything the clock brackets is proved on both counts.
        @test test_signatures([(PureDAQP.solve!, (W,))]; guarantees = (:trim_compatible,)) isa Vector

        @test test_signatures(
            [
                (PureDAQP.run_daqp!, (RD, V, ALG, Int)),
                (PureDAQP.inner_solve!, (RD, V, V, ALG, Int)),
                (PureDAQP.solve_ldp!, (LW, ALG, Int)),
                (PureDAQP.primal_point!, (LW,)),
                (PureDAQP.working_set_multipliers!, (LW,)),
                (PureDAQP.entering_row, (LW, Float64, Bool, UnitRange{Int})),
                (PureDAQP.activate!, (LW, Int, Int8)),
                (PureDAQP.deactivate!, (LW, Int)),
                (PureDAQP.step_and_drop!, (LW, V, Float64)),
                (PureDAQP.singular_step!, (LW, Int, Float64)),
                (PureDAQP.step_toward_multipliers!, (LW, Float64)),
                (PureDAQP.multipliers!, (V, RD)),
                (PureDAQP.primal!, (V, RD, V)),
                (PureDAQP.set_targets!, (RD, V)),
                (PureDAQP.reset_working_set!, (RD,)),
                (PureDAQP.build_solution, (W,)),
            ];
            guarantees = both
        ) isa Vector
    end
end

@testitem "a warm re-solve allocates nothing at run time" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps, Random

    # Measured inside a function so the workspace is a local with a known type; read as a
    # global of the test module, the call itself would allocate. The static proofs above
    # cover every path; this covers the one a caller actually takes, clock included.
    function resolve_bytes(ws)
        solve!(ws)
        return @allocated solve!(ws)
    end

    Random.seed!(2)
    n, m = 10, 24
    X = randn(n, n)
    A = randn(m, n)
    b = A * randn(n)
    q = randn(n)
    l = b .- rand(m)
    u = b .+ rand(m)
    ws = setup(Matrix(X'X / n + I), q, A, l, u, ActiveSet())
    @test iszero(resolve_bytes(ws))

    # The updates a re-solve loop makes carry the same guarantee. Replacing `P` or `A` does
    # not: that branch builds a new reduction, which is what `setup` allocates.
    update_q_bytes(w, qq) = (update!(w; q = qq); @allocated update!(w; q = qq))
    update_lu_bytes(w, ll, uu) = (update!(w; l = ll, u = uu); @allocated update!(w; l = ll, u = uu))
    @test iszero(update_q_bytes(ws, q))
    @test iszero(update_lu_bytes(ws, l, u))

    # A reduction that holds `A` and `R` rather than `A R⁻¹` derives each row from a product and
    # a transposed solve, and prices with a solve and a product, inside the same guarantee. Both
    # pairings of it: an operator `A` against a factor in `P`'s own form, and against a dense
    # triangular one.
    Pk = PureQPBase.KroneckerOperator([2.0 0.5; 0.5 3.0], [4.0 1.0; 1.0 2.0])
    Ak = PureQPBase.KroneckerOperator([1.0 0.5; 0.0 1.0; 1.0 1.0], [1.0 0.0; 0.5 1.0])
    qk, lk, uk = [1.0, 1.0, 0.5, -0.5], fill(-1.0, 6), fill(1.0, 6)
    for Pany in (Pk, kron([2.0 0.5; 0.5 3.0], [4.0 1.0; 1.0 2.0]))
        implicit = PureDAQP.setup(Pany, qk, Ak, lk, uk, ActiveSet())
        @test implicit.red.ws.M isa PureDAQP.ImplicitRows
        @test iszero(resolve_bytes(implicit))
    end

    # An `A` that supplies products only is applied for every row read and every pricing pass,
    # so the guarantee holds exactly as far as the operator's own products allocate nothing.
    # Built inside a function so the closures capture locals of known type.
    function operator_bytes(P, q, A, l, u, allocating)
        m, n = size(A)
        op = if allocating
            LinearMap{Float64}((y, x) -> (y .= A * x), (y, x) -> (y .= A' * x), m, n)
        else
            LinearMap{Float64}((y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x), m, n)
        end
        w = PureDAQP.setup(P, q, op, l, u, ActiveSet())
        @assert w.prob.A isa PureQPBase.ProductOperator
        return resolve_bytes(w)
    end
    Pd = Matrix(X'X / n + I)
    @test iszero(operator_bytes(Pd, q, A, l, u, false))
    @test operator_bytes(Pd, q, A, l, u, true) > 0
end

@testitem "the dual active-set proofs fail on code that allocates or cannot be trimmed" begin
    using StrictMode, StrictModeTest

    # A gate that passes everything proves nothing; each of these must be refused.
    StrictMode.assert_enabled()
    grow(n) = zeros(n)
    @test_throws StrictMode.StrictViolation test_signatures([(grow, (Int,))]; guarantees = (:noalloc,))
    dynamic(r) = r[] + 1
    @test_throws StrictMode.StrictViolation test_signatures(
        [(dynamic, (Base.RefValue{Any},))]; guarantees = (:trim_compatible,)
    )
end
