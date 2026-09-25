@testitem "the matrix-free backend reaches the same solution as the direct one" begin
    using LinearAlgebra, SparseArrays, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(30, 60; seed = 3)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    direct = PureOSQP.solve(P, q, A, l, u; opts...)
    indirect = PureOSQP.solve(P, q, A, l, u; opts..., linsys = :indirect)

    @test direct.status == SOLVED
    @test indirect.status == SOLVED
    # The inner solve is inexact, so the iterates are not identical -- demanding equality
    # here would be demanding the wrong thing. Both must solve the original problem.
    @test indirect.x ≈ direct.x atol = 1.0e-5
    @test indirect.obj_val ≈ direct.obj_val atol = 1.0e-5
    @test maximum(kkt_residuals(P, q, A, l, u, indirect.x, indirect.y)) < 1.0e-5

    @test PureOSQP.backend_name(setup(P, q, A, l, u; linsys = :indirect).linsys) == :indirect
end

@testitem "the matrix-free solve allocates nothing per iteration" begin
    using LinearAlgebra, SparseArrays, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    # The reason for a preallocated Krylov workspace, and the reason `ReducedOperator`
    # carries its element type: Krylov compares `eltype(A)` against the vectors' and drops
    # to an allocating path when they disagree, which is silent apart from a warning.
    P, q, A, l, u = random_qp(20, 40; seed = 8)
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8, linsys = :indirect)
    PureOSQP.solve!(ws)
    # Measured inside a function: from top-level scope the call is a dynamic dispatch on an
    # untyped global, and that dispatch allocates 32 bytes of its own.
    solve_bytes(ws) = @allocated PureOSQP.solve_system!(
        ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde
    )
    solve_bytes(ws)      # warm up
    allocs = [solve_bytes(ws) for _ in 1:4]
    @test all(iszero, allocs)
end

@testitem "the matrix-free backend reads the CG settings setup and update_settings! give it" begin
    using LinearAlgebra, SparseArrays, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(20, 40; seed = 9)
    ws = setup(P, q, A, l, u; linsys = :indirect, cg_max_iter = 7, cg_tol_fraction = 0.3)
    @test (ws.linsys.max_iter, ws.linsys.tol_fraction, ws.linsys.tol_reduction) == (7, 0.3, 10)

    update_settings!(ws; cg_max_iter = 1, cg_tol_fraction = 0.05)
    @test (ws.linsys.max_iter, ws.linsys.tol_fraction, ws.linsys.tol_reduction) == (1, 0.05, 10)
    update_settings!(ws, OperatorSplitting(cg_tol_reduction = 4))
    @test (ws.linsys.max_iter, ws.linsys.tol_fraction, ws.linsys.tol_reduction) == (1, 0.05, 4)
    # And the solve honors it: one CG iteration at most, whatever the tolerance asks for.
    ws.rhs_x .= randn(20)
    ws.rhs_z .= randn(40)
    PureOSQP.set_tolerance_level!(ws.linsys, 0.0)
    PureOSQP.solve_system!(ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde)
    @test ws.linsys.kws.stats.niter == 1
end

@testitem "asking for the matrix-free backend without Krylov says so" begin
    using PureQPBase
    # The core cannot build it, and the error has to name the remedy rather than surface a
    # MethodError from somewhere inside `setup`.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    if isnothing(Base.get_extension(PureQPBase, :PureQPBaseKrylovExt))
        @test_throws "needs Krylov.jl" setup(P, q, A, l, u; linsys = :indirect)
    else
        # Krylov is loaded by the other items in this file, so the backend exists here.
        @test PureOSQP.backend_name(setup(P, q, A, l, u; linsys = :indirect).linsys) == :indirect
    end
    @test_throws "linsys must be" setup(P, q, A, l, u; linsys = :nonsense)
    @test_throws "cg_max_iter must be positive" setup(P, q, A, l, u; cg_max_iter = 0)
    @test_throws "cg_tol_fraction must lie in (0, 1]" setup(P, q, A, l, u; cg_tol_fraction = 2.0)
end

@testitem "a structured A preconditions and solves as its dense form does" begin
    using LinearAlgebra, Random, Krylov
    Random.seed!(91)
    n, k = 24, 3
    structured = (
        Diagonal(randn(n) ./ 2),
        Bidiagonal(randn(n) ./ 2, randn(n - 1) ./ 4, :U),
        PureOSQP.RowCoupled(randn(k, n) ./ 4, ones(n - k), collect(1:(n - k))),
    )
    P = Diagonal(rand(n) .+ 0.5)

    # The preconditioner a structured `A` produces is exact against its dense form, and is
    # asserted below.
    #
    # The run that follows is not. `mul!` against a `Bidiagonal` or a `RowCoupled` sums a row
    # in a different order than the dense `gemv` does, so the iterates differ in the last bit
    # from the first step, and conjugate gradients inside an ADMM loop compounds that: the
    # residual crosses the tolerance at a different check, and once `ρ` adapts on a different
    # iteration the two runs are solving slightly different subproblems. Across 60 seeds and
    # three operators the counts differ in 12 of 180 runs, by as much as 175 against 4000, so
    # an equal-iteration assertion here would pin an accident of the seed. What survives the
    # divergence is the answer, and that is what this checks.
    q = randn(n)
    opts = (linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    for A in structured
        m = size(A, 1)
        lo, hi = -rand(m) .- 0.5, rand(m) .+ 0.5
        ws, ref_ws = setup(P, q, A, lo, hi; opts...), setup(P, q, Matrix(A), lo, hi; opts...)
        @test PureOSQP.factorize!(ws.linsys, ws.prob, ws.weights)
        @test PureOSQP.factorize!(ref_ws.linsys, ref_ws.prob, ref_ws.weights)
        @test ws.linsys.precond.dinv == ref_ws.linsys.precond.dinv

        res, ref = PureOSQP.solve!(ws), PureOSQP.solve!(ref_ws)
        @test res.status == SOLVED
        @test res.x ≈ ref.x rtol = 1.0e-7
    end
end

@testitem "the matrix-free backend reaches a tolerance tighter than sqrt(eps)" begin
    using LinearAlgebra, Random, Krylov
    # CG starts from the previous iterate, so late in a solve it can meet its tolerance
    # without taking a step, and ADMM stops moving. Halving the tolerance after
    # `cg_tol_reduction` such solves, down to a floor relative to the right-hand side, is
    # what lets it continue; a fixed `sqrt(eps)` floor stalled this problem at the limit.
    Random.seed!(5)
    n, m = 100, 200
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    tol = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 20_000)
    cg = solve(P, q, A, l, u; linsys = :indirect, tol...)
    direct = solve(P, q, A, l, u; tol...)
    @test cg.status === SOLVED
    @test cg.x ≈ direct.x rtol = 1.0e-6
end

@testitem "Solution.cg_iters counts CG iterations on :indirect and is zero on direct backends" begin
    using LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(20, 40; seed = 11)
    ws = setup(P, q, A, l, u; linsys = :indirect)
    # Copied: a solve refills the workspace's own `Solution`, so the second run would
    # otherwise overwrite the count this one is compared against.
    first = copy(solve!(ws))
    @test first.cg_iters > 0
    @test first.cg_iters == PureOSQP.inner_iterations(ws.linsys)
    # A second solve reports its own iterations, not the workspace's running total.
    update_settings!(ws; eps_abs = 1.0e-6, eps_rel = 1.0e-6)
    second = solve!(ws)
    @test second.cg_iters == PureOSQP.inner_iterations(ws.linsys) - first.cg_iters

    for linsys in (:auto, :dense, :kkt)
        @test iszero(solve(P, q, A, l, u; linsys).cg_iters)
    end
end

@testitem "a caller preconditioner reaches CG and is refreshed when the weights change" begin
    using LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))

    # The exact inverse of the reduced matrix, rebuilt at every refresh.
    mutable struct ExactReduced
        const P::Matrix{Float64}
        const A::Matrix{Float64}
        F::Cholesky{Float64, Matrix{Float64}}
        const ks::Vector{Int}
    end
    ExactReduced(P, A) = ExactReduced(P, A, cholesky(Matrix(1.0I, size(P)...)), Int[])
    function PureOSQP.update_preconditioner!(M::ExactReduced, prob, wt, k::Int)
        push!(M.ks, k)
        M.F = cholesky(Symmetric(M.P + wt.sigma * I + M.A' * Diagonal(wt.w) * M.A))
        return M
    end
    LinearAlgebra.ldiv!(y::AbstractVector, M::ExactReduced, x::AbstractVector) = ldiv!(y, M.F, x)

    P, q, A, l, u = random_qp(15, 30; seed = 12)
    opts = (linsys = :indirect, scaling = 0, eps_abs = 1.0e-8, eps_rel = 1.0e-8)
    M = ExactReduced(P, A)
    ws = setup(P, q, A, l, u; opts..., preconditioner = M)
    @test ws.linsys.precond === M
    iters = Int[]
    for _ in 1:200
        PureOSQP.admm_step!(ws)
        push!(iters, ws.linsys.kws.stats.niter)
    end
    @test maximum(iters) <= 2
    @test PureOSQP.last_solve_converged(ws.linsys)

    sol = solve!(ws)
    @test sol.status === SOLVED
    ref = solve(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8)
    @test sol.x ≈ ref.x atol = 1.0e-5
    # One refresh per factorization, each handed the refactorizations made before it: a new
    # `ρ` goes through `refactor_weights!`, a new `σ` through `factorize!`.
    update_rho!(ws, 0.3)
    update_settings!(ws, OperatorSplitting(sigma = 1.0e-5))
    @test length(M.ks) >= 3
    @test M.ks == range(0, ws.refactor_count - 1)
    @test M.F.U ≈ cholesky(Symmetric(P + 1.0e-5I + A' * Diagonal(ws.weights.w) * A)).U

    # A refresh must hand back the type the backend was built with.
    struct Retyping end
    PureOSQP.update_preconditioner!(::Retyping, prob, wt, k::Int) = I
    LinearAlgebra.ldiv!(y::AbstractVector, ::Retyping, x::AbstractVector) = copyto!(y, x)
    @test_throws "update_preconditioner! must return a preconditioner of the type" setup(
        P, q, A, l, u; opts..., preconditioner = Retyping()
    )

    @test_throws "pass scaling = 0 with it" setup(
        P, q, A, l, u; linsys = :indirect, preconditioner = ExactReduced(P, A)
    )
    @test_throws "pass linsys = :indirect with it" setup(
        P, q, A, l, u; scaling = 0, preconditioner = ExactReduced(P, A)
    )
    # The built-in preconditioners need nothing from the caller's matrices.
    @test setup(P, q, A, l, u; linsys = :indirect, preconditioner = IdentityPreconditioner()).linsys.precond isa
        IdentityPreconditioner
end

@testitem "IdentityPreconditioner reproduces the default on an operator pair" begin
    using LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    # The Jacobi diagonal of a products-only operator is all ones, so preconditioning by it and
    # not preconditioning at all take the same iterates.
    P, q, A, l, u = random_qp(20, 40; seed = 13)
    Po = PureOSQP.ProductOperator{Float64}(Matrix(P); symmetric = true, posdef = true)
    Ao = PureOSQP.ProductOperator{Float64}(A)
    opts = (linsys = :indirect, scaling = 0, eps_abs = 1.0e-7, eps_rel = 1.0e-7)
    jacobi = solve(Po, q, Ao, l, u; opts...)
    plain = solve(Po, q, Ao, l, u; opts..., preconditioner = IdentityPreconditioner())
    @test jacobi.status === SOLVED
    @test (plain.iter, plain.cg_iters, plain.x) == (jacobi.iter, jacobi.cg_iters, jacobi.x)
end

@testitem "the residual stopping rule and the miss count" begin
    using LinearAlgebra, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(20, 40; seed = 14)
    ws = setup(P, q, A, l, u; linsys = :indirect, cg_max_iter = 200)
    ls = ws.linsys
    solve_once!(ws) = PureOSQP.solve_system!(
        ls, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde
    )
    Random.seed!(1)
    ws.rhs_x .= randn(20)
    ws.rhs_z .= randn(40)
    level = 1.0e-6
    PureOSQP.set_tolerance_level!(ls, level)
    atol = ls.tol_fraction * level

    PureOSQP.use_residual_stop!(ls, true)
    fill!(ws.xtilde, 0.0)
    solve_once!(ws)
    @test PureOSQP.last_solve_converged(ls)
    @test norm(ls.kws.r) <= atol
    @test ls.misses == 0

    # One iteration cannot reach that tolerance from zero: a miss.
    update_settings!(ws; cg_max_iter = 1)
    fill!(ws.xtilde, 0.0)
    solve_once!(ws)
    @test !PureOSQP.last_solve_converged(ls)
    @test ls.misses == 1
    @test ls.kws.stats.niter == 1

    # The same budget under the default rule is a miss too.
    PureOSQP.use_residual_stop!(ls, false)
    fill!(ws.xtilde, 0.0)
    solve_once!(ws)
    @test !PureOSQP.last_solve_converged(ls)
    @test ls.misses == 2
    @test PureOSQP.last_solve_converged(setup(P, q, A, l, u).linsys)
end
