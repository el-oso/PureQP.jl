@testitem "settings merge by keyword as well as by algorithm object" begin
    using PureDAQP, LinearAlgebra

    # The shared keyword method ends by handing the new options to the linear-system backend,
    # which this method does not have, so it carries its own.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    ws = setup(P, q, A, l, u, ActiveSet())
    solve!(ws)

    update_settings!(ws; max_iter = 500)
    @test ws.options.max_iter == 500
    @test solve!(ws).status == SOLVED

    # What the workspace refuses at setup it still refuses here.
    @test_throws ArgumentError update_settings!(ws; linsys = :kkt)
    @test_throws ArgumentError update_settings!(ws; scaling = 10)
    @test_throws ArgumentError update_settings!(ws; polishing = true)
end

@testitem "the solution differentiates without polishing" begin
    using PureDAQP, PureQPBase, LinearAlgebra

    # The active-set method holds the caller's problem rather than an equilibrated copy, so
    # the shared derivative has no scaling to undo. Bounds wide enough to leave the minimum
    # interior: a solution pinned at a vertex has a zero derivative, which a wrong answer
    # would match.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, -2.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = fill(-9.0, 3)
    u = fill(9.0, 3)
    ws = setup(P, q, A, l, u, ActiveSet())
    sol = solve!(ws)
    @test sol.status == SOLVED
    @test sol.x ≈ -(P \ q) atol = 1.0e-9

    # `∂x₁/∂q` of an unconstrained minimum is the first row of `-P⁻¹`.
    d = PureQPBase.adjoint_derivative(ws, [1.0, 0.0], zeros(3))
    @test d.dq ≈ -inv(P)[1, :] atol = 1.0e-8

    f(qq) = solve!(setup(P, qq, A, l, u, ActiveSet())).x[1]
    h = 1.0e-6
    fd = [(f(q .+ h .* (1:2 .== i)) - f(q .- h .* (1:2 .== i))) / (2h) for i in 1:2]
    @test d.dq ≈ fd atol = 1.0e-7
end

@testitem "a solve allocates nothing, proved rather than scanned" begin
    using PureDAQP, StrictMode, StrictModeTest, LinearAlgebra

    # `bench/strictmode_audit.jl` proves the per-iteration kernels. These are the claims the
    # entry points make, which the audit does not reach, proved here from the package's own
    # test environment. A disabled tier prints exactly like a clean one.
    StrictMode.assert_enabled()

    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    ws = setup(P, q, A, l, u, ActiveSet())
    solve!(ws)

    @test_noalloc PureDAQP.report(ws)

    # `solve!` is measured rather than proved: it records its own `solve_time`, and
    # AllocCheck counts the `jl_hrtime` call behind `time_ns` as an allocating runtime call.
    # Every one of the sites it reports for `solve!` is that call.
    #
    # A discarded result is elided by the optimizer, so measuring one says nothing about a
    # caller that keeps it. Storing it is what makes the measurement mean something.
    function hold!(sink, w)
        sink[] = solve!(w)
        return nothing
    end
    # Measured inside a function, so the workspace and the sink are locals with known types.
    # Reading them as globals of the test module makes the call itself allocate, which is
    # what the measurement would then be reporting.
    function held_bytes(w)
        sink = Ref{Any}()
        hold!(sink, w)
        return @allocated hold!(sink, w)
    end
    held_bytes(ws)
    @test iszero(held_bytes(ws))
    @test solve!(ws) === ws.sol

    # `@strict` also reports a union-typed local here. That signal is a lead with no proving
    # counterpart, and it does not answer the same way twice: for this method it reads true
    # in a fresh process and false in one that has already asked inference about the call
    # graph. It is not asserted for that reason. What is provable is asserted instead, and
    # holds: the return type is concrete and the optimization analysis is clean.
    @test_typestable solve!(ws)
end

@testitem "the reported point is the workspace's own" begin
    using PureDAQP, LinearAlgebra

    # What buys the allocation-free solve: `x` and `y` are the workspace's arrays, not copies
    # of them, so the next solve writes through a result still being held. `Solution` says so.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    ws = setup(P, q, A, l, u, ActiveSet())
    sol = solve!(ws)
    @test sol.x === ws.x
    @test sol.y === ws.y

    # Writing the workspace shows through a result already handed out. Asserted directly
    # rather than through a second solve, whose answer need not differ.
    ws.x[1] += 1.0
    @test sol.x[1] == ws.x[1]
    ws.x[1] -= 1.0

    update!(ws; q = [2.0, -1.0])
    again = solve!(ws)
    @test again.x === sol.x            # the same array, refilled
end

@testitem "ActiveSet honours the Solution and Status contract" begin
    using PureQPBase, PureDAQP
    # The assertions live in PureQPBase, which owns `Solution` and `Status`, so every
    # algorithm is held to one statement of what their values mean rather than to whatever
    # its own suite happens to check.
    PureQPBase.conforms(ActiveSet(); eps = 1.0e-8, slow_iters = 1)

    # `using PureDAQP` alone reaches the whole API, because the module re-exports the base's
    # names rather than listing them again. A list would drift from what it copied, and did:
    # four packages each held their own copy of this surface.
    @test issubset(names(PureQPBase), names(PureDAQP))
    # And the names this package adds are its own, so nothing here shadows the base.
    @test setdiff(names(PureDAQP), names(PureQPBase)) ==
        [:ActiveSet, :ActiveSetWorkspace, :PureDAQP, :faster_scan]
end

@testitem "setup then solve! matches a one-shot solve" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(23)
    n, m = 6, 10
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    @test dimensions(ws) == (n, m)
    a = solve!(ws)
    b = solve(P, q, A, -bu, bu, ActiveSet())
    @test a.status == SOLVED
    @test a.x ≈ b.x atol = 1.0e-10
end

@testitem "update! of q, l and u keeps the reduction" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(24)
    n, m = 5, 9
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    solve!(ws)
    red = ws.red

    q2 = randn(rng, n)
    bu2 = bu .* 1.3
    update!(ws; q = q2, l = -bu2, u = bu2)
    # Changing only the vectors must not rebuild the factorization: that is the whole
    # reason a re-solve is cheap here.
    @test ws.red === red
    after = solve!(ws)
    fresh = solve(P, q2, A, -bu2, bu2, ActiveSet())
    @test after.status == SOLVED
    @test after.x ≈ fresh.x atol = 1.0e-9
end

@testitem "update! of P or A rebuilds the reduction" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(25)
    n, m = 5, 8
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    solve!(ws)
    red = ws.red

    A2 = randn(rng, m, n)
    update!(ws; A = A2)
    @test ws.red !== red
    after = solve!(ws)
    @test after.status == SOLVED
    @test after.x ≈ solve(P, q, A2, -bu, bu, ActiveSet()).x atol = 1.0e-9
end

@testitem "cold_start! drops the working set, and the answer is unchanged" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(26)
    n, m = 6, 12
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    warm = solve!(ws)
    cold_start!(ws)
    cold = solve!(ws)
    @test cold.status == SOLVED
    @test cold.x ≈ warm.x atol = 1.0e-10
end

@testitem "what ActiveSet refuses, and why" begin
    using PureDAQP, LinearAlgebra

    P = Matrix(1.0I, 2, 2)
    q = [1.0, 1.0]
    A = [1.0 0.0; 0.0 1.0]
    l = [-1.0, -1.0]
    u = [1.0, 1.0]

    # No backend to choose: the method maintains its own factorization.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); linsys = :kkt)
    # Equilibration would rescale the rows the working set is priced against.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); scaling = 10)
    # Parameters belonging to another algorithm.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); rho = 0.1)
    # The method already ends on the exact solution of the equality QP polishing would solve.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); polishing = true)
    @test solve(P, q, A, l, u, ActiveSet(); polishing = false).status == SOLVED
    # An indefinite P is not a convex problem at all.
    @test_throws ArgumentError solve([1.0 0.0; 0.0 -1.0], q, A, l, u, ActiveSet())
end

@testitem "eps_prox cannot be changed on an existing workspace" begin
    using PureDAQP, LinearAlgebra

    P = Matrix(1.0I, 3, 3)
    q = [1.0, 2.0, 3.0]
    A = Matrix(1.0I, 3, 3)
    ws = setup(P, q, A, fill(-1.0, 3), fill(1.0, 3), ActiveSet())
    # It is built into the Cholesky factor of `P + eps_prox*I`, so changing it would leave
    # the workspace's factorization describing a different problem.
    @test_throws ArgumentError update_settings!(ws, ActiveSet(; eps_prox = 1.0e-4))
    update_settings!(ws, ActiveSet())
    @test solve!(ws).status == SOLVED
end

@testitem "a structurally zero constraint row is priced, not activated" begin
    using PureDAQP, LinearAlgebra

    # The working set is a factorization of its rows, so a zero row raises its column count
    # without its rank and cannot be held. Such a row is therefore an inequality whatever its
    # bounds say: satisfied wherever they admit zero, and violated nowhere else.
    #
    # One variable and two equality rows, the second structurally zero. Before this was
    # distinguished, every case here threw a `DimensionMismatch` out of the factorization, since
    # `l == u` made the zero row an equality that the reset activated unconditionally.
    P = reshape([2.0], 1, 1)
    q = [0.0]
    A = reshape([1.0, 0.0], 2, 1)

    # `0 == 0`: the row holds everywhere, and the problem is the one-row problem.
    s = solve(P, q, A, [1.0, 0.0], [1.0, 0.0], ActiveSet())
    @test s.status == SOLVED
    @test s.x ≈ [1.0] atol = 1.0e-10

    # `0 == 5`: the row holds nowhere, which is infeasibility and is reported as such rather
    # than thrown.
    @test solve(P, q, A, [1.0, 5.0], [1.0, 5.0], ActiveSet()).status == PRIMAL_INFEASIBLE

    # A zero row with slack is the same statement with room in it.
    s = solve(P, q, A, [1.0, -1.0], [1.0, 1.0], ActiveSet())
    @test s.status == SOLVED
    @test s.x ≈ [1.0] atol = 1.0e-10

    # More equality rows than variables, none of them zero: the rank is what bounds the working
    # set, and two consistent rows in one variable are one constraint twice over.
    A2 = reshape([1.0, 2.0], 2, 1)
    s = solve(P, q, A2, [1.0, 2.0], [1.0, 2.0], ActiveSet())
    @test s.status == SOLVED
    @test s.x ≈ [1.0] atol = 1.0e-8
end
