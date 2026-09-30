@testitem "the documented example gives the documented answer" begin
    using PureDAQP
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    sol = solve(P, q, A, l, u, ActiveSet())
    @test sol.status == SOLVED
    @test sol.x ≈ [0.3, 0.7] atol = 1.0e-9
    @test sol.obj_val ≈ 1.88 atol = 1.0e-9
    # An active-set method stops at a vertex of its working set rather than at a tolerance,
    # so the residuals are at rounding level rather than at `eps_abs`. Not exactly zero: the
    # working-set equations hold exactly in exact arithmetic, but these residuals are
    # recomputed in floating point from the original data, and how they round depends on the
    # BLAS underneath.
    @test sol.prim_res < 1.0e-14
    @test sol.dual_res < 1.0e-12
    # Row 2 is inactive. This one *is* exactly zero: `y` is zeroed and only the working set
    # is written into it, so an inactive row is never assigned at all.
    @test iszero(sol.y[2])
end

@testitem "matches libdaqp on random strictly convex problems" begin
    using PureDAQP, LinearAlgebra, Random
    import DAQP

    rng = MersenneTwister(91)
    for _ in 1:150
        n, m = rand(rng, 2:10), rand(rng, 2:14)
        Q = qr(randn(rng, n, n)).Q
        H = Matrix(Symmetric(Q * Diagonal(exp10.(range(0, 2; length = n))) * Q'))
        f = randn(rng, n)
        A = randn(rng, m, n)
        bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
        sol = solve(H, f, A, -bu, bu, ActiveSet())
        @test sol.status == SOLVED
        xr = DAQP.quadprog(H, f, A, bu, -bu)[1]
        @test norm(sol.x - xr, Inf) / (1 + norm(xr, Inf)) < 1.0e-9
        @test sol.prim_res < 1.0e-10
        @test sol.dual_res < 1.0e-9
    end
end

@testitem "matches libdaqp once the working set is gathered" begin
    using PureDAQP, LinearAlgebra, Random
    import DAQP

    # Larger than the random problems above, so the working set grows deep enough for the
    # factorization to be rebuilt by many updates rather than a handful.
    rng = MersenneTwister(77)
    for _ in 1:25
        n, m = rand(rng, 18:40), rand(rng, 30:70)
        Q = qr(randn(rng, n, n)).Q
        H = Matrix(Symmetric(Q * Diagonal(exp10.(range(0, 2; length = n))) * Q'))
        f = randn(rng, n)
        A = randn(rng, m, n)
        bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
        sol = solve(H, f, A, -bu, bu, ActiveSet())
        @test sol.status == SOLVED
        xr = DAQP.quadprog(H, f, A, bu, -bu)[1]
        @test norm(sol.x - xr, Inf) / (1 + norm(xr, Inf)) < 1.0e-9
        @test sol.prim_res < 1.0e-10
        @test sol.dual_res < 1.0e-9
    end
end

@testitem "equality rows leave the gathered working set in order" begin
    using PureDAQP, LinearAlgebra, Random

    # Rows dropped from the middle of a packed working set shift the block that follows them.
    # Equalities never leave, so a run that drops inequalities around them checks that the
    # gathered rows stay paired with the factorization they belong to.
    rng = MersenneTwister(78)
    for _ in 1:20
        n = rand(rng, 20:30)
        P = Matrix(1.0I, n, n)
        q = randn(rng, n)
        A = randn(rng, n + 12, n)
        b = A * randn(rng, n)
        l = copy(b)
        u = copy(b)
        l[4:end] .-= 0.4 .* abs.(randn(rng, n + 9))
        u[4:end] .+= 0.4 .* abs.(randn(rng, n + 9))
        sol = solve(P, q, A, l, u, ActiveSet())
        @test sol.status == SOLVED
        for i in 1:3
            @test abs(dot(A[i, :], sol.x) - b[i]) < 1.0e-9
        end
        @test sol.prim_res < 1.0e-9
    end
end

@testitem "proximal-point iterations accept a singular P" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(92)
    for _ in 1:30
        n, m = rand(rng, 4:9), rand(rng, 6:16)
        Q = qr(randn(rng, n, n)).Q
        # Two zero eigenvalues: `P` is positive semidefinite but not invertible, which the
        # reduction cannot factor without regularization.
        lam = vcat(ones(n - 2), zeros(2))
        P = Matrix(Symmetric(Q * Diagonal(lam) * Q'))
        q = randn(rng, n)
        A = randn(rng, m, n)
        bu = 0.8 .* abs.(A * randn(rng, n)) .+ 0.1
        sol = solve(P, q, A, -bu, bu, ActiveSet(; eps_prox = 1.0e-4, eta_prox = 1.0e-10))
        @test sol.status == SOLVED
        @test sol.prim_res < 1.0e-8
    end
end

@testitem "an exactly singular P is refused without eps_prox" begin
    using PureDAQP, LinearAlgebra

    # Built to be singular exactly rather than by construction and rounding: a random
    # rank-deficient `P` often comes back from `Q Λ Qᵀ` with eigenvalues near ±1e-16, and
    # Cholesky then succeeds or fails depending on which side of zero they land.
    P = [1.0 0.0; 0.0 0.0]
    q = [1.0, -1.0]
    A = [1.0 0.0; 0.0 1.0]
    l = [-1.0, -1.0]
    u = [1.0, 1.0]
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet())
    sol = solve(P, q, A, l, u, ActiveSet(; eps_prox = 1.0e-4, eta_prox = 1.0e-12))
    @test sol.status == SOLVED
    # Minimizing ½x₁² + x₁ − x₂ over the box puts x₁ at −1 and x₂ at its upper bound.
    @test sol.x ≈ [-1.0, 1.0] atol = 1.0e-6
end

@testitem "a linear program is the extreme case of a singular P" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(44)
    for _ in 1:15
        n, m = rand(rng, 3:7), rand(rng, 8:18)
        P = zeros(n, n)
        q = randn(rng, n)
        A = randn(rng, m, n)
        # The origin is strictly feasible, so the program is bounded.
        bu = abs.(randn(rng, m)) .+ 0.5
        bl = -abs.(randn(rng, m)) .- 0.5
        sol = solve(P, q, A, bl, bu, ActiveSet(; eps_prox = 1.0e-4, eta_prox = 1.0e-10, max_prox = 400))
        @test sol.status == SOLVED
        @test sol.prim_res < 1.0e-7
    end
end

@testitem "equality rows stay in the working set" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(13)
    for _ in 1:40
        n = rand(rng, 3:8)
        P = Matrix(1.0I, n, n)
        q = randn(rng, n)
        A = randn(rng, n + 2, n)
        b = A * randn(rng, n)
        l = copy(b)
        u = copy(b)
        # The first two rows are equalities; the rest are a loose box that cannot bind.
        l[3:end] .-= 10.0
        u[3:end] .+= 10.0
        sol = solve(P, q, A, l, u, ActiveSet())
        @test sol.status == SOLVED
        @test abs(dot(A[1, :], sol.x) - b[1]) < 1.0e-9
        @test abs(dot(A[2, :], sol.x) - b[2]) < 1.0e-9
    end
end

@testitem "a feasible problem is never reported infeasible, however ill conditioned" begin
    using PureDAQP, LinearAlgebra, Random

    #=
    `cond(A R⁻¹)` reaches about 1e17 here. A working set held as the `LDLᵀ` of `Mₐ Mₐᵀ`
    squares that, so its pivots collapse from rounding alone and the step along the
    resulting direction finds nothing to block it — which, taken at face value, is a proof
    of infeasibility drawn from a factorization that had stopped being trustworthy. Held as
    the `QR` of `Mₐᵀ` the conditioning enters once rather than twice, and the rank decision
    is made on `|R_ii|` rather than on a difference of squares, so the working set stays
    sound and the problem is solved rather than given up on.

    The claim under test is the weaker one either representation must meet: the problem is
    feasible, `x0` satisfies every row with room to spare, and it must never be reported
    infeasible.
    =#
    rng = MersenneTwister(5)
    n, m = 30, 200
    U, _ = qr(randn(rng, n, n))
    V, _ = qr(randn(rng, m, m))
    # Spectra chosen so that `cond(A R^-1)` reaches about 1e17: the working set's Gram
    # matrix squares that, which is where a pivot collapses from rounding alone.
    A = Matrix(V)[:, 1:n] * Diagonal(exp10.(range(0, -14; length = n))) * Matrix(U)'
    W, _ = qr(randn(rng, n, n))
    P = Matrix(Symmetric(W * Diagonal(exp10.(range(0, -8; length = n))) * W'))
    q = randn(rng, n)
    x0 = randn(rng, n)
    b = A * x0
    # `x0` satisfies every row with room to spare, so a feasible point demonstrably exists.
    l = b .- 1.0
    u = b .+ 1.0
    @test iszero(maximum(max.(A * x0 .- u, l .- A * x0, 0.0)))

    sol = solve(P, q, A, l, u, ActiveSet())
    @test sol.status != PRIMAL_INFEASIBLE
    @test sol.status == SOLVED
    # Feasible in the caller's own rows, with room to spare inside the unit band the bounds
    # leave, and stationary there: `P x + Aᵀy + q` vanishes. Both are measured against the
    # data as given, not against the reduction.
    r = A * sol.x
    @test maximum(max.(r .- u, l .- r)) < 1.0e-6
    @test sol.prim_res < 1.0e-6
    @test sol.dual_res < 1.0e-8
    # `‖x‖` reaches about 1e6, which is the problem and not a symptom: the smallest
    # eigenvalue of `P` is 1e-8 and the smallest singular value of `A` is 1e-14, so the
    # objective keeps falling along directions the rows barely constrain. It lands far below
    # the feasible point the fixture was built around.
    obj(x) = 0.5 * x' * P * x + q' * x
    @test obj(sol.x) < obj(x0)
end

@testitem "a genuinely infeasible problem is still reported infeasible" begin
    using PureDAQP, LinearAlgebra

    # Two rows on the same variable with disjoint bounds: no x satisfies both.
    P = Matrix(1.0I, 2, 2)
    A = [1.0 0.0; 1.0 0.0]
    sol = solve(P, [0.0, 0.0], A, [1.0, -5.0], [2.0, -4.0], ActiveSet())
    @test sol.status == PRIMAL_INFEASIBLE
end

@testitem "linearly dependent rows that agree are solved, not reported infeasible" begin
    using PureDAQP, LinearAlgebra

    # A row that repeats one already held enters the working set at `R_ii = 0`, which is the
    # value the dependency test reads. Reaching it must not be mistaken for infeasibility, and
    # must not leave the working set's bookkeeping disagreeing with the factorization about
    # how many rows are in it.
    P = Matrix(1.0I, 2, 2)
    dup = solve(P, [0.0, 0.0], [1.0 0.0; 1.0 0.0], [1.0, 1.0], [2.0, 2.0], ActiveSet())
    @test dup.status == SOLVED
    @test dup.x[1] ≈ 1.0

    # A third row that is the sum of the first two: dependent without duplicating either.
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    l = [1.0, 1.0, 2.0]
    u = [3.0, 3.0, 4.0]
    sol = solve(P, [0.0, 0.0], A, l, u, ActiveSet())
    @test sol.status == SOLVED
    r = A * sol.x
    @test maximum(max.(r .- u, l .- r)) < 1.0e-9
end

@testitem "a warm start does not drift on a degenerate problem" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random, Statistics

    # Constraints that are exact combinations of others make a singular working set the
    # normal case rather than the exception, and a warm start carries the factorization from
    # one solve to the next, so an error in it compounds over a run. Each warm answer is
    # checked against a cold solve of the same data.
    #
    # The shape is taken from darnstrom/daqp's 13_warm_start_drift.
    n, mi, md, solves = 30, 60, 40, 300
    rng = MersenneTwister(1)
    H = Matrix(Diagonal(2 .+ rand(rng, n)))
    for i in 1:n, j in 1:(i - 1)
        v = 0.3 * randn(rng) / n
        H[i, j] += v
        H[j, i] += v
    end
    Ai = randn(rng, mi, n)
    C = zeros(md, mi)
    for d in 1:md, _ in 1:3
        C[d, rand(rng, 1:mi)] += randn(rng)
    end
    A = vcat(Matrix(1.0I, n, n), Ai, C * Ai)
    m = size(A, 1)
    f0 = 20 .* randn(rng, n)
    bnd = 0.2 .* rand(rng, mi)
    bu0 = vcat(ones(n), bnd, [0.3 * sum(abs.(C[d, :]) .* bnd) for d in 1:md])
    l = vcat(-bu0[1:n], fill(-Inf, m - n))

    perturbed() = (
        f0 .* (1 .+ 0.3 .* randn(rng, n)),
        vcat(bu0[1:n], bu0[(n + 1):end] .* (1 .+ 0.05 .* randn(rng, m - n))),
    )

    alg = ActiveSet()
    q, u = perturbed()
    ws = PureDAQP.setup(H, q, A, l, u, alg; max_iter = 20_000)
    @test PureQPBase.solve!(ws).status == SOLVED

    for _ in 1:solves
        q, u = perturbed()
        PureQPBase.update!(ws; q, l, u)
        warm = PureQPBase.solve!(ws)
        cold = solve(H, q, A, l, u, alg; max_iter = 20_000)
        @test warm.status == SOLVED
        @test cold.status == SOLVED
        # The working set carried between solves must not move the answer.
        @test maximum(abs, warm.x .- cold.x) < 1.0e-6
        r = A * warm.x
        @test maximum(max.(r .- u, l .- r)) < 1.0e-8
    end
end

@testitem "both working sets reach the same answer where both can" begin
    using PureDAQP, LinearAlgebra, Random

    # The two representations answer the same questions, so on a problem neither has trouble
    # with they must reach the same answer. Not necessarily by the same route: the pivots
    # differ in their last digits, so a row priced at the tolerance can enter in one and not
    # the other, and the iteration counts then differ by a step or two.
    rng = MersenneTwister(404)
    for _ in 1:20
        n, m = rand(rng, 3:14), rand(rng, 4:20)
        G = randn(rng, n, n)
        P = Matrix(Symmetric(G' * G + n * I))
        q = randn(rng, n)
        A = randn(rng, m, n)
        b = A * randn(rng, n)
        l = b .- rand(rng, m)
        u = b .+ rand(rng, m)
        sq = solve(P, q, A, l, u, ActiveSet(; working_set = :rows); max_iter = 50_000)
        sg = solve(P, q, A, l, u, ActiveSet(; working_set = :gram); max_iter = 50_000)
        @test sq.status == SOLVED
        @test sg.status == SOLVED
        @test sq.obj_val ≈ sg.obj_val atol = 1.0e-9
        @test sq.x ≈ sg.x atol = 1.0e-7
    end
end

@testitem "the working set is refused an unknown name" begin
    using PureDAQP

    # `:qr` and `:ldl` name factorizations rather than what is factored, which is the axis
    # this setting chooses on, and `:ldl` is additionally what `:gram` already uses.
    @test_throws "working_set must be :rows or :gram" ActiveSet(; working_set = :ldl)
    @test_throws "working_set must be :rows or :gram" ActiveSet(; working_set = :qr)
    @test ActiveSet().working_set === :rows
    @test ActiveSet(; working_set = :gram).working_set === :gram
end

@testitem "infeasibility proved from a full working set is still infeasibility" begin
    using PureDAQP, LinearAlgebra

    # With one variable the working set fills after a single row, so the contradiction
    # between these two is met by `full_set_step!` rather than by the singular branch. The
    # direction that proves it includes the entering row, which is not in the working set;
    # a certificate rebuilt from the set alone does not satisfy `Aᵀy = 0`, fails the check
    # against the caller's rows, and downgrades a sound proof to a numerical breakdown.
    one = reshape([1.0], 1, 1)
    both = reshape([1.0, 1.0], 2, 1)
    for ws in (:rows, :gram)
        sol = solve(one, [0.0], both, [1.0, -5.0], [2.0, -4.0], ActiveSet(; working_set = ws))
        @test sol.status == PRIMAL_INFEASIBLE
    end

    # The same, with the set filled by rows that are not the contradicting pair.
    A = [1.0 0.0; 0.0 1.0; 1.0 0.0]
    sol = solve(Matrix(1.0I, 2, 2), [0.0, 0.0], A, [1.0, 100.0, -5.0], [2.0, 101.0, -4.0], ActiveSet())
    @test sol.status == PRIMAL_INFEASIBLE
end

@testitem "an update! that changes which rows are equalities is not warm started" begin
    using PureDAQP, PureQPBase, LinearAlgebra

    # An equality's multiplier is free in sign and never blocks a step, so a working set
    # carried across a change of equality status holds a row on terms that no longer apply.
    P = Matrix(1.0I, 2, 2)
    q = [-1.0, 1.0]
    A = Matrix(1.0I, 2, 2)
    alg = ActiveSet()

    ws = PureDAQP.setup(P, q, A, [-1.0, -1.0], [-1.0, 1.0], alg)   # row 1 an equality
    PureQPBase.solve!(ws)
    PureQPBase.update!(ws; l = [-1.0, -1.0], u = [1.0, 1.0])       # no longer one
    loosened = PureQPBase.solve!(ws)
    @test loosened.status == SOLVED
    @test loosened.x ≈ solve(P, q, A, [-1.0, -1.0], [1.0, 1.0], alg).x atol = 1.0e-8

    ws2 = PureDAQP.setup(P, q, A, [-1.0, -1.0], [1.0, 1.0], alg)   # no equalities
    PureQPBase.solve!(ws2)
    PureQPBase.update!(ws2; l = [-1.0, -1.0], u = [-1.0, 1.0])     # row 1 becomes one
    tightened = PureQPBase.solve!(ws2)
    @test tightened.status == SOLVED
    @test tightened.x ≈ solve(P, q, A, [-1.0, -1.0], [-1.0, 1.0], alg).x atol = 1.0e-8

    # Bounds that move without changing any equality keep the working set.
    ws3 = PureDAQP.setup(P, q, A, [-2.0, -2.0], [2.0, 2.0], alg)
    PureQPBase.solve!(ws3)
    PureQPBase.update!(ws3; l = [-1.5, -1.5], u = [1.5, 1.5])
    @test ws3.warm
    @test PureQPBase.solve!(ws3).x ≈ solve(P, q, A, [-1.5, -1.5], [1.5, 1.5], alg).x atol = 1.0e-8
end

@testitem "a scan window reaches the same answer as scanning every row" begin
    using PureDAQP, LinearAlgebra, Random

    # A window changes which violated row enters, not which point is optimal.
    rng = MersenneTwister(808)
    for _ in 1:12
        n, m = rand(rng, 5:20), rand(rng, 300:400)
        G = randn(rng, n, n)
        P = Matrix(Symmetric(G' * G + n * I))
        q = randn(rng, n)
        A = randn(rng, m, n)
        b = A * randn(rng, n)
        l = b .- rand(rng, m)
        u = b .+ rand(rng, m)
        every = solve(P, q, A, l, u, ActiveSet(); max_iter = 50_000)
        window = solve(P, q, A, l, u, ActiveSet(; scan = :window); max_iter = 50_000)
        @test every.status == SOLVED
        @test window.status == SOLVED
        @test every.obj_val ≈ window.obj_val atol = 1.0e-9
    end
    @test_throws "scan must be :all or :window" ActiveSet(; scan = :partial)
    @test ActiveSet().scan === :all
end

@testitem "faster_scan measures both settings and picks one" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(99)
    n, m = 20, 400
    G = randn(rng, n, n)
    P = Matrix(Symmetric(G' * G + n * I))
    q = randn(rng, n)
    A = randn(rng, m, n)
    b = A * randn(rng, n)
    r = faster_scan(P, q, A, b .- rand(rng, m), b .+ rand(rng, m); reps = 2, max_iter = 50_000)
    @test r.scan in (:all, :window)
    # Whichever it picked is the one it timed as faster, and the ratio says by how much.
    @test (r.scan === :all) == (r.all_ms <= r.window_ms)
    @test r.ratio >= 1
    @test r.iter_all > 0
    @test r.iter_window > 0
    @test_throws "reps must be at least 1" faster_scan(P, q, A, b .- 1, b .+ 1; reps = 0)
end

@testitem "a solution kept across a solve is refilled without allocating" begin
    using PureDAQP, PureQPBase, LinearAlgebra

    # This method refills the workspace's own `Solution` rather than building one, so two
    # results read from one workspace are the same object. A caller keeping a result needs its
    # own: `copy` builds one and allocates, `copyto!` refills one it already holds and does
    # not, which is what a loop under a no-allocation guarantee needs.
    P = Matrix(2.0I, 3, 3)
    A = Matrix(1.0I, 3, 3)
    ws = PureDAQP.setup(P, [1.0, -2.0, 0.5], A, fill(-1.0, 3), fill(1.0, 3), ActiveSet())
    kept = copy(PureQPBase.solve!(ws))
    firstx = copy(kept.x)

    PureQPBase.update!(ws; q = [-1.0, 2.0, -0.5])
    second = PureQPBase.solve!(ws)
    @test second.x != firstx        # the workspace moved on
    @test kept.x == firstx          # ours did not

    PureQPBase.copyto!(kept, second)
    @test kept.x == second.x
    @test kept.status == second.status
    @test kept.obj_val == second.obj_val
    @test kept.iter == second.iter

    PureQPBase.update!(ws; q = [3.0, 3.0, 3.0])
    third = PureQPBase.solve!(ws)
    @test kept.x != third.x         # still independent

    refill(dest, src) = @allocated PureQPBase.copyto!(dest, src)
    refill(kept, third)
    @test iszero(refill(kept, third))

    smaller = PureDAQP.setup(
        Matrix(2.0I, 2, 2), [1.0, 1.0], Matrix(1.0I, 2, 2),
        fill(-1.0, 2), fill(1.0, 2), ActiveSet()
    )
    other = copy(PureQPBase.solve!(smaller))
    @test_throws "must match the source" PureQPBase.copyto!(other, third)
end
