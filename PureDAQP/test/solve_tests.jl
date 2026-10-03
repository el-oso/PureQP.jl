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

@testitem "the Gram working set gives up loudly where it cannot decide rank" begin
    using PureDAQP, LinearAlgebra, Random

    #=
    The same fixture as the test above, run through both representations. `:gram` factors
    `Mₐ Mₐᵀ`, whose conditioning is `cond(M)` squared, and at `cond(M) ≈ 1e17` its pivots
    carry no rank information at all. What it must not do is turn that into an answer: a
    direction drawn from a collapsed factorization finds nothing blocking it, which reads as a
    proof of infeasibility, and the problem here is feasible with room to spare.

    `certifiable` is what stops it. The infeasibility claim is checked against the caller's own
    rows before it is made, so the representation that cannot decide rank reports
    `NUMERICAL_ERROR` instead. The guarantee is therefore asymmetric and only one half of it
    belongs to `:gram`: `:rows` solves this, and `:gram` may fail, but neither may call it
    infeasible.
    =#
    rng = MersenneTwister(5)
    n, m = 30, 200
    U, _ = qr(randn(rng, n, n))
    V, _ = qr(randn(rng, m, m))
    A = Matrix(V)[:, 1:n] * Diagonal(exp10.(range(0, -14; length = n))) * Matrix(U)'
    W, _ = qr(randn(rng, n, n))
    P = Matrix(Symmetric(W * Diagonal(exp10.(range(0, -8; length = n))) * W'))
    q = randn(rng, n)
    x0 = randn(rng, n)
    b = A * x0
    l, u = b .- 1.0, b .+ 1.0

    # Past the limit by eight orders of magnitude: the Gram path needs `cond(M) < 1/sqrt(eps)`.
    condM = cond(A / cholesky(Symmetric(P)).U)
    @test condM > inv(sqrt(eps(Float64)))

    rows = solve(P, q, A, l, u, ActiveSet(working_set = :rows))
    gram = solve(P, q, A, l, u, ActiveSet(working_set = :gram))

    @test rows.status == SOLVED
    @test gram.status != PRIMAL_INFEASIBLE
    @test gram.status == NUMERICAL_ERROR
    # A status that is not a solution carries no point to read, so nothing here is a near miss
    # that a caller might use.
    @test all(isnan, gram.x)
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

@testitem "a Kronecker pair reaches the dense pair's answer without forming either" begin
    using PureDAQP, PureQPBase, LinearAlgebra
    include(joinpath(@__DIR__, "helpers.jl"))

    P1, P2, A1, A2, q, l, u = kron_problem(11)
    Pop, Aop = kron_operator_pair(P1, P2, A1, A2)
    ws = PureDAQP.setup(Pop, q, Aop, l, u, ActiveSet(); max_iter = 20_000)
    implicit = PureQPBase.solve!(ws)
    dense = solve(kron(P1, P2), q, kron(A1, A2), l, u, ActiveSet(); max_iter = 20_000)
    @test implicit.status == SOLVED
    @test dense.status == SOLVED
    # The two forms factor `P` differently and reach the same answer to the accuracy the
    # conditioning leaves: `cond(A) = 2.4e11` here, so the agreement stops in the sixth digit
    # and bit-for-bit is neither expected nor asserted.
    @test abs(implicit.obj_val - dense.obj_val) <= 1.0e-6 * abs(dense.obj_val)
    # Feasible and stationary in the caller's own rows, measured against the size of the point:
    # `‖x‖` reaches about 1e6 here, because the smallest singular value of `A` is 1e-11 and the
    # objective keeps falling along directions the rows barely constrain.
    r = kron(A1, A2) * implicit.x
    scale = norm(implicit.x, Inf)
    @test maximum(max.(r .- u, l .- r)) < 1.0e-10 * scale
    @test implicit.prim_res < 1.0e-10 * scale
    @test implicit.dual_res < 1.0e-10

    # Neither operand reached an `m × n` array: the whole workspace is smaller than the reduced
    # matrix the dense pair holds for the same problem.
    @test Base.summarysize(ws) < length(A1) * length(A2) * sizeof(Float64)
end

@testitem "a Kronecker A with a dense P solves like the pair it stands for" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random

    # A dense `R` against an operator `A`: the reduction holds the two of them, so this is the
    # implicit form with the dense factor rather than the Kronecker one.
    rng = MersenneTwister(31)
    n1, n2, m1, m2 = 4, 3, 5, 4
    n, m = n1 * n2, m1 * m2
    A1, A2 = randn(rng, m1, n1), randn(rng, m2, n2)
    G = randn(rng, n, n)
    P = Matrix(Symmetric(G' * G + n * I))
    q = randn(rng, n)
    Aop = PureQPBase.KroneckerOperator(A1, A2)
    b = Aop * randn(rng, n)
    l, u = b .- rand(rng, m), b .+ rand(rng, m)
    implicit = solve(P, q, Aop, l, u, ActiveSet())
    dense = solve(P, q, kron(A1, A2), l, u, ActiveSet())
    @test implicit.status == SOLVED
    @test dense.status == SOLVED
    @test implicit.x ≈ dense.x atol = 1.0e-8
    @test implicit.y ≈ dense.y atol = 1.0e-8
end

@testitem "the reduction's storage does not grow with the number of rows" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps

    include(joinpath(@__DIR__, "helpers.jl"))

    # Rows are added at a fixed variable count, and the workspace is asked how much bigger it
    # got. Forming `A R⁻¹` costs one entry per variable for every added row, so its growth
    # scales with `n`. The bound is a constant number of entries per row instead, independent
    # of `n`: an implicit reduction grows only by the length-`m` vectors the loop holds and by
    # the second Kronecker factor's own rows, which together stay far under it, while a
    # reduction that formed the product misses it by an order of magnitude.
    P1, P2, A1, A2, q, = kron_problem(11)
    Pop = PureQPBase.KroneckerOperator(P1, P2)
    n = size(Pop, 1)
    bound(w1, w4) = let dm = size(w4.prob.A, 1) - size(w1.prob.A, 1)
        (Base.summarysize(w4) - Base.summarysize(w1), dm * 64 * sizeof(Float64))
    end
    # `max_iter = 1` because the question is what `setup` holds, not what a solve reaches.
    function kron_ws(reps)
        Aop = PureQPBase.KroneckerOperator(A1, reduce(vcat, (A2 for _ in 1:reps)))
        b = Aop * ones(n)
        return PureDAQP.setup(Pop, q, Aop, b .- 1, b .+ 1, ActiveSet(); max_iter = 1)
    end
    grew, allowed = bound(kron_ws(1), kron_ws(4))
    @test grew < allowed

    # The same with `A` an opaque map over the Kronecker apply, which is the shape a caller who
    # has only products passes. It reaches the reduction as a `ProductOperator`, read one row
    # at a time through one adjoint product each.
    function map_ws(reps)
        Ak = PureQPBase.KroneckerOperator(A1, reduce(vcat, (A2 for _ in 1:reps)))
        mr, nr = size(Ak)
        fm = LinearMap{Float64}(
            (y, x) -> mul!(y, Ak, x), (y, x) -> mul!(y, adjoint(Ak), x), mr, nr
        )
        b = Ak * ones(nr)
        return PureDAQP.setup(Pop, q, fm, b .- 1, b .+ 1, ActiveSet(); max_iter = 1)
    end
    w1 = map_ws(1)
    @test w1.prob.A isa PureQPBase.ProductOperator
    grew, allowed = bound(w1, map_ws(4))
    @test grew < allowed
end

@testitem "an infeasible problem read through products is still reported infeasible" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps

    # The infeasibility proof is checked against the caller's own rows, which needs `Aᵀy`. An
    # operator that supplies products only answers that product and refuses to be read entry by
    # entry, so a proof drawn on such an `A` has to be settled through the product.
    P = PureQPBase.KroneckerOperator(Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2))
    # Rows 1 and 2 are the same row of `A` with disjoint bounds: no x satisfies both.
    A = [1.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0]
    fm = LinearMap{Float64}(
        (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, adjoint(A), x), 4, 4
    )
    l, u = [1.0, -5.0, -1.0, -1.0], [2.0, -4.0, 1.0, 1.0]
    ws = PureDAQP.setup(P, zeros(4), fm, l, u, ActiveSet())
    @test ws.prob.A isa PureQPBase.ProductOperator
    @test PureQPBase.solve!(ws).status == PRIMAL_INFEASIBLE
end

@testitem "an operator A that supplies products only is solved, residuals included" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps, Random

    # The solve ends by reporting residuals against the caller's own data, which needs `Aᵀy`
    # and an operator answers that only as a product.
    rng = MersenneTwister(21)
    n, m = 8, 20
    X = randn(rng, n, n)
    P = X' * X + I
    A = randn(rng, m, n)
    q = randn(rng, n)
    b = A * randn(rng, n)
    l, u = b .- 0.1, b .+ 0.1
    fm = LinearMap{Float64}(
        (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, adjoint(A), x), m, n
    )
    ws = PureDAQP.setup(P, q, fm, l, u, ActiveSet())
    @test ws.prob.A isa PureQPBase.ProductOperator
    op = PureQPBase.solve!(ws)
    dense = solve(P, q, A, l, u, ActiveSet())
    @test op.status == SOLVED
    @test op.iter == dense.iter
    @test op.x ≈ dense.x atol = 1.0e-10
    @test op.obj_val ≈ dense.obj_val atol = 1.0e-10
    @test op.dual_res < 1.0e-10
    @test op.prim_res < 1.0e-10

    # The same through a Kronecker `A` that is not a dense matrix, so the product is the
    # operator's own and not a `gemv`.
    A1, A2 = randn(rng, 3, 2), randn(rng, 4, 3)
    Ak = PureQPBase.KroneckerOperator(A1, A2)
    nk = size(Ak, 2)
    Xk = randn(rng, nk, nk)
    Pk = Xk' * Xk + I
    bk = Ak * randn(rng, nk)
    sk = solve(Pk, randn(rng, nk), Ak, bk .- 0.1, bk .+ 0.1, ActiveSet())
    @test sk.status == SOLVED
    @test sk.dual_res < 1.0e-10
end

@testitem "both working sets reach the same answer on a Kronecker pair" begin
    using PureDAQP, PureQPBase, LinearAlgebra

    include(joinpath(@__DIR__, "helpers.jl"))

    # Well conditioned, because the Gram representation squares the conditioning of `A R⁻¹` and
    # its pivots collapse from rounding alone on the ill-conditioned instance above.
    P1, P2, A1, A2, q, l, u = kron_problem(
        12; n1 = 6, n2 = 5, m1 = 7, m2 = 6, condP = 10.0, condA = 100.0
    )
    Pop, Aop = kron_operator_pair(P1, P2, A1, A2)
    rows = solve(Pop, q, Aop, l, u, ActiveSet(; working_set = :rows); max_iter = 50_000)
    gram = solve(Pop, q, Aop, l, u, ActiveSet(; working_set = :gram); max_iter = 50_000)
    @test rows.status == SOLVED
    @test gram.status == SOLVED
    @test rows.obj_val ≈ gram.obj_val atol = 1.0e-9
    @test rows.x ≈ gram.x atol = 1.0e-7
end

@testitem "a feasible Kronecker problem is never reported infeasible" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random

    include(joinpath(@__DIR__, "helpers.jl"))

    # The rank decision is made on `|R_ii|` of the `QR` of `Mₐᵀ` whether the rows come from
    # memory or from a product and a solve, so the implicit reduction must survive the same
    # conditioning the dense one does. The claim is the weaker one either must meet: a point
    # satisfies every row with room to spare, so the problem must never be reported infeasible.
    P1, P2, A1, A2, q, = kron_problem(
        13; n1 = 12, n2 = 12, m1 = 16, m2 = 16, condP = 1.0e8, condA = 1.0e14
    )
    Pop, Aop = kron_operator_pair(P1, P2, A1, A2)
    n, m = size(Aop, 2), size(Aop, 1)
    x0 = randn(MersenneTwister(14), n)
    b = Aop * x0
    l, u = b .- 1.0, b .+ 1.0
    @test iszero(maximum(max.(b .- u, l .- b, 0.0)))

    sol = solve(Pop, q, Aop, l, u, ActiveSet(); max_iter = 50_000)
    @test sol.status != PRIMAL_INFEASIBLE
    @test sol.status == SOLVED
    r = Aop * sol.x
    @test maximum(max.(r .- u, l .- r)) < 1.0e-6
    obj(x) = 0.5 * dot(x, Pop * x) + dot(q, x)
    @test obj(sol.x) <= obj(x0)
end

@testitem "what the reduction has no factor of is refused by name" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps

    include(joinpath(@__DIR__, "helpers.jl"))

    P1, P2, A1, A2, q, l, u = kron_problem(
        15; n1 = 5, n2 = 4, m1 = 6, m2 = 5, condP = 10.0, condA = 50.0
    )
    Pop, Aop = kron_operator_pair(P1, P2, A1, A2)
    n = size(Pop, 1)

    # An operator that supplies products only has no Cholesky factor, whatever it declares
    # about itself, and the message names every form that has one.
    Pd = kron(P1, P2)
    opaque = LinearMap{Float64}(
        (y, x) -> mul!(y, Pd, x), n, n; issymmetric = true, isposdef = true
    )
    err = try
        PureDAQP.setup(opaque, q, Aop, l, u, ActiveSet())
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("needs a P it can factor", err.msg)
    @test occursin("KroneckerOperator", err.msg)
    @test occursin("OperatorSplitting()", err.msg)

    # `P1 ⊗ P2 + εI` is not a Kronecker product, so the shift has no `R₁ ⊗ R₂`. It does have a
    # square root built from the factors' eigendecompositions, so the proximal-point loop runs.
    wprox = PureDAQP.setup(Pop, q, Aop, l, u, ActiveSet(; eps_prox = 1.0e-6))
    @test wprox.red.R isa PureQPBase.KroneckerSquareRoot
    @test PureQPBase.solve!(wprox).status == SOLVED

    # The reduction is held concretely, so its representation is fixed at setup.
    ws = PureDAQP.setup(Pop, q, Aop, l, u, ActiveSet())
    @test PureQPBase.solve!(ws).status == SOLVED
    @test_throws "must keep the representation" PureQPBase.update!(ws; P = Pd)
    @test_throws "must keep the representation" PureQPBase.update!(ws; A = kron(A1, A2))
    # Replacing them with operators of the same representation is the supported path.
    PureQPBase.update!(ws; P = PureQPBase.KroneckerOperator(P1, P2))
    @test PureQPBase.solve!(ws).status == SOLVED
end

@testitem "a singular Kronecker P solves through the proximal-point loop" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random

    Random.seed!(19)
    k = 25
    # A first factor of rank `k - 2`, so `P₁ ⊗ P₂` is singular and `eps_prox = 0` cannot
    # factor it. `P₁ ⊗ P₂ + εI` is not a Kronecker product, so what factors it is the square
    # root built from the factors' eigendecompositions rather than `R₁ ⊗ R₂`.
    #
    # The deficiency is structural — two rows and columns that are exactly zero — rather than
    # two zero eigenvalues of a full matrix. Those arrive through `V D Vᵀ` as values near
    # `±eps`, and a pivot that rounds positive is factored instead of refused, so whether the
    # refusal below happens at all would depend on the BLAS. An exactly zero pivot cannot.
    Gs = randn(k - 2, k - 2)
    P1 = zeros(k, k)
    P1[1:(k - 2), 1:(k - 2)] = Matrix(Symmetric(Gs'Gs / (k - 2) + I))
    G = randn(k, k)
    P2 = Matrix(Symmetric(G'G / k + I))
    A1, A2 = randn(k + 5, k), randn(k + 3, k)
    Pop = PureQPBase.KroneckerOperator(P1, P2)
    Aop = PureQPBase.KroneckerOperator(A1, A2)
    n = size(Pop, 1)
    b = Aop * randn(n)
    q, l, u = randn(n), b .- 1, b .+ 1

    @test rank(P1) == k - 2
    @test PureQPBase.cholesky_factor(Pop, 1.0e-5) isa PureQPBase.KroneckerSquareRoot

    # Without the shift the reduction has no factor to build, and says so.
    @test_throws "not positive definite" PureDAQP.solve(Pop, q, Aop, l, u, ActiveSet())

    for eps_prox in (1.0e-5, 1.0e-4)
        alg = ActiveSet(; eps_prox)
        sop = PureDAQP.solve(Pop, q, Aop, l, u, alg; max_iter = 20_000)
        sdn = PureDAQP.solve(kron(P1, P2), q, kron(A1, A2), l, u, alg; max_iter = 20_000)
        @test sop.status == SOLVED
        @test sdn.status == SOLVED
        @test isapprox(sop.obj_val, sdn.obj_val; rtol = 1.0e-9)
        @test maximum(abs, sop.x - sdn.x) < 1.0e-6
    end

    # The operator path holds the four factors, not the product the dense pair holds.
    wop = PureDAQP.setup(Pop, q, Aop, l, u, ActiveSet(; eps_prox = 1.0e-5))
    wdn = PureDAQP.setup(kron(P1, P2), q, kron(A1, A2), l, u, ActiveSet(; eps_prox = 1.0e-5))
    @test Base.summarysize(wop) < Base.summarysize(wdn) / 2
end

@testitem "a sparse P is factored sparsely and the reduction is never formed" begin
    using PureDAQP, PureQPBase, LinearAlgebra, SparseArrays, Random

    Random.seed!(31)
    n = 600
    m = n
    # Banded `P` and a few entries per row of `A`: the shape a dense factor is wrong for.
    P = spdiagm(
        -2 => fill(0.3, n - 2), -1 => fill(0.7, n - 1), 0 => fill(4.0, n),
        1 => fill(0.7, n - 1), 2 => fill(0.3, n - 2)
    )
    A = sprandn(m, n, 4 / n) + sparse(1.0I, m, n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- 1, b .+ 1

    ws = PureDAQP.setup(P, q, A, l, u, ActiveSet(); max_iter = 50_000)
    # `P` keeps its representation and its factor keeps the sparsity: a dense factor of this
    # `P` holds `n²/2` entries where the sparse one holds `O(n)`.
    @test PureQPBase.has_cholesky_factor(P)
    @test nameof(typeof(ws.red.R)) === :SparseCholesky
    # The factor holds `L`, its transpose, two permutations and a scratch vector, so it is a
    # multiple of `nnz(L)` rather than `nnz(L)` itself -- and still a fraction of the `n²`
    # entries a dense factor of this `P` would hold.
    @test Base.summarysize(ws.red.R) < n * n * sizeof(Float64) / 20
    # With no dense triangular factor there is no `A R⁻¹` to form, so rows are derived.
    @test nameof(typeof(ws.red.ws.M)) === :ImplicitRows
    # `A` keeps its sparsity too, held with its transpose so a row is a column. That is two
    # copies of the nonzeros and two sets of index arrays, so the bound is against what the
    # dense matrix it replaces would cost rather than against `nnz` itself.
    @test nameof(typeof(ws.red.ws.M.A)) === :SparseRows
    @test Base.summarysize(ws.red.ws.M.A) < m * n * sizeof(Float64) / 20
    # A row read agrees with the dense matrix's, entry for entry.
    rowbuf = zeros(n)
    Adense = Matrix(A)
    for i in (1, m ÷ 3, m)
        PureQPBase.dense_row!(rowbuf, ws.red.ws.M.A, i)
        @test rowbuf == Adense[i, :]
    end

    s = PureDAQP.solve!(ws)
    sdense = PureDAQP.solve(Matrix(P), q, Matrix(A), l, u, ActiveSet(); max_iter = 50_000)
    @test s.status == SOLVED
    @test sdense.status == SOLVED
    @test s.iter == sdense.iter
    @test isapprox(s.obj_val, sdense.obj_val; rtol = 1.0e-9)
    @test maximum(abs, s.x - sdense.x) < 1.0e-7

    # The whole workspace stays well under what the two dense copies alone would cost.
    @test Base.summarysize(ws) < Base.summarysize(
        PureDAQP.setup(Matrix(P), q, Matrix(A), l, u, ActiveSet(); max_iter = 50_000)
    )
end

@testitem "every LinearMaps composition reaches the reduction" begin
    using PureDAQP, PureQPBase, LinearAlgebra, LinearMaps, Random

    Random.seed!(5)
    k = 6
    n = k * k
    f1 = Matrix(Symmetric(rand(k, k) + k * I))
    f2 = Matrix(Symmetric(rand(k, k) + k * I))
    P = kron(f1, f2)
    a1, a2 = randn(k + 2, k), randn(k + 1, k)
    A = kron(a1, a2)
    q = randn(n)

    # `A` is only multiplied and read by row, so every composition serves: the ones the base
    # holds as their own type, and the ones that arrive as a `ProductOperator` and answer a row
    # with one adjoint product.
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
        b = M * randn(n)
        l, u = b .- 1, b .+ 1
        s = copy(PureDAQP.solve(P, q, M, l, u, ActiveSet(); max_iter = 20_000))
        dense = copy(PureDAQP.solve(P, q, Matrix(M), l, u, ActiveSet(); max_iter = 20_000))
        @test s.status == SOLVED
        @test dense.status == SOLVED
        # The same rows in a different representation take the same path to the same point.
        @test s.iter == dense.iter
        @test isapprox(s.obj_val, dense.obj_val; rtol = 1.0e-9)
        @test maximum(abs, s.x - dense.x) < 1.0e-7
    end

    # `P` is the asymmetric case: it needs a factor, so a map that unwraps to one of the forms
    # the base factors is served and a products-only one is refused by name.
    Pk = kron(LinearMap(f1), LinearMap(f2))
    bk = A * randn(n)
    @test PureDAQP.solve(Pk, q, A, bk .- 1, bk .+ 1, ActiveSet(); max_iter = 20_000).status ==
        SOLVED
    Pfun = LinearMap(x -> P * x, n; issymmetric = true, isposdef = true)
    @test_throws "needs a P it can factor" PureDAQP.solve(
        Pfun, q, A, bk .- 1, bk .+ 1, ActiveSet()
    )
end
