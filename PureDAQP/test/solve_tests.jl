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

@testitem "the Gram working set never calls a feasible problem infeasible" begin
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
    # Which side of the disjunction `:gram` lands on at this conditioning is not something the
    # representation promises, so both are checked for what they carry: a status that is not a
    # solution carries no point to read, and a status that is one carries the solution. What
    # `:gram` may not do is report a status that reads as an answer beside a point that is not.
    if gram.status == SOLVED
        @test maximum(abs, gram.x .- rows.x) <= 1.0e-9 * max(1.0, maximum(abs, rows.x))
    else
        @test gram.status == NUMERICAL_ERROR
        @test all(isnan, gram.x)
    end
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
    # Stationarity scales with the terms of `Px + q + Aᵀy`, not with `x`. `refine_primal!`
    # moves `x` onto the active rows and leaves `y` solved for the point before that step, so
    # what it buys in feasibility it gives back here: `2e-9` of these terms, against a primal
    # residual six orders smaller than the unrefined point's.
    dual_scale = max(
        norm(q, Inf), norm(kron(P1, P2) * implicit.x, Inf), norm(kron(A1, A2)' * implicit.y, Inf)
    )
    @test implicit.dual_res < 1.0e-8 * dual_scale

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

@testitem "a joined A is read directly and updated in its own representation" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random

    Random.seed!(23)
    # `A = [K C]`: a Kronecker block beside a dense one, over the same 12 rows.
    K1, K2 = randn(3, 2), randn(4, 3)
    C = randn(12, 4)
    joined(k1, k2, c) = PureQPBase.JoinedOperator(PureQPBase.KroneckerOperator(k1, k2), c)
    A = joined(K1, K2, C)
    m, n = size(A)
    P = Matrix(Symmetric(randn(n, n) * 0.1)) + 3I
    q = randn(n)
    # Bounds a fixed point satisfies, so the problem stays feasible whatever `A` becomes.
    x0 = randn(n)
    bounds(A) = (Matrix(A) * x0 .- 1, Matrix(A) * x0 .+ 1)
    l, u = bounds(A)

    ws = PureDAQP.setup(P, q, A, l, u, ActiveSet())
    # The join is held as it is: rows come from its blocks, not from a dense copy.
    @test ws.red.ws.M isa PureDAQP.ImplicitRows
    sol = PureQPBase.solve!(ws)
    @test sol.status == SOLVED
    ref = PureDAQP.solve(P, q, Matrix(A), l, u, ActiveSet())
    @test sol.x ≈ ref.x atol = 1.0e-8

    # Replacing `A` keeps its representation: another join of the same block types is taken,
    # a matrix of the same numbers is refused.
    C2 = randn(12, 4)
    A2 = joined(K1, K2, C2)
    l2, u2 = bounds(A2)
    PureQPBase.update!(ws; A = A2, l = l2, u = u2)
    sol2 = PureQPBase.solve!(ws)
    @test sol2.status == SOLVED
    @test sol2.x ≈ PureDAQP.solve(P, q, Matrix(A2), l2, u2, ActiveSet()).x atol = 1.0e-8
    @test_throws "must keep the representation" PureQPBase.update!(ws; A = Matrix(A))

    # A stack of joins, which is what an `hvcat` of maps becomes, carries a two-level type and
    # is updated the same way.
    grid(k1, k2, c, d) = PureQPBase.StackedOperator(joined(k1, k2, c), PureQPBase.JoinedOperator(d, c))
    D = randn(12, 6)
    G = grid(K1, K2, C, D)
    lg, ug = bounds(G)
    wsg = PureDAQP.setup(P, q, G, lg, ug, ActiveSet())
    @test PureQPBase.solve!(wsg).status == SOLVED
    G2 = grid(K1, K2, C2, D)
    lg2, ug2 = bounds(G2)
    PureQPBase.update!(wsg; A = G2, l = lg2, u = ug2)
    solg = PureQPBase.solve!(wsg)
    @test solg.status == SOLVED
    @test solg.x ≈ PureDAQP.solve(P, q, Matrix(G2), lg2, ug2, ActiveSet()).x atol = 1.0e-8
    @test_throws "must keep the representation" PureQPBase.update!(wsg; A = Matrix(G))

    # A joined `P` has no factor of its own and is densified for the reduction; it is still
    # held in its representation, so an update must be a join too.
    Pj = PureQPBase.JoinedOperator(
        PureQPBase.StackedOperator(Diagonal(fill(2.0, 6)), zeros(4, 6)),
        PureQPBase.StackedOperator(zeros(6, 4), Matrix(Symmetric(randn(4, 4) * 0.1)) + 2I),
    )
    @test PureQPBase.is_symmetric(Pj)
    wsp = PureDAQP.setup(Pj, q, A, l, u, ActiveSet())
    @test wsp.red.R isa UpperTriangular
    solp = PureQPBase.solve!(wsp)
    @test solp.status == SOLVED
    @test solp.x ≈ PureDAQP.solve(Matrix(Pj), q, Matrix(A), l, u, ActiveSet()).x atol = 1.0e-8
    @test_throws "must keep the representation" PureQPBase.update!(wsp; P = Matrix(Pj))

    # A square stack is densified for the reduction on the same grounds as a join: its blocks
    # are slices of the rows, which a dense factor does not lose.
    Ps = PureQPBase.StackedOperator(Matrix(Pj)[1:5, :], Matrix(Pj)[6:10, :])
    @test PureQPBase.is_symmetric(Ps)
    wss = PureDAQP.setup(Ps, q, A, l, u, ActiveSet())
    @test wss.red.R isa UpperTriangular
    sols = PureQPBase.solve!(wss)
    @test sols.status == SOLVED
    @test sols.x ≈ PureDAQP.solve(Matrix(Ps), q, Matrix(A), l, u, ActiveSet()).x atol = 1.0e-8

    # A block with no entries leaves the stack unreadable, and the refusal says so rather than
    # naming an element type that is not what failed.
    Pop = PureQPBase.StackedOperator(
        Matrix(Pj)[1:5, :], PureQPBase.ProductOperator{Float64}(Matrix(Pj)[6:10, :])
    )
    @test_throws "cannot read this operand" PureDAQP.setup(Pop, q, A, l, u, ActiveSet())
end

@testitem "the proximal-point loop converges on a point far from the origin" begin
    using PureDAQP, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(7)
    n, m = 20, 40
    P = spd_factor(rng, n, 1.0e12)
    A = rect_factor(rng, m, n, 1.0e2)
    q = randn(rng, n)
    b = A * (randn(rng, n) .* 1.0e6)
    l, u = b .- 1.0, b .+ 1.0

    sol = solve(P, q, A, l, u, ActiveSet(eps_prox = 1.0e-4))
    # `eta_prox` is measured against `‖x‖∞`, which is `1.8e6` here. Against one instead, the
    # loop is asked to settle a point of that size to `1.5e-8` absolute -- finer than the pass
    # producing it resolves -- and runs out of passes on a sequence that stopped moving,
    # reporting `MAX_ITER_REACHED` on an answer already good to nine digits. Raising
    # `max_prox` does not reach it: the test is the limit, not the count.
    @test sol.status == SOLVED
    @test norm(sol.x, Inf) > 1.0e5
end

@testitem "the Gram working set never claims more independent rows than there are variables" begin
    using PureQPBase, PureDAQP, LinearAlgebra, Random

    # Five rows within `1e-4` rad of one another in two variables, with tight bounds: the state
    # that puts a row dependent by counting into the working set. Its pivot is formed by
    # cancellation, so reading it decides rank from noise -- the set then answers a singular
    # system as though it were solvable, and reports a point violating a constraint as `SOLVED`.
    function near_parallel(rng)
        base = randn(rng, 2)
        base ./= norm(base)
        rows = Matrix{Float64}(undef, 6, 2)
        for i in 1:5
            th = 1.0e-4 * randn(rng)
            rows[i, 1] = base[1] * cos(th) - base[2] * sin(th)
            rows[i, 2] = base[1] * sin(th) + base[2] * cos(th)
        end
        rows[6, :] = randn(rng, 2)
        l = randn(rng, 6)
        return rows, randn(rng, 2), l, l .+ 1.0e-3 .* rand(rng, 6)
    end

    P = Matrix(1.0I, 2, 2)
    rng = MersenneTwister(4242)
    for _ in 1:400
        A, q, l, u = near_parallel(rng)
        rows = solve(P, q, A, l, u, ActiveSet(working_set = :rows))
        gram = solve(P, q, A, l, u, ActiveSet(working_set = :gram))
        # The two representations factor the same rows, so they answer the same question.
        @test gram.status === rows.status
        # And an answer called `SOLVED` satisfies the constraints it was given.
        if gram.status === PureQPBase.SOLVED
            @test maximum(max.(l .- A * gram.x, A * gram.x .- u)) < 1.0e-6
        end
    end
end

@testitem "more equalities than variables are carried as inequalities however they arrive" begin
    using PureQPBase, PureDAQP, LinearAlgebra, Random

    # The working set holds at most `n` rows and the reset activates every equality, so a problem
    # stating more than `n` of them is capped: each further one is implied by the `n` that span
    # the variables, or contradicts them, and priced as an inequality it says so either way.
    # `update!` rebuilds the classification from new bounds and caps it the same way, where
    # handing the reset more rows than the set holds reached past the factorization's storage.
    Random.seed!(7)
    A = randn(5, 2)
    b = A * randn(2)
    l, u = b .- rand(5), b .+ rand(5)
    q = randn(2)
    P = Matrix(1.0I, 2, 2)

    for ws in (:rows, :gram), neq in (3, 4, 5)
        tightened_l, tightened_u = copy(l), copy(u)
        for j in 1:neq
            tightened_u[j] = tightened_l[j] = b[j]
        end
        # Stated at setup, and reached through `update!`, must agree.
        direct = solve(P, q, A, tightened_l, tightened_u, ActiveSet(working_set = ws))
        w = setup(P, q, A, l, u, ActiveSet(working_set = ws))
        solve!(w)
        update!(w; l = tightened_l, u = tightened_u)
        updated = solve!(w)
        @test direct.status === PureQPBase.SOLVED
        @test updated.status === direct.status
        @test updated.x ≈ direct.x rtol = 1.0e-6
    end
end

@testitem "an equality row implied by the others is dropped, not called infeasible" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # The reset holds every equality row, so a rank-deficient set of them makes the working
    # set singular on the first pass. The direction that dependency opens moves only equality
    # multipliers, which are free in sign and so never block a step, and no row leaves along
    # it. Whether the direction proves anything is the targets' question: it is a proof of
    # infeasibility exactly when the dependency contradicts them. Where it does not, the
    # dependent row is implied by the rows it depends on and the solve carries on without it.
    P, q = Matrix(1.0I, 2, 2), zeros(2)

    for wsk in (:rows, :gram)
        alg = ActiveSet(; working_set = wsk)
        # A duplicated row, and one scaled by two, which the reduction normalizes to the same
        # row from a different row norm.
        for (A, b) in (
                ([1.0 0.0; 1.0 0.0; 0.0 1.0], [1.0, 1.0, 2.0]),
                ([1.0 0.0; 2.0 0.0; 0.0 1.0], [1.0, 2.0, 2.0]),
            )
            implied = PureDAQP.solve(P, q, A, b, b, alg)
            @test implied.status === PureQPBase.SOLVED
            @test implied.x ≈ [1.0, 2.0]

            # The same dependency against targets it contradicts is infeasible, and provably
            # so: the certificate is checked against the caller's own rows.
            contradicted = PureDAQP.solve(P, q, A, [b[1], b[2] + 1.0, b[3]], [b[1], b[2] + 1.0, b[3]], alg)
            @test contradicted.status === PureQPBase.PRIMAL_INFEASIBLE

            # A re-solve resets the working set, so it meets the dependency again every time.
            w = setup(P, q, A, b, b, alg)
            for _ in 1:3
                sol = solve!(w)
                @test sol.status === PureQPBase.SOLVED
                @test sol.x ≈ [1.0, 2.0]
            end
        end
    end
end

@testitem "an unbounded problem under the proximal loop is reported dual infeasible" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # `min ½x₁² − x₂` subject to `x₁ + x₂ ≥ 1`: unbounded along `x₂`. `P` is singular, so the
    # method runs only with `eps_prox > 0`, whose passes then move `x₂` by the same step for as
    # long as they run and never settle.
    P, q, A = [1.0 0.0; 0.0 0.0], [0.0, -1.0], reshape([1.0, 1.0], 1, 2)
    l, u = [1.0], [Inf]
    # Curvature `0.25` in `x₁`, with a cost term that moves it: each pass closes `1/5` of the
    # distance to its minimizer at `eps_prox = 1`, so after 100 passes it still steps by `1e-10`
    # of the step in `x₂`, not zero but below the part of the step that is read as rounding. The
    # direction left once that part is dropped is exact.
    for wsk in (:rows, :gram)
        sol = solve([0.25 0.0; 0.0 0.0], [1.0, -1.0], A, l, u, ActiveSet(; working_set = wsk, eps_prox = 1.0))
        @test sol.status === PureQPBase.DUAL_INFEASIBLE
        @test sol.dual_inf_cert ≈ [0.0, 1.0] atol = 1.0e-8
    end
    for wsk in (:rows, :gram), eps_prox in (1.0e-4, 1.0e-2)
        alg = ActiveSet(; working_set = wsk, eps_prox)
        sol = solve(P, q, A, l, u, alg)
        @test sol.status === PureQPBase.DUAL_INFEASIBLE
        @test !has_solution(sol.status)
        @test sol.obj_val == -Inf
        # The certificate is a unit direction that descends, leaves the cost flat and keeps
        # the constraint satisfied.
        d = sol.dual_inf_cert
        @test norm(d, Inf) ≈ 1
        @test dot(q, d) < 0
        @test norm(P * d, Inf) < 1.0e-6
        @test (A * d)[1] > -1.0e-6
        # The same through a linear program, which has no curvature to leave in the step.
        lp = solve(zeros(2, 2), q, A, l, u, alg)
        @test lp.status === PureQPBase.DUAL_INFEASIBLE
    end

    # A singular `P` whose problem is bounded is still solved: `x₁` does not appear in the
    # cost, so the loop settles at once.
    for wsk in (:rows, :gram)
        sol = solve(zeros(2, 2), [0.0, 1.0], [0.0 1.0], [1.0], [Inf], ActiveSet(; working_set = wsk, eps_prox = 1.0e-4))
        @test sol.status === PureQPBase.SOLVED
        @test sol.x[2] ≈ 1.0 atol = 1.0e-6
        @test isempty(sol.dual_inf_cert)
    end
end

@testitem "an unbounded problem whose direction is exact only to rounding is not certified" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # The master problem of a bundle method, from the upstream reference solver's tracker
    # (darnstrom/daqp#222). It is unbounded, and the last proximal step is a direction along
    # which `Pd` is `1e-14` and the rows drift by `1e-9` against a step of `3e4`: a ray to
    # rounding, not exactly. Only an exact ray is called a certificate, so the run ends on its
    # pass limit with the point it reached.
    n = 28
    H = zeros(n, n); H[6:10, 6:10] = I(5)
    f = zeros(n)
    f[[5;21:28]] = [
        151.27919502648433, -4.619938789665843, 0.24968822098988802,
        1.4361270737496605, -0.5251791730307167, 0.5394847038306813,
        -3.454965943031397, -3.8158174674229137, -3.496349942781965,
    ]
    A = zeros(8, n)
    A[1, 1:3] = [1 1 -1]
    A[2, [1, 4]] = [-1 1]
    A[3, [1, 5]] = [-1 1]
    A[4:8, 1] = [
        -5.9713117389231485, 0.09977180033089894, -4.47153905349424,
        6.565797623189027, 1.423514870996998,
    ]
    for i in 1:5
        A[3 + i, [5 + i, 10 + i, 15 + i]] = [1 1 -1]
    end
    A[4:8, 21:28] = [
        1.872567499533699 0.22920204929038507 1.4063026723533851 -3.133662408092967 1.2660702980107932 -2.127779966040754 0.7742710759881403 -0.33878819195539234;
        -1.6098722046254275 -1.8439221129240346 -1.101305444588689 1.3341738537257875 0.6938841878618023 -0.740454917652404 0.4485469222371885 2.6581696253958027;
        2.160377174613471 -1.5011729929172053 -1.4868191564357816 -0.18737943664088103 1.056193347920882 -2.4147534014903056 -1.1447990492040527 2.5144430112792855;
        -1.1424871008774524 2.730150778780663 -2.5380957159480215 1.688711394142188 0.12060013445052191 0.03314777008331937 2.576926220378722 2.0630587340651836;
        -0.4477084158289073 -0.5738442734668117 -0.5322187171431682 -0.20810993815524018 -2.934332901024774 1.1253763749147792 1.5706264550828064 3.033900521223415
    ]
    blower = zeros(n + 8); blower[6:10] .= -Inf; blower[29] = 1
    bupper = fill(Inf, n + 8)
    eqs = [2, 3, 11, 12, 13, 15, 16, 17, 18, 19, 20, 29, 30, 31, 32, 33, 34, 35, 36]
    bupper[eqs] = blower[eqs]
    Afull = vcat(Matrix(1.0I, n, n), A)
    for wsk in (:rows, :gram), eps_prox in (1.0e-4, 1.0e-2)
        sol = solve(H, f, Afull, blower, bupper, ActiveSet(; working_set = wsk, eps_prox))
        @test sol.status === PureQPBase.MAX_ITER_REACHED
        @test all(isfinite, sol.x)
    end
end

@testitem "a row that moves toward a present bound stops a certificate, however small" begin
    using PureQPBase, PureDAQP

    alg = ActiveSet(; eps_prox = 1.0e-4)
    # `min −x₂` subject to `1e-8·x₂ ≤ 1` is bounded, at `x₂ = 1e8`. The row moves by `1e-8` per
    # unit of `x₂` toward its bound, so the step is not a recession direction and the run ends
    # on its pass limit.
    sol = solve(zeros(2, 2), [0.0, -1.0], [0.0 1.0e-8], [-Inf], [1.0], alg)
    @test sol.status === PureQPBase.MAX_ITER_REACHED

    # The same row on a variable the cost does not use leaves `x₂` unconstrained: the row does
    # not move along the step, and the problem is unbounded.
    sol = solve(zeros(2, 2), [0.0, -1.0], [1.0e-8 0.0], [-Inf], [1.0], alg)
    @test sol.status === PureQPBase.DUAL_INFEASIBLE
    @test sol.dual_inf_cert ≈ [0.0, 1.0] atol = 1.0e-6
end

@testitem "a kept solution follows the workspace across a certificate" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # `min ½x₁² − x₂ s.t. x₁ + x₂ ≥ l`: unbounded without the cap below, bounded with it.
    P, q, A = [1.0 0.0; 0.0 0.0], [0.0, -1.0], [1.0 1.0]
    for wsk in (:rows, :gram)
        alg = ActiveSet(; working_set = wsk, eps_prox = 1.0e-4)
        ws = setup(P, q, A, [1.0], [Inf], alg)
        keep = copy(solve!(ws))
        @test keep.status === PureQPBase.DUAL_INFEASIBLE
        @test length(keep.dual_inf_cert) == 2

        # Bounded: the cap `x₁ + x₂ ≤ 3` leaves the certificate empty.
        update!(ws; u = [3.0])
        @test copyto!(keep, solve!(ws)) === keep
        @test keep.status === PureQPBase.SOLVED
        @test isempty(keep.dual_inf_cert)
        @test keep.x ≈ ws.sol.x

        # And back, with the same certificate the workspace reports.
        update!(ws; u = [Inf])
        @test copyto!(keep, solve!(ws)) === keep
        @test keep.status === PureQPBase.DUAL_INFEASIBLE
        @test keep.dual_inf_cert == ws.sol.dual_inf_cert
        @test keep.dual_inf_cert !== ws.sol.dual_inf_cert
        @test norm(keep.dual_inf_cert, Inf) ≈ 1

        # Refilling the workspace's own solution does not change what the workspace reports
        # for the solves after it.
        update!(ws; u = [3.0])
        solve!(ws)
        copyto!(ws.sol, keep)
        @test solve!(ws).status === PureQPBase.SOLVED
        @test isempty(ws.sol.dual_inf_cert)
        update!(ws; u = [Inf])
        @test solve!(ws).status === PureQPBase.DUAL_INFEASIBLE
        @test length(ws.sol.dual_inf_cert) == 2
    end
end

@testitem "a re-solve whose bounds free a row that the working set held is not called infeasible" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # A solve proves the problem infeasible, and the update removes the lower bound of `10`
    # that made it so: the bounds that follow match a fresh solve. The second loop below holds
    # rows at bounds that the update then removes, which would otherwise leave their targets at
    # infinity.
    P, q, A = Matrix(1.0I, 2, 2), [0.0, -1.0], [1.0 1.0; 1.0 1.0]
    for wsk in (:rows, :gram), eps_prox in (0.0, 1.0e-4)
        alg = ActiveSet(; working_set = wsk, eps_prox)
        ws = setup(P, q, A, [10.0, -Inf], [Inf, 1.0], alg)
        @test solve!(ws).status === PureQPBase.PRIMAL_INFEASIBLE
        update!(ws; l = [-Inf, -Inf], u = [3.0, 5.0])
        sol = solve!(ws)
        fresh = solve(P, q, A, [-Inf, -Inf], [3.0, 5.0], alg)
        @test sol.status === PureQPBase.SOLVED
        @test sol.x ≈ fresh.x
    end

    # The same with a held upper bound removed.
    for wsk in (:rows, :gram)
        alg = ActiveSet(; working_set = wsk)
        ws = setup(P, q, A, [-Inf, -Inf], [1.0, 0.5], alg)
        sol = solve!(ws)
        update!(ws; l = [2.0, -Inf], u = [Inf, Inf])
        sol = solve!(ws)
        fresh = solve(P, q, A, [2.0, -Inf], [Inf, Inf], alg)
        @test sol.status === fresh.status
        @test sol.x ≈ fresh.x
    end
end

@testitem "a row's scale does not turn an infinite bound into a finite one" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # The reduction divides each row and its bounds by the row's norm. A bound that stands in for
    # infinity has to stay infinite: divided by a norm of `1e152` it becomes `1e-122`, a bound the
    # data resolves, and the unbounded problem below is then a thin slab with a minimizer.
    for sc in (1.0, 1.0e20, 1.0e60, 1.0e100, 1.0e200), wsk in (:rows, :gram)
        alg = ActiveSet(; working_set = wsk, eps_prox = 1.0e-4)
        # `min ½x₁² − x₂` subject to `sc·(x₁ + x₂) ≥ 1`: unbounded along `x₂`.
        sol = solve([1.0 0.0; 0.0 0.0], [0.0, -1.0], [sc sc], [1.0], [Inf], alg)
        @test sol.status === PureQPBase.DUAL_INFEASIBLE

        # The same row on a problem whose minimizer `x = (0, 1)` satisfies it.
        sol = solve(Matrix(1.0I, 2, 2), [0.0, -1.0], [sc sc], [1.0], [Inf], alg)
        @test sol.status === PureQPBase.SOLVED
        @test sol.x ≈ [0.0, 1.0] atol = 1.0e-8
    end
end

@testitem "a finite bound far above one is a bound, and only the clamped infinity is not" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # `PureQPBase` clamps a bound to `±1e30`, which is where "no bound" starts. A finite bound
    # below that, however large, is a constraint the row carries.
    P, q = Matrix(1.0I, 2, 2), [-3.0, 0.0]
    for wsk in (:rows, :gram)
        alg = ActiveSet(; working_set = wsk)
        # `1e30·x₁ ≤ 1e28` and `1e30·x₁ ≤ 5e29` cap `x₁` at 0.01 and 0.5, short of its minimizer 3.
        for (u, cap) in ((1.0e28, 0.01), (5.0e29, 0.5), (1.0e27, 1.0e-3))
            sol = solve(P, q, [1.0e30 0.0], [-Inf], [u], alg)
            @test sol.status === PureQPBase.SOLVED
            @test sol.x ≈ [cap, 0.0] atol = 1.0e-9
        end
        # The same row with no upper bound does not constrain the minimizer.
        sol = solve(P, q, [1.0e30 0.0], [-Inf], [Inf], alg)
        @test sol.x ≈ [3.0, 0.0] atol = 1.0e-9
        # And a bound at the clamp is no bound, whatever the row's scale.
        sol = solve(P, q, [1.0e100 0.0], [-Inf], [1.0e100], alg)
        @test sol.x ≈ [3.0, 0.0] atol = 1.0e-9
    end
end

@testitem "a re-solve after an infeasibility proof starts from the equality rows" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # Duplicate rows with disjoint bounds are infeasible with both rows in the working set. The
    # next bounds are finite and feasible, and the solve after them matches a fresh one whether
    # it follows `update!`, `warm_start!`, or both.
    P, q, A = Matrix(1.0I, 2, 2), [0.0, -1.0], [1.0 1.0; 1.0 1.0]
    for wsk in (:rows, :gram), eps_prox in (0.0, 1.0e-4), order in (:update, :warm_then_update, :update_then_warm)
        alg = ActiveSet(; working_set = wsk, eps_prox)
        ws = setup(P, q, A, [10.0, -Inf], [Inf, 1.0], alg)
        @test solve!(ws).status === PureQPBase.PRIMAL_INFEASIBLE
        order === :warm_then_update && warm_start!(ws; x = zeros(2))
        update!(ws; l = [2.0, -Inf], u = [Inf, 5.0])
        order === :update_then_warm && warm_start!(ws; x = zeros(2))
        sol = solve!(ws)
        fresh = solve(P, q, A, [2.0, -Inf], [Inf, 5.0], alg)
        @test sol.status === PureQPBase.SOLVED
        @test sol.x ≈ fresh.x
    end
end

@testitem "a bound at the clamp of the opposite side is a bound" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # `PureQPBase` clamps an absent lower bound to `-1e30` and an absent upper bound to `1e30`. A
    # lower bound of `+1e30` or an upper bound of `-1e30` is on the other side of that clamp: it
    # is a constraint, and an answer that is not finite is not one that satisfies it.
    P, q = Matrix(1.0I, 2, 2), [-3.0, 0.0]
    for wsk in (:rows, :gram)
        alg = ActiveSet(; working_set = wsk)
        sol = solve(P, q, [1.0 0.0], [1.0e30], [Inf], alg)
        @test sol.status === PureQPBase.SOLVED
        @test sol.x ≈ [1.0e30, 0.0] rtol = 1.0e-8
        sol = solve(P, [3.0, 0.0], [1.0 0.0], [-Inf], [-1.0e30], alg)
        @test sol.status === PureQPBase.SOLVED
        @test sol.x ≈ [-1.0e30, 0.0] rtol = 1.0e-8
    end
end

@testitem "an infeasibility proof reads a bound as present below the clamp" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # `x₁ ≥ 2` against `x₁ ≤ 1` on two copies of one row, with the bounds of the second at the
    # size of `1e25` and of `1e27`, which the rest of the package reads as infinite beyond `1e26`.
    P, q, A = Matrix(1.0I, 2, 2), [0.0, -1.0], [1.0 0.0; 1.0 0.0]
    for wsk in (:rows, :gram), scale in (1.0e25, 1.0e27)
        alg = ActiveSet(; working_set = wsk)
        sol = solve(P, q, A, [2.0 * scale, -Inf], [Inf, scale], alg)
        @test sol.status === PureQPBase.PRIMAL_INFEASIBLE
    end
end

@testitem "a bounded problem is never reported unbounded, whatever its scale" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # Each problem below has a minimizer, far from the origin or from the cost's own scale. A
    # tolerance cannot tell it from an unbounded one, so only a direction that satisfies the
    # conditions exactly is called a certificate.
    for wsk in (:rows, :gram), eps_prox in (1.0e-6, 1.0e-4, 1.0e-2)
        alg = ActiveSet(; working_set = wsk, eps_prox)
        # Curvature of `1e-9` along `x₂` puts the minimizer at `x₂ = 1e9`.
        sol = solve([1.0 0.0; 0.0 1.0e-9], [0.0, -1.0], zeros(0, 2), Float64[], Float64[], alg)
        @test sol.status !== PureQPBase.DUAL_INFEASIBLE
        # An ordinary problem with a cost scaled down: the minimizer is `(1, 1)`.
        sol = solve(1.0e-9 * Matrix(1.0I, 2, 2), -1.0e-9 * [1.0, 1.0], [1.0 1.0], [-Inf], [Inf], alg)
        @test sol.status !== PureQPBase.DUAL_INFEASIBLE
        # A row tilted `1e-8` toward the ray caps `x₂` at `1e8`.
        sol = solve(zeros(2, 2), [0.0, -1.0], [1.0 0.0; 1.0 1.0e-8], [0.0, -Inf], [Inf, 1.0], alg)
        @test sol.status !== PureQPBase.DUAL_INFEASIBLE
    end
end

@testitem "a feasible problem is not called infeasible on a certificate that proves little" begin
    using PureQPBase, PureDAQP, LinearAlgebra, Random

    # A ray with a small `Aᵀy` and a negative support function proves only that no feasible point
    # has `‖x‖∞` below `−support/‖Aᵀy‖₁`, and `PRIMAL_INFEASIBLE` is reported when that radius is
    # at least `infeasibility_radius`, `primal_tol^(-1/2)` or about `8e3` by default. The first
    # problem is feasible, but only at `‖x‖∞ = 5.2e9`, so `PRIMAL_INFEASIBLE` is correct within
    # the radius, `NUMERICAL_ERROR` says it was not decided, and an answer of `SOLVED` must be a
    # feasible point. The second is feasible at a point of size one.
    P = [4.720969871801467 0.005417562178060769 1.67127981408794 0.1766169696594515; 0.005417562178060769 0.5804450153849753 -0.6541583893473041 -2.5916361450093013; 1.67127981408794 -0.6541583893473041 1.8371672644054773 4.53789763127223; 0.1766169696594515 -2.5916361450093013 4.53789763127223 17.778414401999722]
    q = [-0.11926406748778533, -1.371592821149857, 0.10070548173207818, 0.8655579997426182]
    A = [-15.228534303437053 -1.8450556238586897 1.1490303002885323 18.146895143764347; -1.3617195880150195 4.7710540938818236 -7.327336917644794 14.147902977478212; -1.156156957170025 -0.028379964292323506 -0.47382868960963714 0.6259886746162785; -0.5584602424093701 -0.21574884434747513 0.3518883248464013 1.9523748030681376; -13.979393329765673 8.843858939959484 56.34705276691109 -12.634475474206102; 0.043562106988349746 0.0010693107308317937 0.01785309160916039 -0.023586231478360885; 0.018292187497792867 -0.01670777726958397 0.012525311950601166 0.019169917420363187; -0.03799499800261318 -0.014678532778696303 0.02394082043636717 0.1328303628190553]
    l = [41.20387074443814, -2.859117756107935, -2.1742879456464013, 6.658229497717323, -Inf, -0.1012363999528594, -Inf, -Inf]
    u = [65.97071927691137, Inf, Inf, Inf, Inf, 0.03865704139647609, Inf, -0.13680507186291063]
    for wsk in (:rows, :gram)
        sol = solve(P, q, A, l, u, ActiveSet(; working_set = wsk))
        @test sol.status in (PureQPBase.SOLVED, PureQPBase.PRIMAL_INFEASIBLE, PureQPBase.NUMERICAL_ERROR)
        if sol.status === PureQPBase.SOLVED
            Ax = A * sol.x
            @test all(Ax .<= u .+ 1.0e-4 .* (1 .+ abs.(Ax))) && all(Ax .>= l .- 1.0e-4 .* (1 .+ abs.(Ax)))
        end
    end

    # The construction of the pivot tests with the third equality row at `σ = 1e-8` from the span
    # of the first two, below `zero_tol`: feasible by construction, and a row the method treats
    # as dependent.
    for wsk in (:rows, :gram)
        wrong = 0
        for seed in 1:100
            rng = MersenneTwister(seed)
            n = 8
            a1 = normalize(randn(rng, n))
            a2 = normalize(randn(rng, n))
            Q = Matrix(qr(hcat(a1, a2)).Q)[:, 1:2]
            w = randn(rng, n)
            w -= Q * (Q' * w)
            w = normalize(w)
            a3 = normalize(randn(rng) * a1 + randn(rng) * a2) + 1.0e-8 * w
            A = vcat(Matrix(hcat(a1, a2, a3)'), randn(rng, 6, n))
            b = A * randn(rng, n)
            l, u = copy(b), copy(b)
            l[4:end] .-= 1
            u[4:end] .+= 1
            sol = solve(Matrix(1.0I, n, n), randn(rng, n), A, l, u, ActiveSet(; working_set = wsk))
            wrong += sol.status === PureQPBase.PRIMAL_INFEASIBLE
        end
        @test wrong == 0
    end
end

@testitem "a SOLVED answer satisfies the rows it was solved against" begin
    using PureQPBase, PureDAQP

    # A random problem with seven rows in six variables, feasible only at a point far from the
    # origin. Both working sets end with the optimality test passed and a point that violates a
    # row by far more than the tolerance, because the multipliers solve a system whose pivots are
    # rounding.
    P = [11.0 1.0 0.0 10.0 2.0 1.0; 1.0 12.0 4.0 3.0 -4.0 5.0; 0.0 4.0 8.0 4.0 -3.0 -5.0; 10.0 3.0 4.0 15.0 0.0 0.0; 2.0 -4.0 -3.0 0.0 8.0 -4.0; 1.0 5.0 -5.0 0.0 -4.0 18.0]
    q = [-2.0, 0.0, 3.0, -2.0, -2.0, -2.0]
    A = [0.0 -9.5367431640625e-7 2.86102294921875e-6 -1.0 1.0 -1.9073486328125e-6; -2.097152e6 -2.097152e6 3.145728e6 1.099511627776e12 0.0 -1.048575e6; 0.0 -3.0 3.0 -3.145728e6 -3.145728e6 -2.0; 0.0 2.097152e6 -3.145728e6 2.199023255552e12 3.298534883328e12 1.048576e6; -1.0 -2.0 -1.0 -3.145728e6 -1.048576e6 -2.0; 4.0 6.0 2.0 -4.194304e6 4.194304e6 4.0; 2.0 3.0 1.0 -2.097151e6 2.097152e6 2.0]
    l = [3.000000814693451, -3.298532414689616e12, 9.427750812e6, -Inf, -Inf, 1.2582908e7, 6.291451e6]
    u = [Inf, -3.298525817631384e12, Inf, -6.597066839598563e12, 1.887437e7, 1.2582908e7, 6.291451e6]
    for wsk in (:rows, :gram)
        sol = solve(P, q, A, l, u, ActiveSet(; working_set = wsk))
        @test sol.status !== PureQPBase.SOLVED || sol.prim_res < 1.0e-6 * (1 + maximum(abs, A * sol.x))
    end
end

@testitem "a certificate computed from rounding proves nothing" begin
    using PureQPBase, PureDAQP

    # Feasible, at a point of size 1.5, with three equality rows. The ray the loop ends on has a
    # support function of `-5.6e-17`, one unit in the last place of the terms that cancel in it,
    # and an `Aᵀy` of `1e-22`: a ratio that holds for rounding as well as for a proof.
    P = [20.0 -3.0 -2.0 4.0 1.0 9.0 -7.0 -5.0 -7.0 13.0; -3.0 18.0 3.0 -5.0 -5.0 -2.0 -4.0 2.0 1.0 4.0; -2.0 3.0 26.0 -2.0 4.0 -6.0 14.0 -2.0 3.0 2.0; 4.0 -5.0 -2.0 14.0 -2.0 0.0 -10.0 -2.0 -7.0 0.0; 1.0 -5.0 4.0 -2.0 34.0 -5.0 17.0 -4.0 -1.0 -8.0; 9.0 -2.0 -6.0 0.0 -5.0 11.0 -3.0 -5.0 0.0 11.0; -7.0 -4.0 14.0 -10.0 17.0 -3.0 32.0 -5.0 5.0 -7.0; -5.0 2.0 -2.0 -2.0 -4.0 -5.0 -5.0 12.0 -3.0 -4.0; -7.0 1.0 3.0 -7.0 -1.0 0.0 5.0 -3.0 18.0 4.0; 13.0 4.0 2.0 0.0 -8.0 11.0 -7.0 -4.0 4.0 29.0]
    q = [3.0, 2.0, -3.0, 1.0, 2.0, 1.0, 1.0, -3.0, -2.0, -1.0]
    A = [-9.0 -2.86102294921875e-6 0.0 6.291456e6 -3.145728e6 6.291456e6 -3.0 9.0 -8.58306884765625e-6 -9.437184e6; 0.0 1.9073486328125e-6 4.0 -6.291456e6 6.291456e6 2.097152e6 6.0 4.0 -1.9073486328125e-6 2.097152e6; -1.9073486328125e-6 0.0 0.0 3.0 -1.0 0.0 -1.9073486328125e-6 2.86102294921875e-6 -1.8189894035458565e-12 -1.0; 0.0 -1.0 -2.097152e6 0.0 2.199023255552e12 2.199023255552e12 3.145728e6 0.0 2.0 -3.298534883328e12; -1.0 9.5367431640625e-7 0.0 -2.097152e6 3.145728e6 3.145728e6 1.0 -1.0 -9.5367431640625e-7 -2.097152e6; 9.5367431640625e-7 9.094947017729282e-13 9.5367431640625e-7 3.0 1.0 2.0 2.86102294921875e-6 -9.5367431640625e-7 0.0 0.0; 9.5367431640625e-7 -9.094947017729282e-13 0.0 2.0 -3.0 -2.9999999999990905 -9.5367431640625e-7 9.5367431640625e-7 9.094947017729282e-13 2.0; -3.0 0.0 2.0 -1.048576e6 2.097152e6 3.145728e6 2.0 5.0 -3.814697265625e-6 -2.097152e6; -5.7220458984375e-6 -3.637978807091713e-12 0.0 -4.0 2.0 -2.0 -3.814697265625e-6 -3.814697265625e-6 3.637978807091713e-12 -2.0; -2.0 -1.9073486328125e-6 -2.0 4.194304e6 -2.097152e6 -5.24288e6 -4.0 -2.0 0.0 5.24288e6; -1.9073486328125e-6 0.0 9.5367431640625e-7 3.0 -3.0 3.0 9.5367431640625e-7 -2.86102294921875e-6 1.8189894035458565e-12 -3.0; 0.0 -1.9073486328125e-6 -2.0 1.048576e6 1.048576e6 -2.097152e6 -1.0 -2.0 -2.86102294921875e-6 2.097152e6; 2.097152e6 2.0 0.0 1.099511627776e12 1.099511627776e12 3.298534883328e12 2.097152e6 3.145728e6 1.0 -1.099511627776e12; -1.9073486328125e-6 -1.8189894035458565e-12 0.0 4.0 -4.0 -1.0 -1.9073486328125e-6 3.814697265625e-6 -1.8189894035458565e-12 -1.0; -3.0 1.9073486328125e-6 4.0 4.194304e6 -2.097152e6 4.194304e6 2.0 -5.0 3.814697265625e-6 -2.097152e6; -2.8610229483305716e-6 -9.094947017729282e-13 0.0 2.0 -1.0 2.0 -9.5367431640625e-7 2.86102294921875e-6 -2.7284841053187847e-12 -3.0; 1.048576e6 -1.0 0.0 2.199023255552e12 -3.298533834752e12 -3.298534883328e12 -1.048576e6 1.048576e6 1.0 2.199023255552e12; 0.0 3.814697265625e-6 4.0 -2.097152e6 -2.097152e6 4.194304e6 2.0 4.0 5.7220458984375e-6 -4.194304e6; 3.0 -2.86102294921875e-6 0.0 6.291456e6 -9.437184e6 -9.437184e6 -3.0 3.0 2.86102294921875e-6 6.291456e6; -3.0 -1.9073486328125e-6 0.0 -2.097152e6 1.0485760009765625e6 -1.048576e6 -2.0 -2.0 1.9073486328125e-6 -1.048576e6; -3.145728e6 -2.0 0.0 -2.199023255552e12 1.099511627776e12 -1.099511627776e12 -2.097152e6 -2.097152e6 2.0 -1.099511627776e12]
    l = [2.1998060169745713e7, -6.815749000004768e6, 0.0, 6.0473144770595e12, -2.200981235487733e6, 3.2499976888527238, -1.0e27, 3.9277035187950437e6, -1.0000004768280633, -7.340031008770155e6, 8.24175119090262, -Inf, -Inf, -Inf, 8.388609750000954e6, 6.999987993213482, -3.2985351510759307e12, 6.808923261e6, -Inf, -524291.0242665587, -5.49756076027e11]
    u = [2.2042100330245703e7, -6.815739621255906e6, 7.499991416927514, 6.047320524373978e12, 8.492437735482965e6, Inf, 2.0990192751766665, 3.936600981197327e6, -1.0000004768280633, -7.340030991224123e6, Inf, Inf, Inf, 4.005825706687016, 8.388609750000954e6, 7.000001993203469, -3.2985351398630693e12, Inf, -9.42114462153639e6, Inf, -5.49756076027e11]
    for wsk in (:rows, :gram)
        sol = solve(P, q, A, l, u, ActiveSet(; working_set = wsk))
        @test sol.status !== PureQPBase.PRIMAL_INFEASIBLE
    end
end

@testitem "the infeasibility radius is a setting" begin
    using PureQPBase, PureDAQP, LinearAlgebra

    # `x = (0, 1e4)` satisfies both rows, so the problem is feasible. The ray the loop finds
    # proves only that no feasible `x` has `‖x‖∞` below `1e4`: infeasible within the default
    # radius `primal_tol^(-1/2)`, about `8e3`, and not within `1e6`.
    P, q = Matrix(1.0I, 2, 2), [0.0, 0.0]
    A, l, u = [1.0 0.0; 1.0 1.0e-9], [-Inf, 1.0e-5], [0.0, Inf]
    for wsk in (:rows, :gram)
        @test solve(P, q, A, l, u, ActiveSet(; working_set = wsk)).status === PureQPBase.PRIMAL_INFEASIBLE
        sol = solve(P, q, A, l, u, ActiveSet(; working_set = wsk, infeasibility_radius = 1.0e6))
        @test sol.status !== PureQPBase.PRIMAL_INFEASIBLE
    end

    @test PureDAQP.ActiveSet{Float64}(ActiveSet()).infeasibility_radius ≈ 1 / sqrt(sqrt(eps(Float64)))
    @test PureDAQP.ActiveSet{Float32}(ActiveSet()).infeasibility_radius ≈ 1 / sqrt(sqrt(eps(Float32)))
    @test PureDAQP.ActiveSet{Float64}(ActiveSet(; primal_tol = 1.0e-4)).infeasibility_radius ≈ 100
    @test PureDAQP.ActiveSet{Float64}(ActiveSet(; infeasibility_radius = 5.0)).infeasibility_radius == 5.0
    for bad in (0.0, -1.0, Inf, NaN)
        @test_throws "infeasibility_radius must be" ActiveSet(; infeasibility_radius = bad)
    end
end

@testitem "rows of large norm are held to the rounding of their own product" begin
    using PureQPBase, PureDAQP, LinearAlgebra, Random

    # `Ax` is formed with an error of `eps‖aᵢ‖‖x‖`, so a row of norm `1e8` at a point of size one
    # cannot be asked to hold to `1.5e-8` of `1 + |zᵢ|`: an answer that is right to the
    # precision of the data is still an answer. Each problem has `x = 0` feasible, so a row
    # that binds does so at a vertex where `zᵢ` is zero and its rounding is not.
    for (T, s) in ((Float64, 1.0e8), (Float32, 1.0e4)), wsk in (:rows, :gram)
        rng = Xoshiro(1)
        errors = 0
        total = 150
        for _ in 1:total
            A = T.(s .* randn(rng, 8, 6))
            sol = solve(Matrix{T}(I, 6, 6), T.(randn(rng, 6)), A, fill(T(-Inf), 8), zeros(T, 8), ActiveSet(; working_set = wsk))
            errors += sol.status !== PureQPBase.SOLVED
        end
        # The unscaled problems fail at a small, fixed rate whatever the guard does.
        @test errors <= 0.1 * total
    end
end

@testitem "the feasibility check does not move when the cost is scaled" begin
    using PureQPBase, PureDAQP, LinearAlgebra, Random

    # Scaling `P` and `q` by one constant leaves the minimizer where it was and changes the norm of
    # the reduced rows `A R⁻¹` by its square root. The rounding of `Ax` does not depend on the
    # cost, so neither does what the check asks of a row.

    # Seven rows in six variables, feasible only at a point far from the origin: a point that
    # violates a row by the size of the data is not an answer at any scale of the cost.
    P = [11.0 1.0 0.0 10.0 2.0 1.0; 1.0 12.0 4.0 3.0 -4.0 5.0; 0.0 4.0 8.0 4.0 -3.0 -5.0; 10.0 3.0 4.0 15.0 0.0 0.0; 2.0 -4.0 -3.0 0.0 8.0 -4.0; 1.0 5.0 -5.0 0.0 -4.0 18.0]
    q = [-2.0, 0.0, 3.0, -2.0, -2.0, -2.0]
    A = [0.0 -9.5367431640625e-7 2.86102294921875e-6 -1.0 1.0 -1.9073486328125e-6; -2.097152e6 -2.097152e6 3.145728e6 1.099511627776e12 0.0 -1.048575e6; 0.0 -3.0 3.0 -3.145728e6 -3.145728e6 -2.0; 0.0 2.097152e6 -3.145728e6 2.199023255552e12 3.298534883328e12 1.048576e6; -1.0 -2.0 -1.0 -3.145728e6 -1.048576e6 -2.0; 4.0 6.0 2.0 -4.194304e6 4.194304e6 4.0; 2.0 3.0 1.0 -2.097151e6 2.097152e6 2.0]
    l = [3.000000814693451, -3.298532414689616e12, 9.427750812e6, -Inf, -Inf, 1.2582908e7, 6.291451e6]
    u = [Inf, -3.298525817631384e12, Inf, -6.597066839598563e12, 1.887437e7, 1.2582908e7, 6.291451e6]
    for c in (2.0^-80, 2.0^-40, 1.0, 2.0^40), wsk in (:rows, :gram)
        sol = solve(c * P, c * q, A, l, u, ActiveSet(; working_set = wsk))
        @test sol.status !== PureQPBase.SOLVED || sol.prim_res < 1.0e-6 * (1 + maximum(abs, A * sol.x))
    end

    # Rows of norm `1e8` against a cost of `c·I`: the answers are the same as at `c = 1`.
    for c in (2.0^-40, 2.0^40), wsk in (:rows, :gram)
        rng = Xoshiro(1)
        errors = 0
        total = 100
        for _ in 1:total
            A = 1.0e8 .* randn(rng, 8, 6)
            q = randn(rng, 6)
            sol = solve(c * Matrix(1.0I, 6, 6), c * q, A, fill(-Inf, 8), zeros(8), ActiveSet(; working_set = wsk))
            errors += sol.status !== PureQPBase.SOLVED
        end
        @test errors <= 0.1 * total
    end
end

@testitem "a warm re-solve after the bounds move matches a fresh solve" begin
    using PureQPBase, PureDAQP, LinearAlgebra, Random

    # Rows that are multiples of one row `a`, with intervals in `t = aᵀx` that overlap or do not
    # and bounds that move between finite, absent, equal and disjoint: infeasible verdicts are
    # common, and the working set that proves one is the one the next bounds start from.
    function disagreements(wsk, eps_prox)
        count = 0
        for t in 1:1500
            rng = MersenneTwister(t)
            n = rand(rng, 2:5)
            m = rand(rng, 2:5)
            P = Matrix(1.0I, n, n)
            q = randn(rng, n)
            a = randn(rng, n)
            c = rand(rng, m) .* 3 .+ 0.2
            A = c * a'
            function bounds()
                lo = randn(rng, m) .* 2
                hi = lo .+ rand(rng, m) .* 3
                l = c .* lo
                u = c .* hi
                for i in 1:m
                    r = rand(rng)
                    r < 0.25 && (l[i] = -Inf)
                    r > 0.7 && (u[i] = Inf)
                    0.25 <= r < 0.32 && (l[i] = u[i] = c[i] * lo[i])
                end
                return l, u
            end
            alg = ActiveSet(; working_set = wsk, eps_prox)
            l, u = bounds()
            ws = setup(P, q, A, l, u, alg)
            solve!(ws)
            for _ in 1:6
                l, u = bounds()
                update!(ws; l, u)
                warm = solve!(ws)
                warm_status, warm_x = warm.status, copy(warm.x)
                fresh = solve(P, q, A, l, u, alg)
                count += warm_status !== fresh.status
                if warm_status === PureQPBase.SOLVED && fresh.status === PureQPBase.SOLVED
                    count += norm(warm_x - fresh.x, Inf) >= 1.0e-5 * (1 + norm(fresh.x, Inf))
                end
            end
        end
        return count
    end
    for wsk in (:rows, :gram), eps_prox in (0.0, 1.0e-4)
        @test disagreements(wsk, eps_prox) == 0
    end
end
