@testitem "an ill-conditioned working set does not settle on the wrong active set" begin
    using PureDAQP, LinearAlgebra
    include(joinpath(@__DIR__, "data", "illconditioned_wrong_set.jl"))

    for i in 1:WRONGSET_COUNT
        P, q, A, l, u, obj_ref, x_ref = illconditioned_wrong_set(i)
        sol = solve(P, q, A, l, u, ActiveSet())
        @test has_solution(sol.status)
        # The objective, not the residual: the point these problems admit carries a residual
        # near `1e-6` from the cancellation in `x = R⁻¹(−u − v)`, while the objective is
        # reached to ten digits. A multiplier noise floor that silences the dual sign test
        # instead converges on an active set larger than the solution's, which lands here as
        # a relative objective error of 13 and 4000.
        @test sol.obj_val ≈ obj_ref rtol = 1.0e-7
        # At most `n` rows can be active at once, and a set that overshoots that is what the
        # silenced sign test produces.
        @test count(>(sqrt(eps(Float64))), abs.(sol.y)) <= length(q)
    end
end

@testitem "the active rows are satisfied to rounding level under an ill-conditioned P" begin
    using PureDAQP, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(11)
    for _ in 1:40
        n, m = 20, 20
        P = spd_factor(rng, n, 1.0e8)
        A = rect_factor(rng, m, n, 1.0e4)
        q = randn(rng, n)
        x0 = randn(rng, n)
        b = A * x0
        l, u = b .- 1.0, b .+ 1.0
        sol = solve(P, q, A, l, u, ActiveSet())
        @test sol.status == SOLVED
        # `x = R⁻¹(−u − v)` subtracts two vectors 20 times larger than their difference at
        # this conditioning, so without the correction `refine_primal!` applies the active
        # rows sit about `1e-6` from their bounds. Measured against the size of the point,
        # which is where the cancellation shows up.
        @test sol.prim_res < 1.0e-11 * (1 + norm(A * sol.x, Inf))
    end
end

@testitem "the Gram working set agrees with the rows it factors the Gram matrix of" begin
    using PureDAQP, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(11)
    converged = Bool[]
    for _ in 1:40
        n, m = 20, 20
        P = spd_factor(rng, n, 1.0e8)
        A = rect_factor(rng, m, n, 1.0e4)
        q = randn(rng, n)
        b = A * randn(rng, n)
        l, u = b .- 1.0, b .+ 1.0
        rows = solve(P, q, A, l, u, ActiveSet(working_set = :rows))
        gram = solve(P, q, A, l, u, ActiveSet(working_set = :gram))
        @test rows.status == SOLVED
        # `:gram` may refuse here, or spend its iterations without converging -- it decides
        # rank through a squared quantity and that is what it costs. What it may not do is
        # report a converged answer that differs from the one the same rows give.
        push!(converged, gram.status == SOLVED)
        if gram.status == SOLVED
            @test gram.obj_val ≈ rows.obj_val rtol = 1.0e-6
        end
    end
    # Most of the family does converge, so the comparison above is not vacuous.
    @test count(converged) >= 30
end

@testitem "the iteration limit reports the point the loop reached" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    rng = MersenneTwister(3)
    n, m = 20, 60
    P = spd_factor(rng, n, 1.0e6)
    A = rect_factor(rng, m, n, 1.0e3)
    q = randn(rng, n)
    b = A * randn(rng, n)
    l, u = b .- 1.0, b .+ 1.0

    full = solve(P, q, A, l, u, ActiveSet())
    @test full.status == SOLVED

    # Two iterations do not reach it; the run took 33. `MAX_ITER_REACHED` carries a point by
    # contract, so what comes back is where the loop stopped rather than the zeros `setup`
    # left in the workspace.
    short = solve(P, q, A, l, u, ActiveSet(); max_iter = 2)
    @test short.status == MAX_ITER_REACHED
    @test has_solution(short.status)
    @test !iszero(short.x)
    @test isfinite(short.obj_val)

    # The same on a re-solve, where what the workspace held before is the previous answer.
    ws = PureDAQP.setup(P, q, A, l, u, ActiveSet())
    first_answer = copy(PureQPBase.solve!(ws))
    @test first_answer.status == SOLVED
    PureQPBase.update!(ws; q = q .+ 50.0)
    PureDAQP.update_settings!(ws; max_iter = 1)
    limited = PureQPBase.solve!(ws)
    @test limited.status == MAX_ITER_REACHED
    @test limited.x != first_answer.x
end

@testitem "the Gram working set's two rank tests agree about which rows are dependent" begin
    using PureDAQP, LinearAlgebra

    # `WorkingSetGram` holds `D`, the squared distance of each active row from the span of the
    # rows before it. `add_row!` admits a row whose square clears `eps(T)`, and
    # `first_dependent` must draw the line in the same place: compared against `zero_tol`
    # itself rather than its square, a row counts as dependent once its distance falls below
    # `sqrt(zero_tol)`, which at the default is `1.2e-4` -- so a row the insertion just
    # accepted is declared singular on the next iteration, the step walks a direction that
    # means nothing, and a feasible problem is reported infeasible.
    T = Float64
    zero_tol = sqrt(eps(T))
    @test zero_tol^2 == eps(T)

    # Two unit rows at a known distance: the second one's pivot is that distance squared.
    pair(d) = ([1.0, 0.0], [sqrt(max(0.0, 1 - d^2)), d])
    for d in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-7)
        W = PureDAQP.build_working_set(PureDAQP.WorkingSetGram, T, 2, 3)
        held, entering = pair(d)
        @test PureDAQP.add_row!(W, held)
        @test PureDAQP.add_row!(W, entering)
        @test PureDAQP.nactive(W) == 2
        @test W.F.D[2] > eps(T)
        @test iszero(PureDAQP.first_dependent(W, zero_tol))
    end

    # A row inside the span of the one before it is dependent, and is still caught.
    W = PureDAQP.build_working_set(PureDAQP.WorkingSetGram, T, 2, 3)
    PureDAQP.add_row!(W, [1.0, 0.0])
    PureDAQP.add_row!(W, [1.0, 0.0])
    @test PureDAQP.first_dependent(W, zero_tol) == 2
end

@testitem "a multiplier the sign test rejects can block the step toward it" begin
    using PureDAQP, LinearAlgebra
    using PureDAQP: nactive, activate!, set_targets!, multiplier_noise, musign,
        step_toward_multipliers!, SIDE_UPPER

    # The dual-feasibility test rejects a multiplier whose sign is wrong by more than
    # `multiplier_noise`, which scales with the working set's conditioning and with the
    # multipliers themselves and is far below `zero_tol` on a well-conditioned set. The step
    # that acts on that rejection has to admit the same rows, or the loop has a state it calls
    # dual infeasible and no move out of it, and reports a numerical breakdown on a problem it
    # can solve.
    n, m = 3, 4
    P = Matrix{Float64}(I, n, n)
    q = zeros(n)
    A = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 1.0 1.0]
    l, u = fill(-1.0, m), fill(1.0, m)
    ws = setup(P, q, A, l, u, ActiveSet())
    lw = ws.red.ws
    set_targets!(ws.red, ws.prob.q0)
    activate!(lw, 1, SIDE_UPPER)
    activate!(lw, 2, SIDE_UPPER)
    k = nactive(lw.W)
    @test k == 2

    # A multiplier wrong in sign by this much sits between the two tolerances: the sign test
    # rejects it, and it is below `zero_tol`, so the step has to block on the lower floor.
    delta = 1.0e-9
    for i in 1:k
        lw.mu[i] = 0.0
        lw.mu_star[i] = -musign(lw, lw.active[i]) * delta
    end
    zero_tol = ws.algorithm.zero_tol
    tol = multiplier_noise(lw, view(lw.mu_star, 1:k))
    @test tol < delta < zero_tol

    @test step_toward_multipliers!(lw, zero_tol)
    @test nactive(lw.W) == k - 1
end

@testitem "a dependency that straddles the primal tolerance stops the pass" begin
    using PureDAQP, PureQPBase, LinearAlgebra
    using PureDAQP: first_dependent, null_direction!, dependency_residual, musign, nactive,
        set_targets!, reset_working_set!, deactivate!, working_set_multipliers!, primal_point!,
        price!, nreduced

    # `singular_step!` reads how far a dependency's own target sits from what the rows it
    # depends on imply, and drops the dependent row where that is within `primal_tol`.
    # Pricing reads the same number from the primal point instead, and the two agree only to
    # about `1e-5` relative on an ill-conditioned set. A `primal_tol` between them is a
    # dependency the step calls implied and pricing calls violated, so the row leaves and
    # comes back for as long as the pass runs. No pass may spend its iteration budget there.
    for theta in (1.0e-6, 3.0e-8), wsk in (:rows, :gram)
        m1 = [1.0, 0.0, 0.0]
        m2 = [cos(theta), sin(theta), 0.0]
        m3 = normalize(m1 .- m2)
        A = Matrix(reduce(hcat, (m1, m2, m3))')
        b = A * [0.3, -0.2, 0.1]
        # The third row is dependent on the first two and inconsistent with them by this much.
        b[3] += 2.0e-6
        P, q = Matrix(1.0I, 3, 3), zeros(3)

        # Both readings of that inconsistency, at the state the reset leaves.
        w = setup(P, q, A, b, b, ActiveSet(; working_set = wsk, primal_tol = 1.0e-3))
        lw = w.red.ws
        set_targets!(w.red, w.prob.q0)
        reset_working_set!(w.red)
        singular = first_dependent(lw.W, w.algorithm.zero_tol)
        iszero(singular) && continue
        p = view(lw.p, 1:nactive(lw.W))
        null_direction!(p, lw.W, singular)
        musign(lw, lw.active[singular]) < 0 && (p .= .-p)
        dependent = lw.active[singular]
        from_targets = dependency_residual(lw, p, singular)
        deactivate!(lw, singular)
        working_set_multipliers!(lw)
        primal_point!(lw)
        price!(lw.price, lw.M, lw.u, 1:nreduced(lw.M))
        s = w.red.scale[dependent]
        priced = max(lw.price[dependent] * s - lw.hi[dependent] * s, lw.lo[dependent] * s - lw.price[dependent] * s)

        lo, hi = minmax(priced, from_targets)
        sol = solve(P, q, A, b, b, ActiveSet(; working_set = wsk, primal_tol = (lo + hi) / 2))
        @test sol.status !== PureQPBase.MAX_ITER_REACHED
        @test sol.iter < 10
    end
end

@testitem "the Gram pivot of a nearly dependent row is read from an explicit residual" begin
    using PureDAQP, LinearAlgebra, Random
    using PureDAQP: WorkingSetGram, build_working_set, add_row!

    # Five rows, the last at distance `sigma` from the span of the first four. Its pivot in
    # `Mₐ Mₐᵀ` is `sigma²`, which is formed by cancellation against a diagonal of one and carries
    # an absolute error near `eps`: at `sigma = 3e-8` that error is half the pivot, and a pivot
    # that rounds to `eps` or below is stored as zero although its exact value is above it. The
    # exact value is `‖m − Mₐᵀc‖²`, which the residual computes without the cancellation. A
    # `sigma` below `sqrt(eps)` is a dependent row, so the band here sits just above it.
    n, k = 8, 4
    for sigma in (1.0e-7, 3.0e-8, 2.0e-8)
        worst = 0.0
        for seed in 1:100
            rng = MersenneTwister(seed)
            rows = [normalize(randn(rng, n)) for _ in 1:k]
            # Orthogonal to the span of all the rows, which sequential projections are not.
            Q = Matrix(qr(reduce(hcat, rows)).Q)[:, 1:k]
            w = randn(rng, n)
            w -= Q * (Q' * w)
            w = normalize(w)
            m = normalize(sum(randn(rng) * a for a in rows)) + sigma * w
            W = build_working_set(WorkingSetGram, Float64, n, n)
            for a in rows
                add_row!(W, a)
            end
            add_row!(W, m)

            # The same pivot in extended precision.
            Mb = BigFloat.(reduce(hcat, rows))
            c = (Mb' * Mb) \ (Mb' * BigFloat.(m))
            exact = Float64(sum(abs2, BigFloat.(m) - Mb * c))
            worst = max(worst, abs(W.F.D[k + 1] - exact) / exact)
        end
        @test worst < 1.0e-6
    end
end

@testitem "the Gram working set solves equality rows that are nearly dependent" begin
    using PureDAQP, PureQPBase, LinearAlgebra, Random

    # The third equality row is within `sigma` of the span of the first two, with bounds from a
    # point that satisfies all three, so the problem is feasible. The pivot that decides whether
    # the row is dependent is formed by cancellation, and a pivot that is noise reports a
    # point that violates a row, or a feasible problem infeasible.
    for sigma in (1.0e-7, 3.0e-8, 2.0e-8)
        failures = 0
        for seed in 1:300
            rng = MersenneTwister(seed)
            n = 8
            a1 = normalize(randn(rng, n))
            a2 = normalize(randn(rng, n))
            # Orthogonal to the span of both rows, which sequential projections are not.
            Q = Matrix(qr(hcat(a1, a2)).Q)[:, 1:2]
            w = randn(rng, n)
            w -= Q * (Q' * w)
            w = normalize(w)
            a3 = normalize(randn(rng) * a1 + randn(rng) * a2) + sigma * w
            A = vcat(Matrix(hcat(a1, a2, a3)'), randn(rng, 6, n))
            x0 = randn(rng, n)
            b = A * x0
            l, u = copy(b), copy(b)
            l[4:end] .-= 1
            u[4:end] .+= 1
            sol = solve(Matrix(1.0I, n, n), randn(rng, n), A, l, u, ActiveSet(; working_set = :gram))
            Ax = A * sol.x
            violation = max(maximum(l .- Ax), maximum(Ax .- u))
            failures += !(sol.status === PureQPBase.SOLVED && violation < 1.0e-6)
        end
        @test failures == 0
    end
end
