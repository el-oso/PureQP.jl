# What a warm start is worth to the active-set method, on a problem hard enough that a cold
# solve is expensive. Writes bench/results/puredaqp_warm_start.json.
#
#     julia --project=bench bench/warm_start.jl
#
# The method carries its working set between solves, so a re-solve after `update!` begins from
# the previous answer's active rows rather than from nothing. How much that saves depends on
# how far the data moved, which is what the perturbation sweep below measures: the working set
# is a good guess exactly to the extent that the active set has not changed.
#
# Setup is excluded from the warm timings and included in the cold one, which is the honest
# split -- a caller re-solving a sequence pays setup once.
#
# A second problem is measured when a directory of CSVs is present, named by the
# PUREQP_BENCH_PROBLEM environment variable or `~/Documents/claude/problem` by default, with
# P.csv A.csv q.csv lb.csv ub.csv. It is skipped without complaint when absent, so this script
# runs on a fresh checkout.
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as
# indicative rather than as a gate.

using PureDAQP, PureQPBase, LinearAlgebra, Random, Printf, Statistics, DelimitedFiles
using Chairmarks: @be

BLAS.set_num_threads(1)

const ITER_LIMIT = 200_000
const PERTURBATIONS = [1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1]
"Draws per perturbation. The spread across them is the point; one draw does not carry it."
const DRAWS = 8

"""
A dense QP with a deliberately ill-conditioned `P`, so that a cold solve takes many
iterations and the working set is worth carrying.
"""
function hard_qp(rng, n, m)
    W, _ = qr(randn(rng, n, n))
    P = Matrix(Symmetric(W * Diagonal(exp10.(range(0, -5; length = n))) * W'))
    A = randn(rng, m, n)
    b = A * randn(rng, n)
    return P, randn(rng, n), A, b .- rand(rng, m), b .+ rand(rng, m)
end

function read_problem(dir)
    isdir(dir) || return nothing
    files = joinpath.(dir, ["P.csv", "A.csv", "q.csv", "lb.csv", "ub.csv"])
    all(isfile, files) || return nothing
    v(f) = vec(readdlm(f, ',', Float64))
    P = readdlm(files[1], ',', Float64)
    A = readdlm(files[2], ',', Float64)
    # A bound at +-1e40 is how several formats spell an absent one.
    l = [x <= -1.0e40 ? -Inf : x for x in v(files[4])]
    u = [x >= 1.0e40 ? Inf : x for x in v(files[5])]
    return (P, v(files[3]), A, l, u)
end

"""
Time warm steps over several perturbation draws: `update!` with moved data, then `solve!`.

Reported as a spread rather than a single figure. What a warm start saves depends on how much
of the active set survived the move, and at the larger perturbations that varies enough
between draws to reverse the answer: one draw is several times cheaper than a cold solve and
another several times dearer. A median alone hides that, and the tail is the part a caller
sizing a control loop needs.

Each draw builds its own workspace and solves the *unperturbed* problem first, so the timed
part always begins from a working set optimal for the problem before the move. Reusing one
workspace across draws would instead measure a solver re-solving what it had just solved.
"""
function warm_steps(P, q, A, l, u, alg, eps, draws)
    ms = Float64[]
    iters = Int[]
    failed = 0
    for seed in 1:draws
        rng = MersenneTwister(seed)
        qp = q .+ eps .* randn(rng, length(q)) .* abs.(q)
        w = PureDAQP.setup(P, q, A, l, u, alg; max_iter = ITER_LIMIT)
        PureQPBase.solve!(w)
        PureQPBase.update!(w; q = qp)
        t = @elapsed s = PureQPBase.solve!(w)
        # A warm step that does not reach an answer is a result about warm starting rather
        # than an error in the measurement: far enough from where the working set was optimal,
        # the carried set is a bad enough guess that the pass breaks down. `cold_start!` then
        # solves the same data, so the count of these belongs in the report beside the times.
        if s.status == PureQPBase.SOLVED
            push!(ms, 1000t)
            push!(iters, s.iter)
        else
            failed += 1
        end
    end
    isempty(ms) && error("no warm step succeeded at eps = $eps")
    return (ms = ms, iters = iters, failed = failed)
end

function report(name, P, q, A, l, u)
    alg = ActiveSet()
    n, m = size(A, 2), size(A, 1)
    cold = PureDAQP.solve(P, q, A, l, u, alg; max_iter = ITER_LIMIT)
    cold.status == PureQPBase.SOLVED ||
        error("$name: the cold solve did not succeed, got $(cold.status)")
    bc = @be PureDAQP.solve($P, $q, $A, $l, $u, $alg; max_iter = ITER_LIMIT) seconds = 6
    cold_ms = 1000 * median(bc).time

    w = PureDAQP.setup(P, q, A, l, u, alg; max_iter = ITER_LIMIT)
    PureQPBase.solve!(w)
    same_ms = 1000 * median(@be PureQPBase.solve!($w) evals = 1 seconds = 3).time
    resolve_bytes = @allocated PureQPBase.solve!(w)

    println("\n", name, "  (n = ", n, ", m = ", m, ")")
    @printf("  cold solve            %9.3f ms   %5d iterations\n", cold_ms, cold.iter)
    @printf(
        "  warm, same data       %9.3f ms   %5d iterations   %d bytes\n",
        same_ms, w.iter, resolve_bytes
    )
    @printf("  %-18s %9s %9s %9s   %s\n", "warm, q moved", "min", "median", "max", "beat cold")
    steps = []
    for eps in PERTURBATIONS
        st = warm_steps(P, q, A, l, u, alg, eps, DRAWS)
        beat = count(<(cold_ms), st.ms)
        push!(
            steps, (
                eps = eps, min_ms = minimum(st.ms), median_ms = median(st.ms),
                max_ms = maximum(st.ms), min_iter = minimum(st.iters),
                max_iter = maximum(st.iters), beat_cold = beat, failed = st.failed,
            )
        )
        @printf(
            "  %-18.0e %9.3f %9.3f %9.3f   %d of %d   iterations %d - %d%s\n",
            eps, minimum(st.ms), median(st.ms), maximum(st.ms), beat, DRAWS - st.failed,
            minimum(st.iters), maximum(st.iters),
            iszero(st.failed) ? "" : "   $(st.failed) of $DRAWS did not reach an answer warm"
        )
        flush(stdout)
    end
    return (
        name = name, n = n, m = m, cold_ms = cold_ms, cold_iter = cold.iter,
        warm_same_ms = same_ms, warm_same_bytes = resolve_bytes, steps = steps,
    )
end

cases = []
rng = MersenneTwister(20260930)
P, q, A, l, u = hard_qp(rng, 250, 500)
push!(cases, report("generated, ill-conditioned P", P, q, A, l, u))

const PROBLEM_DIR = get(ENV, "PUREQP_BENCH_PROBLEM", joinpath(homedir(), "Documents/claude/problem"))
real = read_problem(PROBLEM_DIR)
if isnothing(real)
    println("\nno CSV problem at ", PROBLEM_DIR, "; skipping that case")
else
    push!(cases, report("CSV problem at " * PROBLEM_DIR, real...))
end

open(joinpath(@__DIR__, "results", "puredaqp_warm_start.json"), "w") do io
    println(io, "{")
    println(io, "  \"host\": \"", gethostname(), "\",")
    println(io, "  \"julia\": \"", VERSION, "\",")
    println(io, "  \"blas_threads\": 1,")
    println(io, "  \"tool\": \"Chairmarks, median reported; workspace rebuilt in the untimed setup phase\",")
    println(io, "  \"measures\": \"cold includes setup; warm excludes it\",")
    println(io, "  \"note\": \"clock unpinned on this host; indicative, not a gate\",")
    println(io, "  \"cases\": [")
    for (i, c) in enumerate(cases)
        @printf(
            io, "    {\"name\": \"%s\", \"n\": %d, \"m\": %d, \"cold_ms\": %.6f, \"cold_iter\": %d, \"warm_same_ms\": %.6f, \"warm_same_bytes\": %d, \"steps\": [",
            c.name, c.n, c.m, c.cold_ms, c.cold_iter, c.warm_same_ms, c.warm_same_bytes
        )
        for (j, s) in enumerate(c.steps)
            @printf(
                io,
                "{\"q_perturbation\": %.0e, \"min_ms\": %.6f, \"median_ms\": %.6f, \"max_ms\": %.6f, \"min_iter\": %d, \"max_iter\": %d, \"beat_cold\": %d, \"draws\": %d, \"failed_warm\": %d}%s",
                s.eps, s.min_ms, s.median_ms, s.max_ms, s.min_iter, s.max_iter,
                s.beat_cold, DRAWS, s.failed, j == length(c.steps) ? "" : ", "
            )
        end
        println(io, "]}", i == length(cases) ? "" : ",")
    end
    println(io, "  ]")
    println(io, "}")
end
println("\nwrote bench/results/puredaqp_warm_start.json")
