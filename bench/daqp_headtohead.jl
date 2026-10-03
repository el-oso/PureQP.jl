# PureDAQP against libdaqp, the C implementation of the same method, on random strictly
# convex problems. Writes bench/results/puredaqp_vs_libdaqp.json.
#
# Both solvers are given the same data and timed over setup and solve together, which is what
# a caller pays for a problem it has not seen before; the warm path is measured separately in
# puredaqp_warm_start.json. Agreement is checked before any time is recorded, so a row
# in the output is a comparison of two solvers that found the same answer.
#
#     julia --project=bench bench/daqp_headtohead.jl
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as
# indicative rather than as a gate.

using PureDAQP, PureQPBase, LinearAlgebra, Random, Printf, Statistics
using Chairmarks: @be
import DAQP
import Pkg

BLAS.set_num_threads(1)

"A strictly convex QP whose constraints are all satisfiable, with a known feasible interior."
function random_qp(rng, n, m)
    G = randn(rng, n, n)
    P = Matrix(Symmetric(G' * G + n * I))
    q = randn(rng, n)
    A = randn(rng, m, n)
    b = A * randn(rng, n)
    return P, q, A, b .- rand(rng, m), b .+ rand(rng, m)
end

function dep_version(name)
    for (_, p) in Pkg.dependencies()
        p.name == name && return string(p.version)
    end
    return "not installed"
end

# A grid over both dimensions rather than a line through them. How many rows are active at the
# solution -- and so how many iterations the method takes -- follows the ratio `m / n` more
# than it follows either alone, so a sweep along `m = 2n` cannot show where one solver
# overtakes the other.
const NS = [25, 50, 100, 200, 400]
const RATIOS = [0.5, 1, 2, 4, 8]
"Cells above this are left out: their cost is dominated by the same effects the smaller ones show."
const MAX_CELL = 400 * 3200

# A dual active-set method takes on the order of `m` iterations, so both solvers are given a
# limit well past what the largest size here needs. Left at their defaults the sweep measures
# how long each takes to give up rather than how long it takes to solve.
const ITER_LIMIT = 200_000
const C_SETTINGS = Dict(:iter_limit => Cint(ITER_LIMIT))

rows = []
rng = MersenneTwister(20260930)
for n in NS, ratio in RATIOS
    m = round(Int, n * ratio)
    (m >= 2 && n * m <= MAX_CELL) || continue
    P, q, A, l, u = random_qp(rng, n, m)
    sense = zeros(Cint, m)

    sol = PureDAQP.solve(P, q, A, l, u, ActiveSet(); max_iter = ITER_LIMIT)
    xc, _, exitflag, _ = DAQP.quadprog(P, q, A, u, l, sense; settings = C_SETTINGS)
    sol.status == PureQPBase.SOLVED || error("PureDAQP did not solve n=$n m=$m: $(sol.status)")
    exitflag > 0 || error("libdaqp did not solve n=$n m=$m: exitflag $exitflag")
    rel = norm(sol.x - xc) / max(norm(xc), one(eltype(xc)))
    rel < 1.0e-6 || error("solvers disagree at n=$n m=$m: relative difference $rel")

    bp = @be PureDAQP.solve($P, $q, $A, $l, $u, ActiveSet(); max_iter = ITER_LIMIT) seconds = 2
    bc = @be DAQP.quadprog($P, $q, $A, $u, $l, $sense; settings = $C_SETTINGS) seconds = 2
    pure_ms = 1000 * median(bp).time
    c_ms = 1000 * median(bc).time

    push!(
        rows, (
            n = n, m = m, ratio = ratio, iters = sol.iter,
            puredaqp_ms = pure_ms, libdaqp_ms = c_ms,
            speedup = c_ms / pure_ms,
            rel_err_vs_libdaqp = rel,
            alloc_bytes = (@allocated PureDAQP.solve(P, q, A, l, u, ActiveSet(); max_iter = ITER_LIMIT)),
        )
    )
    @printf(
        "n=%4d m=%4d  PureDAQP %8.3f ms   libdaqp %8.3f ms   %5.2fx   rel %.1e   iters %d\n",
        n, m, pure_ms, c_ms, c_ms / pure_ms, rel, sol.iter
    )
    flush(stdout)
end

# Written by hand rather than through a JSON package: bench carries no JSON dependency, and
# the shape here is flat.
open(joinpath(@__DIR__, "results", "puredaqp_vs_libdaqp.json"), "w") do io
    println(io, "{")
    println(io, "  \"host\": \"", gethostname(), "\",")
    println(io, "  \"julia\": \"", VERSION, "\",")
    println(io, "  \"blas_threads\": 1,")
    println(io, "  \"tool\": \"Chairmarks, seconds=2, median reported\",")
    println(io, "  \"daqp_jl\": \"", dep_version("DAQP"), "\",")
    println(io, "  \"daqp_jll\": \"", dep_version("DAQP_jll"), "\",")
    println(io, "  \"puredaqp\": \"", dep_version("PureDAQP"), "\",")
    println(io, "  \"measures\": \"setup and solve together, cold\",")
    println(io, "  \"note\": \"clock unpinned on this host; indicative, not a gate\",")
    println(io, "  \"rows\": [")
    for (i, r) in enumerate(rows)
        @printf(
            io,
            "    {\"n\": %d, \"m\": %d, \"m_over_n\": %g, \"iters\": %d, \"puredaqp_ms\": %.6f, \"libdaqp_ms\": %.6f, \"speedup\": %.6f, \"rel_err_vs_libdaqp\": %.3e, \"alloc_bytes\": %d}%s\n",
            r.n, r.m, r.ratio, r.iters, r.puredaqp_ms, r.libdaqp_ms, r.speedup,
            r.rel_err_vs_libdaqp, r.alloc_bytes, i == length(rows) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end

# The grid, as a grid. Above 1.00 PureDAQP is the faster of the two.
println("\nlibdaqp / PureDAQP, by n and m/n\n")
@printf("%6s", "n \\ m/n")
for r in RATIOS
    @printf("%10s", string(r))
end
println()
for n in NS
    @printf("%6d", n)
    for ratio in RATIOS
        i = findfirst(r -> r.n == n && r.ratio == ratio, rows)
        isnothing(i) ? @printf("%10s", "-") : @printf("%9.2fx", rows[i].speedup)
    end
    println()
end
println("\nwrote bench/results/puredaqp_vs_libdaqp.json")
