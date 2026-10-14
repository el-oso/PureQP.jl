# Where the low-rank backend stops beating the path a caller reaches when `lowrank_rung`
# declines, as a function of the coupling rank `k`.
#
# `A` is `k` dense rows above one unit row per remaining variable and `P` is `Diagonal`, so
# the reduced matrix is a diagonal plus a rank-`k` correction. Each row is one problem solved
# twice on the same operands — once through the correction, once with `linsys = :dense`, which
# is exactly the rung below — so the iteration counts match and the difference is the backend.
#
# Swept at two BLAS thread counts, and the gate is set from the worse of them. `src/` and
# `ext/` pin nothing, so the dense competitor's `symv` uses whatever threads the caller has;
# a limit derived at one thread promises something the deployed solver does not deliver.
# `Krylov` for the rows past the low-rank rung's limit: those reach the conjugate-gradient
# backend, which lives in a weak dependency and refuses by name without it.
using PureOSQP, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random, Krylov

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = parse(Float64, get(ENV, "GATE_BUDGET", "0.5"))
const SIZES = isempty(ARGS) ? (500, 1000, 2000) : Tuple(parse.(Int, ARGS))
const FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.49)
const THREADS = (1, min(8, Sys.CPU_THREADS))

times(x) = [s.time for s in x.samples]

"""
    abba(a, b) -> (ta, tb)

Time `a` and `b` in the order a, b, b, a and return each one's pooled samples, so a drift
that is monotonic over the pair lands on both equally rather than becoming a ratio.
"""
function abba(a, b)
    ta1, tb1 = a(), b()
    tb2, ta2 = b(), a()
    return (vcat(times(ta1), times(ta2)), vcat(times(tb1), times(tb2)))
end

"""
    problem(n, k) -> (P, q, A, l, u)

A box-constrained QP whose reduced matrix is a diagonal plus a rank-`k` correction.
"""
function problem(n, k)
    rng = MersenneTwister(4919 + 7919 * n + k)
    P = Diagonal(rand(rng, n) .+ 0.5)
    A = PureOSQP.RowCoupled(randn(rng, k, n) ./ 4, ones(n - k), collect(1:(n - k)))
    return P, randn(rng, n), A, -rand(rng, n), rand(rng, n)
end

# The low-rank rung accepts while `10k <= n`, so a higher rank has no low-rank backend to
# compare the dense path against and the sweep stops where the rung's domain does.
rank_grid(n) = sort(unique(clamp.(round.(Int, collect(FRACTIONS) .* n), 1, n ÷ 10)))

"Rewrite the results file with everything measured so far, so a run cut short still leaves
usable samples behind."
function save(rows)
    return open(joinpath(@__DIR__, "results", "gate_crossover_lowrank.json"), "w") do io
        JSON.print(
            io, Dict(
                "julia_version" => string(VERSION),
                "cpu_threads" => Sys.CPU_THREADS,
                "budget_seconds" => BUDGET,
                "cases" => rows,
            ), 2
        )
    end
end

println("\nLow-rank backend against `linsys = :dense` on the same operands.\n")
@printf(
    "%6s %6s %5s %4s | %11s %11s %7s | %10s %10s %7s | %6s\n",
    "n", "k", "k/n", "thr", "lowrank", ":dense", "solve×",
    "setup", ":den setup", "setup×", "iter"
)
println("-"^108)

rows = NamedTuple[]
for nthreads in THREADS
    BLAS.set_num_threads(nthreads)
    for n in SIZES
        for k in rank_grid(n)
            P, q, A, l, u = problem(n, k)
            lr = PureOSQP.solve(P, q, A, l, u; OPTS...)
            dn = PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...)
            # A row whose backend, answers or iteration counts differ is not a comparison of
            # backends.
            PureOSQP.backend_name(PureOSQP.setup(P, q, A, l, u; OPTS...).linsys) === :lowrank ||
                error("n=$n k=$k did not build the low-rank backend")
            lr.iter == dn.iter || error("n=$n k=$k: $(lr.iter) vs $(dn.iter) iterations")
            isapprox(lr.x, dn.x; rtol = 1.0e-6) || error("n=$n k=$k: solutions differ")
            sl, sd = abba(
                () -> @be(PureOSQP.solve(P, q, A, l, u; OPTS...), seconds = BUDGET),
                () -> @be(
                    PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...), seconds = BUDGET
                ),
            )
            # Setup alone isolates the one factorization, which is the `O(nk²)` against
            # `O(n³)` half of the gate's reasoning; the solve above is the other half.
            ul, ud = abba(
                () -> @be(PureOSQP.setup(P, q, A, l, u; OPTS...), seconds = BUDGET),
                () -> @be(
                    PureOSQP.setup(P, q, A, l, u; linsys = :dense, OPTS...), seconds = BUDGET
                ),
            )
            tl, td = median(sl), median(sd)
            el, ed = median(ul), median(ud)
            push!(
                rows, (;
                    n, k, frac = k / n, threads = nthreads, iter = lr.iter,
                    lowrank = tl, dense = td, setup_lowrank = el, setup_dense = ed,
                    lowrank_samples = sl, dense_samples = sd,
                    setup_lowrank_samples = ul, setup_dense_samples = ud,
                )
            )
            @printf(
                "%6d %6d %5.3f %4d | %8.2f ms %8.2f ms %6.2fx | %7.2f ms %7.2f ms %6.2fx | %6d\n",
                n, k, k / n, nthreads, 1.0e3tl, 1.0e3td, td / tl,
                1.0e3el, 1.0e3ed, ed / el, lr.iter
            )
            flush(stdout)
            save(rows)
        end
    end
end

"""
    crossover(rows, nthreads) -> Union{Float64,Nothing}

The rank fraction at which the low-rank path first stops being faster than `linsys = :dense`,
taken as the midpoint between the last winning `k/n` and the first losing one. `nothing` when
it wins at every rank measured.
"""
function crossover(rows, nthreads)
    last_win = nothing
    for r in rows
        r.threads == nthreads || continue
        if r.dense / r.lowrank >= 1
            last_win = r.frac
        elseif isnothing(last_win)
            return r.frac / 2
        else
            return (last_win + r.frac) / 2
        end
    end
    return nothing
end

println("\nCrossover of the whole solve, by thread count:")
for nthreads in THREADS
    for n in SIZES
        c = crossover(filter(r -> r.n == n, rows), nthreads)
        @printf(
            "  %2d thread(s), n = %5d: %s\n", nthreads, n,
            isnothing(c) ? "wins at every rank measured" : @sprintf("k/n ≈ %.3f", c)
        )
    end
end
println("\nsaved PureOSQP/bench/results/gate_crossover_lowrank.json")
