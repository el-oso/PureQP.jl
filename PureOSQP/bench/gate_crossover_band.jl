# Where the banded backend actually stops beating the dense path, as a function of the
# reduced matrix's bandwidth `b`.
#
# `bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`, so a `BandedMatrix` `P` of bandwidth
# `b` with a `Diagonal` `A` gives a reduced matrix of bandwidth `b` exactly. Each row is one
# problem solved three ways — banded, `linsys = :dense` on the same structured operands, and
# the dense path a caller reaches by passing `Matrix` operands — so the iteration counts
# match and the difference is the backend.
#
# Every bandwidth in the sweep is one the shipped backend accepts: `A` is `Diagonal`, so
# `m = n` and the banded rung's storage limit `2b + 1 <= m + n` admits `b` up to `n - 1`. The
# sweep therefore measures the selection a caller gets rather than one arranged for it, and
# the assertion below fails if a change to the rung stops that being true.
using PureOSQP, BandedMatrices, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = parse(Float64, get(ENV, "GATE_BUDGET", "0.5"))
const SIZES = isempty(ARGS) ? (200, 500, 1000, 2000) : Tuple(parse.(Int, ARGS))
const FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

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
    problem(n, b) -> (P, q, A, l, u)

A box-constrained QP whose reduced matrix has bandwidth exactly `b`: `P` is a symmetric
`BandedMatrix` of half-bandwidth `b`, diagonally dominant so it is positive definite, and
`A` is `Diagonal`, which contributes no bandwidth of its own.
"""
function problem(n, b)
    rng = MersenneTwister(1234 + 7919 * n + b)
    P = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(P.data, 0.0)
    for j in 1:n
        for i in max(1, j - b):(j - 1)
            v = (rand(rng) - 0.5) / (4b)
            P[i, j] = v
            P[j, i] = v
        end
        P[j, j] = 1.0 + rand(rng)
    end
    A = Diagonal(rand(rng, n) .+ 0.5)
    return P, randn(rng, n), A, -rand(rng, n), rand(rng, n)
end

# The banded rung accepts while `4b <= n`, so a wider band has no banded backend to compare the
# dense path against and the sweep stops where the rung's domain does.
band_grid(n) = sort(unique(clamp.([2; round.(Int, collect(FRACTIONS) .* n)], 2, n ÷ 4)))

println("\nBanded backend against the dense path, sweeping the reduced bandwidth.\n")
@printf(
    "%6s %6s %5s | %11s %11s %11s | %7s %7s | %10s %10s %7s | %6s\n",
    "n", "b", "b/n", "banded", ":dense", "dense Matrix",
    "vs :den", "vs Mat", "setup", ":den setup", "vs :den", "iter"
)
println("-"^120)

"Rewrite the results file with everything measured so far. A sweep this long is worth
checkpointing: a run cut short still leaves usable samples behind."
function save(rows)
    return open(joinpath(@__DIR__, "results", "gate_crossover_band.json"), "w") do io
        JSON.print(
            io, Dict(
                "julia_version" => string(VERSION),
                "blas_threads" => BLAS.get_num_threads(),
                "budget_seconds" => BUDGET,
                "cases" => rows,
            ), 2
        )
    end
end

rows = NamedTuple[]
for n in SIZES
    for b in band_grid(n)
        P, q, A, l, u = problem(n, b)
        Pm, Am = Matrix(P), Matrix(A)
        banded = PureOSQP.solve(P, q, A, l, u; OPTS...)
        dense = PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...)
        densem = PureOSQP.solve(Pm, q, Am, l, u; OPTS...)
        # A row whose backend, answers or iteration counts differ is not a comparison of
        # backends.
        PureOSQP.backend_name(PureOSQP.setup(P, q, A, l, u; OPTS...).linsys) === :banded ||
            error("n=$n b=$b did not build the banded backend")
        banded.iter == dense.iter ||
            error("n=$n b=$b: $(banded.iter) vs $(dense.iter) iterations")
        isapprox(banded.x, dense.x; rtol = 1.0e-6) ||
            error("n=$n b=$b: solutions differ")
        sb, sd = abba(
            () -> @be(PureOSQP.solve(P, q, A, l, u; OPTS...), seconds = BUDGET),
            () -> @be(PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...), seconds = BUDGET),
        )
        sm = times(@be(PureOSQP.solve(Pm, q, Am, l, u; OPTS...), seconds = BUDGET))
        # Setup alone isolates the one factorization, which is the cost the gate's
        # `O(n b²)` against `O(n³)` reasoning is about.
        ub, ud = abba(
            () -> @be(PureOSQP.setup(P, q, A, l, u; OPTS...), seconds = BUDGET),
            () -> @be(PureOSQP.setup(P, q, A, l, u; linsys = :dense, OPTS...), seconds = BUDGET),
        )
        tb, td, tm = median(sb), median(sd), median(sm)
        eb, ed = median(ub), median(ud)
        push!(
            rows, (;
                n, b, frac = b / n, iter = banded.iter,
                iter_dense_matrix = densem.iter,
                banded = tb, dense = td, dense_matrix = tm,
                setup_banded = eb, setup_dense = ed,
                banded_samples = sb, dense_samples = sd, dense_matrix_samples = sm,
                setup_banded_samples = ub, setup_dense_samples = ud,
            )
        )
        @printf(
            "%6d %6d %5.2f | %8.2f ms %8.2f ms %8.2f ms | %6.2fx %6.2fx | %7.2f ms %7.2f ms %6.2fx | %6d\n",
            n, b, b / n, 1.0e3tb, 1.0e3td, 1.0e3tm, td / tb, tm / tb,
            1.0e3eb, 1.0e3ed, ed / eb, banded.iter
        )
        flush(stdout)
        save(rows)
    end
end

"""
    crossover(rows, key) -> Union{Float64,Nothing}

The bandwidth fraction at which the banded path first stops being faster than the column
`key`, taken as the midpoint between the last winning `b/n` and the first losing one.
Returns `nothing` when the banded path wins at every bandwidth measured.
"""
function crossover(rows, key, base)
    lost = findfirst(r -> r[key] < r[base], rows)
    isnothing(lost) && return nothing
    return lost == 1 ? rows[1].frac : (rows[lost].frac + rows[lost - 1].frac) / 2
end

println("\nCrossover, as a fraction of n:\n")
for n in SIZES
    rs = filter(r -> r.n == n, rows)
    isempty(rs) && continue
    # A problem that converges in a different number of iterations is not comparable across
    # rows, so the crossover is read off the size's most common iteration count only.
    modal = argmax(c -> count(r -> r.iter == c, rs), unique(r.iter for r in rs))
    rc = filter(r -> r.iter == modal, rs)
    @printf(
        "  n = %5d (iter %3d)   solve vs :dense: %-10s   solve vs dense Matrix: %-10s   setup vs :dense: %s\n",
        n, modal,
        something(crossover(rc, :dense, :banded), "none ≤ 0.8"),
        something(crossover(rc, :dense_matrix, :banded), "none ≤ 0.8"),
        something(crossover(rc, :setup_dense, :setup_banded), "none ≤ 0.8")
    )
end

println("\nsaved PureOSQP/bench/results/gate_crossover_band.json")
