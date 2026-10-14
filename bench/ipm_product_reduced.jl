# What the interior-point method's `ProductReduced` backend costs per factorization, and where
# that cost sits.
#
#     julia --project=bench bench/ipm_product_reduced.jl
#         # writes bench/results/ipm_product_reduced.json
#
# `ProductReduced` serves a pair whose entries can be read but whose representation the dense
# terminal declines, which for the interior-point method is its own `indirect_rung`. No `linsys`
# name selects it — `LINSYS_OPTIONS` has none for it — so a pair that reaches it has to be built:
# a stack of joins over Kronecker blocks, which `holds_structure` answers true for.
#
# The backend refactorizes at every outer iteration, so one factorization's cost is the whole
# per-iteration cost of a solve, and it splits three ways:
#
#   the P term    `c·D P D + σI`, which the weights do not enter. Held between iterations, so
#                 `refactor_weights!` starts from it and only `factorize!` rebuilds it.
#   the A term    `D Aᵀ diag(E²ρ) A D`, which they do. A join contracts this block by block:
#                 each diagonal block through its own representation's method, each off-diagonal
#                 block through the column form over the narrower of the pair.
#   the solve     one Cholesky and the inverse the backend stores.
#
# The shapes below mirror a real constraint operator: a stack of two joins, one holding a
# Kronecker product beside a constant block, the other two Kronecker products sharing their row
# factor. `A₁` is narrow in the second join, which is the case the Kronecker contraction runs
# over its first factor's column pairs rather than its second's.
using PureQPBase, PureOSQP, PureIPM
using LinearAlgebra, Random, Printf, JSON, Chairmarks, Statistics
# `Fill` through `PureQPBase`, which holds it already: a constant block is one of the
# representations the base reduces, so this file needs no dependency of its own for it.
using PureQPBase: Fill

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "ipm_product_reduced.json")
const SEED = 2026
const SECONDS = 1.0

"Median of the samples, which is what a per-iteration cost is read from."
med(b) = Statistics.median([x.time for x in b.samples])

"""
    structured_pair(rng, k1, k2, kb) -> (P, A, q, l, u)

A pair reaching `ProductReduced`: a dense `P` and a stack of two joins over Kronecker blocks.

`kb` sets the second join's second factor, so the ratio of rows to columns and the width of the
Kronecker factors both follow it. The constant block is non-zero, since a zero one contributes
nothing and would hide the cross terms the join has to assemble.
"""
function structured_pair(rng, k1, k2, kb)
    j1 = PureQPBase.JoinedOperator(
        PureQPBase.KroneckerOperator(randn(rng, 2k1, k1), randn(rng, k2, k2)),
        Fill(0.25, 2k1 * k2, k1),
    )
    # The two joins must span the same columns, which the first does as `k1*k2 + k1`.
    j2 = PureQPBase.JoinedOperator(
        PureQPBase.KroneckerOperator(randn(rng, 2, 1), randn(rng, kb, k1 * k2)),
        PureQPBase.KroneckerOperator(randn(rng, 2, 1), randn(rng, kb, k1)),
    )
    A = PureQPBase.StackedOperator(j1, j2)
    m, n = size(A)
    X = randn(rng, n, n)
    P = Matrix(Symmetric(X'X ./ n + 8I))
    b = randn(rng, m)
    return P, A, randn(rng, n), b .- rand(rng, m), b .+ rand(rng, m)
end

"One case: the backend reached, the solve, and the factorization split."
function case(name, P, A, q, l, u)
    ws = PureOSQP.setup(P, q, A, l, u, InteriorPoint(); scaling = 0, verbose = false)
    backend = string(nameof(typeof(ws.linsys)))
    solve() = (PureOSQP.cold_start!(ws); PureOSQP.solve!(ws))
    sol = solve()
    solve()
    t_solve = med(@be solve() seconds = SECONDS)
    # `factorize!` rebuilds the weight-independent part and `refactor_weights!` reuses it, which
    # is the difference the interior-point method sees on every iteration but the first.
    fz() = PureQPBase.factorize!(ws.linsys, ws.prob, ws.weights)
    rw() = PureQPBase.refactor_weights!(ws.linsys, ws.prob, ws.weights)
    fz()
    rw()
    t_fz = med(@be fz() seconds = SECONDS)
    t_rw = med(@be rw() seconds = SECONDS)
    m, n = size(A)
    return (;
        name, n, m, backend, iterations = sol.iter, status = string(sol.status),
        rel_kkt_error = sol.rel_kkt_error, solve_seconds = t_solve,
        factorize_seconds = t_fz, refactor_weights_seconds = t_rw,
        hoist_saving_seconds = t_fz - t_rw,
        allocated_per_solve = (@allocated solve()),
    )
end

function main()
    rng = MersenneTwister(SEED)
    cases = Any[]
    for (k1, k2, kb) in ((6, 6, 60), (8, 8, 120), (10, 10, 200))
        P, A, q, l, u = structured_pair(rng, k1, k2, kb)
        push!(cases, case("k1=$k1 k2=$k2 kb=$kb", P, A, q, l, u))
    end

    @printf(
        "%-22s %5s %6s %-16s %5s %10s %12s %12s %9s\n",
        "case", "n", "m", "backend", "iter", "solve ms", "factorize ms", "refactor ms", "alloc B"
    )
    for c in cases
        @printf(
            "%-22s %5d %6d %-16s %5d %10.2f %12.3f %12.3f %9d\n",
            c.name, c.n, c.m, c.backend, c.iterations, 1.0e3c.solve_seconds,
            1.0e3c.factorize_seconds, 1.0e3c.refactor_weights_seconds, c.allocated_per_solve
        )
    end
    # A backend other than `ProductReduced` means the ladder changed and the figures above
    # describe something else, so say so rather than publishing them under this file's name.
    for c in cases
        c.backend == "ProductReduced" ||
            @warn "case reached $(c.backend), not ProductReduced" case = c.name
    end

    mkpath(dirname(RESULTS))
    open(RESULTS, "w") do io
        JSON.json(
            io, (;
                host = gethostname(), julia = string(VERSION),
                blas_threads = BLAS.get_num_threads(), cpu = Sys.CPU_NAME,
                tool = "Chairmarks, seconds=$SECONDS, median reported",
                note = "clock unpinned on neuromancer; indicative, not a gate",
                seed = SEED, cases,
            ); pretty = 2
        )
    end
    println("\nwrote ", RESULTS)
    return nothing
end

main()
