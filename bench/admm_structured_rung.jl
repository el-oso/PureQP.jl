# Which of the two backends that work from a structured operand as it stands is faster on that
# operand.
#
#     julia --project=bench bench/admm_structured_rung.jl
#         # writes bench/results/admm_structured_rung.json
#
# An operand answering `holds_structure` is declined by every rung that reads entries directly,
# since flattening it would discard the representation. Below those sit two backends that both
# work from the representation as it stands:
#
#   `ProductReduced`   assembles the `n × n` reduced matrix from `n` products, factorizes it,
#                      and every solve is one `symv` against the stored inverse.
#   `IndirectCG`       holds nothing and pays its conjugate-gradient iterations on every solve.
#
# Both are built here directly, on one problem and one set of weights, rather than asked of the
# selection ladder. The ladder cannot produce both for every operand: `structured_rung` reaches
# `ProductReduced` only for an operand that also answers `worth_assembling`, so for a bare
# `KroneckerOperator` under `OperatorSplitting` the ladder has only conjugate gradients to give.
# Which backend it picks is `select_backend`'s question and the selection tests answer it; this
# file asks only which of the two is faster, which is what that policy has to be decided
# against.
#
# A backend's cost splits in two, so the comparison is a pair of numbers and not one:
#
#   the factorization   `ProductReduced` spends `n` products with `P` and `A` and a Cholesky;
#                       `IndirectCG` only refreshes its Jacobi diagonal.
#   the solve           one `symv` against the stored inverse, or one run of conjugate
#                       gradients to the tolerance the algorithm asks for.
#
# `breakeven` is where the two meet: the number of solves per factorization past which the
# assembled matrix has paid for itself, zero when its factorization is no dearer either. The
# operator-splitting method factorizes when `rho` moves and solves on every iteration, so the
# solve count per factorization is the iteration count between `rho` updates — tens to hundreds
# on the runs recorded in `iterations` here.
#
# The per-solve ratio is not like for like in accuracy: the assembled arm solves the system to
# machine precision and conjugate gradients stop at the tolerance the algorithm asked for, which
# is what the `residual` columns are there to show. A cheaper inexact solve costs outer
# iterations instead, and `iterations` is from the inexact run.
#
# The conjugate-gradient arm is timed in the state a real solve leaves it: the reference run
# below is an `OperatorSplitting` solve with `linsys = :indirect`, and its backend is timed with
# the settings and the residual level that run ended on, which is the tightest level and so the
# most expensive solve it asked for. The assembled arm is built on that run's own problem and
# weights, so both arms solve the same system at the same weights, and `residual` records how
# accurately each one did.
#
# `compress` is the dense size over what the operand stores. Read it against `breakeven`:
# `worth_assembling` declines a representation on that ratio, and these are the numbers that
# decision costs.
using PureQPBase, PureOSQP
using LinearAlgebra, Random, Printf, JSON, Chairmarks, Statistics
# `Krylov` for the products arm: the conjugate-gradient backend lives in a weak dependency and
# `linsys = :indirect` refuses by name without it.
using Krylov
using PureQPBase: Fill

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "admm_structured_rung.json")
const SEED = 2026
const SAMPLES = 5

med(b) = Statistics.median([x.time for x in b.samples])

"What the operand holds, against the dense matrix it stands for."
stored(A::PureQPBase.KroneckerOperator) = length(A.A1) + length(A.A2)
stored(A::PureQPBase.JoinedOperator) = sum(stored, A.blocks)
stored(A::PureQPBase.StackedOperator) = sum(stored, A.blocks)
stored(::PureQPBase.Fill) = 0
stored(A::AbstractMatrix) = length(A)

"A stack of joins over Kronecker blocks, the shape a constraint operator takes in practice."
function stacked_joins(rng, k1, k2, kb)
    j1 = PureQPBase.JoinedOperator(
        PureQPBase.KroneckerOperator(randn(rng, 2k1, k1), randn(rng, k2, k2)),
        Fill(0.25, 2k1 * k2, k1),
    )
    j2 = PureQPBase.JoinedOperator(
        PureQPBase.KroneckerOperator(randn(rng, 2, 1), randn(rng, kb, k1 * k2)),
        PureQPBase.KroneckerOperator(randn(rng, 2, 1), randn(rng, kb, k1)),
    )
    return PureQPBase.StackedOperator(j1, j2)
end

kron_pair(rng, n1, n2) =
    PureQPBase.KroneckerOperator(randn(rng, n1, n1), randn(rng, n2, n2))

"""
    arm(label, ls, prob, wt, bx, bz) -> NamedTuple

One backend's factorization and solve over the given problem and weights, with the residual of
the system it solved so an arm that is cheap because it is inexact says so.
"""
function arm(label, ls, prob, wt, bx, bz)
    n, m = prob.n, prob.m
    x, z = zeros(n), zeros(m)
    factorize() = PureQPBase.factorize!(ls, prob, wt)
    solve() = PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    factorize()
    solve()
    t_fac = med(@be factorize() samples = SAMPLES evals = 1)
    t_solve = med(@be solve() samples = SAMPLES evals = 1)
    # `R x = rhs_x + Aᵀ(w ⊙ rhs_z)` with `R = c·D P D + σI + D Aᵀ E diag(w) E A D`, which is
    # the system every one of these backends solves, formed densely from the equilibrated data.
    D, E, c = prob.D, prob.E, prob.c
    Ascaled = Diagonal(E) * Matrix(prob.A) * Diagonal(D)
    R = c * Diagonal(D) * Matrix(prob.P) * Diagonal(D) + wt.sigma * I +
        Ascaled' * Diagonal(wt.w) * Ascaled
    rhs = bx + Ascaled' * (wt.w .* bz)
    return (;
        backend = string(PureQPBase.backend_name(ls)), label,
        factorize_seconds = t_fac, solve_seconds = t_solve,
        residual = norm(R * x - rhs) / max(norm(rhs), eps()),
    )
end

"""
    case(name, A, rng) -> NamedTuple

Both backends on one operand. `P` is dense and unstructured, so what the rungs above decline
they decline on `A`, and the reference run is what configures the conjugate-gradient backend.
"""
function case(name, A, rng)
    m, n = size(A)
    X = randn(rng, n, n)
    P = Matrix(Symmetric(X'X ./ n + 8I))
    b = A * randn(rng, n)
    q, l, u = randn(rng, n), b .- rand(rng, m), b .+ rand(rng, m)
    # The reference run: conjugate gradients, reached by name, which both configures that
    # backend and records what the operand costs to solve end to end.
    ws = PureOSQP.setup(P, q, A, l, u, OperatorSplitting(); scaling = 0, linsys = :indirect)
    sol = PureOSQP.solve!(ws)
    prob, wt = ws.prob, ws.weights
    # The level the run's last iteration asked of the backend, which is the tightest it used.
    PureQPBase.set_tolerance_level!(ws.linsys, max(ws.scaled_prim_res, ws.scaled_dual_res))
    bx, bz = randn(n), randn(m)
    products = arm("products", ws.linsys, prob, wt, bx, bz)
    assembled = arm(
        "assembled", PureQPBase.ProductReduced(prob.q0, n, m, prob.A), prob, wt, bx, bz
    )
    # Where the assembled matrix has paid for its factorization: zero when that factorization
    # is no dearer than the other arm's either, so it wins from the first solve, and `Inf` when
    # the products arm is no slower per solve and nothing is ever paid back.
    dsolve = products.solve_seconds - assembled.solve_seconds
    dfac = assembled.factorize_seconds - products.factorize_seconds
    breakeven = dsolve <= 0 ? Inf : max(dfac, zero(dfac)) / dsolve
    return (;
        name, n, m, stored = stored(A), compress = m * n / max(stored(A), 1),
        reference_backend = string(PureQPBase.backend_name(ws.linsys)),
        iterations = sol.iter, status = string(sol.status),
        assembled, products,
        gain_per_solve = products.solve_seconds / assembled.solve_seconds, breakeven,
    )
end

function main()
    rng = MersenneTwister(SEED)
    cases = [
        case("stack k=6 kb=60", stacked_joins(rng, 6, 6, 60), rng),
        case("stack k=10 kb=200", stacked_joins(rng, 10, 10, 200), rng),
        case("kron 8x6", kron_pair(rng, 8, 6), rng),
        case("kron 14x12", kron_pair(rng, 14, 12), rng),
        case("kron 20x18", kron_pair(rng, 20, 18), rng),
    ]

    @printf(
        "%-18s %5s %6s %9s | %8s %9s %8s | %8s %9s %8s | %7s %9s\n",
        "operand", "n", "m", "compress",
        "fac ms", "solve µs", "resid", "fac ms", "solve µs", "resid", "per-solve", "breakeven"
    )
    for c in cases
        @printf(
            "%-18s %5d %6d %8.1fx | %8.3f %9.2f %8.1e | %8.3f %9.2f %8.1e | %6.2fx %8.1f\n",
            c.name, c.n, c.m, c.compress,
            1.0e3c.assembled.factorize_seconds, 1.0e6c.assembled.solve_seconds,
            c.assembled.residual,
            1.0e3c.products.factorize_seconds, 1.0e6c.products.solve_seconds,
            c.products.residual,
            c.gain_per_solve, c.breakeven
        )
    end
    for c in cases
        c.assembled.backend == "product_reduced" ||
            @warn "assembled arm is not ProductReduced" c.name c.assembled.backend
        c.products.backend == "indirect" ||
            @warn "products arm is not the conjugate-gradient backend" c.name c.products.backend
        c.status == "SOLVED" || @warn "the reference run did not solve" c.name c.status
        isfinite(c.breakeven) || @warn "the products arm is no slower per solve" c.name
    end

    mkpath(dirname(RESULTS))
    open(RESULTS, "w") do io
        JSON.json(
            io, (;
                host = gethostname(), julia = string(VERSION),
                blas_threads = BLAS.get_num_threads(), cpu = Sys.CPU_NAME,
                tool = "Chairmarks, samples=$SAMPLES, median reported",
                note = "clock unpinned on neuromancer; indicative, not a gate",
                seed = SEED, cases,
            ); pretty = 2
        )
    end
    println("\nwrote ", RESULTS)
    return nothing
end

main()
