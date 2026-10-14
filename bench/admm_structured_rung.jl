# What the operator-splitting method gains from assembling the reduced matrix for an operand
# that holds structure, instead of working through products with it.
#
#     julia --project=bench bench/admm_structured_rung.jl
#         # writes bench/results/admm_structured_rung.json
#
# An operand answering `holds_structure` is declined by every rung that reads entries directly,
# since flattening it would discard the representation. Below those sits a choice between two
# backends that both work from the representation as it stands:
#
#   `ProductReduced`   assembles the `n × n` reduced matrix from `n` products, factorizes once,
#                      and every solve is one `symv` against the stored inverse.
#   `IndirectCG`       holds nothing and pays its conjugate-gradient iterations on every solve.
#
# The method factorizes only when `rho` changes and solves on every iteration, so the
# factorization amortizes and the per-solve cost decides. The exception is an operand whose
# representation is much smaller than `n²`: there the `n` products that assemble the matrix, and
# the memory holding it, outweigh what the cheaper solves return. A Kronecker pair is the case
# that reaches furthest in that direction, so the shapes below span both regimes — `compress` is
# the dense size over what the operand stores.
#
# The gain falls as `compress` rises, but only once `n` is large enough for the `n²` matrix to
# cost something: the smallest Kronecker pair below compresses 23 times and still gains, because
# its factorization is tens of microseconds either way. Read the two columns together.
#
# Both arms are timed with the factorization counted. Timing only the iteration phase leaves
# the factorization inside `setup`, which is where this backend spends most of its cost.
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
    arm(A, P, q, l, u; kwargs...) -> NamedTuple

One backend's cost: the factorization, then a cold solve, with the backend reached recorded so
a change in the ladder shows up here rather than being absorbed silently.
"""
function arm(A, P, q, l, u; kwargs...)
    ws = PureOSQP.setup(P, q, A, l, u, OperatorSplitting(); scaling = 0, verbose = false, kwargs...)
    sol = PureOSQP.solve!(ws)
    ls, prob, wt = ws.linsys, ws.prob, ws.weights
    factorize() = PureQPBase.factorize!(ls, prob, wt)
    factorize()
    t_fac = med(@be factorize() samples = SAMPLES evals = 1)
    cold() = (PureOSQP.cold_start!(ws); PureOSQP.solve!(ws))
    cold()
    t_iter = med(@be cold() samples = SAMPLES evals = 1)
    return (;
        backend = string(PureQPBase.backend_name(ls)), iterations = sol.iter,
        status = string(sol.status), factorize_seconds = t_fac, iterate_seconds = t_iter,
        total_seconds = t_fac + t_iter,
    )
end

"Bounds bracketing a point the operator reaches, so the pair is feasible for any seed."
function case(name, A, rng)
    m, n = size(A)
    X = randn(rng, n, n)
    P = Matrix(Symmetric(X'X ./ n + 8I))
    b = A * randn(rng, n)
    q, l, u = randn(rng, n), b .- rand(rng, m), b .+ rand(rng, m)
    assembled = arm(A, P, q, l, u)
    products = arm(A, P, q, l, u; linsys = :indirect)
    return (;
        name, n, m, stored = stored(A), compress = m * n / max(stored(A), 1),
        assembled, products, gain = products.total_seconds / assembled.total_seconds,
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
        "%-18s %5s %6s %9s | %5s %8s %8s | %5s %8s %8s | %6s\n",
        "operand", "n", "m", "compress", "iter", "fac ms", "total", "iter", "fac ms", "total", "gain"
    )
    for c in cases
        @printf(
            "%-18s %5d %6d %8.1fx | %5d %8.3f %8.2f | %5d %8.3f %8.2f | %5.2fx\n",
            c.name, c.n, c.m, c.compress,
            c.assembled.iterations, 1.0e3c.assembled.factorize_seconds,
            1.0e3c.assembled.total_seconds,
            c.products.iterations, 1.0e3c.products.factorize_seconds,
            1.0e3c.products.total_seconds, c.gain
        )
    end
    for c in cases
        c.assembled.backend == "product_reduced" || @warn "assembled arm reached another backend" c.name c.assembled.backend
        c.products.backend == "indirect" || @warn "products arm reached another backend" c.name c.products.backend
        c.assembled.status == "SOLVED" || @warn "case did not solve" c.name c.assembled.status
    end

    mkpath(dirname(RESULTS))
    open(RESULTS, "w") do io
        JSON.json(
            io, (;
                host = gethostname(), julia = string(VERSION),
                blas_threads = BLAS.get_num_threads(), cpu = Sys.CPU_NAME,
                tool = "Chairmarks, samples=$SAMPLES, median reported, factorization included",
                note = "clock unpinned on neuromancer; indicative, not a gate",
                seed = SEED, cases,
            ); pretty = 2
        )
    end
    println("\nwrote ", RESULTS)
    return nothing
end

main()
