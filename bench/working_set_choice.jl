# The two working-set representations against each other, on the two kinds of problem that
# separate them. Writes bench/results/working_set_choice.json.
#
#     julia --project=bench bench/working_set_choice.jl
#
# `:gram` factors the Gram matrix of the active rows and `:qr` factors the rows themselves.
# They answer the same questions; what differs is what a row entering costs and how finely
# dependence can be decided.
#
# The two take the same path -- the iteration counts below are equal row for row -- so on a
# problem where both reach an answer the difference is per-iteration cost alone, and `:gram`
# wins it. What `:gram` cannot do is decide rank on a reduction whose conditioning has
# outrun a squared pivot, and the second block below is a problem where that decides the
# answer rather than the speed. The degenerate case is there to show what does *not*
# separate them: rows that are exact combinations of others are handled by both.
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as
# indicative rather than as a gate.

using PureDAQP, PureQPBase, LinearAlgebra, Random, Printf, Statistics
using Chairmarks: @be

BLAS.set_num_threads(1)

const ITER_LIMIT = 200_000

"A strictly convex QP with a well-conditioned constraint matrix."
function benign_qp(rng, n, m)
    G = randn(rng, n, n)
    P = Matrix(Symmetric(G' * G + n * I))
    A = randn(rng, m, n)
    b = A * randn(rng, n)
    return P, randn(rng, n), A, b .- rand(rng, m), b .+ rand(rng, m)
end

"""
A feasible QP whose reduction is conditioned past what a Gram matrix can carry.

`A` has singular values spanning `1e-14`, so `cond(A R⁻¹)` reaches about `1e17`; squaring
that puts it past what double precision holds at all. `x0` satisfies every row with room to
spare, so the problem is demonstrably feasible.
"""
function illconditioned_qp(rng, n, m)
    U, _ = qr(randn(rng, n, n))
    V, _ = qr(randn(rng, m, m))
    A = Matrix(V)[:, 1:n] * Diagonal(exp10.(range(0, -14; length = n))) * Matrix(U)'
    W, _ = qr(randn(rng, n, n))
    P = Matrix(Symmetric(W * Diagonal(exp10.(range(0, -8; length = n))) * W'))
    x0 = randn(rng, n)
    b = A * x0
    return P, randn(rng, n), A, b .- 1.0, b .+ 1.0
end

"A feasible QP in which many rows are exact combinations of others."
function degenerate_qp(rng, n, mi, md)
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
    bnd = 0.2 .* rand(rng, mi)
    u = vcat(ones(n), bnd, [0.3 * sum(abs.(C[d, :]) .* bnd) for d in 1:md])
    return H, 20 .* randn(rng, n), A, vcat(-u[1:n], fill(-Inf, m - n)), u
end

function measure(P, q, A, l, u, kind)
    alg = ActiveSet(; working_set = kind)
    s = PureDAQP.solve(P, q, A, l, u, alg; max_iter = ITER_LIMIT)
    r = A * s.x
    viol = all(isfinite, s.x) ? maximum(max.(r .- u, l .- r)) : NaN
    t = median(
        @be PureDAQP.solve($P, $q, $A, $l, $u, $alg; max_iter = ITER_LIMIT) seconds = 3
    ).time
    return (status = string(s.status), iter = s.iter, ms = 1000 * t, viol = viol, obj = s.obj_val)
end

rows = []

println("Well conditioned: the cost of a row entering decides\n")
@printf(
    "%5s %6s | %-10s %8s %6s | %-10s %8s %6s | %s\n",
    "n", "m", ":gram", "ms", "iters", ":qr", "ms", "iters", "gram/qr"
)
rng = MersenneTwister(20260930)
for (n, m) in [(25, 50), (50, 100), (100, 200), (200, 400)]
    P, q, A, l, u = benign_qp(rng, n, m)
    g = measure(P, q, A, l, u, :gram)
    r = measure(P, q, A, l, u, :qr)
    push!(rows, (case = "benign", n = n, m = m, gram = g, qr = r))
    @printf(
        "%5d %6d | %-10s %8.3f %6d | %-10s %8.3f %6d | %.2fx\n",
        n, m, g.status, g.ms, g.iter, r.status, r.ms, r.iter, r.ms / g.ms
    )
    flush(stdout)
end

println("\nWhere the rank decision decides, and where it does not\n")
@printf("%-22s | %-18s %8s | %-18s %8s\n", "case", ":gram", "ms", ":qr", "ms")
hard = [
    ("ill conditioned 30x200", illconditioned_qp(MersenneTwister(5), 30, 200)),
    ("degenerate 30x130", degenerate_qp(MersenneTwister(1), 30, 60, 40)),
]
for (name, (P, q, A, l, u)) in hard
    g = measure(P, q, A, l, u, :gram)
    r = measure(P, q, A, l, u, :qr)
    push!(rows, (case = name, n = size(A, 2), m = size(A, 1), gram = g, qr = r))
    @printf("%-22s | %-18s %8.3f | %-18s %8.3f\n", name, g.status, g.ms, r.status, r.ms)
    @printf("%-22s |   violation %8.2e |   violation %8.2e\n", "", g.viol, r.viol)
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "working_set_choice.json"), "w") do io
    println(io, "{")
    println(io, "  \"host\": \"", gethostname(), "\",")
    println(io, "  \"julia\": \"", VERSION, "\",")
    println(io, "  \"blas_threads\": 1,")
    println(io, "  \"tool\": \"Chairmarks, seconds=3, median reported\",")
    println(io, "  \"note\": \"clock unpinned on this host; indicative, not a gate\",")
    println(io, "  \"rows\": [")
    for (i, r) in enumerate(rows)
        @printf(
            io,
            "    {\"case\": \"%s\", \"n\": %d, \"m\": %d, \"gram\": {\"status\": \"%s\", \"iter\": %d, \"ms\": %.6f, \"violation\": %.3e, \"obj\": %.10g}, \"qr\": {\"status\": \"%s\", \"iter\": %d, \"ms\": %.6f, \"violation\": %.3e, \"obj\": %.10g}}%s\n",
            r.case, r.n, r.m,
            r.gram.status, r.gram.iter, r.gram.ms, r.gram.viol, r.gram.obj,
            r.qr.status, r.qr.iter, r.qr.ms, r.qr.viol, r.qr.obj,
            i == length(rows) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end
println("\nwrote bench/results/working_set_choice.json")
