# What reducing through a composition's parts costs against reaching the same matrix through
# products of the whole composition. Writes bench/results/composition_reduction.json.
#
#     julia --project=bench bench/composition_reduction.jl
#
# Both columns compute `D Aᵀ diag(w) A D` for the same `A` and agree on it; they differ in how.
# The operator column uses the type's own `add_reduced_term!`, the products column wraps the same
# operator in a `ProductOperator`, which has no structure to exploit and recovers each of the `n`
# columns as `Aᵀ(W(A eⱼ))`.
#
# A composition `B E` is where this pays, and what it pays for is the outer part: the products
# column applies `B` and `Bᵀ` once per column, `2n` times, where `Eᵀ(Bᵀ diag(w) B)E` applies it once.
# So the gain tracks how expensive `B` is to apply, which is why both a dense and a Kronecker outer
# part are measured — a Kronecker `B` is already cheap, and there is correspondingly little to save.
#
# A sum has no reduction of its own, and this records why: expanding `(ΣBᵢ)ᵀW(ΣBᵢ)` into its `K`
# diagonal terms and `K(K-1)/2` cross pairs costs `nK(K+1)` products of a term, where a product of
# the sum already sums the terms in one pass and so costs `2nK`. The ratios below are of the
# expansion against the products, measured before the expansion was removed.
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as indicative
# rather than as a gate.

using PureQPBase, LinearAlgebra, Random, Printf
using Chairmarks: @be

BLAS.set_num_threads(1)

ms(b) = minimum(b).time * 1.0e3

"""
    reduction(A, w, D) -> (R, milliseconds)

The reduced matrix `A` gives, and the time its own `add_reduced_term!` takes to build it.

`R` comes from one application into a zeroed matrix, and the timing runs on a separate one:
`add_reduced_term!` *adds* to what it is given, so repeated samples accumulate, and reading the
value back off the benchmarked matrix would read a sum of every sample. How long the call takes
does not depend on what `R` already holds.
"""
function reduction(A, w, D)
    m, n = size(A)
    T = eltype(A)
    scratch = PureQPBase.reduced_term_scratch(T, A)
    ej, av, col = zeros(n), zeros(m), zeros(n)

    R = zeros(T, n, n)
    PureQPBase.add_reduced_term!(R, T, A, w, D, n, m, scratch, ej, av, col)

    sink = zeros(T, n, n)
    t = ms(
        @be PureQPBase.add_reduced_term!(
            $sink, $T, $A, $w, $D, $n, $m, $scratch, $ej, $av, $col
        ) seconds = 3
    )
    return R, t
end

rows = NamedTuple[]
for (kind, m, k, n, build) in (
        ("dense outer", 400, 50, 100, (m, k) -> randn(m, k)),
        ("dense outer", 800, 50, 200, (m, k) -> randn(m, k)),
        ("dense outer", 1600, 80, 200, (m, k) -> randn(m, k)),
        ("dense outer", 3200, 100, 300, (m, k) -> randn(m, k)),
        ("Kronecker outer", 64, 36, 20, (m, k) -> PureQPBase.KroneckerOperator(randn(8, 6), randn(8, 6))),
        ("Kronecker outer", 144, 64, 40, (m, k) -> PureQPBase.KroneckerOperator(randn(12, 8), randn(12, 8))),
        ("Kronecker outer", 256, 100, 60, (m, k) -> PureQPBase.KroneckerOperator(randn(16, 10), randn(16, 10))),
        ("Kronecker outer", 400, 196, 100, (m, k) -> PureQPBase.KroneckerOperator(randn(20, 14), randn(20, 14))),
    )
    Random.seed!(5)
    outer = build(m, k)
    size(outer) == (m, k) || error("$kind build gave $(size(outer)), expected $((m, k))")
    inner = randn(k, n)
    A = PureQPBase.ComposedOperator(outer, inner)
    products = PureQPBase.ProductOperator{Float64}(A)

    w, D = rand(m) .+ 0.5, rand(n) .+ 0.5
    Rp, tp = reduction(A, w, D)
    Rg, tg = reduction(products, w, D)
    dense = Matrix(outer) * inner
    reference = Diagonal(D) * dense' * Diagonal(w) * dense * Diagonal(D)

    push!(
        rows, (;
            kind, m, k, n, parts_ms = tp, products_ms = tg, ratio = tg / tp,
            err = norm(Rp - reference) / norm(reference),
            agree = norm(Rp - Rg) / norm(Rg),
        )
    )
    @printf(
        "%-16s m=%5d k=%4d n=%4d  parts %9.3f ms  products %9.3f ms  %5.2fx   err %.1e\n",
        kind, m, k, n, tp, tg, tg / tp, rows[end].err
    )
    flush(stdout)
end

mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "composition_reduction.json"), "w") do io
    println(io, "{")
    println(io, "  \"composed\": [")
    for (i, r) in enumerate(rows)
        @printf(
            io,
            "    {\"kind\": \"%s\", \"m\": %d, \"k\": %d, \"n\": %d, \"parts_ms\": %.6f, \"products_ms\": %.6f, \"ratio\": %.4f, \"err\": %.3e, \"agree\": %.3e}%s\n",
            r.kind, r.m, r.k, r.n, r.parts_ms, r.products_ms, r.ratio, r.err, r.agree,
            i == length(rows) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end
println("\nwrote bench/results/composition_reduction.json")
