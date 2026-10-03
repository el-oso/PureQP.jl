# What declaring `Symmetric(P)` costs and what it buys, against the same numbers as a `Matrix`.
# Writes bench/results/symmetric_cost.json.
#
#     julia --project=bench bench/symmetric_cost.jl
#
# The two halves answer different questions and only one of them is a throughput question.
#
# `solve!` is where symmetry pays: `mul_P!` multiplies `prob.P` directly, so a `Symmetric`
# reaches BLAS `symv` instead of `gemv` and halves the flops of a product the iteration runs
# every time. The matrix-free backend gains most, because there the product is per
# conjugate-gradient iteration rather than per solve.
#
# `setup` is where it can be squandered. The wrapper's `getindex` branches per entry and reads
# the half outside its triangle at stride `n`, so a kernel that indexes `P` rather than its
# parent is slower on a `Symmetric` than on a matrix carrying the same numbers. The kernels
# read the parent over the triangle `uplo` names; this records that setup is no worse for
# saying `Symmetric`, which is what makes the `solve!` gain reach a caller.
#
# Parity rather than a gain is the honest ceiling for the column norms: they are maxima over
# full columns, so symmetry halves the reads but forces a scatter to each entry's mirror, and
# the store chain costs what the halved reads save. Where only one triangle is ever required --
# `cholesky_factor`, `is_convex`, the KKT cost block -- reading it is faster.
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as
# indicative rather than as a gate.

using PureQPBase, PureOSQP, PureIPM, PureDAQP, LinearAlgebra, Random, Printf
using Chairmarks: @be
using Krylov

BLAS.set_num_threads(1)

"A strictly convex QP with a dense symmetric `P`."
function symmetric_qp(rng, n, m)
    G = randn(rng, n, n)
    P = Matrix(Symmetric(G' * G / n + I))
    A = randn(rng, m, n)
    x = randn(rng, n)
    b = A * x
    return P, randn(rng, n), A, b .- 1, b .+ 1
end

const SIZES = (100, 200, 500, 900)

ms(b) = minimum(b).time * 1.0e3
us(b) = minimum(b).time * 1.0e6

setup_rows = NamedTuple[]
solve_rows = NamedTuple[]
product_rows = NamedTuple[]

println("The product the iteration runs: mul!(y, P, x)\n")
@printf("%6s  %12s  %12s  %7s\n", "n", "gemv (us)", "symv (us)", "ratio")
for n in SIZES
    rng = MersenneTwister(6)
    P, = symmetric_qp(rng, n, max(n ÷ 2, 2))
    x = randn(rng, n)
    y = zeros(n)
    S = Symmetric(P)
    mul!(y, P, x)
    mul!(y, S, x)
    yg = similar(y)
    ysy = similar(y)
    mul!(yg, P, x)
    mul!(ysy, S, x)
    bg = @be mul!($y, $P, $x) seconds = 3
    bs = @be mul!($y, $S, $x) seconds = 3
    push!(
        product_rows,
        (;
            n, gemv_us = us(bg), symv_us = us(bs), ratio = us(bg) / us(bs),
            maxdiff = maximum(abs, yg - ysy),
        ),
    )
    @printf("%6d  %12.2f  %12.2f  %7.2f\n", n, us(bg), us(bs), us(bg) / us(bs))
end

println("\nsolve!, warm, same workspace re-solved\n")
@printf("%6s  %-9s  %12s  %12s  %7s  %9s\n", "n", "backend", "Matrix (ms)", "Symmetric (ms)", "ratio", "iter")
for n in SIZES
    rng = MersenneTwister(12)
    P, q, A, l, u = symmetric_qp(rng, n, max(n ÷ 2, 2))
    for (backend, kw) in (("direct", (;)), ("indirect", (; linsys = :indirect, scaling = 0)))
        wd = PureQPBase.setup(P, q, A, l, u, OperatorSplitting(); kw...)
        wsym = PureQPBase.setup(Symmetric(P), q, A, l, u, OperatorSplitting(); kw...)
        sd = PureQPBase.solve!(wd)
        ssym = PureQPBase.solve!(wsym)
        bd = @be PureQPBase.solve!($wd) seconds = 4
        bs = @be PureQPBase.solve!($wsym) seconds = 4
        push!(
            solve_rows,
            (;
                n, backend, matrix_ms = ms(bd), symmetric_ms = ms(bs),
                ratio = ms(bd) / ms(bs), iter = sd.iter, iter_sym = ssym.iter,
                objdiff = abs(sd.obj_val - ssym.obj_val) / max(abs(sd.obj_val), 1),
            ),
        )
        @printf(
            "%6d  %-9s  %12.3f  %12.3f  %7.2f  %4d/%-4d\n",
            n, backend, ms(bd), ms(bs), ms(bd) / ms(bs), sd.iter, ssym.iter
        )
    end
end

println("\nsetup, which must not squander the gain above\n")
@printf("%6s  %-18s  %12s  %12s  %7s\n", "n", "algorithm", "Matrix (ms)", "Symmetric (ms)", "ratio")
for n in SIZES
    rng = MersenneTwister(8)
    P, q, A, l, u = symmetric_qp(rng, n, max(n ÷ 2, 2))
    S = Symmetric(P)
    for (name, alg) in (("ActiveSet", ActiveSet()), ("OperatorSplitting", OperatorSplitting()), ("InteriorPoint", InteriorPoint()))
        # `ActiveSet` requires unscaled data, and the others are compared at their own default.
        kw = alg isa ActiveSet ? (; scaling = 0) : (;)
        bd = @be PureQPBase.setup($P, $q, $A, $l, $u, $alg; $kw...) seconds = 4
        bs = @be PureQPBase.setup($S, $q, $A, $l, $u, $alg; $kw...) seconds = 4
        push!(
            setup_rows,
            (;
                n, algorithm = name, matrix_ms = ms(bd), symmetric_ms = ms(bs),
                ratio = ms(bd) / ms(bs),
            ),
        )
        @printf("%6d  %-18s  %12.3f  %12.3f  %7.2f\n", n, name, ms(bd), ms(bs), ms(bd) / ms(bs))
    end
end

mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "symmetric_cost.json"), "w") do io
    println(io, "{")
    println(io, "  \"tool\": \"Chairmarks, minimum reported\",")
    println(io, "  \"blas_threads\": 1,")
    println(io, "  \"note\": \"clock unpinned on this host; indicative, not a gate\",")
    println(io, "  \"product\": [")
    for (i, r) in enumerate(product_rows)
        @printf(
            io, "    {\"n\": %d, \"gemv_us\": %.4f, \"symv_us\": %.4f, \"ratio\": %.4f, \"maxdiff\": %.3e}%s\n",
            r.n, r.gemv_us, r.symv_us, r.ratio, r.maxdiff, i == length(product_rows) ? "" : ","
        )
    end
    println(io, "  ],")
    println(io, "  \"solve\": [")
    for (i, r) in enumerate(solve_rows)
        @printf(
            io, "    {\"n\": %d, \"backend\": \"%s\", \"matrix_ms\": %.6f, \"symmetric_ms\": %.6f, \"ratio\": %.4f, \"iter\": %d, \"iter_symmetric\": %d, \"objdiff\": %.3e}%s\n",
            r.n, r.backend, r.matrix_ms, r.symmetric_ms, r.ratio, r.iter, r.iter_sym,
            r.objdiff, i == length(solve_rows) ? "" : ","
        )
    end
    println(io, "  ],")
    println(io, "  \"setup\": [")
    for (i, r) in enumerate(setup_rows)
        @printf(
            io, "    {\"n\": %d, \"algorithm\": \"%s\", \"matrix_ms\": %.6f, \"symmetric_ms\": %.6f, \"ratio\": %.4f}%s\n",
            r.n, r.algorithm, r.matrix_ms, r.symmetric_ms, r.ratio,
            i == length(setup_rows) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end
println("\nwrote bench/results/symmetric_cost.json")
