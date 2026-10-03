# What an unmaterialized `A` costs on each solver's conjugate-gradient and direct paths, against
# the same problem handed over as a `Matrix`. Writes bench/results/unmaterialized_paths.json.
#
#     julia --project=bench bench/unmaterialized_paths.jl
#
# The six problems are defined once, in bench/unmaterialized_problems.jl, which the examples in
# docs/src/examples.md and one test item per algorithm package also read. The test items pin the
# path each case reaches and compare `case_fingerprint` against the hash recorded here, so a table
# built from problems that have since changed is a failing test rather than a stale number.
#
# The operator column holds only its factors; the matrix column is the same problem with
# `Matrix(A)` and no `linsys` named, which is what a caller gets for handing over a matrix and
# letting the solver choose. The backends therefore differ, and that is the comparison: the
# iteration counts are reported alongside so a time difference that is really an iteration-count
# difference is visible rather than hidden.
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as
# indicative rather than as a gate.

using PureQPBase, PureOSQP, PureIPM, PureDAQP, LinearAlgebra, LinearMaps, Krylov, Random, Printf
using Chairmarks: @be

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "unmaterialized_problems.jl"))

"The median of `b` in milliseconds."
ms(b) = minimum(b).time * 1.0e3

"What `ws` reached: a backend name, or the algorithm's own parameter where the choice is not one."
reached(case, alg, ws) = case.algorithm == "PureDAQP" ? alg.working_set : backend_name(ws.linsys)

"""
    dense_counterpart(case) -> (P, q, A, l, u), options

`case` with `A` formed, and the options a caller gets without naming a backend. The interior-point
method's preconditioner is dropped with it: it is built from Kronecker factors, which a formed
matrix no longer has, so the matrix column takes whichever path the solver chooses for a matrix.
"""
function dense_counterpart(case)
    P, q, A, l, u = case_problem(case)
    Pd = P isa PureQPBase.KroneckerOperator ? Matrix(P) : P
    return (Pd, q, Matrix(A), l, u)
end

rows = NamedTuple[]
for case in unmaterialized_cases()
    P, q, A, l, u = case_problem(case)
    m, n = size(A)
    opts = case.options(P, A)
    alg = case.make_alg()

    ws = setup(P, q, A, l, u, alg; opts..., verbose = false)
    sol = solve!(ws)
    sol.status === SOLVED || error("$(case.algorithm) $(case.path) did not solve: $(sol.status)")
    op_setup = ms(@be setup($P, $q, $A, $l, $u, $alg; opts..., verbose = false) seconds = 3)
    op_solve = ms(@be setup($P, $q, $A, $l, $u, $alg; opts..., verbose = false) solve!(_) seconds = 3)
    op_bytes = Base.summarysize(ws)

    Pd, qd, Ad, ld, ud = dense_counterpart(case)
    wsd = setup(Pd, qd, Ad, ld, ud, alg; scaling = 0, verbose = false)
    sold = solve!(wsd)
    den_setup = ms(@be setup($Pd, $qd, $Ad, $ld, $ud, $alg; scaling = 0, verbose = false) seconds = 3)
    den_solve = ms(
        @be setup($Pd, $qd, $Ad, $ld, $ud, $alg; scaling = 0, verbose = false) solve!(_) seconds = 3
    )
    den_bytes = Base.summarysize(wsd)

    push!(
        rows, (;
            case.algorithm, case.path, case.name, n, m,
            backend = string(reached(case, alg, ws)), iters = sol.iter,
            dense_backend = string(reached(case, alg, wsd)), dense_iters = sold.iter,
            dense_status = string(sold.status),
            op_setup, op_solve, op_bytes, den_setup, den_solve, den_bytes,
            setup_ratio = den_setup / op_setup, solve_ratio = den_solve / op_solve,
            bytes_ratio = den_bytes / op_bytes,
            objdiff = abs(sol.obj_val - sold.obj_val) / max(1.0, abs(sol.obj_val)),
            fingerprint = string(case_fingerprint(case), base = 16),
        )
    )
    @printf(
        "%-9s %-14s %4dx%-4d %-15s %3d it  setup %8.3f ms (%5.1fx)  solve %8.3f ms (%5.1fx)  %6.2f MiB (%5.1fx)\n",
        case.algorithm, case.path, m, n, string(reached(case, alg, ws)), sol.iter,
        op_setup, den_setup / op_setup, op_solve, den_solve / op_solve,
        op_bytes / 2^20, den_bytes / op_bytes
    )
    flush(stdout)
end

mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "unmaterialized_paths.json"), "w") do io
    println(io, "{")
    println(io, "  \"paths\": [")
    for (i, r) in enumerate(rows)
        @printf(
            io,
            "    {\"algorithm\": \"%s\", \"path\": \"%s\", \"name\": \"%s\", \"n\": %d, \"m\": %d, \"backend\": \"%s\", \"iters\": %d, \"dense_backend\": \"%s\", \"dense_iters\": %d, \"dense_status\": \"%s\", \"op_setup_ms\": %.6f, \"op_solve_ms\": %.6f, \"op_bytes\": %d, \"dense_setup_ms\": %.6f, \"dense_solve_ms\": %.6f, \"dense_bytes\": %d, \"setup_ratio\": %.4f, \"solve_ratio\": %.4f, \"bytes_ratio\": %.4f, \"objdiff\": %.3e, \"fingerprint\": \"%s\"}%s\n",
            r.algorithm, r.path, r.name, r.n, r.m, r.backend, r.iters,
            r.dense_backend, r.dense_iters, r.dense_status,
            r.op_setup, r.op_solve, r.op_bytes, r.den_setup, r.den_solve, r.den_bytes,
            r.setup_ratio, r.solve_ratio, r.bytes_ratio, r.objdiff, r.fingerprint,
            i == length(rows) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end
println("\nwrote bench/results/unmaterialized_paths.json")
