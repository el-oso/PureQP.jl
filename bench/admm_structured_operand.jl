# What a structured operand costs the operator-splitting method, against the direct backend the
# same problem reaches as a dense matrix.
#
#     julia --project=bench bench/admm_structured_operand.jl [out.json]
#
# `A` is a stack of joins, the shape a `vcat` of `hcat`s of linear maps arrives as. It reports
# `holds_structure`, so `select_backend` declines the dense terminal and ends at conjugate
# gradients, while the same operator as a dense matrix reaches `cholesky`. Naming
# `linsys = :dense` reaches the factorization anyway, since a stack is materializable, so the
# three columns separate the ladder's choice from what the representation can support:
#
#     dense       `Matrix(A)`, the backend and the accuracy reference
#     forced      the operator with `linsys = :dense`
#     ladder      the operator with the ladder's own choice
#
# BACKLOG.md records why no rung makes the ladder's choice a direct one.
using PureQPBase, PureOSQP, LinearAlgebra, Krylov, Random, Printf, JSON
using PureQPBase: JoinedOperator, StackedOperator, Fill

BLAS.set_num_threads(1)

"A stack of two joins over `n` columns, and the same operator as a dense matrix."
function pair(n, m)
    k = n ÷ 2
    top = JoinedOperator(randn(m ÷ 2, k), Fill(0.5, m ÷ 2, n - k))
    bot = JoinedOperator(Fill(-0.25, m - m ÷ 2, k), randn(m - m ÷ 2, n - k))
    A = StackedOperator(top, bot)
    return A, Matrix(A)
end

function problem(n, m, seed)
    Random.seed!(seed)
    A, Ad = pair(n, m)
    B = randn(n, n)
    return (; P = B' * B / n + I, q = randn(n), A, Ad, l = fill(-1.0, m), u = fill(1.0, m))
end

function run(p, A; kwargs...)
    setup_s = @elapsed ws = setup(p.P, p.q, A, p.l, p.u, OperatorSplitting(); kwargs...)
    solve_s = @elapsed sol = solve!(ws)
    return (;
        backend = string(backend_name(ws.linsys)), status = string(sol.status),
        iterations = sol.iter, setup_s, solve_s, x = sol.x,
    )
end

record(r, err) = Dict(
    "backend" => r.backend, "status" => r.status, "iterations" => r.iterations,
    "setup_s" => r.setup_s, "solve_s" => r.solve_s, "err_vs_dense" => err,
)

rows = []
for (n, m) in ((100, 160), (200, 320), (400, 640))
    p = problem(n, m, 11)
    # Warm each path once: the first call to it is compilation.
    run(p, p.Ad); run(p, p.A; linsys = :dense); run(p, p.A)
    dense = run(p, p.Ad)
    forced = run(p, p.A; linsys = :dense)
    ladder = run(p, p.A)
    err(r) = maximum(abs, r.x .- dense.x)
    push!(
        rows, Dict(
            "n" => n, "m" => m, "dense" => record(dense, 0.0),
            "forced" => record(forced, err(forced)), "ladder" => record(ladder, err(ladder)),
        )
    )
    @printf(
        "n=%4d m=%4d  dense %-13s %4d it %7.3fs | forced %-13s %4d it %7.3fs err %8.2e | ladder %-9s %5d it %7.3fs err %8.2e\n",
        n, m, dense.backend, dense.iterations, dense.solve_s,
        forced.backend, forced.iterations, forced.solve_s, err(forced),
        ladder.backend, ladder.iterations, ladder.solve_s, err(ladder)
    )
    flush(stdout)
end

out = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results", "admm_structured_operand.json")
open(io -> JSON.print(io, rows), out, "w")
println("wrote ", out)
