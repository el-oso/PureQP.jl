# Ship gate for `InteriorPoint()` on `linsys = :indirect` with a caller-supplied preconditioner.
#
# Dense instances with a planted solution, `P` and `A` passed as `LinearMap`s, preconditioned by
# `LaggedCholesky` (refreshed every third outer iteration), at `n ∈ {500, 1000, 2000}`,
# `κ(A) ∈ {1, 1e6}`, active fractions `{0.1, 0.9}`, every row two-sided and with a row mix of
# 20% equality, 20% lower-only, 20% upper-only and 10% free rows. `δ = 1e-8`, `scaling = 0`,
# `eps = 1e-6`, `cg_max_iter = 500`, BLAS single-threaded, configurations run on several Julia
# threads at once (timings are indicative).
#
#   G1  referee ≤ 1e-5 with status SOLVED
#   G2  t ≤ max(10 f, min(100, n/10)), no solve at `cg_max_iter`, largest inner count below n,
#       with f the median inner iterations per solve over outer iterations 0–2 and t over the
#       last three
#
# Also recorded: wall clock including the preconditioner builds; the same instance through the
# dense interior point (`FullKKT`) and through ADMM on the same operators (`:indirect`, `n = 500`
# only, 10 s limit); a primal-infeasible operator instance; a preconditioner that is not
# positive definite, which must end `NUMERICAL_ERROR`; and one sparse instance through
# `IncompleteLDL`, which does not enter the verdict.
#
#     julia -t 4 --project=bench PureIPM/bench/ipm_matrixfree.jl
using PureOSQP, PureIPM, PureQPBase, LinearAlgebra, SparseArrays, Random, Statistics, Krylov, LinearMaps, JSON, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "ipm_preconditioners.jl"))

const KrylovExt = Base.get_extension(PureQPBase, :PureQPBaseKrylovExt)
const EPS = 1.0e-6
const REFEREE = 1.0e-5

"A preconditioner that records `(k, inner iterations, reached)` for every solve it serves."
struct Logged{M}
    inner::M
    log::Vector{NTuple{3, Int}}
end
Logged(M) = Logged(M, NTuple{3, Int}[])
function PureOSQP.update_preconditioner!(L::Logged, prob, wt, k::Int)
    PureOSQP.update_preconditioner!(L.inner, prob, wt, k)
    return L
end
LinearAlgebra.ldiv!(y::AbstractVector, L::Logged, x::AbstractVector) = ldiv!(y, L.inner, x)

function PureOSQP.solve_system!(
        ls::KrylovExt.IndirectCG{T, V, K, <:Logged}, prob, wt, rx, rz, x, z
    ) where {T, V <: AbstractVector{T}, K}
    before = ls.total_iters
    invoke(PureOSQP.solve_system!, Tuple{KrylovExt.IndirectCG, Any, Any, Any, Any, Any, Any}, ls, prob, wt, rx, rz, x, z)
    push!(ls.precond.log, (ls.refresh_index, ls.total_iters - before, Int(ls.last_reached)))
    return nothing
end

function kkt_residuals(P, q, A, l, u, x, y)
    Ax = A * x
    r_prim = maximum(abs, Ax .- clamp.(Ax, l, u))
    r_dual = maximum(abs, P * x .+ q .+ A' * y)
    quad, lin = dot(x, P * x), dot(q, x)
    sup, sign_viol = 0.0, 0.0
    for i in eachindex(y)
        iszero(y[i]) && continue
        b = y[i] > 0 ? u[i] : l[i]
        isfinite(b) ? (sup += b * y[i]) : (sign_viol = max(sign_viol, abs(y[i])))
    end
    r_gap = abs(quad + lin + sup) / max(1.0, abs(quad), abs(lin), abs(sup))
    return max(r_prim, r_dual, r_gap, sign_viol / max(maximum(abs, y), eps()))
end

"Planted instance: the generator of `PureIPM/test/ipm_tests.jl` (`spike_problem`), `m = n`."
function instance(n, κ, frac, seed; mixed = false)
    rng = Xoshiro(seed)
    m = n
    U = Matrix(qr(randn(rng, m, m)).Q)
    V = Matrix(qr(randn(rng, n, n)).Q)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    A = U * Diagonal(exp10.(range(0, -log10(κ); length = n))) * V'
    P = Q * Diagonal(exp10.(range(0, -2; length = n))) * Q'
    P = (P + P') / 2
    xstar = randn(rng, n)
    a = A * xstar
    l = a .- (0.5 .+ rand(rng, m))
    u = a .+ (0.5 .+ rand(rng, m))
    ystar = zeros(m)
    if !mixed
        for i in randperm(rng, m)[1:round(Int, frac * m)]
            mag = 0.5 + rand(rng)
            rand(rng, Bool) ? (ystar[i] = mag; u[i] = a[i]) : (ystar[i] = -mag; l[i] = a[i])
        end
        return P, -(P * xstar + A' * ystar), A, l, u
    end
    kind = fill(:two, m)
    p = randperm(rng, m)
    j = 0
    for (k, f) in ((:eq, 0.2), (:lo, 0.2), (:up, 0.2), (:free, 0.1)), _ in 1:round(Int, f * m)
        kind[p[j += 1]] = k
    end
    fill!(l, -Inf)
    fill!(u, Inf)
    for i in 1:m
        gap, mag, active, k = 0.5 + rand(rng), 0.5 + rand(rng), rand(rng) < frac, kind[i]
        if k === :eq
            l[i] = u[i] = a[i]
            ystar[i] = rand(rng, Bool) ? mag : -mag
        elseif k === :lo
            active ? (l[i] = a[i]; ystar[i] = -mag) : (l[i] = a[i] - gap)
        elseif k === :up
            active ? (u[i] = a[i]; ystar[i] = mag) : (u[i] = a[i] + gap)
        elseif k === :two
            l[i] = a[i] - gap
            u[i] = a[i] + 0.5 + rand(rng)
            active && (rand(rng, Bool) ? (u[i] = a[i]; ystar[i] = mag) : (l[i] = a[i]; ystar[i] = -mag))
        end
    end
    return P, -(P * xstar + A' * ystar), A, l, u
end

# Wrapped explicitly: a `LinearMap` over a matrix reaches the solver as that matrix, and this
# benchmark is about a pair that supplies products only.
operators(P, A) = (
    PureQPBase.ProductOperator{Float64}(LinearMap(P); symmetric = true, posdef = true),
    PureQPBase.ProductOperator{Float64}(LinearMap(A)),
)

function g2(log, iter, n)
    solves = filter(s -> s[1] >= 0, log)
    f = median([s[2] for s in solves if s[1] <= 2])
    t = median([s[2] for s in solves if s[1] >= iter - 3])
    maxinner = maximum(s[2] for s in log)
    misses = count(s -> iszero(s[3]), log)
    bound = max(10f, min(100, n / 10))
    return (; f, t, bound, maxinner, misses, pass = t <= bound && iszero(misses) && maxinner < n)
end

function run_case(n, κ, frac, mixed, seed)
    P, q, A, l, u = instance(n, κ, frac, seed; mixed)
    Pop, Aop = operators(P, A)
    row = Dict{String, Any}("n" => n, "kappa" => κ, "frac" => frac, "rows" => mixed ? "mixed" : "two-sided")

    t0 = time()
    M = Logged(LaggedCholesky(P, A; every = 3))
    s = PureOSQP.solve(
        Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, preconditioner = M,
        scaling = 0, eps_abs = EPS, eps_rel = EPS
    )
    wall = time() - t0
    referee = has_solution(s.status) ? kkt_residuals(P, q, A, l, u, s.x, s.y) : Inf
    G1 = s.status == SOLVED && referee <= REFEREE
    gate2 = g2(M.log, s.iter, n)
    merge!(
        row, Dict(
            "status" => string(s.status), "iter" => s.iter, "referee" => referee, "G1" => G1,
            "cg_iters" => s.cg_iters, "solves" => length(M.log), "builds" => M.inner.builds,
            "wall_s" => wall, "f" => gate2.f, "t" => gate2.t, "g2_bound" => gate2.bound,
            "max_inner" => gate2.maxinner, "misses" => gate2.misses, "G2" => gate2.pass,
            "inner_per_solve" => [s[2] for s in M.log],
        )
    )

    t0 = time()
    d = PureOSQP.solve(P, q, A, l, u, InteriorPoint(); linsys = :kkt, scaling = 0, eps_abs = EPS, eps_rel = EPS)
    row["dense_ipm"] = Dict(
        "status" => string(d.status), "iter" => d.iter, "wall_s" => time() - t0,
        "referee" => has_solution(d.status) ? kkt_residuals(P, q, A, l, u, d.x, d.y) : Inf,
    )

    if n <= 500
        t0 = time()
        a = PureOSQP.solve(
            Pop, q, Aop, l, u; linsys = :indirect, scaling = 0, eps_abs = EPS, eps_rel = EPS,
            max_iter = 1_000_000, time_limit = 10.0
        )
        row["admm_operator"] = Dict(
            "status" => string(a.status), "iter" => a.iter, "cg_iters" => a.cg_iters, "wall_s" => time() - t0,
            "referee" => has_solution(a.status) ? kkt_residuals(P, q, A, l, u, a.x, a.y) : Inf,
        )
    end
    admm = get(row, "admm_operator", nothing)
    @printf(
        "n=%4d κ=%.0e frac=%.1f %-9s %s iter=%2d ref=%.1e G1=%d f=%.1f t=%.1f bound=%.0f max=%d miss=%d G2=%d  %.2fs (dense %.2fs, admm %s)\n",
        n, κ, frac, row["rows"], s.status, s.iter, referee, G1, gate2.f, gate2.t, gate2.bound,
        gate2.maxinner, gate2.misses, gate2.pass, wall, row["dense_ipm"]["wall_s"],
        isnothing(admm) ? "-" : @sprintf("%s %d it %.2fs ref %.1e", admm["status"], admm["iter"], admm["wall_s"], admm["referee"])
    )
    flush(stdout)
    return row
end

"""
A sparse instance (`A` with 10 random nonzeros per row plus the identity, `P = 0.1I`, two-sided
rows around a random point) through `IncompleteLDL`. Published with its own G1 and G2; it does
not enter the verdict.
"""
function sparse_case(n; seed = 7)
    rng = Xoshiro(seed)
    A = sprandn(rng, n, n, 10 / n) + sparse(1.0I, n, n)
    P = sparse(0.1I, n, n)
    b = A * randn(rng, n)
    q, l, u = randn(rng, n), b .- rand(rng, n), b .+ rand(rng, n)
    t0 = time()
    M = Logged(IncompleteLDL(P, A))
    s = PureOSQP.solve(
        P, q, A, l, u, InteriorPoint(); linsys = :indirect, preconditioner = M, scaling = 0,
        eps_abs = EPS, eps_rel = EPS
    )
    wall = time() - t0
    referee = has_solution(s.status) ? kkt_residuals(P, q, A, l, u, s.x, s.y) : Inf
    gate2 = g2(M.log, s.iter, n)
    return Dict(
        "n" => n, "status" => string(s.status), "iter" => s.iter, "referee" => referee,
        "G1" => s.status == SOLVED && referee <= REFEREE, "G2" => gate2.pass, "f" => gate2.f,
        "t" => gate2.t, "max_inner" => gate2.maxinner, "misses" => gate2.misses, "wall_s" => wall,
        "inner_per_solve" => [s[2] for s in M.log],
    )
end

"Two copies of a row with disjoint intervals: `aᵀx ≤ 0` and `aᵀx ≥ 1`."
function infeasible_case(n)
    P, q, A, l, u = instance(n, 1.0, 0.1, 99)
    A[2, :] .= A[1, :]
    l[1], u[1] = -Inf, 0.0
    l[2], u[2] = 1.0, Inf
    Pop, Aop = operators(P, A)
    s = PureOSQP.solve(
        Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, scaling = 0,
        preconditioner = LaggedCholesky(P, A; every = 3), eps_abs = EPS, eps_rel = EPS
    )
    return Dict("n" => n, "status" => string(s.status), "iter" => s.iter, "cg_iters" => s.cg_iters)
end

"A diagonal preconditioner with one negative entry."
function indefinite_case(n)
    P, q, A, l, u = instance(n, 1.0, 0.1, 98)
    Pop, Aop = operators(P, A)
    d = ones(n)
    d[1] = -1.0
    s = PureOSQP.solve(
        Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, scaling = 0,
        preconditioner = Diagonal(d), eps_abs = EPS, eps_rel = EPS
    )
    return Dict("n" => n, "status" => string(s.status), "iter" => s.iter, "cg_iters" => s.cg_iters)
end

"JSON has no `Inf`: a run without a point, whose referee is `Inf`, is written as `null`."
finite_or_null(x::Dict) = Dict(k => finite_or_null(v) for (k, v) in x)
finite_or_null(x::AbstractVector) = map(finite_or_null, x)
finite_or_null(x::AbstractFloat) = isfinite(x) ? x : nothing
finite_or_null(x) = x

function main(; sizes = (500, 1000, 2000), path = joinpath(@__DIR__, "results", "ipm_matrixfree.json"))
    configs = [
        (n, κ, frac, mixed) for n in sizes for κ in (1.0, 1.0e6) for frac in (0.1, 0.9)
            for mixed in (false, true)
    ]
    run_case(60, 1.0e6, 0.9, true, 1)       # compile
    rows = Vector{Dict{String, Any}}(undef, length(configs))
    Threads.@threads :dynamic for i in eachindex(configs)
        n, κ, frac, mixed = configs[i]
        rows[i] = run_case(n, κ, frac, mixed, 1000 + i)
    end
    infeasible = infeasible_case(100)
    indefinite = indefinite_case(100)
    sparse_lldl = sparse_case(1000)
    println("sparse, IncompleteLDL: ", filter(p -> p.first != "inner_per_solve", sparse_lldl))
    println("infeasible: ", infeasible)
    println("indefinite preconditioner: ", indefinite)
    gate = [r for r in rows if r["n"] >= 500]
    verdict = Dict(
        "G1" => all(r -> r["G1"], gate), "G2" => all(r -> r["G2"], gate),
        "G1_count" => count(r -> r["G1"], gate), "G2_count" => count(r -> r["G2"], gate),
        "cases" => length(gate),
        "infeasible_ok" => infeasible["status"] == "PRIMAL_INFEASIBLE",
        "indefinite_ok" => indefinite["status"] == "NUMERICAL_ERROR",
    )
    println("verdict: ", verdict)
    out = Dict(
        "description" => "PureIPM/bench/ipm_matrixfree.jl: :ipm + :indirect + LaggedCholesky(every = 3) over LinearMaps of dense planted instances; eps = 1e-6, scaling = 0, cg_max_iter = 500",
        "julia" => string(VERSION), "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "cases" => rows, "sparse_incomplete_ldl" => sparse_lldl, "infeasible" => infeasible, "indefinite_preconditioner" => indefinite,
        "verdict" => verdict,
    )
    mkpath(dirname(path))
    open(io -> JSON.print(io, finite_or_null(out)), path, "w")
    return out
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
