# Every solver this project is compared against, on two ill-conditioned dense 400x400 problems.
#
# One family carries Kronecker structure a solver can be told about; the other is unstructured,
# so nobody can exploit anything and the comparison is of implementations alone. Both are
# conditioned to `cond(A) = 1e8`, and the bounds cut off the unconstrained minimizer: with the
# minimizer inside the box every solver returns it at once and the timing measures nothing.
#
# `1e8` is the limit of what can be measured here, not a soft choice. At `cond(A) = 1e12` with
# these generators the solution is not determined: Clarabel at `1e-12` and PureIPM at `1e-10`
# then disagree by around `1e-1`, so there is no reference to judge any solver against and the
# error column would be reporting noise. The cross-check below is what licenses that column, and
# it is recorded per family for exactly this reason.
#
# Quality is judged from `x` alone against the caller's own `P, q, A, l, u`, because the dual
# sign conventions differ across these solvers. The reference is Clarabel at `1e-12`,
# cross-checked against PureIPM at `1e-10`; the agreement between those two is reported per
# family and is what makes the error column worth reading.
#
#     julia --project=bench bench/illconditioned_shootout.jl
#         # writes bench/results/illconditioned_shootout.json
using PureOSQP, PureIPM, PureDAQP, PureQPBase
using Clarabel, COSMO, QPALM, DAQP
using LinearAlgebra, SparseArrays, Random, JSON, Printf, Chairmarks

include(joinpath(@__DIR__, "helpers_conditioning.jl"))
include(joinpath(@__DIR__, "helpers_clarabel.jl"))
include(joinpath(@__DIR__, "..", "PureOSQP", "bench", "osqp_v1.jl"))

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "illconditioned_shootout.json")
const TOL = 1.0e-6
const SECONDS = 0.5
const SEED = 2026

"""
    plant(rng, P, A; frac = 0.10) -> (q, l, u, xstar, active)

Bounds and a linear term making a drawn `x⋆` the optimum, with `frac` of the rows active there.

This is the generator of `PureIPM/test/ipm_tests.jl`, applied to whatever `P` and `A` it is
handed, and it is what makes the benchmark measure anything. Bounds drawn around an arbitrary
point leave the unconstrained minimizer inside the box: every row is then slack, every solver
returns `-P⁻¹q` at once, and the timings compare unconstrained solves. Here `q` is set to
`-(P x⋆ + Aᵀ y⋆)`, which is stationarity, so `x⋆` is optimal by construction, its active set is
known, and it serves as the reference every solver is scored against.
"""
function plant(rng, P, A; frac = 0.1)
    m, n = size(A)
    xstar = randn(rng, n)
    a = A * xstar
    l = a .- (0.5 .+ rand(rng, m))
    u = a .+ (0.5 .+ rand(rng, m))
    ystar = zeros(m)
    active = randperm(rng, m)[1:round(Int, frac * m)]
    for i in active
        mag = 0.5 + rand(rng)
        if rand(rng, Bool)
            ystar[i] = mag
            u[i] = a[i]
        else
            ystar[i] = -mag
            l[i] = a[i]
        end
    end
    return -(P * xstar + A' * ystar), l, u, xstar, length(active)
end

"""
    kronecker_case(rng) -> (P, q, A, l, u, xstar, active, A1, A2)

`A = A₁ ⊗ A₂` from two 20×20 factors, so `cond(A)` is the product of the factors'. `P = μI` and
the bounds are two-sided, which is what the Kronecker backend requires. `A1` and `A2` come back
so the same problem can be posed again as a `KroneckerOperator` rather than as the 400×400
matrix it multiplies out to.
"""
function kronecker_case(rng)
    k = 20
    A1 = illconditioned(rng, k, 1.0e4)
    A2 = illconditioned(rng, k, 1.0e4)
    A = kron(A1, A2)
    n = size(A, 2)
    P = Matrix(0.1I, n, n)
    q, l, u, xstar, active = plant(rng, P, A)
    return P, q, A, l, u, xstar, active, A1, A2
end

"Unstructured dense `A` at `cond(A) = 1e8`, with `P`'s eigenvalues spread over `1e6`."
function control_case(rng)
    n = 400
    A = illconditioned(rng, n, 1.0e8)
    P = spd(rng, n, 1.0e6)
    q, l, u, xstar, active = plant(rng, P, A)
    return P, q, A, l, u, xstar, active
end

"∞-norm of the bound violation at `x`, which every solver is judged by rather than by its duals."
violation(A, l, u, x) = (z = A * x; maximum(max.(z .- u, 0.0) .+ max.(l .- z, 0.0); init = 0.0))

objective(P, q, x) = 0.5 * dot(x, P, x) + dot(q, x)

relerr(x, ref) = maximum(abs, x .- ref; init = 0.0) / max(1.0, maximum(abs, ref; init = 0.0))

"Rows resting on a bound at `x`, the quantity that decides an active-set method's cost."
function active_rows(A, l, u, x)
    z = A * x
    # Relative to each row's own scale: one absolute tolerance across rows whose bounds span
    # orders of magnitude counts either every row or none.
    return count(eachindex(z)) do i
        tol = 1.0e-7 * max(1.0, abs(l[i]), abs(u[i]))
        abs(z[i] - l[i]) <= tol || abs(z[i] - u[i]) <= tol
    end
end

"""
    timed(f) -> (; time, x, iters)

Median run time of `f`, which returns `(x, iterations)`. Setup is inside `f`.

Timed with Chairmarks rather than `@elapsed` over a few repetitions. Three `@elapsed` windows
cannot separate a garbage collection from the work: on these instances that read a solver which
allocates during setup as 1.6x slower than it is, while leaving the solvers that allocate less
alone, which is a difference between solvers that is not in the solvers.
"""
function timed(f)
    x, iters = f()
    b = @b f() seconds = SECONDS
    return (; time = b.time, x, iters)
end

function run_qpalm(P, q, A, l, u, tol)
    model = QPALM.Model()
    QPALM.setup!(
        model; Q = sparse(triu(P)), q = q, A = sparse(A), bmin = l, bmax = u,
        eps_abs = tol, eps_rel = tol, verbose = false,
    )
    r = QPALM.solve!(model)
    return (r.x, r.info.iter)
end

function run_cosmo(P, q, A, l, u, tol)
    model = COSMO.Model()
    # COSMO scales the bounds it is given in place. Handing it the harness's own arrays leaves
    # every later solver, and the active-set count, reading rescaled bounds.
    cs = COSMO.Constraint(sparse(A), zeros(length(l)), COSMO.Box(copy(l), copy(u)))
    COSMO.assemble!(
        model, sparse(P), q, cs,
        settings = COSMO.Settings(verbose = false, eps_abs = tol, eps_rel = tol, max_iter = 20_000),
    )
    r = COSMO.optimize!(model)
    return (r.x, r.iter)
end

"Clarabel returns the cone slacks after `x`; only the first `n` entries are the primal point."
function clarabel_x(P, q, A, l, u, tol, n)
    r = run_clarabel(P, q, A, l, u; tol)
    return (r.x[firstindex(r.x):(firstindex(r.x) + n - 1)], r.iterations)
end

function solvers_for(P, q, A, l, u; kron_factors = nothing)
    n = length(q)
    out = Pair{String, Function}[]
    push!(
        out, "PureOSQP" => () -> begin
            s = PureOSQP.solve(P, q, A, l, u; eps_abs = TOL, eps_rel = TOL, max_iter = 20_000)
            (s.x, s.iter)
        end
    )
    if !isnothing(kron_factors)
        A1, A2 = kron_factors
        push!(
            out, "PureOSQP :kronecker" => () -> begin
                Ak = PureQPBase.KroneckerOperator(A1, A2)
                s = PureOSQP.solve(P, q, Ak, l, u; eps_abs = TOL, eps_rel = TOL, max_iter = 20_000, scaling = 0)
                (s.x, s.iter)
            end
        )
    end
    push!(
        out, "PureIPM" => () -> begin
            s = PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); eps_abs = TOL, eps_rel = TOL)
            (s.x, s.iter)
        end
    )
    # `eps_prox = 0` runs the method on its own and needs `P ≻ 0`; a positive value runs
    # proximal-point outer iterations instead, which a positive definite `P` pays for and does
    # not need. Taking the default first and regularizing only on refusal is what a caller
    # would do, and is worth 20% on a family whose `P` is already definite.
    eps_prox = try
        PureOSQP.solve(P, q, A, l, u, PureDAQP.ActiveSet())
        0.0
    catch err
        err isa ArgumentError || rethrow()
        1.0e-4
    end
    push!(
        out, "PureDAQP" => () -> begin
            s = PureOSQP.solve(P, q, A, l, u, PureDAQP.ActiveSet(; eps_prox))
            (s.x, s.iter)
        end
    )
    push!(out, "DAQP" => () -> (DAQP.quadprog(P, q, A, u, l, zeros(Cint, length(l)))[1], -1))
    push!(
        out, "OSQP (libosqp)" => () -> begin
            r = solve_v1(P, q, A, l, u; eps_abs = TOL, eps_rel = TOL, max_iter = 20_000, verbose = false)
            (r.x, r.iter)
        end
    )
    push!(out, "Clarabel" => () -> clarabel_x(P, q, A, l, u, TOL, n))
    push!(out, "QPALM" => () -> run_qpalm(P, q, A, l, u, TOL))
    push!(out, "COSMO" => () -> run_cosmo(P, q, A, l, u, TOL))
    return out
end

function run_family(label, P, q, A, l, u, xstar, planted_active; kron_factors = nothing)
    n = length(q)
    # `x⋆` is the reference: it is optimal by construction, not by another solver's say-so.
    # Clarabel at 1e-12 is run anyway as an independent check that the planting is sound, and
    # its agreement with `x⋆` is reported so the error column can be read with that in mind.
    ref = xstar
    check, _ = clarabel_x(P, q, A, l, u, 1.0e-12, n)
    cross = relerr(check, ref)
    @printf("  planted x* vs Clarabel 1e-12: %.1e   (planted active rows: %d)\n", cross, planted_active)
    flush(stdout)

    rows = []
    # A solver that writes into the problem it was handed corrupts every later one, silently and
    # in a way that looks like the later solver failing. Checked per solver rather than trusted.
    guard = (copy(P), copy(q), copy(A), copy(l), copy(u))
    for (name, f) in solvers_for(P, q, A, l, u; kron_factors)
        r = timed(f)
        (P, q, A, l, u) == guard || error("$name modified the problem data it was given")
        x = collect(r.x)
        v, o, e = violation(A, l, u, x), objective(P, q, x), relerr(x, ref)
        # A solver that reports success and returns an infeasible point is the failure worth
        # catching, so the verdict comes from the returned `x`, not from the solver's status.
        ok = v <= 1.0e-5 * max(1.0, maximum(abs, u[isfinite.(u)]; init = 1.0))
        push!(
            rows, Dict(
                "solver" => name, "time_s" => r.time, "iter" => r.iters,
                "violation" => v, "objective" => o, "relerr" => e, "feasible" => ok,
            )
        )
        @printf(
            "  %-22s %9.4f s  iter %6d  viol %8.1e  obj %14.6f  relerr %8.1e  %s\n",
            name, r.time, r.iters, v, o, e, ok ? "ok" : "INFEASIBLE"
        )
        flush(stdout)
    end
    return Dict(
        "family" => label, "n" => n, "m" => size(A, 1),
        "cond_A" => cond(A), "zeros_frac" => count(iszero, A) / length(A),
        "active_rows" => active_rows(A, l, u, ref), "planted_active" => planted_active,
        "ref_cross_check" => cross,
        "rows" => rows,
    )
end

rng = MersenneTwister(SEED)
println("Kronecker family")
flush(stdout)
Pk, qk, Ak, lk, uk, xk, ak, A1, A2 = kronecker_case(rng)
fam1 = run_family("Kronecker", Pk, qk, Ak, lk, uk, xk, ak; kron_factors = (A1, A2))

println("Dense control family")
flush(stdout)
Pc, qc, Ac, lc, uc, xc, ac = control_case(rng)
fam2 = run_family("Dense control", Pc, qc, Ac, lc, uc, xc, ac)

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "tol" => TOL, "seconds_per_solver" => SECONDS, "seed" => SEED,
            "versions" => Dict(
                "Clarabel" => string(pkgversion(Clarabel)), "COSMO" => string(pkgversion(COSMO)),
                "QPALM" => string(pkgversion(QPALM)), "DAQP" => string(pkgversion(DAQP)),
                "PureOSQP" => string(pkgversion(PureOSQP)), "PureIPM" => string(pkgversion(PureIPM)),
                "PureDAQP" => string(pkgversion(PureDAQP)),
            ),
            "families" => [fam1, fam2],
        ), 2
    )
end
println("\nsaved ", RESULTS)
