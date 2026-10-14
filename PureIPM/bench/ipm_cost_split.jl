# Where the interior-point method's time goes on one problem, part by part.
#
# The Random QP class at its smallest size (n = 6, m = 60, 9 outer iterations) is the one
# class of PureIPM/bench/clarabel_rs_compare.jl's seven where Clarabel.rs is faster, so this splits
# that case into the pieces `setup` and `solve!` are made of and prints microseconds, share
# and allocations for each.
#
# Two passes, because neither alone gives the whole split:
#
#   1. `setup` is replayed part by part — `validate`, `Options`, `is_convex`,
#      `validated_problem` (the Ruiz sweeps), `choose_backend` and the workspace allocation
#      are each called on their own and timed with BenchmarkTools, which is exact because
#      each takes the same arguments `setup` would hand it.
#   2. `solve!` is re-run with `time_ns` around each part of its loop. That costs about
#      0.026 µs per reading, roughly 3 µs over a whole solve, so the instrumented total runs
#      a little above the plain one; the plain total is printed alongside for comparison.
#      Runs in which the garbage collector ran are dropped and the minimum is taken over the
#      rest, matching how PureIPM/bench/clarabel_rs_compare.jl reports its own minima.
#
# The Rust side of the same split is bench/clarabel_rs/src/bin/split.rs, which reads
# Clarabel's own `Solver::timers`.
#
# Rerun:
#     julia --project=bench PureIPM/bench/ipm_cost_split.jl
using PureOSQP, PureIPM, LinearAlgebra, SparseArrays, Random, BenchmarkTools, Printf, Statistics

BLAS.set_num_threads(1)

const PO = PureOSQP
const TOL = 1.0e-8
const CORE = 15

"Pin this process to `cpu` (glibc `cpu_set_t` is a 1024-bit mask on x86_64), as the head-to-head does."
function pin_to_cpu!(cpu::Integer)
    mask = zeros(UInt64, 16)
    mask[cpu ÷ 64 + 1] |= UInt64(1) << (cpu % 64)
    return iszero(ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(mask), mask))
end

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

pin_to_cpu!(CORE)
const P, q, A, l, u = random_qp(6)
mk() = PO.setup(P, q, A, l, u, InteriorPoint(); eps_abs = TOL, eps_rel = TOL)

us(b) = 1.0e-3 * minimum(b.times)
row(name, t, total, allocs, bytes) =
    @printf("  %-34s %8.3f %6.1f%% %7d %9d\n", name, t, 100t / total, allocs, bytes)

b_all = @benchmark PO.solve($P, $q, $A, $l, $u, InteriorPoint(); eps_abs = TOL, eps_rel = TOL)
b_setup = @benchmark mk()
total = us(b_all)
@printf("Random QP n=%d m=%d, tol=%.0e: %.3f us total, %d allocations, %d B\n", size(A, 2), size(A, 1), TOL, total, b_all.allocs, b_all.memory)

# --- setup, replayed part by part ---------------------------------------------------------
T = Float64
alg = InteriorPoint()
nv, mv = PO.validate(P, q, A, l, u)
opts = PO.Options{T}(; PO.algorithm_defaults(alg, T)..., linsys = :auto, eps_abs = TOL, eps_rel = TOL)
algv = PO.element_typed(alg, T, opts)
prob = PO.validated_problem(T, nv, mv, P, q, A, l, u, opts.scaling)
wt = PO.SystemWeights(
    fill!(similar(prob.q0, T, mv), one(T)), fill!(similar(prob.q0, T, mv), one(T)), algv.reg_primal
)
sel = PO.IPMSelection()
backend = first(PO.choose_backend(P, A, prob, wt, sel))

@printf("\nsetup: %.3f us (%.1f%% of total), %d allocations, %d B\n", us(b_setup), 100us(b_setup) / total, b_setup.allocs, b_setup.memory)
println("  part                                     us  share  allocs     bytes")
for (name, b) in (
        "validate" => @benchmark(PO.validate($P, $q, $A, $l, $u)),
        "Options" => @benchmark(PO.Options{T}(; PO.algorithm_defaults($alg, T)..., linsys = :auto, eps_abs = TOL, eps_rel = TOL)),
        "is_convex" => @benchmark(PO.is_convex(T, $P, $(algv.reg_primal))),
        "equilibration (validated_problem)" => @benchmark(PO.validated_problem(T, $nv, $mv, $P, $q, $A, $l, $u, $(opts.scaling))),
        "backend selection + 1st factor" => @benchmark(PO.choose_backend($P, $A, $prob, $wt, $sel)),
        "workspace allocation" => @benchmark(
            PureIPM.ipm_workspace($backend, $prob, $wt, $algv, $opts)
        ),
    )
    row(name, us(b), total, b.allocs, b.memory)
end

# The starting point refactorizes the system `setup` already factored, at the same unit
# weights, so the cost of one numeric factorization is paid twice before the loop begins.
b_factor = @benchmark PO.factorize!(w.linsys, w.prob, w.weights) setup = (w = mk()) evals = 1
b_start = @benchmark PureIPM.starting_point!(w) setup = (w = mk()) evals = 1
w = mk()
fact_before, L_before = w.linsys.fact, copy(w.linsys.L.nzval)
PureIPM.starting_point!(w)
@printf(
    "\none numeric factorization: %.3f us, %d allocations, %d B\n", us(b_factor), b_factor.allocs, b_factor.memory
)
@printf(
    "starting_point! on a fresh workspace: %.3f us; refactorizes the same factor object (%s) to the same values (%s)\n",
    us(b_start), fact_before === w.linsys.fact, L_before == w.linsys.L.nzval
)

# --- solve!, instrumented part by part ----------------------------------------------------
const ACC = Dict{Symbol, Float64}()
const CNT = Dict{Symbol, Int}()

macro t(key, ex)
    return quote
        local t0 = time_ns()
        local v = $(esc(ex))
        local dt = time_ns() - t0
        ACC[$key] = get(ACC, $key, 0.0) + dt
        CNT[$key] = get(CNT, $key, 0) + 1
        v
    end
end

"`PureOSQP.solve!`'s loop with `time_ns` around each part. Safeguards this problem never trips are left out."
function instrumented_solve!(ws)
    s, alg = ws.options, ws.algorithm
    ws.status = PO.UNSOLVED
    ws.iter = 0
    ws.reg_bumps = 0
    PureIPM.set_regularization!(ws, alg.reg_primal, alg.reg_dual)
    ws.short_steps = 0
    ws.flat_merit = 0
    ws.last_merit = PO.INFTY(Float64)
    ws.alert = false
    ws.diverged = false
    ws.cg_misses = 0
    bound = PureIPM.iterate_bound(ws)
    @t :starting_point PureIPM.starting_point!(ws)
    @t :residuals PureIPM.ipm_residuals!(ws)
    for iter in 1:s.max_iter
        ws.status == PO.UNSOLVED || break
        ws.iter = iter
        @t :weights PureIPM.weights!(ws)
        PO.set_refresh_index!(ws.linsys, iter - 1)
        @t :factorize PureIPM.factorize_newton!(ws, false)
        @t :predictor_corrector PureIPM.ipm_step!(ws)
        @t :residuals PureIPM.ipm_residuals!(ws)
        stall = @t :stall PureIPM.stalled!(ws, bound)
        if stall || ws.alert || (s.check_termination > 0 && iszero(iter % s.check_termination))
            st = @t :termination PO.check_termination(ws, false, stall || ws.alert)
            st == PO.UNSOLVED || (ws.status = st; break)
        end
        stall && break
    end
    @t :build_solution PureIPM.build_solution(ws)
    return ws.iter, ws.status
end

const LOOP_PARTS = (
    :starting_point, :factorize, :predictor_corrector, :residuals,
    :termination, :stall, :weights, :build_solution,
)

"Minimum per part over `nrep` unseeded runs, dropping every run in which the collector ran."
function loop_split(nrep)
    per = Dict(k => Float64[] for k in LOOP_PARTS)
    tots = Float64[]
    collected = Bool[]
    for _ in 1:nrep
        g0 = Base.gc_num().total_time
        ws = mk()
        empty!(ACC)
        empty!(CNT)
        t0 = time_ns()
        instrumented_solve!(ws)
        push!(tots, Float64(time_ns() - t0))
        push!(collected, Base.gc_num().total_time != g0)
        for k in LOOP_PARTS
            push!(per[k], get(ACC, k, 0.0))
        end
    end
    keep = .!collected
    return per, tots, keep
end

instrumented_solve!(mk())
per, tots, keep = loop_split(3000)
loop_total = 1.0e-3 * minimum(tots[keep])
@printf(
    "\nsolve!: %.3f us instrumented (%.3f us plain, the difference is the time_ns readings), %d of %d runs collector-free\n",
    loop_total, total - us(b_setup), count(keep), length(keep)
)
println("  part                                     us  share   calls")
attributed = 0.0
for k in LOOP_PARTS
    t = 1.0e-3 * minimum(per[k][keep])
    global attributed += t
    @printf("  %-34s %8.3f %6.1f%% %7d\n", k, t, 100t / total, CNT[k])
end
@printf("  %-34s %8.3f %6.1f%%\n", "unattributed", loop_total - attributed, 100(loop_total - attributed) / total)
