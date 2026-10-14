# Matrix-free IPM spike: how hard are the reduced systems of a Mehrotra IPM for CG?
#
# A dense Mehrotra predictor–corrector prototype (Float64, `bunchkaufman!` on the proximally
# regularized KKT matrix) solves
#
#     min ½xᵀPx + qᵀx   s.t.   l ≤ Ax ≤ u
#
# on instances with a known optimum, a prescribed κ(A) and a prescribed fraction of active rows.
# Every row is two-sided with finite bounds, so there are no equality or free rows. The script
# is bench-only and independent of the package's internals.
#
# Per instance and per δ = δ_p = δ_d:
#   * variant runs: exact solves only, for each regularized-step variant in `VARIANTS`.
#   * exact run (variant `PRIMARY`): the outer loop with exact KKT solves. At each predictor and corrector solve the
#     reduced system `(P + δI + Aᵀ diag(w) A) Δx = −r_d − Aᵀg`, `w = 1/(1/W + δ)`, is also solved
#     by `Krylov.cg!` with each preconditioner (none, exact Jacobi, probed Woodbury k = 5, 20),
#     and the iterations to `atol_k` are recorded (the CG result is discarded).
#   * restart experiment (`RESTART_PRECONDS`): for solves that need more than `CG_DEFAULT_ITER` iterations, CG is
#     restarted every `CG_DEFAULT_ITER` iterations from its current point (iterative refinement
#     through CG) and the total count is compared with the unrestarted run.
#   * inexact runs: the outer loop with the CG solve in place of the exact one, per
#     preconditioner.
#
# CG stops on the unpreconditioned residual `‖b − KΔx‖₂ ≤ atol_k` (checked through a callback on
# Krylov's recursively updated residual, then recomputed from scratch), so iteration counts are
# comparable across preconditioners. The starting point is computed by an exact solve in every
# run. The P and A seen by CG and the Woodbury construction are `LinearMap`s over the dense
# matrices; exact Jacobi reads the dense matrices directly.
#
# Run from the repository root:  julia +1.12 -t 7 --project=bench PureIPM/bench/ipm_matrixfree_spike.jl
# Output: PureIPM/bench/results/ipm_matrixfree_spike.json (raw samples) and
#         PureIPM/bench/results/ipm_matrixfree_spike_summary.json (compact summary, `write_summary`)

using LinearAlgebra, Krylov, LinearMaps, Clarabel, JSON, Random, Statistics, SparseArrays, Printf

const NS = (200, 500, 1000)
const KAPPAS = (1.0e0, 1.0e3, 1.0e6)
const FRACS = (0.1, 0.5, 0.9)
const DELTAS = (1.0e-8, 1.0e-4, 1.0e-2)
const PRECONDS = (:none, :jacobi, :woodbury5, :woodbury20)
# Measured on exact trajectories only: Woodbury k = 20 with the exact core diagonal.
const DIAGNOSTIC_PRECONDS = (:woodbury20_exactdiag,)
const BATTERY = (PRECONDS..., DIAGNOSTIC_PRECONDS...)
# Preconditioners whose slow solves are also rerun restarted.
const RESTART_PRECONDS = (:none, :jacobi)
const EPS_LIST = (1.0e-6, 1.0e-8)       # termination tolerances; the run goes to the last one
const MAX_ITER = 100
const CG_CAP = 2000                     # generous cap: 10× the design's cg_max_iter default
const CG_DEFAULT_ITER = 200             # the design's cg_max_iter default
const CG_FAIL_LIMIT = 3                 # consecutive missed solves end an inexact run
const ETA = 0.1                         # cg_tol_fraction
const PROBES = 10                       # Hutchinson probes for the Woodbury core
const TAU = 0.99
# Exact runs without CG, one per (step recovery, μ safeguard) variant; see `ipm`.
const VARIANTS = ((:design, false), (:consistent, false), (:perside, false), (:perside, true))
# The variant that produces the trajectories CG is measured on and the inexact runs.
const PRIMARY = (; recovery = :perside, safeguard = true)

# ---------------------------------------------------------------------------------------------
# Instances

struct Instance
    n::Int
    kappa::Float64
    frac::Float64
    P::Matrix{Float64}
    q::Vector{Float64}
    A::Matrix{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    xstar::Vector{Float64}
    ystar::Vector{Float64}
    nactive::Int
end

"""
    make_instance(n, κ, frac; seed)

`m = n` rows. `A = U Σ Vᵀ` with singular values log-spaced from 1 to 1/κ; `P = Q Λ Qᵀ` with
eigenvalues log-spaced from 1 to 1e-2. A random `x⋆` and a set of `round(frac·m)` rows are
chosen; those rows get a multiplier of magnitude in [0.5, 1.5] and the matching bound placed at
`(Ax⋆)_i`, the other bound 0.5–1.5 away. Inactive rows have both bounds 0.5–1.5 away and a zero
multiplier. `q = −Px⋆ − Aᵀy⋆` makes `(x⋆, y⋆)` a strictly complementary KKT point, and the
unique optimum because `P` is positive definite.
"""
function make_instance(n, κ, frac; seed)
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
    nactive = round(Int, frac * m)
    for i in randperm(rng, m)[1:nactive]
        mag = 0.5 + rand(rng)
        if rand(rng, Bool)
            ystar[i] = mag
            u[i] = a[i]
        else
            ystar[i] = -mag
            l[i] = a[i]
        end
    end
    q = -(P * xstar + A' * ystar)
    return Instance(n, κ, frac, P, q, A, l, u, xstar, ystar, nactive)
end

objective(inst, x) = 0.5 * dot(x, inst.P * x) + dot(inst.q, x)

"""
    kkt_residuals(P, q, A, l, u, x, y) -> (r_prim, r_dual, r_opt)

The package's test referee (`PureIPM/test/helpers.jl`), restricted to finite bounds: absolute primal
and dual residuals and the relative duality gap `|xᵀPx + qᵀx + uᵀmax(y,0) + lᵀmin(y,0)|`.
"""
function kkt_residuals(P, q, A, l, u, x, y)
    Ax = A * x
    r_prim = maximum(abs, Ax .- clamp.(Ax, l, u))
    r_dual = maximum(abs, P * x .+ q .+ A' * y)
    quad = dot(x, P * x)
    lin = dot(q, x)
    sup = dot(u, max.(y, 0)) + dot(l, min.(y, 0))
    r_gap = abs(quad + lin + sup) / max(1.0, abs(quad), abs(lin), abs(sup))
    return (r_prim, r_dual, r_gap)
end
referee(inst, x, y) = maximum(kkt_residuals(inst.P, inst.q, inst.A, inst.l, inst.u, x, y))

function clarabel_reference(inst)
    m = size(inst.A, 1)
    settings = Clarabel.Settings(
        verbose = false, tol_gap_abs = 1.0e-10, tol_gap_rel = 1.0e-10, tol_feas = 1.0e-10,
        max_iter = 200,
    )
    solver = Clarabel.Solver()
    Ast = sparse([inst.A; -inst.A])
    Clarabel.setup!(
        solver, sparse(triu(inst.P)), inst.q, Ast, [inst.u; -inst.l],
        [Clarabel.NonnegativeConeT(2m)], settings,
    )
    t = @elapsed sol = Clarabel.solve!(solver)
    return (; x = sol.x, status = string(sol.status), obj = objective(inst, sol.x), time = t)
end

# ---------------------------------------------------------------------------------------------
# Reduced operator and preconditioners (products only, except exact Jacobi)

"`y = (P + δI + Aᵀ diag(w) A) x` through LinearMap products; `tmp` has length m."
function reduced_mul!(y, Pop, Aop, w, δ, tmp, x)
    mul!(tmp, Aop, x)
    tmp .*= w
    mul!(y, Aop', tmp)
    mul!(y, Pop, x, 1.0, 1.0)
    y .+= δ .* x
    return y
end

function reduced_map(Pop, Aop, w, δ)
    n = size(Pop, 1)
    tmp = zeros(size(Aop, 1))
    return LinearMap{Float64}(
        (y, x) -> reduced_mul!(y, Pop, Aop, w, δ, tmp, x), n;
        ismutating = true, issymmetric = true, isposdef = true,
    )
end

jacobi_map(inst, w, δ) = Diagonal(1 ./ (diag(inst.P) .+ δ .+ (inst.A .^ 2)' * w))

"""
    woodbury_map(Pop, Aop, w, winv, δ, k, rng)

`R ≈ C + Vᵀ diag(w_K) V`, `K` the `k` rows with the largest `w`. `V` is read through `k`
adjoint products `Aᵀeᵢ`. `C` is a Hutchinson estimate (`PROBES` ±1 probes, three products each)
of the diagonal of `P + δI + Aᵀ diag(w with w_K = 0) A`, floored at `δ`. Applied as
`C⁻¹x − Yᵀ cap⁻¹ Y x` with `Y = V C⁻¹`, `cap = diag(1/w_K) + Y Vᵀ`.

With `exact_diag = c`, `c` replaces the Hutchinson estimate (a diagnostic that separates the
estimate's error from the low-rank structure). With `stats`, the median relative error of the
estimate against the exact diagonal (computed from the dense matrices of `inst`) and the
fraction of entries raised to the floor are pushed to it.
"""
function woodbury_map(Pop, Aop, w, winv, δ, k, rng; exact_diag = nothing, inst = nothing, stats = nothing)
    n = size(Pop, 1)
    m = size(Aop, 1)
    K = partialsortperm(w, 1:k; rev = true)
    V = zeros(k, n)
    e = zeros(m)
    row = zeros(n)
    for (j, i) in enumerate(K)
        e[i] = 1.0
        mul!(row, Aop', e)
        e[i] = 0.0
        V[j, :] .= row
    end
    wrest = copy(w)
    wrest[K] .= 0
    tmp = zeros(m)
    est = zeros(n)
    v = zeros(n)
    Mv = zeros(n)
    for _ in 1:PROBES
        v .= rand(rng, (-1.0, 1.0), n)
        reduced_mul!(Mv, Pop, Aop, wrest, δ, tmp, v)
        est .+= v .* Mv
    end
    est ./= PROBES
    if !isnothing(stats)
        cex = diag(inst.P) .+ δ .+ (inst.A .^ 2)' * wrest
        push!(stats, Dict("k" => k, "hutch_relerr_median" => median(abs.(est .- cex) ./ cex), "floored_fraction" => mean(est .< δ)))
    end
    isnothing(exact_diag) || (est .= exact_diag)
    cinv = 1 ./ max.(est, δ)
    Y = V .* cinv'
    capf = cholesky!(Symmetric(Diagonal(winv[K]) + Y * V'))
    ky = zeros(k)
    apply! = function (y, x)
        mul!(ky, Y, x)
        ldiv!(capf, ky)
        y .= cinv .* x
        mul!(y, Y', ky, -1.0, 1.0)
        return y
    end
    return LinearMap{Float64}(apply!, n; ismutating = true, issymmetric = true)
end

function build_precond(pc, inst, Pop, Aop, w, winv, δ, rng; stats = nothing)
    pc === :none && return I
    pc === :jacobi && return jacobi_map(inst, w, δ)
    pc === :woodbury5 && return woodbury_map(Pop, Aop, w, winv, δ, 5, rng; inst, stats)
    pc === :woodbury20 && return woodbury_map(Pop, Aop, w, winv, δ, 20, rng; inst, stats)
    if pc === :woodbury20_exactdiag
        K = partialsortperm(w, 1:20; rev = true)
        wrest = copy(w)
        wrest[K] .= 0
        cex = diag(inst.P) .+ δ .+ (inst.A .^ 2)' * wrest
        return woodbury_map(Pop, Aop, w, winv, δ, 20, rng; exact_diag = cex)
    end
    error("unknown preconditioner $pc")
end

"""
    cg_solve(op, M, b, atol, cap; restart = cap)

CG from zero on `op x = b`, stopping when the unpreconditioned residual 2-norm reaches `atol`
or after `cap` iterations; with `restart < cap`, restarted from its current point every
`restart` iterations. Returns the solution, total iterations and the recomputed residual.

Krylov throws when the preconditioned inner product `rᵀMr` turns negative (a preconditioner
that is numerically indefinite). That solve is reported as `breakdown` with the iteration it
happened at, `iters = cap` and `reached = false`, and the current iterate.
"""
function cg_solve(op, M, b, atol, cap; restart = cap)
    n = length(b)
    ws = CgWorkspace(n, n, Vector{Float64})
    calls = Ref(0)
    stop = w -> (calls[] += 1; norm(w.r) <= atol)
    x = zeros(n)
    total = 0
    res = norm(b)
    first = true
    while total < cap && res > atol
        first || Krylov.warm_start!(ws, x)
        first = false
        calls[] = 0
        try
            cg!(ws, op, b; M, ldiv = false, atol = 0.0, rtol = 0.0, itmax = min(restart, cap - total), callback = stop)
        catch e
            (e isa ErrorException && occursin("positive definite", e.msg)) || rethrow()
            return (; x = copy(ws.x), iters = cap, res = norm(b - op * ws.x), reached = false, breakdown = total + calls[])
        end
        total += ws.stats.niter
        x .= ws.x
        res = norm(b - op * x)
        iszero(ws.stats.niter) && break
    end
    return (; x, iters = total, res, reached = res <= atol, breakdown = nothing)
end

# ---------------------------------------------------------------------------------------------
# Mehrotra predictor–corrector

function termination(inst, x, y, Ax, ϵ)
    P, q, A, l, u = inst.P, inst.q, inst.A, inst.l, inst.u
    z = clamp.(Ax, l, u)
    Px = P * x
    Aty = A' * y
    prim = maximum(abs, Ax .- z)
    dual = maximum(abs, Px .+ q .+ Aty)
    quad = dot(x, Px)
    lin = dot(q, x)
    sup = dot(u, max.(y, 0)) + dot(l, min.(y, 0))
    gap = abs(quad + lin + sup)
    ok_prim = prim <= ϵ + ϵ * max(norm(Ax, Inf), norm(z, Inf))
    ok_dual = dual <= ϵ + ϵ * max(norm(Px, Inf), norm(Aty, Inf), norm(q, Inf))
    ok_gap = gap <= ϵ + ϵ * max(abs(quad), abs(lin), abs(sup))
    return (; prim, dual, gap, pass = ok_prim && ok_dual && ok_gap)
end

function max_step(v, dv)
    α = Inf
    for i in eachindex(v, dv)
        dv[i] < 0 && (α = min(α, -v[i] / dv[i]))
    end
    return α
end

"""
    ipm(inst, δ; inner = :exact, precond = :none, battery = false, recovery = :perside, safeguard = false)

`recovery` selects the weights and how `Δs, Δz` are recovered from `Δx`:
`:design` takes `w = 1/(1/W + δ)` and recovers `Δs_l = AΔx + r_l` (so `Δz_u − Δz_l` is not the
solved `Δy`); `:consistent` keeps that `w` and rescales `AΔx` so the two agree; `:perside`
regularizes each one-sided row, `w = z_l/(s_l + δz_l) + z_u/(s_u + δz_u)`. `safeguard` floors
Mehrotra's `σ` at `min(1, 0.1‖r‖∞/μ)`.

Runs the outer loop to `last(EPS_LIST)` or `MAX_ITER`. `inner = :exact` solves the regularized
KKT system with `bunchkaufman!`; `inner = :cg` solves the reduced system with `cg_solve` and
`precond`. With `battery = true` (exact runs), every predictor and corrector reduced system is
also solved by CG under each preconditioner, unrestarted and restarted every
`CG_DEFAULT_ITER` iterations, and the counts are recorded.
"""
function ipm(inst, δ; inner = :exact, precond = :none, battery = false, recovery = :perside, safeguard = false)
    P, q, A, l, u = inst.P, inst.q, inst.A, inst.l, inst.u
    n = inst.n
    m = size(A, 1)
    Pop = LinearMap(P; issymmetric = true)
    Aop = LinearMap(A)
    rng = Xoshiro(12345)
    t0 = time()

    # Unseeded starting point: (P + δI + AᵀA) x = −q + Aᵀt, t the row midpoints.
    x = cholesky(Symmetric(P + δ * I + A' * A)) \ (-q + A' * ((l + u) / 2))
    Ax = A * x
    sl = Ax - l
    su = u - Ax
    θ = max(0.0, -1.5 * min(minimum(sl), minimum(su)))
    sl .+= θ
    su .+= θ
    zl = ones(m)
    zu = ones(m)
    sz = dot(sl, zl) + dot(su, zu)
    δs = 0.5 * sz / (sum(zl) + sum(zu))
    δz = 0.5 * sz / (sum(sl) + sum(su))
    sl .+= δs
    su .+= δs
    zl .+= δz
    zu .+= δz
    Ns = 2m

    iters = Any[]
    hits = Dict{String, Any}()
    status = "max_iter"
    fails = 0
    small_steps = 0
    for k in 0:MAX_ITER
        Ax = A * x
        y = zu - zl
        for ϵ in EPS_LIST
            key = string(ϵ)
            haskey(hits, key) && continue
            if termination(inst, x, y, Ax, ϵ).pass
                hits[key] = Dict(
                    "iter" => k, "referee" => referee(inst, x, y),
                    "obj" => objective(inst, x), "xerr" => norm(x - inst.xstar, Inf),
                    "time" => time() - t0,
                )
            end
        end
        if haskey(hits, string(last(EPS_LIST)))
            status = "solved"
            break
        end
        k == MAX_ITER && break
        if !all(isfinite, x) || !all(isfinite, zl) || !all(isfinite, zu)
            status = "nan"
            break
        end

        rd = P * x + q + A' * y
        rl = Ax - l - sl
        ru = u - Ax - su
        μ = (dot(sl, zl) + dot(su, zu)) / Ns
        rnorm = max(norm(rd, Inf), norm(rl, Inf), norm(ru, Inf))
        W = zl ./ sl + zu ./ su
        # `:perside` regularizes each one-sided row, `AΔx − Δs_l + δΔz_l = −r_l` (and the
        # upper row likewise), so each side's weight is `z/(s + δz)`. The other two use the
        # harmonic form `w = 1/(1/W + δ)`.
        dl = recovery === :perside ? sl .+ δ .* zl : sl
        du = recovery === :perside ? su .+ δ .* zu : su
        w = recovery === :perside ? zl ./ dl + zu ./ du : 1 ./ (1 ./ W .+ δ)
        winv = 1 ./ w
        level = min(μ, rnorm)

        F = inner === :exact ? bunchkaufman!(Symmetric([P + δ * I A'; A -Diagonal(winv)], :U)) : nothing
        op = reduced_map(Pop, Aop, w, δ)
        pcs = battery ? BATTERY : (inner === :cg ? (precond,) : ())
        hutch = Any[]
        Ms = Dict(pc => build_precond(pc, inst, Pop, Aop, w, winv, δ, rng; stats = hutch) for pc in pcs)

        rec = Dict{String, Any}(
            "k" => k, "mu" => μ, "rnorm" => rnorm, "rd_norm" => norm(rd, Inf),
            "rlu_norm" => max(norm(rl, Inf), norm(ru, Inf)), "xerr" => norm(x - inst.xstar, Inf),
            "W_min" => minimum(W), "W_median" => median(W), "W_max" => maximum(W),
            "w_max" => maximum(w), "n_w_saturated" => count(>=(0.5 / δ), w),
            "n_W_gt_1e2" => count(>(1.0e2), W), "hutchinson" => hutch,
            "solves" => Any[],
        )
        solve_failed = false
        direction = function (rcl, rcu, label)
            g = (rcl + zl .* rl) ./ dl - (rcu + zu .* ru) ./ du
            b = -rd - A' * g
            atol = max(ETA * level, eps() * max(1.0, norm(b, Inf)))
            srec = Dict{String, Any}("which" => label, "atol" => atol, "rhs_norm2" => norm(b))
            if inner === :exact
                dx = (F \ [-rd; -winv .* g])[1:n]
                srec["exact_reduced_res2"] = norm(b - op * dx)
            else
                r = cg_solve(op, Ms[precond], b, atol, CG_CAP)
                dx = r.x
                srec["cg"] = Dict("iters" => r.iters, "reached" => r.reached, "res2" => r.res, "breakdown" => r.breakdown)
                if r.reached
                    fails = 0
                else
                    fails += 1
                    fails >= CG_FAIL_LIMIT && (solve_failed = true)
                end
            end
            if battery
                bat = Dict{String, Any}()
                for pc in BATTERY
                    r = cg_solve(op, Ms[pc], b, atol, CG_CAP)
                    e = Dict{String, Any}(
                        "iters" => r.iters, "reached" => r.reached, "res2" => r.res, "breakdown" => r.breakdown,
                        "dx_relerr" => norm(r.x - dx) / max(norm(dx), eps()),
                    )
                    if r.iters > CG_DEFAULT_ITER && pc in RESTART_PRECONDS
                        rr = cg_solve(op, Ms[pc], b, atol, CG_CAP; restart = CG_DEFAULT_ITER)
                        e["restarted"] = Dict("iters" => rr.iters, "reached" => rr.reached, "res2" => rr.res)
                    end
                    bat[string(pc)] = e
                end
                srec["battery"] = bat
            end
            push!(rec["solves"], srec)
            # `:design` recovers the slack steps from `AΔx`, so `Δz_u − Δz_l = W AΔx + g`,
            # which differs from the solved `Δy = w AΔx + g` by `(W − w) AΔx`. `:consistent`
            # uses `v = AΔx / (1 + δW)`, for which `Δz_u − Δz_l = W v + g = w AΔx + g`.
            # `:perside` gives `Δz_u − Δz_l = w AΔx + g` directly.
            Adx = A * dx
            recovery === :consistent && (Adx ./= 1 .+ δ .* W)
            if recovery === :perside
                dzl = -(rcl + zl .* (Adx + rl)) ./ dl
                dzu = -(rcu + zu .* (-Adx + ru)) ./ du
                dsl = Adx + rl + δ .* dzl
                dsu = -Adx + ru + δ .* dzu
            else
                dsl = Adx + rl
                dsu = -Adx + ru
                dzl = -(rcl + zl .* dsl) ./ sl
                dzu = -(rcu + zu .* dsu) ./ su
            end
            return dx, dsl, dsu, dzl, dzu
        end

        # Predictor
        dx, dsl, dsu, dzl, dzu = direction(sl .* zl, su .* zu, "predictor")
        αa = min(1.0, max_step(sl, dsl), max_step(su, dsu), max_step(zl, dzl), max_step(zu, dzu))
        μa = (dot(sl + αa * dsl, zl + αa * dzl) + dot(su + αa * dsu, zu + αa * dzu)) / Ns
        σ = (μa / μ)^3
        # With `safeguard`, the centering target stays above a tenth of the Newton residual,
        # so μ is not driven to underflow while the residuals are still large.
        safeguard && (σ = max(σ, min(1.0, 0.1 * rnorm / μ)))
        # Corrector
        rcl = sl .* zl + dsl .* dzl .- σ * μ
        rcu = su .* zu + dsu .* dzu .- σ * μ
        dx, dsl, dsu, dzl, dzu = direction(rcl, rcu, "corrector")
        αmax = min(max_step(sl, dsl), max_step(su, dsu), max_step(zl, dzl), max_step(zu, dzu))
        α = min(1.0, TAU * αmax)
        x .+= α .* dx
        sl .+= α .* dsl
        su .+= α .* dsu
        zl .+= α .* dzl
        zu .+= α .* dzu
        rec["alpha_aff"] = αa
        rec["alpha"] = α
        rec["sigma"] = σ
        push!(iters, rec)
        if solve_failed
            status = "cg_fail"
            break
        end
        small_steps = α < 1.0e-8 ? small_steps + 1 : 0
        if small_steps >= 3
            status = "stall"
            break
        end
    end
    y = zu - zl
    return Dict{String, Any}(
        "delta" => δ, "inner" => string(inner), "precond" => string(precond),
        "recovery" => string(recovery), "safeguard" => safeguard, "status" => status,
        "outer_iters" => length(iters), "hits" => hits, "final_referee" => referee(inst, x, y),
        "final_obj" => objective(inst, x), "time" => time() - t0, "iters" => iters,
    )
end

# ---------------------------------------------------------------------------------------------
# Driver

function run_instance(n, κ, frac)
    seed = hash((n, κ, frac)) % UInt32
    inst = make_instance(n, κ, frac; seed)
    sv = svdvals(inst.A)
    ref = clarabel_reference(inst)
    obj_star = objective(inst, inst.xstar)
    out = Dict{String, Any}(
        "n" => n, "m" => n, "kappa" => κ, "active_fraction" => frac, "nactive" => inst.nactive,
        "seed" => Int(seed), "cond_A_measured" => sv[1] / sv[end], "obj_star" => obj_star,
        "referee_star" => referee(inst, inst.xstar, inst.ystar),
        "clarabel" => Dict(
            "status" => ref.status, "obj" => ref.obj, "time" => ref.time,
            "xerr" => norm(ref.x - inst.xstar, Inf),
        ),
        "variants" => Any[], "exact" => Any[], "cg" => Any[],
    )
    for δ in DELTAS, (recovery, safeguard) in VARIANTS
        r = ipm(inst, δ; recovery, safeguard)
        r["iters"] = [Dict(k => v for (k, v) in it if k != "solves") for it in r["iters"]]
        push!(out["variants"], r)
        @printf("n=%d κ=%.0e f=%.1f δ=%.0e %s/%s: %s in %d (%.1fs)\n", n, κ, frac, δ, recovery, safeguard, r["status"], r["outer_iters"], r["time"])
        flush(stdout)
    end
    for δ in DELTAS
        r = ipm(inst, δ; battery = true, PRIMARY...)
        push!(out["exact"], r)
        @printf("n=%d κ=%.0e f=%.1f δ=%.0e exact: %s in %d (%.1fs)\n", n, κ, frac, δ, r["status"], r["outer_iters"], r["time"])
        flush(stdout)
    end
    for δ in DELTAS, pc in PRECONDS
        r = ipm(inst, δ; inner = :cg, precond = pc, PRIMARY...)
        push!(out["cg"], r)
        @printf("n=%d κ=%.0e f=%.1f δ=%.0e cg/%s: %s in %d (%.1fs)\n", n, κ, frac, δ, pc, r["status"], r["outer_iters"], r["time"])
        flush(stdout)
    end
    return out
end

"""
    g2(per_outer) -> (first3, last3, ratio)

Median CG iterations per solve over the solves of the first three and the last three outer
iterations.
"""
function g2(per_outer)
    length(per_outer) < 6 && return (NaN, NaN, NaN)
    f = median(reduce(vcat, per_outer[1:3]))
    t = median(reduce(vcat, per_outer[(end - 2):end]))
    return (f, t, t / f)
end

"Gate summaries per instance: G2 on the exact trajectory and on each inexact run, and G1."
function summarize(inst)
    rows = Any[]
    for ex in inst["exact"], pc in BATTERY
        per_outer = [[s["battery"][string(pc)]["iters"] for s in it["solves"]] for it in ex["iters"]]
        all_solves = reduce(vcat, per_outer; init = Int[])
        entries = [s["battery"][string(pc)] for it in ex["iters"] for s in it["solves"]]
        reached = [e["reached"] for e in entries]
        restarted = filter(e -> haskey(e, "restarted"), entries)
        f, t, ratio = g2(per_outer)
        push!(
            rows, Dict(
                "restart_nsolves" => length(restarted),
                "restart_iters_plain" => sum((e["iters"] for e in restarted); init = 0),
                "restart_iters_restarted" => sum((e["restarted"]["iters"] for e in restarted); init = 0),
                "restart_reached_plain" => count(e -> e["reached"], restarted),
                "restart_reached_restarted" => count(e -> e["restarted"]["reached"], restarted),
                "source" => "exact_trajectory", "delta" => ex["delta"], "precond" => string(pc),
                "outer_iters" => ex["outer_iters"], "status" => ex["status"],
                "g2_first3" => f, "g2_last3" => t, "g2_ratio" => ratio, "g2_pass" => ratio <= 10,
                "cg_median" => median(all_solves), "cg_max" => maximum(all_solves),
                "frac_reached" => mean(reached), "frac_over_default" => mean(all_solves .> CG_DEFAULT_ITER),
            )
        )
    end
    for r in inst["cg"]
        per_outer = [[s["cg"]["iters"] for s in it["solves"]] for it in r["iters"]]
        all_solves = reduce(vcat, per_outer; init = Int[])
        f, t, ratio = g2(per_outer)
        hit6 = get(r["hits"], string(1.0e-6), nothing)
        push!(
            rows, Dict(
                "source" => "inexact_run", "delta" => r["delta"], "precond" => r["precond"],
                "outer_iters" => r["outer_iters"], "status" => r["status"],
                "g2_first3" => f, "g2_last3" => t, "g2_ratio" => ratio, "g2_pass" => ratio <= 10,
                "cg_median" => isempty(all_solves) ? NaN : median(all_solves),
                "cg_total" => sum(all_solves),
                "g1_iter" => isnothing(hit6) ? nothing : hit6["iter"],
                "g1_referee" => isnothing(hit6) ? nothing : hit6["referee"],
                "g1_pass" => !isnothing(hit6) && hit6["referee"] <= 1.0e-5,
            )
        )
    end
    return rows
end

# ---------------------------------------------------------------------------------------------
# Compact summary

"""
    compact_row(; method, delta, status, outer_iters, hits, final_referee,
                per_outer = nothing, reached = nothing, products = nothing) -> Dict

One row per instance × method × δ: status, outer iterations, final referee, G1 (the first
tolerance of `EPS_LIST` reached with referee ≤ 1e-5), and, when inner solves are given, the G2
ratio of `g2`, the median and maximum inner iterations per solve, the total products and the
fraction of solves that reached the inner tolerance. `per_outer[k]` holds the inner iteration
counts of outer iteration `k`'s solves; `reached` has one entry per solve.
"""
function compact_row(;
        method, delta, status, outer_iters, hits, final_referee,
        per_outer = nothing, reached = nothing, products = nothing,
    )
    hit = get(hits, string(first(EPS_LIST)), nothing)
    row = Dict{String, Any}(
        "method" => method, "delta" => delta, "status" => status, "converged" => status == "solved",
        "outer_iters" => outer_iters, "final_referee" => final_referee,
        "g1_iter" => isnothing(hit) ? nothing : hit["iter"],
        "g1_referee" => isnothing(hit) ? nothing : hit["referee"],
        "g1_pass" => !isnothing(hit) && hit["referee"] <= 1.0e-5,
    )
    if !isnothing(per_outer)
        solves = reduce(vcat, per_outer; init = Int[])
        _, _, ratio = g2(per_outer)
        row["g2_ratio"] = ratio
        row["g2_pass"] = ratio <= 10
        row["inner_median"] = isempty(solves) ? nothing : median(solves)
        row["inner_max"] = isempty(solves) ? nothing : maximum(solves)
        row["frac_reached"] = isempty(reached) ? nothing : mean(reached)
        row["products"] = products
    end
    return row
end

"""
    products_first_spike(pc, cg_iters, outer_iters)

Products of a first-spike CG run, derived rather than counted: three per CG iteration (`A`,
`Aᵀ`, `P`) plus the preconditioner build once per outer iteration (`k` adjoint products and
three per Hutchinson probe for probed Woodbury, `k` for the exact-diagonal variant, none for
Jacobi, which reads the dense matrices). The residual recomputations of `cg_solve` are
instrumentation and are not counted.
"""
function products_first_spike(pc, cg_iters, outer_iters)
    build = pc == "woodbury5" ? 5 + 3PROBES : pc == "woodbury20" ? 20 + 3PROBES :
        pc == "woodbury20_exactdiag" ? 20 : 0
    return 3cg_iters + build * outer_iters
end

"""
    write_summary(doc, path)

Compact summary of a first-spike document (as returned by `main` or read back from its JSON):
per instance, one `compact_row` per exact variant (`exact/<recovery>[+sigmafloor]`), per
preconditioner measured on the exact trajectory (`battery/<pc>`, outer statistics of that exact
run) and per inexact run (`cg/<pc>`).
"""
function write_summary(doc, path)
    instances = Any[]
    for inst in doc["instances"]
        haskey(inst, "error") && continue
        rows = Any[]
        common(r) = (;
            delta = r["delta"], status = r["status"], outer_iters = r["outer_iters"],
            hits = r["hits"], final_referee = r["final_referee"],
        )
        for r in inst["variants"]
            push!(rows, compact_row(; method = "exact/" * r["recovery"] * (r["safeguard"] ? "+sigmafloor" : ""), common(r)...))
        end
        for r in inst["exact"], pc in doc["settings"]["battery"]
            per_outer = [[s["battery"][pc]["iters"] for s in it["solves"]] for it in r["iters"]]
            reached = [s["battery"][pc]["reached"] for it in r["iters"] for s in it["solves"]]
            products = products_first_spike(pc, sum(sum, per_outer; init = 0), r["outer_iters"])
            push!(rows, compact_row(; method = "battery/" * pc, per_outer, reached, products, common(r)...))
        end
        for r in inst["cg"]
            per_outer = [[s["cg"]["iters"] for s in it["solves"]] for it in r["iters"]]
            reached = [s["cg"]["reached"] for it in r["iters"] for s in it["solves"]]
            products = products_first_spike(r["precond"], sum(sum, per_outer; init = 0), r["outer_iters"])
            push!(rows, compact_row(; method = "cg/" * r["precond"], per_outer, reached, products, common(r)...))
        end
        push!(
            instances, Dict(
                "family" => "dense", "n" => inst["n"], "kappa" => inst["kappa"],
                "active_fraction" => inst["active_fraction"], "rows" => rows,
            )
        )
    end
    out = Dict(
        "description" => "Compact summary of PureIPM/bench/results/ipm_matrixfree_spike.json (first matrix-free IPM spike)",
        "products_note" => "derived, not counted: 3 per CG iteration + preconditioner build per outer iteration (Woodbury k + 3*probes; exact-diagonal k; Jacobi 0)",
        "g1" => "eps = $(first(EPS_LIST)) reached with referee <= 1e-5",
        "g2" => "median inner iterations per solve, last 3 outer iterations / first 3, pass <= 10 (NaN if fewer than 6 outer iterations)",
        "instances" => instances,
    )
    open(io -> JSON.json(io, out; allownan = true), path, "w")
    return out
end

function main(; ns = NS, kappas = KAPPAS, fracs = FRACS, path = joinpath(@__DIR__, "results", "ipm_matrixfree_spike.json"))
    BLAS.set_num_threads(1)
    cases = sort!(vec([(n, κ, f) for n in ns, κ in kappas, f in fracs]); by = c -> -c[1])
    results = Vector{Any}(undef, length(cases))
    # A failing instance is recorded and rethrown after the JSON is written, so the other
    # instances' samples are kept.
    failures = Any[]
    failures_lock = ReentrantLock()
    Threads.@threads :dynamic for i in eachindex(cases)
        try
            results[i] = run_instance(cases[i]...)
        catch e
            results[i] = Dict("case" => collect(cases[i]), "error" => sprint(showerror, e, catch_backtrace()))
            lock(() -> push!(failures, (cases[i], e)), failures_lock)
            println(stderr, "FAILED ", cases[i], ": ", sprint(showerror, e))
            flush(stderr)
        end
    end
    for r in results
        haskey(r, "error") || (r["summary"] = summarize(r))
    end
    doc = Dict(
        "description" => "Matrix-free IPM spike: dense Mehrotra prototype, CG on the reduced system per outer iteration",
        "julia" => string(VERSION), "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "settings" => Dict(
            "eps_list" => collect(EPS_LIST), "max_iter" => MAX_ITER, "cg_cap" => CG_CAP,
            "cg_default_iter" => CG_DEFAULT_ITER, "cg_fail_limit" => CG_FAIL_LIMIT, "eta" => ETA,
            "probes" => PROBES, "tau" => TAU, "deltas" => collect(DELTAS),
            "preconds" => string.(collect(PRECONDS)), "battery" => string.(collect(BATTERY)),
            "restart_preconds" => string.(collect(RESTART_PRECONDS)),
            "variants" => [Dict("recovery" => string(r), "safeguard" => s) for (r, s) in VARIANTS],
            "primary" => Dict("recovery" => string(PRIMARY.recovery), "safeguard" => PRIMARY.safeguard),
            "start_point" => "exact Cholesky solve in every run", "m" => "m = n, all rows two-sided finite",
            "P_eigs" => "log-spaced 1 .. 1e-2", "A_singular_values" => "log-spaced 1 .. 1/kappa",
            "cg_stopping" => "unpreconditioned ||b - K dx||_2 <= atol_k",
        ),
        "instances" => results,
    )
    mkpath(dirname(path))
    open(io -> JSON.json(io, doc; allownan = true), path, "w")
    write_summary(doc, replace(path, r"\.json$" => "_summary.json"))
    isempty(failures) || error("$(length(failures)) instance(s) failed; see \"error\" entries in $path")
    return doc
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
