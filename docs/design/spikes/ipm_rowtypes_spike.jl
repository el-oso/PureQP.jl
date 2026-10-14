# Matrix-free IPM spike 3: row classes, the realizable inner stopping test, CG start, and
# quasi-definite LDLᵀ in place of Bunch–Kaufman
#
# Reuses spike 2's instances (`ipm_matrixfree_spike2.jl`: dense `make_instance`, sparse family)
# and generalizes its Mehrotra prototype to the row classes of the design document
# (docs/design/modular-ipm.md): equality rows (`w_inv = δ`, `rhs_z = −r_e`), lower-only and
# upper-only rows (absent side masked, placeholders `s = 1`, `z = 0`), free rows
# (`w_inv = 1/δ`, `rhs_z = 0`, `Δy` zeroed), with spike 2's per-side recovery and σ floor.
#
# Row mixes (`MIXES`), applied to a spike-2 instance:
#   spike      the spike-2 instance unchanged (every row two-sided)
#   eq20       20% equality rows, the rest two-sided, re-planted
#   mixed      20% equality, 20% lower-only, 20% upper-only, 10% free, 30% two-sided, re-planted
#   eqonly     the first n/2 rows of A as equality rows, nothing else (N_s = 0)
# The spike-2 active fraction applies to the inequality rows.
#
# `N_s = 0` rule: μ ≡ 0, the tolerance level is ‖r‖∞, one solve per outer iteration (no
# corrector, no σ), α = 1, starting point = the w = 1 solve with y = 0 and no slack steps.
#
# Inner solvers: `exact` (bunchkaufman! on the KKT matrix), `exact_cholmod` (CHOLMOD `ldlt`,
# no pivoting, fill-reducing ordering: what `SparseKKT` runs), `exact_ldlf` (LDLFactorizations
# `ldl`: what `LDLKKT` runs), `exact_redchol` (Cholesky of the reduced matrix, dense LAPACK or
# CHOLMOD: what the reduced backends run), and CG on the reduced matrix with `none`, `lagchol3` (Cholesky of
# the reduced matrix at outer iterations 0, 3, 6, …) or `lldl10` (limited-memory LDLᵀ every
# iteration, spike 2's shift search). A `_ref1` suffix on a KKT solver adds one refinement step;
# no other solver refines.
#
# CG stopping (`stop`):
#   oracle    ‖b − KΔx‖₂ ≤ atol on the explicit reduced matrix every iteration (spike 2)
#   design    callback on Krylov's recursive residual ‖r‖₂ ≤ atol, one recomputation through
#             the operator after the solve, reached = recomputed ≤ 2·atol
#   revised   callback on ‖r‖₂ ≤ atol; recompute; while recomputed > atol and budget remains,
#             warm-restart CG from the current point (resetting the recursion); reached =
#             recomputed ≤ atol (spike 1's `cg_solve`)
#   capfail   as `design`, but a solve counts as missed only when it spends the whole budget
#             or breaks down; the recomputed residual is recorded, not tested
# Every solve records `res_over_floor`: the recomputed residual over `attainable`'s rounding scale.
# CG start (`start`): `zero`, or `prev` (the run's previous solve: the predictor's Δx_a for the
# corrector, the previous corrector's Δx for the next predictor).
#
# Run from the repository root:
#     julia +1.12 -t 6 --project=bench PureIPM/bench/ipm_rowtypes_spike.jl
# Writes PureIPM/bench/results/ipm_rowtypes_spike_raw.json (per-run records) and
# PureIPM/bench/results/ipm_rowtypes_spike.json (aggregated summary).

include(joinpath(@__DIR__, "ipm_matrixfree_spike2.jl"))
using LDLFactorizations

const MIXES = (:spike, :eq20, :mixed, :eqonly)
const FREE, INEQ, EQ = Int8(-1), Int8(0), Int8(1)

struct RowInst{MP, MA}
    family::Symbol
    label::Dict{String, Any}
    n::Int
    m::Int
    P::MP
    q::Vector{Float64}
    A::MA
    l::Vector{Float64}
    u::Vector{Float64}
    cls::Vector{Int8}
    hl::BitVector
    hu::BitVector
end

"Row classes of `mix` on the spike-2 instance `base`, re-planted so `(x⋆, y⋆)` stays optimal."
function rowtype_instance(base::Inst2, mix::Symbol)
    label = merge(base.label, Dict{String, Any}("mix" => string(mix)))
    n = base.n
    if mix === :spike
        m = length(base.l)
        return RowInst(base.family, label, n, m, base.P, base.q, base.A, base.l, base.u, fill(INEQ, m), trues(m), trues(m))
    end
    rng = Xoshiro(hash((base.label, mix)) % UInt32)
    A = mix === :eqonly ? base.A[1:(n ÷ 2), :] : base.A
    m = size(A, 1)
    xstar = randn(rng, n)
    a = A * xstar
    kind = fill(:two, m)
    if mix === :eqonly
        fill!(kind, :eq)
    else
        fracs = mix === :eq20 ? ((:eq, 0.2),) : ((:eq, 0.2), (:lo, 0.2), (:up, 0.2), (:free, 0.1))
        p = randperm(rng, m)
        j = 0
        for (k, f) in fracs, _ in 1:round(Int, f * m)
            kind[p[j += 1]] = k
        end
    end
    l = fill(-Inf, m)
    u = fill(Inf, m)
    ystar = zeros(m)
    cls = fill(INEQ, m)
    for i in 1:m
        gap = 0.5 + rand(rng)
        mag = 0.5 + rand(rng)
        active = rand(rng) < base.frac
        k = kind[i]
        if k === :eq
            cls[i] = EQ
            l[i] = u[i] = a[i]
            ystar[i] = rand(rng, Bool) ? mag : -mag
        elseif k === :free
            cls[i] = FREE
        elseif k === :lo
            active ? (l[i] = a[i]; ystar[i] = -mag) : (l[i] = a[i] - gap)
        elseif k === :up
            active ? (u[i] = a[i]; ystar[i] = mag) : (u[i] = a[i] + gap)
        else
            l[i] = a[i] - gap
            u[i] = a[i] + 0.5 + rand(rng)
            if active
                rand(rng, Bool) ? (u[i] = a[i]; ystar[i] = mag) : (l[i] = a[i]; ystar[i] = -mag)
            end
        end
    end
    q = -(base.P * xstar + A' * ystar)
    hl = BitVector(cls[i] == INEQ && isfinite(l[i]) for i in 1:m)
    hu = BitVector(cls[i] == INEQ && isfinite(u[i]) for i in 1:m)
    return RowInst(base.family, label, n, m, base.P, q, A, l, u, cls, hl, hu)
end

"`sup_{z ∈ [l,u]} yᵀz` over the finite bounds (an infinite bound carries `y = 0`)."
support(l, u, y) = sum((isfinite(u[i]) ? u[i] * max(y[i], 0.0) : 0.0) + (isfinite(l[i]) ? l[i] * min(y[i], 0.0) : 0.0) for i in eachindex(y))

function termination3(inst, x, y, Ax, ϵ)
    z = clamp.(Ax, inst.l, inst.u)
    Px = inst.P * x
    Aty = inst.A' * y
    prim = maximum(abs, Ax .- z)
    dual = maximum(abs, Px .+ inst.q .+ Aty)
    quad, lin, sup = dot(x, Px), dot(inst.q, x), support(inst.l, inst.u, y)
    gap = abs(quad + lin + sup)
    pass = prim <= ϵ + ϵ * max(norm(Ax, Inf), norm(z, Inf)) &&
        dual <= ϵ + ϵ * max(norm(Px, Inf), norm(Aty, Inf), norm(inst.q, Inf)) &&
        gap <= ϵ + ϵ * max(abs(quad), abs(lin), abs(sup))
    return (; prim, dual, gap_rel = gap / max(1.0, abs(quad), abs(lin), abs(sup)), pass)
end

referee3(inst, x, y) = (t = termination3(inst, x, y, inst.A * x, 0.0); max(t.prim, t.dual, t.gap_rel))

function ipm3(inst::RowInst, δ, method; stop = :oracle, start = :zero)
    P, q, A, l, u, cls, hl, hu = inst.P, inst.q, inst.A, inst.l, inst.u, inst.cls, inst.hl, inst.hu
    n, m = inst.n, inst.m
    eq = cls .== EQ
    fr = cls .== FREE
    ineq = cls .== INEQ
    Ns = count(hl) + count(hu)
    c = Counter(0, 0, 0, 0)
    Pop, Aop = LinearMap(P; issymmetric = true), LinearMap(A)
    Pc = LinearMap{Float64}((y, x) -> (c.P += 1; mul!(y, Pop, x)), n; ismutating = true, issymmetric = true)
    Ac = LinearMap{Float64}((y, x) -> (c.A += 1; mul!(y, Aop, x)), (y, x) -> (c.At += 1; mul!(y, Aop', x)), m, n; ismutating = true)
    ms = string(method)
    state = Dict{Symbol, Any}()
    cgws = CgWorkspace(n, n, Vector{Float64})
    xprev = zeros(n)
    tmp = zeros(n)
    Anorm = opnorm(Matrix(A))
    t0 = time()

    t = [eq[i] ? l[i] : fr[i] ? 0.0 : hl[i] && hu[i] ? (l[i] + u[i]) / 2 : hl[i] ? l[i] : u[i] for i in 1:m]
    x = cholesky(Symmetric(Matrix(P + δ * I + A' * A))) \ (-q + A' * t)
    Ax = A * x
    sl = [hl[i] ? Ax[i] - l[i] : 1.0 for i in 1:m]
    su = [hu[i] ? u[i] - Ax[i] : 1.0 for i in 1:m]
    zl = Float64.(hl)
    zu = Float64.(hu)
    yE = zeros(m)
    if Ns > 0
        θ = max(0.0, -1.5 * min(minimum(sl[hl]; init = Inf), minimum(su[hu]; init = Inf)))
        sl[hl] .+= θ
        su[hu] .+= θ
        sz = dot(sl, zl) + dot(su, zu)
        δs = 0.5 * sz / (sum(zl) + sum(zu))
        δz = 0.5 * sz / (sum(sl[hl]) + sum(su[hu]))
        sl[hl] .+= δs
        su[hu] .+= δs
        zl[hl] .+= δz
        zu[hu] .+= δz
    end

    solves = Any[]
    iters = Any[]
    hits = Dict{String, Any}()
    status = "max_iter"
    fails = 0
    for k in 0:MAX_ITER
        Ax = A * x
        y = (zu .- zl) .* ineq .+ yE .* eq
        for ϵ in EPS_LIST
            key = string(ϵ)
            haskey(hits, key) && continue
            termination3(inst, x, y, Ax, ϵ).pass && (hits[key] = Dict("iter" => k, "referee" => referee3(inst, x, y)))
        end
        haskey(hits, string(last(EPS_LIST))) && (status = "solved"; break)
        k == MAX_ITER && break
        all(isfinite, x) && all(isfinite, zl) && all(isfinite, zu) || (status = "nan"; break)

        rd = P * x + q + A' * y
        rl = (Ax .- l .- sl) .* hl
        ru = (u .- Ax .- su) .* hu
        re = (Ax .- l) .* eq
        # Masked-out entries of l, u are infinite; zero them before they meet the mask.
        rl[.!hl] .= 0.0
        ru[.!hu] .= 0.0
        re[.!eq] .= 0.0
        μ = Ns > 0 ? (dot(sl, zl) + dot(su, zu)) / Ns : 0.0
        rnorm = max(norm(rd, Inf), norm(rl, Inf), norm(ru, Inf), norm(re, Inf))
        dl = sl .+ δ .* zl
        du = su .+ δ .* zu
        w = [eq[i] ? 1 / δ : fr[i] ? δ : zl[i] / dl[i] + zu[i] / du[i] for i in 1:m]
        level = Ns > 0 ? min(μ, rnorm) : rnorm
        K = reduced_matrix(P, A, w, δ)
        if startswith(ms, "exact_redchol")
            F = cholesky(K isa Symmetric ? K : Symmetric(K, :U); check = false)
            issuccess(F) || (status = "factor_fail"; break)
            state[:F] = F
        elseif startswith(ms, "exact")
            winv = [eq[i] ? δ : fr[i] ? 1 / δ : 1 / w[i] for i in 1:m]
            Kkkt = [sparse(P) + δ * I sparse(A)'; sparse(A) -Diagonal(winv)]
            state[:F] = method === :exact ? bunchkaufman!(Symmetric(Matrix(Kkkt), :U)) :
                startswith(ms, "exact_cholmod") ? ldlt(Symmetric(Kkkt, :U); check = false) :
                ldl(Symmetric(triu(Kkkt), :U))
            state[:winv] = winv
            state[:Kkkt] = Kkkt
            if startswith(ms, "exact_cholmod") && !issuccess(state[:F])
                status = "factor_fail"
                break
            end
        elseif method === :cg_none
            state[:M] = I
        elseif method === :cg_lagchol3
            if iszero(k % 3)
                F = cholesky(K; check = false)
                issuccess(F) ? (state[:M] = F) : (state[:refresh_failures] = get(state, :refresh_failures, 0) + 1)
            end
            haskey(state, :M) || (state[:M] = I)
        elseif method === :cg_lldl10
            state[:M], _ = lldl_spd(K, 10)
        else
            error("unknown method $method")
        end

        solve_failed = false
        direction = function (rcl, rcu, label)
            g = [eq[i] ? re[i] / δ : fr[i] ? 0.0 : (rcl[i] + zl[i] * rl[i]) / dl[i] - (rcu[i] + zu[i] * ru[i]) / du[i] for i in 1:m]
            b = -rd - A' * g
            atol = max(ETA * level, eps() * max(1.0, norm(b, Inf)))
            rec = Dict{String, Any}("which" => label, "k" => k, "atol" => atol)
            if startswith(ms, "exact_redchol")
                dx = state[:F] \ b
                endswith(ms, "_ref1") && (dx .+= state[:F] \ (b - K * dx))
                rec["iters"] = 0
                res = norm(b - K * dx)
                rec["res_over_atol"] = res / atol
                rec["res_over_floor"] = res / attainable(P, A, Anorm, w, δ, b, dx)
                rec["reached"] = true
            elseif startswith(ms, "exact")
                rhs = [-rd; -state[:winv] .* g]
                sol = state[:F] \ rhs
                # `_ref1`: one refinement step against the regularized KKT matrix.
                endswith(ms, "_ref1") && (sol .+= state[:F] \ (rhs - state[:Kkkt] * sol))
                dx = sol[1:n]
                rec["iters"] = 0
                res = norm(b - K * dx)
                rec["res_over_atol"] = res / atol
                rec["res_over_floor"] = res / attainable(P, A, Anorm, w, δ, b, dx)
                rec["reached"] = true
            else
                dx = cg_inner!(rec, cgws, reduced_map(Pc, Ac, w, δ), state[:M], b, atol, K, xprev, tmp; stop, start, floor = dxv -> attainable(P, A, Anorm, w, δ, b, dxv))
                if rec["reached"]
                    fails = 0
                else
                    fails += 1
                    fails >= CG_FAIL_LIMIT && (solve_failed = true)
                end
            end
            push!(solves, rec)
            Adx = A * dx
            dy = (w .* Adx .+ g) .* .!fr
            dzl = (-(rcl .+ zl .* (Adx .+ rl)) ./ dl) .* hl
            dzu = (-(rcu .+ zu .* (-Adx .+ ru)) ./ du) .* hu
            dsl = (Adx .+ rl .+ δ .* dzl) .* hl
            dsu = (-Adx .+ ru .+ δ .* dzu) .* hu
            return dx, dy, dsl, dsu, dzl, dzu
        end

        if Ns > 0
            dx, dy, dsl, dsu, dzl, dzu = direction(sl .* zl, su .* zu, "predictor")
            αa = min(1.0, max_step(sl, dsl), max_step(su, dsu), max_step(zl, dzl), max_step(zu, dzu))
            μa = (dot(sl + αa * dsl, zl + αa * dzl) + dot(su + αa * dsu, zu + αa * dzu)) / Ns
            σ = max((μa / μ)^3, min(1.0, 0.1 * rnorm / μ))
            rcl = (sl .* zl + dsl .* dzl .- σ * μ) .* hl
            rcu = (su .* zu + dsu .* dzu .- σ * μ) .* hu
            dx, dy, dsl, dsu, dzl, dzu = direction(rcl, rcu, "corrector")
            α = min(1.0, TAU * min(max_step(sl, dsl), max_step(su, dsu), max_step(zl, dzl), max_step(zu, dzu)))
        else
            dx, dy, dsl, dsu, dzl, dzu = direction(zeros(m), zeros(m), "newton")
            α = 1.0
        end
        x .+= α .* dx
        yE .+= α .* dy .* eq
        sl .+= α .* dsl
        su .+= α .* dsu
        zl .+= α .* dzl
        zu .+= α .* dzu
        push!(iters, Dict{String, Any}("k" => k, "mu" => μ, "rnorm" => rnorm, "alpha" => α, "w_max" => maximum(w)))
        solve_failed && (status = "abort_cap"; break)
    end
    y = (zu .- zl) .* ineq .+ yE .* eq
    return Dict{String, Any}(
        "delta" => δ, "method" => ms, "stop" => string(stop), "start" => string(start), "status" => status,
        "outer_iters" => length(iters), "hits" => hits, "final_referee" => referee3(inst, x, y), "Ns" => Ns,
        "time" => time() - t0, "products" => products(c), "refresh_failures" => get(state, :refresh_failures, 0),
        "iters" => iters, "solves" => solves,
    )
end

"""
    attainable(P, A, Anorm, w, δ, b, dx)

Rounding scale of the reduced residual `b − (P + δI + Aᵀdiag(w)A)Δx`:
`eps·(‖b‖ + ‖PΔx‖ + δ‖Δx‖ + ‖A‖₂‖w ⊙ AΔx‖)`, the size of the terms before they cancel. Every
term but `‖A‖₂` is a by-product of one application of the operator.
"""
function attainable(P, A, Anorm, w, δ, b, dx)
    return eps() * (norm(b) + norm(P * dx) + δ * norm(dx) + Anorm * norm(w .* (A * dx)))
end

"""
One CG solve of `op Δx = b` under the stopping rule `stop` and the start `start`; fills `rec`
(`iters`, `reached`, `res_over_atol`, `res_over_floor`, `restarts`, `krylov_self_stops`,
`breakdown`) and returns `Δx`. `floor(Δx)` is the rounding scale of the residual (`attainable`).
"""
function cg_inner!(rec, ws, op, M, b, atol, K, xprev, tmp; stop, start, floor)
    n = length(b)
    base = start === :prev ? copy(xprev) : zeros(n)
    warm = any(!iszero, base)
    fired = Ref(false)
    # While a warm start runs, `ws.x` holds the correction; the iterate is `base + ws.x`.
    cb = stop === :oracle ?
        (w -> (mul!(tmp, K, w.x); warm && mul!(tmp, K, base, 1.0, 1.0); tmp .-= b; fired[] = norm(tmp) <= atol)) :
        (w -> (fired[] = norm(w.r) <= atol))
    total = 0
    restarts = 0
    selfstops = 0
    bd = false
    recursive = NaN
    xfull = base
    while true
        warm && Krylov.warm_start!(ws, base)
        fired[] = false
        itmax = KRYLOV_CAP - total
        bd = breakdown(() -> cg!(ws, op, b; M, ldiv = true, atol = 0.0, rtol = 0.0, itmax, callback = cb))
        total += bd ? itmax : ws.stats.niter
        xfull = copy(ws.x)              # Krylov adds the warm-start point back at exit
        recursive = norm(ws.r)
        !bd && !fired[] && ws.stats.niter < itmax && (selfstops += 1)
        stop === :oracle && break
        mul!(tmp, op, xfull)            # the one recomputation, counted as products
        tmp .-= b
        res = norm(tmp)
        (stop === :design || stop === :capfail || res <= atol || bd || total >= KRYLOV_CAP || iszero(ws.stats.niter)) && break
        restarts += 1
        base = xfull
        warm = true
    end
    res = norm(b - K * xfull)
    fl = floor(xfull)
    rec["iters"] = total
    rec["res_over_atol"] = res / atol
    rec["res_over_floor"] = res / fl
    rec["recursive_over_atol"] = recursive / atol
    rec["reached"] = stop === :design ? res <= 2atol : stop === :capfail ? !bd && total < KRYLOV_CAP : res <= atol
    rec["restarts"] = restarts
    rec["krylov_self_stops"] = selfstops
    rec["breakdown"] = bd
    xprev .= xfull
    return xfull
end

# ---------------------------------------------------------------------------------------------
# Experiments and summary

"`(experiment, family, mix, method, δ, stop, start)` jobs."
function jobs3()
    J = Tuple{Symbol, Symbol, Symbol, Symbol, Float64, Symbol, Symbol}[]
    for fam in (:dense, :sparse)
        pc = fam === :dense ? :cg_lagchol3 : :cg_lldl10
        # (a) row classes: exact and preconditioned CG; unpreconditioned CG on dense for the 1/δ effect
        for mix in MIXES
            push!(J, (:rowtypes, fam, mix, :exact, 1.0e-8, :oracle, :zero))
            push!(J, (:rowtypes, fam, mix, pc, 1.0e-8, :oracle, :zero))
            push!(J, (:rowtypes, fam, mix, pc, 1.0e-8, :revised, :zero))
            push!(J, (:rowtypes, fam, mix, pc, 1.0e-8, :design, :zero))
            push!(J, (:rowtypes, fam, mix, pc, 1.0e-8, :capfail, :zero))
            fam === :dense && push!(J, (:rowtypes, fam, mix, :cg_none, 1.0e-8, :oracle, :zero))
        end
        # (b) stopping test on the spike instances
        for δ in (1.0e-8, 1.0e-6), stop in (:oracle, :design, :revised, :capfail)
            push!(J, (:stopping, fam, :spike, pc, δ, stop, :zero))
        end
        # (c) CG start
        for stop in (:oracle, :revised)
            push!(J, (:cgstart, fam, :spike, pc, 1.0e-8, stop, :prev))
        end
        # (d) quasi-definite LDLᵀ without pivoting in place of Bunch–Kaufman
        for mix in (:spike, :mixed), meth in (:exact_cholmod, :exact_ldlf, :exact_cholmod_ref1, :exact_ldlf_ref1, :exact_redchol, :exact_redchol_ref1)
            push!(J, (:kktldl, fam, mix, meth, 1.0e-8, :oracle, :zero))
        end
    end
    return J
end

function row_of(r, n)
    per_outer = Dict{Int, Vector{Int}}()
    for s in r["solves"]
        push!(get!(per_outer, s["k"], Int[]), s["iters"])
    end
    po = [per_outer[k] for k in sort!(collect(keys(per_outer)))]
    row = compact_row(;
        method = r["method"], delta = r["delta"], status = r["status"], outer_iters = r["outer_iters"],
        hits = r["hits"], final_referee = r["final_referee"], per_outer = po,
        reached = [s["reached"] for s in r["solves"]], products = r["products"],
    )
    row["hit_1e-8"] = haskey(r["hits"], "1.0e-8")
    if length(po) >= 6
        f = median(reduce(vcat, po[1:3]))
        t = median(reduce(vcat, po[(end - 2):end]))
        row["g2_floor100"] = t <= max(10f, 100)
        row["g2_floor_n10"] = t <= max(10f, min(100, n / 10))
    end
    ra = [s["res_over_atol"] for s in r["solves"]]
    row["res_over_atol_max"] = maximum(ra; init = 0.0)
    row["solves_in_1to2atol"] = count(x -> 1 < x <= 2, ra)
    row["solves_over_2atol"] = count(>(2), ra)
    rf = [s["res_over_floor"] for s in r["solves"]]
    row["res_over_floor_median"] = isempty(rf) ? nothing : median(rf)
    if !startswith(r["method"], "exact")
        row["restarts"] = sum((s["restarts"] for s in r["solves"]); init = 0)
        row["krylov_self_stops"] = sum((s["krylov_self_stops"] for s in r["solves"]); init = 0)
        row["inner_max_ge_n"] = !isnothing(row["inner_max"]) && row["inner_max"] >= n
    end
    return row
end

function aggregate(recs)
    groups = Dict{Tuple, Vector{Any}}()
    for rec in recs
        key = (rec["experiment"], rec["family"], rec["mix"], rec["run"]["method"], rec["run"]["delta"], rec["run"]["stop"], rec["run"]["start"])
        push!(get!(groups, key, Any[]), rec)
    end
    out = Any[]
    for (key, g) in sort!(collect(groups); by = p -> string(p[1]))
        rows = [x["row"] for x in g]
        get_all(f) = [r[f] for r in rows if haskey(r, f) && !isnothing(r[f])]
        outer = [r["outer_iters"] for r in rows]
        d = Dict{String, Any}(
            "experiment" => key[1], "family" => key[2], "mix" => key[3], "method" => key[4], "delta" => key[5],
            "stop" => key[6], "start" => key[7], "instances" => length(rows),
            "g1_pass" => count(r -> r["g1_pass"], rows), "converged_1e-8" => count(r -> r["hit_1e-8"], rows),
            "outer_min" => minimum(outer), "outer_median" => median(outer), "outer_max" => maximum(outer),
            "statuses" => Dict(s => count(r -> r["status"] == s, rows) for s in unique(r["status"] for r in rows)),
            "Ns_zero" => count(x -> iszero(x["run"]["Ns"]), g),
            "solves" => sum(length(x["run"]["solves"]) for x in g),
            "solves_in_1to2atol" => sum(get_all("solves_in_1to2atol")),
            "solves_over_2atol" => sum(get_all("solves_over_2atol")),
            "res_over_atol_max" => maximum(get_all("res_over_atol_max")),
            "res_over_floor_median_of_medians" => median(get_all("res_over_floor_median")),
        )
        if !startswith(string(key[4]), "exact")
            d["products_total"] = sum(r["products"] for r in rows)
            d["inner_median_of_medians"] = median(get_all("inner_median"))
            d["inner_max"] = maximum(get_all("inner_max"))
            d["inner_max_ge_n"] = count(identity, get_all("inner_max_ge_n"))
            d["g2_floor100_pass"] = count(identity, get_all("g2_floor100"))
            d["g2_floor_n10_pass"] = count(identity, get_all("g2_floor_n10"))
            d["g2_evaluable"] = length(get_all("g2_floor100"))
            d["solves_missed"] = sum(count(s -> !s["reached"], x["run"]["solves"]) for x in g)
            d["restarts"] = sum(get_all("restarts"))
            d["krylov_self_stops"] = sum(get_all("krylov_self_stops"))
        end
        push!(out, d)
    end
    return out
end

"Per instance, outer iterations and products of each non-oracle spike-mix run against the oracle zero-start run of the same instance, method and δ."
function oracle_deltas(recs)
    idx = Dict((x["tag"], x["run"]["method"], x["run"]["delta"]) => x for x in recs if x["experiment"] == "stopping" && x["run"]["stop"] == "oracle")
    out = Dict{String, Any}()
    for x in recs
        x["experiment"] in ("stopping", "cgstart") || continue
        (x["run"]["stop"] == "oracle" && x["run"]["start"] == "zero") && continue
        o = get(idx, (x["tag"], x["run"]["method"], x["run"]["delta"]), nothing)
        isnothing(o) && continue
        key = join((x["family"], x["run"]["method"], x["run"]["delta"], x["run"]["stop"], x["run"]["start"]), "/")
        e = get!(out, key, Dict{String, Any}("outer_diff" => Int[], "products_ratio" => Float64[], "g1_changed" => 0))
        push!(e["outer_diff"], x["run"]["outer_iters"] - o["run"]["outer_iters"])
        push!(e["products_ratio"], x["run"]["products"] / max(1, o["run"]["products"]))
        e["g1_changed"] += x["row"]["g1_pass"] != o["row"]["g1_pass"]
    end
    for e in values(out)
        pr = e["products_ratio"]
        e["products_ratio"] = Dict("min" => minimum(pr), "median" => median(pr), "max" => maximum(pr))
        od = e["outer_diff"]
        e["outer_diff"] = Dict("min" => minimum(od), "max" => maximum(od), "nonzero" => count(!iszero, od))
    end
    return out
end

"Per instance, outer iterations of each no-pivoting LDLᵀ run minus the Bunch–Kaufman run on the same instance and mix."
function kkt_deltas(recs)
    idx = Dict((x["tag"], x["mix"]) => x["run"]["outer_iters"] for x in recs if x["experiment"] == "rowtypes" && x["run"]["method"] == "exact")
    out = Dict{String, Any}()
    for x in recs
        x["experiment"] == "kktldl" || continue
        key = join((x["family"], x["mix"], x["run"]["method"]), "/")
        push!(get!(out, key, Int[]), x["run"]["outer_iters"] - idx[(x["tag"], x["mix"])])
    end
    return Dict(k => Dict("min" => minimum(v), "max" => maximum(v), "abs_gt_2" => count(d -> abs(d) > 2, v), "n" => length(v)) for (k, v) in out)
end

function main3(; raw_path = joinpath(@__DIR__, "results", "ipm_rowtypes_spike_raw.json"), summary_path = joinpath(@__DIR__, "results", "ipm_rowtypes_spike.json"))
    BLAS.set_num_threads(1)
    bases = Dict(:dense => instances(:dense), :sparse => instances(:sparse))
    work = [(j, b) for j in jobs3() for b in eachindex(bases[j[2]])]
    recs = Vector{Any}(undef, length(work))
    Threads.@threads :dynamic for t in eachindex(work)
        (exp, fam, mix, method, δ, stop, start), b = work[t]
        base = bases[fam][b]
        inst = rowtype_instance(base, mix)
        r = try
            ipm3(inst, δ, method; stop, start)
        catch e
            Dict{String, Any}("error" => sprint(showerror, e), "method" => string(method), "delta" => δ, "stop" => string(stop), "start" => string(start))
        end
        recs[t] = Dict{String, Any}(
            "experiment" => string(exp), "family" => string(fam), "mix" => string(mix), "tag" => label_tag(base.label),
            "label" => inst.label, "run" => r,
        )
    end
    errors = [x for x in recs if haskey(x["run"], "error")]
    good = [x for x in recs if !haskey(x["run"], "error")]
    for x in good
        x["row"] = row_of(x["run"], x["label"]["n"])
    end
    mkpath(dirname(raw_path))
    open(io -> JSON.json(io, Dict("julia" => string(VERSION), "krylov_cap" => KRYLOV_CAP, "runs" => good, "errors" => errors); allownan = true), raw_path, "w")
    summary = Dict(
        "description" => "Spike 3 (PureIPM/bench/ipm_rowtypes_spike.jl): row classes, realizable CG stopping, CG start, quasi-definite LDLt; aggregated per (experiment, family, mix, method, delta, stop, start)",
        "g1" => "eps = 1e-6 reached with referee <= 1e-5", "g2_floor100" => "t <= max(10f, 100)", "g2_floor_n10" => "t <= max(10f, min(100, n/10))",
        "krylov_cap" => KRYLOV_CAP, "fail_limit" => CG_FAIL_LIMIT, "errors" => length(errors),
        "groups" => aggregate(good), "vs_oracle" => oracle_deltas(good), "kkt_vs_bunchkaufman" => kkt_deltas(good),
    ) |> compact_numbers
    open(io -> JSON.json(io, summary; allownan = true), summary_path, "w")
    return summary, errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    main3()
end
