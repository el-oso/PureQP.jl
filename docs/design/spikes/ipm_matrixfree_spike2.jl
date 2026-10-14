# Matrix-free IPM spike 2: augmented-system Krylov solvers and caller-supplied preconditioners
#
# Reuses the first spike's Mehrotra prototype (`ipm_matrixfree_spike.jl`: instances, referee,
# termination, step length) with its per-side regularized step recovery and the σ floor
# `σ ≥ min(1, 0.1‖r‖∞/μ)`, and replaces the inner solve by one of `METHODS`:
#
#   exact                 bunchkaufman! on the regularized KKT matrix (reference trajectory)
#   cg_none               CG on the reduced system P + δI + Aᵀ diag(w) A, no preconditioner
#   minres_none           MINRES on the augmented system [P+δI Aᵀ; A −diag(1/w)]
#   minres_blockdiag      MINRES preconditioned by diag(I, w) — identity (1,1), exact (2,2)
#   trimr / tricg         TriMR / TriCG (Krylov.jl) on the same augmented system with
#                         M = (P+δI)⁻¹, N = diag(w). TriMR and TriCG require the (1,1) block to
#                         be exactly τM⁻¹, so M must be the (1,1) inverse: on the Kronecker family
#                         (P = αI) it is the scalar τ = α + δ with M = I (product-only); on the
#                         dense family it is applied through a dense Cholesky factor of P + δI
#                         (not product-only; counted as preconditioner applications).
#   cg_kron_*             CG with a Kronecker eigendecomposition preconditioner of
#                         (α+δ)I + (BᵀD_B B) ⊗ (CᵀD_C C), D_B ⊗ D_C fitted to diag(w) by
#                         `scalar_arith` (arithmetic mean), `rank1_geo` / `rank1_arith`
#                         (geometric / arithmetic row and column means of the weight grid)
#   cg_lagchol<r>         CG preconditioned by the Cholesky factor of the reduced matrix from the
#                         most recent outer iteration whose index is a multiple of r
#   cg_lldl<p>            CG preconditioned by a limited-memory incomplete LDLᵀ
#                         (LimitedLDLFactorizations, memory = p) of the sparse reduced matrix,
#                         refreshed every outer iteration
#   cg_jacobi             CG with the exact diagonal of the sparse reduced matrix
#
# Families (per active fraction in FRACS):
#   dense   the first spike's LinearMap-over-dense generator (`make_instance`, same seed rule),
#           n ∈ NS2, κ(A) ∈ KAPPAS
#   kron    A = B ⊗ C, n = n₁² with n₁ ∈ KRON_SIDES, κ(A) ∈ KAPPAS, P = αI, active rows either
#           random or a Kronecker pattern S₁ × S₂
#   sparse  n ∈ SPARSE_NS; A a 2-D Dirichlet Laplacian, or random sparse (sprandn + I) at 1% or 5% density,
#           1% also with column scalings of κ ∈ {1e3, 1e6}; P diagonal, eigenvalues 1 .. 1e-2
#
# Inner stopping is an oracle on the step: every Krylov method stops when the reduced residual
# of its current Δx, ‖b − KΔx‖₂ with K = P + δI + Aᵀ diag(w) A formed explicitly, reaches
# `atol_k`, so every method delivers a step of the same quality. The oracle's own products are
# instrumentation and are not counted. Counted products are applications of P, A and Aᵀ inside
# the Krylov solve (a Kronecker product counts once); preconditioner applications are counted
# separately. Outer-loop products (right-hand side, step recovery, residuals) are the same for
# every method, except one Aᵀ per solve for the reduced right-hand side, and are not counted.
# Preconditioner construction from materialized matrices (Kronecker factors, reduced matrix)
# is not counted as products.
#
# Every Krylov solve is capped at KRYLOV_CAP iterations. A run is aborted once CG_FAIL_LIMIT
# consecutive solves miss `atol_k` (`abort_cap`) or, from six outer iterations on, once the G2
# ratio (median inner iterations of the last three outer iterations over the first three)
# exceeds 10 (`abort_g2`); `outer_iters` is then the iteration where it stopped. Instance
# records carry `krylov_cap` and `early_stop`, since records from other settings may be kept.
#
# Run from the repository root:
#     julia +1.12 -t 6 --project=bench PureIPM/bench/ipm_matrixfree_spike2.jl [family[:n[:κ]] ...]
# e.g. `dense:1000:1e6 kron sparse:500`. With no argument every family runs. The instances run
# are replaced in
#     PureIPM/bench/results/ipm_matrixfree_spike2_raw.json  (raw samples)
# and the compact summary PureIPM/bench/results/ipm_matrixfree_spike2.json is rewritten from it.

include(joinpath(@__DIR__, "ipm_matrixfree_spike.jl"))
using LimitedLDLFactorizations

const DELTAS2 = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)
const NS2 = (100, 200)
const KRON_SIDES = (10, 14)
const KRYLOV_CAP = 500
# Abort a run on the G2 ratio (`--no-g2-stop` turns it off and writes `*_nog2stop*` files).
const G2_STOP = Ref(true)
const KRON_ALPHA = 1.0e-2
const SPARSE_NS = (100, 200)
const SPARSE_KINDS = ((:laplace, 0.0, 1.0), (:rand, 0.01, 1.0), (:rand, 0.05, 1.0), (:rand, 0.01, 1.0e3), (:rand, 0.01, 1.0e6))
const METHODS = Dict(
    :dense => (:exact, :cg_none, :minres_none, :minres_blockdiag, :trimr, :tricg, :cg_lagchol3, :cg_lagchol5),
    :kron => (:exact, :cg_none, :minres_blockdiag, :trimr, :tricg, :cg_kron_scalar_arith, :cg_kron_rank1_geo, :cg_kron_rank1_arith),
    :sparse => (:exact, :cg_none, :cg_jacobi, :cg_lldl0, :cg_lldl10),
)
# Outer iterations of a TriMR run on which GPMR and TriMR are also run with their native
# residual stopping at the same tolerance (Kronecker family only, where every block
# preconditioner is diagonal).
const GPMR_CHECK_ITERS = 3
# Shifts tried in turn for the limited-memory LDLᵀ (see `lldl_spd`).
const LLDL_SHIFTS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2)

# ---------------------------------------------------------------------------------------------
# Instances

"""
Instance with materialized `P`, `A` (dense or sparse, used for the exact solve, the oracle, the
referee and Clarabel) and product maps `Pop`, `Aop` (what the Krylov solvers see).
"""
struct Inst2{MP, MA, OP, OA, KR}
    family::Symbol
    label::Dict{String, Any}
    n::Int
    kappa::Float64
    frac::Float64
    P::MP
    q::Vector{Float64}
    A::MA
    l::Vector{Float64}
    u::Vector{Float64}
    xstar::Vector{Float64}
    ystar::Vector{Float64}
    nactive::Int
    Pop::OP
    Aop::OA
    kron::KR          # (B, C, α) for the Kronecker family, `nothing` otherwise
end

"Bounds, multipliers and `q` making `(x⋆, y⋆)` the optimum, as `make_instance` does, on the rows `active`."
function plant(rng, P, A, active)
    m, n = size(A)
    xstar = randn(rng, n)
    a = A * xstar
    l = a .- (0.5 .+ rand(rng, m))
    u = a .+ (0.5 .+ rand(rng, m))
    ystar = zeros(m)
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
    q = -(P * xstar + A' * ystar)
    return xstar, ystar, l, u, q
end

function dense_instance(n, κ, frac)
    inst = make_instance(n, κ, frac; seed = hash((n, κ, frac)) % UInt32)
    label = Dict{String, Any}("family" => "dense", "n" => n, "kappa" => κ, "active_fraction" => frac)
    return Inst2(
        :dense, label, n, κ, frac, inst.P, inst.q, inst.A, inst.l, inst.u, inst.xstar, inst.ystar,
        inst.nactive, LinearMap(inst.P; issymmetric = true), LinearMap(inst.A), nothing,
    )
end

function kron_instance(side, κ, frac, pattern)
    rng = Xoshiro(hash((:kron, side, κ, frac, pattern)) % UInt32)
    factor() = Matrix(qr(randn(rng, side, side)).Q) * Diagonal(exp10.(range(0, -log10(κ) / 2; length = side))) * Matrix(qr(randn(rng, side, side)).Q)'
    B = factor()
    C = factor()
    n = side^2
    A = kron(B, C)
    P = sparse(KRON_ALPHA * I, n, n)
    if pattern === :random
        active = randperm(rng, n)[1:round(Int, frac * n)]
    else
        k = round(Int, sqrt(frac) * side)
        S1 = randperm(rng, side)[1:k]
        S2 = randperm(rng, side)[1:k]
        # kron(B, C) row (i₁ − 1)·side + i₂ couples row i₁ of B with row i₂ of C.
        active = [(i1 - 1) * side + i2 for i1 in S1 for i2 in S2]
    end
    xstar, ystar, l, u, q = plant(rng, P, A, active)
    label = Dict{String, Any}("family" => "kron", "n" => n, "side" => side, "kappa" => κ, "active_fraction" => frac, "pattern" => string(pattern))
    Pop = LinearMap{Float64}((y, x) -> (y .= KRON_ALPHA .* x), n; ismutating = true, issymmetric = true)
    return Inst2(:kron, label, n, κ, frac, P, q, A, l, u, xstar, ystar, length(active), Pop, LinearMap(B) ⊗ LinearMap(C), (B, C, KRON_ALPHA))
end

function laplace2d(n)
    g1, g2 = n == 100 ? (10, 10) : n == 200 ? (20, 10) : n == 500 ? (25, 20) : n == 1000 ? (40, 25) : error("no grid for n = $n")
    T(g) = spdiagm(-1 => fill(-1.0, g - 1), 0 => fill(2.0, g), 1 => fill(-1.0, g - 1))
    return kron(sparse(1.0I, g2, g2), T(g1)) + kron(T(g2), sparse(1.0I, g1, g1))
end

function sparse_instance(n, kind, density, κ, frac)
    rng = Xoshiro(hash((:sparse, n, kind, density, κ, frac)) % UInt32)
    A = kind === :laplace ? laplace2d(n) : sprandn(rng, n, n, density) + sparse(1.0I, n, n)
    isone(κ) || (A = A * Diagonal(exp10.(range(0, -log10(κ); length = n))))
    P = sparse(Diagonal(exp10.(range(0, -2; length = n))[randperm(rng, n)]))
    active = randperm(rng, n)[1:round(Int, frac * n)]
    xstar, ystar, l, u, q = plant(rng, P, A, active)
    label = Dict{String, Any}(
        "family" => "sparse", "n" => n, "kind" => string(kind), "density" => density,
        "kappa_scaling" => κ, "active_fraction" => frac, "nnz_A" => nnz(A),
    )
    return Inst2(:sparse, label, n, κ, frac, P, q, A, l, u, xstar, ystar, length(active), LinearMap(P; issymmetric = true), LinearMap(A), nothing)
end

function instances(family)
    family === :dense && return [dense_instance(n, κ, f) for n in NS2 for κ in KAPPAS for f in FRACS]
    family === :kron && return [kron_instance(s, κ, f, p) for s in KRON_SIDES for κ in KAPPAS for f in FRACS for p in (:random, :kron)]
    family === :sparse && return [sparse_instance(n, k, d, κ, f) for n in SPARSE_NS for (k, d, κ) in SPARSE_KINDS for f in FRACS]
    return error("unknown family $family")
end

# ---------------------------------------------------------------------------------------------
# Inner solvers

mutable struct Counter
    P::Int
    A::Int
    At::Int
    M::Int
end
products(c::Counter) = c.P + c.A + c.At

function counted_maps(inst, c)
    n, m = inst.n, length(inst.l)
    Pc = LinearMap{Float64}((y, x) -> (c.P += 1; mul!(y, inst.Pop, x)), n; ismutating = true, issymmetric = true)
    Ac = LinearMap{Float64}(
        (y, x) -> (c.A += 1; mul!(y, inst.Aop, x)), (y, x) -> (c.At += 1; mul!(y, inst.Aop', x)), m, n;
        ismutating = true,
    )
    return Pc, Ac
end

reduced_matrix(P::Matrix, A::Matrix, w, δ) = Symmetric(P + δ * I + A' * (w .* A))
reduced_matrix(P, A::Matrix, w, δ) = Symmetric(Matrix(P) + δ * I + A' * (w .* A))
reduced_matrix(P::SparseMatrixCSC, A::SparseMatrixCSC, w, δ) = P + δ * I + A' * Diagonal(w) * A

"Stops a Krylov method once the reduced residual of `getx(ws)` reaches `atol`."
function oracle(K, b, atol, getx)
    tmp = similar(b)
    return ws -> begin
        mul!(tmp, K, getx(ws))
        tmp .-= b
        norm(tmp) <= atol
    end
end

"Runs `f()` (a Krylov call); returns `true` if the preconditioned inner product turned non-positive."
function breakdown(f)
    try
        f()
        return false
    catch e
        (e isa ErrorException && occursin("definite", e.msg)) || rethrow()
        return true
    end
end

"`y₁ = (P+δI)x₁ + Aᵀx₂`, `y₂ = Ax₁ − x₂/w` on the augmented vector."
function augmented_map(Pc, Ac, w, δ, n, m)
    tmp = zeros(n)
    return LinearMap{Float64}(
        (y, x) -> begin
            x1 = view(x, 1:n)
            x2 = view(x, (n + 1):(n + m))
            y1 = view(y, 1:n)
            y2 = view(y, (n + 1):(n + m))
            mul!(y1, Pc, x1)
            y1 .+= δ .* x1
            mul!(tmp, Ac', x2)
            y1 .+= tmp
            mul!(y2, Ac, x1)
            y2 .-= x2 ./ w
            y
        end, n + m; ismutating = true, issymmetric = true,
    )
end

"Kronecker eigendecomposition preconditioner for `(α+δ)I + (BᵀD_B B) ⊗ (CᵀD_C C)`."
function kron_precond(inst, w, δ, fit, c)
    B, C, α = inst.kron
    side = size(B, 1)
    Wg = reshape(w, side, side)          # Wg[i₂, i₁] is the weight of kron row (i₁ − 1)·side + i₂
    if fit === :scalar_arith
        wB = fill(mean(w), side)
        wC = ones(side)
    elseif fit === :rank1_geo
        L = log.(Wg)
        wC = exp.(vec(mean(L; dims = 2)))
        wB = exp.(vec(mean(L; dims = 1)) .- mean(L))
    elseif fit === :rank1_arith
        wC = vec(mean(Wg; dims = 2))
        wB = vec(mean(Wg; dims = 1)) ./ mean(Wg)
    else
        error("unknown fit $fit")
    end
    EB = eigen(Symmetric(B' * (wB .* B)))
    EC = eigen(Symmetric(C' * (wC .* C)))
    denom = (α + δ) .+ max.(EC.values, 0) * max.(EB.values, 0)'
    UB, UC = EB.vectors, EC.vectors
    return LinearMap{Float64}(
        (y, x) -> begin
            c.M += 1
            X = reshape(x, side, side)
            y .= vec(UC * ((UC' * X * UB) ./ denom) * UB')
        end, inst.n; ismutating = true, issymmetric = true,
    )
end

"""
    lldl_spd(K, memory) -> (F, α)

Limited-memory LDLᵀ of the SPD matrix `K` with the smallest shift `α` of `LLDL_SHIFTS` for
which every pivot of `D` is positive, so `F` is an SPD preconditioner. `lldl` itself raises the
shift only when a pivot vanishes, and on these reduced matrices it returns negative pivots at
`α = 0`.
"""
function lldl_spd(K, memory)
    T = tril(K)
    for α in LLDL_SHIFTS
        F = lldl(T; memory, α)
        all(>(0), F.D) && return F, F.α_out
    end
    return error("no positive-definite lldl factor up to α = $(last(LLDL_SHIFTS))")
end

"""
    make_inner(method, inst, δ, Pc, Ac, c, state) -> (refresh!, solve)

`refresh!(k, w, K)` runs at the start of outer iteration `k` with the weights and the explicit
reduced matrix; `solve(rd, g, w, b, atol, K, k)` returns `Δx` and a record of the solve. The
solver keeps its preconditioner and diagnostics (`:refresh_failures`, `:lldl_shifts`) in
`state`.
"""
function make_inner(method, inst, δ, Pc, Ac, c, state)
    n, m = inst.n, length(inst.l)
    ms = string(method)
    finish(dx, iters, K, b, atol, bd, p0, m0) = begin
        res = norm(b - K * dx)
        rec = Dict{String, Any}(
            "iters" => iters, "reached" => res <= atol, "res2" => res, "breakdown" => bd,
            "products" => products(c) - p0, "precond_applies" => c.M - m0,
        )
        (dx, rec)
    end
    if method === :exact
        refresh! = (k, w, K) -> (state[:F] = bunchkaufman!(Symmetric([Matrix(inst.P) + δ * I Matrix(inst.A)'; Matrix(inst.A) -Diagonal(1 ./ w)], :U)))
        solve = (rd, g, w, b, atol, K, k) -> finish((state[:F] \ [-rd; -g ./ w])[1:n], 0, K, b, atol, false, products(c), c.M)
        return refresh!, solve
    end
    if startswith(ms, "cg_")
        refresh! = (k, w, K) -> begin
            if method === :cg_none
                state[:M] = I
            elseif method === :cg_jacobi
                state[:M] = Diagonal(1 ./ diag(K))
            elseif startswith(ms, "cg_kron_")
                state[:M] = kron_precond(inst, w, δ, Symbol(ms[9:end]), c)
            elseif startswith(ms, "cg_lagchol")
                if iszero(k % parse(Int, ms[11:end]))
                    F = cholesky(K; check = false)
                    if issuccess(F)
                        state[:M] = LinearMap{Float64}((y, x) -> (c.M += 1; ldiv!(y, F, x)), n; ismutating = true, issymmetric = true)
                    else
                        state[:refresh_failures] = get(state, :refresh_failures, 0) + 1
                    end
                end
                haskey(state, :M) || (state[:M] = I)
            elseif startswith(ms, "cg_lldl")
                F, α = lldl_spd(K, parse(Int, ms[8:end]))
                push!(get!(state, :lldl_shifts, Float64[]), α)
                state[:M] = LinearMap{Float64}((y, x) -> (c.M += 1; ldiv!(y, F, x)), n; ismutating = true, issymmetric = true)
            else
                error("unknown method $method")
            end
        end
        solve = (rd, g, w, b, atol, K, k) -> begin
            p0, m0 = products(c), c.M
            op = reduced_map(Pc, Ac, w, δ)
            ws = CgWorkspace(n, n, Vector{Float64})
            bd = breakdown(() -> cg!(ws, op, b; M = state[:M], ldiv = false, atol = 0.0, rtol = 0.0, itmax = KRYLOV_CAP, callback = oracle(K, b, atol, v -> v.x)))
            finish(copy(ws.x), bd ? KRYLOV_CAP : ws.stats.niter, K, b, atol, bd, p0, m0)
        end
        return refresh!, solve
    end
    if startswith(ms, "minres_")
        refresh! = (k, w, K) -> nothing
        solve = (rd, g, w, b, atol, K, k) -> begin
            p0, m0 = products(c), c.M
            op = augmented_map(Pc, Ac, w, δ, n, m)
            M = method === :minres_blockdiag ? Diagonal([ones(n); w]) : I
            ws = MinresWorkspace(n + m, n + m, Vector{Float64})
            minres!(ws, op, [-rd; -g ./ w]; M, atol = 0.0, rtol = 0.0, etol = 0.0, conlim = 0.0, itmax = KRYLOV_CAP, callback = oracle(K, b, atol, v -> view(v.x, 1:n)))
            finish(ws.x[1:n], ws.stats.niter, K, b, atol, false, p0, m0)
        end
        return refresh!, solve
    end
    if method === :trimr || method === :tricg
        refresh! = (k, w, K) -> begin
            if inst.family === :kron
                state[:M] = I
                state[:τ] = inst.kron[3] + δ
            elseif !haskey(state, :M)
                F = cholesky(Symmetric(Matrix(inst.P) + δ * I))
                state[:M] = LinearMap{Float64}((y, x) -> (c.M += 1; ldiv!(y, F, x)), n; ismutating = true, issymmetric = true)
                state[:τ] = 1.0
            end
        end
        solve = (rd, g, w, b, atol, K, k) -> begin
            p0, m0 = products(c), c.M
            M, τ = state[:M], state[:τ]
            N = Diagonal(w)
            b2 = -g ./ w
            if method === :trimr
                ws = TrimrWorkspace(n, m, Vector{Float64})
                trimr!(ws, Ac', -rd, b2; M, N, τ, ν = -1.0, atol = 0.0, rtol = 0.0, itmax = KRYLOV_CAP, callback = oracle(K, b, atol, v -> v.x))
            else
                ws = TricgWorkspace(n, m, Vector{Float64})
                tricg!(ws, Ac', -rd, b2; M, N, τ, ν = -1.0, atol = 0.0, rtol = 0.0, itmax = KRYLOV_CAP, callback = oracle(K, b, atol, v -> v.x))
            end
            dx, rec = finish(copy(ws.x), ws.stats.niter, K, b, atol, false, p0, m0)
            if method === :trimr && inst.family === :kron && k < GPMR_CHECK_ITERS
                # Native residual stopping at the same tolerance. With D = F = diag(√w), GPMR's
                # preconditioned system is diag(I, √w) [τI Aᵀ; A −diag(1/w)] diag(I, √w).
                sw = Diagonal(sqrt.(w))
                wt = TrimrWorkspace(n, m, Vector{Float64})
                trimr!(wt, Ac', -rd, b2; M, N, τ, ν = -1.0, atol, rtol = 0.0, itmax = KRYLOV_CAP)
                wg = GpmrWorkspace(n, m, Vector{Float64}; memory = 50)
                gpmr!(wg, Ac', Ac, -rd, b2; D = sw, F = sw, λ = τ, μ = -1.0, atol, rtol = 0.0, itmax = KRYLOV_CAP)
                rec["native_check"] = Dict(
                    "atol" => atol, "trimr_iters" => wt.stats.niter, "gpmr_iters" => wg.stats.niter,
                    "trimr_reduced_res2" => norm(b - K * wt.x), "gpmr_reduced_res2" => norm(b - K * wg.x),
                    "trimr_solved" => wt.stats.solved, "gpmr_solved" => wg.stats.solved,
                )
            end
            (dx, rec)
        end
        return refresh!, solve
    end
    return error("unknown method $method")
end

# ---------------------------------------------------------------------------------------------
# Mehrotra predictor–corrector with per-side recovery and σ floor

function ipm2(inst, δ, method)
    P, q, A, l, u = inst.P, inst.q, inst.A, inst.l, inst.u
    m = length(l)
    c = Counter(0, 0, 0, 0)
    Pc, Ac = counted_maps(inst, c)
    state = Dict{Symbol, Any}()
    refresh!, inner_solve = make_inner(method, inst, δ, Pc, Ac, c, state)
    t0 = time()

    x = cholesky(Symmetric(Matrix(P + δ * I + A' * A))) \ (-q + A' * ((l + u) / 2))
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
                hits[key] = Dict("iter" => k, "referee" => referee(inst, x, y), "obj" => objective(inst, x), "time" => time() - t0)
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
        dl = sl .+ δ .* zl
        du = su .+ δ .* zu
        w = zl ./ dl + zu ./ du
        level = min(μ, rnorm)
        K = reduced_matrix(P, A, w, δ)
        refresh!(k, w, K)
        rec = Dict{String, Any}("k" => k, "mu" => μ, "rnorm" => rnorm, "w_max" => maximum(w), "w_min" => minimum(w), "solves" => Any[])
        solve_failed = false
        direction = function (rcl, rcu, label)
            g = (rcl + zl .* rl) ./ dl - (rcu + zu .* ru) ./ du
            b = -rd - A' * g
            atol = max(ETA * level, eps() * max(1.0, norm(b, Inf)))
            dx, srec = inner_solve(rd, g, w, b, atol, K, k)
            srec["which"] = label
            srec["atol"] = atol
            push!(rec["solves"], srec)
            if method !== :exact
                # A solve that misses atol_k has run to KRYLOV_CAP, broken down, or (MINRES)
                # stopped on its condition estimate.
                if srec["reached"]
                    fails = 0
                else
                    fails += 1
                    fails >= CG_FAIL_LIMIT && (solve_failed = true)
                end
            end
            Adx = A * dx
            dzl = -(rcl + zl .* (Adx + rl)) ./ dl
            dzu = -(rcu + zu .* (-Adx + ru)) ./ du
            dsl = Adx + rl + δ .* dzl
            dsu = -Adx + ru + δ .* dzu
            return dx, dsl, dsu, dzl, dzu
        end

        dx, dsl, dsu, dzl, dzu = direction(sl .* zl, su .* zu, "predictor")
        αa = min(1.0, max_step(sl, dsl), max_step(su, dsu), max_step(zl, dzl), max_step(zu, dzu))
        μa = (dot(sl + αa * dsl, zl + αa * dzl) + dot(su + αa * dsu, zu + αa * dzu)) / Ns
        σ = max((μa / μ)^3, min(1.0, 0.1 * rnorm / μ))
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
        rec["alpha"] = α
        rec["sigma"] = σ
        push!(iters, rec)
        if solve_failed
            status = "abort_cap"
            break
        end
        # Early stop on unbounded inner work: G2's ratio above 10 once six outer iterations exist.
        if G2_STOP[] && method !== :exact && length(iters) >= 6
            _, _, ratio = g2([[s["iters"] for s in it["solves"]] for it in iters])
            if ratio > 10
                status = "abort_g2"
                break
            end
        end
        small_steps = α < 1.0e-8 ? small_steps + 1 : 0
        if small_steps >= 3
            status = "stall"
            break
        end
    end
    y = zu - zl
    return Dict{String, Any}(
        "delta" => δ, "method" => string(method), "status" => status, "outer_iters" => length(iters),
        "hits" => hits, "final_referee" => referee(inst, x, y), "final_obj" => objective(inst, x),
        "time" => time() - t0, "products" => products(c), "precond_applies" => c.M,
        "counts" => Dict("P" => c.P, "A" => c.A, "At" => c.At),
        "refresh_failures" => get(state, :refresh_failures, 0), "lldl_shifts" => get(state, :lldl_shifts, nothing),
        "iters" => iters,
    )
end

# ---------------------------------------------------------------------------------------------
# Driver

"Instance record without runs: Clarabel reference and measured κ(A)."
function instance_record(inst)
    ref = clarabel_reference(inst)
    sv = svdvals(Matrix(inst.A))
    return Dict{String, Any}(
        "label" => inst.label, "nactive" => inst.nactive, "obj_star" => objective(inst, inst.xstar),
        "referee_star" => referee(inst, inst.xstar, inst.ystar), "cond_A_measured" => sv[1] / sv[end],
        "clarabel" => Dict("status" => ref.status, "obj" => ref.obj, "time" => ref.time, "xerr" => norm(ref.x - inst.xstar, Inf)),
        "runs" => Any[], "krylov_cap" => KRYLOV_CAP, "early_stop" => G2_STOP[] ? "abort_cap+abort_g2" : "abort_cap",
    )
end

"Adds a run to its instance record; the exact run at the smallest δ is the validation against Clarabel."
function add_run!(out, r)
    push!(out["runs"], r)
    if r["method"] == "exact" && r["delta"] == first(DELTAS2)
        cobj = out["clarabel"]["obj"]
        relerr = abs(r["final_obj"] - cobj) / max(1.0, abs(cobj))
        out["validation"] = Dict(
            "exact_status" => r["status"], "exact_obj" => r["final_obj"], "obj_relerr_vs_clarabel" => relerr,
            "pass" => r["status"] == "solved" && relerr <= 1.0e-6,
        )
    end
    return out
end

function instance_rows(inst)
    rows = Any[]
    for r in inst["runs"]
        per_outer = [[s["iters"] for s in it["solves"]] for it in r["iters"]]
        reached = [s["reached"] for it in r["iters"] for s in it["solves"]]
        row = compact_row(;
            method = r["method"], delta = r["delta"], status = r["status"], outer_iters = r["outer_iters"],
            hits = r["hits"], final_referee = r["final_referee"], per_outer, reached, products = r["products"],
        )
        if length(per_outer) >= 6
            # G2 with the first-three median floored at 10 inner iterations, so a preconditioner
            # that starts at one or two iterations is not failed for growing to a few tens.
            f = median(reduce(vcat, per_outer[1:3]))
            t = median(reduce(vcat, per_outer[(end - 2):end]))
            row["g2_ratio_floor10"] = t / max(f, 10)
        end
        iszero(r["precond_applies"]) || (row["precond_applies"] = r["precond_applies"])
        iszero(r["refresh_failures"]) || (row["refresh_failures"] = r["refresh_failures"])
        isnothing(r["lldl_shifts"]) || (row["lldl_shift_max"] = maximum(r["lldl_shifts"]; init = 0.0))
        checks = [s["native_check"] for it in r["iters"] for s in it["solves"] if haskey(s, "native_check")]
        if !isempty(checks)
            row["native_check"] = Dict(
                "solves" => length(checks),
                "trimr_iters" => sum(e["trimr_iters"] for e in checks),
                "gpmr_iters" => sum(e["gpmr_iters"] for e in checks),
                "max_abs_iter_diff" => maximum(abs(e["trimr_iters"] - e["gpmr_iters"]) for e in checks),
            )
        end
        push!(rows, row)
    end
    return rows
end

"Floats rounded to three significant digits, recursively; keeps the summary small."
compact_numbers(x::AbstractFloat) = isfinite(x) ? round(x; sigdigits = 3) : x
compact_numbers(x::AbstractDict) = Dict{String, Any}(k => compact_numbers(v) for (k, v) in x)
compact_numbers(x::AbstractVector) = map(compact_numbers, x)
compact_numbers(x) = x

# Each summary row is an array in this column order (`nothing` where a field does not apply).
const SUMMARY_COLUMNS = (
    "method", "delta", "status", "converged", "outer_iters", "final_referee", "g1_pass", "g1_iter",
    "g1_referee", "g2_ratio", "g2_pass", "inner_median", "inner_max", "products", "frac_reached",
    "g2_ratio_floor10", "precond_applies", "refresh_failures", "lldl_shift_max", "native_check",
)

function write_summary2(raw, path)
    out = Dict(
        "description" => "Compact summary of PureIPM/bench/results/ipm_matrixfree_spike2_raw.json (matrix-free IPM spike 2)",
        "settings" => raw["settings"],
        "g1" => "eps = $(first(EPS_LIST)) reached with referee <= 1e-5",
        "g2" => "median inner iterations per solve, last 3 outer iterations / first 3, pass <= 10 (NaN if fewer than 6 outer iterations)",
        "products" => "applications of P, A, A' inside the Krylov solves (counted); preconditioner applications in precond_applies",
        "instances" => [
            Dict{String, Any}(
                "label" => i["label"], "clarabel_obj" => i["clarabel"]["obj"], "cond_A_measured" => i["cond_A_measured"],
                "validation" => i["validation"], "krylov_cap" => i["krylov_cap"],
                # Boolean records predate the `--no-g2-stop` switch: `true` ran with both stops,
                # `false` with none.
                "early_stop" => i["early_stop"] === true ? "abort_cap+abort_g2" : i["early_stop"] === false ? "none" : i["early_stop"],
                "rows" => [[get(r, c, nothing) for c in SUMMARY_COLUMNS] for r in instance_rows(i)],
            ) for i in raw["instances"] if !haskey(i, "error")
        ],
        "columns" => collect(SUMMARY_COLUMNS),
    ) |> compact_numbers
    open(io -> JSON.json(io, out; allownan = true), path, "w")
    return out
end

# Numbers as Float64: a label read back from the JSON may hold integers as floats.
label_tag(label) = join(sort!(["$k=$(v isa Real ? Float64(v) : v)" for (k, v) in label]), ",")

"""
    select(spec) -> instances

`spec` is `family`, `family:n` or `family:n:κ` (κ as in `KAPPAS` or the sparse column scaling).
"""
function select(spec)
    parts = split(spec, ':')
    insts = instances(Symbol(parts[1]))
    length(parts) >= 2 && filter!(i -> i.n == parse(Int, parts[2]), insts)
    length(parts) >= 3 && filter!(i -> i.kappa == parse(Float64, parts[3]), insts)
    return insts
end

function main2(
        specs = ("dense", "kron", "sparse");
        raw_path = joinpath(@__DIR__, "results", "ipm_matrixfree_spike2_raw.json"),
        summary_path = joinpath(@__DIR__, "results", "ipm_matrixfree_spike2.json"),
    )
    BLAS.set_num_threads(1)
    insts = reduce(vcat, [select(s) for s in specs])
    results = Vector{Any}(undef, length(insts))
    Threads.@threads :dynamic for i in eachindex(insts)
        results[i] = instance_record(insts[i])
    end
    # One job per instance × method × δ, largest instances first.
    jobs = sort!([(i, method, δ) for i in eachindex(insts) for method in METHODS[insts[i].family] for δ in DELTAS2]; by = j -> -insts[j[1]].n)
    runs = Vector{Any}(undef, length(jobs))
    failures = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for j in eachindex(jobs)
        i, method, δ = jobs[j]
        label = insts[i].label
        tag = join(("$k=$v" for (k, v) in sort(collect(label); by = first)), " ")
        try
            r = ipm2(insts[i], δ, method)
            runs[j] = r
            @printf("%s δ=%.0e %s: %s in %d, products %d (%.1fs)\n", tag, δ, method, r["status"], r["outer_iters"], r["products"], r["time"])
            flush(stdout)
        catch e
            runs[j] = nothing
            results[i]["error"] = sprint(showerror, e, catch_backtrace())
            Threads.atomic_add!(failures, 1)
            println(stderr, "FAILED ", tag, " δ=", δ, " ", method, ": ", sprint(showerror, e))
            flush(stderr)
        end
    end
    for j in eachindex(jobs)
        isnothing(runs[j]) || add_run!(results[jobs[j][1]], runs[j])
    end
    tags = Set(label_tag(i.label) for i in insts)
    kept = isfile(raw_path) ? [i for i in JSON.parsefile(raw_path; allownan = true)["instances"] if !(label_tag(i["label"]) in tags)] : Any[]
    # Records written before the cap of 500 and the early stop ran with a cap of 2000 and no G2 stop.
    for i in kept
        haskey(i, "krylov_cap") || (i["krylov_cap"] = 2000; i["early_stop"] = "none")
    end
    raw = Dict(
        "description" => "Matrix-free IPM spike 2: augmented-system Krylov solvers and caller-supplied preconditioners inside a Mehrotra prototype",
        "julia" => string(VERSION), "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "settings" => Dict(
            "eps_list" => collect(EPS_LIST), "max_iter" => MAX_ITER, "krylov_cap" => KRYLOV_CAP, "fail_limit" => CG_FAIL_LIMIT,
            "early_stop" => "abort_cap: fail_limit consecutive solves missing atol_k; abort_g2: G2 ratio > 10 from 6 outer iterations on",
            "sizes" => Dict("dense" => collect(NS2), "kron_sides" => collect(KRON_SIDES), "sparse" => collect(SPARSE_NS)),
            "eta" => ETA, "tau" => TAU, "deltas" => collect(DELTAS2), "recovery" => "perside", "sigma_floor" => true,
            "inner_stopping" => "oracle: ||b - K dx||_2 <= atol_k on the explicit reduced matrix, checked every Krylov iteration",
            "methods" => Dict(string(k) => string.(collect(v)) for (k, v) in METHODS),
            "kron_alpha" => KRON_ALPHA, "kron_sides" => collect(KRON_SIDES),
            "sparse_kinds" => [Dict("kind" => string(k), "density" => d, "kappa_scaling" => κ) for (k, d, κ) in SPARSE_KINDS],
            "trimr_tricg_M" => "kron: tau = alpha + delta, M = I (product-only); dense: M = (P + delta I)^-1 by dense Cholesky (counted in precond_applies)",
        ),
        "instances" => vcat(kept, results),
    )
    mkpath(dirname(raw_path))
    open(io -> JSON.json(io, raw; allownan = true), raw_path, "w")
    write_summary2(raw, summary_path)
    iszero(failures[]) || error("$(failures[]) instance(s) failed; see \"error\" entries in $raw_path")
    return raw
end

if abspath(PROGRAM_FILE) == @__FILE__
    specs = filter(!=("--no-g2-stop"), ARGS)
    if length(specs) < length(ARGS)
        G2_STOP[] = false
        main2(
            isempty(specs) ? ("dense", "kron", "sparse") : Tuple(specs);
            raw_path = joinpath(@__DIR__, "results", "ipm_matrixfree_spike2_nog2stop_raw.json"),
            summary_path = joinpath(@__DIR__, "results", "ipm_matrixfree_spike2_nog2stop.json"),
        )
    else
        main2(isempty(specs) ? ("dense", "kron", "sparse") : Tuple(specs))
    end
end
