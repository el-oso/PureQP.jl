"""
    OperatorSplitting(; kwargs...)

The OSQP operator-splitting (ADMM) method, the default algorithm of [`setup`](@ref) and
[`solve`](@ref). The keyword arguments are its parameters; every default is libosqp 1.0's,
including `adaptive_rho` adapting on a fixed iteration interval of 50.

- `rho = 0.1`, `sigma = 1e-6`, `alpha = 1.6` — the step size, the primal regularization and
  the relaxation of the iteration. `P + sigma*I` must be positive definite.
- `adaptive_rho = true` — `:disabled`, `:iterations` or `:kkt_error`; `true` is `:iterations`
  and `false` is `:disabled`.
- `adaptive_rho_interval = 50`, `adaptive_rho_fraction = 0.4`, `adaptive_rho_tolerance = 5.0`
  — how often `ρ` is retuned, the fall in the KKT error `:kkt_error` requires first, and the
  factor `ρ` must move by before a refactorization is paid for.
- `rho_is_vec = true` — give equality rows a `ρ` a thousand times larger than inequality rows.
- `cg_tol_reduction = 10` — the factor the conjugate-gradient tolerance is divided by when CG
  stops iterating, with `linsys = :indirect`.
- `profile_primdual = false` — fill [`Solution`](@ref)'s `primdual_int` and
  `primdual_int_log`; the only parameter that makes the solve read a clock it would not
  otherwise read, at a measured cost under 1%. Only ADMM reads a clock this way, so this
  parameter stays here rather than in [`Options`](@ref); `verbose` moved there because both
  algorithms now print a progress report.

One mode is deliberately absent. libosqp 1.0 offers a fourth `adaptive_rho`, adapting once
a fraction of the setup time has elapsed; a solver that decides when to refactorize by
reading a clock takes a different number of iterations on a different machine, so the modes
here are `:disabled`, `:iterations` and `:kkt_error` only.

The object is built without an element type; [`setup`](@ref) converts it to the solve's
element type, and the workspace holds that `OperatorSplitting{T}` as `ws.algorithm`.
"""
struct OperatorSplitting{T <: Real} <: QPAlgorithm
    rho::T
    sigma::T
    alpha::T
    adaptive_rho::Symbol
    adaptive_rho_interval::Int
    adaptive_rho_fraction::T
    adaptive_rho_tolerance::T
    rho_is_vec::Bool
    cg_tol_reduction::Int
    profile_primdual::Bool
end

function OperatorSplitting(;
        rho = 0.1, sigma = 1.0e-6, alpha = 1.6, adaptive_rho = true, adaptive_rho_interval = 50,
        adaptive_rho_fraction = 0.4, adaptive_rho_tolerance = 5.0, rho_is_vec = true,
        cg_tol_reduction = 10, profile_primdual = false,
    )
    # `adaptive_rho` names a mode. A `Bool` is also accepted: `true` is `:iterations`.
    rho_mode = adaptive_rho isa Bool ? (adaptive_rho ? :iterations : :disabled) :
        Symbol(adaptive_rho)
    rho_mode in (:disabled, :iterations, :kkt_error) || throw(
        ArgumentError(
            "adaptive_rho must be :disabled, :iterations, :kkt_error or a Bool, got :$rho_mode"
        )
    )
    0 < adaptive_rho_fraction <= 1 || throw(
        ArgumentError("adaptive_rho_fraction must lie in (0, 1], got $adaptive_rho_fraction")
    )
    cg_tol_reduction > 0 || throw(ArgumentError("cg_tol_reduction must be positive"))
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    rho > 0 || throw(ArgumentError("rho must be positive, got $rho"))
    0 < alpha < 2 || throw(ArgumentError("alpha must lie in (0, 2), got $alpha"))
    adaptive_rho_interval >= 0 || throw(ArgumentError("adaptive_rho_interval must be non-negative"))
    adaptive_rho_tolerance >= 1 || throw(ArgumentError("adaptive_rho_tolerance must be at least 1"))
    F = float(
        promote_type(
            typeof(rho), typeof(sigma), typeof(alpha), typeof(adaptive_rho_fraction),
            typeof(adaptive_rho_tolerance),
        )
    )
    return OperatorSplitting{F}(
        F(rho), F(sigma), F(alpha), rho_mode, Int(adaptive_rho_interval),
        F(adaptive_rho_fraction), F(adaptive_rho_tolerance), Bool(rho_is_vec),
        Int(cg_tol_reduction), Bool(profile_primdual),
    )
end

"`a` in element type `T`."
OperatorSplitting{T}(a::OperatorSplitting) where {T <: Real} = OperatorSplitting{T}(
    T(a.rho), T(a.sigma), T(a.alpha), a.adaptive_rho, a.adaptive_rho_interval,
    T(a.adaptive_rho_fraction), T(a.adaptive_rho_tolerance), a.rho_is_vec,
    a.cg_tol_reduction, a.profile_primdual,
)

const OPERATOR_SPLITTING_NAMES = fieldnames(OperatorSplitting{Float64})

"""
    element_typed(alg, T, options) -> QPAlgorithm

`alg` in element type `T`, with every default that depends on `T` or on `options` resolved.
"""
element_typed(a::OperatorSplitting, ::Type{T}, ::Options) where {T} = OperatorSplitting{T}(a)

"The [`Options`](@ref) defaults of an algorithm that differ from the other algorithm's."
algorithm_defaults(::OperatorSplitting, ::Type{T}) where {T} = (
    max_iter = 4000, eps_abs = 1.0e-3, eps_rel = 1.0e-3, eps_prim_inf = 1.0e-4,
    eps_dual_inf = 1.0e-4, check_termination = 25, cg_max_iter = 20, cg_tol_fraction = 0.15,
)

"""
    OperatorSplittingWorkspace{T,MP,MA,V,VI,LS,AC} <: QPWorkspace{T}

Solver state of [`OperatorSplitting`](@ref), built by [`setup`](@ref). The caller's `P` and
`A` are held by reference and never mutated: Ruiz equilibration lives in the factors `D`,
`E`, `c` and is applied lazily on every product.

The buffers are `similar` to the `q` that built the workspace, so they follow the array
type of the caller's data rather than always being `Vector`.
"""
mutable struct OperatorSplittingWorkspace{
        T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix,
        V <: AbstractVector{T}, VI <: AbstractVector{Int8}, LS <: LinearSystem,
        AC,
    } <: QPWorkspace{T}
    prob::Problem{T, MP, MA, V}
    x::V
    y::V
    z::V
    x_prev::V
    z_prev::V
    xtilde::V
    ztilde::V
    delta_x::V
    delta_y::V
    Ax::V
    Px::V
    Aty::V
    rhs_x::V
    rhs_z::V
    rho::T
    # `w = ρ`, `w_inv = ρ⁻¹` per row, `sigma = σ`; replaced whenever `σ` changes.
    weights::SystemWeights{T, V}
    constr_type::VI
    linsys::LS
    # `nothing` unless the caller supplied an accelerator. Its type is a parameter so the
    # per-iteration hooks dispatch statically and cost nothing when there is none.
    accel::AC
    refactor_count::Int
    prim_res::T
    dual_res::T
    scaled_prim_res::T
    scaled_dual_res::T
    obj_val::T
    dual_obj_val::T
    duality_gap::T
    scaled_duality_gap::T
    # The three terms of the gap, kept for its termination tolerance.
    xtPx::T
    qtx::T
    SCy::T
    rel_kkt_error::T
    last_rel_kkt::T
    # The primal-dual integral, accumulated only when `profile_primdual` is set. Two rules
    # over the same samples: `primdual_int` interpolates the gap linearly between them, as a
    # trapezoid; `primdual_int_log` interpolates it exponentially, which is what a
    # geometrically decaying gap actually does between samples. `last_gap_time` and
    # `last_gap` are the previous sample. Times are seconds since the loop started.
    primdual_int::Float64
    primdual_int_log::Float64
    last_gap_time::Float64
    last_gap::T
    # `solve!` stamps this before the loop, so the integral's clock starts where the loop
    # does rather than where the workspace was built.
    loop_start::UInt64
    rho_estimate::T
    rho_updates::Int
    accel_declined::Int
    cg_iters::Int
    iter::Int
    status::Status
    polished::Bool
    status_polish::PolishStatus
    setup_time::Float64
    update_time::Float64
    first_run::Bool
    solve_time::Float64
    polish_time::Float64
    algorithm::OperatorSplitting{T}
    options::Options{T}
    # Where the result is unscaled, in the workspace's own array type. The `Solution` holds
    # `Vector`s, so an array that forbids scalar indexing is scaled here and copied across
    # once rather than indexed element by element.
    xout::V
    yout::V
    # Refilled and handed back by every solve, so a solve allocates nothing at all. Its
    # arrays are plain `Vector`s whatever the workspace was built from: the result is what a
    # caller reads, not a buffer the solver iterates on, and a GPU array here would make
    # every field access a device transfer. `Solution` says what reuse means for a caller
    # holding one across a solve.
    sol::Solution{T}
end

"""
    OperatorSplittingWorkspace show, in one line: the shape, the backend the workspace is built on, and the
    state of its last run.
"""
function Base.show(io::IO, ws::OperatorSplittingWorkspace)
    print(
        io, "PureOSQP OperatorSplittingWorkspace: ", ws.prob.n, "×", ws.prob.m,
        ", backend ", backend_name(ws.linsys),
        ", status ", status_name(ws.status),
        ", rho ", ws.rho,
    )
    return nothing
end
function setup_backend(
        alg::OperatorSplitting, ::Val{LS}, ::Type{T}, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, options::Options,
        preconditioner, accelerator
    ) where {LS, T <: Real}
    t0 = time_ns()
    nv, mv = validate(P, q, A, l, u)
    algorithm = element_typed(alg, T, options)
    isnothing(preconditioner) || LS === :indirect || throw(
        ArgumentError(
            "preconditioner is used only by the matrix-free backend: pass linsys = :indirect with it."
        )
    )
    # A caller's preconditioner approximates the caller's own reduced matrix, which
    # equilibration would change underneath it. The two built-in ones need nothing from the
    # caller's matrices.
    preconditioner isa Union{Nothing, IdentityPreconditioner, JacobiPreconditioner} ||
        iszero(options.scaling) || throw(
        ArgumentError(
            "a caller-supplied preconditioner approximates P + sigma*I + A' * Diagonal(rho) * A " *
                "for the P and A passed to setup, which equilibration would change: pass scaling = 0 with it."
        )
    )
    if !is_convex(T, P, algorithm.sigma)
        throw(ArgumentError("P + sigma*I is not positive definite: P is indefinite, so the problem is not convex. Increase sigma if P + sigma*I can be made positive definite."))
    end
    # Equilibration and the ρ split run before the backend exists, because choosing a backend
    # well means building the reduced matrix and factoring it, and doing that with the values
    # the solver will actually use makes that factorization the setup factorization.
    prob = validated_problem(T, nv, mv, P, q, A, l, u, options.scaling)
    n, m, q0, l, u = prob.n, prob.m, prob.q0, prob.l, prob.u
    # A single definition, and no default argument: a local function assigned more than
    # once is boxed, which turns every call through it into a dynamic dispatch and makes
    # the entry points fail `--trim`.
    buf(k, v) = fill!(similar(q0, T, k), v)
    z = zero(T)
    o = one(T)
    ctype = fill!(similar(q0, Int8, m), zero(Int8))
    rho = clamp(algorithm.rho, RHO_MIN(T), RHO_MAX(T))
    rho_vec, rho_inv_vec = buf(m, o), buf(m, o)
    classify_rho!(
        ctype, rho_vec, rho_inv_vec, l, u, rho,
        INFTY(T) * MIN_SCALING(T), algorithm.rho_is_vec
    )
    ac = init_accelerator(accelerator, T, n, m)
    wt = SystemWeights(rho_vec, rho_inv_vec, algorithm.sigma)
    # `built`, not `ws`: assigning a name the enclosing function also assigns would capture
    # that variable, and a captured variable that is assigned is boxed.
    function make(ls)
        built = OperatorSplittingWorkspace{
            T, typeof(P), typeof(A), typeof(q0), typeof(ctype), typeof(ls), typeof(ac),
        }(
            prob,
            buf(n, z), buf(m, z), buf(m, z), buf(n, z), buf(m, z), buf(n, z), buf(m, z), buf(n, z), buf(m, z),
            buf(m, z), buf(n, z), buf(n, z),
            buf(n, z), buf(m, z),
            rho, wt, ctype,
            ls, ac, 0,
            zero(T), zero(T), zero(T), zero(T), zero(T),
            zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), INFTY(T),
            0.0, 0.0, 0.0, zero(T), zero(UInt64),
            algorithm.rho, 0, 0, 0, 0, UNSOLVED, false, POLISH_NOT_PERFORMED,
            0.0, 0.0, true, 0.0, 0.0,
            algorithm, options,
            buf(n, z), buf(m, z),
            # The certificates are reserved rather than sized: only one of them is reported,
            # and only by a run that ends infeasible, so their lengths are what a solve sets.
            empty_solution(
                Vector{T}(undef, n), Vector{T}(undef, m), reserved(T, m), reserved(T, n)
            ),
        )
        adopt_settings!(built.linsys, algorithm, options)
        return built
    end
    # `LS` is a type parameter, so a named backend leaves exactly one of `named_backend`'s
    # branches live and the rest are gone before the trimmer sees them. `options` still holds
    # and validates the same value; reading it back here instead would put the choice beyond
    # inference's reach and make every branch reachable again.
    named = named_backend(Val(LS), P, A, prob, wt, ADMMSelection(), preconditioner)
    if isnothing(named)
        # `choose_backend` picks by representation; the choice is settled here, once. The
        # backend is then part of the workspace's type, so the per-iteration solve dispatches
        # statically.
        ls, factored = choose_backend(P, A, prob, wt, ADMMSelection())
        ws = make(ls)
        # A factorization that fails here throws, as it does at every later refactorization,
        # and names `linsys = :kkt` as the remedy rather than switching backends unannounced.
        factored ? (ws.refactor_count += 1) : refactor!(ws)
        return finish_setup!(ws, t0)
    end
    ls, factored = named
    ws = make(ls)
    factored || refactor!(ws)
    return finish_setup!(ws, t0)
end

"Record how long `setup` took. Called on each of its return paths."
function finish_setup!(ws::OperatorSplittingWorkspace, t0::UInt64)
    ws.setup_time = (time_ns() - t0) / 1.0e9
    return ws
end

"""
    warm_start!(ws; x = nothing, y = nothing)

Seed the iterates in problem space. `z` is set to the scaled `Ax`.
"""
function warm_start!(ws::OperatorSplittingWorkspace{T}; x = nothing, y = nothing) where {T}
    prob = ws.prob
    if !isnothing(x)
        length(x) == prob.n || throw(ArgumentError("length(x) must be $(prob.n)"))
        all(isfinite, x) || throw(ArgumentError("x must be finite, found NaN or Inf"))
        ws.x .= T.(x) ./ prob.D
    end
    if !isnothing(y)
        length(y) == prob.m || throw(ArgumentError("length(y) must be $(prob.m)"))
        all(isfinite, y) || throw(ArgumentError("y must be finite, found NaN or Inf"))
        ws.y .= prob.c .* T.(y) ./ prob.E
    end
    prob.m > 0 && mul_A!(ws.z, prob, ws.x)
    return ws
end

"""
    cold_start!(ws) -> ws

Zero the iterates `x`, `y` and `z`, discarding whatever warm-start state the workspace
held. The equilibration factors, the factorization and the problem data are untouched, so
the next [`solve!`](@ref) restarts the ADMM iteration without rebuilding anything.

The solver already does this where it must: at the start of a solve when
`warm_starting = false`, and after a run that ended without a meaningful primal-dual point,
since those iterates lie on a diverging ray. Call it directly to discard a warm start you
no longer want — after a large change in the data, say, when the previous solution is a
worse starting point than the origin.
"""
function cold_start!(ws::OperatorSplittingWorkspace{T}) where {T}
    fill!(ws.x, zero(T))
    fill!(ws.y, zero(T))
    fill!(ws.z, zero(T))
    return ws
end
