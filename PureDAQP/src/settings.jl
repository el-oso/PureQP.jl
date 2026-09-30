"""
    ActiveSet(; kwargs...)

A dual active-set method, passed as the algorithm of [`setup`](@ref) and
[`solve`](@ref). The keyword arguments are its parameters:

- `eps_prox = 0` — the proximal regularization `ε`. Zero runs the method on its own, which
  needs `P ≻ 0`. Any positive value runs outer proximal-point iterations instead, solving a
  sequence of problems in `P + εI`, which accepts a singular `P` and improves conditioning.
  `1e-4` is the value the method's authors report.
- `eta_prox = sqrt(eps(T))` — the proximal-point loop stops once the iterate moves less than
  this in the ∞-norm.
- `max_prox = 100` — the most outer proximal-point iterations in one solve.
- `zero_tol = sqrt(eps(T))` — a diagonal entry of the working set's `LDLᵀ` at or below this
  marks a linearly dependent row, which sends the iteration down its singular branch.
- `primal_tol = sqrt(eps(T))` — a row priced below `-primal_tol` is violated and enters the
  working set. Rows are normalized first, so this means the same thing on every row.
- `working_set = :rows` — what the working set factors. `:rows` factors the active rows
  themselves, and decides dependence on how much of an entering row is orthogonal to those
  already held, which carries the conditioning of `M` once. `:gram` factors their Gram
  matrix `Mₐ Mₐᵀ` instead — the normal equations of the same rows — which is the faster of
  the two but squares that conditioning, so it cannot separate a dependent row from an
  independent one much below `sqrt(k·eps)`. Prefer `:gram` only for problems measured to be
  well conditioned; on one that is not, it reports a feasible problem infeasible rather than
  solving it. The manual's section on choosing between them says how to tell.
- `scan = :all` — how many rows are examined to choose the one that enters. `:all` examines
  every row and takes the worst violator. `:window` examines a window of them, resumed where
  the last row entered, and takes the worst violator within it; it still examines every row
  before ending a run, so it cannot stop early or miss a violated row. A window makes an
  iteration cheaper and, by entering a row that is not the worst, usually costs iterations.
  Which effect wins is not predictable from the problem's dimensions: measured across eight
  problems, `:window` ranges from 1.8× faster to 1.6× slower, and neither the ratio of rows
  to active rows, nor how often the window finds a violator, nor how good its choice is
  separates the two outcomes. So there is no automatic setting, and `:all` is the default
  because it is the one that is never much worse. [`faster_scan`](@ref) measures both on a
  problem of yours and says which to pass.

The method reduces the problem to a least-distance problem, `min ‖u‖²` subject to `Mu ≤ d`
with `M = A R⁻¹` for the Cholesky factor `R` of `P` (or of `P + εI`), and solves that by
maintaining an `LDLᵀ` of the Gram matrix of the working set under rank-one updates.

Two consequences follow from that reduction and are not settings you can turn off. `P` must
have a Cholesky factor, so an operator that supplies products only is refused by name; and
**a sparse `P` or `A` is read into a dense matrix**, since the factor and the row reads are
dense either way. A structured or unmaterialized pair is not: `M` is then held as `A` and `R`
rather than formed, and each row and each product is derived from them.

`linsys` is refused other than `:auto` and `:dense`: the method has no choice of backend to
make. `scaling` must be `0` — the reduction does its own row normalization.
"""
struct ActiveSet{T <: Real, EE, EZ, EP} <: QPAlgorithm
    eps_prox::T
    eta_prox::EE
    max_prox::Int
    zero_tol::EZ
    primal_tol::EP
    working_set::Symbol
    scan::Symbol
end

"The real type a parameter given as `x` is stored in; `nothing` leaves it to the default."
stored_real(x::Real) = typeof(x)
stored_real(::Nothing) = Float64

function ActiveSet(;
        eps_prox = 0.0, eta_prox = nothing, max_prox = 100,
        zero_tol = nothing, primal_tol = nothing, working_set = :rows, scan = :all,
    )
    working_set in (:rows, :gram) ||
        throw(ArgumentError("working_set must be :rows or :gram, got $(repr(working_set))"))
    scan in (:all, :window) ||
        throw(ArgumentError("scan must be :all or :window, got $(repr(scan))"))
    eps_prox >= 0 || throw(ArgumentError("eps_prox must be non-negative, got $eps_prox"))
    max_prox > 0 || throw(ArgumentError("max_prox must be positive, got $max_prox"))
    isnothing(eta_prox) || eta_prox > 0 || throw(ArgumentError("eta_prox must be positive, got $eta_prox"))
    isnothing(zero_tol) || zero_tol > 0 || throw(ArgumentError("zero_tol must be positive, got $zero_tol"))
    isnothing(primal_tol) || primal_tol > 0 || throw(ArgumentError("primal_tol must be positive, got $primal_tol"))
    F = float(
        promote_type(
            stored_real(eps_prox), stored_real(eta_prox),
            stored_real(zero_tol), stored_real(primal_tol)
        )
    )
    et = isnothing(eta_prox) ? nothing : F(eta_prox)
    zt = isnothing(zero_tol) ? nothing : F(zero_tol)
    pt = isnothing(primal_tol) ? nothing : F(primal_tol)
    # Each optional tolerance carries its own field type, so any mix of given and defaulted
    # is representable and every instance stays concretely typed.
    return ActiveSet{F, typeof(et), typeof(zt), typeof(pt)}(
        F(eps_prox), et, Int(max_prox), zt, pt, Symbol(working_set), Symbol(scan)
    )
end

"""
    ActiveSet{T}(a::ActiveSet)

`a` in element type `T`, with every tolerance left out resolved to `sqrt(eps(T))`.
"""
function ActiveSet{T}(a::ActiveSet) where {T <: Real}
    tol = sqrt(eps(float(T)))
    return ActiveSet{T, T, T, T}(
        T(a.eps_prox),
        isnothing(a.eta_prox) ? T(tol) : T(a.eta_prox),
        a.max_prox,
        isnothing(a.zero_tol) ? T(tol) : T(a.zero_tol),
        isnothing(a.primal_tol) ? T(tol) : T(a.primal_tol),
        a.working_set,
        a.scan,
    )
end

const ACTIVE_SET_NAMES = fieldnames(ActiveSet)

element_typed(a::ActiveSet, ::Type{T}, ::Options) where {T} = ActiveSet{T}(a)

function algorithm_defaults(::ActiveSet, ::Type{T}) where {T}
    tol = sqrt(eps(float(T)))
    # An active-set method stops at an exact vertex of the working set rather than at a
    # tolerance, so the residual tolerances only decide when a run is called inaccurate.
    # `check_termination = 1` because there is no periodic test: the run ends when no row
    # prices in. `cg_*` exist only for the matrix-free backend, which this method has not.
    return (
        max_iter = 1000, eps_abs = T(tol), eps_rel = T(tol),
        eps_prim_inf = T(tol), eps_dual_inf = T(tol), scaling = 0,
        check_termination = 1, cg_max_iter = 1, cg_tol_fraction = T(tol),
    )
end

"""
    faster_scan(P, q, A, l, u; alg = ActiveSet(), reps = 5, kwargs...) -> NamedTuple

Solve a problem both ways and report which [`ActiveSet`](@ref) `scan` setting is faster on it.

Returns `(; scan, all_ms, window_ms, ratio, iter_all, iter_window)`, where `scan` is the value
to pass and `ratio` is how much faster it is than the other. `kwargs` reach [`solve`](@ref),
so options such as `max_iter` carry over.

There is no rule that predicts this from a problem's dimensions -- across eight problems
`:window` ranged from 1.8× faster to 1.6× slower, and no cheap property of the problem
separated those cases -- so measuring one representative problem is the way to decide. The
answer then holds for problems of that shape and conditioning, not for a different family.

Each variant is solved `reps` times and the fastest run of each is compared, which is enough
to separate the large differences and not the small ones. When `ratio` is near one the setting
does not matter; prefer `:all`, which is the default and the one that is never much worse.

Both variants must reach the same status, or this throws: a setting is not faster if it does
not solve the problem.
"""
function faster_scan(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector;
        alg::ActiveSet = ActiveSet(), reps::Integer = 5, kwargs...
    )
    reps >= 1 || throw(ArgumentError(lazy"reps must be at least 1, got $reps"))
    best = Dict{Symbol, Float64}()
    seen = Dict{Symbol, Any}()
    for which in (:all, :window)
        a = ActiveSet(;
            eps_prox = alg.eps_prox, eta_prox = alg.eta_prox, max_prox = alg.max_prox,
            zero_tol = alg.zero_tol, primal_tol = alg.primal_tol,
            working_set = alg.working_set, scan = which,
        )
        sol = solve(P, q, A, l, u, a; kwargs...)   # also the warm-up for the timing below
        t = Inf
        for _ in 1:reps
            t = min(t, @elapsed solve(P, q, A, l, u, a; kwargs...))
        end
        best[which] = 1000 * t
        seen[which] = sol
    end
    seen[:all].status == seen[:window].status || throw(
        ArgumentError(
            lazy"the two settings disagree about this problem: scan = :all gives $(seen[:all].status) and scan = :window gives $(seen[:window].status), so neither is simply faster"
        )
    )
    pick = best[:all] <= best[:window] ? :all : :window
    return (
        scan = pick,
        all_ms = best[:all], window_ms = best[:window],
        ratio = max(best[:all], best[:window]) / min(best[:all], best[:window]),
        iter_all = seen[:all].iter, iter_window = seen[:window].iter,
    )
end
