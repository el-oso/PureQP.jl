"""
    ActiveSet(; kwargs...)

A dual active-set method, passed as the algorithm of [`setup`](@ref) and
[`solve`](@ref). The keyword arguments are its parameters:

- `eps_prox = 0` — the proximal regularization `ε`. Zero runs the method on its own, which
  needs `P ≻ 0`. Any positive value runs outer proximal-point iterations instead, solving a
  sequence of problems in `P + εI`, which accepts a singular `P` and improves conditioning.
  `1e-4` is the value the method's authors report. A [`PureQPBase.KroneckerOperator`](@ref) `P`
  takes a positive value through a [`PureQPBase.KroneckerSquareRoot`](@ref), which factors
  `P₁ ⊗ P₂ + εI` from the factors' eigendecompositions; that matrix is not a Kronecker product,
  so the factor is not `R₁ ⊗ R₂` and not triangular, and neither property is one the reduction
  needs.
- `eta_prox = sqrt(eps(T))` — the proximal-point loop stops once the iterate moves less than
  this in the ∞-norm, relative to the size of the iterate.
- `max_prox = 100` — the most outer proximal-point iterations in one solve. The loop contracts
  at a rate set by `eps_prox` against the curvature of `P`, so an `eps_prox` well above
  `λ_min(P)` needs more passes than this: raise it, or lower `eps_prox`, if a solve of a
  nearly singular `P` returns `MAX_ITER_REACHED` having moved steadily toward the answer.
- `zero_tol = sqrt(eps(T))` — a diagonal entry of the working set's `LDLᵀ` at or below this
  marks a linearly dependent row, which sends the iteration down its singular branch.
- `primal_tol = sqrt(eps(T))` — a row priced below `-primal_tol` is violated and enters the
  working set. It is absolute, in the units of the row's value `aᵢ·x`: a violation of a row whose
  values are of the order of `primal_tol`, such as a row of norm `1e-8`, is never priced in, so
  scale such a row up. It also bounds how far a row of the answer may leave its bounds,
  `primal_tol·(max(1, ‖aᵢ‖₂‖x‖∞) + |zᵢ|)`, and an answer that leaves a row by more ends the run
  as `NUMERICAL_ERROR`.
- `infeasibility_radius = primal_tol^(-1/2)` — `PRIMAL_INFEASIBLE` is reported when the ray
  that proves it rules out every `x` with `‖x‖∞` below this, in the units of `x`. A ray whose
  `Aᵀy` is not zero proves only that much, and a problem that is feasible only at a point
  larger than the radius is reported infeasible within it. A ray with a smaller radius ends the
  run as `NUMERICAL_ERROR`: it could not be decided. Raise it when the variables are of a size
  that makes the default too small, for example variables in the millions, and it must be
  finite and positive.
- `working_set = :rows` — what the working set factors. `:rows` factors the active rows
  themselves, and decides dependence on how much of an entering row is orthogonal to those
  already held, which carries the conditioning of `M` once. `:gram` factors their Gram
  matrix `Mₐ Mₐᵀ` instead — the normal equations of the same rows — which is the faster of
  the two but squares that conditioning. A new pivot small enough for rounding to dominate it
  is recomputed from the explicit residual of the entering row against the rows held, and
  the multipliers still carry the squared conditioning. Prefer `:gram` only for problems
  measured to be well conditioned; on one that is not, it can refuse a problem the rows
  solve. The manual's section on choosing between them says how to tell.
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
  problem of yours and says which to pass. When `M` is not stored (below), one product prices
  every row, so a window saves nothing; and in an iteration where the window holds no violated
  row, every row is priced a second time before the run ends, so `:window` can cost more than
  `:all` there.

With `eps_prox > 0`, a problem that is unbounded below keeps the proximal-point loop from
settling. When the loop ends on `max_prox` passes and its last step `d` satisfies `qᵀd < 0` and
`Pd = 0` exactly, and no row moves toward a bound that is present, the status is
`DUAL_INFEASIBLE` and `Solution.dual_inf_cert` holds `d`, scaled to unit ∞-norm. A direction
that is exact only to rounding leaves the status at `MAX_ITER_REACHED`, because a tolerance
cannot tell it from a bounded problem with a distant minimizer. Without `eps_prox` the method
needs `P ≻ 0` and cannot be unbounded.

The method reduces the problem to a least-distance problem, `min ‖u‖²` subject to
`lo ≤ Mu ≤ hi` with `M = A R⁻¹` for the Cholesky factor `R` of `P` (or of `P + εI`), and solves
that by maintaining an `LDLᵀ` of the Gram matrix of the working set under rank-one updates.

What it keeps of `M` depends on `P` and `A`, and is not a setting. When `A` is a dense matrix
and `R` is a dense triangular factor, `M` is formed once and stored, `m×n`. In every other case
`M` is never formed: the reduction holds `A` and `R`, where `R` is the Cholesky factor in `P`'s
own form ([`PureQPBase.cholesky_factor`](@ref)), and derives a row of `M` as one row of `A`
([`PureQPBase.dense_row!`](@ref)) and one triangular solve, and `Mu` as one solve and one
product with `A`. Storage then grows with `m` only through vectors of length `m`, not through
`m·n`; the working set's own factor, which is `n × (min(m, n) + 1)`, does not depend on `m`
once `m ≥ n`. This applies to a `Diagonal`, a
[`PureQPBase.BlockDiagonal`](@ref) or a [`PureQPBase.KroneckerOperator`](@ref) in either
position, and to an `A` that supplies products only, such as a `LinearMap`.

Two requirements follow, and neither is a setting. `P` must have a Cholesky factor: a dense
matrix, a `Diagonal`, a `BlockDiagonal` of dense blocks, or a `KroneckerOperator` of two dense
matrices, at any `eps_prox`. An operator that supplies products only is refused as `P`, by
name. `A` has no such requirement, since it is only multiplied and read by row, but an operator
`A` needs a transpose. **A sparse or banded `P` or `A`, and a `RowCoupled` one, is read into a
dense matrix**, since there is no sparse factor and no row read in `O(nnz)`.

Of the options shared with the other algorithms, `max_iter` is the only one this method reads:
`eps_abs`, `eps_rel`, `eps_prim_inf` and `eps_dual_inf` decide nothing here, and `primal_tol`,
`zero_tol` and `infeasibility_radius` decide its statuses. A singular `P` needs `eps_prox` above
about `eps·‖P‖`, or `setup` refuses it as not positive definite.

`linsys` is refused other than `:auto` and `:dense`: the method has no choice of backend to
make. `scaling` must be `0` — the reduction does its own row normalization.
"""
struct ActiveSet{T <: Real, EE, EZ, EP, ER, WS} <: QPAlgorithm
    eps_prox::T
    eta_prox::EE
    max_prox::Int
    zero_tol::EZ
    primal_tol::EP
    infeasibility_radius::ER
    working_set::Symbol
    scan::Symbol
end

"""
The working set's type, as the sixth type parameter of `a`.

The reduction carries its working set as a type parameter, and the factor `P` reduces through
is one of two types for a Kronecker `P`, chosen by whether its blocks are positive definite.
Two choices against two is four, and inference keeps two concrete results but widens four to
`DAQPReduction{T}` — which `--trim` cannot resolve. Carrying the working set in the algorithm's
type leaves one choice to make while the reduction is built, so `setup` stays trim-compatible.
`working_set` reads back the `Symbol` the caller passed.
"""
working_set_param(::ActiveSet{T, EE, EZ, EP, ER, WS}) where {T, EE, EZ, EP, ER, WS} = WS

"The real type a parameter given as `x` is stored in; `nothing` leaves it to the default."
stored_real(x::Real) = typeof(x)
stored_real(::Nothing) = Float64

# `:aggressive` so a literal `working_set` resolves the type parameter at the call site: this
# returns one of two types, and a caller that names the working set should get one of them
# rather than their union.
Base.@constprop :aggressive function ActiveSet(;
        eps_prox = 0.0, eta_prox = nothing, max_prox = 100,
        zero_tol = nothing, primal_tol = nothing, infeasibility_radius = nothing,
        working_set = :rows, scan = :all,
    )
    working_set in (:rows, :gram) ||
        throw(ArgumentError(lazy"working_set must be :rows or :gram, got $(repr(working_set))"))
    scan in (:all, :window) ||
        throw(ArgumentError(lazy"scan must be :all or :window, got $(repr(scan))"))
    eps_prox >= 0 || throw(ArgumentError(lazy"eps_prox must be non-negative, got $eps_prox"))
    max_prox > 0 || throw(ArgumentError(lazy"max_prox must be positive, got $max_prox"))
    isnothing(eta_prox) || eta_prox > 0 || throw(ArgumentError(lazy"eta_prox must be positive, got $eta_prox"))
    isnothing(zero_tol) || zero_tol > 0 || throw(ArgumentError(lazy"zero_tol must be positive, got $zero_tol"))
    isnothing(primal_tol) || primal_tol > 0 || throw(ArgumentError(lazy"primal_tol must be positive, got $primal_tol"))
    isnothing(infeasibility_radius) || (isfinite(infeasibility_radius) && infeasibility_radius > 0) ||
        throw(ArgumentError(lazy"infeasibility_radius must be finite and positive, got $infeasibility_radius"))
    F = float(
        promote_type(
            stored_real(eps_prox), stored_real(eta_prox), stored_real(zero_tol),
            stored_real(primal_tol), stored_real(infeasibility_radius)
        )
    )
    et = isnothing(eta_prox) ? nothing : F(eta_prox)
    zt = isnothing(zero_tol) ? nothing : F(zero_tol)
    pt = isnothing(primal_tol) ? nothing : F(primal_tol)
    rt = isnothing(infeasibility_radius) ? nothing : F(infeasibility_radius)
    # Each optional tolerance carries its own field type, so any mix of given and defaulted
    # is representable and every instance stays concretely typed. One branch per working set,
    # so the type parameter is a constant in each: the branches rejoin as a two-way union
    # here, which inference keeps, and `setup` is reached with one of them.
    ws = Symbol(working_set)
    args = (F(eps_prox), et, Int(max_prox), zt, pt, rt, ws, Symbol(scan))
    ws === :rows && return ActiveSet{F, typeof(et), typeof(zt), typeof(pt), typeof(rt), WorkingSetQR}(args...)
    return ActiveSet{F, typeof(et), typeof(zt), typeof(pt), typeof(rt), WorkingSetGram}(args...)
end

"""
    ActiveSet{T}(a::ActiveSet)

`a` in element type `T`, with every tolerance left out resolved to `sqrt(eps(T))`, and the
infeasibility radius to the inverse square root of the resolved `primal_tol`.
"""
function ActiveSet{T}(a::ActiveSet) where {T <: Real}
    tol = sqrt(eps(float(T)))
    primal_tol = isnothing(a.primal_tol) ? T(tol) : T(a.primal_tol)
    # The working set is carried over rather than re-derived: it is already a parameter of `a`,
    # so this stays one concrete type rather than reopening the choice.
    return ActiveSet{T, T, T, T, T, working_set_param(a)}(
        T(a.eps_prox),
        isnothing(a.eta_prox) ? T(tol) : T(a.eta_prox),
        a.max_prox,
        isnothing(a.zero_tol) ? T(tol) : T(a.zero_tol),
        primal_tol,
        isnothing(a.infeasibility_radius) ? inv(sqrt(primal_tol)) : T(a.infeasibility_radius),
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
            infeasibility_radius = alg.infeasibility_radius,
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
