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
- `working_set = :qr` — how the working set is represented. `:qr` factors the active rows
  themselves and decides dependence on `|R_ii|`, which carries the conditioning of `M` once.
  `:gram` factors their Gram matrix instead, which is the faster of the two but squares that
  conditioning, so it cannot separate a dependent row from an independent one much below
  `sqrt(k·eps)`. Prefer `:gram` only for problems measured to be well conditioned; on one
  that is not, it reports a feasible problem infeasible rather than solving it.

The method reduces the problem to a least-distance problem, `min ‖u‖²` subject to `Mu ≤ d`
with `M = A R⁻¹` for the Cholesky factor `R` of `P` (or of `P + εI`), and solves that by
maintaining an `LDLᵀ` of the Gram matrix of the working set under rank-one updates.

Two consequences follow from that reduction and are not settings you can turn off. `M` is
dense whatever `A` was, so **this algorithm ignores sparsity and declared structure**; and
because the working set is tracked one side at a time, each two-sided row becomes two rows
internally.

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
end

"The real type a parameter given as `x` is stored in; `nothing` leaves it to the default."
stored_real(x::Real) = typeof(x)
stored_real(::Nothing) = Float64

function ActiveSet(;
        eps_prox = 0.0, eta_prox = nothing, max_prox = 100,
        zero_tol = nothing, primal_tol = nothing, working_set = :qr,
    )
    working_set in (:qr, :gram) ||
        throw(ArgumentError("working_set must be :qr or :gram, got $(repr(working_set))"))
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
        F(eps_prox), et, Int(max_prox), zt, pt, Symbol(working_set)
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
