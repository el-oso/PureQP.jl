"""
    InteriorPoint(; kwargs...)

The Mehrotra predictor–corrector interior-point method, passed as the algorithm of
[`setup`](@ref) and [`solve`](@ref). The keyword arguments are its parameters:

- `reg_primal = ipm_floor(T)`, `reg_dual = ipm_floor(T)` — the proximal regularization `δ_p`
  on the primal block and `δ_d` on each slack row, in scaled space. They are never refined
  away, and convexity is checked on `P + reg_primal*I`.
- `max_reg_bumps = 5` — when the Newton system cannot be factorized, both regularizations
  are multiplied by ten and the factorization retried, at most this many times in one solve;
  past that the run ends `NUMERICAL_ERROR`. Every solve starts from `reg_primal` and
  `reg_dual`.
- `refine_iter` — refinement steps per Newton solve against the regularized system, which
  correct rounding in the factorization: `1` by default, `0` with `linsys = :indirect`.
- `step_fraction = 0.99` — the fraction of the step to the boundary that is taken.
- `cg_fail_limit = 3` — this many missed conjugate-gradient solves in a row, with
  `linsys = :indirect`, end the run `NUMERICAL_ERROR`. A solve that spends `cg_max_iter`
  iterations, or that conjugate gradients abandons because the operator or the
  preconditioner is not positive definite, is a missed solve.

Its [`Options`](@ref) defaults are its own, and tighter than an operator-splitting method
would ask for: `max_iter = 100`,
`eps_abs`, `eps_rel`, `eps_prim_inf` and `eps_dual_inf` at `ipm_floor(T)`,
`check_termination = 1`, `cg_max_iter = 500` and `cg_tol_fraction = 0.1`, where each
conjugate-gradient solve stops once the two-norm of its recursively updated residual is
below that fraction of `min(μ, ‖r‖∞)`, the barrier parameter and the largest Newton
residual, floored at `eps(T)` relative to the right-hand side. With `linsys = :indirect`
it runs only with a caller-supplied `preconditioner` (see [`setup`](@ref)); `:kronecker` and
`:lowrank` are refused. `polishing = true` is required before taking a derivative: an
interior-point solution carries inactive-row multipliers of size `μ_final`, which is small
but not the near-zero [`active_kkt`](@ref) requires, and
[`adjoint_derivative`](@ref)/[`forward_derivative`](@ref) refuse an unpolished workspace.

A reduced backend solves `P̃ + δ_p I + Ãᵀ diag(w) Ã`, whose weights reach `1/δ_d` on
equality rows and on active inequality rows, so its conditioning is bounded by
`(λ_max(P̃) + ‖Ã‖²/δ_d) / (λ_min(P̃) + δ_p)`. That bound is not checked.

Any `T <: Real` is accepted. `ipm_floor(T)` is `1e-8` in `Float64` and finer arithmetic
(`BigFloat`) and `sqrt(eps(T))` in coarser arithmetic (`Float32`); a number type that wraps
another, such as `ForwardDiff.Dual`, takes the precision of `float(T)`. A value passed
explicitly is used as given.

The object is built without an element type, and a parameter left out holds `nothing` until
[`setup`](@ref) resolves it for the solve's element type and `linsys`; the workspace holds
the resolved `InteriorPoint{T, T, T, Int}`, every field concrete, as `ws.algorithm`.

`T` is the element type of `step_fraction`, which is always given. The three parameters
after it are the field types of `reg_primal`, `reg_dual` and `refine_iter`: `Nothing` while
the parameter is unresolved and `T`, `T`, `Int` once it is. They are internal — the
constructors compute them — and they exist so that every instance is concretely typed,
seeds included.
"""
struct InteriorPoint{T <: Real, RP, RD, RI} <: QPAlgorithm
    reg_primal::RP
    reg_dual::RD
    max_reg_bumps::Int
    refine_iter::RI
    step_fraction::T
    cg_fail_limit::Int
end

"The real type a parameter given as `x` is stored in; `nothing` leaves it to the default."
stored_real(x::Real) = typeof(x)
stored_real(::Nothing) = Float64

function InteriorPoint(;
        reg_primal = nothing, reg_dual = nothing, max_reg_bumps = 5, refine_iter = nothing,
        step_fraction = 0.99, cg_fail_limit = 3,
    )
    max_reg_bumps >= 0 || throw(ArgumentError(lazy"max_reg_bumps must be non-negative, got $max_reg_bumps"))
    isnothing(reg_primal) || reg_primal > 0 || throw(ArgumentError(lazy"reg_primal must be positive, got $reg_primal"))
    isnothing(reg_dual) || reg_dual > 0 || throw(ArgumentError(lazy"reg_dual must be positive, got $reg_dual"))
    isnothing(refine_iter) || refine_iter >= 0 || throw(ArgumentError(lazy"refine_iter must be non-negative, got $refine_iter"))
    0 < step_fraction < 1 || throw(ArgumentError(lazy"step_fraction must lie in (0, 1), got $step_fraction"))
    cg_fail_limit > 0 || throw(ArgumentError(lazy"cg_fail_limit must be positive, got $cg_fail_limit"))
    F = float(promote_type(stored_real(reg_primal), stored_real(reg_dual), typeof(step_fraction)))
    rp = isnothing(reg_primal) ? nothing : F(reg_primal)
    rd = isnothing(reg_dual) ? nothing : F(reg_dual)
    ri = isnothing(refine_iter) ? nothing : Int(refine_iter)
    # Straight to the inner constructor: the three trailing parameters are the field types
    # just computed, so there is nothing left for an outer method to narrow.
    return InteriorPoint{F, typeof(rp), typeof(rd), typeof(ri)}(
        rp, rd, Int(max_reg_bumps), ri, F(step_fraction), Int(cg_fail_limit),
    )
end

"""
    InteriorPoint{T}(a::InteriorPoint, linsys::Symbol)

`a` in element type `T`, with `reg_primal` and `reg_dual` left out resolved to `ipm_floor(T)`
and `refine_iter` left out to `0` under `linsys = :indirect` and `1` otherwise.
"""
function InteriorPoint{T}(a::InteriorPoint, linsys::Symbol) where {T <: Real}
    floor = ipm_floor(T)
    return InteriorPoint{T, T, T, Int}(
        isnothing(a.reg_primal) ? floor : T(a.reg_primal),
        isnothing(a.reg_dual) ? floor : T(a.reg_dual),
        a.max_reg_bumps,
        isnothing(a.refine_iter) ? (linsys === :indirect ? 0 : 1) : a.refine_iter,
        T(a.step_fraction), a.cg_fail_limit,
    )
end

const INTERIOR_POINT_NAMES = fieldnames(InteriorPoint)

element_typed(a::InteriorPoint, ::Type{T}, options::Options) where {T} = InteriorPoint{T}(a, options.linsys)

function algorithm_defaults(::InteriorPoint, ::Type{T}) where {T}
    f = ipm_floor(T)
    return (
        max_iter = 100, eps_abs = f, eps_rel = f, eps_prim_inf = f, eps_dual_inf = f,
        check_termination = 1, cg_max_iter = 500, cg_tol_fraction = 0.1,
    )
end

"""
    precision_eps(T) -> eps

The spacing of `T`'s arithmetic, `eps(float(T))`: the value type's `eps` for a number type
that wraps one and defines `eps` through it (as `ForwardDiff.Dual` does), `Float64`'s for
integers and rationals.
"""
@inline precision_eps(::Type{T}) where {T <: Real} = eps(float(T))

"""
    ipm_floor(T) -> T

The default of the interior-point regularizations, tolerances and short-step threshold:
`1e-8` in `Float64` and finer arithmetic, `sqrt(eps)` in arithmetic coarser than `Float64`.
"""
@inline function ipm_floor(::Type{T}) where {T <: Real}
    e = precision_eps(T)
    return e > eps(Float64) ? T(sqrt(e)) : T(1.0e-8)
end
