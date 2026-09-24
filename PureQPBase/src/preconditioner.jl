"""
    update_preconditioner!(M, prob, wt, k::Int) -> M

Refresh the preconditioner of `P + wt.sigma*I + A' * Diagonal(wt.w) * A`, where `P` and `A`
are the matrices or operators passed to [`setup`](@ref) (a caller-supplied preconditioner
requires `scaling = 0`, so no equilibration intervenes). `wt.w` and `wt.sigma` are the only
documented reads of `wt`; `prob` is passed for dispatch and is not part of the documented
interface.

`k` is the refresh index the algorithm sets before calling: under ADMM, the number of
refactorizations so far; under the interior-point method, `-1` for the starting-point solve and the
outer iteration `0, 1, 2, …` afterwards, where a regularization retry calls again with the
same `k` and a larger `wt.sigma`.

Called from the matrix-free backend's `factorize!` and `refactor_weights!`, before the solves
that use it. The returned object replaces `M` and must have the same type; another type throws
an `ArgumentError`. Refresh lazily by returning `M` unchanged, for instance except when `k` is
negative, a multiple of three, or `wt.sigma` changed.

`M` is applied through `LinearAlgebra.ldiv!(y, M, x)`, so a `Cholesky`, `LDLt` or any
factorization object is usable as it is, and `M` must be symmetric positive definite. The
default method never refreshes.
"""
update_preconditioner!(M, prob, wt, k::Int) = M

"""
    Preconditioner

The built-in preconditioners of the matrix-free backend, [`IdentityPreconditioner`](@ref) and
[`JacobiPreconditioner`](@ref). A preconditioner is used through two methods:
[`update_preconditioner!`](@ref), whose default never refreshes, and
`LinearAlgebra.ldiv!(y, M, x)`, which has no default.

A caller's preconditioner need not be a subtype, so a `Cholesky` or any other factorization
object is usable as it is; `TypeContracts.check_contract(typeof(M), Preconditioner)` checks one
against the same contract. [`setup`](@ref) refuses a preconditioner with no `ldiv!` method for
the backend's vectors.
"""
abstract type Preconditioner end

# A StrictMode contract: `ldiv!` runs once per conjugate-gradient iteration and is held to
# allocation-free, type-stable code on top of the method surface TypeContracts checks.
@strict_contract Preconditioner begin
    update_preconditioner!(::Self, ::Problem, ::SystemWeights, ::Int)::Self => "refresh for the current weights and return the preconditioner, of the same type"
    LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector) => "write the preconditioned vector into `y`"
end

"""
    check_preconditioner(M, V) -> Nothing

Throw unless `ldiv!(y::V, M, x::V)` has a method, `V` being the matrix-free backend's vector
type. `nothing` selects the default preconditioner and passes. [`update_preconditioner!`](@ref)
is not checked, since its default applies to any object.
"""
function check_preconditioner(M, ::Type{V}) where {V}
    isnothing(M) && return nothing
    hasmethod(LinearAlgebra.ldiv!, Tuple{V, typeof(M), V}) || throw(
        ArgumentError(
            lazy"the preconditioner, a $(typeof(M)), has no method LinearAlgebra.ldiv!(y::$V, M, x::$V), through which conjugate gradients applies it: define one."
        )
    )
    return nothing
end

"""
    IdentityPreconditioner()

No preconditioning: conjugate gradients on the reduced system as it stands. Needs nothing
from `P` or `A`, so it runs with equilibration on. The matrix-free backend skips it rather
than calling `ldiv!`, which copies.
"""
struct IdentityPreconditioner <: Preconditioner end

LinearAlgebra.ldiv!(y::AbstractVector, ::IdentityPreconditioner, x::AbstractVector) = copyto!(y, x)

"""
    JacobiPreconditioner(dinv)

The inverted diagonal of the reduced matrix, which [`reduced_diagonal!`](@ref) fills at every
[`update_preconditioner!`](@ref) from the current weights. It is the matrix-free backend's
default, and it runs with equilibration on.

`ldiv!(y, J, x)` computes `y = dinv .* x`. It multiplies by the stored reciprocal rather than
dividing by the diagonal, and the two differ in the last bit on about a quarter of entries,
so a `Diagonal` of either vector is not a substitute.
"""
mutable struct JacobiPreconditioner{V <: AbstractVector} <: Preconditioner
    const dinv::V
end

LinearAlgebra.ldiv!(y::AbstractVector, J::JacobiPreconditioner, x::AbstractVector) =
    multiply!(y, J.dinv, x)

function update_preconditioner!(J::JacobiPreconditioner, prob, wt, k::Int)
    reduced_diagonal!(
        J.dinv, eltype(J.dinv), prob.P, prob.A, wt.w, prob.E, prob.D, wt.sigma, prob.c
    )
    return J
end

"""
    set_refresh_index!(ls, k) -> Nothing

Hand the backend the refresh index its next [`update_preconditioner!`](@ref) passes on. A
backend without a preconditioner ignores it, which is the default.
"""
set_refresh_index!(ls::LinearSystem, k::Int) = nothing

"""
    use_residual_stop!(ls, on::Bool) -> Nothing

Choose the inner stopping rule of an iterative backend. Off (the default), conjugate gradients
stops at the absolute tolerance on its preconditioned residual norm. On, it stops once the
two-norm of its recursively updated, unpreconditioned residual reaches the tolerance, or at
its own machine-precision stop. A direct backend ignores it.
"""
use_residual_stop!(ls::LinearSystem, on::Bool) = nothing

"""
    last_solve_converged(ls) -> Bool

Whether the backend's most recent solve met its stopping test. An iterative solve that spent
its whole iteration budget, or broke down, did not. A direct backend always does.
"""
last_solve_converged(ls::LinearSystem) = true

"""
    inner_iterations(ls) -> Int

The inner iterations an iterative backend has spent over its life; zero for a direct one.
`Solution.cg_iters` is its difference across one solve.
"""
inner_iterations(ls::LinearSystem) = 0
