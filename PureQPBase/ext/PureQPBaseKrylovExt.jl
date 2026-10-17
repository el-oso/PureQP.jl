"""
    PureQPBaseKrylovExt

The matrix-free linear-system backend, loaded when Krylov.jl is.

Every other backend forms the reduced matrix `P̃ + σI + Ãᵀ diag(ρ) Ã` and factors it. This
one never forms it: it applies it through the same `mul_A!`, `mul_At!` and `mul_P!` the
iteration already uses, so a problem that can only supply matrix-vector products is
solvable. Selected with `linsys = :indirect`.

Krylov.jl is a weak dependency, not a core one. It has no non-stdlib dependencies of its
own, but it costs about 185 ms to load against the 76 ms of this whole package, and a
direct factorization is some two orders of magnitude faster whenever the matrix *can* be
formed. Nobody should pay that who is not using this backend.
"""
module PureQPBaseKrylovExt

using PureQPBase: PureQPBase, LinearSystem, SystemWeights, mul_A!, mul_At!, mul_P!, norm_inf
using Krylov: Krylov, CgWorkspace, cg!
using LinearAlgebra: LinearAlgebra, mul!
using TypeContracts: TypeContracts, @verify

"""
    ReducedOperator{T,PB,WT}

`x ↦ (P̃ + σI + Ãᵀ diag(w) Ã) x` for a problem and its weights, applied without forming
anything.

Built inside `solve_system!` from the problem and weights it is given, so it is concretely
typed at the call site.
"""
struct ReducedOperator{T <: Real, PB <: PureQPBase.Problem{T}, WT <: SystemWeights{T}}
    prob::PB
    wt::WT
end

# `T` is carried in the operator's own type. Krylov checks `eltype(A)` against the vectors'
# and falls back to a slower, allocating path when they disagree, so this cannot be left to
# a generic `eltype` of the problem type.
ReducedOperator(prob::PureQPBase.Problem{T}, wt::SystemWeights{T}) where {T} =
    ReducedOperator{T, typeof(prob), typeof(wt)}(prob, wt)

Base.size(op::ReducedOperator) = (op.prob.n, op.prob.n)
Base.size(op::ReducedOperator, d::Integer) = op.prob.n
Base.eltype(::ReducedOperator{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, op::ReducedOperator, x::AbstractVector)
    prob, wt = op.prob, op.wt
    # `mul_A!` and `mul_At!` use `prob.tmp_m`/`prob.tmp_n` as scratch, so `work_m` carries
    # the intermediate here rather than aliasing theirs.
    if prob.m > 0
        mul_A!(prob.work_m, prob, x)
        PureQPBase.multiply!(prob.work_m, prob.work_m, wt.w)
        mul_At!(y, prob, prob.work_m)
    else
        fill!(y, zero(eltype(y)))
    end
    mul_P!(prob.work_n, prob, x)
    PureQPBase.add_scaled!(y, prob.work_n, wt.sigma, x)
    return y
end

"""
    IndirectCG{T,V,K,M} <: LinearSystem

Conjugate gradients on the reduced system, preconditioned by `precond::M`.

`factorize!` refreshes the preconditioner rather than building a factorization, through
`PureQPBase.update_preconditioner!` with the refresh index last set by
`PureQPBase.set_refresh_index!`. The default `JacobiPreconditioner` holds the inverted diagonal
of the reduced matrix, which is computable column by column without assembling the matrix
itself. The Krylov workspace is allocated once and reused, so the per-iteration solve
allocates nothing.

`level` is the residual level the next solve's tolerance is relative to, set through
`PureQPBase.set_tolerance_level!`. `max_iter` and `tol_fraction` are the workspace's
`cg_max_iter` and `cg_tol_fraction` options and `tol_reduction` its algorithm's
`cg_tol_reduction`, copied in by `PureQPBase.adopt_settings!` when the workspace is built and
whenever its options or algorithm parameters are replaced.

`total_iters` counts CG iterations over the backend's life and `misses` the solves that did
not meet their stopping test; `last_reached` is whether the most recent one did.
`residual_stop` selects the stopping rule, set through `PureQPBase.use_residual_stop!`.
"""
mutable struct IndirectCG{T <: Real, V <: AbstractVector{T}, K, M} <: LinearSystem
    const kws::K        # Krylov's CgWorkspace, reused across solves
    const rhs::V
    precond::M
    reduction::T        # multiplies the tolerance; halved when CG stops iterating
    idle_solves::Int    # consecutive solves in which CG took no iteration
    level::T
    max_iter::Int
    tol_fraction::T
    tol_reduction::Int
    refresh_index::Int
    total_iters::Int
    misses::Int
    last_reached::Bool
    residual_stop::Bool
end

function PureQPBase.indirect_backend(
        proto::AbstractVector{T}, n::Integer, m::Integer, preconditioner
    ) where {T <: Real}
    kws = CgWorkspace(n, n, typeof(similar(proto, T, n)))
    # `cg!` allocates its preconditioned vector on first use, when it finds the field empty.
    # Filling it here is what makes the very first solve allocation-free, not merely every
    # solve after the first.
    kws.z = similar(proto, T, n)
    precond = isnothing(preconditioner) ?
        PureQPBase.JacobiPreconditioner(fill!(similar(proto, T, n), one(T))) : preconditioner
    precond isa PureQPBase.JacobiPreconditioner && length(precond.dinv) != n && throw(
        DimensionMismatch("a JacobiPreconditioner needs one entry per variable")
    )
    # The level and the settings hold placeholders until `set_tolerance_level!` and
    # `adopt_settings!` fill them, which `setup` and `admm_step!` do before any solve.
    return IndirectCG{T, typeof(similar(proto, T, n)), typeof(kws), typeof(precond)}(
        kws, similar(proto, T, n), precond, one(T), 0,
        zero(T), 0, zero(T), 0,
        0, 0, 0, true, false,
    )
end

# Krylov tests `M === I` and then skips the preconditioned vector entirely, which is exact
# and cheaper than copying the residual into it.
krylov_preconditioner(M) = M
krylov_preconditioner(::PureQPBase.IdentityPreconditioner) = LinearAlgebra.I

PureQPBase.backend_name(::IndirectCG) = :indirect

# Matrix-free: the preconditioner is not a factorization this backend owns, so there is no
# factor to count.
PureQPBase.backend_info(ls::IndirectCG) = PureQPBase.BackendInfo(
    PureQPBase.backend_name(ls), false, :reduced, length(ls.rhs), 0
)

function PureQPBase.set_tolerance_level!(ls::IndirectCG, level)
    ls.level = level
    return nothing
end

"""
    adopt_settings!(ls::IndirectCG, alg::QPAlgorithm, options)

Copy `cg_max_iter` and `cg_tol_fraction` from `options` and the CG tolerance-reduction
factor from `alg`. `alg.cg_tol_reduction`, when `alg` has that field, is how much the
tolerance is allowed to shrink between the conjugate-gradient solves of an iterative
algorithm as its outer residual falls; an algorithm with no such field (it sets a fresh
tolerance and starts CG from zero before every solve, so the tolerance would never shrink
between solves) reads as `typemax(Int)`, disabling the shrink outright.
"""
function PureQPBase.adopt_settings!(ls::IndirectCG, alg::PureQPBase.QPAlgorithm, options)
    ls.max_iter = options.cg_max_iter
    ls.tol_fraction = options.cg_tol_fraction
    ls.tol_reduction = hasfield(typeof(alg), :cg_tol_reduction) ?
        alg.cg_tol_reduction : typemax(Int)
    return nothing
end

"""
    no_settings_adopted()

What this backend raises when it reaches a factorization still holding its placeholder
settings. `PureQPBase.adopt_settings!`'s default does nothing, which is right for a backend
that reads no settings and silently wrong for this one.
"""
@noinline function no_settings_adopted()
    throw(
        ArgumentError(
            "the matrix-free backend never received its conjugate-gradient settings: no " *
                "`PureQPBase.adopt_settings!(::IndirectCG, alg, options)` method matched this " *
                "workspace's algorithm, so `cg_max_iter` is still zero and every solve would " *
                "return its starting point unchanged. Define `PureQPBase.adopt_settings!` for " *
                "the algorithm; it must set `max_iter`, `tol_fraction` and `tol_reduction`."
        )
    )
end

function PureQPBase.set_refresh_index!(ls::IndirectCG, k::Int)
    ls.refresh_index = k
    return nothing
end

function PureQPBase.use_residual_stop!(ls::IndirectCG, on::Bool)
    ls.residual_stop = on
    return nothing
end

PureQPBase.last_solve_converged(ls::IndirectCG) = ls.last_reached
PureQPBase.inner_iterations(ls::IndirectCG) = ls.total_iters

"""
    factorize!(ls::IndirectCG, prob, wt) -> Bool

Refresh the preconditioner for the current weights through `update_preconditioner!`. For the
default `JacobiPreconditioner` that is the reduced diagonal
`c·D[j]²·P[j,j] + σ + Σᵢ wᵢ (E[i] A[i,j] D[j])²`, which each column yields directly.

Succeeds unless `update_preconditioner!` returns an object of another type, which throws:
there is no factorization here that can be singular.

Also where an inner budget that never arrived is caught. `max_iter` holds the placeholder
`0` until `PureQPBase.adopt_settings!` copies `cg_max_iter` in, and `Options` refuses a
`cg_max_iter` of zero, so a zero here means no `adopt_settings!` method ran for the
workspace's algorithm — under which every solve would take no iteration and return its
starting point. Every path to a solve factorizes first, so the check sits here rather than
in `solve_system!`, where it would be on the per-iteration path.
"""
function PureQPBase.factorize!(ls::IndirectCG{T, V, K, M}, prob, wt)::Bool where {T, V, K, M}
    ls.max_iter > 0 || no_settings_adopted()
    fresh = PureQPBase.update_preconditioner!(ls.precond, prob, wt, ls.refresh_index)
    fresh isa M || throw(
        ArgumentError(
            lazy"update_preconditioner! must return a preconditioner of the type it was given, $M, and returned a $(typeof(fresh))"
        )
    )
    ls.precond = fresh
    return true
end

"""
    residual_stop_cg!(ls, op, M, atol) -> Bool

Run CG from zero with zero Krylov tolerances, stopping once the two-norm of its recursively
updated, unpreconditioned residual `kws.r` reaches `atol`, which costs a norm and no product.
The callback runs after every iteration, so `ls.last_reached` holds the verdict of the last
one.

`false` means Krylov abandoned the solve because `rᵀM⁻¹r` came out negative or NaN, which
happens when the operator or the preconditioner is not symmetric positive definite. The throw
is caught here, in a function of its own, so the solve kernel carries no exception path.
"""
@noinline function residual_stop_cg!(ls::IndirectCG{T}, op, M, atol::T) where {T}
    ls.last_reached = false
    try
        cg!(
            ls.kws, op, ls.rhs;
            M, ldiv = true, atol = zero(T), rtol = zero(T), itmax = ls.max_iter,
            callback = kws -> (ls.last_reached = LinearAlgebra.norm(kws.r) <= atol),
        )
    catch e
        # `msg` is declared `AbstractString`; Krylov's `error` builds a `String`.
        msg = e isa ErrorException ? e.msg : ""
        (msg isa String && occursin("not symmetric positive definite", msg)) || rethrow()
        return false
    end
    return true
end

"""
    solve_system!(ls::IndirectCG, prob, wt, rhs_x, rhs_z, x, z) -> Nothing

Solve the reduced system by preconditioned CG.

The tolerance follows the solver's residuals rather than being fixed: an early iterate does not
deserve an exact inner solve, and a late one does. It is `cg_tol_fraction` of the residual
level last set by `set_tolerance_level!`, floored at `eps(T)` relative to the right-hand side
so it cannot chase zero. This makes the solve *inexact*, so iterates differ from the direct
backends in the last digits even though both converge to the same solution.

CG starts from `x`, which holds the previous solve's `x̃`. Late in a solve that guess already
meets the tolerance, CG takes no step, and the iterate stops moving; after `cg_tol_reduction`
such solves in a row the tolerance is halved, which is what that setting counts. The floor is
relative rather than a fixed `sqrt(eps)` because a fixed floor sits above tight outer
tolerances, and no amount of halving gets below it.

The preconditioner is applied through `ldiv!`. With `residual_stop` on, CG starts from zero,
runs with zero Krylov tolerances and stops once the two-norm of its recursive residual reaches
the same `atol` (see `residual_stop_cg!`); a solve Krylov abandons for a
preconditioner or operator that is not positive definite returns `x = 0`. Either way the solve
counts as converged when it stopped on its test (or on Krylov's machine-precision stop), and as
a miss when it spent `cg_max_iter` iterations or broke down.
"""
function PureQPBase.solve_system!(ls::IndirectCG{T}, prob, wt, rhs_x, rhs_z, x, z)::Nothing where {T}
    m = prob.m
    if m > 0
        PureQPBase.multiply!(prob.work_m, wt.w, rhs_z)
        mul_At!(ls.rhs, prob, prob.work_m)
        PureQPBase.increment!(ls.rhs, rhs_x)
    else
        copyto!(ls.rhs, rhs_x)
    end

    floor = eps(T) * max(one(T), norm_inf(ls.rhs))
    atol = max(ls.reduction * ls.tol_fraction * ls.level, floor)
    op = ReducedOperator(prob, wt)
    M = krylov_preconditioner(ls.precond)
    if ls.residual_stop
        if residual_stop_cg!(ls, op, M, atol)
            ls.last_reached = ls.last_reached || ls.kws.stats.solved
            ls.total_iters += ls.kws.stats.niter
        else
            ls.last_reached = false
            fill!(ls.kws.x, zero(T))
        end
    else
        # The previous step's `x̃` is the best available guess: consecutive subproblems differ
        # by one step of the outer iteration, so starting from zero discards most of the work
        # and the inner budget is spent recovering it.
        Krylov.warm_start!(ls.kws, x)
        cg!(
            ls.kws, op, ls.rhs;
            M, ldiv = true, atol, rtol = zero(T), itmax = ls.max_iter,
        )
        ls.last_reached = ls.kws.stats.solved
        ls.total_iters += ls.kws.stats.niter
    end
    ls.last_reached || (ls.misses += 1)
    ls.idle_solves = iszero(ls.kws.stats.niter) ? ls.idle_solves + 1 : 0
    if ls.idle_solves >= ls.tol_reduction
        ls.reduction /= 2
        ls.idle_solves = 0
    end
    copyto!(x, ls.kws.x)
    m > 0 && mul_A!(z, prob, x)
    return nothing
end

@verify IndirectCG trim_compat = true

# The bare name above checks method existence for every parameter; a declared return type
# resolves only for concrete parameters, so the instantiation `indirect_backend` builds is
# checked too.
@verify IndirectCG{
    Float64, Vector{Float64}, CgWorkspace{Float64, Float64, Vector{Float64}},
    PureQPBase.JacobiPreconditioner{Vector{Float64}},
} trim_compat = true

end # module PureQPBaseKrylovExt
