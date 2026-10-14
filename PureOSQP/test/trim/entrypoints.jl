# Entry points for the `--trim` compatibility check. Each wraps one public path with a
# concrete signature, because `juliac --trim` and TrimCheck both analyse from a call, not
# from a module.
module TrimEntry

using PureOSQP
using PureIPM
# For the operator types PureOSQP does not re-export. Imported without names, so nothing here
# shadows the `PureOSQP.`-qualified calls below.
import PureQPBase
using Krylov
using LinearAlgebra
using BandedMatrices
using SparseArrays
# Sparse compatibility is conditional on this: `is_convex` reaches a factorization for a
# sparse `P`, and only the one this supplies is resolvable statically.
using LDLFactorizations
# An operator hierarchy that is not this package's own, so the wrapper is checked against a
# real `mul!` rather than against a `Matrix` standing in for one.
using SciMLOperators
using COSMOAccelerators

const M = Matrix{Float64}
const V = Vector{Float64}
const SPM = SparseMatrixCSC{Float64, Int}
const DM = Diagonal{Float64, Vector{Float64}}
const STM = SymTridiagonal{Float64, Vector{Float64}}
const TM = Tridiagonal{Float64, Vector{Float64}}

solve_default(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)
solve_polish(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; polishing = true)
solve_kkt(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :kkt)
solve_unscaled(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; scaling = 0)

# ADMM never calls `solve_multiplier!`, so it needs its own entry point to be reached at all.
# `:kkt` is what selects a backend whose override, rather than the default, is exercised.
function solve_multiplier(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u; linsys = :kkt)
    nu = similar(ws.rhs_z)
    PureOSQP.solve_multiplier!(ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, nu)
    return nu[1]
end

# The diagonal backend is reached by representation rather than by a setting, so it needs a
# signature of its own to be analysed at all.
solve_diagonal(P::DM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)
solve_tridiagonal(P::STM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The same band spelled `Tridiagonal` reaches the same backend over a different `OperatorSplittingWorkspace`
# type, which is a specialization of every solve method in its own right.
solve_tridiagonal_unsym(P::TM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# A Tridiagonal A squares past SymTridiagonal, so this is the BandedMatrices extension.
solve_banded(P::STM, q::V, A::TM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The Woodbury backend, whose `A` is a matrix type of this package's own rather than one of
# LinearAlgebra's.
const RC = PureOSQP.RowCoupled{Float64, Matrix{Float64}, Vector{Float64}}
solve_lowrank(P::DM, q::V, A::RC, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The matrix-free backend exists only once Krylov is loaded, so this entry point is what
# checks that a weak dependency on the solve path does not cost the trim guarantee.
solve_indirect(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :indirect)

# A caller preconditioner is a type parameter of the backend, so a factorization object in
# that slot is a specialization of every solve method of its own.
solve_indirect_preconditioned(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(
    P, q, A, l, u; linsys = :indirect, scaling = 0, preconditioner = cholesky(Symmetric(P + I))
)

# The interior-point method on its default (dense pair, `FullKKT`), on the KKT backend named
# directly, unscaled, and with polishing -- the four ways of reaching `InteriorPointWorkspace`
# that a Matrix pair takes.
solve_ipm_default(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint())
solve_ipm_kkt(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); linsys = :kkt)
solve_ipm_unscaled(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); scaling = 0)
solve_ipm_polish(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); polishing = true)

# `verbose` on the interior-point method: the header/row/footer printing is new code the
# trimmer must resolve, on the direct backend and, separately, on the matrix-free one, whose
# row takes the extra branch that prints the CG column.
solve_ipm_verbose(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); verbose = true)
solve_ipm_verbose_indirect(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(
    P, q, A, l, u, PureIPM.InteriorPoint(); verbose = true, linsys = :indirect, scaling = 0,
    preconditioner = cholesky(Symmetric(P + I))
)

# Chosen by representation under the IPM exactly as under ADMM: `choose_backend` for a
# `Diagonal` pair serves either algorithm.
solve_ipm_diagonal(P::DM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint())

# The sparse KKT family, named so the pair does not depend on which sparse `LDLᵀ` extension
# is loaded.
solve_ipm_sparse_kkt(P::SPM, q::V, A::SPM, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); linsys = :kkt)

function setup_ipm_update(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u, PureIPM.InteriorPoint())
    PureOSQP.solve!(ws)
    PureOSQP.update!(ws; q = q, l = l, u = u)
    return PureOSQP.solve!(ws)
end

function derivatives_ipm(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u, PureIPM.InteriorPoint(); polishing = true)
    PureOSQP.solve!(ws)
    d = PureOSQP.adjoint_derivative(ws, q, l)
    fx, fy = PureOSQP.forward_derivative(ws; dq = q)
    return d.dq[1] + fx[1] + fy[1]
end

# The interior-point method on the matrix-free backend, whose Krylov call sits inside the
# `try` that turns a definiteness failure into a missed solve.
solve_ipm_indirect(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(
    P, q, A, l, u, PureIPM.InteriorPoint(); linsys = :indirect, scaling = 0,
    preconditioner = cholesky(Symmetric(P + I))
)

# The block backend, whose `P` and `A` are both this package's own matrix type.
const BD = PureOSQP.BlockDiagonal{Float64, Matrix{Float64}}
solve_block(P::BD, q::V, A::BD, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# These two pin the `@constprop :aggressive` on `setup` and `solve`. A keyword alone does not:
# `solve_unscaled` above passes without either annotation, because a dense pair's ladder
# backend *is* `ReducedCholesky`, so only three `OperatorSplittingWorkspace` types merge and the union stays
# under `MAX_TYPEUNION_LENGTH`. What discriminates is a pair whose backend is a fourth type —
# none of `ReducedCholesky`, `FullKKT` or `IndirectCG` — carrying a keyword, so the merge is
# four wide and widens to `OperatorSplittingWorkspace{…} where LS`.
const KO = PureOSQP.KroneckerOperator{Float64, Matrix{Float64}}
solve_kronecker(P::DM, q::V, A::KO, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; scaling = 0)

# A second such pair, so the guard does not rest on one backend.
solve_lowrank_scaled(P::DM, q::V, A::RC, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; scaling = 0)

# A stack, which answers both `holds_structure` and `worth_assembling`, so ADMM's
# `structured_rung` serves it with `ProductReduced`. That is the only ADMM route to that
# backend, and no `linsys` name reaches it, so without this entry point the trimmer never sees
# the stack's `add_reduced_term!` or the inverse-holding solve on it.
const SK = PureQPBase.StackedOperator{Float64, Tuple{KO, DM}, V}
solve_stacked(P::M, q::V, A::SK, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; scaling = 0)

# The interior-point method on an `A` that holds structure: every rung above declines it, so
# `indirect_rung` serves it with `ProductReduced` — holding the Cholesky factor here, where
# `solve_stacked` gets the inverse, which makes the two different specializations of the solve.
solve_ipm_kronecker(P::DM, q::V, A::KO, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); scaling = 0)

# `setup` with a keyword, reached directly rather than through `solve`. Its constants flow
# through the untyped five-argument forwarder, which carries no annotation and is small enough
# for the const-prop heuristic to enter; this pins that, since logic added there later would
# otherwise regress the path with the gate still green.
function setup_kronecker(P::DM, q::V, A::KO, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u; scaling = 0)
    return PureOSQP.solve!(ws)
end

# An operator that supplies only products. Its `getindex` throws on a branch the trimmer
# analyses whether or not it runs, which is what keeps that message free of the string
# formatting `--trim` rejects.
const PO = PureOSQP.ProductOperator{
    Float64, Matrix{Float64}, Adjoint{Float64, Matrix{Float64}}, Vector{Float64},
}
solve_operator(P::PO, q::V, A::PO, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u; scaling = 0, linsys = :indirect)

# The interior-point method on the same operator pair, with a caller preconditioner: the IPM
# has no automatic route onto an operator, so `:indirect` and a preconditioner are named
# together, as the operator path requires.
solve_ipm_operator(P::PO, q::V, A::PO, l::V, u::V) = PureOSQP.solve(
    P, q, A, l, u, PureIPM.InteriorPoint();
    scaling = 0, linsys = :indirect, preconditioner = Diagonal(ones(length(q)))
)

# A real operator library inside the wrapper, rather than the `Matrix` above: this is what
# checks that an operator's own `mul!` and `adjoint` resolve statically.
const SO = typeof(
    PureOSQP.ProductOperator{Float64}(
        FunctionOperator(
            (w, v, u, p, t) -> (w .= v), zeros(Float64, 2), zeros(Float64, 2);
            op_adjoint = (w, v, u, p, t) -> (w .= v), islinear = true,
        )
    )
)
solve_sciml(P::SO, q::V, A::SO, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u; scaling = 0, linsys = :indirect)

# An accelerated solve. The accelerator is built here rather than taken as an argument,
# because its type is what puts the per-iteration hooks on the trimmed path at all.
solve_accelerated(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u; accelerator = PureOSQP.anderson(Float64, length(q) + length(l)))

# `verbose` prints through hand-written formatting precisely because `--trim` rejects
# Printf and `Base.stdout`. Pinned as its own entry point so that stays checked.
solve_verbose(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u; verbose = true)

# The `try`/`catch` guarding the ADMM loop against an interrupt is on every solve path,
# so the trimmer sees it whether or not one is ever raised.
solve_interruptible(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; max_iter = 50)

# `time_limit` puts `time_ns` and the UInt64 budget arithmetic on the solve path.
solve_time_limited(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; time_limit = 10.0)

# The primal-dual integral adds a `time_ns` and a `log` to the residual update. Both are
# resolvable, but the branch reaching them is analysed whether or not the setting is on, so
# this pins that the accumulator did not put anything unresolvable on the path.
solve_profiled(P::M, q::V, A::M, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u, PureOSQP.OperatorSplitting(profile_primdual = true))

# The rest of the exported surface. `update_settings!` in particular compares algorithm
# parameters field by field rather than looping over a tuple of symbols, because `getfield`
# with a symbol the compiler cannot see is a dynamic call -- this is what checks that reasoning.
function settings_and_rho(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    PureOSQP.update_settings!(ws; eps_abs = 1.0e-9)
    PureOSQP.update_settings!(ws, PureOSQP.OperatorSplitting(rho = 0.5))
    PureOSQP.update_rho!(ws, 0.25)
    PureOSQP.cold_start!(ws)
    n, m = PureOSQP.dimensions(ws)
    c = PureOSQP.capabilities()
    return n + m + (c.direct_solver ? 1 : 0)
end

function derivatives(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    PureOSQP.solve!(ws)
    d = PureOSQP.adjoint_derivative(ws, q, l)
    fx, fy = PureOSQP.forward_derivative(ws; dq = q)
    return d.dq[1] + fx[1] + fy[1]
end

function setup_solve_update(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    PureOSQP.solve!(ws)
    PureOSQP.update!(ws; q = q, l = l, u = u)
    return PureOSQP.solve!(ws)
end

function warm_started(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    s = PureOSQP.solve!(ws)
    PureOSQP.warm_start!(ws; x = s.x, y = s.y)
    return PureOSQP.solve!(ws)
end

# A sparse pair with the backend named. Naming one is what makes it compatible: `linsys`
# reaches `setup` as a type parameter, so the branch that descends the ladder is eliminated
# rather than merely unused, and with it `choose_backend` and the SuiteSparse bindings the
# sparse rungs reach.
solve_sparse_kkt(P::SPM, q::V, A::SPM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :kkt)
solve_sparse_dense(P::SPM, q::V, A::SPM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :dense)

# A sparse `A` under a structured `P`, left on `:auto` so the ladder itself is checked: the
# rungs this pair descends serve it without reaching a sparse factorization at all. This is
# the shape a problem takes when the objective is a multiple of the identity and the
# constraints are what carry the structure.
solve_sparse_diagonal(P::DM, q::V, A::SPM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# There is no `:auto` entry point for a pair that is sparse on both sides, and the reason is
# not incidental. Two obstructions sit there, either of which is fatal on its own. That ladder
# must consider every sparse rung, and the reduced rung factors with CHOLMOD, whose bindings
# read and write their C structs through `getproperty` on pointer-backed wrappers that no
# static analysis resolves. It also reaches more backend types than `tmerge` will keep apart,
# so the workspace widens to `OperatorSplittingWorkspace{…} where LS` and the following `solve!` is a dynamic
# dispatch. Leaving the choice open is therefore incompatible by construction, and naming the
# backend is what closes it.

# Negative control: reflection is exactly what `--trim` cannot resolve. If the checker
# calls this one compatible it is not discriminating and its verdict on the real entry
# points means nothing.
not_trimmable(P::M, q::V, A::M, l::V, u::V) = length(Base.return_types(PureOSQP.solve, (M, V, M, V, V)))

end # module TrimEntry
