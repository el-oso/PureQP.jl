"""
    PureDAQP

A dual active-set method for

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

The problem representation and the generic `setup`/`solve`/`solve!` come from PureQPBase.jl
and are re-exported here, so `using PureDAQP` is enough to solve a problem:

    solve(P, q, A, l, u, ActiveSet())

The method reduces the problem to a least-distance problem and maintains an `LDLᵀ` of the
working set's Gram matrix under rank-one updates, so each iteration costs `O(k²)` in the size
of the working set rather than a fresh factorization. It follows Algorithm 1 of Arnström,
Bemporad and Axehill, *A dual active-set solver for embedded quadratic programming using
recursive LDLᵀ updates*, IEEE Transactions on Automatic Control 67(8):4362-4369, 2022, with
that paper's Algorithm 2 as the proximal-point outer loop. The reference implementation is
MIT-licensed and was read alongside the paper.

It needs a `P` it can factor, which rules out an operator supplying products only, and it reads
a sparse `P` or `A` into a dense matrix. A dense pair reduces to `A R⁻¹` formed once; a
structured or unmaterialized one is held as `A` and the factor of `P` in their own forms, and
each row and each product is derived from them. It is at its best where an active-set method
always is, with few rows active at the solution.
"""
module PureDAQP

using LinearAlgebra
using ModifiableFactorizations: ModifiableFactorizations, ModifiableQR, try_insert_column!,
    delete_column!, FixedCapacity
using TypeContracts: TypeContracts, @contract, @verify
using StrictMode: @strict_function, @strict, @assert_noalloc, @assert_trim_compatible,
    @assert_typestable
using Reexport
# The base's own exports reach a caller through here, so `using PureDAQP` gives the whole API:
# the verbs, the `Solution`, the statuses and the options. They stay PureQPBase's — this says
# where they come from in one line, where a list repeated per solver would make one generic
# function look like it had four owners.
@reexport using PureQPBase

import PureQPBase:
    setup, solve, solve!, warm_start!, cold_start!, update!, update_settings!,
    derivative_ready, setup_backend, algorithm_defaults, element_typed, dimensions,
    QPData, Options, Solution, Status, QPAlgorithm, QPWorkspace, PolishStatus,
    adopt_update!, check_update, has_solution, is_convex, is_materializable, validate,
    validated_data, validate_update!, check_option_names, settings_tuple, paired,
    empty_solution, norm_inf, support_plain, project_polar_reccone!, DIVISION_TOL,
    has_cholesky_factor, cholesky_factor, factorable_operand, rows_operand,
    scalar_diagonal, dense_row!,
    BlockDiagonal, KroneckerOperator, KroneckerCholesky, ProductOperator,
    StackedOperator, JoinedOperator, ComposedOperator, SumOperator


# What this package owns. Everything else a caller needs is PureQPBase's and arrives through the
# `@reexport` above.
export ActiveSet, faster_scan, ActiveSetWorkspace

# `settings.jl` first: the loop takes its tolerances as an `ActiveSet`, so the type has to
# exist before the methods that name it.
include("settings.jl")
include("qrset.jl")
include("ldl.jl")
include("ldp.jl")
include("workspace.jl")
include("solution.jl")

"""
    Optimizer(; kwargs...)

MathOptInterface optimizer, available once MathOptInterface is loaded. Keyword arguments are
the fields of [`Options`](@ref) and the parameters of [`ActiveSet`](@ref), each checked by name
when it is set.

`ActiveSet` refuses what does not apply to it rather than ignoring it, so a `scaling` other
than zero, a `polishing`, a `linsys` or an operator-splitting parameter set through this
wrapper throws as it would through [`setup`](@ref).

The wrapper lives in a package extension, so it costs nothing to a caller who does not use
it; this name is the only part of it this package owns.
"""
function Optimizer end

# The entry points that reach forward, held to the same guarantee as the ones declared at
# their definitions. `solve!` calls `build_solution`, which `solution.jl` defines after it,
# so inference has the whole call graph only once every file is in. Running them on a problem
# small enough to solve here also compiles the path a caller arrives on.
let
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    # `setup` builds the workspace, so it allocates by contract and carries no such claim.
    ws = setup(P, q, A, l, u, ActiveSet())
    # `solve!` is type-stable and allocates nothing, but the allocation claim is measured
    # rather than scanned: it reads the clock, and the `jl_hrtime` foreign call behind
    # `time_ns` is opaque to both the scan and AllocCheck.
    @strict solve!(ws)
    @strict update!(ws; q = q)
    @strict update!(ws; l = l, u = u)
    @strict update_settings!(ws, ActiveSet())

    # The same on a Kronecker pair, which reduces to `ImplicitRows` over a `KroneckerCholesky`
    # and so puts different code in the iteration: a row is a product and a solve rather than a
    # view, and pricing is a solve and a product rather than one `gemv`.
    Pk = KroneckerOperator([2.0 0.5; 0.5 3.0], [4.0 1.0; 1.0 2.0])
    Ak = KroneckerOperator([1.0 0.5; 0.0 1.0; 1.0 1.0], [1.0 0.0; 0.5 1.0])
    qk = [1.0, 1.0, 0.5, -0.5]
    lk, uk = fill(-1.0, 6), fill(1.0, 6)
    wsk = setup(Pk, qk, Ak, lk, uk, ActiveSet())
    @strict solve!(wsk)

    # The dual active-set loop and the pieces a solve runs around it, on a workspace a solve
    # has already brought to a state each call is legal in. `solve!` is asserted trim-
    # compatible but not allocation-free: it reads the clock, and AllocCheck counts the
    # `jl_hrtime` foreign call as an allocation it cannot see through. Everything under the
    # clock carries both claims, and `run_daqp!` is the whole iteration. These report rather
    # than throw; `test/strictmode_tests.jl` proves every kernel with StrictModeTest.
    #
    # A function rather than a loop over the two workspaces: each call specializes on one
    # concrete workspace type, where a loop would hand every assertion their union.
    function proofs(w)
        red = w.red
        @assert_trim_compatible solve!(w)
        @assert_noalloc run_daqp!(red, w.prob.q0, w.algorithm, w.options.max_iter)
        @assert_trim_compatible run_daqp!(red, w.prob.q0, w.algorithm, w.options.max_iter)
        @assert_noalloc multipliers!(w.y, red)
        @assert_trim_compatible multipliers!(w.y, red)
        @assert_noalloc build_solution(w)
        @assert_trim_compatible build_solution(w)
        @assert_noalloc reset_working_set!(red)
        @assert_trim_compatible reset_working_set!(red)
        return nothing
    end
    proofs(ws)
    proofs(wsk)
end

# Every workspace and algorithm this package defines must satisfy its contract, asserted for
# each subtype defined by the time the module finishes rather than type by type, so a new
# type cannot acquire the guarantee only by someone remembering to ask for it.
@verify QPWorkspace subtypes = true trim_compat = true
@verify QPAlgorithm subtypes = true trim_compat = true

end # module PureDAQP
