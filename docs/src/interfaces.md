# Interfaces

To add a new algorithm, a new linear-system backend or a new preconditioner, you write a short
list of methods for a subtype of an abstract type. [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl)'s
`@contract` declares each list. When the package precompiles, it checks every subtype against
that list, including whether each method's inferred return type matches.
`TypeContracts.describe(T)` prints the list for `T`. The tables below give the same lists.

A required method must exist for the subtype. An optional method either has a default that
serves every subtype, or belongs to one algorithm. `TypeContracts.satisfies(S, T)` tells you
what a type `S` is missing, and `TypeContracts.check_contract(S, T)` throws and names it.

## An algorithm

An algorithm is an object that holds the parameters only that method reads. It is a subtype of
[`QPAlgorithm`](@ref). [`setup`](@ref) calls these on it:

| method | returns | what it does |
|---|---|---|
| `setup_backend(alg, Val(linsys), T, P, q, A, l, u; kwargs...)` | a workspace | validates the data, builds the options, the problem and the backend, and factorizes |
| `algorithm_defaults(alg, T)` | `NamedTuple` | the [`Options`](@ref) defaults of this algorithm that have no common default |
| `default_options(alg, T)` | `Options` | the options a solve runs with when no keyword is passed; a default serves every algorithm |
| `element_typed(alg, T, options)` | `QPAlgorithm` | the object in the solve's element type, with every default resolved |

Optional: `adopt_settings!(ls, alg, options)`, which copies the parameters a backend reads into
that backend. Only the matrix-free backend reads any, so an algorithm that runs on it writes
this method for that backend. It is optional because a direct backend reads none. The
matrix-free one throws if it factorizes before its settings arrive, rather than solving quietly
with a zero iteration budget.

`setup_backend` declares no return type in the contract. Inferred through the abstract data
arguments of the contract's signature, that type is `Any`, though a call with concrete arguments
returns a concrete workspace.

An algorithm also needs a [`PureQPBase.SelectionFor`](@ref) tag of its own, plus the four
selection methods whose answer depends on the algorithm:
[`select_backend`](@ref PureQPBase.select_backend), the order it tries candidates in;
[`dense_rung`](@ref PureQPBase.dense_rung), its last candidate;
[`indirect_rung`](@ref PureQPBase.indirect_rung), what serves an operator it cannot build; and,
if the sparse candidates are in that order, the SparseArrays extension's `sparse_form`. Every
other selection method already takes any tag: the candidates that are skipped by default so
selection moves on, the `choose_backend` methods for a structured pair, and the errors the GPU
extension raises when a GPU array reaches a backend that cannot serve it. Leave one of the four
out and you get an error that names the method, not a `MethodError`.

## A workspace

The solver state that an algorithm's `setup_backend` builds is a subtype of
[`QPWorkspace`](@ref). Code outside the algorithm calls these on it:

| method | returns | what it does |
|---|---|---|
| [`solve!(ws)`](@ref solve!) | [`Solution`](@ref) | runs the algorithm from the workspace's state, and refills the workspace's result |
| [`warm_start!(ws; x, y)`](@ref warm_start!) | `ws` | seeds the next solve, in problem space |
| [`cold_start!(ws)`](@ref cold_start!) | `ws` | discards the iterates |
| [`update!(ws; q, l, u, P, A)`](@ref update!) | `ws` | replaces problem data |
| [`update_settings!(ws; kwargs...)`](@ref update_settings!) | `ws` | merges options; a default serves every workspace |
| [`update_settings!(ws, alg)`](@ref update_settings!) | `ws` | replaces the algorithm parameters; the default throws for another algorithm's object |
| [`dimensions(ws)`](@ref dimensions) | `Tuple{Int, Int}` | the number of variables and of rows; a default serves every workspace |

A solve allocates nothing: `setup` reserves everything the iteration and the result need,
so `solve!` refills the workspace's own [`Solution`](@ref) rather than building one. Two
results read from the same workspace are therefore the same object. `copy` the one you
need before solving again.

Optional, because only [`OperatorSplittingWorkspace`](@ref) implements them:
[`update_rho!`](@ref) and [`constraint_violation`](@ref).

For a method that takes keywords, the contract checks the positional signature. That is all it
can see. It does not check fields. The methods written for every workspace —
[`dimensions`](@ref), the keyword form of [`update_settings!`](@ref),
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref) — read `prob`, `linsys`,
`algorithm`, `options`, `x`, `y`, `z`, `status` and `polished`.

## A linear-system backend

A backend is a subtype of [`LinearSystem`](@ref). The workspace holds one and hands it the
[`PureQPBase.Problem`](@ref) and the [`PureQPBase.SystemWeights`](@ref) on every call.

| method | returns | what it does |
|---|---|---|
| [`factorize!(ls, prob, wt)`](@ref PureQPBase.factorize!) | `Bool` | rebuilds the factorization; `false` when it cannot |
| [`solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z)`](@ref PureQPBase.solve_system!) | `Nothing` | solves for `x` and writes `z = Ãx` |
| [`backend_info(ls)`](@ref backend_info) | [`BackendInfo`](@ref) | what the backend is and how large its factorization is |

Optional, each with a default for every backend:

| method | returns | default |
|---|---|---|
| [`refactor_weights!(ls, prob, wt)`](@ref PureQPBase.refactor_weights!) | `Bool` | calls `factorize!` |
| [`solve_multiplier!(ls, prob, wt, rhs_x, rhs_z, x, nu)`](@ref PureQPBase.solve_multiplier!) | `Nothing` | derives `ν` from `solve_system!` |
| [`check_update(ls, P, A)`](@ref PureQPBase.check_update) | `Nothing` | accepts |
| [`set_tolerance_level!(ls, level)`](@ref PureQPBase.set_tolerance_level!) | `Nothing` | ignores it |
| [`set_refresh_index!(ls, k)`](@ref PureQPBase.set_refresh_index!) | `Nothing` | ignores it |
| [`adopt_settings!(ls, alg, options)`](@ref PureQPBase.adopt_settings!) | `Nothing` | ignores them |
| [`use_residual_stop!(ls, on)`](@ref PureQPBase.use_residual_stop!) | `Nothing` | ignores it |
| [`last_solve_converged(ls)`](@ref PureQPBase.last_solve_converged) | `Bool` | `true` |
| [`inner_iterations(ls)`](@ref PureQPBase.inner_iterations) | `Int` | `0` |

A backend defined in an extension gets checked when the extension loads.

## A preconditioner

The matrix-free backend (`linsys = :indirect`) applies a preconditioner `M` through two
methods:

| method | returns | what it does |
|---|---|---|
| [`update_preconditioner!(M, prob, wt, k)`](@ref update_preconditioner!) | `M`'s type | refreshes for the current weights; the default returns `M` unchanged |
| `LinearAlgebra.ldiv!(y, M, x)` | | writes the preconditioned vector into `y`; no default |

[`IdentityPreconditioner`](@ref) and [`JacobiPreconditioner`](@ref) are subtypes of
[`Preconditioner`](@ref). Yours does not have to be. A `Cholesky`, or any other factorization
object, already has `ldiv!`, and `TypeContracts.check_contract(typeof(M), Preconditioner)`
checks one against the same list. If a preconditioner has no `ldiv!` method for the backend's
vectors, [`setup`](@ref) throws and names the missing method.
