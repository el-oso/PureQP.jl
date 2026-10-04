# Guarantees

This page gives the evidence for three claims about PureOSQP, and says where each one stops.

| claim | meaning | limit |
|---|---|---|
| **no allocation on the hot path** | No memory allocation occurs during iterations, avoiding garbage collection. | Guaranteed for `Vector`-backed workspaces. Not claimed for custom operators. |
| **type stability** | Performance is consistent because there is no hidden dynamic dispatch. | Checked for all reachable backends. |
| **compiles under `juliac --trim`** | Can be built into a standalone binary with no Julia runtime. | Entry points are enumerated; sparse operands require a named backend. |

All claims are machine-checked.

## The `LinearSystem` contract

The factorization backend is a declared interface.
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl) enforces it at precompilation.

```@eval
using PureOSQP, TypeContracts, Markdown
Markdown.parse(replace(contract_md_string(PureOSQP.LinearSystem), r"\A# [^\n]*\n+" => ""))
```

`PureQPBase.refactor_weights!` is not part of the contract, because its default is to rebuild
the factorization.

`@verify` checks at precompilation that a backend has the required methods and returns the right
types.

To add a backend, subtype `LinearSystem`, write the mandatory methods, and use `@verify`. The
backend is fixed when the workspace is built, so every per-iteration call dispatches
statically.

## No allocation, no type instability

`bench/strictmode_audit.jl` uses [StrictMode.jl](https://github.com/el-oso/StrictMode.jl) to
check that these functions are type-stable and allocation-free:

| function | guarantees |
|---|---|
| `admm_step!` | type-stable, allocation-free |
| `update_residuals!` | type-stable, allocation-free |
| `solve_system!` | type-stable, allocation-free |
| `check_termination` | type-stable |
| `factorize!` | type-stable |
| `refactor_weights!` | type-stable |
| `solve!` | type-stable |

The interior-point method's own hot kernels carry the same guarantees. We check them over every
backend its candidate list reaches: `FullKKT`, the sparse KKT family, the structured reduced
backends, and `:indirect` with a caller preconditioner.

| function | guarantees |
|---|---|
| `ipm_step!` | type-stable, allocation-free |
| `ipm_residuals!` | type-stable, allocation-free |
| `solve_multiplier!` | type-stable, allocation-free |
| `check_termination` | type-stable |
| `factorize!` | type-stable |
| `refactor_weights!` | type-stable |
| `solve!` | type-stable |

Two notes:
- A sparse arithmetic backend can allocate inside its own library. That covers `factorize!`,
  `refactor_weights!` and `solve!` on the sparse KKT family under both algorithms that use one.
  `solve_multiplier!` there is this package's own code, so it keeps the full guarantee.
- We check the matrix-free backend by measurement, because its static analysis shows branches
  that are possible but never taken. Under the interior-point method, the `try`/`catch` that
  turns a non-positive-definite preconditioner into a missed solve, rather than an exception,
  sits in a helper outside the audited kernel, so it does not change what we measure.

An operator you supply is only as fast as its own `mul!` method.

## The active-set method

`PureDAQP/test/strictmode_tests.jl` proves the iteration allocation-free and trim-compatible:
`run_daqp!`, the functions it calls, `multipliers!`, `primal!`, `build_solution` and
`reset_working_set!`. It proves them on a dense pair, on a singular `P` under `eps_prox > 0`,
and on a pair of [`PureQPBase.KroneckerOperator`](@ref)s, where reading a row of `A R⁻¹` is a
product and a solve and pricing is a solve and a product, so the loop runs different code. A
warm `solve!` and an `update!` of `q`, `l` or `u` are measured at 0 bytes. `solve!` reads the
clock, which the static check counts as an allocation, so it is proved trim-compatible and its
allocation claim is the measurement.

**For an `A` you supply, the allocation guarantee is only as good as your `mul!`.** An `A` that
supplies products only is multiplied for every pricing pass and for the final residual, and it
is read one row at a time through its adjoint product,
[`PureQPBase.dense_row!`](@ref), each time a row enters the working set. A product that
allocates therefore allocates on every iteration. In the test that checks this, a warm `solve!`
over a `LinearMap` whose functions allocate nothing allocates 0 bytes, and over one whose
functions allocate it does not. A LinearMaps product `B * C` allocates scratch per application
([Two packages supply operators](@ref)), so it breaks the guarantee in the same way.

## `--trim` compatibility

`juliac --trim` needs every function call resolved statically.

**We check every backend.** `PureQPBase/src/PureQPBase.jl` uses
`@verify LinearSystem subtypes = true trim_compat = true`, and each solver package does the same
for the workspace and algorithm it defines. The check runs during precompilation, for every
subtype.

**We list the entry points.** `PureOSQP/test/trim_tests.jl` validates concrete calls for every
public path:
- `solve` with various settings and sparse or dense operands.
- Every structured representation: diagonal, tridiagonal, banded, low-rank, block-diagonal,
  Kronecker, and products.
- `setup` → `solve!` → `update!` → `solve!`, and similar sequences with `warm_start!`.
- `update_settings!`, `update_rho!`, and `cold_start!`.
- The derivatives.
- `InteriorPoint()` on `FullKKT` (its default and its named KKT backend), the sparse KKT
  family, a diagonal pair, `:indirect` with a caller preconditioner on both a matrix pair and
  a `ProductOperator` pair, and a `setup` → `solve!` → `update!` → `solve!`
  sequence and the derivatives, each under `InteriorPoint()`.

For a sparse problem you must name the backend — `:kkt`, `:dense` or `:indirect` — to stay
trim-compatible.
