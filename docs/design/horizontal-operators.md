# Horizontal operator composition

## What does not work

`unwrap(::Type{T}, M::LinearMaps.BlockMap, λ)` in `PureQPBaseLinearMapsExt` declines any
`BlockMap` whose `rows` is not all ones, which is every `hcat`. The block becomes a products-only
`ProductOperator`, and `is_materializable(::ProductOperator) = false` (`operator.jl:131`), so the
whole composition answers `false` and the dense rungs decline it (`linsys.jl:847`).

A `LinearMaps.FillMap` is declined the same way, and that matters more than the `hcat`: it
appears in **all 72** problems, so even with an `hcat` type every composition would still contain
one non-materializable block and still be refused. The Fill is the load-bearing fix.

## The data

72 serialized problems from an external interior-point solver, arriving as LinearMaps.

| family | n | m | files |
|---|---|---|---|
| 101 | 220 | 330 | 16 |
| 102 | 629 | 2396 | 16 |
| 103 | 302 | 2382 | 16 |
| 104 | 139 | 330 | 24 |

`n` is 139 to 629. A dense terminal is `O(n³)` at worst 2.5e8 flops, so the dense rungs are the
right destination and `holds_structure` should be `false` for the new type — stated here because
the decision depends on these numbers and nothing else.

| `P` | files |
|---|---|
| `Wrapped{Array}`, `Wrapped{Adjoint}` | 56 |
| `hcat(vcat(Diagonal, Fill), vcat(Fill, Array))` | 8 |
| `hcat(vcat(Diagonal, Fill), vcat(Fill, Diagonal), vcat(Fill, Array))` | 8 |

| `A` | files |
|---|---|
| `hcat(kron, Fill)` | 40 |
| `vcat(hcat(…), hcat(…))`, nested two deep | 32 |

Mapping to PureQP, verified against each file's stored solution: `P = H`, `q = f`, `l = b`,
`u = +∞`, objective `½xᵀPx + qᵀx`. `min(Ax - b) ≥ 0` on every file, tight to 1e-12 on family 101,
while `Ax ≤ b` is violated by every row. `x0` is the unconstrained minimum.

## Step 1 — `FillMap` becomes a `FillArrays.Fill`

One rule in the LinearMaps extension:

```julia
unwrap(::Type{T}, M::LinearMaps.FillMap, λ) where {T} = Fill(T(λ * M.λ), size(M))
```

`FillArrays` joins PureQPBase's `[deps]` and `[compat]`; it is already in the manifest
transitively, so load time does not change. A `Fill` is materializable with `O(1)` `getindex`,
its products contract to `O(n + m)` as `FillMap`'s do, `issymmetric` works, and
`AbstractMatrix{T}(::Fill)` returns a `Fill` so `conform` is unaffected.

This step alone changes what the 32 `vcat` files can reach, because their `StackedOperator`
blocks stop containing a non-materializable one. **Re-measure the 72-file table here, before
writing any new type.** No `ConstantOperator`: it would reinvent `Fill`, measured equivalent at
85 ns and no allocations for a 2000×500 product.

## Step 2 — `JoinedOperator`

`PureQPBase/src/joined.jl`, a thin sibling of `StackedOperator`: blocks share rows, columns
concatenate, `colstart` in place of `rowstart`. Every traversal recurses on the block tuple's
type, as `stacked.jl` does, because iterating a heterogeneous tuple by runtime index boxes.

| verb | horizontal form |
|---|---|
| `mul!(y, A, x)` | `y = Σᵢ Bᵢ·x[colrange(i)]` through `work`; three-argument `mul!` per block and an explicit add, since the five-argument form is not defined for every block type |
| `mul!(y, A', x)` and `mul!(y, transpose(A), x)` | `y[colrange(i)] = Bᵢ'·x`, independent per block. Both wrappers, as `stacked.jl:120` does |
| `getindex`, `dense_row!` | locate the block by column; a row spans every block |
| `structural_rows(A, j)` | column-local: delegate to the block owning `j`, so equilibration does not walk `m` per column |
| `check_finite` | per block, so a `KroneckerOperator` block uses its own rather than a generic entry walk |
| `reduced_diagonal!` | column-local, each block answering for its own columns |
| `is_materializable` | `all` over the blocks |
| `holds_structure` | **`false`**, from the dimensions above — this is what lets the dense rungs serve it |

**No `JoinedOperator` method for `add_reduced_term!`.** The generic one is correct, and for this
data it is never called: `add_reduced_term!` runs only from `ProductReduced.factorize!`
(`linsys.jl:1613`), which only `PureIPM`'s `indirect_rung` builds, and with `holds_structure =
false` and every block materializable the dense rungs take these problems instead — ADMM to
`ReducedCholesky`, IPM to `FullKKT`. Record in a comment what a block-pair reduction would cost,
for whoever adds an ADMM `ProductReduced` rung: an `hcat` partitions the columns, so the pairs
sum to `Σᵢ nᵢ(fᵢ+aᵢ) + Σᵢ<ⱼ min(nᵢ,nⱼ)(fⱼ+aᵢ)`, strictly below the generic `n·Σᵢ(fᵢ+aᵢ)`. The
generic method's own cost is **not** lower for an `hcat` than for an opaque operator: `mul!(av,
A, ej)` applies the whole operator (`linsys.jl:1545`), every block included.

## Step 3 — the `unwrap` rule

`all(isone, rows)` → `StackedOperator` as today; `length(rows) == 1` → `JoinedOperator`;
`rows == (k, k, …)` describing an `hvcat` → `StackedOperator` of `JoinedOperator`s by grouping
`maps` against `rows`. Blocks go through `as_operator`, so anything unrecognised still becomes a
`ProductOperator` and the rest keeps its structure.

## Step 4 — ADMM has no rung for a structured operand

Independent of `hcat`, and the reason "the 32 vcat files already reach a direct backend" is true
only under `InteriorPoint`. Under `OperatorSplitting`, `select_backend` reaches `dense_rung`,
which declines on `holds_structure(P) || holds_structure(A)` (`linsys.jl:848`);
`holds_structure(::StackedOperator)` is `true` (`stacked.jl:77`); `formed_rung` is defined only
for a `SparseMatrixCSC` `A`; and the block, Kronecker and low-rank rungs are typed to their own
pairs. So a `StackedOperator` `A` under ADMM reaches `:indirect` today and would continue to.

`ProductReduced` is algorithm-agnostic and lives in PureQPBase. An ADMM rung returning it for a
`holds_structure` operand is a few lines and gives ADMM a direct backend for those operands. This
is where a block-pair `add_reduced_term!` would start to pay, so the two belong together and
after the litmus test, not before it.

## Step 5 — the refactorization path

`add_reduced_term!` is a per-iteration cost where it is reached at all:
`PureIPM.factorize_newton!` calls `refactor_weights!` once per iteration (`ipm.jl:61`). Three
obligations hold for any new operator in that path.

**Scratch is built once and held.** `reduced_term_scratch(T, A)` is called in `ProductReduced`'s
constructor and stored, and the backend's type parameter carries it, so a method must be
type-stable and shaped per block as `StackedOperator`'s is.

**`update!` must preserve the concrete type.** `validate_update!` requires `A isa MA`
(`problem.jl:201`), which a re-wrapped LinearMap of the same shape satisfies. The nested case
needs checking, since a `StackedOperator` of `JoinedOperator`s carries a two-level type.

**The refactorization stays allocation-free.** `refactor_weights!` is among the signatures
StrictMode proves; a new operator inherits that, and the gate is the existing audit.

## The litmus test

Every problem must solve, on every solver, in every representation it can be expressed in, with
the backend named and the time recorded. Four representations:

| representation | how |
|---|---|
| unmaterialized | the operators as deserialized, through `unwrap` |
| dense | `Matrix(H)`, `Matrix(A)` |
| sparse | `sparse(Matrix(H))`, `sparse(Matrix(A))` |
| structured | where a family's `P` is `Diagonal`-blocked or its `A` is a Kronecker pair, the `PureQPBase` type rather than the LinearMap |

For each of 72 problems × 4 representations × 3 solvers, record: `backend_name(ws.linsys)`,
status, iterations, wall time, `‖x − solution‖∞`, and the objective gap. Two conditions decide
the test:

1. no cell is an error, and
2. the unmaterialized cell's backend is a direct one wherever the dense cell's is — an
   unmaterialized input must not be punished for its representation, which is the whole point of
   the operator types.

`:indirect` is a legitimate answer only where it is chosen on the merits for every
representation, not where it is the residue of a declined `unwrap`.

Baselines to record **before** step 1, so each step's effect is separable: the current table has
family 101 solving under `ActiveSet` at 6e-8 in two iterations while `OperatorSplitting` reports
`SOLVED` 14 to 22 from the stored solution, running unequilibrated conjugate gradients at
`cond(P) = 1e12`, and every problem whose `P` is an `hcat` — 16 of the 72 — refused outright at
`setup` for that reason. Note that equilibration becoming *possible* and the backend changing are two
separate levers; measure them separately rather than attributing the error change to one.

Artifacts: a `bench/` script that regenerates the whole matrix from a fresh checkout, samples
saved as JSON under `bench/results/`, and `PureQPBase`/`PureDAQP` test items covering
`JoinedOperator` against a dense `hcat` reference, the `unwrap` results for both shapes,
`update!` over a `JoinedOperator` including the nested case, and `dense_row!` plus the
refactorization added to the StrictMode proofs with a Kronecker block and a `Fill` block.
