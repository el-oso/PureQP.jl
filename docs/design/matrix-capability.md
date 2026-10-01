# A matrix capability the base owns, so every algorithm takes every representation

PureQPBase is meant to hold all support for the four matrix representations — dense, sparse,
structured, unmaterialized — with PureOSQP, PureIPM and PureDAQP deriving from it. Today that
does not hold for `ActiveSet`, and `docs/src/matrices.md:30` promises four answers that the code
does not give. This document is the plan that makes it hold, written to be read cold by an
implementer who was not present: every file it names was read, every number it quotes was run.

It **supersedes `docs/design/matrixfree-activeset.md`**, which is deleted in the same commit.
§9 says what of it survives.

Claims marked **measured** were run in the session this document was written in, on
`neuromancer` at one BLAS thread, in the warm `bench` environment, against a Kronecker problem
matched to a real caller's problem: `n = 25·25 = 625`, `m = 46·48 = 2208`,
`cond(P) = 8.3e8`, `cond(A) = 2.4e11`, `P = P₁ ⊗ P₂`, `A = A₁ ⊗ A₂`. Claims marked
**algebra** are identities checked numerically at that size to the precision quoted. Claims
marked **literature** rest on a reference, not on this code. Claims marked **unverified** were
not checked; each names what would check it.

Conventions: `R` is the upper Cholesky factor, `P + εI = RᵀR`; `M = A R⁻¹` is the reduced
constraint matrix of the least-distance problem; `ε` is `ActiveSet`'s `eps_prox`, `0` by
default (`PureDAQP/src/settings.jl:65`).

---

## 1. Where things stand

**Measured**, one problem passed in each form to each algorithm:

| | dense | sparse | structured | unmaterialized |
|---|---|---|---|---|
| `OperatorSplitting` | yes | yes | yes | yes |
| `InteriorPoint` | yes | yes | yes | only with a caller-supplied preconditioner |
| `ActiveSet` | yes | yes, **by densifying** | yes, **by densifying** | **refused** |

The refusal is a deliberate `throw` in `PureDAQP/src/workspace.jl` (`setup_backend`, the
`is_materializable(P) && is_materializable(A) ||` line), not a `MethodError`: the base's
`ProductOperator <: AbstractMatrix` (`PureQPBase/src/operator.jl`) already carries a
`LinearMap`, a SciMLOperator or an AbstractOperator to every algorithm's `setup_backend`.

The two "by densifying" cells matter as much as the empty one. `setup_backend` calls
`reduce_qp(convert(Matrix{T}, P), convert(Matrix{T}, A), …)`, so a `SparseMatrixCSC`, a
`BlockDiagonal` and a `KroneckerOperator` are all read into dense `Matrix`es before anything
else happens. **Measured**: `setup(Pop, q, Aop, l, u, ActiveSet(); scaling = 0)` on the two
`KroneckerOperator`s solves (`SOLVED`, 1510 iterations, the same objective as the dense pair to
`3e-11` relative) while holding a `36.3 MiB` workspace of which the reduction's `Mt` alone is
`10.5 MiB`, against `28 KiB` for the four Kronecker factors. A caller who passed the operator to
avoid the dense form gets the dense form.

`setup` itself also densifies a structured `P` once, before any algorithm runs: `is_convex`
has no `KroneckerOperator` method, so the generic one forms `Matrix(P) + σI` and factors it
(`PureQPBase/src/types.jl:460`). That is `n²`, not `mn`, but it is the same violation in a
smaller size and it costs every algorithm. §7 S1 closes it.

## 2. What the active-set loop needs, read from `PureDAQP/src/ldp.jl`

The loop never solves a KKT system, so the base's `LinearSystem` contract
(`PureQPBase/src/linsys.jl`) gives it nothing. What it does with `M = A R⁻¹`, and where:

| operation | where | today |
|---|---|---|
| row `r` of `M` as a dense length-`n` vector | `activate!` → `add_row!` every time a row enters; `full_set_step!` | `row(ws, r) = view(ws.Mt, :, r)`, contiguous |
| `Mu` over all rows, or a window of rows | `solve_ldp!` pricing; `set_targets!` once per pass | `mul!(price, transpose(Mt), u)` |
| `R⁻ᵀv` | `inner_solve!`, once per pass | `ldiv!(red.Rt, v)` with `Rt = transpose(R.U)` held once |
| `R⁻¹x` | `primal!`, once per pass | `ldiv!(transpose(red.Rt), x)` |
| the row norm of every row of `M` | `finish_reduction`, once at setup | computed from the dense `Mr` |

Two identities make every row of that table available without forming `M` (**algebra**,
relative agreement against the dense `M` at the §0 size):

    M[r, :] = R⁻ᵀ (Aᵀ e_r)          3.97e-10   (one row of A, then one triangular solve)
    M u     = A (R⁻¹ u)             (a triangular solve, then one apply of A)

So the loop needs three things of its operands, and nothing else:

1. `mul!(y, A, x)` — every `AbstractMatrix` the base accepts has it, including `ProductOperator`.
2. row `r` of `A` as a dense vector — a seam the base does not yet have (§3.2).
3. an `R` with `ldiv!(R, v)` and `ldiv!(transpose(R), v)`, in place and allocation-free, in
   `P`'s own form — a capability the base does not yet have (§3.1). **This is the one that
   decides support**: `A` is only ever applied and read row by row, so any `A` works; `P`
   must be factorable.

The working set itself is already fine: `WorkingSetQR` factors `Mₐᵀ`, which is `n × k` with
`k ≤ n` (`PureDAQP/src/qrset.jl`), and `WorkingSetGram` gathers the active rows as `n × kmax`
(`ldl.jl`). Neither scales with `m·n`. Both take a dense row vector through `add_row!`, so
neither changes.

Per-iteration cost of the implicit form against the dense form on the §0 problem (**measured**,
Chairmarks medians):

| | dense `Mt` | implicit, dense `R` | implicit, Kronecker operands |
|---|---|---|---|
| row of `M` | 0.20 µs (view copy) | 3.84 µs (`Aᵀe_r` Kronecker adjoint apply) + 17.1 µs (`R⁻ᵀv`, `n = 625` triangular solve) | 0.29 µs (outer product of the two factor rows) + a `25×25` two-sided solve |
| pricing `Mu` | 96.7 µs (`gemv` against `n×m`) | 18.1 µs (`R⁻¹u`) + 2.51 µs (Kronecker apply) | ≈ 3 µs |

Pricing is the bulk of an iteration (the comment in `solve_ldp!` says why), and the implicit
Kronecker form prices in about 1/30 of the dense time. The old plan's worry that row extraction
"is cheap enough not to dominate" is settled: it is cheaper than the dense pricing it replaces.
The Kronecker adjoint apply and the Kronecker apply allocate nothing (**measured**, 0 allocs);
the outer-product row written with `reshape` and `view` allocates 144 bytes, so it is written
as loops (§7 S1).

## 3. The design: two seams in the base, one consumer change in PureDAQP

### 3.1 `has_cholesky_factor(P)` and `cholesky_factor(P, shift)` — in PureQPBase

Split predicate from value, as `is_scalar_multiple`/`scalar_multiple` are in
`PureQPBase/src/kronecker.jl` and for the same reason: a `Union{Nothing, R}` return is a call
`--trim` refuses to resolve however plainly the caller narrows it.

    has_cholesky_factor(P) -> Bool         # true for every representation below; false for ProductOperator
    cholesky_factor(P, shift) -> R          # the upper factor of P + shift*I, in P's own form;
                                            # throws ArgumentError, naming the remedy, when it is not positive definite

`R`'s contract, which is what PureDAQP consumes and what `@verify` asserts per type at the end
of the module, as `LinearSystem` and `Preconditioner` are asserted today:

- `size(R) == size(P)`;
- `LinearAlgebra.ldiv!(R, v::AbstractVector)` overwrites `v` with `R⁻¹v`;
- `LinearAlgebra.ldiv!(Rt, v)` for `Rt = transpose(R)` overwrites `v` with `R⁻ᵀv`. The consumer
  forms `transpose(R)` **once** and holds it: on Julia 1.13 a `transpose` wrapper built inside a
  solve is a heap allocation (**measured**, 16 bytes per `ldiv!(transpose(UpperTriangular), v)`),
  which is the reason `DAQPReduction.Rt` exists today;
- both allocate nothing on `Vector{T}` operands, proved by `@assert_noalloc` in the module-level
  check block that `PureQPBase/src/PureQPBase.jl` already runs for every backend.

Methods, by representation the base owns:

| `P` | `R` | notes |
|---|---|---|
| `StridedMatrix`, `Symmetric{<:StridedMatrix}` | `UpperTriangular{T, Matrix{T}}` from `cholesky!(Symmetric(Hs), NoPivot(); check = false)` on a symmetrized, shifted copy | the code in `reduce_qp` today, moved. Its `μI` shortcut (`scalar_diagonal`) stays and returns the same type, `UpperTriangular(Matrix(s*I, n, n))`, so `update!` keeps one reduction type |
| `Diagonal` | `Diagonal(sqrt.(d .+ shift))` | throws on a non-positive entry |
| `BlockDiagonal` | a `BlockDiagonal` of `UpperTriangular` blocks | `ldiv!` block by block. **Unverified**: whether `ldiv!` over `transpose(block)` allocates a wrapper per block; if so, call `BLAS.trsv!('U', 'T', 'N', parent(block), view)` directly. `@assert_noalloc` decides |
| `KroneckerOperator` | `KroneckerCholesky{T}`: `R₁ = chol(P₁)`, `R₂ = chol(P₂)` as `UpperTriangular`, their two transposes held, one `n₂×n₁` scratch | **requires `shift == 0`**, see §5. `chol(P₁ ⊗ P₂) = chol(P₁) ⊗ chol(P₂)` (**algebra**, 5.7e-13) |
| `SparseMatrixCSC` | CHOLMOD factor with its permutation, `L` extracted into CSC as `factor_csc!` in `PureQPBaseSparseArraysExt.jl` already does | S6, on request. CHOLMOD's own `ldiv!` allocates per call (`SparseArraysExt.jl:872`), so the two solves are written over the extracted `L` |
| `SymTridiagonal`, `BandedMatrix` | LAPACK banded Cholesky | S6, on request. `hasmethod(cholesky, Tuple{SymTridiagonal{Float64, Vector{Float64}}})` is `true` (**measured**) |
| `ProductOperator` | — | `has_cholesky_factor` is `false`. A product cannot be factored |

The Kronecker solves, with `X` the `n₂×n₁` scratch and `vec` column-major (**algebra**, the
same convention `KroneckerOperator.mul!` uses):

    (R₁ ⊗ R₂) vec(X)        = vec(R₂ X R₁ᵀ)
    (R₁ ⊗ R₂)⁻¹ vec(Y)      = vec(R₂⁻¹ Y R₁⁻ᵀ)     copyto!(X, y); ldiv!(R₂, X); rdiv!(X, R₁ᵀ); copyto!(y, X)
    (R₁ ⊗ R₂)⁻ᵀ vec(Y)      = vec(R₂⁻ᵀ Y R₁⁻¹)     copyto!(X, y); ldiv!(R₂ᵀ, X); rdiv!(X, R₁); copyto!(y, X)

where `R₁ᵀ`, `R₂ᵀ` are the held transposes. `copyto!` rather than `reshape`, for the reason
`KroneckerOperator.mul!` gives.

### 3.2 `dense_row!(dest, A, i)` — in PureQPBase

Row `i` of `A` written into `dest::AbstractVector` of length `n`. The column-side seam
`structural_rows(A, j)` exists for equilibration; this is its row-side counterpart for a
consumer that reads rows.

| `A` | method | cost |
|---|---|---|
| any materializable `AbstractMatrix` | `dest[j] = A[i, j]` over `axes(A, 2)` | `n` reads; the fallback |
| `StridedMatrix` | the strided copy | as above, kept for clarity |
| `KroneckerOperator` | `dest` viewed as `n₂×n₁` is the outer product of row `i₂` of `A₂` and row `i₁` of `A₁`, with `(i₁, i₂) = divrem(i - 1, m₂)` (**algebra**, 3.97e-10, same as the adjoint apply) | `n` multiplies, written as loops |
| `BlockDiagonal` | zero `dest`, copy the one block's row into its column range | |
| `ProductOperator` | `fill!(column, 0); column[i] = 1; mul!(dest, opt, column)` — one adjoint apply | the wrapped map's own cost; allocation-free iff its `mul!` is |

`ProductOperator.column` is allocated only with `probe = true` today (length `rows`). Allocate it
always — `8m` bytes — so the adjoint-apply method has its basis vector without `probe`, and
`no_entries()` is no longer the answer to a row request. `basis` (length `cols`) stays
probe-only.

### 3.3 PureDAQP consumes both, and keeps its dense path bit for bit

`LDPWorkspace{T, WS}` gains a third parameter for the reduced rows, replacing the field
`Mt::Matrix{T}` with `M::MR` and two internal methods:

    row(M, r)             -> AbstractVector     # row r of M; a view for the dense form, a filled buffer otherwise
    price!(dest, M, u, rows::UnitRange)         # dest[rows] = (M u)[rows]

Two implementations, both in `ldp.jl`:

- `DenseRows{T}` holds `Mt::Matrix{T}` exactly as today. `row` is the contiguous view, `price!`
  is the `mul!` against `transpose(view(Mt, :, rows))` — the code that is there now, moved
  behind the two names. **Selected whenever `A isa StridedMatrix` and
  `R isa UpperTriangular{T, <:StridedMatrix}`**, which is the dense pair. Nothing about its
  arithmetic changes, so `test/solve_tests.jl`'s libdaqp comparisons and every dense item pass
  unchanged; that is the gate. A `SparseMatrixCSC` `P` or `A` is converted to a `Matrix` in
  `setup_backend` first, as today, so the sparse cells of §1 keep their "yes, by densifying"
  until S6 replaces the conversion; the conversion is the one place in PureDAQP that forms a
  matrix, and its comment says S6 is what removes it.
- `ImplicitRows{T, MA, R, RT}` holds `A`, `R`, `Rt = transpose(R)` and two length-`n` buffers.
  `row(M, r)` is `dense_row!(buf, A, r); ldiv!(Rt, buf); buf ./= scale[r]`. `price!` is
  `copyto!(tmp, u); ldiv!(R, tmp); mul!(dest, A, tmp); dest ./= scale` over **all** rows,
  whatever `rows` was: a window buys nothing when the apply prices everything, and pricing all
  then selecting inside the window keeps `scan = :window`'s trajectory identical to the dense
  one rather than silently changing which row enters. The `ActiveSet` docstring says the
  window's saving does not apply here.

`DAQPReduction` holds whatever `R` `cholesky_factor` returned (its `F` parameter widens from
`<: Cholesky` to unconstrained) and `Rt = transpose(R)` as it does now; `inner_solve!` and
`primal!` are unchanged. `finish_reduction` computes the row scales as
`‖R⁻ᵀ (Aᵀ e_r)‖` row by row through `row` on the implicit form — `m` row extractions at setup,
about `2208 × 21 µs ≈ 46 ms` with a dense `R` and far less with a Kronecker one (**measured**
components, §2). The Kronecker outer-product shortcut for the norms (**algebra**, 1.1e-10, from
the superseded plan) is an optimization for later, not part of this design.

`setup_backend` replaces the `is_materializable` refusal with

    has_cholesky_factor(P) || refuse_unfactorable_P()
    R = cholesky_factor(P, resolved.eps_prox)

and the same in `update!` when `P` or `A` changes. This section first called for a check in
PureDAQP that the replacement `P` and `A` keep the reduction's type, to refuse a Kronecker `P`
replaced by a dense one by name instead of failing at the `ws.red = …` assignment. The base's
`validate_update!` already does that: it requires `P isa MP` and `A isa MA` for the types the
workspace was built with, which fixes the reduction's type, and refuses with "P must keep the
representation the workspace was built with". PureDAQP adds nothing, and its test asserts the
base's message.

The type aliases the guarantees are stated against grow by one: `DenseWorkspace{T}` stays, and
`KroneckerWorkspace{T}` names the `KroneckerOperator` pair on `ImplicitRows`. `@strict_function
signatures = […]` on `warm_start!`/`cold_start!` and the module-level `let` block in
`PureDAQP/src/PureDAQP.jl` run `run_daqp!`, `multipliers!`, `build_solution` and
`reset_working_set!` on both. For a caller's own operator wrapped in `ProductOperator`, the
allocation guarantee is conditional on the wrapped `mul!`, and `docs/src/guarantees.md` says so.

## 4. Which compositions of `P` and `A` are supportable — for the converter

`A` composes freely. It is only ever applied (`mul!`) and read row by row (`dense_row!`, one
adjoint apply per row for an opaque map), so **every** LinearMaps composition works — `vcat`,
`hcat`, `*`, `+`, `kron`, `blockdiag`, `FunctionMap` — at the cost of `m` adjoint applies at
setup for the row norms and one per row entering the working set. A composition the LinearMaps
extension can unwrap to a base type (below) gets the base's cheaper `dense_row!` instead.

`P` is where support branches, because `M = A R⁻¹` needs `R`. Types are what LinearMaps 3.11.4
builds (**measured**):

| you write | LinearMaps builds | factorable? | how |
|---|---|---|---|
| `LinearMap(B)`, `B` a matrix | `WrappedMap{T, Matrix{T}}` | yes | the extension unwraps it to `B`; `cholesky_factor(B, ε)` |
| `kron(LinearMap(P₁), LinearMap(P₂))` | `KroneckerMap{T, Tuple{WrappedMap, WrappedMap}}` | yes, **with `eps_prox = 0`** and `Matrix` factors | unwrapped to `KroneckerOperator(P₁, P₂)`; `R = R₁ ⊗ R₂`. A `Diagonal` or `Symmetric` factor cannot be held by a `KroneckerOperator` (`similar` of it is not that type) and throws in the conversion; sparse factors unwrap but have no factor |
| `blockdiag(LinearMap(P₁), …)` | `BlockDiagonalMap{T, Tuple{WrappedMap, …}}` | yes, when every block is a dense matrix | unwrapped to `BlockDiagonal([P₁, …])`; block-wise `R`. A `Diagonal` or sparse block unwraps but has no factor, and `ActiveSet` refuses it |
| `c * P` for `c > 0` | `ScaledMap{T, T, …}` | yes when its map is | `R = √c · R(P)`; the extension folds `c` into the innermost matrix or into one Kronecker factor |
| `B' * B` or `transpose(B) * B` | `CompositeMap{T, Tuple{WrappedMap{T, Matrix{T}}, WrappedMap{T, Adjoint{T, Matrix{T}}}}}` — `maps[1]` is `B`, `maps[2]` its adjoint | yes when `B` has entries | `R = qr([B; √ε I]).R` (**literature**: `BᵀB + εI = RᵀR` for the `R` of that QR). Recognized by checking `parent(maps[2].lmap) === maps[1].lmap`. S6, on request |
| `P₁ + P₂` | `LinearCombination` | **no** | `chol(P₁ + P₂)` is not a function of `chol(P₁)` and `chol(P₂)`. Refused |
| `B * C` in general | `CompositeMap` | **no** unless it is the Gram pattern above | not symmetric in general; refused |
| `FunctionMap`, any opaque map | `FunctionMap` | **no** | products only; refused by name |
| `vcat`, `hcat` | `BlockMap` | not a `P` | not square |

Two consequences the converter should build to:

1. **Emit the base's types, or the LinearMaps forms the extension unwraps to them.** A
   `KroneckerOperator` or `BlockDiagonal` `P` reaches every algorithm's structured path:
   `ActiveSet` through `cholesky_factor`, `OperatorSplitting` through its `:kronecker` and
   `:block` backends, `InteriorPoint` through `:block`. Today the LinearMaps extension wraps
   *every* map, `kron` of two matrices included, into an opaque `ProductOperator`
   (`PureQPBaseLinearMapsExt.jl`, `as_operator`), so a Kronecker `P` built with LinearMaps is
   currently invisible to all three. S4 fixes that.
2. **Do not sum maps to build `P`.** `P = P₀ + εI` is the one sum with a factor, and it is
   handled by passing `ε` as `eps_prox` rather than by building the sum — except for a
   Kronecker `P₀`, §5.

## 5. What this design cannot do, written down now

- **An opaque `P` under `ActiveSet`.** `has_cholesky_factor` is `false` for a
  `ProductOperator`, and `setup` refuses with a message naming what would work: a matrix, a
  `KroneckerOperator`, a `BlockDiagonal`, a LinearMaps `kron`/`blockdiag` of matrices, or
  `OperatorSplitting()`. The alternative the superseded plan carried — running the loop on
  `P`-solves alone, which forces `working_set = :gram` — is dropped from this design on a
  measurement: on the §0 problem `working_set = :gram` reports `NUMERICAL_ERROR` at iteration
  504 where `:rows` solves in 1510 (**measured**; the Gram form squares the conditioning and
  its pivot cannot separate a dependent row from an independent one below about `1e-8`,
  `qrset.jl` header). A path that buys memory by giving up the rank test on the very problems
  that need the memory is not worth its code.
- **A Kronecker `P` with `eps_prox > 0`.** `P₁ ⊗ P₂ + εI` has no Kronecker Cholesky factor
  (`(P₁ + aI) ⊗ (P₂ + bI)` carries cross terms `bP₁ ⊗ I + aI ⊗ P₂`, **algebra**). The default
  `eps_prox = 0` reaches the Kronecker path; a positive one is refused by name. A singular
  Kronecker `P` therefore has no `ActiveSet` path in this design.
- **`InteriorPoint` on operators without a caller-supplied preconditioner.** The proposal that
  `cholesky_factor(P)` is "exactly the IPM's missing piece" is **refuted by measurement**.
  With `preconditioner = cholesky(Symmetric(P))` on the §0 Kronecker operators
  (`linsys = :indirect, scaling = 0`) the interior-point method ends `NUMERICAL_ERROR` at
  outer iteration 41 after 8565 CG iterations; the exact lagged Cholesky of the whole reduced
  matrix (`bench/ipm_preconditioners.jl`'s `LaggedCholesky`, `every = 1`) solves in 46 outer /
  114 CG iterations and `every = 3` in 47 / 1133. The mechanism (**measured** on the dense
  reduced matrix `P + δI + Aᵀ diag(w) A`, `δ = 1e-8 = ipm_floor(Float64)`):

  | weights | `cond(K)` | preconditioned by `chol(P)` | Jacobi |
  |---|---|---|---|
  | `w = 1`, the starting point | 5.6e7 | 2.2e7 | 2.8e7 |
  | 10% of rows active at `1/δ`, rest at `δ` | 1.1e15 | 3.4e14 | 4.2e14 |
  | 50% active | 8.0e13 | 8.9e14 | 3.9e13 |

  `chol(P)` preconditions `P`; the interior-point matrix is dominated by `Aᵀ diag(w) A` whose
  weights spread over sixteen orders of magnitude, and a factor of `P` does nothing about that
  term. The IPM column of the §1 table therefore keeps its qualification, and the docs say so.
  **Unverified** candidates for closing it, outside this design: a Woodbury correction of
  `chol(P)` over the rows whose weights are large, which is `RᵀR + (1/δ)AₐᵀAₐ` and the same
  `M = A R⁻¹` object the active-set loop holds; and the low-rank reduced form as a
  preconditioner rather than a solver, which `PureIPM/src/workspace.jl`'s `lowrank_rung`
  docstring measured too inaccurate as a solver (`9e-9`) but CG may tolerate. Either is a
  measurement first.
- **`scan = :window` saves nothing on an implicit reduction.** The trajectory is preserved
  (§3.3); the cost is not. Documented in the `ActiveSet` docstring.
- **Sparse `P` and sparse `A` stay densified until S6.** `has_cholesky_factor(::SparseMatrixCSC)`
  is `false` in the base until S6, and PureDAQP converts a sparse operand to a `Matrix` before
  the reduction exactly as it does today (§3.3), so the sparse cells of §1 neither improve nor
  regress in S3. A caller who needs a sparse `P` factored sparsely, or a sparse `A` read by
  rows in `O(nnz)`, asks for S6.
- **The equilibrated forms.** `ActiveSet` requires `scaling = 0` today and keeps requiring it.

## 6. Requirements

| # | requirement | status |
|---|---|---|
| R1 | `ActiveSet` accepts every representation the base owns; the dense, structured and unmaterialized forms never form `P` or `A`, and sparse keeps its densifying path until S6; a `ProductOperator` `A` with a factorable `P` solves | **done** (S3; a solve that reaches `SOLVED` over a `ProductOperator` `A` needed `report` to ask for `Aᵀy` through `adjoint`, closed in S5) |
| R2 | The base owns `has_cholesky_factor`/`cholesky_factor` for `StridedMatrix`, `Symmetric`, `Diagonal`, `BlockDiagonal`, `KroneckerOperator`; a `ProductOperator` declines; `R`'s `ldiv!` pair is allocation-free and `@verify`-asserted | **done** (S2). `has_cholesky_factor` is `true` for a `BlockDiagonal` only when every block is a dense matrix, and for a `KroneckerOperator` only when its factors are strided |
| R3 | The base owns `dense_row!` for materializable matrices, `KroneckerOperator`, `BlockDiagonal`, `ProductOperator` | **done** (S1) |
| R4 | The implicit reduction runs `working_set = :rows` with the exact `\|R_ii\|` rank test; the Kronecker pair reaches the dense pair's objective | **done** (S3) |
| R5 | A test asserts the workspace's storage does not grow with `m·n` (§8) | **done** (S3) |
| R6 | `run_daqp!` and the other loop kernels are proved allocation-free and trim-compatible on `KroneckerWorkspace{Float64}` as they are on `DenseWorkspace{Float64}`; the guarantee is stated as conditional for a caller's operator | **done** (S3 proofs; S5 states the condition in `docs/src/guarantees.md` and tests it). The proof item `a dual active-set iteration allocates nothing and trims, proved` passes on all three workspaces: `active_product!` copied between two views of a vector, which allocates because the copy cannot tell whether they alias, and an explicit loop replaces it |
| R7 | Every unsupported form is refused by name: opaque `P`; Kronecker `P` with `eps_prox > 0`; an `update!` that changes `P`'s or `A`'s representation | **done** (S3). The last is refused by the base's `validate_update!`, which already requires `P isa MP`, so PureDAQP carries no check of its own |
| R8 | The LinearMaps extension unwraps `WrappedMap`, `KroneckerMap` and `BlockDiagonalMap` of matrices, and `ScaledMap` of any of them, to the base's types; everything else stays a `ProductOperator` | **done** (S4). A `KroneckerOperator` can only be built from `Matrix` factors, so a `kron` of wrapped `Diagonal` or `Symmetric` matrices throws in the conversion rather than staying a `ProductOperator` |
| R9 | `docs/src/matrices.md` carries one table of algorithm × representation saying what each does, and the §4 composition table; `linearmaps.md` says what the extension unwraps | **done** (S5; the composition table is under "What a composed map becomes" in `matrices.md`) |
| R10 | The dense path is unchanged: every existing PureDAQP test item passes without edit, libdaqp comparisons included | **done** (S3) |
| R11 | `setup` no longer densifies a `KroneckerOperator` `P` for `is_convex`, `is_symmetric` or `check_finite` | **done** (S1) |

Dropped from the superseded plan: its R3 (opaque `P` via `P`-solves on `:gram`) and R7 (the
IPM gap), both on the measurements in §5.

### 6a. The invariant the whole package owes, and where it is broken

R1–R11 cover one algorithm reaching every representation. They do not state the property that
motivates it, which applies to all three:

> A dense `P` and `A` are dense. Symmetry stated by the caller is used, not discarded. A sparse
> `P` or `A` stays sparse. A structured one stays structured. An unmaterialized one is never
> materialized. For every algorithm.

**Measured** on `kron_problem(11)` at `scaling = 0` — `n = 625`, `m = 2208`, four Kronecker
factors totalling 0.027 MiB, against 13.5 MiB for the dense pair:

| | dense | `Symmetric` `P` | sparse | structured | unmaterialized |
|---|---|---|---|---|---|
| `OperatorSplitting` | held | discarded | stays sparse | **densified**, 13.99 MiB, backend `cholesky` | held, 0.52 MiB, backend `indirect` |
| `InteriorPoint` | held | discarded | stays sparse | **densified**, 124.63 MiB, backend `bunchkaufman` | **refused** |
| `ActiveSet` | held | discarded | **densified** | held (R1) | held (R1) |

R14 closes the structured column for the first two: `OperatorSplitting` reaches the matrix-free
backend and `InteriorPoint` refuses rather than forming. One consequence to state plainly,
because it reaches callers who never name a backend: the matrix-free backend lives in the
Krylov extension, so a structured pair that no structured rung accepts now needs `using Krylov`
where it previously fell through to a factorization. A `BlockDiagonal` pair is unaffected — the
block rung serves it, and the refusal sits below every structured rung rather than in front of
them.

Two findings behind that table. **Unmaterialized does not force an iterative solve**: the base
owns direct backends that form nothing — `:kronecker` eigendecomposes the two factors,
`:lowrank` solves by Woodbury, `:block` solves block by block. What is true is narrower, that
those backends are unreachable here. `:kronecker` refuses an `A` that is not itself a
`KroneckerOperator` (`PureQPBase`), and `InteriorPoint` refuses `:kronecker` and `:lowrank`
outright, because both assume one weight for every row while an interior-point method's weights
differ per row and an active row's reaches `1/reg_dual`. And **symmetry costs rather than
saves**: `setup` allocates 32 bytes *more* for `Symmetric(P)` than for the same `Matrix`, in
all three, the wrapper and nothing else.

| # | requirement | status |
|---|---|---|
| R12 | A caller's `Symmetric` is used rather than re-derived: the symmetrising copy is skipped and the factorisation reads one triangle. All three algorithms, so it belongs in the base | not started |
| R13 | A sparse `P` or `A` keeps its representation through `ActiveSet`'s reduction | not started |
| R14 | `linsys = :auto` does not choose a materialising backend for a structured `P` or `A`, in any algorithm | **done**. `holds_structure(M)` in the base is the predicate, separate from `is_materializable` because polishing and the derivatives do read a structured operand's entries; the rungs that form the reduced matrix decline on it. `OperatorSplitting` reaches the matrix-free backend instead, measured on a Kronecker pair at `n = 1600` as 0.72 MiB against 43.77 MiB and 43.3 ms against 190.6 ms. `InteriorPoint` has no structured rung for such a pair and refuses by name, since its own matrix-free path needs a caller's preconditioner |
| R15 | `:kronecker` serves a structured `P` with a differently-represented `A`, and the reverse | not started |
| R16 | The structured backends admit a weight per row, so `InteriorPoint` can reach them | not started |
| R17 | `InteriorPoint` accepts an unmaterialized pair | not started |

R14 is a choice among backends the base already has, so it is the cheapest of these and worth
measuring first. R16 carries the algebra: a Kronecker product times a general diagonal weight is
not a Kronecker product, the same obstruction as `chol(P₁⊗P₂ + εI)` in §3.1. Only the active
rows reach `1/reg_dual`, so splitting the weights into a uniform part and a low-rank correction
over those rows is the candidate — **unverified**, and to be measured before it is designed.

## 7. Steps

Each step leaves the repository green and is committed on its own; the session may stop after
any of them. Gate for every step: the named test items through `julia_run_testitems` with
`max_workers` set (1 while iterating, 2–4 for a suite); `runic -i .` before the commit; the
package's `Pkg.test()` cold as the pre-commit gate for a step that touches `src/`. **Sonnet**
marks a step a Sonnet-class model implements from this document alone; **Opus** marks one that
needs judgment about the loop's invariants.

| # | step | done when | who |
|---|---|---|---|
| S1 | **Kronecker `P` without densifying, and `dense_row!`.** In `PureQPBase/src/kronecker.jl`: `is_symmetric(K) = issymmetric(K.A1) && issymmetric(K.A2)`; `is_convex(T, K, σ)` from the two factors' eigenvalues, `min λ(P₁) ⋅ λ(P₂) + σ > 0` over all pairs (both factors symmetric, so `eigvals(Symmetric(·))`), which is exact and `O(n₁³ + n₂³)`; `check_finite` over the factors. In a new `PureQPBase/src/rows.jl` (included after `operator.jl`): `dense_row!` with the five methods of §3.2, the Kronecker one as two loops over `axes(A2, 2)` and `axes(A1, 2)`; `ProductOperator.column` allocated unconditionally. Tests in `PureQPBase/test/kronecker_tests.jl` and a new `rows_tests.jl`: each `dense_row!` equals `Matrix(A)[i, :]` for every type, on `Float32` too; `is_convex` agrees with the dense answer on a Kronecker `P` at `σ` above and below the threshold; `@assert_noalloc dense_row!` on each type in `test/strictmode_tests.jl`. | those items pass; `Base.summarysize` of `setup(Pop, …, OperatorSplitting(); scaling = 0)` no longer includes an `n×n` dense array (assert in the test) | Sonnet |
| S2 | **`has_cholesky_factor`/`cholesky_factor` in the base.** New `PureQPBase/src/cholesky.jl` (included after `kronecker.jl` and `blockdiagonal.jl`): the dense method moved from `reduce_qp` with its two error messages and its `μI` shortcut; `Diagonal`; `BlockDiagonal`; `KroneckerCholesky{T}` with the `ldiv!` pair of §3.1 and the `shift == 0` refusal; `has_cholesky_factor(::ProductOperator) = false`, `true` for the rest, `false` by default for any other `AbstractMatrix`. `@strict_contract`/`@verify` for the factor types, mirroring `Preconditioner`. Tests in a new `cholesky_tests.jl`: for each type at small size, `ldiv!(R, ldiv!(transpose(R), copy(v))) ≈ (Matrix(P) + shift I) \ v`; the not-positive-definite refusals; the Kronecker shift refusal message; `@assert_noalloc` on both solves. | items pass; `PureDAQP` untouched and its suite still green | Sonnet |
| S3 | **The implicit reduction in PureDAQP.** §3.3 in full: `DenseRows`/`ImplicitRows`, `row`/`price!`, `LDPWorkspace` and `DAQPReduction` parameters, `finish_reduction` for both forms, `setup_backend`/`update!` through `cholesky_factor`, the `validate_update!` representation check, `KroneckerWorkspace{T}`, the module-level proofs, `refuse_unfactorable_P`. `reduce_qp` keeps its signature for dense inputs so `test/` needs no edit. New items in `PureDAQP/test/solve_tests.jl`: the §0 Kronecker pair (`KroneckerOperator(P1, P2)`, `KroneckerOperator(A1, A2)`, built from a seeded generator in `test/helpers.jl` — the session's `kron_problem` is the model) solves `SOLVED` with `:rows` to the dense pair's objective within `1e-6` relative, and a small Kronecker `A` with a `Matrix` `P` too; the §8 storage test; `working_set = :gram` on the Kronecker pair reaches the same answer as `:rows` on a well-conditioned instance; `update!` that changes representation is refused by name; `eps_prox > 0` with a Kronecker `P` is refused by name; a `ProductOperator` `P` is refused by name and the message names the remedies. `test/strictmode_tests.jl` proves the kernels on `KroneckerWorkspace{Float64}`. | every existing PureDAQP item passes unedited (R10); the new items pass; the `bench/daqp_headtohead.jl` numbers for the dense suite are unchanged within noise (run it once, compare to `bench/results/`) | **Opus** |
| S4 | **The LinearMaps extension unwraps what the base can hold.** `as_operator` in `PureQPBaseLinearMapsExt.jl` gains methods: `WrappedMap` → its matrix; `KroneckerMap` of two `WrappedMap`s → `KroneckerOperator`; `BlockDiagonalMap` of `WrappedMap`s → `BlockDiagonal`; `ScaledMap` of any of those → the scalar folded into the matrix (a copy) or into the first Kronecker factor; anything else → `ProductOperator` as now. The same for the SciMLOperators extension is out of scope. Tests in `PureQPBase/test/operator_tests.jl`: `kron(LinearMap(P1), LinearMap(P2))` under `OperatorSplitting` with `P = μI` reaches `backend_name == :kronecker`; the §0 problem written with `kron` on both sides solves under `ActiveSet`; a `FunctionMap` still arrives as a `ProductOperator`. | items pass | Sonnet |
| S5 | **Documentation.** `docs/src/matrices.md`: replace the single backend table's "an operator — never formed" row with a table of algorithm × representation (dense, sparse, structured, unmaterialized) stating for each cell what happens — `ActiveSet`'s cells read "implicit `A R⁻¹`, `R` in `P`'s form" / "refused: `P` must have a Cholesky factor"; `InteriorPoint`'s unmaterialized cell keeps "with a caller-supplied preconditioner" and points at §5's measurement. Add the §4 composition table under a heading the converter can link to. `docs/src/linearmaps.md`: what the extension unwraps. `ActiveSet` docstring: the operator support, the `scan = :window` note, the `eps_prox` restriction. `PureDAQP`'s module docstring loses "reads `P` and `A` as dense matrices". `docs/src/guarantees.md`: the conditional allocation guarantee. | `docs` build green; `grep -n "entry by entry" docs/src PureDAQP/src` is empty | Sonnet |
| S6 | **On request only, each its own commit:** (a) `SparseMatrixCSC` `P` in the SparseArrays extension over the extracted CSC factor and its permutation, and a CSR copy of a sparse `A` for `dense_row!`; (b) `B' * B` recognition in the LinearMaps extension with `qr([B; √ε I]).R`; (c) `SymTridiagonal`/`BandedMatrix` `P`; (d) the Kronecker row-norm shortcut in `finish_reduction`; (e) `bench/working_set_choice.jl` and `bench/daqp_headtohead.jl` extended with the operator forms so the cost of not materializing is a saved number. Each carries the same agreement, refusal and `@assert_noalloc` items as S2/S3. | per item | Sonnet |

Order: S1, S2, S4 are independent of each other and of S3; S3 needs S1 and S2; S5 needs S3
and S4. Dispatch S1, S2 and S4 in parallel if hands allow; nothing in S4 touches a file S1 or
S2 touches.

Status: S1 (`ea83d38`), S2 (`1c543ed`), S4 (`675828d`), S3 (`fcadf30`) and S5 are done. Only S6
remains, on request. S5 also fixed what the documentation pass found wrong in the code: `report`
asked for `Aᵀy` through `transpose`, which an operator `A` answers by reading entries, so a
solve over a `ProductOperator` `A` threw when it came to report its residuals.

## 8. Verification

- **Storage does not scale with `m·n`** (R5), the requirement a materializing shortcut would
  violate, so it is a test and not a comment. In `PureDAQP/test/solve_tests.jl`: build the
  Kronecker pair at `(n₁, n₂, m₁, m₂) = (25, 25, 46, 48)` and again with `m₂ = 4·48`, so `m`
  grows by `Δm = 3·2208 = 6624` rows at fixed `n = 625`; assert
  `Base.summarysize(ws₄) - Base.summarysize(ws₁) < Δm · 64 · sizeof(T)`. A materialized
  reduction grows by `Δm · n · sizeof(T)`, which is `Δm · 625 · sizeof(T)` here and so misses a
  bound stated per row rather than per row times `n` — the property being asserted is that
  growth does not scale with `n` at all. The implicit one grows by
  the dozen length-`m` vectors the loop holds plus `A₂`'s own rows, about `Δm · 16 · sizeof(T)`
  (**measured** components: `Mt` is `10.5 MiB` at `m = 2208`, the whole dense workspace
  `36.3 MiB`). The same assertion once more with `A` as a `ProductOperator` over a
  `FunctionMap` of the Kronecker apply, which is the caller's actual shape.
- **Agreement**: every operator form reaches the dense form's objective to a tolerance recorded
  per form in the test (`1e-6` relative is what the §0 problem gives at `cond(A) = 2.4e11`;
  bit-for-bit is not expected and not asserted, `docs/src/matrices.md` already explains why a
  different factorization moves the last digits).
- **The rank test survives**: the "a feasible problem is never reported infeasible, however ill
  conditioned" item in `solve_tests.jl` runs once more on the implicit reduction with a
  Kronecker `A` of the same conditioning.
- **The dense path is untouched** (R10): the existing items pass unedited, and
  `bench/daqp_headtohead.jl` reproduces `bench/results/` within noise.
- **Allocation and trim** (R6): `PureDAQP/test/strictmode_tests.jl` proves the kernels on both
  workspace aliases with StrictModeTest; the module-level block reports on both; the base's
  block reports on every `cholesky_factor` type and every `dense_row!` method.
- **Refusals** (R7): each has an item asserting the message text with `@test_throws "…"`, per
  the house convention.
- **The audit**: `julia --project=bench bench/strictmode_audit.jl` unchanged in verdict after S3,
  with the new signatures added to its list.

## 9. What survives of `matrixfree-activeset.md`, and what does not

Survives, relocated into this design: the §1 measurement table; the diagnosis that `R`, not
the products, is the blocker; the Kronecker identities of its §2a (now §3.1 here); the idea of
a Cholesky-in-own-form capability (its §4 step 1), moved from PureDAQP into the base where the
representations live; the parametric `Mt` and buffered `row` (its §4 steps 2–3, now §3.3); the
storage test, the agreement test and the bench extension of its §9; requirements R1, R2, R4, R5,
R6, R8 in substance.

Dropped, with the reason: its §5 and R3 (opaque `P` via `P`-solves on the Gram working set),
because it gives up the rank test and is measured to fail on the motivating problem (§5 here);
its §7 and R7 (`InteriorPoint` served by the same capability), because the preconditioner it
proposed is measured not to work (§5 here) — the correct part of that section, that the IPM's
gap is a preconditioner and not an interface, stands; its placement of the capability in
PureDAQP, because the base owns the representations and each algorithm is meant to derive
from it rather than to grow a private copy.
