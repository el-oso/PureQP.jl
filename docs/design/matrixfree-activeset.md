# A dual active-set method that never forms its matrices

Design for accepting an unmaterialized `P` and `A` in [`ActiveSet`](@ref), so that all three
algorithms cover all four matrix representations — dense, sparse, structured, unmaterialized.

Claims marked **measured** were run in the session this document was written in, on `neuromancer`
at one BLAS thread, against a Kronecker problem matched to the shape and conditioning of a real
caller's problem: `n = 625`, `m = 2208`, `cond(P) = 8.3e8`, `cond(A) = 2.4e11`. Claims marked
**algebra** were checked numerically at that size to the precision quoted. Claims marked
**unverified** were not.

---

## 1. Where the four representations stand today

**Measured**, by passing each form of the same problem to each algorithm:

| | dense | sparse | structured | unmaterialized |
|---|---|---|---|---|
| `OperatorSplitting` | accepted | accepted | accepted | accepted |
| `InteriorPoint` | accepted | accepted | accepted | accepted only with a caller-supplied preconditioner |
| `ActiveSet` | accepted | accepted | accepted | **refused** |

So one cell is empty and one is qualified. `docs/src/matrices.md:30` promises four answers and
`:156` lists "an operator — never formed", so the documentation already claims what the code
does not do.

Two qualifications matter more than the table:

- `ActiveSet` densifies everything it accepts. `M = A R⁻¹` is dense whatever `A` was, so sparse
  and structured inputs are read entry by entry and the reduction's storage is `O(mn)`
  regardless. Accepting an operator by materializing it would therefore be *consistent* with
  the other three columns and still wrong: a caller passes an operator because the dense form
  is what they are avoiding. On the problem above the four Kronecker factors are 28 KiB against
  13.5 MiB for dense `P` and `A`, **492×** (**measured**). Materializing is not an option.
- `InteriorPoint` refuses an operator without a preconditioner, and the refusal is measured
  rather than cautious: "without one, or with the Jacobi diagonal, it does not reach the
  tolerance on most problems". At `cond(P) = 8.3e8` a Jacobi diagonal will not do, so its
  unmaterialized support is nominal for the class of problem this document is about. §7.

---

## 2. What the method needs, and which parts need `R`

The reduction is `M = A R⁻¹` for the Cholesky factor `R` of `P` (or `P + εI`), turning the QP
into `min ‖u‖²` subject to `lo ≤ Mu ≤ hi`. A Cholesky needs entries, so `R` is the blocker — not
the products.

Every quantity the loop uses, and its form without `R` (**algebra**, agreement quoted):

| used today | without `R` | agreement |
|---|---|---|
| `M(Mₐᵀμ)`, the price vector | `A P⁻¹ Aₐᵀ μ` | 3.6e-10 |
| `Mₐ · m_r`, Gram entries as a row enters | `Aₐ P⁻¹ a_r` | 7.3e-10 |
| `m_rᵀ m_r`, the new pivot | `a_rᵀ P⁻¹ a_r` | 7.5e-10 |
| `‖R⁻ᵀaᵢ‖`, the row scales | `√(aᵢᵀ P⁻¹ aᵢ)` | 1.1e-10 |

So the method can be written with **`P`-solves and `A`-applies only**. Three consequences set
the shape of the work:

1. **It must use the Gram working set.** `WorkingSetGram` needs the inner products
   `aᵢᵀP⁻¹aⱼ`, which a `P`-solve supplies. `WorkingSetQR` factors `Mₐᵀ` and needs the vectors
   `R⁻ᵀaᵢ` themselves, which need `R`. This is an unwelcome tension: `:gram` squares the
   conditioning of `Mₐ`, which is why `:rows` is the default. A matrix-free path is therefore on
   the numerically weaker representation, and on an ill-conditioned problem it will report
   `NUMERICAL_ERROR` where the dense `:rows` path solves. That is a real limit of this design,
   not a defect to fix later, and §6 says what to do about it.
2. **The iterate must stay `μ`, never `u`.** `primal_point!` forms `u = Mₐᵀμ` in the reduced
   space. Matrix-free, the loop carries `μ` and computes `A P⁻¹ Aₐᵀ μ`: one `Aᵀ` apply, one
   `P`-solve, one `A` apply per iteration.
3. **Row normalization costs `m` `P`-solves at setup**, one per row, or is dropped. It only sets
   what `primal_tol` means per row; `entering_row` already multiplies the scale back out to
   price in the caller's units, so dropping it changes which row enters first and not which
   point is optimal. **Unverified**: whether dropping it costs iterations.

### 2a. Structured operators are the easy case and should come first

When the operator carries an algebra, `R` is available *unmaterialized* and nothing above is
needed. For Kronecker (**algebra**):

    chol(P₁ ⊗ P₂) = chol(P₁) ⊗ chol(P₂)                    rel 5.7e-13
    (A₁ ⊗ A₂)(R₁ ⊗ R₂)⁻¹ = (A₁R₁⁻¹) ⊗ (A₂R₂⁻¹)             rel 4.3e-10
    row norms of M = outer product of the factors' row norms  rel 1.1e-10

`M` stays two small factors: 18.4 KiB against 10.5 MiB materialized, **587×** (**measured**).
The loop runs exactly as it does today, on factored operands, with an exact `R` and so an exact
`|R_ii|` rank test — the `:rows` representation keeps working. The same holds for any structure
closed under Cholesky and under division: block-diagonal, and `P = BᵀB` where `R = qr(B).R`.

This splits the work into a good case worth doing first and a general case that costs numerics.

### 2b. Composition, which is what a caller's converter produces

`LinearMaps` composes with `vcat`, `hcat`, `*`, `+`. For **`A`** composition is free: `A` is
only ever applied, and `M = A R⁻¹` inherits whatever composition `A` had. For **`P`** it is
where the plan branches, because `chol(P₁ + P₂)` is not a function of `chol(P₁)` and
`chol(P₂)`. A summed `P` has no factored Cholesky and lands in the general case; a `P` that is a
Gram, a Kronecker product, or block-diagonal lands in §2a.

**This is the open question that decides how much of §4 and §5 is needed**, and it is a question
about the caller's problem rather than about this package.

---

## 3. Requirements, as a checklist

| # | requirement | status |
|---|---|---|
| R1 | `ActiveSet` accepts an operator and never forms `P` or `A` | not started |
| R2 | A structured operator whose algebra supplies `R` uses `:rows` and the exact rank test | not started |
| R3 | An opaque product-only operator solves via `P`-solves on `:gram` | not started |
| R4 | Composed maps — `vcat`, `hcat`, `*`, `+` of `A`; Gram/Kronecker/block `P` | not started |
| R5 | No path materializes `P` or `A`, and a test asserts the memory does not grow with `m·n` | not started |
| R6 | The warm path stays allocation-free, or the guarantee is restated for operators | not started |
| R7 | `InteriorPoint`'s operator support reaches an ill-conditioned problem | not started, §7 |
| R8 | `docs/src/matrices.md` describes what each algorithm does with each of the four | not started |

---

## 4. Step 1 — structured operators, exact `R` (R2, part of R1 and R4)

The smallest change that closes the common case without touching the numerics.

1. A trait on the operator: can it supply a Cholesky factor of itself, in its own form?
   Spelled as a function returning `nothing` by default, so an operator opts in:
   `factored_cholesky(op) -> R or nothing`, with `R` supporting `ldiv!` and `rdiv!`.
   `KroneckerOperator` implements it as `KroneckerOperator(chol(A₁).U, chol(A₂).U)`.
2. `reduce_qp` asks for it. When it comes back non-`nothing`, `M = A / R` is formed **in the
   operator's own algebra** — for Kronecker, two small divisions — and the row norms from the
   factors. Everything downstream is unchanged: `Mt` becomes an operator rather than a
   `Matrix`, and `LDPWorkspace` is already parametric over the working set but **not** over
   `Mt`. That parameter has to be added.
3. `row(ws, r)` currently returns `view(ws.Mt, :, r)`, contiguous by construction. For an
   operator it becomes an apply into a buffer. This is the one hot-path change: it is called by
   `add_row!` every iteration. **Unverified**: whether a Kronecker row extraction is cheap
   enough not to dominate.

Deliverable: the Kronecker problem in §1 solves through `ActiveSet` with `:rows`, storage
independent of `m·n`, and the same answer as the dense path.

## 5. Step 2 — opaque operators, `P`-solves (R3)

Needed only if the caller's `P` has no factored Cholesky (§2b).

1. `ActiveSet` gains a `P`-solver for operators. PureQPBase already carries the machinery —
   `linsys = :indirect` with Krylov — and `ActiveSet` currently refuses `linsys` precisely
   because it factors instead. That refusal becomes conditional on the input being a matrix.
2. `solve_ldp!` is restructured to carry `μ` and never `u`, per §2.2. This touches
   `primal_point!`, `working_set_multipliers!` and the pricing call — the centre of the loop.
3. The working set is forced to `:gram`, and `working_set = :rows` with an opaque operator is
   refused with a message saying why, since the alternative is silently changing the rank test.
4. Row scales: `m` `P`-solves at setup, behind a setting, defaulting off with the scales left at
   one. §2.3.
5. Inexactness: every `P⁻¹` is now a CG solve to a tolerance. The rank test, the multiplier
   signs and `multiplier_noise` all assume exact solves today. **Unverified** and the main
   risk in this step: what CG tolerance the rank decision needs, and whether
   `multiplier_noise` has to grow to cover it.

Deliverable: an opaque `FunctionMap` of the §1 problem reaches an answer, with the accuracy it
reaches recorded rather than assumed.

## 6. What this cannot do, to be written down rather than discovered

The §5 path is on `:gram` with inexact solves, so on an ill-conditioned reduction it will report
`NUMERICAL_ERROR` where the dense `:rows` path solves. **Measured** on the §1 problem today:
`:gram` gives `NUMERICAL_ERROR` where `:rows` solves. An opaque operator therefore buys memory
at the cost of the rank test that today's default exists to provide, and `docs/src/matrices.md`
must say so next to the table rather than leaving a caller to find it. A structured operator
(§4) does not pay this.

## 7. Does `InteriorPoint` need work too? (R7)

Yes, but of a different kind. It already accepts operators; **measured**, it refuses this
problem without a caller-supplied preconditioner, and its own refusal message says a Jacobi
diagonal is not enough. So the gap is not the interface but the preconditioner: at
`cond(P) = 8.3e8` a caller has to supply something that works, and the package offers no way to
build one from a structured operator it already understands. A Kronecker `P` has an exact
preconditioner available — `R₁ ⊗ R₂` — and nothing exposes it.

Smaller than §4 and §5, and independent of them. It is a preconditioner story, not an algorithm
change.

**Measured, for scale**: on the §1 problem, `InteriorPoint` dense solves to 1.4e-11 of the
`ActiveSet` objective; `OperatorSplitting` dense reports `DUAL_INFEASIBLE`, and with an operator
reaches the iteration limit with an objective 7e18 out. So on this class of problem the
operator-capable algorithm is the one that cannot solve it, and the one that solves it is the
one that refuses operators. That is the whole reason this document exists.

## 8. Order, and what to decide first

1. **Answer §2b**: what is `P`'s structure in the real problem? If it has a factored Cholesky,
   §4 is the whole job and §5 may never be needed.
2. §4, structured operators. Self-contained, keeps the numerics, closes R2 and most of R1/R4.
3. §7, `InteriorPoint` preconditioners from structured operators. Independent, small.
4. §5, opaque operators. Largest, and the one that costs accuracy; worth doing only if step 1
   says the real problem needs it.
5. R8, the documentation, last — it describes what the code ended up doing.

## 9. Verification

- A memory test: solve the §1 problem through each path and assert the workspace's storage does
  not scale with `m·n` (R5). This is the requirement that a materializing shortcut would
  violate, so it is the one that must be a test rather than a comment.
- Agreement: every operator path reaches the dense path's objective, to a tolerance recorded per
  path rather than assumed equal.
- `bench/working_set_choice.jl` and `bench/daqp_headtohead.jl` extended with the operator forms,
  so the cost of not materializing is a measured number.
- The StrictMode audit over the new signatures, and R6 settled either way: an operator apply
  that allocates cannot be inside the no-allocation guarantee, so either the operator protocol
  requires an in-place apply or the guarantee is restated.
