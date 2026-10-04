# Backlog

Work that is scoped and measured but not yet done. `docs/src/roadmap.md` is a different
document: it records what libosqp does that PureOSQP does not.

## Error messages in PureQPBase interpolate eagerly

`ArgumentError("… $x … $y")` lowers to a `string` call, and from four arguments up stock
inference declines to specialize it and falls back to `print_to_string(::Symbol, ::Vararg{Any})`
— an `Any` vararg that `--trim=safe` cannot resolve. `juliac` raises that threshold in
`juliac-trim-base.jl`, so a real build accepts these and only an in-process verifier against
stock Base refuses them. `lazy"…"` builds a `LazyString` that formats in `show` instead, which
is clean under both and renders identically.

PureDAQP's messages are already `lazy`. PureQPBase has about thirty that are not, `validate`'s
among them, which every solver's `setup` reaches. Converting them would let `setup` be asserted
trim-compatible in the test suites rather than only by `bench/juliac_trim_build.jl`.

Worth doing as a deliberate sweep over one package, not as collateral while chasing a verdict:
the one measurement that settles whether `setup` trims is the real build, which passes.

## Line coverage does not identify a redundant test here

`bench/coverage_per_item.jl` measures what each `@testitem` covers and solves the set cover.
Measured on PureDAQP: 14 of 46 items reach all 736 covered lines in 354 s of 886 s, and 36
items cover no line no other item covers. Taken as a cut list that would be a 60% saving.

Read item by item, it is not one. The items the cover leaves out include the proof that a solve
allocates nothing (44 s), the proof that a warm re-solve allocates nothing (17 s), agreement
with libdaqp on random problems (15 s), and the control that a feasible Kronecker problem is
never reported infeasible (15 s). Nearly every item runs a solve, so the line sets overlap;
what differs is the property each asserts, which line coverage does not see.

The clearest case covers **zero** lines and would be dropped first: the item asserting the
gates refuse code that allocates or cannot be trimmed. Every other proof in that file passes
when nothing throws, so without it a disabled checker reports a fully proved solver.

So the measurement answers the question, in the negative. Where the time goes is not
redundancy: 160 s is the strictmode item's own six gates over three problem configurations, and
the three configurations share most of their assertions — reducing those is the lever, not
dropping items. PureOSQP's 166 items have not been measured and this result is reason to expect
the same shape.

`coverage_per_item.jl` prints `greedy cover: N items` without naming them; the names are only
recoverable from `in_cover` in the JSON. Worth fixing before the harness is used again.

## PureOSQP exports `Optimizer` and the others do not

Each solver defines its own `Optimizer`, one line that builds the base's wrapper type with its
algorithm. PureOSQP also exports the name; PureIPM and PureDAQP do not.

Two modules exporting one name make the unqualified one unusable: `using PureOSQP, PureIPM`
then bare `Optimizer` is an `UndefVarError` naming both. So the set works only because two of
the three abstain, and exporting from exactly one is a property of which name was claimed
first rather than a decision.

MathOptInterface expects the qualified form — a caller writes `PureOSQP.Optimizer` as it writes
`HiGHS.Optimizer` — and nothing in MOI reads exports. Dropping PureOSQP's export makes the
three agree and removes the latent clash. It is a one-line change plus the `names` assertion in
`PureOSQP/test/contract_tests.jl`, which lists `:Optimizer` among the package's own names.

## PureQPBase re-tests StrictMode's own checkers

`PureQPBase/test/strictmode_tests.jl` ends with an item that defines two deliberately broken
functions — `grow(n) = zeros(n)` and `dynamic(r) = r[] + 1` over a `Ref{Any}` — and asserts the
`:noalloc` and `:trim_compatible` gates refuse them. It covers no line of `PureQPBase/src`,
because what it exercises is the checker.

StrictModeTest tests this itself, in seven places: `@test_throws StrictViolation @test_noalloc
Fixtures.allocs(4)`, the same for `@test_noboxing`, `@test_strict`, `test_compiled` and
`test_registered`. So the item duplicates a dependency's own suite, and costs about 37 s of
package loads and one child process to do it.

What is not duplicated is `StrictMode.assert_enabled()`, which asserts a property of this test
environment rather than of StrictMode: the tier depends on whether StrictModeTest is loaded and
on `LocalPreferences.toml`, and a disabled tier prints exactly like a clean one. Every real
proof item already calls it, so removing the broken-function item keeps that guard.

The three solver packages no longer carry this item. PureQPBase still does.

## PureOSQP's suite tests PureIPM

`PureIPM` appears in eight of `PureOSQP/test`'s files — `contract_tests.jl`, `linsys_tests.jl`,
`chainrules_tests.jl`, `derivative_tests.jl`, `setup_tests.jl`, `update_tests.jl`,
`polish_tests.jl` and `moi_tests.jl` — and `PureOSQP/test/Project.toml` depends on it.

Two consequences. PureIPM's behaviour is partly asserted in another package's suite, so a
reader of PureIPM's tests does not see everything that holds it to account; and PureOSQP cannot
be tested without PureIPM installed, which is a test-only dependency between two packages that
have none between their sources.

Some of it is deliberate: a shared contract is more convincingly checked against two
implementations at once, which is why `contract_tests.jl` exercises both workspaces. The rest
reads as PureOSQP having been the original package, with the interior-point comparisons added
where the fixtures already were. Separating those would mean moving the interior-point halves
into `PureIPM/test` and keeping only the genuinely two-implementation assertions behind.

## `ActiveSet` needs `eps_prox` chosen against the problem's scale

The reduction factors `P`, so `P` must be positive definite unless the proximal-point iterations
are on. A linear program presents `P = 0`, which means every LP reaches this method through
`eps_prox > 0`.

How large it has to be is bounded from below by conditioning, not accuracy. With `P = 0` the
factor is `√eps_prox * I` and the reduced rows carry `1 / √eps_prox`, so too small a value makes
`A R⁻¹` ill-conditioned and the outer loop stalls short of the optimum at any iteration count.
Measured against MOI's `test_linear_add_constraints`, whose data reaches `7e4`: `1e-6` never
converges, `1e-4` solves it. `PureDAQP/test/moi_tests.jl` therefore sets `1e-4`.

A caller has no guidance for this, and the failure is silent in the sense that the status is
`ITERATION_LIMIT` rather than a refusal naming the cause. Either `eps_prox`'s default should
scale with the data when `P` is semidefinite, or the refusal that currently names `eps_prox`
should also say what magnitude the problem needs.

## `ActiveSet` reports no dual-infeasibility certificate

An unbounded problem reaches `max_iter` and is reported as `ITERATION_LIMIT`. The status is
honest — the method did not converge — but it is not `DUAL_INFEASIBLE`, which is what a caller
testing for unboundedness looks for, and MOI's `test_linear_DUAL_INFEASIBLE` is excluded for
this reason. The dual active-set iteration has a direction of unbounded descent available to it
when this happens; recognizing it would turn a timeout into an answer.

## Operator splitting has no direct backend for a structured operand

`select_backend` for an `ADMMSelection` ends at `dense_rung`, which declines an operand
reporting `holds_structure`, and then at `indirect_rung`. So a `StackedOperator` `A` reaches
conjugate gradients under `OperatorSplitting` even when every one of its blocks is readable,
while the same problem as a dense matrix reaches `cholesky`. A forced `linsys = :dense` or
`:kkt` does reach a factorization, since a stack is materializable; only the ladder's own choice
differs. The interior-point method has no such gap: its `indirect_rung` returns
`ProductReduced`, a factorization.

A direct backend is the better one for this operand. `bench/admm_structured_operand.jl` measures
a stack of joins at `n = 400`: named `linsys = :dense` takes 50 iterations, lands `1.5e-15` from
the dense answer and spends 2 ms solving against 22.8 ms of setup, where the ladder's conjugate
gradients take 100 iterations, land `8.5e-5` away and spend 16 ms solving against 17.6 ms of
setup. The factorization wins on accuracy, iterations and total time at every size measured.

A rung returning `ProductReduced` would make the ladder choose one. Two things block it, and
both must be answered first:

  - it costs the `--trim` guarantee. Eight of the 43 signatures in
    `PureOSQP/test/trim_tests.jl` stop being trim-compatible, because one more backend reachable
    from `:auto` widens the inferred workspace union past `Base.Compiler.MAX_TYPEUNION_LENGTH`
    (`PureQPBase/src/types.jl` states this constraint);
  - `PureQPBase/test/selection_tests.jl` asserts, by name, that no ladder forms a matrix for a
    pair that holds structure. `ProductReduced` holds the `n²` reduced matrix, so the rung
    contradicts that guarantee rather than extending it.

This is also where a block-pair `add_reduced_term!` for a join would begin to pay
(`docs/design/horizontal-operators.md`).

## A square vcat is refused as a P while an hcat is accepted

`PureDAQP` reads a `JoinedOperator` `P` into a dense matrix and factors it, and refuses a
`StackedOperator` `P` by name even when it is square and every block has entries. Nothing about
the reduction needs the distinction: both are materializable and `factorable_operand` densifies
either.

## Operator splitting does not reach the tolerance at cond(P) = 1e12

On 72 serialized problems from an external solver, `OperatorSplitting` reaches
`MAX_ITER_REACHED`, or `SOLVED` tens of units from the stored answer, in every representation —
dense, sparse, structured and unmaterialized alike. Their `cond(P)` is 1e12 and the default
tolerances and iteration limit do not reach it. The interior-point method and the dual
active-set method with the `:rows` working set solve the same problems in every representation,
so this is the method's conditioning behaviour and not a representation or backend gap.

## The Gram working set fails where the QR one solves

On one of the four families above (`n = 505` and `629`, `m = 1420` and `2396`, `cond(P)` of
7e8), `ActiveSet(working_set = :gram)` ends in `NUMERICAL_ERROR` after 111 or 323 iterations in
every representation — dense, sparse, structured and unmaterialized alike — while
`working_set = :rows` solves the same problems to a lower objective than the stored answer, with
constraint violations below 1e-6. The failure is in the Gram working set's own arithmetic, not
in the operand's representation.
