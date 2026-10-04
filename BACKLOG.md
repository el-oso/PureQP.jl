# Backlog

Work that is scoped and measured but not yet done. `docs/src/roadmap.md` is a different
document: it records what libosqp does that PureOSQP does not.

## `setup` is not trim-compatible for a Kronecker pair

`PureQPBase.setup` is one of the package's two entry points, and `--trim` rejects it when `P`
and `A` are `KroneckerOperator`s. The dense and diagonal paths pass.

The cause is two independent runtime branches whose product exceeds what inference keeps as a
union. `cholesky_factor(::KroneckerOperator, shift)` returns a `KroneckerCholesky` when
`shift` is zero and both factors are positive definite, and a `KroneckerSquareRoot`
otherwise — data-dependent, so not resolvable at compile time. `build_working_set` returns a
`WorkingSetQR` or a `WorkingSetGram` from the `working_set` option. Two by two is four
concrete results, and inference keeps two but widens four:

| `P` | factor arms | × working set | inferred `reduce_qp` return |
|---|---|---|---|
| `Matrix` | 1, `UpperTriangular` | 2 | a 2-way union of concrete types |
| `KroneckerOperator` | 2 | 2 | `DAQPReduction{Float64}`, four parameters unresolved |

The factor branch has to stay dynamic, so the fix is to resolve the working set at compile
time instead: branch once in `setup_backend` and thread a type rather than a `Symbol` through
`reduce_qp`, `build_reduction` and `build_working_set`. That leaves two arms per
specialization. `setup_backend` already takes the backend name as a `::Val{LS}`, so the shape
is established. Doing it in `ActiveSet`'s type parameters would work too, but that changes a
public type and breaks reading `alg.working_set` as a `Symbol`.

Until then the trim roots are `solve!` and `set_targets!`, the latter standing in for the
setup and rebuild paths that `solve!` does not reach. With `setup` fixed it becomes the second
root and `set_targets!` drops out, since it is internal to `setup`.

## Minimize the test suite against per-item coverage

`bench/coverage_per_item.jl` measures what each `@testitem` covers and solves the set cover.
For PureDAQP, 14 of 46 items reach all 733 covered lines; 36 items contribute no line that no
other item reaches, and 4 of the 14 contribute none individually yet are collectively
required. The selection is recorded in `bench/results/coverage_sets_PureDAQP.tsv` and
`bench/results/coverage_per_item_PureDAQP.json`.

Not yet done: running the 14-item selection to confirm it passes and covers what the analysis
says, then deciding what to cut. Coverage equality is not sufficient grounds on its own — two
items can cover identical lines and still assert different things about them. PureOSQP has 166
items and has not been measured.

## `Optimizer` is exported by one solver and not the other

PureOSQP exports `Optimizer`; PureIPM defines it and does not. Two modules exporting one name
make it unusable unqualified — `using PureOSQP, PureIPM` then `Optimizer` is an
`UndefVarError` naming both — so the pair works only because PureIPM abstains.

One `Optimizer` generic per solver package is right and is what MathOptInterface expects:
a caller writes `PureOSQP.Optimizer`, as it writes `HiGHS.Optimizer`. The base does not own it
and should not, which is the opposite of the `solve!` case. What is left to settle is the
export: qualified access is the convention, so neither package needs to export it, and
exporting from exactly one is a property of which name happened to be claimed first.

## PureDAQP has no MathOptInterface wrapper

PureQPBase, PureOSQP and PureIPM each ship a MathOptInterface extension. PureDAQP ships none
and has no `Optimizer`, so a JuMP model cannot select the dual active-set method.

It is the method suited to the case JuMP users meet in control work: a small dense QP re-solved
many times from a warm start. A wrapper has to carry the refusals across, since `ActiveSet`
rejects rather than ignores what does not apply to it — `scaling` other than zero, `polishing`,
any `linsys`, and the operator-splitting parameters each throw an `ArgumentError`. MOI
attributes that map onto those need the same answer rather than a silent default.
