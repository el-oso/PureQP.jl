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
