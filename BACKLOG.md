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
