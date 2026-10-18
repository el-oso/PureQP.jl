# Benchmarks

Each package's benchmarks live in its own `bench/` directory and write their samples to that
package's `bench/results/`, so either set can be run without the other. Every section below
names the script that produced it. The shared problem generators, the snapshot gate and the
StrictMode audit sit in the top-level `bench/`, which also holds the index the last section
renders.

## Against OSQP

Dense random QPs, compared with libosqp 1.0, the C library OSQP itself ships. Reproduce with
`julia --project=bench PureOSQP/bench/headtohead.jl`; samples are in
`PureOSQP/bench/results/headtohead.json`.

Both solvers run with `eps_abs = eps_rel = 1e-6`, single-threaded BLAS, ρ adapted every 50
iterations, and the duality-gap test (`check_dualgap`) turned off. Each solver checks the gap
at a different point in the iteration, so with it on they stop at different iterations and the
times would compare stopping rules instead of solvers. libosqp reads only sparse (CSC) matrices,
so it is timed on its setup and solve calls from CSC arrays built beforehand.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "headtohead")
benchtable(
    d["results"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "PureOSQP" => r -> ms(r["t_openblas"]),
        "libosqp 1.0" => r -> ms(r["t_osqp"]),
        "vs libosqp" => r -> ratio(r["speedup"]),
        "iterations" => r -> int(r["iter_pure"]),
        "objective rel. Δ" => r -> sci(r["objdiff"]),
        "max \\|Δx\\|" => r -> sci(r["dx_osqp"]),
    ]
)
```

Both solvers take the same number of iterations in every case, so the equilibration, the ρ
schedule and the residual tests match libosqp step for step. The solutions agree to about
`1e-14`. The one exception is the `100 × 1000` case, which runs 7475 iterations and agrees to
`4e-9`. PureOSQP is faster in every case.

libosqp has no Julia wrapper, so `PureOSQP/bench/osqp_v1.jl` calls the library in `OSQP_jll` v100
through `ccall`. The header that library ships describes both its single- and double-precision
builds, and some of its type definitions are wrong for the double build. So the wrapper takes the
integer and float sizes from the library itself. `verify_abi()` checks them each time the file is
loaded, and throws if a rebuilt library changes them.

## Sequential re-solves

`P` and `A` stay fixed while `q`, `l` and `u` change every step, as in a receding-horizon
control loop. Each case runs 20 solves. Reproduce with
`julia --project=bench PureOSQP/bench/update_bench.jl`; samples are in
`PureOSQP/bench/results/update_bench.json`.

libosqp runs the same loop through its own update call, `osqp_update_data_vec`, which keeps
the factorization. Both solvers use `eps_abs = eps_rel = 1e-6` with `check_dualgap` off, and
the script checks that their objectives agree to `1e-4` at every step.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "update_bench")
benchtable(
    d["results"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "`update!`" => r -> ms(r["t_update"]),
        "fresh `setup` each step" => r -> ms(r["t_fresh"]),
        "saved" => r -> ratio(num(r["t_fresh"]) / num(r["t_update"])),
        "libosqp 1.0" => r -> ms(r["t_osqp"]),
        "vs libosqp" => r -> ratio(num(r["t_osqp"]) / num(r["t_update"])),
        "factorizations" => r -> int(r["refactorizations"]),
    ]
)
```

"Factorizations" counts PureOSQP's factorizations over all 20 steps: one from `setup`, plus
one for each step where a row changed between equality and inequality.

**PureOSQP's `update!` is several times faster than libosqp's update loop**, and the margin
grows with size, as it does for single solves above.

**What `update!` saves over calling `setup` again varies with the case**, because `setup` is
already fast (see [Sparse A](@ref "Sparse A")). At the two largest sizes it is slightly
slower. The two loops also start from different points (`update!` keeps the previous
solution), so they take different numbers of iterations. `update!` helps most when each solve
is short, so that setup is a large share of it. It is also the way to keep that previous
solution as the starting point.

The factorization count is often more than 1. Equilibration rescales the bounds, and a row
whose scaled gap `ũ - l̃` falls below `RHO_TOL` is treated as an equality, which changes its
`ρ`. libosqp applies the same rule, so an `update!` can still trigger a factorization.

## What a warm start is worth to the active-set method

[`ActiveSet`](@ref) carries its working set between solves, so a re-solve after
[`update!`](@ref) starts from the previous answer's active rows. How much that saves depends on
how far the data moved, which is what a sweep over the size of a perturbation to `q` measures.
Reproduce with `julia --project=bench bench/warm_start.jl`; samples are in
`bench/results/puredaqp_warm_start.json`. Single-threaded BLAS, `n = 250`, `m = 500`, a dense
`P` with eigenvalues spread over `1e0` to `1e-5`, 8 draws per perturbation. Setup is included
in the cold time and excluded from the warm ones, which is the split a caller re-solving a
sequence pays.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "puredaqp_warm_start")
# The sweep also runs any problem read from a directory of CSVs, named by its path. Only the
# generated case is reproducible from a checkout, so only that one is shown.
c = first(x for x in d["cases"] if !occursin("/", string(x["name"])))
cold = num(c["cold_ms"])
# The unchanged re-solve is timed and weighed but its iteration count is not recorded, so that
# cell is a dash.
rows = Any[Dict(
    "label" => "none", "median" => c["warm_same_ms"], "iters" => "—",
    "vs" => cold / num(c["warm_same_ms"]),
)]
for s in c["steps"]
    push!(
        rows, Dict(
            "label" => "`" * sci(s["q_perturbation"]) * "`",
            "median" => s["median_ms"],
            "iters" => int(s["min_iter"]) * "–" * int(s["max_iter"]),
            "vs" => cold / num(s["median_ms"]),
        )
    )
end
benchtable(
    rows, [
        "perturbation to `q`" => r -> r["label"],
        "median" => r -> msval(r["median"]),
        "iterations" => r -> r["iters"],
        "vs cold" => r -> ratio(r["vs"]),
    ]
)
```

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "puredaqp_warm_start")
c = first(x for x in d["cases"] if !occursin("/", string(x["name"])))
Markdown.parse(
    "A cold solve of the same problem takes " * msval(c["cold_ms"]) * " and " *
        int(c["cold_iter"]) * " iterations."
)
```

**The saving is the share of the active set that did not change.** The working set is a good
guess exactly to that extent, so the iteration count grows with the perturbation while the cold
count stays where it is. At the largest perturbation here the warm solve still takes about a
seventh of the cold time, and every one of the 40 draws across the sweep beat the cold solve.

**A warm solve of the unchanged data allocates nothing** — 0 bytes.

The script also measures a second, larger problem read from a directory of CSVs when one is
present, and skips it without complaint otherwise, so it runs on a fresh checkout.

## Linear-system backend

`PureOSQP/bench/kkt_backend.jl` reproduces the cost and accuracy comparison between the reduced
Cholesky and the full-KKT Bunch-Kaufman factorization described under
[Algorithm](@ref "The linear system"), including the near-parallel-row family that
equilibration cannot fix.

## On PureBLAS instead of OpenBLAS

[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl) is a pure-Julia BLAS/LAPACK. Its
`activate()` overlays per-symbol forwards onto libblastrampoline, so PureOSQP runs on it
**with no code changes** — the same `mul!`, `cholesky!` and `symv` calls are rerouted in
process. Reproduce with `PureOSQP/bench/pureblas_backend.jl` (setup instructions in its header).

Note `BLAS.get_config()` cannot show this: OpenBLAS stays loaded and the forwards sit on
top of it. `PureBLAS.is_active()` is the check, and the benchmark asserts it at every
measurement — otherwise a rerouting that silently failed would look like a clean tie.

Measured against PureBLAS at commit `ea79919`, which `PureOSQP/bench/results/pureblas_backend.json`
records alongside the timings.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "pureblas_backend")
benchtable(
    d["solve"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "OpenBLAS" => r -> ms(r["t_openblas"]),
        "PureBLAS" => r -> ms(r["t_pureblas"]),
        "ratio" => r -> ratio(r["ratio"]),
        "iterations" => r -> int(r["iter_pureblas"]),
        "\\|Δx\\|" => r -> sci(r["dx"]),
    ]
)
```

**Correctness is exact** — identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in — **and it is faster than OpenBLAS on every case**:

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "pureblas_backend")
# How often PureOSQP's inner loop reaches each operation. A dash marks one measured for
# reference that the loop does not call.
runs = Dict(
    "gemv  A*x" => "every iteration",
    "gemv  A'y (trans)" => "every iteration",
    "gemv  At*y (dense)" => "—",
    "syrk  W'W" => "on a ρ update",
    "potrf" => "on a ρ update",
    "potri" => "on a ρ update",
    "symv  R⁻¹b" => "every iteration",
    "trsv  F\\b" => "—",
)
benchtable(
    d["ops"], [
        "operation (n=200, m=400)" => r -> "`" * replace(strip(r["op"]), "  " => " ") * "`",
        "OpenBLAS" => r -> us(r["t_openblas"]),
        "PureBLAS" => r -> us(r["t_pureblas"]),
        "ratio" => r -> ratio(num(r["t_openblas"]) / num(r["t_pureblas"])),
        "runs" => r -> get(runs, r["op"], "—"),
    ]
)
```

PureOSQP's inner loop is Level-2-bound: it does a handful of `gemv` and one `symv` per
iteration and touches `syrk`/`potrf`/`potri` only when ρ changes, so the whole-solve ratio
tracks `gemv`. `potri` is the one operation where PureBLAS is behind, and inverting the
factor put it on the ρ-update path — but at one call per update against hundreds of
iterations, it does not move the whole-solve number.

`trsv` is listed for reference only: it is what `symv` replaced. The two are the same
`2n²` flops on either BLAS, and the 7× between them is the sequential dependency, not the
library.

### On the threading

The two sides are not symmetric. OpenBLAS is pinned with `BLAS.set_num_threads(1)`;
PureBLAS is plain Julia, so it is bounded by `Threads.nthreads()` — 12 in these runs —
which `BLAS.set_num_threads` does not affect. Giving OpenBLAS more threads does not change
the picture at these sizes: at n=200, m=400 it took 25.4 ms on 1 thread, 25.8 ms on 4 and
26.3 ms on 8.



## Against other solvers

The same dense QPs solved by all three algorithms here and by the outside implementation of
each: libosqp 1.0 for operator splitting, DAQP for the active-set method, Clarabel for the
interior-point one. Reproduce with `julia --project=bench PureOSQP/bench/solvers.jl`; samples
are in `PureOSQP/bench/results/solvers.json`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "solvers")
cols = Any["n" => r -> int(r["n"]), "m" => r -> int(r["m"])]
for (key, header) in [
        "PureOSQP" => "PureOSQP", "OSQP" => "libosqp 1.0", "PureDAQP" => "PureDAQP",
        "DAQP" => "DAQP", "PureIPM" => "PureIPM", "Clarabel" => "Clarabel",
    ]
    push!(cols, header => r -> ms(get(r["times"], key, nothing)))
end
benchtable(d["results"], cols)
```

libosqp and Clarabel read sparse matrices, so each is timed from sparse copies built
beforehand. Both DAQP implementations read dense ones, which is what they are built for.

**An active-set method wins this shape.** These are dense problems with few rows active at
the solution, which is what an active-set method is for: a few expensive steps, then it stops
at the exact vertex. PureDAQP is the fastest solver here from `n = 50` up, and over those
sizes it beats the C implementation it follows by 1.2× to 2.3×. PureIPM beats Clarabel at
every size, by 1.8× to 5.6×.

Below `n = 50` it does not, and the reason is per-iteration work rather than a fixed cost:
setup is about a fifth of a solve at `n = 25`, and the remainder divides into iterations that
each cost more than the C implementation's. The next table gives the whole picture.

These are the figures for the default `working_set = :rows`, which trades 1.2×–1.5× for a
rank test that reads the conditioning once rather than twice; `:gram` is the faster setting,
and
[choosing a working set for `ActiveSet`](@ref "Choosing a working set for `ActiveSet`") says
when each is right.

For repeated small solves, [`setup`](@ref) with [`update!`](@ref) and [`solve!`](@ref) pays
the fixed cost once and warm starts from the previous working set, which is a different
question from the one this table asks.

ADMM is the better fit when you re-solve a problem many times: warm starts, updates that keep
the factorization, and one factorization reused across hundreds of iterations. It is also the
only one of the three that takes an operator it can only multiply by.

The six solutions agree to about `1e-4`, which is expected at `eps_abs = eps_rel = 1e-6`.
The active-set and interior-point solvers stop at exact optimality conditions; ADMM stops when
its residuals fall below the tolerance.

## Against the C implementation, over both dimensions

The table above runs along `m = 2n`, which is one line through a plane. This one is the
plane: `libdaqp / PureDAQP`, so above `1.00` PureDAQP is faster. Reproduce with
`julia --project=bench bench/daqp_headtohead.jl`; samples are in
`bench/results/puredaqp_vs_libdaqp.json`. Single-threaded BLAS, `working_set = :rows`, both
solvers given the same iteration limit.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "puredaqp_vs_libdaqp")
ns = sort(unique(num(r["n"]) for r in d["rows"]))
aspects = sort(unique(num(r["m_over_n"]) for r in d["rows"]))
by_shape = Dict((num(r["n"]), num(r["m_over_n"])) => r for r in d["rows"])
rows = [Dict("n" => n, "cells" => [get(by_shape, (n, a), nothing) for a in aspects]) for n in ns]
cols = Any["n \\ m/n" => r -> int(r["n"])]
for (j, a) in enumerate(aspects)
    push!(
        cols, (a == round(a) ? int(a) : string(a)) =>
            r -> (c = r["cells"][j]; isnothing(c) ? "—" : ratio(c["speedup"]))
    )
end
benchtable(rows, cols)
```

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "puredaqp_vs_libdaqp")
errs = [num(r["rel_err_vs_libdaqp"]) for r in d["rows"]]
Markdown.parse(
    "The two agree to between `" * sci(minimum(errs)) * "` and `" * sci(maximum(errs)) *
        "` in every cell, and the script checks that before recording a time."
)
```

**`n` decides this, not the aspect ratio.** Reading down a column is a clean progression;
reading across a row is not. The crossover sits near `n = 50` whichever shape the problem
has.

**The best case is tall and thin.** `m / n = 0.5` is the strongest column, because few rows
means a small working set and the per-iteration cost follows `O(nk)`. A sweep along `m = 2n`
runs through one of the weaker columns, which is why the table above reports a narrower
range than this one.

**Below `n = 50`, `:gram` recovers it.** The deficit is the extra arithmetic the `:rows`
representation spends per iteration, not a fixed cost, so changing the representation removes
it:

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "puredaqp_vs_libdaqp")
benchtable(
    d["working_sets"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "`:rows`" => r -> msval(r["rows_ms"]),
        "`:gram`" => r -> msval(r["gram_ms"]),
        "libdaqp" => r -> msval(r["libdaqp_ms"]),
        "`:rows` / C" => r -> ratio(num(r["libdaqp_ms"]) / num(r["rows_ms"])),
        "`:gram` / C" => r -> ratio(num(r["libdaqp_ms"]) / num(r["gram_ms"])),
    ]
)
```

## Choosing a working set for the active-set method

[`ActiveSet`](@ref) keeps its working set in one of two forms, chosen with `working_set`.
`:rows` factors the active rows themselves; `:gram` factors their Gram matrix `Mₐ Mₐᵀ`. These
are the measurements; [choosing a working set for `ActiveSet`](@ref "Choosing a working set
for `ActiveSet`") says how to decide. Reproduce with
`julia --project=bench bench/working_set_choice.jl`; samples are in
`bench/results/working_set_choice.json`. Single-threaded BLAS.

**Well conditioned: `:gram` is faster.**

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "working_set_choice")
benchtable(
    [r for r in d["rows"] if r["case"] == "benign"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "`:gram`" => r -> msval(r["gram"]["ms"]),
        "`:rows`" => r -> msval(r["rows"]["ms"]),
        "iterations" => r -> int(r["gram"]["iter"]) * " / " * int(r["rows"]["iter"]),
        "`:rows` cost" => r -> ratio(num(r["rows"]["ms"]) / num(r["gram"]["ms"])),
    ]
)
```

The iteration counts are equal row for row here, so on these problems the whole difference is
what one iteration costs. The counts are not guaranteed to match — the pivots differ in their
last digits, so a row priced at the tolerance can enter one and not the other — but the
answers agree.

Where that cost sits, measured at `k = 100`, `n = 200` against a whole iteration of 14.8 µs:

| | `:rows` |
|---|---|
| `add_row!` | 4.5 µs |
| `remove_row!`, early in the set | 8.1 µs |
| `remove_row!`, middle | 4.3 µs |
| `remove_row!`, last | 0.06 µs |

A deletion is repaired by rotations over the columns after it, so dropping the newest row is
free and dropping the oldest costs the whole factor.

**What does not separate them on these problems: conditioning, or rows that are exact
combinations of others.**

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "working_set_choice")
benchtable(
    [r for r in d["rows"] if r["case"] != "benign"], [
        "case" => r -> str(r["case"]),
        "`:gram`" => r -> str(r["gram"]["status"]) * ", violation " * sci(r["gram"]["violation"]),
        "`:rows`" => r -> str(r["rows"]["status"]) * ", violation " * sci(r["rows"]["violation"]),
        "iterations" => r -> int(r["gram"]["iter"]) * " / " * int(r["rows"]["iter"]),
    ]
)
```

Both settings solve both cases. Dependence that is exact is not the same difficulty as
dependence blurred by rounding, and both forms carry the dependent row and walk the direction
it opens. The condition number at which the Gram pivot stops being able to tell a dependent
row from an independent one is a property of the arithmetic rather than of these two problems;
[choosing a working set for `ActiveSet`](@ref "Choosing a working set for `ActiveSet`") derives
it and says what to do about it.

## Choosing a representation

A matrix can be passed dense, sparse, as a structured type, or as an operator that is never
formed. These three cases measure when each is the better choice. Reproduce with
`julia --project=bench PureOSQP/bench/representation_choice.jl`; samples are in
`PureOSQP/bench/results/representation_choice.json`. Single-threaded BLAS, `eps_abs = eps_rel = 1e-6`.
The script checks that every run reaches `SOLVED` before it reports a time.

**An operator that is cheap to apply, in a problem where that is not the main cost.** `P` is a
diagonal plus a rank-one term, stored as `O(n)` numbers, compared with the `n×n` matrix it
represents. `A` is dense in both, so the products with `A` cost `O(n²)` either way.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "representation_choice")
benchtable(
    [r for r in d["cases"] if r["case"] == "no_advantage"], [
        "n" => r -> int(r["n"]),
        "iterations (operator / dense)" => r -> int(r["operator_iter"]) * " / " * int(r["dense_iter"]),
        "operator" => r -> ms(r["operator_seconds"]),
        "dense" => r -> ms(r["dense_seconds"]),
        "speedup" => r -> ratio(r["speedup"]),
    ]
)
```

The operator skips the factorization but solves a linear system with CG in every iteration. At
`n = 200` that costs more than it saves. From `n = 500` the factorization's `O(n³)` cost grows
faster than the CG work, and the operator is faster.

**An operator applied in `O(n)` whose dense form costs `O(n²)`.** A moving average with a window
of `n/20`, so about a tenth of the entries are nonzero, which is too many for a sparse format to
be the obvious choice. It is diagonally dominant, so conditioning plays no part here.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "representation_choice")
benchtable(
    [r for r in d["cases"] if r["case"] == "asymptotic_win"], [
        "n" => r -> int(r["n"]),
        "window" => r -> int(r["bandwidth"]),
        "fill" => r -> sigfig(num(r["fill_percent"]), 2) * "%",
        "iterations (operator / dense)" => r -> int(r["operator_iter"]) * " / " * int(r["dense_iter"]),
        "operator" => r -> ms(r["operator_seconds"]),
        "dense" => r -> ms(r["dense_seconds"]),
        "speedup" => r -> "**" * ratio(r["speedup"]) * "**",
        "dense `A`" => r -> sigfig(num(r["dense_mib"])) * " MiB",
    ]
)
```

Compared with the first case, what changed is neither size nor sparsity. It is that applying the
operator costs less than multiplying by its dense form. Once the dense matrix no longer fits in
cache, its product is limited by memory speed, so the operator's lead holds as `n` grows.

**An ill-conditioned problem with a structured backend.** `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so a
Kronecker operator with `κ = 1e12` is two factors with `κ = 1e6` each. Its backend
eigendecomposes the two factors and never forms the product.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "representation_choice")
benchtable(
    [r for r in d["cases"] if r["case"] == "direct_not_cg"], [
        "n" => r -> int(r["n"]),
        "κ(A)" => r -> sci(r["cond_A"]),
        "iterations" => r -> int(r["kronecker_iter"]) * " / " * int(r["dense_iter"]),
        "`kronecker`" => r -> ms(r["kronecker_seconds"]),
        "dense" => r -> ms(r["dense_seconds"]),
        "speedup" => r -> "**" * ratio(r["speedup"]) * "**",
        "CG on the same problem" => r -> "`" * str(r["cg_status"]) * "` in " * int(r["cg_iter"]) * " iterations",
    ]
)
```

The Kronecker backend and the dense path reach the same objective, and the script checks this
before reporting a time. The matrix-free backend also solves both problems and reaches the same
objective, so an operator without a structured backend still works here. The Kronecker backend
is still the fast choice: its cost depends on the structure, not on the conditioning.

## Matrix types

PureOSQP keeps the caller's `P` and `A` and applies equilibration as it goes, so every product
in the iteration calls `mul!` on the matrix that was passed in. Each row below is one problem
solved twice, once as a plain `Matrix` and once in a structured type, so both take the same
number of iterations and the difference is the cost of the products. `n = 150`, `m = 300`.
Reproduce with `julia --project=bench PureOSQP/bench/matrix_types.jl`; samples are in
`PureOSQP/bench/results/matrix_types.json`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "matrix_types")
# The type each problem is compared against is the script's choice of structured form, which the
# samples record by problem rather than by type name.
against = Dict(
    "dense P" => "dense `P` (vs `Symmetric`)",
    "diagonal P" => "diagonal `P` (vs `Diagonal`)",
    "tridiagonal P" => "tridiagonal `P` (vs `SymTridiagonal`)",
    "A as a view" => "`A` as a `SubArray`",
)
benchtable(
    d["structured"], [
        "problem" => r -> against[r["problem"]],
        "as `Matrix`" => r -> ms(r["t_matrix"]),
        "structured" => r -> ms(r["t_structured"]),
        "speedup" => r -> ratio(r["ratio"]),
        "iterations" => r -> int(r["iters"]),
    ]
)
```

**A structured `P` barely moves the solve here.** Equilibration reads only the entries a band
type can hold, but equilibration is a small part of a solve. Every iteration also multiplies by
`A` twice, and with `m = 2n` and `A` dense in every row above, that is four times the work of
the `P` product.

What the table does show is that any `AbstractMatrix` works, the caller's matrices are never
copied or changed, and a type with a cheaper `mul!` gets it. With a dense `A`, the structured
type is close to a wash. [Structured backends](@ref) covers the cases where the structure
changes which linear system is solved, which is where the large gains are.

## Structured backends

A structured `A` is where it becomes one, and by a great deal, because then the *reduced
matrix* is structured rather than merely the input. Eliminating `ν` gives
`R = c D P D + σI + Ãᵀ diag(ρ) Ã`; diagonal scaling preserves a bandwidth and `ÃᵀρÃ` doubles
`A`'s, so

```math
\mathrm{bandwidth}(R) = \max\bigl(\mathrm{bandwidth}(P),\; 2\,\mathrm{bandwidth}(A)\bigr)
```

Each row below is one problem solved twice, once in the structured types and once as a
`Matrix`, so the iteration counts match exactly and the difference is the backend.
Reproduce with `julia --project=bench PureOSQP/bench/structured_backends.jl`; samples are written to
`PureOSQP/bench/results/structured_backends.json`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "structured_backends")
benchtable(
    [r for r in d["cases"] if num(r["n"]) in (400, 2000)], [
        "`P`, `A`" => r -> "`" * replace(str(r["name"]), ", " => "`, `") * "`",
        "n" => r -> int(r["n"]),
        "bw(`R`)" => r -> int(r["bw"]),
        "backend" => r -> "`" * str(r["backend"]) * "`",
        "setup" => r -> us(r["setup"]),
        "dense setup" => r -> ms(r["dense_setup"]),
        "setup×" => r -> "**" * ratio(num(r["dense_setup"]) / num(r["setup"])) * "**",
        "total" => r -> ms(r["total"]),
        "dense total" => r -> ms(r["dense_total"]),
        "total×" => r -> "**" * ratio(num(r["dense_total"]) / num(r["total"])) * "**",
    ]
)
```

The ratios grow with `n` because the two sides have different exponents, not because the
constant is better: the dense path factors in `O(n³)` and applies in `O(n²)`, and these are
`O(n b²)` and `O(n b)`. At bandwidth 0 there is no factorization at all — a solve is `n`
divisions.

**What is selected, and by what.** `Diagonal` with `Diagonal` gives a diagonal `R`;
`SymTridiagonal` or `Tridiagonal` with `Diagonal`, and any of the three with a `Bidiagonal`
`A`, give bandwidth 1 and an `ldlt`. Both are core LinearAlgebra. Bandwidth 2 and up needs
BandedMatrices.jl loaded, since LinearAlgebra stores no symmetric banded type past
`SymTridiagonal`; without it those problems take the dense path, correctly but densely.
Selection is dispatch on the pair of types — no setting, no density gate — and the banded
backend is not used in either direction: below bandwidth 2 the LinearAlgebra backends are
cheaper, and above a quarter of the matrix the dense path wins per iteration.

**Keyed on `A`, not `P`.** `ÃᵀρÃ` is dense for a general `A` whatever `P` looked like, so a
`Diagonal` `P` with a dense `A` has a dense reduced matrix and correctly gets the dense
backend. Widening `A` costs twice what widening `P` does.

**The structure has to be established, not inherited.** Julia's arithmetic does not carry it
through: `D P D` on a `SymTridiagonal` returns a `Tridiagonal`, which `cholesky` rejects as
not Hermitian though it is symmetric to `1e-17`, and a `Diagonal` `P` with a `Bidiagonal` `A`
returns a dense `Array` despite having bandwidth 1. The bands are computed entry by entry for
that reason. An `ldlt` is likewise no substitute for a Cholesky's failure reporting — it
returns a negative pivot for an indefinite matrix and throws only on an exact zero — so the
tridiagonal backend tests the pivots itself, where the banded one can rely on `issuccess`.

## Block-diagonal structure

A [`PureQPBase.BlockDiagonal`](@ref) `P` and `A` split the reduced matrix into `K` independent
blocks, which are factored one at a time and never assembled into one matrix. The factorization
cost falls from `n³` to `Σnᵢ³` and the storage from `n²` to `Σnᵢ²`, so both improve as the same
`n` is split into more blocks. Each row is one problem solved as a `BlockDiagonal`, as a
`Matrix`, and with the matrix-free backend, so the direct paths take the same number of
iterations. Reproduce with `julia --project=bench PureOSQP/bench/block_backend.jl`; samples are in
`PureOSQP/bench/results/block_backend.json`. Single-threaded BLAS, `n = 240`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "block_backend")
rows = d["cases"]
best_speedup = maximum(num(r["dense"]) / num(r["block"]) for r in rows)
best_memory = maximum(num(r["dense_words"]) / num(r["words"]) for r in rows)
emph(v, best) = (s = ratio(v); v ≈ best ? "**" * s * "**" : s)
benchtable(
    rows, [
        "`K`" => r -> int(r["K"]),
        "block size" => r -> int(r["block_size"]),
        "iterations" => r -> int(r["iter"]),
        "`block`" => r -> ms(r["block"]),
        "dense" => r -> ms(r["dense"]),
        "speedup" => r -> emph(num(r["dense"]) / num(r["block"]), best_speedup),
        "matrix-free" => r -> ms(r["indirect"]),
        "block vs matrix-free" => r -> ratio(num(r["indirect"]) / num(r["block"])),
        "factor words" => r -> int(r["words"]),
        "dense words" => r -> int(r["dense_words"]),
        "memory saved" => r -> emph(num(r["dense_words"]) / num(r["words"]), best_memory),
    ]
)
```

**The block backend is the fastest of the three at every split.** Its lead over the dense path
grows from 1.18× to 2.78× as the blocks get smaller. At `n = 240` a dense factorization is
already cheap and each iteration's `symv` takes most of the time, so the time gain is smaller
than the memory gain. The memory saved matches `n²/Σnᵢ²` and does not depend on the problem
being small. The iteration count changes down the table because each `K` is a different
problem, so compare across a row, not down a column.

The matrix-free backend is slower than the block backend at every split, and slower than the
dense path at all but `K = 12`. The blocks are dense, and CG does not use the block structure.

## Low-rank structure

A `Diagonal` `P` with a [`PureQPBase.RowCoupled`](@ref) `A` makes the reduced matrix a diagonal
plus a rank-`k` correction, which Woodbury solves without forming it: two `gemv`s against a
`k×n` block and one `k×k` solve, in `O(nk)` time and storage rather than `O(n²)`. Reproduce
with `julia --project=bench PureOSQP/bench/lowrank_backend.jl`; samples in
`PureOSQP/bench/results/lowrank_backend.json`, single-threaded BLAS.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "lowrank_backend")
benchtable(
    d["cases"], [
        "n" => r -> int(r["n"]),
        "k" => r -> int(r["k"]),
        "iters" => r -> int(r["iter"]),
        "setup" => r -> us(r["setup"]),
        "dense setup" => r -> ms(r["dense_setup"]),
        "total" => r -> ms(r["total"]),
        "dense total" => r -> ms(r["dense_total"]),
        "total×" => r -> "**" * ratio(num(r["dense_total"]) / num(r["total"])) * "**",
    ]
)
```

The ratio grows with `n` at fixed `k`, and broadly shrinks as `k` climbs, which is what `O(nk)`
against `O(n²)` predicts. It is not monotonic in `k`: the rows do not share an iteration count,
so a row that took twice the iterations carries that into both of its times. The solver drops
this backend once `10k > n`, which sits below the measured
crossing, so the limit holds at any BLAS thread count — see the gate discussion in
`PureOSQP/bench/results/gate_crossover_lowrank.json`. Each row is a different problem, so read across
a row rather than down a column.

## Sparse A

Random sparse `P` and `A`, the kind of problem libosqp's sparse factorization is designed for.
PureOSQP is measured twice: given dense copies, and given the sparse matrices, where the
SparseArrays extension reads only the stored entries. libosqp gets the sparse matrices. The
results come from the same script as the table above.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "matrix_types")
benchtable(
    d["sparse"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "density" => r -> sigfig(num(r["density"]) * 100, 2) * "%",
        "PureOSQP, dense copies" => r -> ms(r["t_densified"]),
        "PureOSQP, sparse" => r -> ms(r["t_sparse"]),
        "libosqp 1.0" => r -> ms(r["t_osqp"]),
        "best vs libosqp" => r -> ratio(r["ratio"]),
    ]
)
```

Dense and sparse storage take the same number of iterations in every row; the script checks
this.

**PureOSQP is faster in every row.** The smallest margin, 2.15×, is on the smallest and
sparsest problem, where libosqp's factorization has the least to do.

**Which storage to pass depends on density.** At 1% and 5%, pass the sparse matrices: the
solve is 1.9× to 3.8× faster than with dense copies. At 20%, pass dense copies: sparse `mul!`
is slower than dense BLAS once each row holds enough entries, and the sparse path takes about
1.25× longer. The solver does not choose for you, because it would have to measure the density
first.


## Sparse against libosqp, both sides sparse

Both solvers get `SparseMatrixCSC` and neither makes a dense copy. `eps_abs = eps_rel = 1e-6`,
with `check_dualgap` off on both. Reproduce with
`julia --project=bench PureOSQP/bench/sparse_headtohead.jl`; samples are in
`PureOSQP/bench/results/sparse_headtohead.json`. Both solvers take the same number of iterations in
every row, and the script stops if their objectives differ by more than `1e-6`.

The two problem families below favor different approaches, so both are shown.

**Banded**, as in a model-predictive control horizon. The reduced matrix keeps the band, which
PureOSQP factors sparsely; the backend column names which one it chose at each size. This is
the kind of problem libosqp's sparse factorization of the full KKT system is designed for.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "sparse_headtohead")
sparse_rows(family) = benchtable(
    d[family], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "nnz(A)" => r -> int(r["nnz_A"]),
        "PureOSQP backend" => r -> "`" * str(r["backend"]) * "`",
        "PureOSQP" => r -> ms(r["t_pure"]),
        "libosqp 1.0" => r -> ms(r["t_osqp"]),
        "vs libosqp" => r -> ratio(r["ratio"]),
        "iterations" => r -> int(r["iter_pure"]),
    ]
)
sparse_rows("banded")
```

**Uniformly random sparsity**, with no structure a sparse factorization can use. The reduced
matrix fills in, so PureOSQP builds it from the stored entries and factors it as a dense
matrix.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "sparse_headtohead")
benchtable(
    d["random"], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "nnz(A)" => r -> int(r["nnz_A"]),
        "PureOSQP backend" => r -> "`" * str(r["backend"]) * "`",
        "PureOSQP" => r -> ms(r["t_pure"]),
        "libosqp 1.0" => r -> ms(r["t_osqp"]),
        "vs libosqp" => r -> ratio(r["ratio"]),
        "iterations" => r -> int(r["iter_pure"]),
    ]
)
```

The margin is smaller on the banded family (1.5× to 1.6×) than on most of the random one (up
to 4.8×). A banded problem suits libosqp's sparse factorization well. A random sparse problem
fills in, and there building the smaller reduced matrix and factoring it densely beats
factoring the larger KKT matrix sparsely.

For banded problems, how convexity is checked matters. Checking a tridiagonal `P` by copying
it into a dense `n×n` array and running a dense Cholesky costs `O(n³)` for a matrix with `O(n)`
entries; at `n = 2000, m = 4000` that check alone takes as long as the rest of the solve. The
SparseArrays extension checks it through CHOLMOD instead, which is 93× faster on that matrix.


## The OSQP benchmark suite

The seven problem classes from OSQP's own benchmark suite, built from its problem definitions
and compared with libosqp 1.0. Reproduce the totals with
`julia --project=bench PureOSQP/bench/osqp_suite.jl` and the setup and loop split with
`julia --project=bench PureOSQP/bench/suite_split.jl`; samples are in `PureOSQP/bench/results/osqp_suite.json`.

Both solvers get `SparseMatrixCSC` and run with `eps_abs = eps_rel = 1e-5` and
`check_dualgap` off. They stop at the same iteration in every class, so the times compare the
cost of the work, not the number of iterations. libosqp is timed on `osqp_setup` and
`osqp_solve` from CSC arrays built beforehand. Measured on Julia 1.13.0, single-threaded BLAS,
pinned to core 15, on a workstation with an unpinned clock.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "osqp_suite")
benchtable(
    sort(collect(d["cases"]); by = r -> -num(r["ratio"])), [
        "class" => r -> str(r["name"]),
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "PureOSQP backend" => r -> "`" * str(r["backend"]) * "`",
        "PureOSQP" => r -> ms(r["t_pure"]),
        "libosqp 1.0" => r -> ms(r["t_osqp"]),
        "vs libosqp" => r -> "**" * ratio(r["ratio"]) * "**",
        "iterations" => r -> int(r["iter_pure"]),
    ]
)
```
`vs libosqp` is libosqp's time divided by PureOSQP's. The two take the same number of
iterations in every class, so it is also the ratio per iteration. The objectives agree to
`1e-13` or better in six classes and to `1e-9` in the seventh. Random QP and Eq QP read
`sparse_formed`: their reduced matrix is accumulated from the stored entries of `P` and `A`
rather than formed with a dense product, the same arithmetic without the `m×n` buffer.

**PureOSQP is faster in all seven classes, by 1.05× to 1.95×.** These problems have the block
and band structure real problems tend to have. The random sparse families elsewhere on this
page have none, which is the hardest case for any sparse factorization.

The split between setup and iteration is where the classes differ, and the saved samples record
only the whole solve, so the paragraphs below name the setup costs without a table behind them.

Every figure is a median over ten seconds of samples. The two solvers run in the order
PureOSQP, libosqp, libosqp, PureOSQP, and the median is taken over both turns together, so a
slow drift during a row affects both solvers equally.

**Control's setup is the slower half on purpose.** PureOSQP forms the reduced matrix `R` and
inverts it, which costs `O(n³)` once and makes each iteration a single `symv`. The faster
iterations more than pay for it over the hundreds the class takes. A sparse `LDLᵀ` of the
reduced matrix or of the KKT system matches libosqp's setup but makes the iterations slower, so
the dense path is kept.

Most of that setup is the inverse, not forming `R`: `potrf` costs `n³/3` and the `potri` after
it another `2n³/3`. Keeping the factor and applying it with two `trsv` calls costs as much but
cannot run in parallel. Inverting only the triangle with `trtri` and applying two `trmv` calls
halves the setup but doubles the cost of each iteration. Over 325 iterations neither is better.

**Small setup ratios are noisy.** Control's setup is one large dense factorization and times
reliably. Setups made of many small steps and allocations do not, because the ratio divides two
numbers within a few percent of each other. Read Portfolio's, SVM's, Lasso's and Huber's setup
ratios as roughly equal. Other programs running on the machine move these ratios more than
anything else does, so run the benchmarks on an otherwise idle machine.

**Portfolio is what a dense row costs, read without forming anything.** Its `A` is 0.9%
dense, but one row — the budget constraint `1ᵀx = 1` — touches 99% of the columns.
`sparse_form` reads that row directly: once the densest row spans half of `n`,
`Ãᵀ diag(ρ) Ã` is dense however sparse the rest of the pattern is, so the rule sends the pair
straight past the reduced form. Nothing is accumulated or factored to reach that answer — the
reduced matrix's own fill (`nnz(R)/n²` in the table below, which would be 99% here) is never
computed for a pattern the densest-row test has already ruled out. The `ldl_kkt` backend then
factors the full quasi-definite system instead, which keeps that row as one sparse row: its
factor holds 2322 nonzeros against the KKT's 3305, so the elimination fills in nothing.
Forming the reduced matrix here would mean a dense `505×505` factorization in place of a
sparse `1011×1011` one.

| class | nnz(A)/mn | nnz(R)/n² | densest row of A | backend |
|---|---|---|---|---|
| Portfolio | 0.009 | **0.990** | **0.990** | `ldl_kkt` |
| Lasso | 0.003 | 0.004 | 0.007 | `ldlfactorizations` |
| Huber | 0.001 | 0.003 | 0.004 | `ldlfactorizations` |
| Control | 0.038 | 0.209 | 0.097 | `sparse_formed` |

`nnz(R)/n²` is shown for context — what the reduced matrix's fill turns out to be — not
because the rule computes it for every pattern. `sparse_form` reads only the densest row of
`A`, the number of stored entries in the KKT matrix, and the symbolic `AᵀA ∪ P` pattern count
that `reduced_nnz` returns; it stops counting once a pattern would fail its budget anyway, so
even that count costs a fraction of a full pass on the patterns most expensive to check.
Portfolio's densest row alone settles the question. Lasso, Huber and Control have no row that
wide, so for them the rule goes on to compare the symbolic reduced-pattern count against a
budget of 5% of `n²`. Lasso and Huber stay under it and reach the sparse `:reduced` form,
factored here by `ldlfactorizations`. Control's reduced pattern fills 21% of `n²`, well past
that budget, so the rule does not use the sparse reduced form, and the pair falls through to
the `sparse_formed` terminal, which accumulates the same matrix from the stored entries and
factors it densely.

**Eq QP pays for its storage.** Its `P` is 99% dense but is passed as a `SparseMatrixCSC`, so
equilibration reads it entry by entry through the sparse structure. Passing the same matrix as a
`Matrix` makes that step about 10× faster, as the advice under [Sparse A](@ref "Sparse A")
predicts. Each equilibration sweep reads `P` once, not twice: the cost normalization at the end
of a sweep computes the column norms the next sweep starts from, so `cost_norms!` returns both.

### The SparseArrays extension

PureOSQP's per-iteration products go through `mul!`, which sparse matrices already handle
well. Equilibration and the factorization were the problem: they walk the caller's matrices
entry by entry, and the generic loop visits every structural zero and reaches each through
`M[i, j]`, which on CSC is a binary search. Equilibration was **10.7× slower on a sparse
matrix than on a dense one** — the opposite of what the storage should give.

The four column traversals are now overridable, and `PureQPBase/ext/PureQPBaseSparseArraysExt.jl`
specialises them for `SparseMatrixCSC`. `SparseArrays` is a weak dependency, so the core
still has none beyond `LinearAlgebra` and TypeContracts, and the extension can only load
when the caller already has sparse matrices to pass.

| at 1% density, n=200, m=400 | dense input | sparse input |
|---|---|---|
| `scale!` | 548 µs | **67.6 µs** |
| `factorize!` | 767 µs | 626 µs |
| `setup` | 1475 µs | **844 µs** |
| full solve | 8.16 ms | **3.54 ms** |

A test asserts the two storages produce **identical** `D`, `E` and `c`, so the extension
cannot drift into being a behaviour change.

## The interior-point method against Clarabel

[`InteriorPoint`](@ref) and [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl) 0.11.1
are both interior-point methods; `PureIPM/bench/ipm_vs_clarabel.jl` runs them on the smallest instance
of each OSQP suite problem class, alongside [`OperatorSplitting`](@ref) on the same instance.
`InteriorPoint` and Clarabel both run at `eps_abs = eps_rel = 1e-8`; `OperatorSplitting` runs
at `1e-6`, the tightest tolerance ADMM reaches in a modest iteration count on these problems.
These are not committed benchmarks to reproduce and compare against: the figures below are
`PureIPM/bench/results/ipm_vs_clarabel.json`, measured once at commit `c7d2c92` on Julia 1.13.0, pinned
to core 15 with BLAS at one thread, on a workstation with an unpinned clock, so the times are
indicative rather than a claim about relative speed.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureIPM", "ipm_vs_clarabel")
# The samples record each solver's time in milliseconds; this table reads in microseconds.
usms(x) = (t = num(x); isfinite(t) ? sigfig(t * 1000) * " µs" : "—")
benchtable(
    d["results"], [
        "class" => r -> str(r["name"]),
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "ADMM iter" => r -> int(r["admm"]["iter"]),
        "ADMM time" => r -> usms(r["admm"]["time_ms"]),
        "IPM iter" => r -> int(r["ipm"]["iter"]),
        "IPM time" => r -> usms(r["ipm"]["time_ms"]),
        "Clarabel iter" => r -> int(r["clarabel"]["iter"]),
        "Clarabel time" => r -> usms(r["clarabel"]["time_ms"]),
        "`x`, IPM vs Clarabel" => r -> sci(r["dx_ipm_clarabel"]),
    ]
)
```
The last column is the largest component of `x` where the interior-point solution and
Clarabel's differ, relative to the largest component of `x`. It is `8.2e-11` on Eq QP, the
class with the fewest active rows, and `1.8e-5`–`2.1e-5` on Huber and Portfolio, the two with
the loosest conditioning; the other four classes fall between `1e-9` and `1e-6`. Both solvers
aim at the same `1e-8` accuracy, so this is agreement between two independent implementations
of the same method, not a referee against a known solution.

The interior-point method reaches 2–11 outer iterations on every class here, close to
Clarabel's count on the same problem. ADMM's iteration counts are not comparable to either —
it stops at a looser tolerance and by a different test — so its column states what ADMM costs
on the same instance, not a claim about which method is faster.

## Clarabel's two factorizations, at sizes where one can win

Clarabel ships twice from one algorithm: as the Julia package compared above, and as a Rust
crate. The crate carries a sparse `LDLᵀ` factorization through
[faer](https://github.com/sarah-quinones/faer-rs) that the Julia package has no equivalent of,
so the crate is run here under both of its factorizations — `faer` and its own bundled
`qdldl` — over the same problem files. The pair isolates the factorization: everything else
about the solver is held fixed between those two columns.

Reproduce with
`PUREQP_CLARABEL_GRID=large julia --project=bench PureIPM/bench/clarabel_rs_compare.jl`;
samples are in `PureIPM/bench/results/clarabel_rs_compare_large.json`, and the smallest-size
grid of the section above writes `clarabel_rs_compare.json`. Clarabel.jl 0.11.1 against crate
0.11.1 (`default-features = false`, `faer-sparse`), all at `1e-8`, single-threaded BLAS and
`max_threads = 1`, pinned to one core on Julia 1.13.1. The host's clock is not pinned, so read
the ratios rather than the absolute times. The two Julia solvers are measured in one process
and interleaved A-B-B-A; the crate is a separate process timing `DefaultSolver::new` plus
`solve()` on its own clock, so a Julia-against-Rust difference carries that much more
uncertainty than a difference within either language. `setup` is the share of that solver's own
total spent before it starts iterating.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureIPM", "clarabel_rs_compare_large")
msus(x) = (t = num(x); isfinite(t) ? sigfig(t / 1000) * " ms" : "—")
pct(part, whole) = (f = num(part) / num(whole); isfinite(f) ? int(round(f * 100)) * "%" : "—")
emph(s, cond) = cond ? "**" * s * "**" : s
benchtable(
    d["results"], [
        "class" => r -> str(r["name"]),
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "IPM it" => r -> int(r["ipm"]["iter"]),
        "IPM" => r -> msus(r["ipm"]["median_us"]),
        "setup" => r -> pct(r["ipm"]["setup_median_us"], r["ipm"]["median_us"]),
        ".jl it" => r -> int(r["clarabel_jl"]["iter"]),
        "Clarabel.jl" => r -> emph(
            msus(r["clarabel_jl"]["median_us"]),
            num(r["clarabel_jl"]["median_us"]) < num(r["ipm"]["median_us"]),
        ),
        "setup " => r -> pct(r["clarabel_jl"]["setup_median_us"], r["clarabel_jl"]["median_us"]),
        ".rs faer" => r -> msus(r["clarabel_rs_by_ldl"]["faer"]["min_us"]),
        ".rs qdldl" => r -> msus(r["clarabel_rs_by_ldl"]["qdldl"]["min_us"]),
        "faer/qdldl" => r -> (
            v = num(r["clarabel_rs_by_ldl"]["faer"]["min_us"]) / num(r["clarabel_rs_by_ldl"]["qdldl"]["min_us"]);
            emph(ratio(v), v < 1)
        ),
    ]
)
```

All four solvers reach the same answer: the largest component of `x` where `InteriorPoint` and
any of the others differ is `1.3e-5` on the loosest class and below `1e-7` on nine of the
thirteen.

The `IPM` and `Clarabel.jl` columns are medians over their samples; the two `.rs` columns are
minima, because the Rust binary reports a minimum over its repetitions and nothing else. A
minimum flatters the side that reports it, so read the `.rs` times as a floor.

**faer needs fill before its factorization is worth anything.** faer is simplicial below a
flops/nnz(L) ratio of 40 and supernodal above it, and a supernodal factorization blocks its
work into dense kernels. Clarabel hardcodes that threshold, so the problem decides which path
runs rather than a setting. Only `Eq QP` crosses it here, and that is the class whose `P` is
dense — 640000 nonzeros at `n = 800`. There faer takes half of QDLDL's time and 59% of
Clarabel.jl's, which has no faer backend at all to reach. On the other eleven cases faer runs
between equal to QDLDL and 29% slower than it, and the three genuinely sparse classes
(`Portfolio`, `Lasso`, `SVM`) never cross the threshold at any size here. A factorization that
exploits dense blocks is worth exactly as much as the dense blocks a problem has.

**On a sparse class, setup is most of the solve, and both solvers pay about the same for it.**
The `setup` columns are over half the total on `Lasso` at `n = 3060` and on `SVM` at
`n = 3030`. That time is the symbolic analysis of a sparse `KKT` pattern the two solvers share:
on `SVM` at `n = 3030` it is 14.50 ms for `InteriorPoint` against 15.00 ms for Clarabel.jl, a
3% difference on a term that is half the answer. Neither solver can win a class where that
term dominates.

**`SVM` at `n = 3030` is the one case Clarabel.jl takes**, 27.81 ms against 28.69 ms, and
subtracting setup says why. The iteration phase is 14.19 ms over 11 iterations against
12.81 ms over 10, which is 1.290 ms and 1.281 ms per iteration — the same cost per iteration to
within a percent. `InteriorPoint` is 0.5 ms ahead on setup and then spends 1.29 ms on an
eleventh iteration, and that is the whole of the 0.88 ms it loses by. The step is a real one
rather than a refinement: its duality gap is `8.2e-5` at the tenth iterate and `9.0e-7` at the
eleventh, and `refine_iter = 0` changes nothing. The counts are not a like-for-like measure
either, since the two stopping tests are different tests: `InteriorPoint` finishes exactly
primal-feasible with a dual residual of `7.4e-13`, Clarabel with residuals near `2e-10` and a
tighter relative gap, and the two objectives agree to `5.2e-10` relative. The reading of that
row is that the two are within a few percent and the stopping rule decides it.

The `Eq QP` rows are not a comparison of solvers at all. `InteriorPoint` reaches the answer
there in one iteration against Clarabel's five or six, which is a property of an
equality-constrained problem under each method's own test rather than a speed.

## The matrix-free backend

`linsys = :indirect` never forms the reduced matrix. It multiplies by it through the caller's
own products and solves each linear system with preconditioned conjugate gradients (CG). Whether
that is faster depends on the problem.

On dense QPs the direct backend is faster: the matrix-free backend takes the same number of
iterations or close to it, but each costs more, so the whole solve is 2.6× slower at
`n = 50, m = 100` and 3.6× slower at `n = 200, m = 400`.

On sparse QPs it depends on size. The direct backend builds the reduced matrix from the stored
entries, but it still factors and inverts an `n×n` dense matrix, which costs `O(n³)` however
sparse the input is. The matrix-free backend costs `O(nnz)` per CG iteration and stores only
vectors. Keeping about five nonzeros per row of `A` and growing the problem
(`PureOSQP/bench/indirect_backend.jl`, `eps_abs = eps_rel = 1e-6`, single-threaded BLAS; samples in
`PureOSQP/bench/results/indirect_backend.json`):

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "indirect_backend")
mib(x) = sigfig(num(x) / 1024^2) * " MiB"
emph(v) = (s = ratio(v); v > 1 ? "**" * s * "**" : s)
indirect_rows(family) = benchtable(
    d[family], [
        "n" => r -> int(r["n"]),
        "m" => r -> int(r["m"]),
        "density" => r -> sigfig(num(r["density"]) * 100, 2) * "%",
        "direct" => r -> ms(r["t_direct"]),
        "matrix-free" => r -> ms(r["t_indirect"]),
        "speedup" => r -> emph(num(r["speedup"])),
        "direct memory" => r -> mib(r["bytes_direct"]),
        "matrix-free memory" => r -> mib(r["bytes_indirect"]),
        "iterations (direct / matrix-free)" => r ->
            int(r["iter_direct"]) * " / " * int(r["iter_indirect"]),
    ]
)
indirect_rows("by_size")
```

The matrix-free backend becomes faster between `n = 500` and `n = 1000`. It also uses less
memory at every size, from 2.6× less at `n = 200` to 50× less at `n = 4000`, because the direct
backend stores an `n×n` inverse and the matrix-free one stores vectors. For large problems,
memory can decide whether the direct backend fits at all.

At a fixed size, density decides. At `n = 1000, m = 2000`:

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "indirect_backend")
emph(v) = (s = ratio(v); v > 1 ? "**" * s * "**" : s)
benchtable(
    d["by_density"], [
        "density" => r -> sigfig(num(r["density"]) * 100, 2) * "%",
        "direct" => r -> ms(r["t_direct"]),
        "matrix-free" => r -> ms(r["t_indirect"]),
        "speedup" => r -> emph(num(r["speedup"])),
        "iterations (direct / matrix-free)" => r ->
            int(r["iter_direct"]) * " / " * int(r["iter_indirect"]),
    ]
)
```

Both backends get slower as `nnz` grows, but the matrix-free one slows much faster. It pays
for every CG iteration, and it also needs more ADMM iterations, while the direct backend pays
only when it refactorizes.

The two backends take different numbers of iterations because CG solves each linear system
only approximately, so these are end-to-end times, not times per iteration. The script checks
that the two objectives agree to `1e-6`.

## An operator that is never materialized

The sections above all give the solver a matrix. This one gives it an operator that has no
entries at all, only `mul!`. Nothing can be formed from it, so none of the direct backends
apply and the solver uses the matrix-free backend.

The comparison is against the same operator stored as a `Matrix`, which is the other option
when the entries exist. Reproduce with `julia --project=bench PureOSQP/bench/operator_protocol.jl`;
samples are in `PureOSQP/bench/results/operator_protocol.json`.

The operator is `P = Diagonal(d) + α v vᵀ`, written three ways:

- **protocol**: a type that implements the
  [operator protocol](@ref "What to implement, in order"), with `size`, `mul!`,
  `is_materializable` returning `false`, `is_convex`, and `structural_rows`. It is defined in
  `PureOSQP/bench/lazy_operator.jl` and stores a vector and a scalar, never an `n×n` array.
- **linearmap**: the same operator as a `LinearMaps.LinearMap`, which implements none of the
  protocol and is wrapped in [`PureQPBase.ProductOperator`](@ref) to reach the same backend.
- **matrix**: the same operator stored as an `n×n` `Matrix`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "operator_protocol")
benchtable(
    d["cases"], [
        "n" => r -> int(r["n"]),
        "BLAS threads" => r -> int(r["blas_threads"]),
        "protocol setup" => r -> us(r["protocol_setup"]),
        "linearmap setup" => r -> us(r["linearmap_setup"]),
        "matrix setup" => r -> ms(r["matrix_setup"]),
        "matrix / protocol" => r ->
            "**" * ratio(num(r["matrix_setup"]) / num(r["protocol_setup"])) * "**",
        "protocol step" => r -> us(r["protocol_step"]),
        "linearmap step" => r -> us(r["linearmap_step"]),
        "matrix step" => r -> us(r["matrix_step"]),
    ]
)
```

**Compared with the matrix, setup is 20× to 57× faster and each step is 8× to 10× slower.**
Setup has no `O(n³)` factorization to do. Each step runs a CG solve through the operator, where
the matrix path does one multiplication by a stored inverse. Which is faster overall depends on
how many iterations the problem needs. The setup saving grows with `n`; the step cost does not.
More BLAS threads speed up only the matrix setup, so the setup ratio shrinks from 32× to 20× at
`n = 500`.

**The LinearMap is set up about 3× faster than the protocol type, and each of its steps is
1.33× to 1.44× slower.** Both come from the same cause: a `LinearMap` has no preconditioner.
The matrix-free backend preconditions CG with the diagonal of the reduced matrix. The protocol
type provides that diagonal through `structural_rows`, here `P[j,j] = d[j] + α v[j]²`, and its
extra setup time goes to computing it. A `LinearMap` has no entries, so the preconditioner
stays at ones and CG needs more work each iteration. `probe = true` does not help, because
probing is used for equilibration, not for this diagonal.

The wrapping itself costs nothing measurable. Whether the preconditioner is worth its setup
depends on how many iterations the problem runs. To give a map one, define
`PureQPBase.structural_rows` for its type, the same method that enables equilibration.

The solver is only allocation-free if the operator's `mul!` is: a `mul!` that allocates makes
every iteration allocate.

## Conditioning

At `n = 300`, with `κ(P) = κ(A)` swept up to `1e12`. Reproduce with
`julia --project=bench PureOSQP/bench/illconditioned.jl`; samples are in
`PureOSQP/bench/results/illconditioned.json`. Single-threaded BLAS, `eps_abs = eps_rel = 1e-6`,
`max_iter = 20000`. Two shapes: a dense `P` and `A`, and the same size split into six
`BlockDiagonal` blocks of 50, each with the same `κ`.

**Check the status column first.** Where a backend reaches `SOLVED`, its time is the time to a
solution and its iterations and objective can be compared. Where it stops at `MAX_ITER`, its time
is just the cost of 20 000 iterations and its objective is not a solution.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "illconditioned")
fastest = Dict{Any, Float64}()
for r in d["cases"]
    key = (r["kappa"], r["shape"])
    t = num(r["seconds"])
    if isfinite(t) && (!haskey(fastest, key) || t < fastest[key])
        fastest[key] = t
    end
end
benchtable(
    d["cases"], [
        "κ" => r -> sci(r["kappa"]),
        "shape" => r -> str(r["shape"]),
        "backend" => r -> "`" * str(r["backend"]) * "`",
        "status" => r -> str(r["status"]),
        "iterations" => r -> int(r["iter"]),
        "objective" => r -> sigfig(num(r["obj_val"]), 5),
        "words" => r -> int(r["words"]),
        "time" => r -> (
            s = ms(r["seconds"]);
            num(r["seconds"]) ≈ get(fastest, (r["kappa"], r["shape"]), NaN) ? "**" * s * "**" : s
        ),
    ]
)
```
**The matrix-free backend only works at the lowest `κ`.** At `κ = 1e4` it reaches the same
objectives as the direct backends, but it needs 2.3× to 3.4× more iterations and is slower
than the `cholesky` and `block` backends. From `κ = 1e8` it does not converge. The reduced matrix it works with has condition
number `κ(A)²`, so `κ = 1e8` means `1e16` for CG. For badly conditioned problems, use a direct
backend.

## Every solver on one ill-conditioned problem

Two dense `400×400` problems at `cond(A) = 1e8`, with 40 of the 400 rows active at the optimum
and the bounds placed so that they cut off the unconstrained minimizer. One family carries
Kronecker structure a solver can be told about; the other is unstructured, so the comparison is
of implementations alone. Reproduce with
`julia --project=bench bench/illconditioned_shootout.jl`; samples are in
`bench/results/illconditioned_shootout.json`. Single-threaded BLAS, `tol = 1e-6` for every
solver that takes one.

`x⋆` is planted: the bounds and `q` are built so that a drawn point is the optimum, so the
reference is optimal by construction rather than by another solver's say-so. The error column is
`‖x − x⋆‖∞ / max(1, ‖x⋆‖∞)`. Clarabel at `1e-12` is run as an independent check that the
planting is sound, and agrees with `x⋆` to `9.6e-13` on the Kronecker family and `1.2e-9` on the
dense one. `cond(A) = 1e8` is the limit at which that check holds; higher, and the solution is
not determined well enough for the error column to mean anything.

**Kronecker structure, `cond(A) = 1e8`**

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "illconditioned_shootout")
family(name) = only(f for f in d["families"] if f["family"] == name)
# A solver that reports no iteration count records -1 rather than a count.
iters(r) = (v = num(get(r, "iter", nothing)); isfinite(v) && v > 0 ? int(v) : "—")
shootout(name) = (
    rows = sort(collect(family(name)["rows"]); by = r -> num(r["time_s"]));
    best_time = minimum(num(r["time_s"]) for r in rows);
    best_err = minimum(num(r["relerr"]) for r in rows);
    emph(s, cond) = cond ? "**" * s * "**" : s;
    benchtable(
        rows, [
            "solver" => r -> str(r["solver"]),
            "time" => r -> emph(ms(r["time_s"]), num(r["time_s"]) ≈ best_time),
            "error in `x`" => r -> emph(sci(r["relerr"]), num(r["relerr"]) ≈ best_err),
            "iterations" => iters,
            "violation" => r -> sci(r["violation"]),
        ]
    )
)
shootout("Kronecker")
```

**Unstructured, `cond(A) = 1e8`**

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "illconditioned_shootout")
family(name) = only(f for f in d["families"] if f["family"] == name)
iters(r) = (v = num(get(r, "iter", nothing)); isfinite(v) && v > 0 ? int(v) : "—")
rows = sort(collect(family("Dense control")["rows"]); by = r -> num(r["time_s"]))
best_time = minimum(num(r["time_s"]) for r in rows)
best_err = minimum(num(r["relerr"]) for r in rows)
emph(s, cond) = cond ? "**" * s * "**" : s
benchtable(
    rows, [
        "solver" => r -> str(r["solver"]),
        "time" => r -> emph(ms(r["time_s"]), num(r["time_s"]) ≈ best_time),
        "error in `x`" => r -> emph(sci(r["relerr"]), num(r["relerr"]) ≈ best_err),
        "iterations" => iters,
        "violation" => r -> sci(r["violation"]),
    ]
)
```

Every solver returns a feasible point; the verdict comes from the returned `x`, not from the
status the solver reports. The iteration count is not recorded for DAQP.

**Few active rows is where an active-set method belongs.** 40 of 400 rows are active, which is
the regime the [iteration-count split](@ref "Accuracy and iteration count") identifies. Both
active-set solvers are six to nine orders of magnitude closer to `x⋆` than any other solver
here. Against the fastest solver that is not an active-set method, PureDAQP is 30× faster on
the structured family and 19× on the unstructured one.

**On the unstructured family the objective does not separate the solvers and `x` does.** The
objectives agree to `3.5e-7` relative across the whole table, while the errors in `x` spread
over ten orders of magnitude. The problem is flat along the directions the first-order and
interior-point solvers leave, so a comparison made on objective alone would report these runs
as equivalent.

**Declaring the Kronecker structure costs time here.** `PureOSQP` on the same family is 6×
faster left to its default ladder than with `:kronecker`, which routes to the indirect solver:
the reduced matrix conjugate gradients work with has condition number `κ(A)²`, which is `1e16`
at this `κ`. [Matrix types](@ref "What each algorithm does with each type") says which
structures pay and when.

## The primal-dual integral

`∫|gap| dt` over the solve, under two quadrature rules, with `profile_primdual = true`. See
[Measuring how fast a solve converges](@ref) for what the numbers mean and which to use.
Reproduce with `julia --project=bench PureOSQP/bench/primdual_integral.jl`; samples in
`PureOSQP/bench/results/primdual_integral.json`, single-threaded BLAS, `eps_abs = eps_rel = 1e-9`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "primdual_integral")
num3(x) = (v = num(x); isfinite(v) ? sigfig(v, 3) : "—")
benchtable(
    d["cases"], [
        "problem" => r -> str(r["case"]),
        "iterations" => r -> int(r["iter"]),
        "trapezoid" => r -> sci(r["trapezoid"]),
        "log-mean" => r -> sci(r["logmean"]),
        "log/trap" => r -> num3(r["ratio"]),
        "overhead" => r -> sigfig(num(r["overhead_percent"]), 2) * "%",
    ]
)
```

The same problem, sampled at five intervals:

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "primdual_integral")
rows = d["sampling"]
tightest = maximum(num(r["ratio"]) for r in rows)
benchtable(
    rows, [
        "sample every" => r -> int(r["check_termination"]),
        "iterations" => r -> int(r["iter"]),
        "trapezoid" => r -> sci(r["trapezoid"]),
        "log-mean" => r -> sci(r["logmean"]),
        "log/trap" => r -> (
            s = sigfig(num(r["ratio"]), 3);
            num(r["ratio"]) ≈ tightest ? "**" * s * "**" : s
        ),
    ]
)
```

Both estimates converge on about `2.1e-6`. `check_termination` sets the sampling interval and
also decides where the solve stops, so the iteration count moves down the column too: this is
a trend, not a controlled single-variable sweep.

Measuring costs under 1% either way — the `overhead` column straddles zero, so the cost is
below what this timing can resolve — and it does not change what it measures: the benchmark
asserts the iteration count and objective are identical with profiling on and off before
reporting a time.

**Neither number is reproducible.** Both integrate against wall-clock time, so they cannot be
compared across machines, or between runs on a machine whose clock is not pinned. No test
asserts a value for either.

## The ρ schedule

Iterations to `eps_abs = eps_rel = 1e-6` under the three `adaptive_rho` modes, on the OSQP
benchmark classes. `:iterations` retunes on a fixed schedule; `:kkt_error` retunes only when
the relative KKT error has fallen by `adaptive_rho_fraction` since the last look. Reproduce
with `julia --project=bench PureOSQP/bench/rho_schedule.jl`; samples in
`PureOSQP/bench/results/rho_schedule.json`, single-threaded BLAS.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "rho_schedule")
# The totals the samples carry, as a final row; it has no refactorization count of its own.
total = Dict(
    "class" => "**total**",
    "iter_disabled" => d["totals"]["disabled"],
    "iter_iterations" => d["totals"]["iterations"],
    "iter_kkt_error" => d["totals"]["kkt_error"],
)
bold(r, s) = str(r["class"]) == "**total**" ? "**" * s * "**" : s
benchtable(
    vcat(collect(d["cases"]), Any[total]), [
        "class" => r -> str(r["class"]),
        "`:disabled`" => r -> bold(r, int(r["iter_disabled"])),
        "`:iterations`" => r -> bold(r, int(r["iter_iterations"])),
        "`:kkt_error`" => r -> bold(r, int(r["iter_kkt_error"])),
        "refactorizations" => r -> haskey(r, "refac_disabled") ? join(
            (int(r[k]) for k in ("refac_disabled", "refac_iterations", "refac_kkt_error")), " / "
        ) : "",
    ]
)
```

Adapting is worth **3.8×** across the corpus. The two triggers are indistinguishable: same
iterations and same refactorizations on every class.

Iterations to a fixed tolerance, not wall clock, and not the regime in [Conditioning](@ref),
where nothing converges on any schedule.

## Declared structure against a sparse factorization

libosqp accepts `P` and `A` only as sparse matrices, so the only structure it can use is where
the zeros are. PureOSQP accepts the same sparse matrices and also structured types. Running one
problem three ways separates two effects: PureOSQP sparse against libosqp sparse compares the
implementations, and PureOSQP structured against PureOSQP sparse shows what passing the
structure is worth.

Reproduce with `julia --project=bench PureOSQP/bench/structured_vs_osqp.jl`; samples are in
`PureOSQP/bench/results/structured_vs_osqp.json`. Single-threaded BLAS, the same settings on all three,
and libosqp timed from CSC arrays built beforehand.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("PureOSQP", "structured_vs_osqp")
emph(v) = (s = ratio(v); v > 1 ? "**" * s * "**" : s)
benchtable(
    d["cases"], [
        "structure" => r -> str(r["case"]),
        "nnz(`A`)" => r -> sigfig(num(r["density_percent"]), 2) * "%",
        "iterations" => r -> int(r["iter"]),
        "libosqp 1.0 (sparse)" => r -> ms(r["osqp_seconds"]),
        "PureOSQP (sparse)" => r -> ms(r["sparse_seconds"]),
        "PureOSQP (structured)" => r -> ms(r["structured_seconds"]),
        "sparse vs libosqp" => r -> ratio(r["sparse_vs_osqp"]),
        "structured vs sparse" => r -> emph(num(r["structured_vs_sparse"])),
    ]
)
```

The script checks that all three agree before it reports a row: the same status, the same
number of iterations, and objectives within `1e-9`.

**The Kronecker row shows the difference between sparsity and structure.** `A₁ ⊗ A₂` has no
zeros, so a sparse factorization has nothing to skip and both sparse paths factor a dense
400×400 matrix. The same matrix passed as its two 20×20 factors is solved through their
eigenvectors, 51.6× faster.

**The banded row is slower with the structure declared**: 0.77× of PureOSQP's own sparse path.
The sparse factor of a banded matrix is already banded, so declaring the band gains nothing and
the banded backend's extra bookkeeping costs a little. The banded backend is chosen when it beats
the dense path, not the sparse one, which is why it is used here.

## What was measured, and when

Every table above is rendered at documentation-build time from the saved samples named in its
own section, through the helpers in `docs/benchtables.jl`, so no figure on this page is
transcribed. Three small tables are the exception, and say what they are: the working set's
`add_row!` and `remove_row!` costs, and the OSQP suite's sparsity ratios and setup-phase split,
none of which the scripts record.

This section is rendered the same way, from `bench/results/index.json`, which
`bench/consolidate.jl` writes by reading each package's cache. A benchmark that has never
been run, or whose samples came from a different Julia than the rest, shows up here.

```@eval
using JSON, Markdown
index = JSON.parsefile(
    joinpath(@__DIR__, "..", "..", "bench", "results", "index.json"); allownan = true
)
entries = [e for e in index["entries"] if !endswith(e["name"], "_raw")]
io = IOBuffer()
println(io, "| package | benchmark | script | Julia | BLAS threads |")
println(io, "|---|---|---|---|---|")
for e in sort(entries; by = e -> (e["package"], e["name"]))
    script = isnothing(e["script"]) ? "—" : "`" * e["script"] * "`"
    threads = isnothing(e["blas_threads"]) ? "—" : string(Int(e["blas_threads"]))
    println(
        io, "| ", e["package"], " | ", e["name"], " | ", script, " | ",
        something(e["julia_version"], "—"), " | ", threads, " |"
    )
end
Markdown.parse(String(take!(io)))
```

Regenerate the index with

```sh
julia --project=bench bench/consolidate.jl
```

which also names any cache no script writes any more, and any benchmark script that has
never been run.

## What unmaterialized operators cost

Each solver reaches an operator `A` on two paths, one that iterates the linear system and one that
factors it, and each row below is one problem solved twice: once with `A` held as its factors, once
with `Matrix(A)` and no backend named, which is what a caller gets for handing over a matrix and
letting the solver choose. The backends therefore differ between the two columns — that is the
comparison — so the iteration counts are given for both, and a time ratio that is really an
iteration-count difference is visible rather than hidden.

Reproduce with `julia --project=bench bench/unmaterialized_paths.jl`; samples are written to
`bench/results/unmaterialized_paths.json`. The six problems are defined in
`bench/unmaterialized_problems.jl`, which the [Unmaterialized operators](@ref) examples and one
test item per algorithm package also read; the test items pin the backend each path reaches and
compare the problems against the fingerprint recorded in the JSON, so a table built from problems
that have since changed is a failing test.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "unmaterialized_paths")
emph(v, cut) = (s = ratio(v); v >= cut ? "**" * s * "**" : s)
benchtable(
    d["paths"], [
        "algorithm" => r -> str(r["algorithm"]),
        "path" => r -> str(r["path"]),
        "n" => r -> int(r["n"]),
        "backend" => r -> "`" * str(r["backend"]) * "`",
        "its" => r -> int(r["iters"]),
        "matrix backend" => r -> "`" * str(r["dense_backend"]) * "`",
        "its " => r -> int(r["dense_iters"]),
        "setup×" => r -> emph(num(r["setup_ratio"]), 100),
        "solve×" => r -> emph(num(r["solve_ratio"]), 10),
        "memory×" => r -> ratio(r["bytes_ratio"]),
    ]
)
```

The iteration counts match across the two columns on every row, so the ratios are the
representation and nothing else.

Both columns reach the same answer. The objectives agree to machine precision on five rows and to
`1.8e-08` on the first, where both sides stop at the default tolerance and conjugate gradients
solve each step only to a tolerance of their own.

The two `PureDAQP` rows are one problem under both of its working-set representations, so their
absolute times compare directly: 0.697 ms through the rows and a `QR`, 0.243 ms through the Gram
matrix and an `LDLᵀ`.

**Where the gain comes from differs by path.** On `kronecker` it is algebraic: the reduced matrix is
diagonalized by the factors' own eigenvectors, so two 24×24 eigenproblems replace a 576×576
factorization, and setup falls by two orders of magnitude. On `indirect` it is the product: a
Kronecker product applies in `O(n(m₁+m₂))` against `O(mn)` dense, per conjugate-gradient iteration.
On `product_reduced` the matrix is assembled rather than avoided, so the saving is bounded by what
assembling it costs — 2.5× on the solve, against 55× on a setup that no longer forms `A`.

**The dual active-set method gains least on setup and on storage**, both 1.3×, and that is a
property of the method: its setup factors the reduction rather than reading `A`, and its working
set is what it stores. The solve gains more, 2.8× through the rows and 4.6× through the Gram
matrix, because each working-set change reads a row, and a Kronecker operator answers that from its
factors rather than from a stored matrix.

**Memory is a secondary consequence, not the point.** The 35× at `n = 576` is two 24×24 factors
standing for a 576×576 matrix and the `n²` reduced matrix a direct backend would hold; the reason
to keep the operator is that the structure survives into the arithmetic, and the storage follows.

## Reducing through a composition's parts

A composition `B E` held as its parts reduces as `Eᵀ (Bᵀ diag(w) B) E`, so the outer part is
applied once. Reaching the same matrix through products of the composition applies it twice per
column, `2n` times, which is what the gain below is made of — and why it tracks how expensive the
outer part is to apply rather than how large the problem is.

Both columns compute the same matrix and agree with it to `7.5e-16` or better on every row.
Reproduce with `julia --project=bench bench/composition_reduction.jl`; samples go to
`bench/results/composition_reduction.json`.

```@eval
include(joinpath(@__DIR__, "..", "benchtables.jl"))
d = cache("shared", "composition_reduction")
emph(v) = (s = ratio(v); v >= 2 ? "**" * s * "**" : s)
benchtable(
    d["composed"], [
        "outer part" => r -> str(r["kind"]),
        "m" => r -> int(r["m"]),
        "k" => r -> int(r["k"]),
        "n" => r -> int(r["n"]),
        "parts" => r -> msval(r["parts_ms"]),
        "products" => r -> msval(r["products_ms"]),
        "ratio" => r -> emph(num(r["ratio"])),
    ]
)
```

A dense outer part costs `mk` per application, so removing `2n` of them is most of the work and the
ratio sits between 2.2× and 4.0×. A Kronecker outer part is already cheap to apply — that is what
its own factors buy — so there is little left for the composition to save, and the ratio is 1.1× to
1.4×. Keeping the parts is still right there: it is what lets the Kronecker factors be used at all.

The scaling is applied to the inner part's columns once, `(E D)ᵀ G (E D)`, so the result is two
matrix products accumulated into the reduced matrix rather than `n` matrix-vector products. That
choice is the difference between this table and a slower one: going column by column measured
**0.73×–0.85×** against the products it was meant to beat, which is the shape of a reduction that
reproduces the arithmetic and loses the rate.

**A sum has no reduction of its own, by measurement.** Expanding `(Σ Bᵢ)ᵀ W (Σ Bᵢ)` gives `K`
diagonal terms at `2n` products of a term each and `K(K-1)/2` cross pairs at `2n` more, so
`nK(K+1)` in all; a product of the sum costs `2nK`, because `A eⱼ = Σ Bₜ eⱼ` sums the terms in one
pass. The expansion measured 1.04× at two terms, 0.69× at three and 0.49× at four, and was removed.
`SumOperator` keeps the terms for every other reason — the sum is never formed, each term keeps its
representation — and leaves the reduction to the generic path.
