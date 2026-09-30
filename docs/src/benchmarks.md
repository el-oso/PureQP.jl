# Benchmarks

Each package's benchmarks live in its own `bench/` directory and write their samples to that
package's `bench/results/`, so either set can be run without the other. Every section below
names the script that produced it. The shared problem generators, the snapshot gate and the
StrictMode audit sit in the top-level `bench/`, which also holds the index the last section
renders.

## Against OSQP

Dense random QPs, compared with libosqp 1.0, the C library OSQP itself ships. Reproduce with
`julia --project=bench PureOSQP/bench/headtohead.jl`, or with `--project=bench/pureblas` to add the
PureBLAS column; samples are in `PureOSQP/bench/results/headtohead.json`.

Both solvers run with `eps_abs = eps_rel = 1e-6`, single-threaded BLAS, ρ adapted every 50
iterations, and the duality-gap test (`check_dualgap`) turned off. Each solver checks the gap
at a different point in the iteration, so with it on they stop at different iterations and the
times would compare stopping rules instead of solvers. libosqp reads only sparse (CSC) matrices,
so it is timed on its setup and solve calls from CSC arrays built beforehand. PureOSQP is timed
with OpenBLAS and again with [PureBLAS](https://github.com/el-oso/PureBLAS.jl) at commit
`ea79919`.

| n | m | PureOSQP | PureOSQP + PureBLAS | libosqp 1.0 | vs libosqp | vs libosqp (PureBLAS) | iterations | objective rel. Δ | max \|Δx\| |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 0.036 ms | 0.031 ms | 0.065 ms | 1.80× | 2.09× | 100 | 1.1e-15 | 8.9e-16 |
| 25 | 50 | 0.382 ms | 0.329 ms | 1.41 ms | 3.69× | 4.28× | 675 | 6.4e-16 | 6.9e-15 |
| 50 | 100 | 1.10 ms | 1.02 ms | 5.08 ms | 4.60× | 4.98× | 900 | 7.5e-16 | 6.9e-15 |
| 100 | 200 | 2.95 ms | 2.35 ms | 21.1 ms | 7.17× | 9.01× | 850 | 6.5e-16 | 8.8e-15 |
| 200 | 400 | 14.6 ms | 12.8 ms | 129 ms | 8.87× | 10.08× | 1175 | 6.5e-16 | 1.3e-14 |
| 400 | 800 | 41.2 ms | 40.0 ms | 723 ms | **17.57×** | 18.07× | 625 | 0.0 | 5.4e-14 |
| 100 | 50 | 0.304 ms | 0.290 ms | 1.05 ms | 3.45× | 3.62× | 50 | 2.1e-15 | 1.3e-14 |
| 200 | 100 | 1.05 ms | 1.01 ms | 6.99 ms | 6.64× | 6.92× | 50 | 1.2e-16 | 1.9e-14 |
| 100 | 1000 | 98.2 ms | 97.3 ms | 691 ms | 7.04× | 7.11× | 7475 | 4.5e-11 | 4.0e-9 |
| 200 | 2000 | 215 ms | 211 ms | 1329 ms | 6.18× | 6.31× | 3025 | 1.4e-16 | 3.3e-14 |

Both solvers take the same number of iterations in every case, so the equilibration, the ρ
schedule and the residual tests match libosqp step for step. The solutions agree to about
`1e-14`. The one exception is the `100 × 1000` case, which runs 7475 iterations and agrees to
`4e-9`. PureOSQP is faster in every case, and PureBLAS changes the time but not the iteration
count.

libosqp has no Julia wrapper, so `PureOSQP/bench/osqp_v1.jl` calls the library in `OSQP_jll` v100
through `ccall`. The header that library ships describes both its single- and double-precision
builds, and some of its type definitions are wrong for the double build. So the wrapper takes the
integer and float sizes from the library itself. `verify_abi()` checks them each time the file is
loaded, and throws if a rebuilt library changes them.

## Sequential re-solves

`P` and `A` stay fixed while `q`, `l` and `u` change every step, as in a receding-horizon
control loop. Each case runs 20 solves. Reproduce with
`julia --project=bench PureOSQP/bench/update_bench.jl`, or with `--project=bench/pureblas` to add the
PureBLAS column; samples are in `PureOSQP/bench/results/update_bench.json`.

libosqp runs the same loop through its own update call, `osqp_update_data_vec`, which keeps
the factorization. Both solvers use `eps_abs = eps_rel = 1e-6` with `check_dualgap` off, and
the script checks that their objectives agree to `1e-4` at every step.

| n | m | `update!` | fresh `setup` each step | saved | libosqp 1.0 | vs libosqp | `update!` + PureBLAS | vs | factorizations |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 0.96 ms | 1.37 ms | 1.42× | 1.74 ms | 1.81× | 0.74 ms | 1.29× | 8 |
| 25 | 50 | 2.56 ms | 5.30 ms | 2.07× | 10.2 ms | 4.00× | 2.18 ms | 1.17× | 14 |
| 50 | 100 | 12.7 ms | 19.4 ms | 1.53× | 60.1 ms | 4.72× | 11.8 ms | 1.08× | 7 |
| 100 | 200 | 75.0 ms | 69.1 ms | **0.92×** | 524 ms | 6.98× | 68.5 ms | 1.09× | 3 |
| 200 | 400 | 311 ms | 323 ms | 1.04× | 2603 ms | 8.36× | 289 ms | 1.08× | 1 |

"Factorizations" counts PureOSQP's factorizations over all 20 steps: one from `setup`, plus
one for each step where a row changed between equality and inequality.

**PureOSQP's `update!` is 1.8× to 8.4× faster than libosqp's update loop.** The gap grows with
size, as it does for single solves above.

**`update!` saves little over calling `setup` again**, because `setup` is already fast (see
[Sparse A](@ref "Sparse A")). At `n = 100` it is slightly slower. The two loops also start
from different points (`update!` keeps the previous solution), so they take different
numbers of iterations. `update!` helps most when each solve is short, so that setup is a
large share of it. It is also the way to keep that previous solution as the starting point.

**With PureBLAS this loop is 1.08× to 1.29× faster.** It refactorizes more often per iteration
than a single solve does, so the factorization routines matter more here. PureBLAS is faster
than OpenBLAS at `potrf` and slower at `potri`, which is why its margin here does not follow
the factorization count.

The factorization count is often more than 1. Equilibration rescales the bounds, and a row
whose scaled gap `ũ - l̃` falls below `RHO_TOL` is treated as an equality, which changes its
`ρ`. libosqp applies the same rule, so an `update!` can still trigger a factorization.

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

| n | m | OpenBLAS | PureBLAS | ratio | iterations | \|Δx\| |
|---|---|---|---|---|---|---|
| 25 | 50 | 0.198 ms | 0.186 ms | 1.07× | 225 | 4.2e-15 |
| 50 | 100 | 0.611 ms | 0.582 ms | 1.05× | 350 | 4.3e-15 |
| 100 | 200 | 5.94 ms | 5.20 ms | 1.14× | 1500 | 8.7e-15 |
| 200 | 400 | 9.44 ms | 8.12 ms | 1.16× | 650 | 1.0e-14 |
| 100 | 50 | 0.346 ms | 0.333 ms | 1.04× | 50 | 8.4e-15 |

**Correctness is exact** — identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in — **and it is faster than OpenBLAS on every case**:

| operation (n=200, m=400) | OpenBLAS | PureBLAS | ratio | runs |
|---|---|---|---|---|
| `gemv A*x` | 3.38 µs | 3.07 µs | 1.10× | every iteration |
| `gemv Aᵀy` (transposed) | 3.91 µs | 2.53 µs | **1.55×** | every iteration |
| `gemv At*y` (materialized transpose) | 3.79 µs | 3.00 µs | 1.26× | — |
| `syrk WᵀW` | 313 µs | 233 µs | **1.34×** | on a ρ update |
| `potrf` | 114 µs | 57.7 µs | **1.98×** | on a ρ update |
| `potri` | 216 µs | 322 µs | **0.67×** | on a ρ update |
| `symv R⁻¹b` | 1.58 µs | 1.35 µs | 1.17× | every iteration |
| `trsv F\b` | 11.6 µs | 12.1 µs | 0.96× | — |

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

| n | m | PureOSQP | libosqp 1.0 | PureDAQP | DAQP | PureIPM | Clarabel |
|---|---|---|---|---|---|---|---|
| 10 | 20 | 0.089 ms | 0.163 ms | 0.008 ms | **0.003 ms** | 0.066 ms | 0.121 ms |
| 25 | 50 | 0.163 ms | 0.532 ms | 0.043 ms | **0.035 ms** | 0.296 ms | 0.631 ms |
| 50 | 100 | 0.487 ms | 2.13 ms | **0.162 ms** | 0.207 ms | 1.14 ms | 2.87 ms |
| 100 | 200 | 4.51 ms | 34.9 ms | **1.20 ms** | 1.87 ms | 5.06 ms | 17.0 ms |
| 200 | 400 | 8.18 ms | 76.3 ms | **11.6 ms** | 16.0 ms | 28.9 ms | 108 ms |
| 100 | 50 | 0.297 ms | 1.03 ms | **0.147 ms** | 0.356 ms | 1.08 ms | 6.02 ms |

libosqp and Clarabel read sparse matrices, so each is timed from sparse copies built
beforehand. Both DAQP implementations read dense ones, which is what they are built for.

**An active-set method wins this shape.** These are dense problems with few rows active at
the solution, which is what an active-set method is for: a few expensive steps, then it stops
at the exact vertex. PureDAQP is the fastest solver here from `n = 50` up and beats the C
implementation it follows by 1.28× to 2.42×. PureIPM beats Clarabel at every size, by 1.9× to
5.6×.

Below `n = 50` it does not, and the reason is per-iteration work rather than a fixed cost:
setup is about a fifth of a solve at `n = 25`, and the remainder divides into iterations that
each cost more than the C implementation's. The next table gives the whole picture.

These are the figures for the default `working_set = :rows`, which trades 1.2×–1.5× for a
rank test that holds on an ill-conditioned reduction; `:gram` is the faster setting, and
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

| n \ m/n | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 25 | 0.89× | 0.66× | 0.61× | 0.94× | 0.93× |
| 50 | 1.40× | 1.10× | 1.09× | 1.14× | 1.04× |
| 100 | 2.30× | 1.80× | 1.64× | 1.11× | 1.64× |
| 200 | 3.34× | 2.32× | 1.39× | 1.71× | 2.14× |
| 400 | **4.05×** | 1.70× | 1.49× | 1.85× | 2.19× |

The two agree to between `4.5e-16` and `3.9e-14` in every cell, and the script checks that
before recording a time.

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

| n | m | `:rows` | `:gram` | libdaqp | `:rows` / C | `:gram` / C |
|---|---|---|---|---|---|---|
| 25 | 25 | 0.026 ms | 0.017 ms | 0.017 ms | 0.63× | **0.96×** |
| 25 | 50 | 0.067 ms | 0.047 ms | 0.041 ms | 0.60× | 0.86× |
| 25 | 100 | 0.080 ms | 0.059 ms | 0.062 ms | 0.78× | **1.06×** |
| 50 | 50 | 0.117 ms | 0.089 ms | 0.124 ms | 1.06× | **1.39×** |
| 100 | 100 | 0.527 ms | 0.437 ms | 0.819 ms | 1.55× | **1.87×** |

## Choosing a working set for the active-set method

[`ActiveSet`](@ref) keeps its working set in one of two forms, chosen with `working_set`.
`:rows` factors the active rows themselves; `:gram` factors their Gram matrix `Mₐ Mₐᵀ`. These
are the measurements; [choosing a working set for `ActiveSet`](@ref "Choosing a working set
for `ActiveSet`") says how to decide. Reproduce with
`julia --project=bench bench/working_set_choice.jl`; samples are in
`bench/results/working_set_choice.json`. Single-threaded BLAS.

**Well conditioned: `:gram` is faster.**

| n | m | `:gram` | `:rows` | iterations | `:rows` cost |
|---|---|---|---|---|---|
| 25 | 50 | 0.039 ms | 0.057 ms | 47 / 47 | 1.48× |
| 50 | 100 | 0.189 ms | 0.233 ms | 99 / 99 | 1.23× |
| 100 | 200 | 1.048 ms | 1.236 ms | 214 / 214 | 1.18× |
| 200 | 400 | 9.631 ms | 12.028 ms | 817 / 817 | 1.25× |

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

**Ill conditioned: `:gram` cannot decide rank, and stops.**

| case | `:gram` | `:rows` |
|---|---|---|
| `cond(A R⁻¹) ≈ 1e17`, 30 variables, 200 rows | `NUMERICAL_ERROR` | `SOLVED`, violation 1.2e-06 |

The problem is feasible and `:rows` solves it.

**What does not separate them: rows that are exact combinations of others.**

| case | `:gram` | `:rows` |
|---|---|---|
| 40 of 130 rows are combinations of the rest | `SOLVED`, violation 1.7e-13 | `SOLVED`, violation 7.4e-14 |

Dependence that is exact is not the same difficulty as dependence blurred by rounding. Both
forms carry the dependent row and walk the direction it opens.

## Choosing a representation

A matrix can be passed dense, sparse, as a structured type, or as an operator that is never
formed. These three cases measure when each is the better choice. Reproduce with
`julia --project=bench PureOSQP/bench/representation_choice.jl`; samples are in
`PureOSQP/bench/results/representation_choice.json`. Single-threaded BLAS, `eps_abs = eps_rel = 1e-6`.
The script checks that every run reaches `SOLVED` before it reports a time.

**An operator that is cheap to apply, in a problem where that is not the main cost.** `P` is a
diagonal plus a rank-one term, stored as `O(n)` numbers, compared with the `n×n` matrix it
represents. `A` is dense in both, so the products with `A` cost `O(n²)` either way.

| n | iterations (operator / dense) | operator | dense | speedup |
|---|---|---|---|---|
| 200 | 225 / 125 | 3.4 ms | 2.2 ms | 0.65× |
| 500 | 150 / 125 | 18.6 ms | 20.0 ms | 1.08× |
| 1000 | 175 / 175 | 85.2 ms | 128.8 ms | 1.51× |

The operator skips the factorization but solves a linear system with CG in every iteration. At
`n = 200` that costs more than it saves. From `n = 500` the factorization's `O(n³)` cost grows
faster than the CG work, and the operator is faster.

**An operator applied in `O(n)` whose dense form costs `O(n²)`.** A moving average with a window
of `n/20`, so about a tenth of the entries are nonzero, which is too many for a sparse format to
be the obvious choice. It is diagonally dominant, so conditioning plays no part here.

| n | window | fill | iterations (operator / dense) | operator | dense | speedup | dense `A` |
|---|---|---|---|---|---|---|---|
| 500 | 25 | 9.9% | 200 / 75 | 12.2 ms | 15.1 ms | **1.23×** | 1.9 MiB |
| 1000 | 50 | 9.8% | 100 / 100 | 21.9 ms | 109 ms | **5.01×** | 7.6 MiB |
| 2000 | 100 | 9.8% | 100 / 100 | 227 ms | 737 ms | **3.24×** | 30.5 MiB |
| 4000 | 200 | 9.8% | 100 / 75 | 904 ms | 4756 ms | **5.26×** | 122 MiB |

Compared with the first case, what changed is neither size nor sparsity. It is that applying the
operator costs less than multiplying by its dense form. Once the dense matrix no longer fits in
cache, its product is limited by memory speed, so the operator's lead holds as `n` grows.

**An ill-conditioned problem with a structured backend.** `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so a
Kronecker operator with `κ = 1e12` is two factors with `κ = 1e6` each. Its backend
eigendecomposes the two factors and never forms the product.

| n | κ(A) | iterations | `kronecker` | dense | speedup | CG on the same problem |
|---|---|---|---|---|---|---|
| 400 | 1e12 | 625 / 625 | 2.3 ms | 27.9 ms | **12.1×** | `SOLVED` in 625 iterations |
| 1600 | 1e12 | 1100 / 1100 | 21.8 ms | 1566 ms | **71.8×** | `SOLVED` in 850 iterations |

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

| problem | as `Matrix` | structured | speedup | iterations |
|---|---|---|---|---|
| dense `P` (vs `Symmetric`) | 30.6 ms | 35.8 ms | 0.85× | 4375 |
| diagonal `P` (vs `Diagonal`) | 4.67 ms | 4.32 ms | 1.08× | 550 |
| tridiagonal `P` (vs `SymTridiagonal`) | 19.0 ms | 18.1 ms | 1.05× | 2550 |
| `A` as a `SubArray` | 6.88 ms | 6.77 ms | 1.02× | 900 |

**A structured `P` helps only a little here.** Equilibration reads only the entries a band
type can hold, which makes that step 2.3× faster on a `Diagonal` `P` and 2.2× on a
`SymTridiagonal`. The whole solve gains less, because equilibration is a small part of it.
Every iteration also multiplies by `A` twice, and with `m = 2n` and `A` dense in every row
above, that is four times the work of the `P` product.

What the table does show is that any `AbstractMatrix` works, the caller's matrices are never
copied or changed, and a type with a cheaper `mul!` gets it. With a dense `A`, the structured
type is not a speedup. [Structured backends](@ref) covers the cases where the structure changes
which linear system is solved, which is where the large gains are.

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

| `P`, `A` | n | bw(`R`) | backend | setup | dense setup | setup× | total | dense total | total× |
|---|---|---|---|---|---|---|---|---|---|
| `Diagonal`, `Diagonal` | 400 | 0 | `diagonal` | 49.6 µs | 5.37 ms | **108×** | 0.08 ms | 7.03 ms | **83×** |
| `SymTridiagonal`, `Diagonal` | 400 | 1 | `tridiagonal` | 65.1 µs | 5.40 ms | **83×** | 0.23 ms | 12.1 ms | **52×** |
| `SymTridiagonal`, `Tridiagonal` | 400 | 2 | `banded` | 114 µs | 5.37 ms | **47×** | 0.62 ms | 11.3 ms | **18×** |
| `Diagonal`, `Diagonal` | 2000 | 0 | `diagonal` | 230 µs | 393 ms | **1710×** | 0.40 ms | 488 ms | **1216×** |
| `SymTridiagonal`, `Diagonal` | 2000 | 1 | `tridiagonal` | 306 µs | 389 ms | **1270×** | 1.15 ms | 831 ms | **725×** |
| `SymTridiagonal`, `Tridiagonal` | 2000 | 2 | `banded` | 549 µs | 384 ms | **699×** | 3.08 ms | 778 ms | **252×** |

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

| `K` | block size | iterations | `block` | dense | speedup | matrix-free | block vs matrix-free | factor words | dense words | memory saved |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 120 | 50 | 3.69 ms | 4.39 ms | 1.19× | 5.75 ms | 1.56× | 14 520 | 28 920 | 2.0× |
| 3 | 80 | 50 | 2.50 ms | 3.27 ms | 1.31× | 4.17 ms | 1.66× | 9 720 | 28 920 | 3.0× |
| 4 | 60 | 50 | 1.92 ms | 2.70 ms | 1.41× | 3.27 ms | 1.70× | 7 320 | 28 920 | 4.0× |
| 6 | 40 | 150 | 1.65 ms | 2.55 ms | 1.55× | 2.56 ms | 1.55× | 4 920 | 28 920 | 5.9× |
| 8 | 30 | 100 | 1.23 ms | 2.06 ms | 1.68× | 2.18 ms | 1.77× | 3 720 | 28 920 | 7.8× |
| 12 | 20 | 250 | 1.44 ms | 2.47 ms | 1.71× | 2.09 ms | 1.45× | 2 520 | 28 920 | 11.5× |
| 20 | 12 | 100 | 0.84 ms | 2.34 ms | **2.80×** | 3.29 ms | 3.94× | 1 560 | 28 920 | **18.5×** |

**The block backend is the fastest of the three at every split.** Its lead over the dense path
grows from 1.19× to 2.80× as the blocks get smaller. At `n = 240` a dense factorization is
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

| n | k | iters | setup | dense setup | total | dense total | total× |
|---|---|---|---|---|---|---|---|
| 500 | 1 | 75 | 80.3 µs | 9.31 ms | 0.24 ms | 19.2 ms | **81×** |
| 500 | 2 | 75 | 88.5 µs | 9.34 ms | 0.26 ms | 19.3 ms | **75×** |
| 500 | 6 | 100 | 138 µs | 9.34 ms | 0.58 ms | 14.3 ms | **25×** |
| 500 | 16 | 125 | 282 µs | 9.34 ms | 0.79 ms | 15.5 ms | **20×** |
| 1000 | 1 | 75 | 154 µs | 57.7 ms | 0.45 ms | 119 ms | **267×** |
| 1000 | 6 | 100 | 270 µs | 56.4 ms | 1.11 ms | 79.1 ms | **71×** |
| 1000 | 16 | 150 | 557 µs | 55.8 ms | 1.64 ms | 98.2 ms | **60×** |
| 2000 | 1 | 75 | 304 µs | 391 ms | 0.87 ms | 806 ms | **923×** |
| 2000 | 6 | 100 | 533 µs | 385 ms | 2.16 ms | 559 ms | **259×** |
| 2000 | 16 | 175 | 1.11 ms | 392 ms | 3.71 ms | 691 ms | **186×** |

The ratio grows with `n` at fixed `k` and shrinks as `k` climbs, which is what `O(nk)` against
`O(n²)` predicts. The solver drops this backend once `10k > n`, which sits below the measured
crossing, so the limit holds at any BLAS thread count — see the gate discussion in
`PureOSQP/bench/results/gate_crossover_lowrank.json`. Each row is a different problem, so read across
a row rather than down a column.

## Sparse A

Random sparse `P` and `A`, the kind of problem libosqp's sparse factorization is designed for.
PureOSQP is measured twice: given dense copies, and given the sparse matrices, where the
SparseArrays extension reads only the stored entries. libosqp gets the sparse matrices. The
results come from the same script as the table above.

| n | m | density | PureOSQP, dense copies | PureOSQP, sparse | libosqp 1.0 | best vs libosqp |
|---|---|---|---|---|---|---|
| 200 | 400 | 1% | 7.18 ms | 2.52 ms | 4.85 ms | 1.92× |
| 200 | 400 | 5% | 11.5 ms | 6.03 ms | 24.3 ms | 4.03× |
| 200 | 400 | 20% | 11.6 ms | 14.7 ms | 37.3 ms | 3.22× |
| 400 | 800 | 1% | 38.7 ms | 10.2 ms | 38.9 ms | 3.80× |
| 400 | 800 | 5% | 56.2 ms | 25.7 ms | 111 ms | 4.33× |
| 400 | 800 | 20% | 139 ms | 172 ms | 394 ms | 2.82× |

Dense and sparse storage take the same number of iterations in every row; the script checks
this.

**PureOSQP is faster in every row.** The smallest margin, 1.92×, is on the smallest and
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

**Banded**, as in a model-predictive control horizon. The reduced matrix keeps the band, so
PureOSQP factors it with CHOLMOD. This is the kind of problem libosqp's sparse factorization of
the full KKT system is designed for.

The `cholmod` figures below predate the change to how `SparseCholmod` reads its factor
(commit `1609934`), which cut its `factorize!` cost — they have not been re-measured since, so
treat PureOSQP's margin here as a lower bound rather than the current number.

| n | m | nnz(A) | PureOSQP backend | PureOSQP | libosqp 1.0 | vs libosqp | iterations |
|---|---|---|---|---|---|---|---|
| 200 | 400 | 2775 | `cholmod` | 7.57 ms | 11.4 ms | 1.50× | 1400 |
| 500 | 1000 | 6975 | `cholmod` | 15.3 ms | 23.2 ms | 1.52× | 1125 |
| 1000 | 2000 | 13975 | `cholmod` | 57.9 ms | 89.8 ms | 1.55× | 2200 |
| 2000 | 4000 | 27975 | `cholmod` | 48.2 ms | 71.1 ms | 1.47× | 825 |

**Uniformly random sparsity**, with no structure a sparse factorization can use. The reduced
matrix fills in, so PureOSQP builds it from the stored entries and factors it as a dense
matrix.

| n | m | nnz(A) | PureOSQP backend | PureOSQP | libosqp 1.0 | vs libosqp | iterations |
|---|---|---|---|---|---|---|---|
| 200 | 400 | 753 | `sparse_formed` | 1.81 ms | 2.35 ms | 1.30× | 225 |
| 200 | 400 | 4014 | `sparse_formed` | 7.33 ms | 30.0 ms | 4.09× | 1100 |
| 500 | 1000 | 4994 | `sparse_formed` | 27.0 ms | 129 ms | 4.77× | 1100 |
| 1000 | 2000 | 9926 | `sparse_formed` | 103 ms | 421 ms | 4.09× | 850 |
| 2000 | 4000 | 20104 | `sparse_formed` | 664 ms | 2667 ms | 4.02× | 800 |

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

| class | n | m | PureOSQP backend | PureOSQP | libosqp 1.0 | vs libosqp | per iteration | setup |
|---|---|---|---|---|---|---|---|---|
| Random QP | 50 | 500 | `sparse_formed` | 4.94 ms | 9.47 ms | **1.92×** | 1.80× | 3.68× |
| Eq QP | 200 | 100 | `sparse_formed` | 2.37 ms | 4.13 ms | **1.74×** | 5.44× | 1.16× |
| SVM | 808 | 1600 | `ldlfactorizations` | 3.63 ms | 5.82 ms | **1.61×** | 1.70× | 0.98× |
| Portfolio | 505 | 506 | `ldl_kkt` | 3.52 ms | 4.31 ms | **1.22×** | 1.25× | 1.01× |
| Control | 320 | 540 | `sparse_formed` | 6.18 ms | 7.44 ms | **1.20×** | 1.69× | 0.30× |
| Lasso | 816 | 816 | `ldlfactorizations` | 1.51 ms | 1.65 ms | **1.09×** | 1.23× | 0.95× |
| Huber | 1806 | 1800 | `ldlfactorizations` | 3.47 ms | 3.77 ms | **1.08×** | 1.17× | 0.88× |

The last two columns are libosqp's time divided by PureOSQP's, for the iterations and for
setup. The objectives agree to `1e-13` or better in six classes and to `1e-9` in the seventh.
Random QP and Eq QP read `sparse_formed`: their reduced matrix is now accumulated from the
stored entries of `P` and `A` rather than formed with a dense product, the same arithmetic
without the `m×n` buffer. The other five classes keep the backend they had.

**PureOSQP is faster in all seven classes, by 1.08× to 1.92×.** These problems have the block
and band structure real problems tend to have. The random sparse families elsewhere on this
page have none, which is the hardest case for any sparse factorization.

**Each iteration is faster in every class, and setup is where libosqp does well.** libosqp's
setup is faster in five classes, so the classes with the smallest overall lead are the ones where
PureOSQP's faster iterations make up for a slower setup.

Every figure is a median over ten seconds of samples. The two solvers run in the order
PureOSQP, libosqp, libosqp, PureOSQP, and the median is taken over both turns together, so a
slow drift during a row affects both solvers equally.

**Control's setup is 3.4× slower on purpose.** PureOSQP forms the reduced matrix `R` and
inverts it, which costs `O(n³)` once and makes each iteration a single `symv`. That makes the
iterations 1.69× faster over 325 of them. A sparse `LDLᵀ` of the reduced matrix or of the KKT
system matches libosqp's setup but makes the iterations slower, so the dense path is kept.

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

| class | n | m | ADMM iter | ADMM time | IPM iter | IPM time | Clarabel iter | Clarabel time | `x`, IPM vs Clarabel |
|---|---|---|---|---|---|---|---|---|---|
| Random QP | 6 | 60 | 200 | 67.1 µs | 9 | 103.7 µs | 9 | 128.5 µs | 2.4e-8 |
| Eq QP | 20 | 10 | 50 | 34.0 µs | 2 | 45.1 µs | 6 | 94.4 µs | 8.2e-11 |
| Portfolio | 101 | 102 | 125 | 238.7 µs | 10 | 321.8 µs | 11 | 530.8 µs | 2.1e-5 |
| Lasso | 204 | 204 | 100 | 293.7 µs | 6 | 270.5 µs | 9 | 771.6 µs | 1.0e-6 |
| SVM | 202 | 400 | 375 | 957.3 µs | 9 | 468.2 µs | 8 | 661.0 µs | 3.0e-9 |
| Huber | 602 | 600 | 125 | 1070.1 µs | 9 | 1145.5 µs | 10 | 1943.9 µs | 1.8e-5 |
| Control | 64 | 108 | 50 | 131.1 µs | 7 | 311.0 µs | 8 | 573.3 µs | 8.4e-9 |

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

| n | m | density | direct | matrix-free | speedup | direct memory | matrix-free memory | iterations (direct / matrix-free) |
|---|---|---|---|---|---|---|---|---|
| 200 | 400 | 2.5% | 5.07 ms | 44.1 ms | 0.11× | 0.5 MiB | 0.19 MiB | 1050 / 6200 |
| 500 | 1000 | 1% | 27.3 ms | 34.5 ms | 0.79× | 2.4 MiB | 0.49 MiB | 1100 / 1275 |
| 1000 | 2000 | 0.5% | 104 ms | 86.7 ms | **1.20×** | 8.5 MiB | 0.97 MiB | 850 / 1500 |
| 2000 | 4000 | 0.25% | 699 ms | 249 ms | **2.81×** | 32.4 MiB | 1.97 MiB | 800 / 1900 |
| 3000 | 6000 | 0.17% | 2143 ms | 367 ms | **5.85×** | 71.5 MiB | 3.01 MiB | 1025 / 1250 |
| 4000 | 8000 | 0.125% | 3998 ms | 655 ms | **6.10×** | 125.7 MiB | 3.89 MiB | 875 / 1700 |

The matrix-free backend becomes faster between `n = 500` and `n = 1000`. It also uses less
memory at every size, from 3× less at `n = 200` to 32× less at `n = 4000`, because the direct
backend stores an `n×n` inverse and the matrix-free one stores vectors. For large problems,
memory can decide whether the direct backend fits at all.

At a fixed size, density decides. At `n = 1000, m = 2000`:

| density | direct | matrix-free | speedup | iterations (direct / matrix-free) |
|---|---|---|---|---|
| 0.2% | 72.0 ms | 22.7 ms | **3.17×** | 550 / 775 |
| 0.5% | 101 ms | 85.1 ms | **1.19×** | 850 / 1500 |
| 1% | 105 ms | 260 ms | 0.40× | 875 / 1975 |
| 2% | 140 ms | 1168 ms | 0.12× | 1025 / 4300 |

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

| n | BLAS threads | protocol setup | linearmap setup | matrix setup | matrix / protocol | protocol step | linearmap step | matrix step |
|---|---|---|---|---|---|---|---|---|
| 500 | 1 | 229 µs | 79 µs | 7.27 ms | **32×** | 403 µs | 580 µs | 47 µs |
| 1000 | 1 | 900 µs | 283 µs | 51.6 ms | **57×** | 1491 µs | 2119 µs | 181 µs |
| 500 | 8 | 231 µs | 81 µs | 4.65 ms | **20×** | 402 µs | 535 µs | 41 µs |
| 1000 | 8 | 909 µs | 285 µs | 23.3 ms | **26×** | 961 µs | 1300 µs | 98 µs |

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

| κ | shape | backend | status | iterations | objective | words | time |
|---|---|---|---|---|---|---|---|
| 1e4 | dense | `cholesky` | SOLVED | 225 | 300.97 | 45 150 | 9.7 ms |
| 1e4 | dense | `bunchkaufman` | SOLVED | 225 | 300.97 | 180 300 | 27.8 ms |
| 1e4 | dense | `indirect` | SOLVED | 775 | 300.97 | 0 | 95.1 ms |
| 1e4 | blocks | `block` | SOLVED | 175 | 164.60 | 7 650 | **3.4 ms** |
| 1e4 | blocks | `cholesky` | SOLVED | 175 | 164.60 | 45 150 | 8.3 ms |
| 1e4 | blocks | `indirect` | SOLVED | 400 | 164.60 | 0 | 21.8 ms |
| 1e8 | dense | `cholesky` | SOLVED | 500 | 298.49 | 45 150 | 15.5 ms |
| 1e8 | dense | `indirect` | MAX_ITER | 20 000 | 323.89 | 0 | 8004 ms |
| 1e8 | blocks | `block` | SOLVED | 1 200 | 440.33 | 7 650 | **8.8 ms** |
| 1e10 | dense | `cholesky` | **MAX_ITER** | 20 000 | 344.72 | 45 150 | 354 ms |
| 1e10 | blocks | `block` | **SOLVED** | 13 975 | 616.87 | 7 650 | **72.4 ms** |
| 1e10 | blocks | `cholesky` | SOLVED | 14 025 | 616.87 | 45 150 | 105.9 ms |
| 1e12 | dense | `cholesky` | MAX_ITER | 20 000 | 1552.3 | 45 150 | 353 ms |
| 1e12 | blocks | `block` | MAX_ITER | 20 000 | 1180.3 | 7 650 | 104.2 ms |

**Where ADMM stops converging.** The dense problem solves up to `κ = 1e8` and fails from
`1e10`. Between those, it reaches `SOLVED` at `1e9` in 5 900 iterations and
`SOLVED_INACCURATE` at `3e9`. The iteration count grows quickly before that point: 225 at
`1e4`, 500 at `1e8`, 5 900 at `1e9`. A looser tolerance does not help: at `κ = 1e12` the dense
problem does not converge at `eps = 1e-3` in 50 000 iterations.

**The block problem converges at a higher `κ`.** At `κ = 1e10` the block problem converges and
the dense one does not. They are different problems, not one problem written two ways, so this
does not show that structure fixes conditioning. It shows that a problem that splits into six
50×50 systems is still solvable at a `κ` where one 300×300 system is not. The block backend is
also the fastest direct backend at every `κ` where it converges, and uses a sixth of the
storage.

**The matrix-free backend only works at the lowest `κ`.** At `κ = 1e4` it reaches the same
objectives as the direct backends, but it needs 2.3× to 3.4× more iterations and is slower
than the `cholesky` and `block` backends. From `κ = 1e8` it does not converge. The reduced matrix it works with has condition
number `κ(A)²`, so `κ = 1e8` means `1e16` for CG. For badly conditioned problems, use a direct
backend.

## The primal-dual integral

`∫|gap| dt` over the solve, under two quadrature rules, with `profile_primdual = true`. See
[Measuring how fast a solve converges](@ref) for what the numbers mean and which to use.
Reproduce with `julia --project=bench PureOSQP/bench/primdual_integral.jl`; samples in
`PureOSQP/bench/results/primdual_integral.json`, single-threaded BLAS, `eps_abs = eps_rel = 1e-9`.

| problem | iterations | trapezoid | log-mean | log/trap | overhead |
|---|---|---|---|---|---|
| well conditioned | 75 | 6.84e-6 | 1.68e-6 | 0.246 | 0.29% |
| moderate | 175 | 4.64e-6 | 3.14e-6 | 0.677 | 0.09% |
| ill conditioned | 50 000 | 1.98e-5 | 1.52e-5 | 0.768 | 0.19% |
| larger | 150 | 8.72e-4 | 5.23e-4 | 0.600 | 0.72% |

The same problem, sampled at five intervals:

| sample every | iterations | trapezoid | log-mean | log/trap |
|---|---|---|---|---|
| 25 iterations | 75 | 7.11e-6 | 1.76e-6 | 0.247 |
| 10 | 60 | 3.20e-6 | 1.50e-6 | 0.470 |
| 5 | 55 | 1.96e-6 | 1.32e-6 | 0.674 |
| 2 | 52 | 1.96e-6 | 1.72e-6 | 0.876 |
| 1 | 51 | 2.28e-6 | 2.11e-6 | **0.927** |

Both estimates converge on about `2.1e-6`. `check_termination` sets the sampling interval and
also decides where the solve stops, so the iteration count moves down the column too: this is
a trend, not a controlled single-variable sweep.

Measuring costs under 1% and does not change what it measures — the benchmark asserts the
iteration count and objective are identical with profiling on and off before reporting a time.

**Neither number is reproducible.** Both integrate against wall-clock time, so they cannot be
compared across machines, or between runs on a machine whose clock is not pinned. No test
asserts a value for either.

## The ρ schedule

Iterations to `eps_abs = eps_rel = 1e-6` under the three `adaptive_rho` modes, on the OSQP
benchmark classes. `:iterations` retunes on a fixed schedule; `:kkt_error` retunes only when
the relative KKT error has fallen by `adaptive_rho_fraction` since the last look. Reproduce
with `julia --project=bench PureOSQP/bench/rho_schedule.jl`; samples in
`PureOSQP/bench/results/rho_schedule.json`, single-threaded BLAS.

| class | `:disabled` | `:iterations` | `:kkt_error` | refactorizations |
|---|---|---|---|---|
| Random QP | 5200 | 1225 | 1225 | 1 / 2 / 2 |
| Eq QP | 50 | 50 | 50 | 1 / 1 / 1 |
| Portfolio | 2750 | 600 | 600 | 1 / 2 / 2 |
| Lasso | 450 | 125 | 125 | 1 / 2 / 2 |
| SVM | 1825 | 325 | 325 | 1 / 2 / 2 |
| Huber | 175 | 125 | 125 | 1 / 2 / 2 |
| Control | 450 | 450 | 450 | 1 / 1 / 1 |
| **total** | **10 900** | **2 900** | **2 900** | |

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

| structure | nnz(`A`) | iterations | libosqp 1.0 (sparse) | PureOSQP (sparse) | PureOSQP (structured) | sparse vs libosqp | structured vs sparse |
|---|---|---|---|---|---|---|---|
| Kronecker | 100% | 250 | 120 ms | 54.6 ms | 1.06 ms | 2.19× | **51.6×** |
| tridiagonal | 0.2% | 75 | 0.58 ms | 0.47 ms | 0.19 ms | 1.26× | **2.49×** |
| low-rank | 0.7% | 75 | 0.53 ms | 0.51 ms | 0.22 ms | 1.04× | **2.29×** |
| block-diagonal | 12.5% | 175 | 2.55 ms | 2.24 ms | 1.34 ms | 1.14× | **1.67×** |
| banded | 0.5% | 75 | 0.66 ms | 0.66 ms | 0.86 ms | 1.00× | 0.77× |

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

Every table above is transcribed by hand from a saved run. This section is not: it is
rendered at documentation-build time from `bench/results/index.json`, which
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
