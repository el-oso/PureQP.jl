# Changelog

Notable changes, and the measurements that drove them. The reference documentation states
what is true now; this file is where the history lives.

## Unreleased

### Changed

- **Every per-iteration call allocates nothing, and the test suites prove it.** StrictMode
  checks the kernels at load time and StrictModeTest proves them allocation-free and `--trim`
  compatible on every backend in `PureQPBase`'s own source. Allocations removed, per call on
  a 12-variable, 30-row problem unless noted: `ReducedCholesky`'s refactorization 304 B,
  `FullKKT`'s 22 008 B (it now calls `sytrf` into pivot and work arrays it holds, giving the
  same factor as `bunchkaufman!`), `KroneckerReduced`'s refactorization 5 424 B on a 20-variable
  problem (it keeps its eigenbases when only the weights move), and so the interior-point
  refactorization and ADMM's `adapt_rho!` by the same amounts. A regularization bump sets
  `SystemWeights.sigma` in place instead of building a new object. `LinearSystem` and
  `Preconditioner` are StrictMode contracts.

- **Three packages, one repository.** `PureQPBase` holds everything the algorithms share:
  the problem representation, the linear-system backends and their selection, equilibration,
  termination, the polishing and derivative kernels, the `QPAlgorithm`/`QPWorkspace`
  contracts, and the generic `setup`, `solve` and `solve!`, which take the algorithm as their
  sixth positional argument. `PureOSQP` supplies `OperatorSplitting` and the five-argument
  forms that run it by default; `PureIPM` supplies `InteriorPoint`. Either solver re-exports
  the base, so `using` one is enough; `using` both puts both algorithms on the same `solve`.
  The six matrix-support extensions and the ChainRulesCore rules moved to `PureQPBase`, since
  they extend backends rather than either algorithm.

  Solve times are unchanged: on the seven suite classes the medians sit within this machine's
  spread of the single-package build, at identical iteration counts.

- **`setup_backend` takes the options, the preconditioner and the accelerator by position**,
  and the backend name reaches it as a `Val` built where the caller's keyword is still a
  literal. A keyword call carries a `NamedTuple` whose names inference loses track of once
  several keywords survive to it, which `--trim` rejects as an unresolved call.

- **An unrecognized keyword is refused by name.** It no longer reaches the `Options` keyword
  constructor, whose `MethodError` lists every option without saying which name was wrong.

### Removed

- **The MathOptInterface `algorithm` raw attribute.** Each package supplies its own
  optimizer — `PureOSQP.Optimizer` runs `OperatorSplitting`, `PureIPM.Optimizer` runs
  `InteriorPoint` — and each accepts the shared options and only its own algorithm's
  parameters. One wrapper implementation serves both, in `PureQPBase`, carrying the algorithm
  as a field and reporting its package as the solver name.

### Added

- **`setup` and `solve` take `InteriorPoint()`**, a Mehrotra predictor–corrector
  interior-point method, alongside the default `OperatorSplitting()`. `setup` returns an
  `InteriorPointWorkspace`; its default tolerances are `1e-8`, not ADMM's `1e-3`. It handles equality, one-sided, two-sided and free rows, uses
  equilibration as ADMM does, and solves its Newton systems with the direct backends: the
  dense full KKT factorization for dense data and for a dense `P` with a sparse `A`, and for
  a sparse pair whichever sparse form the sparsity pattern calls for. Diagonal, tridiagonal, banded, block and
  sparse reduced pairs keep their structured backends: on problems of each structure, QPs and
  LPs with and without equality rows, each solves in the same outer iterations as the full KKT
  factorization with a referee residual of at most `1.1e-8` (`bench/ipm_backends.jl`). A diagonal `P`
  with a `RowCoupled` `A` gets the full KKT factorization instead, since the low-rank backend
  does not solve the LPs. A re-solve starts from the previous
  point. It is generic over the element type `T <: Real`: its tolerances, regularizations
  and short-step threshold default to `1e-8` in `Float64` and finer arithmetic and to
  `sqrt(eps(T))` in coarser arithmetic, so `Float32` data solves at the defaults; `BigFloat` and
  `ForwardDiff.Dual` run as well, dual numbers on the reduced Cholesky only, where
  `bunchkaufman!` is not needed. In this version it refuses by name GPU arrays,
  `linsys = :kronecker` and `linsys = :lowrank`. `verbose = true` prints a progress report:
  a header, one row per termination check with the barrier parameter `mu` and the step
  length `alpha` in place of ADMM's `rho`, and a footer with the status, the iteration count,
  the residuals and the run time; on the matrix-free backend the row gains a `cg iters`
  column and the footer adds the total CG iterations and the number of missed inner solves.
- **The interior-point method runs on operators with a caller-supplied preconditioner.**
  `InteriorPoint()` takes `linsys = :indirect` with a `preconditioner` other than the built-in
  ones and `scaling = 0`, for matrices and for operators that supply products only
  (`ProductOperator`, LinearMaps, SciMLOperators); anything else on that path is refused by
  name, and `linsys = :auto` never chooses it. Conjugate gradients starts each Newton solve from
  zero and stops on its recursively updated residual at `cg_tol_fraction` (default `0.1`) of
  `min(μ, ‖r‖∞)`; a solve that spends `cg_max_iter` (default `500`) iterations, or that Krylov
  abandons because the preconditioner is not positive definite, is missed, and `cg_fail_limit`
  (default `3`) misses in a row end the run `NUMERICAL_ERROR`. `update_preconditioner!` receives
  `k = -1` for the starting point and the outer iteration afterwards. Measured by
  `bench/ipm_matrixfree.jl` on 24 dense planted instances as `LinearMap`s (`n` 500–2000,
  `κ(A) ∈ {1, 1e6}`, active fractions 0.1 and 0.9, two-sided and mixed rows) with a lagged
  Cholesky preconditioner refreshed every third outer iteration: all 24 solve at `eps = 1e-6`
  with a referee residual of at most `7.9e-7`, in the same outer iterations as the dense KKT
  factorization, with median inner iterations over the last three outer iterations at most
  46 and no solve above 138 inner iterations. A primal-infeasible instance returns its
  certificate, and a preconditioner with a negative entry ends `NUMERICAL_ERROR`. One sparse
  instance with a limited-memory incomplete `LDLᵀ` reaches the iteration cap and ends
  `NUMERICAL_ERROR`. `docs/src/operators.md` records the table.
- **The interior-point method detects infeasibility.** It reports `PRIMAL_INFEASIBLE` and
  `DUAL_INFEASIBLE` (and their `*_INACCURATE` variants) with certificates in
  `Solution.prim_inf_cert` and `Solution.dual_inf_cert`, found by ADMM's certificate tests on
  its last step and its normalized iterate at `eps_prim_inf` and `eps_dual_inf` (default
  `1e-8`). On random `n = 20`, `m = 40` primal- and dual-infeasible problems, all 40 runs
  (ten seeds each, with and without equilibration) end with a certificate that checks
  against the data, in 8–20 and 14–34 iterations.
- **The interior-point method takes `time_limit` and returns on `Ctrl-C`**, with
  `TIME_LIMIT_REACHED` and `INTERRUPTED` and the point reached, as ADMM does.
- **`NUMERICAL_ERROR`**, a new `Status` value. The interior-point method returns it when its
  Newton system stays unfactorizable after `max_reg_bumps` (default `5`) tenfold increases of
  the regularization, when a residual stops being finite, when conjugate gradients misses
  `cg_fail_limit` Newton solves in a row, and when three consecutive steps
  are shorter than `1e-8` (`sqrt(eps(T))` in arithmetic coarser than `Float64`) without a
  certificate. `has_solution` is false for it, and MathOptInterface reports
  `MOI.NUMERICAL_ERROR` with no result. ADMM never returns it.
- **`Solution.cg_iters`** reports how many conjugate-gradient iterations a solve took on
  `linsys = :indirect`. It is zero on every direct backend.
- **`setup` and `solve` take a `preconditioner` keyword** for `linsys = :indirect`. The default
  is the Jacobi diagonal, as before; `IdentityPreconditioner()` turns preconditioning off; any
  other object is applied through `LinearAlgebra.ldiv!`, so a `Cholesky` works as it is, and is
  refreshed by a method of `update_preconditioner!` whenever `ρ` or `σ` changes. A caller's own
  preconditioner requires `scaling = 0`.
- **The interior-point method supports polishing, derivatives, `update!`, `update_settings!`
  and MathOptInterface.** The `polishing`, `polish_refine_iter` and `delta` options apply to
  it; a `SOLVED` or `SOLVED_INACCURATE` run polishes exactly as ADMM does
  and reports `Solution.polished`/`status_polish`. `adjoint_derivative` and
  `forward_derivative` accept an `InteriorPointWorkspace`, taking the derivative at its unscaled point
  through the same active-set KKT matrix ADMM uses, and refuse an unpolished one by name: an
  interior-point solution's inactive-row multipliers sit at `μ_final` rather than at zero,
  which the active-set test cannot otherwise tell apart from a genuinely active row.
  `update!` validates and adopts `q`, `l`, `u`, `P` and `A` through the same path as ADMM and
  reclassifies rows, without refactorizing: every outer iteration refactorizes at its own
  weights regardless. `update_settings!` rejects a changed `linsys` or `scaling`; every other
  option and parameter, including the two regularizations, takes effect on the next solve
  without any refactorization here, since a solve always resets the regularizations before its
  first iteration. `MOI.RawOptimizerAttribute("algorithm")` selects `"admm"` or `"ipm"`, and
  every other raw setting is routed to the options or to the selected algorithm's parameters
  and checked when set; setting `algorithm` itself is refused when a raw setting already stored
  does not belong to the algorithm being switched to. `MOI.optimize!` dispatches to whichever
  `setup` returns, and `MOI.BarrierIterations` reports `Solution.iter` for either.
- **ChainRulesCore's `rrule` and `frule` for `solve` work with `InteriorPoint()`.** Both call
  `adjoint_derivative`/`forward_derivative`, which now accept either workspace, so no change
  to the rules themselves was needed.
- **The interior-point method's hot path carries the same StrictMode and `--trim`
  guarantees as ADMM's.** `ipm_step!`, `ipm_residuals!` and `solve_multiplier!` are
  type-stable and allocation-free on every backend the interior-point selection reaches,
  including the sparse KKT family, where `solve_multiplier!` is this package's own code and
  keeps the guarantee even though `factorize!` there reaches foreign sparse arithmetic. Trim
  entries cover `InteriorPoint()` on its default (`FullKKT`), the KKT backend named directly,
  unscaled, with polishing, on a diagonal pair, on the sparse KKT family, on `:indirect` with a
  caller preconditioner on both a matrix pair and a `ProductOperator` pair, and through a
  `setup` → `solve!` → `update!` → `solve!` sequence and the derivatives.
- **`bench/ipm_vs_clarabel.jl`** runs `InteriorPoint()` against Clarabel, also an
  interior-point method, and against `OperatorSplitting()` at a looser tolerance, on the
  smallest instance of each OSQP suite problem class. `x` agrees with Clarabel's to `8e-11`
  relative on the tightest class and `1.8e-5`–`2.1e-5` on the loosest two (Huber and
  Portfolio); the other four classes fall between `1e-9` and `1e-6`. Results are written to
  `bench/results/ipm_vs_clarabel.json`.
- **`bench/clarabel_rs_compare.jl`** adds Clarabel.rs (crates.io `clarabel` 0.11.1, built with
  its `faer-sparse` feature and `direct_solve_method = "faer"` forced, since `"auto"` stays on
  its bundled QDLDL below faer's fill-based switch threshold at these sizes) to the same seven
  classes and tolerance as `ipm_vs_clarabel.jl`. Its solver iterations and objective match
  Clarabel.jl's on every class; its own `DefaultSolver::new` + `solve()` time (via
  `bench/clarabel_rs`, a Cargo-built driver read through a subprocess, not a package
  dependency) runs from about 0.65x to 1.1x Clarabel.jl's time on these small instances, with no
  consistent direction. Results are written to `bench/results/clarabel_rs_compare.json`; the
  comparison degrades to Julia-only when `cargo` or the crate build is unavailable.
- **`QPAlgorithm`, `QPWorkspace` and `Preconditioner` declare interface contracts**, and
  `LinearSystem`'s contract lists its optional methods too, so `TypeContracts.describe` prints
  what a new algorithm, workspace, backend or preconditioner implements. Every subtype in the
  package is checked at precompilation. The Interfaces page of the documentation states the
  same lists. `IdentityPreconditioner` and `JacobiPreconditioner` are subtypes of the new
  `Preconditioner`; a caller's preconditioner need not be one, and `setup` refuses one that has
  no `LinearAlgebra.ldiv!` method for the backend's vectors, naming the method.
  `dimensions` accepts an `InteriorPointWorkspace`.
- **`recommend_linsys(P, q, A, l, u, alg)` measures the backend choice instead of predicting
  it.** It builds every backend the pair admits, times `setup` and a bounded number of
  iterations on each, and returns a `LinsysAdvice` ranked fastest first, carrying the `linsys`
  name to pin, the times and the factor fills. `linsys = :auto` applies a rule fitted to a
  benchmark suite; this runs the experiment on the problem in hand. It is a tool for the
  caller: nothing on the solve path reaches it, and it adds no dependency.
- **`DiagonalReduced`**, the backend for a `Diagonal` `P` with a `Diagonal` `A` — a
  separable objective under box constraints. Both diagonal leaves
  `R = c D P D + σI + Ãᵀ diag(ρ) Ã` diagonal, so there is nothing to factor and a solve is
  `n` divisions. Previously such a problem took the dense default: at `n = 400` the reduced
  matrix was stored as a 1.25 MB dense array with zero off-diagonal nonzeros, then Cholesky
  factored and inverted. Setup is 17.6× faster at `n = 100` and 1710× at `n = 2000`, end to
  end 15.0× and 1216×, on identical iterates. `is_convex` gains a `Diagonal` method for the
  same reason: a diagonal matrix is positive definite exactly when its diagonal is, so the
  test no longer densifies into an `n×n` Cholesky.

  Selection is by dispatch on the pair, not by a setting or a density gate, and keyed on
  `A` rather than `P`: `Ãᵀ diag(ρ) Ã` fills in for any other `A`, so a `Diagonal` `P` with a
  general `A` still has a dense reduced matrix and correctly gets the dense backend.

- **`TridiagonalReduced`**, the same idea one band wider. Diagonal scaling preserves a
  bandwidth and squaring `A` doubles it, so `bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`
  — which is 1 for a `SymTridiagonal` `P` with a `Diagonal` `A`, a `Diagonal` `P` with a
  `Bidiagonal` `A`, or both together. `ldlt` solves that in `O(n)` and its `ldiv!` allocates
  nothing. Setup is 13.3× faster at `n = 100` and 1270× at `n = 2000`, end to end 8.4× and
  725×. `is_convex` gains a `SymTridiagonal` method that reads the `ldlt` pivots rather than
  densifying.

  The bands are computed entry by entry rather than by forming the product, because the
  arithmetic does not preserve the structure: `D P D` on a `SymTridiagonal` returns a
  `Tridiagonal`, which `cholesky` rejects as not Hermitian though it is symmetric to `1e-17`,
  and a `Diagonal` `P` with a `Bidiagonal` `A` returns a dense `Array` despite having
  bandwidth 1. An `ldlt` reports neither indefiniteness nor a zero pivot the way a Cholesky
  does — a negative pivot for the first, a throw for the second — so `factorize!` tests both
  rather than trusting the factorization to complain.

- **`PureOSQPBandedMatricesExt`**, the same rule past what LinearAlgebra can store. A
  `Tridiagonal` `A` squares to bandwidth 2, which no symmetric type in LinearAlgebra holds;
  BandedMatrices.jl holds any bandwidth and its `cholesky` is LAPACK's banded factorization,
  `O(n b²)` to factor and `O(n b)` to solve. Setup is 7.6× faster at `n = 100` and 699× at
  `n = 2000`, end to end 3.1× and 252×. A weak dependency, so without it those problems take
  the dense path as before. The backend declines twice: below bandwidth 2 the LinearAlgebra
  backends are cheaper, and at half the matrix or wider the dense factorization wins.

  A banded Cholesky reports indefiniteness through `issuccess`, so unlike the tridiagonal
  backend this one needs no separate test of the pivots.

- **`Solution.accel_declined`**, how many accelerated steps the solve discarded because they
  did worse than the plain step allows. A count close to `iter` means the accelerator is only
  adding work.
- **`update_time`**, the time spent in `update!` since the previous solve, accumulated
  across however many calls were made and counted in `run_time`. In the receding-horizon
  loop `update!` exists for, a cycle is an update followed by a solve, and that pair is
  what the caller pays; it resets once reported, so no solve carries another one's updates.
- **`SparseLDL` and `LDLKKT`**, the reduced and full-KKT backends factored by
  LDLFactorizations.jl, a pure-Julia `LDLᵀ`. A weak dependency: loading it changes which
  engine factors, nothing a caller can observe but speed, and without it the CHOLMOD
  backends serve. Its numeric factorization is 2.3–3.1× faster than CHOLMOD's on these
  problems, allocates nothing, produces identical fill, and hands back `L` and `D` as Julia
  arrays so nothing has to be extracted from a foreign factor on every refactorization.
  Only the factorization is delegated; the substitutions and the diagonal scaling stay here,
  being as fast or faster than the library's own.
- **`check_factor`**, which establishes once per factorization that every index the
  substitutions will use is in range, so those loops can run unchecked instead of
  bounds-checking a row index read out of the factor on every nonzero. Removing that check
  measured 1.18× (Lasso), 1.23× (Huber) and 1.32× (Portfolio) faster on the substitution
  pair, bit-identical results — the single largest item in closing the OSQP benchmark suite
  gap, taking the Portfolio class from 0.88× of libosqp to 1.00×.
- **`check_storage`**, the same guarantee for the sparse equilibration traversals: it
  establishes once from `validate` that a row index read out of `P` or `A` is in range, so
  the four Ruiz sweeps run unchecked. On the OSQP suite's Eq QP class, whose `P` holds
  39 638 entries in a 200×200 matrix, a sweep went from 144.5 µs to 18.7 µs (7.7×), and the
  class's setup from 0.73× of libosqp to 1.49×.
- **`ReducedGram`**, a stored map from each contribution to its slot in the reduced matrix,
  so a refactorization refills values in one allocation-free pass instead of rebuilding
  through four sparse products.
- **`SparseKKT`**, a sparse `LDLᵀ` factorization of the full `(n+m)×(n+m)` quasi-definite
  system, selected when a dense row in `A` would densify the reduced matrix.
- **`SparseCholmod`** and **`SparseFormedInverse`**, the reduced backends for a sparse `A`:
  one forms the reduced matrix from stored entries and factors it with CHOLMOD, the other
  forms it sparsely and factors it densely.
- **`IndirectCG`**, a matrix-free preconditioned-CG backend over Krylov.jl.
- **GPU arrays**, through the matrix-free backend only, with JLArrays as the correctness
  gate.
- **Solution derivatives**, by implicit differentiation of the KKT conditions.
- **A MathOptInterface wrapper**, as a package extension.
- **`linsys = :dense`**, to overrule the representation gates.
- **`bench/osqp_suite.jl`**, the OSQP benchmark suite's seven problem classes.

### Changed

- **The two CHOLMOD backends gather their factor instead of rebuilding it.** A
  refactorization runs every outer iteration of the interior-point method and every time ADMM
  retunes `ρ`, and `SparseKKT` and `SparseCholmod` each ended one by asking CHOLMOD for
  `sparse(F.LD)` or `sparse(F.L)`, which copies the whole factor inside CHOLMOD, converts the
  copy, and allocates three fresh arrays from it. A simplicial factor already holds its
  values column by column, so `factor_csc!` reads them into the buffers the backend already
  owns in one pass, bounding the row indices as it goes; `D⁻¹` is read from the diagonal each
  column stores first, the transposed factor is refreshed through `transpose!`, and the
  ordering is reread only when a new symbolic analysis is done, which is the only thing that
  moves it. A supernodal factor, which stores no column-wise pattern, still goes through
  CHOLMOD's conversion. Minima on one core with one BLAS thread, on the classes
  `bench/clarabel_rs_compare.jl` times: Random QP (`n = 6`, `m = 60`, `sparse_kkt`)
  `factorize!` 5.46 → 3.43 µs and 70 → 30 allocations, whole solve 133.2 → 110.5 µs and
  993 → 593 allocations; Lasso (`n = m = 204`, `cholmod`) `factorize!` 12.08 → 10.03 µs and
  61 → 28 allocations, solve 298.6 → 281.5 µs; Huber (`n = 602`, `m = 600`, `cholmod`)
  `factorize!` 40.85 → 39.31 µs and 71 → 31 allocations, solve 1251.6 → 1197.9 µs. What a
  refactorization saves is 1.5–2 µs across all three, so its share is largest on the smallest
  problem. The factors are bit-identical to what the conversion returned: the 41-case
  snapshot matches unchanged and iteration counts do not move.
- **Every selection point answers a third algorithm, by serving it or by naming what it
  must define.** A new `SelectionFor` subtype used to reach a `MethodError` from inside
  `select_backend`, `dense_rung`, `indirect_rung`, `kronecker_rung`, `formed_rung`, the
  SparseArrays extension's `sparse_form` and the GPU extension's refusals. The rungs whose
  default is to decline now take any subtype, as `kkt_rung` and `block_rung` already did, and
  the GPU refusal is generic. The four points with no algorithm-independent answer —
  `select_backend`, `dense_rung`, `indirect_rung` and `sparse_form` — throw through
  `PureOSQP.refuse_selection`, naming themselves. `test/contract_tests.jl` asserts the whole
  set against a dummy subtype.
- **The matrix-free backend refuses to factorize without its settings.** `adopt_settings!`'s
  default does nothing, which is right for a backend that reads no settings and was silently
  wrong for `IndirectCG`: under an algorithm with no `adopt_settings!` method of its own it
  kept `cg_max_iter = 0` and every solve returned its starting point. `factorize!` now throws
  and names `adopt_settings!`. `Options` refuses a `cg_max_iter` of zero, so the check cannot
  fire on a legitimate setting, and it is off the per-iteration path.
- **One `named_backend` serves both algorithms' named `linsys` kinds.** `setup_backend`
  carried the same `Val{LS}` chain twice, once per algorithm. The refusals, the messages and
  the two-stage rule for `:sparse`, `:block` and `:lowrank` are unchanged; `:sparse`'s message
  still differs by algorithm, because what the two chains require of `P` differs.
- **`recommend_linsys` ranks by what a whole solve costs**, `setup_ms + iterate_ms *
  solve_iters`, rather than by cost per iteration alone. One unbounded run on `linsys = :auto`
  measures `solve_iters`, and `LinsysAdvice` reports it beside the new `total_ms` column and
  the existing `ms/iter` one. Per-iteration ranking treats setup as free, which decides
  against a factorization that is slower to build and faster to solve against — the right
  choice over four thousand ADMM iterations and the wrong one over twenty interior-point
  iterations.
- **`InteriorPoint()` is concretely typed.** The seed was
  `InteriorPoint{Union{Nothing,Float64},Union{Nothing,Int}}`, an abstract type leaking into
  `show` and into every error message naming it. `reg_primal`, `reg_dual` and `refine_iter`
  now carry a type parameter each, so the seed is `InteriorPoint{Float64,Nothing,Nothing,
  Nothing}`, `InteriorPoint(reg_primal = 1e-7)` is `InteriorPoint{Float64,Float64,Nothing,
  Nothing}`, and the workspace holds `InteriorPoint{T,T,T,Int}`. `OperatorSplitting` has no
  parameter it defaults from `T` and was already `OperatorSplitting{Float64}`.
- **Files moved to the directory their contents belong to.** `src/core/update.jl` is
  `src/admm/update.jl`, keeping ADMM's `update!`; the `Problem`-level `validate_update!` and
  `adopt_update!` it also held are in `src/core/problem.jl`. The generic
  `update_settings!(::QPWorkspace)` moved from `src/admm/api.jl` to `src/core/options.jl`.
  The parts of `src/admm/termination.jl` the interior-point method calls — `eps_prim`,
  `eps_dual`, `eps_duality_gap`, `is_primal_infeasible`, `is_dual_infeasible`, `gap_terms` and
  the norm and recession-cone helpers they are built from — are in a new
  `src/core/termination.jl`.
- **`linsys = :auto` chooses from the sparsity pattern, and never factors a matrix to
  decide.** For a `SparseMatrixCSC` pair the choice among the sparse KKT form, the sparse
  reduced form and the algorithm's terminal now reads the densest row of `A`, `Σᵢ nnzᵢ²`, the
  stored entries of the KKT matrix, the symbolic `AᵀA ∪ P` pattern count, `n` and `m`. The two
  gates it replaces both cost something they then threw away: one factored a trial matrix and
  read its fill, the other compared `nnz(A)/mn` against a measured density. The rule is fitted
  to `bench/results/ipm_selection.json`, which times every backend each of 71 problems admits
  under both algorithms: the backend it picks is within 1.3× of the fastest measured one on 69
  of the 71 under `InteriorPoint` (worst case 1.52×) and within 1.03× on all 71 under
  `OperatorSplitting`. Both figures are in-sample, and the `OperatorSplitting` comparison is
  per iteration, since 10 of the 71 reach the iteration cap at the sweep's tolerance. Under `InteriorPoint` the benchmark suite's Random QP class goes from
  the dense KKT factorization to the sparse reduced one (22.5 ms to 1.0 ms) and Control from
  the dense KKT factorization to the sparse KKT one (56.8 ms to 1.9 ms); Lasso, SVM and Huber
  move from the sparse KKT form to the sparse reduced one. Under `OperatorSplitting` the
  Random QP and Eq QP classes accumulate the reduced matrix from the stored entries rather
  than forming it with a dense product, which is the same arithmetic without the `m×n` buffer,
  and every other class keeps the backend it had. `linsys = :sparse`, `:kkt` and `:dense`
  remain unconditional overrides. The `56.8 ms to 1.9 ms` figure above uses the sparse KKT
  backend and predates the CHOLMOD factor-read change below, which cut that backend's
  `factorize!` cost; it has not been re-measured since, so read it as a lower bound on the
  margin rather than the current number.
- **The algorithm is an object, and the settings are split into its parameters and shared
  options.** `solve(P, q, A, l, u, alg; kwargs...)` and `setup` take the algorithm as an
  optional sixth argument: `OperatorSplitting(; rho, sigma, alpha, adaptive_rho, …,
  profile_primdual)`, the default, or `InteriorPoint(; reg_primal, reg_dual, max_reg_bumps,
  refine_iter, step_fraction, cg_fail_limit)`. The keyword arguments are the fields of
  `Options`, which both algorithms read (`max_iter`, `time_limit`, the tolerances, `scaling`,
  `check_termination`, `check_dualgap`, `scaled_termination`, `warm_starting`, `linsys`,
  `polishing`, `polish_refine_iter`, `delta`, `cg_max_iter`, `cg_tol_fraction`, `verbose`), with
  defaults that depend on the algorithm, reported by `default_options(alg, T)`. A workspace
  holds `ws.algorithm` and `ws.options` in the solve's element type, and
  `update_settings!(ws; kwargs...)` changes options while `update_settings!(ws, alg)` replaces
  the parameters. A parameter passed as a keyword throws an `ArgumentError` naming the
  algorithm it belongs to. The workspaces are `OperatorSplittingWorkspace` and
  `InteriorPointWorkspace`, subtypes of `QPWorkspace`. The `algorithm` keyword and the
  `Settings`, `IPMSettings`, `Workspace` and `IPMWorkspace` names are removed.
  `cg_tol_fraction` must lie in `(0, 1]` under either algorithm, and `verbose` is an option
  both algorithms accept: `InteriorPoint()` prints its own per-iteration report (see
  "Watching a solve" in the docs).
- **The `polish` setting is now `polishing`**, the name libosqp 1.0 uses. This is a breaking
  change: passing `polish = true` throws a `MethodError` for the unsupported keyword.
- **The matrix-free backend warm-starts CG and halves its tolerance when CG stops moving.**
  Starting each CG solve from the previous `x̃` cut CG iterations 2.5× to 60×, but late in a
  solve the starting point already met the tolerance, CG took no step, and ADMM stalled. The
  tolerance is now halved after `cg_tol_reduction` such solves in a row, which is what that
  setting means in libosqp, and its floor is `eps(T)` relative to the right-hand side instead of
  a fixed `sqrt(eps)`. A problem that ran to `max_iter` at `eps = 1e-9` now converges, and the
  matrix-free backend now solves the `κ = 1e4` case in the conditioning benchmark.
- **`setup` throws when the backend it chose cannot factorize the problem.** The error names
  `linsys = :kkt` as the fix. Before, `setup` switched to the full KKT backend without saying so,
  while a later refactorization of the same problem threw.
- **The MathOptInterface wrapper checks a raw setting when it is set.** A bad value throws
  from `MOI.set` instead of from `optimize!`. A setting that takes a symbol also accepts a
  string, so `"kkt"` works for `linsys`. Reading a setting that was never set returns its
  default instead of throwing.
- **The GPU tests are excluded from the test run.** They are tagged `:gpu` in
  `test/gpu_tests.jl`, and `test/runtests.jl` filters them out: compiling the JLArrays path
  crashes Julia 1.13.0 inside LLVM on AVX-512 targets.
- **Benchmarks compare against libosqp 1.0** through `bench/osqp_v1.jl`, with
  `check_dualgap` off on both sides. `bench/update_bench.jl` gains a libosqp column that uses
  `osqp_update_data_vec`. `bench/headtohead_v1.jl` is removed; `bench/headtohead.jl` reports the
  same agreement figures.

### Fixed

- **A named `linsys` is honored whenever the representation allows it**, under either
  algorithm. Before, `linsys = :sparse` still ran the `:auto` ladder's fill and density gates
  internally and could throw on a pair with a perfectly good sparse representation — a sparse
  `A` whose reduced or KKT factor simply came out denser than the ladder's threshold, or one
  the ladder never reached because a cheaper rung answered first. `:block` and `:lowrank` had
  the same problem: `:block` declined a single-block pair and `:lowrank` declined a coupling
  wide enough that the correction stopped paying, both cost decisions rather than
  representation ones. A named backend now tries the same rungs `:auto` would, at the same
  threshold, first — which is what reaches the same backend `:auto` reaches wherever the gate
  already passes — and only when every one of those declines retries with the gate disabled,
  which is what reaches a backend for a pair `:auto` would otherwise send to the dense
  terminal. It refuses only when it genuinely cannot serve the pair at either stage (wrong
  matrix type, a missing weak dependency, or a factorization that does not succeed at the
  current regularization) — never because a cheaper backend was measured to win. `:auto`'s
  own choices are unchanged: the gates still decide which rung the ladder reaches on its own.
- The infeasibility certificates test the sign strictly, as libosqp 1.0 does. A tolerance there
  certified feasible problems as infeasible.
- A `NaN` residual ends the solve as `NON_CONVEX` instead of running to `max_iter` with a point
  that `has_solution` accepts.
- A refused `update!` leaves the workspace unchanged. Before, a valid argument passed alongside
  an invalid one was applied first.
- The derivative is refused unless the last solve reached `SOLVED`.
- A one-sided constraint row no longer makes every row look weakly active, which refused the
  derivative of any problem with an infinite bound.
- A refactorization drops the accelerator's history.
- `solve` throws when `x0` or `y0` is passed with `warm_starting = false`, since the seed would
  be discarded.
- A failed factorization on the full KKT backend no longer suggests switching to it.
- A non-finite Kronecker factor is reported by factor, and checked without reading the product.
- `setup` and `update!` check a `ProductOperator` `P` declared symmetric by comparing
  `dot(v, P*w)` with `dot(P*v, w)`, and throw if they differ. A false declaration had been
  accepted.
- The finiteness check in `setup` reads a `SparseMatrixCSC`'s stored entries only. It had read
  every position, one search each, which made `setup` 13× slower on the suite's Huber class
  (6.9 ms against 0.52 ms) and the whole solve 3.4× slower.
- `polish_kernel!` no longer allocates three zero vectors on a path it declines: it returns
  the caller's own `x`, `y`, `z` instead, which the caller already knows not to use.
- `mul_At!` no longer allocates on every iteration for a `Tridiagonal` `A`. `mul!` against a
  `Tridiagonal`'s adjoint allocates 48 bytes in LinearAlgebra, which cost `admm_step!`,
  `update_residuals!` and `solve_system!` their no-allocation guarantee for any backend
  holding one. It now walks the three bands directly.
