# API

```@docs
PureQPBase.solve
PureQPBase.setup
PureQPBase.solve!
PureQPBase.update!
PureQPBase.warm_start!
PureQPBase.cold_start!
PureQPBase.update_settings!
PureQPBase.update_rho!
PureQPBase.dimensions
PureQPBase.capabilities
PureQPBase.constraint_violation
PureQPBase.constraint_violation!
PureOSQP.Optimizer
PureQPBase.adjoint_derivative
PureQPBase.forward_derivative
PureQPBase.Solution
Base.copy(::PureQPBase.Solution)
Base.copyto!(::PureQPBase.Solution{T}, ::PureQPBase.Solution{T}) where {T}
```

## Algorithms and options

```@docs
PureQPBase.QPAlgorithm
PureOSQP.OperatorSplitting
PureIPM.InteriorPoint
PureDAQP.ActiveSet
PureDAQP.faster_scan
PureQPBase.Options
PureQPBase.LINSYS_OPTIONS
PureQPBase.default_options
PureQPBase.QPWorkspace
PureOSQP.OperatorSplittingWorkspace
PureIPM.InteriorPointWorkspace
PureDAQP.ActiveSetWorkspace
PureQPBase.conforms
```

## Status values

```@docs
PureQPBase.Status
```

## Settings

[`OperatorSplitting`](@ref) documents its parameters and [`Options`](@ref) the options. This section shows how both correspond to the reference implementation's settings; under `OperatorSplitting` every setting in the right-hand column is an `Options` keyword except `profile_primdual`, `rho`, `rho_is_vec`, `sigma`, `alpha`, `cg_tol_reduction` and the four `adaptive_rho` settings, which are `OperatorSplitting` parameters.

The defaults in the "here" column are `OperatorSplitting`'s. [`InteriorPoint`](@ref) has no
upstream counterpart, so it is not in this table; its own `Options` defaults — a tighter
`max_iter`, `eps_abs`, `eps_rel`, a `check_termination` of `1`, and different `cg_max_iter`
and `cg_tol_fraction` — are in [`default_options`](@ref) and in
[Choosing an algorithm](@ref).

| upstream | default | here | default | note |
|---|---|---|---|---|
| `device` | `0` | — | | GPU device selection; no counterpart |
| `osqp_linsys_solver_type` | direct | `linsys` | `:auto` | upstream picks a *library*, this picks a *formulation* — see [Algebra backends](@ref) |
| `allocate_solution` | `true` | — | | an embedded-allocation concern; no counterpart |
| `verbose` | `true` | `verbose` | `false` | a library that prints by default is the wrong default for a package; `OperatorSplitting` and `InteriorPoint` read it, `ActiveSet` keeps no iteration log |
| `profiler_level` | `0` | `profile_primdual` | `false` | one switch over the one measurement that needs a clock |
| `warm_starting` | `true` | `warm_starting` | `true` | |
| `scaling` | `10` | `scaling` | `10` | |
| `polishing` | `false` | `polishing` | `false` | |
| `rho` | `0.1` | `rho` | `0.1` | |
| `rho_is_vec` | `true` | `rho_is_vec` | `true` | |
| `sigma` | `1e-6` | `sigma` | `1e-6` | |
| `alpha` | `1.6` | `alpha` | `1.6` | |
| `cg_max_iter` | `20` | `cg_max_iter` | `20` | |
| `cg_tol_reduction` | `10` | `cg_tol_reduction` | `10` | |
| `cg_tol_fraction` | `0.15` | `cg_tol_fraction` | `0.15` | |
| `cg_precond` | diagonal | — | | always diagonal here; nothing to select |
| `adaptive_rho` | `true` | `adaptive_rho` | `:iterations` | a mode, not a flag; `true` is accepted and means `:iterations` |
| `adaptive_rho_interval` | `50` | `adaptive_rho_interval` | `50` | the published table says `0`; the library ships `50` |
| `adaptive_rho_fraction` | `0.4` | `adaptive_rho_fraction` | `0.4` | **same name, different meaning** — see below |
| `adaptive_rho_tolerance` | `5` | `adaptive_rho_tolerance` | `5` | |
| `max_iter` | `4000` | `max_iter` | `4000` | |
| `eps_abs` | `1e-3` | `eps_abs` | `1e-3` | |
| `eps_rel` | `1e-3` | `eps_rel` | `1e-3` | |
| `eps_prim_inf` | `1e-4` | `eps_prim_inf` | `1e-4` | |
| `eps_dual_inf` | `1e-4` | `eps_dual_inf` | `1e-4` | |
| `scaled_termination` | `false` | `scaled_termination` | `false` | |
| `check_termination` | `25` | `check_termination` | `25` | |
| `check_dualgap` | `true` | `check_dualgap` | `true` | in the C header and the library's defaults, absent from the published table |
| `time_limit` | `1e10` | `time_limit` | `Inf` | the same "no limit", spelled as the thing it means |
| `delta` | `1e-6` | `delta` | `1e-6` | |
| `polish_refine_iter` | `3` | `polish_refine_iter` | `3` | |

Of the thirty-one settings: twenty-two match upstream, three have no counterpart, five are renamed or defaulted differently, and one is a trap.

**Same name, different meaning — the one to watch.** `adaptive_rho_fraction` means different things in upstream and here. Upstream it is a fraction of *setup time*. Here it is a fraction of the *previous KKT error*. Under `adaptive_rho = :kkt_error`, `ρ` is retuned only when the relative KKT error falls to `adaptive_rho_fraction` of its previous value. Porting tuned values without reading this will cause quiet errors.

**Upstream only.** `device` and `allocate_solution` are for GPU and embedded systems. `cg_precond` always uses the diagonal preconditioner.

## Algebra backends

The reference implementation selects an *algebra* (`builtin` for CPU, `mkl` for Intel, `cuda` for NVIDIA GPUs).

In this package, the library is determined by what is loaded in the Julia session.

The `linsys` setting selects the *formulation* of the linear system:

| | selects | values |
|---|---|---|
| upstream algebra | which library does the arithmetic | `builtin`, `mkl`, `cuda` |
| `linsys` here | which system is formed and how it is factored | ten values, listed under [What each `linsys` value means](@ref) |

## Internals

```@docs
PureQPBase.equilibrate!
PureQPBase.mul_A!
PureQPBase.mul_At!
PureQPBase.mul_P!
PureQPBase.refactor!
PureQPBase.refactor_rho!
PureQPBase.factorize!
PureQPBase.refactor_weights!
PureQPBase.assemble_kkt0!
PureQPBase.solve_system!
PureQPBase.solve_multiplier!
PureQPBase.element_typed
PureOSQP.admm_step!
PureOSQP.set_rho_vec!
PureOSQP.adapt_rho!
PureOSQP.update_residuals!
PureQPBase.gap_terms
PureQPBase.eps_prim
PureQPBase.eps_dual
PureQPBase.eps_duality_gap
PureQPBase.check_termination
PureQPBase.is_primal_infeasible
PureQPBase.is_dual_infeasible
PureQPBase.residuals_at!
PureQPBase.polish_kernel!
PureQPBase.polish!
PureQPBase.active_kkt
PureQPBase.derivative_ready
PureIPM.ipm_step!
PureIPM.ipm_residuals!
PureIPM.weights!
PureIPM.factorize_newton!
PureIPM.primal_certificate!
PureIPM.dual_certificate!
PureIPM.stalled!
PureIPM.iterate_bound
PureQPBase.has_solution
PureQPBase.status_name
PureQPBase.PolishStatus
```

## Linear-system backends

```@docs
PureQPBase.LinearSystem
PureQPBase.QPData
PureQPBase.Problem
PureQPBase.SystemWeights
PureQPBase.check_update
PureQPBase.set_tolerance_level!
PureQPBase.adopt_settings!
PureQPBase.set_refresh_index!
PureQPBase.use_residual_stop!
PureQPBase.last_solve_converged
PureQPBase.inner_iterations
PureQPBase.update_preconditioner!
PureQPBase.Preconditioner
PureQPBase.check_preconditioner
PureQPBase.IdentityPreconditioner
PureQPBase.JacobiPreconditioner
PureQPBase.KroneckerPreconditioner
PureQPBase.choose_backend
PureQPBase.ReducedInverse
PureQPBase.ReducedCholesky
PureQPBase.ProductReduced
PureQPBase.DiagonalReduced
PureQPBase.TridiagonalReduced
PureQPBase.DiagonalLowRank
PureQPBase.BlockReduced
PureQPBase.KroneckerReduced
PureQPBase.FullKKT
PureQPBase.indirect_backend
```

## Structured inputs

```@docs
PureQPBase.RowCoupled
PureQPBase.coupling_rank
PureQPBase.BlockDiagonal
PureQPBase.KroneckerOperator
PureQPBase.StackedOperator
PureQPBase.ComposedOperator
PureQPBase.SumOperator
PureQPBase.parts
PureQPBase.nterms
PureQPBase.add_reduced_term!
PureQPBase.factors
PureQPBase.has_cholesky_factor
PureQPBase.cholesky_factor
PureQPBase.CholeskyFactor
PureQPBase.KroneckerCholesky
PureQPBase.KroneckerSquareRoot
PureQPBase.scalar_diagonal
PureQPBase.dense_row!
PureQPBase.is_scalar_multiple
PureQPBase.scalar_multiple
PureQPBase.nblocks
PureQPBase.rowrange
PureQPBase.colrange
PureQPBase.structural_rows
PureQPBase.is_convex
PureQPBase.is_symmetric
PureQPBase.is_materializable
PureQPBase.holds_structure
PureQPBase.reduced_diagonal!
PureQPBase.reduced_rhs!
PureQPBase.ProductOperator
PureQPBase.unpreconditioned!
PureQPBase.probe_column!
PureQPBase.no_entries
PureQPBase.check_symmetric_products
PureQPBase.CoupledRows
PureQPBase.divide!
PureIPM.caller_preconditioner
```

## Backend selection

```@docs
PureQPBase.SelectionFor
PureQPBase.ADMMSelection
PureQPBase.IPMSelection
PureQPBase.refuse_selection
PureQPBase.named_backend
PureQPBase.sparse_refusal
PureQPBase.select_backend
PureQPBase.kkt_rung
PureQPBase.reduced_rung
PureQPBase.block_rung
PureQPBase.kronecker_rung
PureQPBase.lowrank_rung
PureQPBase.formed_rung
PureQPBase.dense_rung
PureQPBase.indirect_rung
```

## Backend introspection

```@docs
PureQPBase.backend_name
PureQPBase.backend_info
PureQPBase.factor_fill
PureQPBase.BackendInfo
```

## Measuring the backend choice

```@docs
PureQPBase.recommend_linsys
PureQPBase.LinsysAdvice
PureQPBase.measure_linsys
PureQPBase.solve_iterations
```
