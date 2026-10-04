# PureOSQP.jl: shared matrix support for more than one QP algorithm

**Every path in this document is from the single-package layout it was written against.**
`src/`, `test/` and `bench/` are now split across `PureQPBase`, `PureOSQP` and `PureIPM`;
`three-package-split.md` records where each part landed.

Design for steps 1 and 2 of the agreed strategy (internal boundary, then a Mehrotra IPM in the
same package), with the step-3 package layout sketched. Every file:line below was read; claims
that could not be checked by reading are marked **unverified**; claims that rest on the
literature rather than on this code are marked **literature**. Claims marked **measured** come
from the two bench-only spikes `bench/ipm_matrixfree_spike.jl` (27 dense instances, `n ∈ {200,
500, 1000}`, results `bench/results/ipm_matrixfree_spike_summary.json`) and
`bench/ipm_matrixfree_spike2.jl` (18 dense, 36 Kronecker, 30 sparse instances, `n ≤ 200`,
results `bench/results/ipm_matrixfree_spike2_nog2stop.json`; three dense `n = 1000` instances
in `ipm_matrixfree_spike2.json`), and `bench/ipm_rowtypes_spike.jl` (spike 3: spike 2's 18 dense
and 30 sparse instances with equality, one-sided and free rows, inner stopping rules, CG start,
and no-pivoting `LDLᵀ`; results `bench/results/ipm_rowtypes_spike.json`). All spikes ran at
`n ≤ 1000` on neuromancer and are indicative. Choices the user has taken or must confirm are in
§10.

Conventions: `ws` is the ADMM workspace, `OperatorSplittingWorkspace`; `prob` is the shared
problem object introduced here; `wt` is the weights object. Line numbers in §1 refer to the
tree before step S2a; the settings are named as §11 defines them. In prose, `:admm` and `:ipm`
name the two algorithms, `OperatorSplitting()` and `InteriorPoint()` (§11), as the MOI
attribute `algorithm` does.

---

## 1. Coupling inventory

What the backends, the selection ladder, equilibration, `update!`, polishing and the derivatives
read off the ADMM `OperatorSplittingWorkspace`.

### 1.1 The `OperatorSplittingWorkspace` itself (`src/types.jl:294-378`)

| group | fields | lines |
|---|---|---|
| problem data + equilibration (algorithm-neutral) | `P A n m q0 l0 u0 q l u D E c` | 299-311 |
| scratch used by products and backends (neutral) | `tmp_n tmp_m work_n work_m` | 326-329 |
| ADMM weights | `rho rho_vec rho_inv_vec constr_type` | 330-333 |
| backend + accelerator | `linsys accel refactor_count` | 334-338 |
| ADMM iterates | `x y z x_prev z_prev xtilde ztilde delta_x delta_y Ax Px Aty rhs_x rhs_z` | 312-325 |
| ADMM run state (residuals, gap terms, primdual integral, counters, timing, status) | 339-376 |
| the settings, now `algorithm::OperatorSplitting{T}` and `options::Options{T}` (`sigma` and `scaling` are read by neutral code) | 377 |

`setup_backend` (`types.jl:541-698`) builds all of it in one constructor call (579-597) and is
the only place the ladder is reached (617-619, 635, 649, 654, 666, 678, 692).
`Base.show(::Workspace)` reads `n m rho` (`:399-407`); `factor_fill(ws)` reads `ws.n`
(`linsys.jl:57`); `dimensions` reads `n m` (`api.jl:79`).

### 1.2 Backends: every read of ADMM state

All backends take `ws` and read `n m P A D E c` (neutral) and `rho_vec | rho_inv_vec` +
`settings.sigma` (ADMM). `solve_system!` writes `ws.xtilde`/`ws.ztilde` and uses
`ws.work_n`/`ws.work_m` through `reduced_rhs!` and `mul_A!`.

| backend | `factorize!` reads ρ/σ | `solve_system!` reads ρ | other |
|---|---|---|---|
| `ReducedCholesky` | `linsys.jl:573`, `:593` | via `reduced_rhs!` `:697` | writes `xtilde` `:713`, `ztilde` `:714`; binds locals at entry `:573` |
| `DiagonalReduced` | `:604` | `:719` | |
| `TridiagonalReduced` | `:623` | `:726` | |
| `FullKKT` | `:671` (`sigma`), `:679` (`rho_inv_vec`) | `:748` | reads `ws.D[j]`, `ws.c` per entry `:666-679` (no local binding); reads `A[i, j]` entry by entry at every factorization |
| `BlockReduced` | `block.jl:76` | `:107` | `colrange(ws.A, i)` `:111` |
| `DiagonalLowRank` | `lowrank.jl:114`, `:98` | `:157` | `refactor_rho!` override `:154` — the only one |
| `KroneckerReduced` | `kronsolve.jl:93-94` | `:111` | `mu` set by rung `:88`, refreshed by `update!` `update.jl:161` |
| `SparseFormedInverse` (ext) | `SparseArraysExt.jl:283,291` | ReducedInverse method | inverts `:293-295` |
| `SparseKKT` (ext) | `:487,492` | `:570` | allocates `sparse(LD)`, `diag`, `inv.(d)`, `perm` every call `:496-504` |
| `SparseCholmod` (ext) | `:982,987` | `:1095` | |
| `SparseLDL` (LDL ext) | `LDLFactorizationsExt.jl:147,150` | `:207` | allocates `fact_L/fact_perm` `:155-156`; reaches SparseArrays ext by `Base.get_extension` `:142` |
| `LDLKKT` (LDL ext) | `:286,289` | `:328` | allocates `:294-295` |
| `BandedReduced` (Banded ext) | `BandedMatricesExt.jl:143` | `:175` | |
| `IndirectCG` (Krylov ext) | `KrylovExt.jl:107` | `:132`; **also** `:139-142` `settings.cg_tol_fraction`, `ws.scaled_prim_res/dual_res`; `:147` warm start from `ws.xtilde`; `:151,154` `cg_max_iter`, `cg_tol_reduction` | `ReducedOperator` holds `ws` `:30-37`, reads `rho_vec` `:49`, `sigma` `:55`, `work_m/work_n` `:45-56`; preconditioner is `Diagonal(ls.prec)` `:150`, identity for any `ProductOperator` operand (`operator.jl:220-228`) |

`refactor_rho!(ls, ws)` default `linsys.jl:100`; `refactor!(ws)`/`refactor_rho!(ws)`
`:764-777`; `refactored!` `:779-798` counts and resets the accelerator (`:796`).
`reduced_diagonal!` (`scaling.jl:323-337`, GPU override `GPUArraysCoreExt.jl:111-118`,
operator overrides `operator.jl:222-228`) already takes `rho, E, D, sigma, c`.

### 1.3 Selection ladder

Every rung and `choose_backend` carries `(P, A, proto, n, m, D, E, c, rho_vec, sigma)`:
`choose_backend` `linsys.jl:355-357, 505-525`; `select_backend` `:407-427`; rungs `:437-503`;
`kronecker_rung` `kronsolve.jl:70-89`; `block_rung` `block.jl:57-72`; `lowrank_rung`
`lowrank.jl:71-87`; extensions `SparseArraysExt.jl:151-186, 588-615, 1146-1181`,
`BandedMatricesExt.jl:89-127`, `GPUArraysCoreExt.jl:44-53`, `linsys.jl:831,843` →
`LDLFactorizationsExt.jl:88-106, 258-277`.

ADMM-specific decisions inside rungs: `kronecker_rung` requires uniform `rho_vec`
(`kronsolve.jl:82-83`) and identity scaling (`:86`); the sparse KKT and reduced rungs require
`P isa SparseMatrixCSC`, so a dense `P` with a sparse `A` lands on `formed_rung` →
`SparseFormedInverse`; `ReducedInverse` inverts on the solve-many assumption
(`linsys.jl:597`, `block.jl:101`, `SparseArraysExt.jl:295`). For a `SparseMatrixCSC` pair the
choice among the KKT form, the reduced form and the algorithm's terminal is `sparse_form`
(`SparseArraysExt.jl`), which reads the sparsity pattern and has one method per selection tag;
its constants are fitted to `bench/results/ipm_selection.json`. The gates that remain
algorithm-independent because the accepted region wins on both factorization and solve are
`10k ≤ n` (`lowrank.jl:78-85`) and `b ≤ n/4` (`BandedMatricesExt.jl:102-118`).
The `select_backend` docstring's claim that a failed `factorize!` is "rebuilt on `FullKKT`"
(`linsys.jl:398-401`) is stale: `setup_backend:696` throws through `refactored!`.

### 1.4 Equilibration and products

`equilibrate!` (`scaling.jl:193-240`) takes arrays. `mul_A!` `:250-258`, `mul_At!`
`:267-272` and its `Tridiagonal` specialization `:278-296`, `mul_P!` `:305-310` read
`ws.A/P/D/E/c` and scratch `ws.tmp_n/tmp_m`. `warm_start!` reads `D E c`
(`types.jl:715,720`).

### 1.5 `update!` (`src/update.jl:35-184`)

Neutral: `:43-53, 56-57, 67-77, 106-153` (validation), `:159-166` (adoption; nothing is
written before every check passed, `:155-158`), `:169-172` (rescale). ADMM: `is_convex(T,
P, ws.algorithm.sigma)` `:54`; `set_rho_vec!` `:175`; `refactor!` `:178`; uniform-ρ class
guard `:119-135`. Backend invariants through `ws.linsys isa …`: `KroneckerReduced`
`:58, 161`, `DiagonalLowRank` `:78`, `BlockReduced` `:87-105`.

### 1.6 Polishing, residuals, termination

`polish!` (`polish.jl:43-134`) reads `z y l u q D E c P A` (`:51-72`), `settings.delta`
`:47`, `polish_refine_iter` `:98`, `prim_res/dual_res` `:122-124`, calls `residuals_at`
(`:7-26`: reads `settings.scaling`, `E D c`, and uses `ws.work_m`, `ws.Px`, `ws.work_n`,
`ws.Aty` as scratch `:13-24`) and `update_residuals!` `:132`. `update_residuals!`
(`termination.jl:34-85`) writes `Ax Px Aty` `:38,46,49` and reads `settings.scaling` `:36`,
`E D c` `:41,56,79`; `eps_prim/eps_dual` read `Ax Aty Px` back `:119,129`;
`eps_duality_gap` `:142-148`; `is_primal_infeasible` `:207-221` **overwrites `delta_y` with
its projection** `:209`; `is_dual_infeasible` `:246-262`; `check_termination` `:272-300`.
`build_solution` (`admm.jl:282-309`) reads `delta_y/delta_x` `:286,293` and unscales with
`D E c` `:304-305`. `constraint_violation!` (`api.jl:115-129`) reads `settings.scaling`
`:121`, `E` `:126`.

### 1.7 Derivatives (`src/derivative.jl`)

`active_kkt` `:29-115` reads `ws.status` `:33`, `x y z` and `D E c` `:48-50`, `l0 u0` `:56`,
`P A` `:100-105`; the active-set threshold is `τ = sqrt(eps)·max(‖y‖∞, 1)` `:52,64`. The
ChainRules ext (`ChainRulesCoreExt.jl:35-46`) calls `setup(…; polishing = true)` + `solve!`.

### 1.8 Accelerator, verbose, MOI, settings

`pack_fixed_point!`/`unpack_fixed_point!` (`src/accelerate.jl:62-85`) read `rho_inv_vec`
`:65` and `rho_vec` `:83`; the COSMO extension reads no weight field itself
(`COSMOAcceleratorsExt.jl:100-133`) and calls `admm_step!` `:130`. `print_header`
(`admm.jl:63-83`) reads `backend_name(ws.linsys)`. `update_settings!` (`api.jl:31-56`)
accepts a new `sigma` and refactorizes `:48-53` (exercised by the trim entry
`settings_and_rho`, `test/trim/entrypoints.jl:124-132`). MOI `optimize!`
(`MathOptInterfaceExt.jl:169-177`); `ResultCount` returns 1 for every status but
`NON_CONVEX` `:203-206`.

### 1.9 Readers outside `src/` and `ext/`

- Tests: `ws.rho_vec` `test/solve_tests.jl:84-88,182,188,409-411,533-535`,
  `test/linsys_tests.jl:10,37,435,486,491,538`, `test/banded_tests.jl:15,20`;
  `ws.algorithm.sigma` same lines; `ws.xtilde/ztilde` `linsys_tests.jl:14-15,39,437-438,
  493-494,540-541,595`, `banded_tests.jl:22-23`; `ws.constr_type` `setup_tests.jl:124`,
  `update_tests.jl:40`, `solve_tests.jl:88,411`; `ws.D/E/c` `scaling_tests.jl:12-59`,
  `block_tests.jl:33-34`.
- Direct backend/rung calls: `test/linsys_tests.jl:12,34,270-271,434,490,537,571-573,594,
  601,603,658,712-714`, `test/selection_tests.jl:112-124,223`, `test/block_tests.jl:59`,
  `test/indirect_tests.jl:30-31,94-95`, `test/banded_tests.jl:19`,
  `test/coverage_tests.jl:24`; bench: `rho_update.jl`, `gate_band_beyond.jl`,
  `tridiagonal_rung.jl`, `gate_fill_periteration.jl`, `loop_breakdown.jl:53`,
  `gate_crossover_fill.jl:32`, `strictmode_audit.jl:265-276` (signatures), `:279`
  (`ReducedOperator(ws)`), `:287-288` (`ws.linsys.gram`, `typeof(ws.P)`).
- Docs: `docs/src/matrices.md:276` (`ws.P`, `ws.A`), `docs/src/examples.md:609` (`ws.rho`,
  `ws.algorithm`), `docs/src/operators.md:112` (`ws.xtilde`, `ws.ztilde`).

---

## 2. Target layout

### 2.1 Steps 1 and 2: one package, three directories

```
src/PureOSQP.jl                 module; includes in the order below; exports
src/core/
  constants.jl                  INFTY, MIN/MAX_SCALING, RHO_* … (types.jl:84-94)
  status.jl                     Status, PolishStatus, has_solution, status_name, Solution
  blockdiagonal.jl kronecker.jl rowcoupled.jl operator.jl   (unchanged)
  problem.jl                    Problem, validate, check_*, is_symmetric, is_convex,
                                validate_update!, adopt_update!
  scaling.jl                    equilibrate!, traversals, mul_A!/mul_At!/mul_P!(·, prob, ·),
                                reduced_diagonal!
  elementwise.jl                unchanged, plus the IPM's elementwise kernels
  weights.jl                    SystemWeights, SelectionFor tags
  linsys.jl                     LinearSystem contract, BackendInfo, ReducedCholesky, FullKKT,
                                DiagonalReduced, TridiagonalReduced, reduced_rhs!, ladder
  block.jl lowrank.jl kronsolve.jl   (backends, on prob + wt)
  preconditioner.jl             update_preconditioner! interface, JacobiPreconditioner (§9.3)
  termination.jl                norm kernels, gap_terms, the tolerances both algorithms test
                                against, certificate tests, support kernels
  options.jl                    Options, default_options, update_settings!(::QPWorkspace)
  polish.jl                     polish_kernel!
  derivative.jl                 active_kkt(prob, x, y, z) and the two derivatives
src/admm/
  settings.jl workspace.jl rho.jl accelerate.jl termination.jl admm.jl update.jl api.jl
src/ipm/
  settings.jl workspace.jl ipm.jl api.jl
src/api.jl                      setup/solve(…; algorithm), capabilities, dimensions,
                                constraint_violation
```

`@verify LinearSystem subtypes = true trim_compat = true` stays at the end of the module.

### 2.2 The ten extensions in steps 1–2

| extension | touches | change |
|---|---|---|
| SparseArrays | traversals, rungs, three backends, `is_convex`, `check_finite` | backend/rung signatures; `IPMSelection` methods (§5) |
| LDLFactorizations | `ldl_backend`, `ldl_kkt_backend`, `ldl_posdef`, two backends | backend signatures |
| BandedMatrices | `choose_backend`, `BandedReduced`, `structural_rows`, `is_symmetric/is_convex` | signatures |
| Krylov | `IndirectCG`, `ReducedOperator`, `indirect_backend` | signatures, tolerance seam, preconditioner slot applied by `ldiv!`, residual callback, `inner_iterations` (§3.6, §9) |
| GPUArraysCore | `choose_backend` refusals, traversals | `choose_backend` signature; an `IPMSelection` refusal (§5) |
| LinearMaps, SciMLOperators | `ProductOperator` constructors, `setup`/`solve` overloads | forward `algorithm` and `preconditioner` keywords (they already forward `kwargs...`) |
| COSMOAccelerators | ADMM hooks, `admm_step!` | none (its reads are in `src/accelerate.jl`) |
| ChainRulesCore | `setup`/`solve!`/derivatives | passes `algorithm` through in step 2 |
| MathOptInterface | `setup`/`solve!`/`Solution` | `algorithm` raw attribute, `NUMERICAL_ERROR` mapping in step 2 |

### 2.3 Step 3: monorepo layout (design only)

```
<repo>/                         umbrella <Umbrella>.jl: solve/setup(…; algorithm), MOI ext,
                                ChainRulesCore ext, docs, oracle/corpus tests, trim entry
                                points, StrictMode audit
lib/<Core>.jl/                  src/core/*; extensions: SparseArrays, LDLFactorizations,
                                BandedMatrices, Krylov, GPUArraysCore, LinearMaps, SciMLOperators
lib/<Core>ADMM.jl/              src/admm/*; depends on <Core>; extension: COSMOAccelerators
lib/<Core>IPM.jl/               src/ipm/*; depends on <Core>
```

An extension goes where the function it extends lives. `PureOSQP.Optimizer` (stub at
`src/PureOSQP.jl:62`) and `solve` → umbrella, so MOI and ChainRules go there.
`LDLFactorizationsExt` reaches `SparseArraysExt` through `Base.get_extension(<Core>, …)`
(`LDLFactorizationsExt.jl:102,142,272,280,296`); both stay Core extensions. Registration
order Core → ADMM, IPM → umbrella through Registrator's `subdir` (**unverified** on this
workstation). Each subpackage has its own `test/`; the trim
test and the audit need every extension and live in the umbrella; a Core-only trim entry set
(`Problem` + `factorize!`/`solve_system!` on `ReducedCholesky` and `FullKKT`) is added.

---

## 3. The backend interface

### 3.1 Weights

```julia
"""
    SystemWeights{T,V}

The diagonal weights and the primal regularization the linear system is built from:

    reduced   P̃ + σI + Ãᵀ diag(w) Ã
    KKT       [P̃ + σI   Ãᵀ  ;  Ã   −diag(w_inv)]

Invariants, maintained by the owner: `w[i] > 0`, `w_inv[i] == inv(w[i])` as the owner
computed it, `sigma > 0`, `length(w) == m`. The vectors are read in place, so a change to
their contents reaches the next `refactor_weights!` without a new object; a change to `sigma`
needs a new object.
"""
struct SystemWeights{T <: Real, V <: AbstractVector{T}}
    w::V
    w_inv::V
    sigma::T
end
```

ADMM: `w = rho_vec`, `w_inv = rho_inv_vec`, `sigma = settings.sigma`; `update_settings!`
assigns `ws.weights = SystemWeights(ws.weights.w, ws.weights.w_inv, new.sigma)` whenever it
refactorizes (`api.jl:48-53`). IPM: `w` is the per-side regularized weight of §8.2,
`w_inv = inv.(w)`, `sigma = δ_p`; equality and free rows set `w_inv` directly (§8.2). The
backend interface does not know which algorithm built the weights.

The object is a field of each workspace, not built per call, so the hot path constructs
nothing. `rho_vec`/`rho_inv_vec` as ADMM workspace fields are removed; the tests in §1.9 read
`ws.weights.w`.

`refactor_weights!(ls, prob, wt)` is the backend-level function; the workspace-level
`refactor_rho!(ws)` keeps its name.

### 3.2 Contract

```julia
abstract type LinearSystem end

@contract LinearSystem begin
    factorize!(::Self, ::Problem, ::SystemWeights)::Bool
    solve_system!(::Self, ::Problem, ::SystemWeights, ::Any, ::Any, ::Any, ::Any)::Nothing
    backend_info(::Self)::BackendInfo
    :optional                                   # every one has a LinearSystem default
    refactor_weights!(::Self, ::Problem, ::SystemWeights)::Bool
    solve_multiplier!(::Self, ::Problem, ::SystemWeights, ::Any, ::Any, ::Any, ::Any)::Nothing
    check_update(::Self, ::Any, ::Any)::Nothing
    set_tolerance_level!(::Self, ::Any)::Nothing
    set_refresh_index!(::Self, ::Int)::Nothing
    adopt_settings!(::Self, ::QPAlgorithm, ::Options)::Nothing
    use_residual_stop!(::Self, ::Bool)::Nothing
    last_solve_converged(::Self)::Bool
    inner_iterations(::Self)::Int
end
```

**As built (S13b).** The optional section above is in `src/core/linsys.jl`, each entry with a
one-line description, so `describe(LinearSystem)` lists the whole backend interface. TypeContracts
checks an optional entry by `hasmethod` only, never its return type, and every optional function
has a `LinearSystem` default, so the section documents rather than constrains.
`@verify LinearSystem subtypes = true trim_compat = true` is unchanged and passes.

**Preconditioner contract (S13b, `src/core/preconditioner.jl`).**

```julia
abstract type Preconditioner end          # exported; IdentityPreconditioner and JacobiPreconditioner subtype it
@contract Preconditioner begin
    update_preconditioner!(::Self, ::Problem, ::SystemWeights, ::Int)::Self
    LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)
end
@verify Preconditioner subtypes = true trim_compat = true
```

- The vector slots of `ldiv!` are `AbstractVector`, not `Any`: `hasmethod(ldiv!, Tuple{Any,
  Cholesky, Any})` is false, since LinearAlgebra's factorization method restricts both vectors
  to `AbstractVecOrMat`, while `Tuple{AbstractVector, Cholesky, AbstractVector}` is covered.
  The backend only ever passes its own vectors.
- `IdentityPreconditioner` gains `ldiv!(y, ::IdentityPreconditioner, x) = copyto!(y, x)` to
  satisfy it; the Krylov extension still maps it to `I`, so the method is never on the solve path.
- `update_preconditioner!` has an `Any` default (§9.3), so every type has it; the refresh-index
  form is the only form.
- A caller's preconditioner is not a subtype. `check_preconditioner(M, typeof(q0))` runs in both
  `setup_backend` methods on the `:indirect` branch, after the problem is built and before the
  backend: `hasmethod(LinearAlgebra.ldiv!, Tuple{V, typeof(M), V})` with `V` the backend's vector
  type, throwing an `ArgumentError` that names the missing `ldiv!` signature. `nothing` passes.
  `update_preconditioner!` is not checked (its default applies). `TypeContracts.check_contract`
  passes for `Cholesky{Float64, Matrix{Float64}}` and for both bench references
  (`LaggedCholesky`, `IncompleteLDL`), asserted in `test/contract_tests.jl`. The bench file
  `bench/ipm_preconditioners.jl` is brought in with `include`'s `mapexpr` stripping its `using
  LimitedLDLFactorizations` line, so the two preconditioner types are defined for the contract
  check without adding that package to the test environment; their bodies are never run.

TypeContracts 0.14 checks `hasmethod(f, Tuple{Self, arg_types...})` then infers the return
(`~/.julia/packages/TypeContracts/wFVz0/src/check.jl:167-175`). With `::Problem` and
`::SystemWeights` in the contract, a method may annotate those two slots with exactly those
types or leave them unannotated; anything narrower is reported missing. The vector slots are
`::Any` as today, and every method annotates its return so inference through the abstract
`Problem` gives the declared type.

```julia
"""
    factorize!(ls, prob, wt) -> Bool

Rebuild the factorization of the system `prob` and `wt` define. `false` means this backend
cannot factor it (not positive definite for a reduced backend; a zero pivot for a KKT one).
Reads `prob.P A D E c n m` and `wt`; may use `prob.work_n work_m` as scratch. Every method
binds `P A D E c n m` to locals at entry (as `linsys.jl:573, 603-604` do).
"""
function factorize! end

"""
    refactor_weights!(ls, prob, wt) -> Bool

Refresh after only `wt` changed since the last `factorize!`. Default rebuilds; an override
may keep every part that depends on `P A D E c` alone. (`DiagonalLowRank` keeps its
override, `lowrank.jl:154`.)
"""
refactor_weights!(ls::LinearSystem, prob, wt) = factorize!(ls, prob, wt)

"""
    solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z) -> Nothing

Solve for `x` and write `z = Ã x` as this backend computes it: a reduced backend forms the
product, a KKT backend recovers it from the eliminated multiplier as `rhs_z + w_inv ⊙ ν`.
None of `rhs_x rhs_z x z` may alias each other or `prob.work_n`, `prob.work_m`, `prob.tmp_n`,
`prob.tmp_m` (`reduced_rhs!` writes `work_n` and `work_m`; the products use `tmp_*`).
"""
function solve_system! end

"""
    solve_multiplier!(ls, prob, wt, rhs_x, rhs_z, x, nu) -> Nothing

Solve the same system for `x` and the multiplier `ν`. Row two of the KKT system reads
`Ã x − w_inv ⊙ ν = rhs_z`, so `ν = w ⊙ (Ã x − rhs_z)`, which is what the default derives
from `solve_system!`; a KKT backend overrides it to hand `ν` over directly, which does not
lose digits when `w_inv` is small.
"""
function solve_multiplier!(ls::LinearSystem, prob, wt, rhs_x, rhs_z, x, nu)
    solve_system!(ls, prob, wt, rhs_x, rhs_z, x, nu)      # nu holds z for a moment
    subtract!(nu, nu, rhs_z)
    scale_by!(nu, wt.w)
    return nothing
end
```

Two solve methods rather than one returning `ν`: ADMM's `z̃` must stay the value `Ãx̃`
computed as today (`linsys.jl:714`) for the oracle's 1e-10 iterate match, and IPM wants `ν`
without the cancellation of `w ⊙ (z − rhs_z)` on a KKT backend near convergence. Overrides:
`FullKKT`, `SparseKKT`, `LDLKKT` (each is the existing `solve_system!` minus its last loop).

`reduced_rhs!(prob, wt, rhs_x, rhs_z) -> prob.work_n` keeps its arithmetic order
(`linsys.jl:695-704`) with `wt.w` for `ws.rho_vec`.

`check_update(ls::LinearSystem, P, A) -> Nothing` (default no-op) hosts the three backend
invariants that `update!` tests today through `ws.linsys isa …` (`update.jl:58,78,87-105`).
`update!` calls it with the effective matrices (the replacement if given, otherwise the current
`prob.P`/`prob.A`) whenever either is replaced. `KroneckerReduced.factorize!` returns `false`
when `P` is not a scalar multiple, removing the separate refresh at `update.jl:161`.

`inner_iterations(ls::LinearSystem) -> Int` (default `0`) reports the inner iterations an
iterative backend has spent over its life; `IndirectCG` counts `stats.niter` per solve.
`Solution.cg_iters` is the difference across one solve, under both algorithms.

### 3.3 Cost reporting

`BackendInfo` is unchanged. The per-backend flop model used to compare backends under the
IPM lives in the bench scripts (`bench/ipm_vs_clarabel.jl`, `bench/ipm_matrixfree.jl`),
computed from `backend_info(ls).factor_nnz` and the dimensions, next to the measurement it is
checked against.

### 3.4 How ADMM calls it

```julia
# admm_step!
scale_subtract!(ws.rhs_x, ws.weights.sigma, ws.x_prev, ws.prob.q)
subtract_scaled!(ws.rhs_z, ws.z_prev, ws.weights.w_inv, ws.y)
# set_tolerance_level!(ws.linsys, max(scaled_prim_res, scaled_dual_res))  immediately before:
solve_system!(ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde)
update_x!(…); update_zy!(…, ws.weights.w, ws.weights.w_inv, …)

# refactor!(ws)     = refactored!(ws, factorize!(ws.linsys, ws.prob, ws.weights))
# refactor_rho!(ws) = refactored!(ws, refactor_weights!(ws.linsys, ws.prob, ws.weights))
# refactored! keeps the count, the throw and accelerator_reset! (ADMM policy)
# setup_backend and update_settings! call adopt_settings!(ws.linsys, ws.algorithm, ws.options) after the workspace is built
```

### 3.5 How IPM calls it

`setup_backend(::InteriorPoint, …)` calls `use_residual_stop!(ws.linsys, true)` once, after the workspace is
built — a no-op for every direct backend, and for `IndirectCG` the switch that turns on the
inner stopping rule of §9.4 for the rest of the workspace's life. `setup_backend(::OperatorSplitting, …)` never
calls it, so ADMM keeps its tolerance stop.

```julia
# outside ipm_step!, once per outer iteration (the backend's allocation lives here):
weights!(ws)                        # §8.2: w = z_l/(s_l + δ_d z_l) + z_u/(s_u + δ_d z_u), masked;
                                    # equality rows w_inv = δ_d, free rows w_inv = 1/δ_d
set_refresh_index!(ws.linsys, k)    # outer iteration; −1 for the starting-point solve (§9.3)
refactor_weights!(ws.linsys, ws.prob, ws.weights) || bump_regularization!(ws)
# ipm_step!: two solves on the same factorization, no allocation
set_tolerance_level!(ws.linsys, min(μ, ‖r‖∞))
solve_multiplier!(ws.linsys, ws.prob, ws.weights, rhs_x, rhs_z, dx_aff, dy_aff)   # predictor
…                                                                                # corrector
```

For `IndirectCG`, `refactor_weights!` is where `update_preconditioner!` runs (§9.3). With the
residual stop on, `IndirectCG.solve_system!` skips `Krylov.warm_start!` and starts CG from zero
on its own, at no extra product; the IPM also zeroes `dx_aff`/the corrector's `Δx` before each
`solve_multiplier!` call (§8.4), so the `x` argument holds zero either way.

### 3.6 The CG seam (Krylov ext)

`IndirectCG{T,V,K,M}` holds a preconditioner `M` (§9.3), a mutable `level::T` to track the
tolerance, a mutable `refresh_index::Int` handed to `update_preconditioner!`, and counters
`total_iters::Int`, `misses::Int`. The three CG settings (max_iter,
tol_fraction, tol_reduction) are not stored at construction; `indirect_backend(proto, n, m, preconditioner)`
takes the preconditioner as its fourth argument, with CG settings arriving through `adopt_settings!`.

```julia
adopt_settings!(ls::LinearSystem, alg, options) = nothing   # direct backends
adopt_settings!(ls::IndirectCG, alg::OperatorSplitting, options) = (ls.max_iter = options.cg_max_iter;
                                             ls.tol_fraction = options.cg_tol_fraction;
                                             ls.tol_reduction = alg.cg_tol_reduction;
                                             nothing)
adopt_settings!(ls::IndirectCG, ::InteriorPoint, options) = (ls.max_iter = options.cg_max_iter;
                                             ls.tol_fraction = options.cg_tol_fraction;
                                             ls.tol_reduction = typemax(Int);   # the IPM has no idle-solve rule (§8.10)
                                             nothing)

set_tolerance_level!(ls::LinearSystem, level) = nothing     # direct backends
set_tolerance_level!(ls::IndirectCG, level) = (ls.level = level; nothing)
set_refresh_index!(ls::LinearSystem, k) = nothing           # direct backends
set_refresh_index!(ls::IndirectCG, k) = (ls.refresh_index = k; nothing)
last_solve_converged(ls::LinearSystem) = true
last_solve_converged(ls::IndirectCG) = ls.last_reached     # set by the miss rule of §9.4; consecutive misses counted by the IPM from this
use_residual_stop!(ls::LinearSystem, flag) = nothing        # direct backends; no-op by default
use_residual_stop!(ls::IndirectCG, flag) = (ls.use_residual_stop = flag; nothing)  # §9.4; off by default, ADMM keeps tolerance stop
```

`adopt_settings!`'s default does nothing, which is right for a direct backend and silently
wrong for `IndirectCG` under an algorithm with no method of its own: `max_iter` stays at its
`0` placeholder and every solve returns its starting point. `factorize!(ls::IndirectCG, …)`
refuses a zero `max_iter`, naming `adopt_settings!` — `Options` refuses a `cg_max_iter` of
zero, so the placeholder is the only way to reach it, and the check is off the per-iteration
path.

`refresh_index` is owned by the algorithm: the IPM sets it to the outer iteration before
`refactor_weights!` (`−1` before the starting-point factorization; a regularization bump
refactors with the same index); ADMM sets it to `refactor_count` before `factorize!` and
`refactor_weights!`.

Options passed at construction do not reach the `:auto` operator path through `indirect_rung`,
and nothing would refresh them on `update_settings!`; `adopt_settings!` is called by `setup_backend`
after the workspace is built and by `update_settings!` whenever the options or the algorithm
parameters change.

ADMM calls `set_tolerance_level!` immediately before `solve_system!` with
`max(scaled_prim_res, scaled_dual_res)`; the backend computes `atol` exactly as at
`KrylovExt.jl:140-142`, so `:indirect` iterates do not move. With the residual stop off (ADMM),
CG warm-starts from the `x` argument, which holds the previous solution (the same values as
`ws.xtilde` today). With the residual stop on (IPM only, §3.5), `solve_system!` does not call
`Krylov.warm_start!` at all: a warm start from the previous step measured no G1 gain, 1.02×
(dense) and 1.21× (sparse) the median products, and five extra misses from Krylov's
machine-precision stop (**measured**, `ipm_rowtypes_spike.json`, `cgstart` groups).
`ReducedOperator{T,PB,WT}(prob, wt)` replaces `ReducedOperator(ws)`.

---

## 4. Workspace split

### 4.1 `Problem` (shared)

```julia
mutable struct Problem{T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T}}
    P::MP                 # mutable: update! replaces them (update.jl:160,163)
    A::MA
    n::Int
    m::Int
    q0::V; l0::V; u0::V   # caller's data, clamped to ±INFTY
    q::V;  l::V;  u::V    # equilibrated
    D::V;  E::V;  c::T    # Ruiz factors
    scaling::Int          # sweeps run; 0 ⇒ D, E, c are identity (replaces settings.scaling reads)
    tmp_n::V; tmp_m::V    # scratch of mul_A!/mul_At!/mul_P!
    work_n::V; work_m::V  # scratch of reduced_rhs! and the backends
end
```

`Problem(T, P, q, A, l, u; scaling)` runs `validate`, allocates with the `similar(q0, …)`
discipline (`types.jl:553-560`), and calls `equilibrate!`. `validated_problem(T, n, m, P, q, A, l, u, scaling)`
builds a `Problem` from already-validated data, bypassing the validation step; `setup` validates
once and uses it. It does not run `is_convex`: the shift is the algorithm's (`σ` for ADMM, `δ_p`
for IPM), so each `setup` calls `is_convex(T, P, shift)` where `types.jl:548` does today, and each
documents its shift.

### 4.2 ADMM `OperatorSplittingWorkspace`

```julia
mutable struct OperatorSplittingWorkspace{T, MP, MA, V, VI <: AbstractVector{Int8}, LS <: LinearSystem, AC} <: QPWorkspace{T}
    prob::Problem{T, MP, MA, V}
    linsys::LS
    weights::SystemWeights{T, V}     # w = ρ, w_inv = ρ⁻¹, sigma = σ
    rho::T
    constr_type::VI
    x y z x_prev z_prev xtilde ztilde delta_x delta_y Ax Px Aty rhs_x rhs_z :: V
    accel::AC
    refactor_count::Int
    … every run-state field from types.jl:339-376, unchanged …
    algorithm::OperatorSplitting{T}
    options::Options{T}
end
```

Same seven type parameters, so the `MAX_TYPEUNION_LENGTH` argument at `types.jl:514-527` is
untouched; `Problem{T,MP,MA,V}` is determined by them. Field moves: `P A n m q0 l0 u0 q l u D
E c tmp_n tmp_m work_n work_m` → `ws.prob`; `rho_vec rho_inv_vec` → `ws.weights`. No
`getproperty` forwarding; every read is rewritten, including §1.9. `setup(::Type{T}, …, alg;
linsys, kwargs...)` keeps `@constprop :aggressive` and the `Val(linsys)` lift;
`setup_backend` becomes `prob = Problem(…; scaling = options.scaling)` + `is_convex` + ρ
classification + weights + the same `if LS === …` ladder over `choose_backend(P, A, prob,
wt, ADMMSelection())`.

### 4.3 Who adapts

- **Products** take `prob`; the `Tridiagonal` specialization dispatches on
  `Problem{T, <:AbstractMatrix, <:Tridiagonal}`.
- **Residual kernels** (`core/residuals.jl`): `residuals_at!(prob, x, y, z, Ax, Px, Aty) ->
  (pr, dr, obj)` (today's `polish.jl:7-26`, writing the three vectors it was reading off
  `ws`), `gap_terms(prob, y, Px, x) -> (xtPx, qtx, SCy)` (`termination.jl:67-73`, `tmp_m` as
  scratch as today `:71`), `eps_prim(prob, settings, z, Ax)`, `eps_dual(prob, settings, Aty,
  Px)`. Both workspaces own `Ax Px Aty`. `update_residuals!(ws)` keeps its name, its field
  writes and its arithmetic order, calling the kernels.
- **Certificates**: `is_primal_infeasible(prob, dy, eps) -> Bool` projects `dy` in place, as
  today (`termination.jl:209`); each workspace owns the buffers it hands in (`delta_y` for
  ADMM; `cert_y`, `cert_x` for IPM, §8.7).
- **`update!`**: `validate_update!(prob, ls; P, A, q, l, u)` (`update.jl:43-153` minus the
  ADMM and backend lines of §1.5, which become `check_update` and the ADMM tail) and
  `adopt_update!(prob; …)` (`:159-172`). `update!` calls `check_update(ls, P, A)` with the
  effective matrices (the replacement if given, otherwise the current `prob.P`/`prob.A`) whenever
  either is replaced. ADMM: validate (+ `is_convex` with `σ`, the Kronecker ρ-class guard) →
  adopt → `set_rho_vec!` if bounds moved → `refactor!`. IPM: validate (with `δ_p`) → adopt →
  recompute row classes; `s z` are untouched until the next starting point; no refactorization,
  the next iteration factorizes anyway.
- **Derivatives**: `active_kkt(prob, x, y, z)` on the problem-space point; each workspace's
  method checks `status === SOLVED`, `require_host`, unscales, delegates.
- **Polishing**: `polish_kernel!(prob, x, y, z, prim_res, dual_res, Ax, Px, Aty; delta,
  refine_iter) -> (status, xpol, ypol, zpol)` is `polish.jl:43-125`; ADMM's `polish!(ws)`
  copies on success and runs `update_residuals!`.
- **MOI**: step 2 adds the raw attribute `algorithm` (`"admm"` or `"ipm"`, validated at `MOI.set`, §11)
  and the `NUMERICAL_ERROR` rows (§8.6).
- **`Solution`**: one struct; new field `cg_iters::Int` (both algorithms). IPM fills
  `rho_estimate rho_updates accel_declined primdual_int*` with zeros and documents it.

### 4.4 Workspace and algorithm contracts (as built, S13b)

Declared in `src/types.jl`, after `Solution` (the return types need it), and verified at the end of
`src/PureOSQP.jl` with `@verify QPWorkspace subtypes = true trim_compat = true` and
`@verify QPAlgorithm subtypes = true trim_compat = true`. No supertype changed: both workspaces
and both algorithm objects already subtyped these types (S12b).

```julia
@contract QPWorkspace begin
    solve!(::Self)::Solution
    warm_start!(::Self)::Self                         # keywords x, y
    cold_start!(::Self)::Self
    update!(::Self)::Self                             # keywords q, l, u, P, A
    update_settings!(::Self)::Self                    # options form; QPWorkspace default
    update_settings!(::Self, ::QPAlgorithm)::Self     # algorithm-object form
    dimensions(::Self)::Tuple{Int, Int}               # now a QPWorkspace method
    :optional
    update_rho!(::Self, ::Real)::Self                 # ADMM only
    constraint_violation(::Self)::AbstractVector      # ADMM only
end

@contract QPAlgorithm begin
    setup_backend(::Self, ::Val, ::Type{<:Real}, ::AbstractMatrix, ::AbstractVector,
                  ::AbstractMatrix, ::AbstractVector, ::AbstractVector)
    algorithm_defaults(::Self, ::Type{<:Real})::NamedTuple
    default_options(::Self, ::Type{<:Real})::Options  # QPAlgorithm default
    element_typed(::Self, ::Type{<:Real}, ::Options)::QPAlgorithm
    :optional
    adopt_settings!(::LinearSystem, ::Self, ::Options)::Nothing
end
```

- Keyword methods are checked through their positional signature, all `hasmethod` sees.
- `dimensions` was defined for `OperatorSplittingWorkspace` only; it is now
  `dimensions(ws::QPWorkspace) = (ws.prob.n, ws.prob.m)`, so the IPM workspace has it.
- Left out of the workspace contract: `polish!`, `check_termination`, `build_solution` and the
  step functions. Both workspaces have the first three, but only each algorithm's own `solve!`
  calls them, so nothing outside the algorithm depends on their shape. `constraint_violation!`
  is left out because its method is `(::AbstractVector{T}, ::OperatorSplittingWorkspace{T})`,
  which `hasmethod` on `Tuple{AbstractVector, Self}` does not find; the allocating form stands for
  both. Fields are not expressible in a contract; the `QPWorkspace` docstring lists the ones the
  shared methods read (`prob linsys algorithm options x y z status polished`).
- `Type{<:Real}`, not `Type`, in the algorithm slots: the methods take `::Type{T} where {T <:
  Real}`, which `Tuple{…, Type, …}` is not covered by.
- `setup_backend` has no return type: `Base.return_types` through the contract's abstract data
  arguments is `Any`. An annotation on `setup_backend` would put a conversion on setup's return
  path for no gain in what is checked, so none was added.
- The selection tags are not an algorithm hook: each `setup_backend` constructs `ADMMSelection()`
  or `IPMSelection()` itself (§11), so `setup_backend` is the whole dispatch surface.
- Precompile time of PureOSQP with the three new `@verify` lines: 4.5 s against 3.3 s before
  (one cold precompile each, neuromancer).
- Measured (S13b): ADMM snapshot 41/41 identical; StrictMode audit unchanged; trim item green.
  `admm_step!` minimums, A/B/B/A on one core, 100 steps from a fresh setup (base/new): Random QP
  3.086/3.107 µs (+0.7%), SVM 6.820/6.786 µs, Huber 13.600/13.480 µs.
- Two existing tests built a caller preconditioner with no `ldiv!` to reach the
  `update_preconditioner!` retyping error; they now define `ldiv!` so that error is still the
  one reached.

---

## 5. Algorithm-aware selection

```julia
abstract type SelectionFor end
struct ADMMSelection <: SelectionFor end
struct IPMSelection <: SelectionFor end
```

The tag is the last argument of `choose_backend(P, A, prob, wt, sel)`, `select_backend`, and
every rung (`formed_rung(P, A, prob, sel)`, the rest `(P, A, prob, wt, sel)`); extension
helpers take it too. `P` and `A` stay explicit because the
extensions dispatch on them. Under `ADMMSelection` every rung is the method it is today with
the ten-argument tail collapsed; the ADMM ladder does not change. The IPM owns its deviations
as extra methods:

| rung / backend | `ADMMSelection` | `IPMSelection` |
|---|---|---|
| `kronecker_rung` | as today (uniform `wt.w`, identity scaling) | `nothing` |
| `sparse_form` (consulted by `kkt_rung` and `reduced_rung`) | `:kkt` for a row of `A` spanning half the columns whose KKT pattern stays under `n²/2`; `:reduced` while the reduced pattern fills at most `SPARSE_PATTERN_FRACTION` of `n²`; `:none` otherwise, so the pair falls to `formed_rung` | `:none` once the KKT pattern fills a quarter of `(n+m)²`; then `:kkt` for a spanning row, `:kkt` while `2m ≤ n`, `:reduced` for a reduced pattern past `FULL_REDUCED_PATTERN`, and otherwise `:kkt` when `Σᵢ nnzᵢ²` exceeds `REDUCED_ASSEMBLY_LIMIT · n²` |
| `formed_rung`, `dense_rung` | as today; `formed_rung` is the terminal for every `SparseMatrixCSC` pair `sparse_form` declines | `FullKKT(proto, n, m)` for `T <: BlasFloat` (reads `A[i, j]` entry by entry at every factorization; covers dense-`P`/sparse-`A` and LP pairs); `ReducedCholesky` otherwise (`bunchkaufman!` has no generic method) |
| structured reduced backends (diagonal, tridiagonal, banded, block, sparse reduced), `SparseCholmod/SparseLDL` | as today | as today; reached only when the pair has that structure; kept as measured (S10) |
| `lowrank_rung` | as today (declines above `10k ≤ n`) | declines unconditionally: the pair goes to `FullKKT` (measured, S10 — the Woodbury solve does not reach tolerance on the low-rank family's linear programs) |
| `indirect_rung` (operator pairs) | as today | declines; an operator pair under `:ipm` is served only by `linsys = :indirect` **with** a caller-supplied preconditioner (§9.2); without one `setup` refuses by name |
| GPU ext `choose_backend` | refusal as today | `choose_backend(P::AbstractGPUMatrix, …, ::IPMSelection)` refuses `:ipm` by name (v1 is CPU-only) |
| named kinds | as today | `:kronecker` and `:lowrank` throw naming the pair's requirement |

`ReducedInverse` under IPM: the inversion costs `2n³/3` per iteration and buys a `symv` for
2–4 solves. `FullKKT` is preferred instead; a factor-keeping `ReducedCholeskyFactor` is added
only if the S13 bench shows the inversion above ~25% of an IPM iteration on a reached rung.

**A third `SelectionFor`.** Every point above answers one, either by serving it or by naming
what it still owes. The rungs whose default is to decline — `kkt_rung`, `reduced_rung`,
`kronecker_rung`, `block_rung`, `lowrank_rung`, `formed_rung` — and the `choose_backend`
methods for a structured pair take any subtype, as does the GPU refusal, which is a property
of the factorization rather than of the algorithm. `select_backend`, `dense_rung`,
`indirect_rung` and `sparse_form` have no algorithm-independent answer and throw through
`refuse_selection`, naming themselves; a new algorithm defines those four (`sparse_form` only
if the sparse rungs are in its ladder). `test/contract_tests.jl` asserts the whole set against
a dummy subtype, so nothing here can regress to a `MethodError`.

**Named kinds.** `named_backend(Val{LS}, P, A, prob, wt, sel, preconditioner)` in
`src/core/linsys.jl` holds the branch per named `linsys` for both algorithms; `nothing` means
`:auto` and the ladder. The kinds an algorithm cannot serve are refused by its own
`setup_backend` before the problem is built, so the message can name the reason.

---

## 6. Migration sequence

Gates, run after every step unless stated: (a) full suite via `julia_run_testitems` with
`max_workers` set; (b) `julia --project=bench bench/strictmode_audit.jl`; (c) the trim test
item; (d) the S0 snapshot artifact through S7. "Identical" means the oracle, c-suite and
corpus items pass unchanged and the snapshot matches. **M** mechanical, **J** judgment.

| # | step | gate | diff. |
|---|---|---|---|
| S0 | **Snapshot artifact.** `bench/snapshot.jl`: for every suite class in `bench/suite_problems.jl` and every structured family in `selection_tests.jl`, record `(backend_name, iter, refactor_count, status, round(obj_val, 10))` at fixed settings with `BLAS.set_num_threads(1)`, on the S0 tree, into `bench/results/snapshot_s0.json`; a bench check compares the current tree to it. Not a committed test item. | artifact generated | M |
| S-spike | **Done**: `bench/ipm_matrixfree_spike.jl` and `spike2.jl` (commit f2aee68), `bench/ipm_rowtypes_spike.jl`; findings in §8.2, §8.4, §8.5, §9.1, §9.4. | — | — |
| S2a | **`Problem` extraction.** `core/problem.jl`; `OperatorSplittingWorkspace.prob`; every `ws.<moved field>` rewritten; `mul_*` on `prob`; `settings.scaling` reads → `prob.scaling`; backends keep `(ls, ws)` for now but bind `P A D E c n m` to locals at entry; docs lines in §1.9. Gate adds `bench/loop_breakdown.jl` step timings against S0 on neuromancer, ABBA-interleaved with the S0 tree (indicative, §10.7). | identical; audit; trim; timings within noise | M |
| S1+S2b | **Weights and backend signatures, one pass.** `SystemWeights`; `factorize!(ls, prob, wt)`, `refactor_weights!`, `solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z)`; every `rho_vec/rho_inv_vec/settings.sigma` read of §1.2 → `wt`; `OperatorSplittingWorkspace.weights`; `update_settings!` rebuilds `weights` (+ test that changes `sigma` and checks the factorized matrix); `set_rho_vec!`, `pack/unpack_fixed_point!`, `admm_step!` follow; `reduced_rhs!(prob, wt, …)`; `ReducedOperator(prob, wt)`; `check_update` + Kronecker `mu` in `factorize!`; `update!` split; contract; tests/bench/audit signatures `(LS, PB, WT, V, V, V, V)`, `(LS, PB, WT)`. CG seam: `set_tolerance_level!(ls::LinearSystem, level) = nothing` (default), mutable `level::T` field on `IndirectCG`, `admm_step!` sets it to `max(scaled_prim_res, scaled_dual_res)` immediately before `solve_system!`; `adopt_settings!(ls::LinearSystem, alg, options) = nothing` (default, signature as of S12b) with `IndirectCG` method filling `max_iter`, `tol_fraction`, `tol_reduction` from the options and algorithm parameters, called by `setup_backend` after workspace build and by `update_settings!` on every settings change; CG warm-starts from `x` argument. | identical; audit; trim | M |
| S3 | **Selection tag.** `SelectionFor`; collapse rung signatures; `ADMMSelection` threaded from `setup`; no `IPMSelection` methods yet. Superseded in part by S14, which replaced the measured gates the rungs carried with `sparse_form` and removed `density_gate_rung` from both ladders. | identical; `selection_tests` backends unchanged | M |
| S4 | **Shared kernels.** `residuals_at!`, `gap_terms`, `eps_*`, certificate tests on `(prob, buffer, eps)`, `polish_kernel!`, `active_kkt(prob, x, y, z)`; ADMM wrappers keep names and order. | identical; audit (`update_residuals!` row); trim (`derivatives`, `solve_polish`) | M |
| S5 | **CG seam continued.** Counters `total_iters`, `misses`, `last_reached`; `last_solve_converged(ls)`; `inner_iterations(ls)`; `Solution.cg_iters`; the preconditioner slot `M` applied through `ldiv!` (Krylov `ldiv = true`) with `update_preconditioner!(M, prob, wt, ls.refresh_index)` called from `factorize!`/`refactor_weights!`, `set_refresh_index!` (§3.6), and the default `JacobiPreconditioner` whose `ldiv!(y, J, x)` is `y .= J.dinv .* x`, reproducing today's `Diagonal(ls.prec)` product (§9.3); `IdentityPreconditioner` and `JacobiPreconditioner` defined; a caller-supplied preconditioner requires `scaling = 0` under every algorithm, ADMM included; `indirect_backend(proto, n, m, preconditioner)` signature; passing `preconditioner` with `linsys ≠ :indirect` throws; the stopping and miss rule of §9.4 (callback on Krylov's recursively updated `r`; miss only on budget or breakdown) switched on by `use_residual_stop!(ls, true)`, off by default so ADMM stays bit-identical; consecutive miss counter counted by the IPM from `last_solve_converged`; functions in `src/core/preconditioner.jl`. | identical on `:indirect` (`indirect_tests`, `solve_indirect` trim entry) | J |
| S6 | **`solve_multiplier!`** default + `FullKKT`/`SparseKKT`/`LDLKKT` overrides; test at `w_inv = 1e-12`. | suite; audit unchanged | M |
| S7 | **Directory move** to `src/core`, `src/admm`, `src/ipm` (empty). | identical; audit; trim | M |
| S8 | **IPM skeleton on direct backends.** Builds `src/ipm/{settings,workspace,ipm}.jl`; `setup_backend` takes a leading algorithm argument (dispatch on `OperatorSplitting`/`InteriorPoint` since S12b); the IPM settings hold only the fields the solve uses so far (time limit, infeasibility tolerances, `max_reg_bumps`, CG settings, polishing and verbose arrive with the steps that use them); factorization failure and non-finite residuals throw until S9; `update!`, `update_settings!`, polishing and derivatives are not defined for `InteriorPointWorkspace` until S12; `linsys = :dense` builds `ReducedCholesky` under `:ipm`; GPU arrays are refused by `:auto` selection, while a named backend with GPU arrays fails with the scalar-indexing error as under ADMM. Additionally, `InteriorPointWorkspace`, `setup(…, InteriorPoint())`, `seeded`, starting point, `ipm_step!` with the per-side recovery and the σ floor of §8.2/§8.4 (`τ = 0.99`), regularization of §8.5, residuals through S4 kernels, `SOLVED`/`SOLVED_INACCURATE`/`MAX_ITER_REACHED`, `solve!`, `build_solution`; `IPMSelection` methods of §5 (FullKKT routing, KKT-first sparse, kronecker decline, GPU refusal, operator refusal: unconditional in S8, lifted by name in S11), the `N_s = 0` rule of §8.4. Corpus items under `:ipm` with `:auto`/`:kkt`, referee `< 1e-5`, backend name asserted for the dense-`P`/sparse-`A` and LP cases; objective vs `osqp_ref`; a reproduction test item on the spike-1 dense generator (`make_instance`) at `n = 200`, `κ ∈ {1, 1e3, 1e6}`, fractions `{0.1, 0.5, 0.9}`, in the spike's two-sided form and in spike 3's `mixed` row form, with `scaling = 0`, the spike's iteration count (outer iterations before the `1e-8` termination check passes), reproducing the exact-solve outer counts (`δ = 1e-8`: 6–11) within ±2 through `FullKKT` with `refine_iter = 0` and through `SparseKKT` with `refine_iter = 1`. | new items pass; ADMM gates identical | J |
| S9 | **Robustness.** Dynamic regularization bump; `NUMERICAL_ERROR` (+ every switch of §8.6); certificate buffers, stall rule, certificate tests on step and normalized iterates; `time_limit`, interrupt. Tests: c-suite ported cases under `:ipm`; a random infeasible `n = 20, m = 40` primal case and a dual one; equality-only (`N_s = 0`) and free-row corpus cases; a `Float32` item as §10.8 question 3 decides. Built as commit ba1bc19; S9b not needed. | items pass | J |
| S10 | **Structured backends under IPM, measured.** Each structured family through its recorded backend under `:ipm`: referee tolerance and iteration count recorded per backend into `bench/results/ipm_backends.json` (`bench/ipm_backends.jl`), not the snapshot. Result: only the low-rank family is routed, for failing the referee on its linear programs; no other backend exceeds `2×` the `FullKKT` iterations. `SparseFormedInverse` is unreachable under `:ipm` — the ladder has no formed rung. A sparse pair the KKT form does not suit lands on the sparse reduced backend, observed on `banded_qp(200, 300)`; which of the two forms a pattern reaches is S14's rule. `Float32` as §10.8 question 3 decides. Built as commit c1745cd. | items pass; table in docs | J |
| S10b | **Generic element types under IPM** (§10.8 question 3). The `Float32` refusal is removed; `ipm_floor(T)` supplies the defaults of the tolerances, regularizations and short-step threshold, `precision_eps(T)` the divergence bound (§8.5); test items for `Float32`, `BigFloat`, `ForwardDiff.Dual` (§8.10). | items pass | J |
| S11 | **IPM `:indirect` with a caller-supplied preconditioner** (§9): `preconditioner` keyword, `update_preconditioner!(M, prob, wt, k)`, refusal by name without one (matrices and operators) and without `scaling = 0`, inner stopping and miss rule, zero start, budgets, `cg_fail_limit`, reporting; reference preconditioners in `bench/ipm_preconditioners.jl` (lagged Cholesky over the dense reduced matrix with an in-place `WA`/`K` buffer, refreshed every 3 outer iterations; limited-memory LDLᵀ over the sparse one, §9.3); the test suite's own inline `LaggedCholesky` logs the refresh index `k` it is called with (`M.ks == -1:(iter-1)`) — `IndirectCG.factorize!` never fails, so the bump case that would repeat a `k` is not reachable and is not tested; a `Diagonal` preconditioner with a negative entry ends `NUMERICAL_ERROR` (the cause is named in documentation, not carried in `Solution`, §9.3); `update_preconditioner!` returning another type throws the `ArgumentError`; `bench/ipm_matrixfree.jl` per §9.6, run on the reduced grid of decision 1(a). Verdict: **SHIP** — G1 24/24 and G2 24/24 on the dense `LaggedCholesky` grid (`n ∈ {500, 1000, 2000}`, `κ ∈ {1, 1e6}`, fractions `{0.1, 0.9}`, two-sided and mixed rows, G1 at `eps = 1e-6`), outer counts equal to `FullKKT` on all 24; the sparse `IncompleteLDL` family includes one instance that ends `NUMERICAL_ERROR`, capped at outer iteration 8. Built as commit 2951256. | items pass; bench under `bench/results/`; SHIP recorded in docs | J |
| S12 | **Polish, derivatives, `update!`, warm start, MOI for IPM.** The IPM settings gain `polishing`, `polish_refine_iter`, `delta` (all `Options` since S12b); `InteriorPointWorkspace` gains `polished`, `status_polish`, `polish_time`, `update_time`, and `n_sides` is no longer `const` so `update!` can reclassify rows. `active_kkt`, `adjoint_derivative` and `forward_derivative` take `QPWorkspace{T}`. `update!` reuses `validate_update!`/`adopt_update!`/`check_update` unchanged and never refactorizes; `update_settings!` never refactorizes either, since a solve resets the regularization from the algorithm parameters before its first iteration regardless. `warm_start!`/`cold_start!` needed no change past S8. The MOI extension gains `algorithm` as a `RawOptimizerAttribute`, validating every other raw setting against the options or the selected algorithm's parameters (§11); setting `algorithm` itself is refused when a raw setting already stored does not belong to the algorithm being switched to; `MOI.optimize!` and `BarrierIterations` were already generic. The ChainRulesCore rules needed no change: they call `adjoint_derivative`/`forward_derivative` rather than dispatching on the workspace type. `polish_kernel!`'s early-exit paths return the caller's own `x`, `y`, `z` instead of allocating zeros. `adjoint_derivative` and `forward_derivative` refuse an unpolished `InteriorPointWorkspace` by name: its inactive-row multipliers sit at the barrier parameter rather than at zero, which the active-set test cannot otherwise tell apart from a genuinely active row; a non-materializable operator is refused first, since it was never a candidate for polishing either, and the workspace-polish check would otherwise name the wrong obstacle. `verbose = true` is refused under the IPM rather than accepted and ignored, since the interior-point method has no per-iteration report (since S12b `verbose` is an `OperatorSplitting` parameter, §11). Built as commits 43d0199, af1f856. | `derivative_tests`, `update_tests`, `moi_tests`, `polish_tests` parametrized where semantics carry; `MOI.Test` subset (`test_linear_`, `test_quadratic_`) at `InteriorPoint()` needs no exclusion beyond the ADMM run's three; the IPM derivative referee runs at `eps = 1e-10` rather than `1e-12`, since at `1e-12` the outer solve is already accurate enough that polishing correctly declines (`POLISH_FAILED`) and the derivative then refuses rather than differentiate an unpolished point | J |
| S12b | **Public API: algorithm objects plus shared `Options`** (§10.9, §11). `OperatorSplitting`, `InteriorPoint`, `Options`, `default_options`, `QPAlgorithm`, `QPWorkspace`; workspaces renamed `OperatorSplittingWorkspace`, `InteriorPointWorkspace`; `algorithm` keyword, `Settings`, `IPMSettings`, `Workspace`, `IPMWorkspace` removed from src, ext, test, bench, docs; misplaced-setting errors; MOI routing by name. Built as commit 40df94c. | snapshot 41 identical; audit; trim; full suite; ADMM step and setup timings within 3% | J |
| S13 | **StrictMode + trim for IPM; Clarabel bench; docs.** Audit rows of §8.12; trim entries; `bench/ipm_vs_clarabel.jl`; iteration bounds; API/guarantees/algorithm pages. `SystemWeights` had no `@docs` entry: a comment between its docstring and the struct it documents silently dropped the docstring, breaking every page that referenced it; the comment moved inside the struct body. Built as commit 822e29f. `verbose` output for the interior-point method is built later: §11 and §9.4 record the report, and `test/ipm_tests.jl` tests it directly. | audit green; trim green; bench committed; docs build clean | J |
| S13b | **Workspace, algorithm and preconditioner contracts.** `LinearSystem`'s optional section (§3.2); the `QPWorkspace`/`QPAlgorithm` contracts (§4.4); the `Preconditioner` supertype and its `ldiv!`/`update_preconditioner!` contract, checked by `check_preconditioner` in both `setup_backend` methods (§3.2); `docs/src/interfaces.md`. Built as commit 51c8df4. | audit unchanged; trim green; `contract_tests` pass; `admm_step!` timings within noise | M |
| S14 | **Selection from the sparsity pattern.** `sparse_form` replaces the gates that factored a trial matrix and read its fill (`DENSE_FACTOR_FILL`) and the density gate that read `nnz(A)/mn` (`DENSE_FORM_DENSITY`); `density_gate_rung` is gone from both ladders and `formed_rung` is the ADMM terminal for every `SparseMatrixCSC` pair. `kkt_rung` and `reduced_rung` take `gated::Bool` where they took `fill_limit::Real`, and `ldl_backend`/`ldl_kkt_backend` lose the fill argument. `recommend_linsys` is the measuring tool the rule is not: it builds every backend a pair admits and ranks them. Constants fitted to `bench/results/ipm_selection.json` (71 problems × 8 candidates × both algorithms): the pick is within 1.3× of the fastest measured backend on 69 of 71 under `InteriorPoint` (worst 1.52×) and within 1.03× on all 71 under `OperatorSplitting`. | snapshot regenerated (3 cases change backend); audit; trim; full suite | J |

The spikes' verdicts are in; S2a onward can start. The remaining measurements that could still
invalidate a design choice are S10's (structured reduced backends at `δ = 1e-8`) and S11's
(wall clock including preconditioner builds, `n ≥ 500`, the Kronecker family with a
reference preconditioner), which is why S11 runs before S12 and S13 and before the docs claim
operator support.

---

## 7. Risks, ranked

1. **Bitwise drift in the ADMM step.** Rewrite reads, never expressions; S0 snapshot + oracle
   after every step.
2. **`@constprop` fragility in `setup`** (`types.jl:514-527`). `Problem` takes only
   `scaling::Int`; `linsys` stays a `Val`; trim entries `solve_kronecker`,
   `solve_lowrank_scaled`, `setup_kronecker` are the detectors.
3. **The inner stopping and miss rule of §9.4 is measured at `n ≤ 200` from the spikes, and at
   `n ∈ {500, 1000, 2000}` from S11's reduced grid** (§9.6: G1 24/24, G2 24/24, `FullKKT`-equal
   outer counts on all 24). It reproduces the spikes' explicit-residual oracle on two-sided rows
   at `n ≤ 200` (identical inner and outer counts on 96 runs) and keeps the exact solver's G1
   with equality rows where the oracle does not (`ipm_rowtypes_spike.json`). It accepts steps
   whose reduced residual exceeds `atol_k`, which the exact solver's steps also do; a step that
   is poor for another reason is caught only by the outer stall rule (§8.7). S11's grid does not
   record the per-solve explicit residual over `atol_k` at `n ≥ 500`; the fuller §9.6 sweep that
   would (`κ = 1e3`, fraction `0.5`, the Kronecker family, sparse `n = 5000`, G1 at `1e-8`) is
   not run.
4. **Operator IPM wall clock, `n ≥ 500`.** **Measured** by S11 (`bench/ipm_matrixfree.jl`) on
   the dense `LaggedCholesky` grid of decision 1(a) (`n ∈ {500, 1000, 2000}`, `κ ∈ {1, 1e6}`,
   fractions `{0.1, 0.9}`, two-sided and mixed rows), wall clock including every
   `update_preconditioner!` call: G1 24/24, G2 24/24 (§9.6). Not measured: the full sweep's
   `κ = 1e3`, fraction `0.5`, the Kronecker family, sparse `n = 5000`, and G1 at `1e-8`.
5. **FullKKT reads `A[i, j]` entry by entry at every factorization.** Under the IPM, which
   refactorizes every iteration, this cost is paid every iteration; **measured** (S10,
   `bench/ipm_backends.jl`, `fill_ms` against one factorization): 3% of an iteration on a
   dense pair (`n = 200, m = 400`), 7.5% on a sparse one (`Lasso`, `816×816`), 16% on a banded
   one (`banded_qp(200, 300)`) — it does not dominate an iteration, so `FullKKT` is unchanged.
   Reduced backends under IPM carry `w` up to `1/δ_d`; with `δ = 1e-8` the reduced
   matrix's conditioning bound is `~1e8‖Ã‖²`. **Measured** at `n ≤ 200`: a Cholesky of that
   matrix (dense and CHOLMOD) reproduces the Bunch–Kaufman KKT outer counts on 96 of 96 runs,
   equality rows included (`kktldl` groups). The structured reduced backends (diagonal,
   tridiagonal, banded, block, sparse reduced) are now measured at `δ = 1e-8`, LPs included
   (S10, `bench/ipm_backends.jl`): each reaches the full KKT factorization's outer count within
   `2×` on every variant it accepts; only the low-rank family fails, on its linear programs, and
   is routed to `FullKKT` (§5, §8.8). Larger `δ` is not a remedy: it costs outer convergence
   (§8.5, **measured**).
6. **Infeasibility detection.** Certificate tests pass on all 40 random infeasible runs
   (`n = 20, m = 40`, seeds 1–10, scaling 0 and 10, primal and dual cases). Primal infeasibility
   detected at iteration 5, dual at iteration 3. A stall-based rule (before S9) would lose
   detections because primal-infeasible mu rises for up to 16 iterations before the certificate
   appears. HSD is not needed.
7. **Derivatives after an IPM solve**: inactive-row multipliers are `O(μ_final)`, above the
   `sqrt(eps)` threshold at default tolerances (`derivative.jl:52`); polishing cleans the
   active set and is required for IPM derivatives in v1.
8. **A caller's preconditioner that is not SPD** makes Krylov throw when `rᵀM⁻¹r` is negative
   or NaN, or makes CG stall to its budget; §9.3 states how each surfaces and what the status is.
9. **`NUMERICAL_ERROR`** is a new `Status` value; every switch in §8.6 must grow with it.
10. **Test churn**: ~120 lines outside `src/ext` read moved fields (§1.9).
11. **Step-3 registration mechanics** untested here.

---

## 8. Mehrotra IPM specification

### 8.1 Problem form and row classes

The IPM runs on the equilibrated `prob` (`P̃ q̃ Ã l̃ ũ`). Each row is classified at setup and
in `update!` into an `Int8` class vector `rclass` and two `Bool` masks `has_l`, `has_u`:

| class | condition (`loose = INFTY(T) * MIN_SCALING(T)`, as `rho_class`) | slacks |
|---|---|---|
| free (`-1`) | `l̃ < −loose && ũ > loose` | none; `y_i ≡ 0`; `w_inv = 1/δ_d` |
| equality (`1`) | `l0 == u0` exactly | none; free multiplier `y_i`; `w_inv = δ_d` |
| inequality (`0`) | otherwise; `has_l = l̃ > −loose`, `has_u = ũ < loose` | `s_l, z_l` if `has_l`; `s_u, z_u` if `has_u` |

    s_l = Ãx − l̃,  s_u = ũ − Ãx,   y_i = z_u,i − z_l,i   (inequality rows; y ∈ N_[l,u](Ax))

Storage: `s_l s_u z_l z_u` as four length-`m` vectors, unused entries `1` for slacks and `0`
for multipliers so every loop is branch-free apart from the mask; `N_s = count(has_l) +
count(has_u)`. Variable bounds arrive as rows of `A` (MOI does this,
`MathOptInterfaceExt.jl:130-139`); there is no separate box-slack path.

### 8.2 Newton system, weights and step recovery

Residuals of the *original* (scaled) problem at the current iterate:

    r_d = P̃x + q̃ + Ãᵀy
    r_l = Ãx − l̃ − s_l,   r_u = ũ − Ãx − s_u   (masked)
    r_e = Ãx − l̃                              (equality rows)
    r_cl = s_l ∘ z_l − σμ e (+ Δs_a ∘ Δz_a),  r_cu likewise

**Regularized rows.** The dual regularization `δ_d` sits on each *side's* slack row, not on
the eliminated multiplier `y`:

    Ã_iΔx − Δs_l,i + δ_d Δz_l,i = −r_l,i          (lower side, if has_l)
    −Ã_iΔx − Δs_u,i + δ_d Δz_u,i = −r_u,i         (upper side, if has_u)
    z_l ∘ Δs_l + s_l ∘ Δz_l = −r_cl,   z_u ∘ Δs_u + s_u ∘ Δz_u = −r_cu
    (P̃ + δ_p I)Δx + ÃᵀΔy = −r_d,   Δy = Δz_u − Δz_l

Substituting the slack row into the complementarity row of each side gives, with

    d_l = s_l + δ_d z_l,   d_u = s_u + δ_d z_u

    Δz_l = −(r_cl + z_l ∘ (ÃΔx + r_l)) / d_l
    Δz_u = −(r_cu + z_u ∘ (−ÃΔx + r_u)) / d_u
    Δs_l = ÃΔx + r_l + δ_d Δz_l,   Δs_u = −ÃΔx + r_u + δ_d Δz_u

and therefore `Δy = Δz_u − Δz_l = w ⊙ ÃΔx + g` with

    w = z_l/d_l + z_u/d_u                                    (absent side contributes 0)
    g = (r_cl + z_l ∘ r_l)/d_l − (r_cu + z_u ∘ r_u)/d_u

which is exactly the package's KKT form with `w_inv = 1/w`, `sigma = δ_p`, `rhs_x = −r_d`,
`rhs_z = −w_inv ⊙ g`. An implementer can check the derivation by substituting `Δz_l` back:
`z_l(ÃΔx + r_l + δ_dΔz_l) + s_lΔz_l = z_l(ÃΔx + r_l) + Δz_l d_l = −r_cl`.

**Why this form and not `w = 1/(1/W + δ_d)` with `Δs = ÃΔx + r`.** With `W = z_l/s_l +
z_u/s_u`, the harmonic weight makes the solved `Δy = wÃΔx + g` differ from the recovered
`Δz_u − Δz_l = WÃΔx + g` by `(W − w)ÃΔx`, so the step is not a solution of any one Newton
system. **Measured** (`ipm_matrixfree_spike_summary.json`, `exact/design`): 0/27 instances
converge at every `δ`. Rescaling `ÃΔx` by `1/(1 + δW)` to make the two agree (`exact/
consistent`) converges 25/27 at `δ = 1e-8` and 0/27 at `1e-4` and `1e-2`. The per-side form
above (`exact/perside`) converges 27/27 at `δ = 1e-8`, 18/27 at `1e-4`, 15/27 at `1e-2`
(`spike.jl:391-396, 457-461`).

**Equality rows**: `Ã_iΔx − δ_d Δy_i = −r_e,i`, i.e. `w_inv,i = δ_d`, `rhs_z,i = −r_e,i`; no
slack recovery. **Free rows**: `w_inv,i = 1/δ_d`, `rhs_z,i = 0`, and `Δy_i` is zeroed after the
solve (one masked store), so `y_i ≡ 0` and the row contributes `δ_d ãᵢãᵢᵀ` to the reduced
matrix and nothing else. Zeroing leaves the solved and the applied `Δy` apart by `δ_d ÃΔx` on
that row; it acts as an extra proximal term `δ_d ãᵢãᵢᵀΔx`, which vanishes at a fixed point.
**One-sided rows**: the absent side's terms are masked out of `w`, `g` and the recovery; its
slack and multiplier stay at their placeholders. After `solve_multiplier!` returns `Δx, Δy`,
the recovery needs one `mul_A!` for `ÃΔx` and four masked elementwise passes.

**Measured with every row class** (`ipm_rowtypes_spike.json`, `rowtypes` groups; spike 2's 18
dense and 30 sparse instances re-planted with 20% equality rows, or with 20% equality, 20%
lower-only, 20% upper-only, 10% free and 30% two-sided rows, or as `n/2` equality rows only):
exact solves reach G1 and `1e-8` on 18/18 and 30/30 for every mix, in 6–10 outer iterations
with mixed rows and 2 with equality rows only.

**Centering safeguard.** Mehrotra's `σ = (μ_a/μ)³` is floored:

    σ ← max(σ, min(1, 0.1 · ‖r‖∞ / μ)),   ‖r‖∞ = max(‖r_d‖∞, ‖r_l‖∞, ‖r_u‖∞, ‖r_e‖∞)

so `μ` is not driven toward zero while the Newton residuals are still large. **Measured**
(spike 1, `exact/perside` without and with the floor): at `δ = 1e-8` nothing changes (27/27
converged either way); at `δ = 1e-4` convergence rises from 18 to 21 of 27 and G1 falls from 25
to 24; at `1e-2` convergence stays at 15 and G1 rises from 15 to 18. It costs one comparison.

### 8.3 Starting point and `seeded`

`InteriorPointWorkspace.seeded::Bool` is set by `warm_start!`, by `x0`/`y0`, and by a completed
`solve!`; cleared by `cold_start!`. With default equilibration (`scaling = 10`), equality-only problems converge in 1 outer iteration; with `scaling = 0`, they converge in 2. A solve starts:

1. If `!seeded`: one factorization and one `solve_system!` with `w = w_inv = 1`, `sigma =
   δ_p`, `rhs_x = −q̃`, `rhs_z = t` (`t_i` = midpoint of a two-sided row, the finite bound of a
   one-sided one, `l̃_i` of an equality, `0` of a free row), i.e.
   `(P̃ + δ_p I + ÃᵀÃ) x = −q̃ + Ãᵀt`. If `seeded`: `x` is the workspace's. On an iterative
   backend this solve runs at tolerance level `0` (`set_tolerance_level!(ls, 0)`), so it solves
   from `x = 0` to the `eps(T)·max(1, ‖rhs‖)` floor; it counts toward
   `InteriorPointWorkspace.cg_misses` (§9.4) like every other solve.
2. `s_l = Ãx − l̃`, `s_u = ũ − Ãx` (masked); `θ = max(0, −1.5·min(s))`; `s .+= θ`; `z .= 1`
   on masked entries, or from `y`: `z_u = max(y, 0) + 1`, `z_l = max(−y, 0) + 1`.
3. Mehrotra's balancing: `δ_s = ½(sᵀz)/(eᵀz)`, `δ_z = ½(sᵀz)/(eᵀs)`; `s .+= δ_s`, `z .+= δ_z`.
   Equality multipliers keep `y` or start at `0`.

Sums, minima and `θ` run over the masked entries only. With `N_s = 0` (no inequality side,
§8.4) steps 2 and 3 are skipped. This is the starting point all three spikes ran
(`spike.jl:337-354`; `ipm_rowtypes_spike.jl` for the row classes and `N_s = 0`). `warm_starting = false`
clears `seeded` before every solve. No claim is made about the iteration count of a
warm-started re-solve; S13 records the measured number.

### 8.4 One outer iteration

Outside `ipm_step!` (allocation allowed, `:warm` audit rows):

1. `μ = (s_lᵀz_l + s_uᵀz_u)/N_s`; `‖r‖∞`; weights per §8.2 (`weights!`);
   `set_refresh_index!(ls, k)`; `refactor_weights!` (for `IndirectCG` this runs
   `update_preconditioner!`, §9.3); on `false`, `bump_regularization!` (§8.5) and retry with
   the same `k`.

Inside `ipm_step!` (`:hot`, no allocation):

2. **Predictor**: `r_cl = s_l∘z_l`, `r_cu = s_u∘z_u`; `g`, `rhs_x`, `rhs_z`;
   `set_tolerance_level!(ls, min(μ, ‖r‖∞))`; zero `Δx` (the CG start, §3.6);
   `solve_multiplier!` → `Δx_a, Δy_a`; recover
   `Δs_a, Δz_a` per §8.2; `α_a` = largest step in `(0, 1]` keeping `s, z ≥ 0` (one step length
   for primal and dual: the dual residual couples `x` and `y`); `μ_a`, `σ = (μ_a/μ)³` with
   the floor of §8.2.
3. **Corrector**: `r_cl = s_l∘z_l + Δs_a∘Δz_a − σμ e` (and `u`); rebuild `g`, `rhs_z`; same
   factorization; zero `Δx`; `solve_multiplier!`; optional refinement (§8.5); recover `Δs, Δz`.
4. `α = min(1, τ·α_max)`, `τ = step_fraction = 0.99`; update `x, y_E, s, z`; `y = z_u − z_l`
   on inequality rows; zero `Δy` on free rows.

**No inequality side (`N_s = 0`).** `μ ≡ 0` and the tolerance level is `‖r‖∞`; each iteration
makes one solve (steps 2 and 3 collapse: `r_c = 0`, no `σ`) and takes `α = 1`, which is the
proximal-point iteration on the regularized KKT system. **Measured** (`ipm_rowtypes_spike.json`,
`eqonly` groups, `n/2` equality rows): 2 outer iterations on 18/18 dense and 30/30 sparse
instances with exact solves; 18/18 in 2 with the lagged Cholesky and 30/30 in 2–5 with the
incomplete LDLᵀ.

Then residuals and termination (§8.6) every `check_termination` iterations and at exit, with
`time_limit` and `InterruptException` handled as in `admm.jl:161-225`. No Gondzio
correctors, no adaptive `τ`, no separate step lengths in v1.

### 8.5 Regularization model and defaults

One model for every backend: **primal–dual proximal regularization** — `δ_p I` on the primal
block and `δ_d` on each slack row (§8.2) at every iteration, the right-hand side built from
the original residuals, the regularization never refined away, termination measured on the
original residuals. The motivation is Friedlander–Orban's primal–dual regularization
(**literature**), which regularizes every primal variable including the slacks; here only `x`
carries `δ_p` and the per-side `δ_d` term is the dual regularization in slack form, so the
scheme is a variant and its convergence is a **measured** property (two-sided rows: spikes 1
and 2; every row class: spike 3), not a cited theorem.

**What `δ` costs (measured, exact solves, `δ_p = δ_d = δ`).** Convergence to `eps = 1e-8`
(`ipm_matrixfree_spike2_nog2stop.json`, `exact` rows): dense 18/18 at `δ = 1e-8` and `1e-6`,
14/18 at `1e-4`, 11/18 at `1e-2`; Kronecker 36/36, 33/36, 28/36, 21/36; sparse 30/30, 30/30,
20/30, 18/30. G1 (`eps = 1e-6`, referee `≤ 1e-5`): dense 18, 18, 17, 12 of 18. Outer
iterations on the dense instances that do converge: 6–11 at `1e-8` and `1e-6` (two outliers
at 86–87 at `1e-6`), 6–71 at `1e-4`, 10–97 at `1e-2`. A larger `δ` therefore buys nothing the
linear solver can use without paying it back in outer iterations; the preconditioned-CG runs
of §9.1 confirm that their G1 ceiling equals the exact one at the same `δ` and falls with it.

- **Defaults**: `reg_primal = reg_dual = 1e-8` for `Float64` on every backend, direct or
  iterative; there is no separate `reg_indirect`. Where that default is **measured** at
  `δ = 1e-8` (outer counts equal to Bunch–Kaufman on the `FullKKT` matrix, `n ≤ 200`,
  `ipm_rowtypes_spike.json` `kktldl` groups, two-sided and mixed rows, dense and sparse, 96 runs
  each): `FullKKT` (spikes 1–2); no-pivoting quasi-definite `LDLᵀ` through CHOLMOD `ldlt` (the
  factorization `SparseKKT` calls) and through LDLFactorizations `ldl` (`LDLKKT`), each
  identical on 96/96 with `refine_iter = 1`; without refinement 93/96 identical and 95/96
  within ±2 (worst +7 and +5, both sparse with mixed rows); Cholesky of the reduced matrix,
  dense and CHOLMOD (`ReducedCholesky`, `SparseCholmod`), identical on 96/96; the
  preconditioned `IndirectCG` of §9. The package's own backend code for these is measured in
  S8; the structured reduced backends (diagonal, tridiagonal, banded, block, sparse reduced) in
  S10, at `δ = 1e-8` with LPs included — every one of them, and only those, reproduces the
  `FullKKT` outer count within `2×`; the low-rank family does not (§5, §8.8).
  Held in `InteriorPoint{T}` in scaled space, changeable through
  `update_settings!(ws, InteriorPoint(…))` without refactorization.
- **Element types** (§10.8 question 3): the IPM is generic over `T <: Real` and refuses none.
  The `1e-8` defaults above (`reg_primal`, `reg_dual`, `eps_abs`, `eps_rel`, `eps_prim_inf`,
  `eps_dual_inf`) and the short-step threshold of §8.7 come from `ipm_floor(T)`: `1e-8` when
  `precision_eps(T) ≤ eps(Float64)`, `sqrt(precision_eps(T))` otherwise, with
  `precision_eps(T) = eps(float(T))` (a wrapper type such as `ForwardDiff.Dual` defines `eps`
  through its value type; no dependency is added). The divergence bound is
  `1/sqrt(precision_eps(T))`. A value passed explicitly is used as given. `Float64` keeps
  exactly `1e-8` (decisions 5 and 6); `sqrt(eps(Float64))` is `1.49e-8`, so a plain
  `max(1e-8, sqrt(eps(T)))` would move the `Float64` defaults. The `1e-4`/`1e-2` columns above
  are `Float64` runs. **Measured in `Float32`**: S10's 18-instance dense grid (`FullKKT`,
  `eps = 1e-4`, `δ = sqrt(eps(Float32))`, `scaling = 0`) — all 18 solve, referee
  `1.1e-5`–`1.0e-4`, iteration count equal to the same `Float64` tolerance on 15/18 and within
  2 on the rest (`ipm_backends.json` `float32`). The `Float32` defaults
  (`eps = δ = 3.45e-4`) are covered by the §8.10 test items, not by a saved benchmark.
- `is_convex(T, P, reg_primal)` at setup; documented as the IPM's shift.
- Dynamic: on `factorize!` returning `false`, multiply both by `10` and retry, up to
  `max_reg_bumps = 5`; then `NUMERICAL_ERROR` with the last point. The table says what each
  bump costs in outer convergence, so a bumped run is reported with its final `δ` in the
  verbose footer.
- Refinement (direct backends only, `refine_iter` default `1`): one step against the
  *regularized* operator to correct rounding in the factorization —
  `r = [P̃Δx + δ_pΔx + ÃᵀΔy − rhs_x ; ÃΔx − w_inv ⊙ Δy − rhs_z]` (the second block is zero
  by construction after `solve_multiplier!`), correction through the same factorization.
  For `IndirectCG` `refine_iter` is `0`: **measured** (`spike.jl` restart experiment, 238 slow
  solves at `δ = 1e-8`, unpreconditioned), restarting CG every 200 iterations reached the
  tolerance on 124 solves against 144 unrestarted, at 318k against 288k iterations.
- The `InteriorPoint` docstring states the conditioning bound of a reduced backend,
  `κ ≤ (λ_max(P̃) + ‖Ã‖²/δ_d)/(λ_min(P̃) + δ_p)`, as the reason a structured backend is measured
  before it is routed (S10). It is not checked by the code. Every QP instance has
  `λ_min(P) = 1e-2`, so the bound stays near `‖Ã‖²·1e10`, below `1/eps(Float64)`; an LP
  (`P = 0`) reaches `‖Ã‖²·1e16`. S10 includes an LP on each structured backend that accepts
  one: every reduced backend still reaches the `FullKKT` outer count within `2×` there; only
  the low-rank family's LPs fail (§5, §8.8).
- The bump is triggered by `factorize!` returning `false` only. An inaccurate factor that
  does not fail is not detected by the backend; its symptom is the outer stall rule (§8.7).

### 8.6 Termination and status

Residuals from the S4 kernels on `(x, y, z = clamp(Ãx, l̃, ũ))`, tolerances `eps_prim`,
`eps_dual`, `eps_duality_gap` with the same formulas as ADMM, so `eps_abs`, `eps_rel`,
`check_dualgap`, `scaled_termination` mean the same thing under both algorithms — the
*defaults* differ (`1e-8` vs `1e-3`, §8.10) and the API and MOI docs say so. `SOLVED` when all
three pass; at `max_iter`, retry at ten times for `SOLVED_INACCURATE`, else
`MAX_ITER_REACHED`. `check_termination = 1` by default: every iteration pays the residual
kernels, which an IPM's iteration count justifies.

`Status` gains `NUMERICAL_ERROR` (regularization ceiling, inner-solve failure, stall without
certificate, NaN residual — an IPM NaN after `is_convex` passed is a numerical failure, not
`NON_CONVEX`). Switches to update: the `Status` docstring ("Eleven values", `types.jl:4`),
`status_name` (`admm.jl:28-40`), the export list (`src/PureOSQP.jl:27-31`), MOI
`_TERMINATION` (→ `MOI.NUMERICAL_ERROR`) and `ResultCount` (→ `0`,
`MathOptInterfaceExt.jl:203-206`), `docs/src/algorithm.md:383-386`, the API table.
`has_solution` is `false` for it.

### 8.7 Infeasibility detection

v1: certificate tests, not a homogeneous embedding.

- The tests are the package's, formulation-independent, after S4 on `(prob, buffer, eps)`.
  They project the buffer in place (`termination.jl:209`), so `InteriorPointWorkspace` owns `cert_x`,
  `cert_y`; `build_solution` reads them.
- Every termination check runs the certificate tests at `eps_prim_inf`/`eps_dual_inf` on
  `(x/‖x‖∞, y/‖y‖∞)` if a certificate exists; with the `*_INACCURATE` retry at ten times. If
  a certificate does not exist at max_iter, the run returns `MAX_ITER_REACHED`.
- Short-step rule: once `α < ipm_floor(T)` (`1e-8` in `Float64`, §8.5) on three consecutive iterations, or `μ` (`‖r‖∞` when
  `N_s = 0`) not decreasing for ten consecutive iterations, the tests run every iteration
  on `(Δx_k, Δy_k)` (step direction) and normalized iterates. If neither certificate passes,
  the run ends `NUMERICAL_ERROR` with the last point. Divergence guard: `‖x‖∞` or `‖y‖∞`
  above `1/sqrt(eps(T))` relative to the data triggers every-iteration testing without ending
  the run; exceeding a multiple of that ceiling ends `NUMERICAL_ERROR`.
- Measurements (S9, 40 random infeasible runs, `n = 20, m = 40`, seeds 1–10, scaling 0 and 10):
  primal-infeasible and dual-infeasible cases detected at iterations 5 and 3 respectively,
  before any stall condition fires. A rule that ended runs on a flat `μ` lost detections because
  primal-infeasible `μ` rises for up to 16 iterations before the certificate appears.

### 8.8 Backends and selection

`IPMSelection` (§5). The IPM ladder has no `formed_rung` step and ends at `FullKKT`, skipping the low-rank rung: a diagonal `P` with a `RowCoupled` `A` reaches `FullKKT` instead, since the Woodbury solve does not reach tolerance on that pair's linear programs (measured, S10). The `sparse_kkt_backend` is split into the density pre-gate and `factored_kkt_backend`, which the IPM path calls directly; a sparse pair whose sparse KKT factor fails the fill gate lands on the sparse reduced backend, observed on `banded_qp(200, 300)`. `IndirectCG` under `:ipm` follows §9. Kronecker declines. GPU arrays are
refused for `:ipm` through the GPU extension's `choose_backend` method.

This fill gate, and `block_rung`'s single-block decline and `lowrank_rung`'s crossover
decline, are what `:auto`'s ladder applies. A caller who names `linsys = :sparse`, `:block`
or `:lowrank` tries the same rungs at the same threshold first — which is what reaches the
same backend `:auto` would wherever the gate already passes — and only when every one of
those declines retries `kkt_rung`/`reduced_rung`/`block_rung`/`lowrank_rung` with the gate
disabled (`fill_limit = Inf`, `require_multiple = false`, `require_crossover = false`), which
is what reaches a backend for a pair `:auto` would send to the dense terminal instead.
Refusing still means the representation itself cannot serve the pair or the factorization
does not succeed at either stage. `:auto` keeps the gates above unchanged.

### 8.9 Equilibration

Identical to ADMM: the workspace holds `prob` with `D E c`; iterates are scaled;
`Solution.x = D ⊙ x̃`, `Solution.y = E ⊙ ỹ / c`; residuals are reported unscaled by the shared
kernels. `scaling = 0` turns it off. Under `:ipm` a caller-supplied `preconditioner` requires
`scaling = 0` (§9.2), for matrices and operators alike, so the matrix the preconditioner
approximates is built from the caller's own `P` and `A`; `probe = true` is refused with a
preconditioner in v1.

### 8.10 v1 support matrix

| feature | IPM v1 |
|---|---|
| `update!(q, l, u, P, A)` | yes; validate → adopt → reclassify; no refactorization at update |
| `warm_start!` / `x0, y0` | seed (§8.3); sets `seeded` |
| `cold_start!` | zeroes `x, y`, clears `seeded` |
| `update_settings!` | yes; `linsys`, `scaling` fixed; `reg_*` free |
| `polishing` | yes, through the kernel; required before derivatives |
| derivatives | yes on `SOLVED` with polishing; a non-materializable operator refuses first, since it was never a candidate for polishing |
| MOI | `InteriorPoint()`; `BarrierIterations = iter`; `NUMERICAL_ERROR` mapped; switching `algorithm` refuses a raw setting left over from the other algorithm |
| `time_limit`, `Ctrl-C` | as ADMM |
| `verbose` | an `Options` field (§11); prints a header, one row per termination check with `mu` and `alpha` in place of ADMM's `rho`, and a footer naming the run time, adding a `cg iters` row column and total/missed-solve footer lines on the matrix-free backend (§9.4) |
| `linsys = :indirect` (matrices or operators: `ProductOperator`, LinearMaps, SciMLOperators) | only with a caller-supplied `preconditioner` (§9) and `scaling = 0`; operators: `P` declared `posdef`, no polishing, no derivatives, no `:kkt`; never chosen by `:auto`. Without a preconditioner, or with `scaling ≠ 0`, `setup` refuses by name. `update_settings!` does not change `linsys`; a regularization change reaches the next solve's own factorization, which runs `update_preconditioner!` regardless. |
| accelerator, GPU arrays, `profile_primdual` | refused by name / absent |
| element type `T <: Real` | generic, none refused (§10.8 question 3); defaults from `ipm_floor(T)` (§8.5) |
| `Float32` | `FullKKT` and reduced backends; tests: dense QP, LP, equality/one-sided/free rows through `:kkt` and `:dense`, referee `< 10·sqrt(eps(Float32))` (`test/ipm_tests.jl`); measured per §8.5 |
| `BigFloat` | `FullKKT` and `ReducedCholesky`; test: tiny QP through `:kkt` at `eps = 1e-20`, referee `< 1e-18` |
| `ForwardDiff.Dual` | reduced backends (`:auto` reaches `ReducedCholesky`); `FullKKT` fails with a `MethodError` (`bunchkaufman!` has no `Dual` method), documented in `docs/src/algorithm.md`; tests: objective derivative vs `x[1]` (`test/ipm_backends_tests.jl`) and vs a central difference of the `Float64` objective to `1e-6` (`test/ipm_tests.jl`) |

`InteriorPoint` parameters: `reg_primal = reg_dual = ipm_floor(T)`, `max_reg_bumps = 5`,
`refine_iter = 1` (left out, it resolves from `linsys` at setup and is `0` when
`linsys = :indirect`, per §8.5; an explicit value is honored under either default),
`step_fraction = 0.99`, `cg_fail_limit = 3`. `default_options(InteriorPoint(), T)`:
`max_iter = 100`, `eps_abs = eps_rel = eps_prim_inf = eps_dual_inf = ipm_floor(T)`,
`check_termination = 1`, `cg_max_iter = 500` (spike 2's cap; spike 1 and spike 2's three
`n = 1000` records ran 2000; at 500, `cg_lagchol5` would have hit the cap on two of those three,
whose maxima were 618 and 678), `cg_tol_fraction = 0.1`, and the options both algorithms
share: `time_limit = Inf`, `scaling = 10`, `check_dualgap = true`, `scaled_termination = false`,
`polishing = false`, `polish_refine_iter = 3`, `delta = 1e-6`, `warm_starting = true`,
`linsys = :auto` (§11). `ipm_floor(T)` is `1e-8`, `sqrt(eps(T))` for `T` coarser than
`Float64`. (`cg_tol_reduction` is ADMM's idle-solve rule and has no IPM counterpart.)

**Fields used by `solve!`:** `time_limit` (clock includes starting point); `eps_prim_inf`,
`eps_dual_inf` (certificate tolerances); `max_reg_bumps` (limit on regularization bumps, each
multiplies both `reg_primal` and `reg_dual` by 10 and triggers a full `factorize!`; counted
per solve); `reg_primal`, `reg_dual` (run in scaled space, changeable through `update_settings!`
with no refactorization at update time on any backend, direct or `:indirect`: every solve
resets the regularization from `ws.algorithm` before its first iteration and factorizes there
regardless, which runs `update_preconditioner!` on `:indirect`). A starting point without
seeding returns `NaN` for `x` and `y` while the workspace keeps the last iterate and clears
`seeded`.

### 8.11 Validation plan

1. **Referee** `kkt_residuals` (`test/helpers.jl:36-60`) `< 1e-5` on the structural corpus
   (`corpus_tests.jl:12-42`) under `:ipm` with `:auto` and `:kkt`; on `random_qp` sizes; on
   every structured family of `selection_tests.jl:26-103`, with the backend name asserted.
2. **Oracle**: objective vs `osqp_ref` to `1e-6·max(1,|obj|)`; `x` vs the ADMM solution at
   `eps = 1e-9` to `1e-6`.
3. **Spike reproduction** (test): the spike-1 generator at `n = 200`, two-sided and with spike
   3's mixed row classes, reproduces the exact-solve outer counts within ±2 through `FullKKT`
   and `SparseKKT` (S8).
4. **Clarabel** (`bench/ipm_vs_clarabel.jl`, Clarabel via `bench/solvers.jl:46-51`): `x` to
   `1e-5` relative on the six `CASES`, iteration counts and wall clock side by side. (Both
   spikes validated their exact runs against Clarabel: 0 of 84 instances off by more than
   `1e-6` in objective.)
5. **Iteration sanity** (test): `iter ≤ 40` on corpus and c-suite; `≤ 25` on the dense random
   QPs. The warm-started re-solve count is recorded in the snapshot, not asserted.
6. **c-suite** under `:ipm`, plus S9's random infeasible cases.
7. **Generic `T`** (§8.10 rows): `Float32` end to end through `:kkt` and `:dense`;
   `BigFloat` on a tiny QP through `:kkt`; `ForwardDiff.Dual` on `:auto` (reduced path, no
   `bunchkaufman!`) against a central finite difference — done, `test/ipm_tests.jl`,
   `test/ipm_backends_tests.jl`.
8. **Operators**: §9.6.
9. **Ambiguous status**: a problem that is both primal-infeasible and dual-infeasible may be
   reported as either status, provided its certificate passes the independent check
   (`is_primal_infeasible` or `is_dual_infeasible`). Example: test case `c_suite_tests.jl`
   A34, `u = [0, 3, Inf]`, returns `DUAL_INFEASIBLE` under `:ipm`.

### 8.12 StrictMode and trim

Audit rows per backend: `ipm_step!(W)` and `ipm_residuals!(W)` hot (`typestable`,
`noalloc`); `solve_system!`/`solve_multiplier!` hot; `refactor_weights!`, `factorize!`,
`check_termination(W, Bool)`, `solve!(W)` warm (`typestable`); the `:indirect` row measured.
`check_termination` never touches a factorization and keeps `typestable` on every backend. The
sparse KKT family (`SparseCholmod`, `SparseKKT`, `SparseLDL`, `LDLKKT`) claims neither
`typestable` nor `noalloc` on `factorize!`, `refactor_weights!` and `solve!`: their shared
`reduced_gram`/`kkt_gram` assembly (`PureOSQPSparseArraysExt`) routes through SparseArrays'
sparse-matrix constructors, whose dimension-validating error path a static analyzer cannot see
past, which costs the caller its inferrability — measured as JET's `internal instability /
runtime dispatch` finding on `factorize!` and `solve!` (88 reports on the KKT family). The docs state
that the IPM's per-iteration allocation is the backend's factorization (and, for
`IndirectCG`, the caller's `update_preconditioner!`). Rules: masks and classes preallocated;
every elementwise update a two-schedule function in `elementwise.jl` style (`max_step`,
`complementarity!`, `weights!`, `recover_slack_steps!`); loops over `eachindex`, no `findall`,
no broadcasting on `Vector`; `norm_inf` only; `InteriorPoint{T, T, T, Int}` and `Options{T}` concrete; `lazy"…"` messages;
verbose via `Core.stdout`. The `try`/`catch` that turns Krylov's definiteness throw into a miss
(§9.3) lives in a `@noinline` helper outside the audited `solve_system!` kernel, so the
`noalloc` row does not see the catch path. Trim entries: `solve_ipm_default`, `solve_ipm_kkt`,
`solve_ipm_unscaled`, `solve_ipm_polish`, `solve_ipm_diagonal`, `solve_ipm_sparse_kkt`,
`solve_ipm_indirect` (dense pair, `:indirect` with a `Cholesky` preconditioner, added in S11),
`setup_ipm_update`, `derivatives_ipm`; `solve_ipm_operator` (`ProductOperator` pair with the
same) is added in S13.

---

## 9. Operators under the IPM

The case the user cares about most: `P` and `A` as `LinearMap`s or SciMLOperators (through
`ProductOperator`, `operator.jl`), which supply products and nothing else. Under the IPM such a
pair has exactly one backend, `IndirectCG`, and the reduced matrix it must solve,
`P̃ + δ_p I + Ãᵀ diag(w) Ã`, has a spectrum that splits as `μ → 0` — `O(w_max‖ã‖²)` on the
active-row span, `O(λ_min(P̃) + δ_p)` on the rest (**literature**: Wright 1998).

### 9.1 What the spikes established

All **measured**, at the G1 criterion "`eps = 1e-6` reached with referee `≤ 1e-5`" and, in
spikes 1 and 2, an inner stopping oracle: the explicit reduced residual `‖b − KΔx‖₂ ≤ atol_k`
(spike 2 every iteration; spike 1 through the recursive residual with warm restarts until the
recomputed residual passes). "n/N" is instances passing out of the family's instances at that
`δ`, whatever the run's status; every row class is two-sided unless spike 3 is named.

- **Every product-only option fails G1 broadly** (`ipm_matrixfree_spike_summary.json`,
  `ipm_matrixfree_spike2_nog2stop.json`). Unpreconditioned CG: 16/27 (spike 1, `δ = 1e-8`),
  11/18 dense, 27/36 Kronecker, 16/30 sparse (spike 2); at `δ = 1e-4` 24/27 / 14/18 / 29/36 /
  23/30, at `1e-2` no better than the exact ceiling minus the failures of §8.5. Exact Jacobi
  (which needs entries): 17/27, 21/30 sparse. Probed Woodbury (a Hutchinson estimate of the diagonal plus the `k ∈ {5, 20}`
  largest-weight rows): 4/27 and 8/27 at `δ = 1e-8`, *worse than no preconditioner*; its Hutchinson diagonal
  estimate had a median relative error of 0.64 (min 0.05, max 1.39 over 5172 builds). MINRES on
  the augmented system: 2/18; with the `diag(I, w)` block preconditioner 10/18 dense, 22/36
  Kronecker. TriMR/TriCG need the `(1,1)` block's inverse and are product-only only for
  `P = αI` (13/18 dense with a dense Cholesky of `P + δI`, 27–28/36 Kronecker). GPMR matched
  TriMR iteration for iteration on the Kronecker checks and adds nothing. A Kronecker rank-one
  fit of the weights helps only when the active set is itself Kronecker-structured (17/18
  Kronecker-pattern instances with 0.07–0.64× the products of plain CG; on random active sets
  5/18 fail and the rest cost more than plain CG).
- **A lagged Cholesky of the reduced matrix reaches the exact ceiling on the dense family.**
  Refreshed every third outer iteration (`cg_lagchol3`): G1 18/18 at `δ = 1e-8` and `1e-6`,
  17/18 at `1e-4`, 12/18 at `1e-2` — the exact ceiling — at 0.074–0.244× (median 0.13×) the
  products of plain CG at `1e-8` and 0.070–0.312× at `1e-6` (all 18 instances as
  denominator), inner medians 8–32 per solve, maxima 15–110. Refreshed every fifth iteration
  (`cg_lagchol5`): same G1, 0.145–0.745× products at `1e-8`. With row classes (spike 3, §9.4
  miss rule): G1 18/18 with 20% equality rows and 18/18 with mixed rows, inner median per solve
  17.0 and 16.5 against 15.2 two-sided, maxima 100 and 115.
- **The limited-memory incomplete LDLᵀ is weaker evidence.** On the sparse family
  (`cg_lldl10`, refreshed every outer iteration) G1 is 30/30 at `δ ≤ 1e-6`, but at
  `cg_max_iter = 500 > n`: on 5/30 runs a solve takes more than `n` inner iterations
  (`cg_lldl0`: 14/30). Products against plain CG at `1e-8`: 0.0027–0.66× (median ≈ 0.02×) over
  all 30 instances, 0.0027–0.28× (median 0.014×) over the 16 where plain CG passed G1. 1182 of
  6801 factor builds needed a diagonal shift to come out positive definite
  (`spike2.jl:295-302`). With row classes (spike 3) G1 falls to 26/30 (20% equality rows) and
  27/30 (mixed rows), every failure a solve at the 500 cap, under every stopping rule measured.
- **Unpreconditioned CG and equality rows.** Median inner iterations per solve 118 on the
  two-sided dense instances, 392 with 20% equality rows, 406 with mixed rows, 426 with equality
  rows only (spike 3); G1 11/18, 6/18, 7/18, 12/18.
- **The Kronecker family has no measured reference preconditioner.** Spike 2 ran plain CG,
  MINRES, TriMR/TriCG and the Kronecker fits there, none of the preconditioners above.
- **`δ` does not help a preconditioned solve.** For `lagchol*` and `lldl*` the G1 count at
  `δ = 1e-4` and `1e-2` equals the exact solver's, which is lower than at `1e-8` (§8.5). (Not
  for `cg_jacobi`, 25 against 29 at `1e-4`, nor `cg_kron_rank1_geo`, 28 against 31.)
- **Restarting CG is not refinement** (§8.5).
- **Not measured**: `n ≥ 500` for the caller-supplied preconditioners (three dense `n = 1000`,
  `κ = 1e6` instances in `ipm_matrixfree_spike2.json` only, cap 2000); wall clock, and the cost
  of building any preconditioner (products were counted, factorizations were not); the
  Kronecker family with a reference preconditioner; infeasible problems; the ADMM-operator
  comparison.

Consequence: the package ships no product-only preconditioner, and `:ipm` runs `IndirectCG`
only when the caller supplies one.

### 9.2 What runs and what is refused

- `setup(P, q, A, l, u, InteriorPoint(); linsys = :indirect, preconditioner = M, scaling = 0)`
  with an operator pair or with matrices: allowed. `IndirectCG` is built with `M`; the ladder
  never chooses it (§5).
- A caller-supplied `preconditioner` (anything other than `nothing`, `IdentityPreconditioner()` or `JacobiPreconditioner`)
  requires `scaling = 0` under every algorithm, ADMM included: the preconditioner approximates the caller's own
  `P + σI + Aᵀdiag(w)A`, which equilibration would change.
- Passing `preconditioner` with any `linsys` other than `:indirect` throws an `ArgumentError`, so it is never silently ignored.
- The same without `preconditioner`, or an operator pair with `linsys = :auto`: `setup` throws
  an `ArgumentError` naming the requirement — "the interior-point method uses conjugate
  gradients only with a caller-supplied `preconditioner`; measured without one, or with the
  Jacobi diagonal, it does not reach the tolerance on most problems. Pass one, choose a direct
  `linsys`, or use `OperatorSplitting()`." Unpreconditioned and exact-Jacobi CG failed G1 (§9.1:
  16/27 and 17/27 dense, 16/30 and 21/30 sparse), so decision §10.2 refuses them for matrices
  as for operators. `JacobiPreconditioner` stays the ADMM default.
- Under `InteriorPoint()`, a `preconditioner` and `probe = true`: `setup` throws an
  `ArgumentError` — the preconditioner must approximate the caller's own `P + σI + Aᵀdiag(w)A`
  without equilibration. The refusal is implemented under `:ipm` only; `OperatorSplitting()` with
  a `preconditioner` and `probe = true` is not refused by this check.
- There is no experimental product-only path.

### 9.3 Preconditioner interface

```julia
"""
    update_preconditioner!(M, prob, wt, k::Int) -> M

Refresh the preconditioner of `P + wt.sigma*I + A' * Diagonal(wt.w) * A`, where `P` and `A`
are the matrices or operators passed to `setup` (a preconditioner requires `scaling = 0`, so
no equilibration intervenes). `wt.w` and `wt.sigma` are the only documented reads of `wt`;
`prob` is passed for dispatch and is not part of the documented interface.

`k` is the refresh index the algorithm sets before calling: under `InteriorPoint()`, `-1`
for the starting-point solve and the outer iteration `0, 1, 2, …` afterwards; a
regularization retry calls again with the same `k` and a larger `wt.sigma`. Under
`OperatorSplitting()`, `k` is the number of refactorizations so far.

Called from `IndirectCG`'s `factorize!` and `refactor_weights!`, before the solves that use
it. The returned object replaces `M` and must have the same type. Refresh lazily by
returning `M` unchanged: the reference lagged Cholesky rebuilds when `k < 0`,
`k % every == 0`, or `wt.sigma` differs from the value it was built with.

`M` is applied through `LinearAlgebra.ldiv!(y, M, x)`, so a `Cholesky`, `LDLt` or any
factorization object is usable as it is, and `M` must be symmetric positive definite.
"""
update_preconditioner!(M, prob, wt, k::Int) = M          # default: never refreshed
```

- `IndirectCG{T,V,K,M}` carries `M` as a type parameter (static dispatch, trim-safe). The
  `preconditioner` keyword of `setup` fixes the type at construction; `update_preconditioner!`
  must return that type (checked, `ArgumentError` otherwise). `k` is `ls.refresh_index`
  (§3.6).
- Krylov is called with `ldiv = true`, so the user writes one method: `ldiv!(y, M, x)`.
  `IdentityPreconditioner` exists: it maps to Krylov's `I`, so CG skips the preconditioned vector.
  `JacobiPreconditioner` is a mutable struct with a `const dinv` field; it is the default everywhere.
  Built-in `JacobiPreconditioner{V}` holds `dinv` from `reduced_diagonal!` (the hook at
  `operator.jl:208-228`, refreshed at every `update_preconditioner!` with the current `wt.w`;
  identity for an operator pair, so `dinv` holds ones) and defines `LinearAlgebra.ldiv!(y, J::JacobiPreconditioner,
  x) = (y .= J.dinv .* x)` through the elementwise-multiply kernel. That is the product today's
  `Diagonal(ls.prec)` computes under `ldiv = false`, so ADMM's `:indirect` iterates stay bit
  for bit. A `Diagonal` cannot stand in: `ldiv!(y, Diagonal(dinv), x)` applies `diag` rather
  than its inverse, and `ldiv!(y, Diagonal(d), x)` computes `x ./ d`, which differs bitwise from
  `(1 ./ d) .* x` on 26% of entries (**measured**, 1e6 log-uniform entries over 1e-8..1e8).
- **SPD requirement and how a violation surfaces.** Krylov 0.10.9's `cg!` throws
  `ErrorException("The linear operator `A` or the preconditioner `M` is not symmetric positive
  definite.")` when `rᵀM⁻¹r` is negative or NaN (`cg.jl:163,243`; **measured** with a
  `Diagonal` preconditioner holding a `-1` and one holding a `NaN`). `IndirectCG.solve_system!`
  calls `cg!` through `residual_stop_cg!`, a `@noinline` helper that matches on that exact
  message as a `String` (an exception-type match does not survive `juliac --trim`): a caught
  throw zeroes `x` and counts as a missed solve without adding its iterations to
  `Solution.cg_iters`, and the run ends `NUMERICAL_ERROR` after `cg_fail_limit` misses.
  `Solution` carries no message field, so the preconditioner is named as the likely cause only
  in documentation — the `Status` table entry and the `solve!`/`InteriorPoint` docstrings — never
  in the returned `Solution`. An indefinite `M` that does
  not trigger the throw stalls CG to `cg_max_iter`, which is also a miss (§9.4). No SPD
  pre-check is made: the one-application check `dot(x, M⁻¹x) > 0` on the right-hand side proves
  nothing about definiteness.
- Reference preconditioners for the ship gate live in `bench/ipm_preconditioners.jl`, not in
  `src/`: `LaggedCholesky` keeps an in-place `WA` (weighted `Ã`) and `K` (reduced matrix) buffer
  across refreshes, so a lagged rebuild reuses both rather than reallocating; `IncompleteLDL` is
  limited-memory LDLᵀ through LimitedLDLFactorizations, with the shift search of
  `spike2.jl:295-302`. The test suite (`test/ipm_tests.jl`) defines its own `LaggedCholesky`
  inline, under 30 lines, logging the sequence of refresh indices `k` it is called with (the
  `M.ks == -1:(iter-1)` check, §9.6); `docs/src/operators.md` carries the same worked example.
  Whether the bench version becomes an extension is §10.8.
- Core functions live in `src/core/preconditioner.jl`: `update_preconditioner!`, `IdentityPreconditioner`,
  `JacobiPreconditioner`, `set_refresh_index!`, `use_residual_stop!`, `last_solve_converged`, `inner_iterations`.
  `Solution.cg_iters` exists and holds total inner iterations of the solve under both algorithms.

### 9.4 Inner stopping test, budgets, reporting, failure

**Tolerance.** `atol_k = η · min(μ_k, ‖r_k‖∞)`, `η = cg_tol_fraction (0.1)`, floored at
`eps(T)·max(1, ‖rhs‖∞)`, set through `set_tolerance_level!` before each iteration's solves.
This is the rule all three spikes ran (`spike.jl:398, 418`).

**Stopping test.** CG starts from zero (§3.6). The inner stopping and miss rule of §9.4 is switched on by
the internal `use_residual_stop!(ls, true)` (§3.6); it is off by default, so ADMM keeps its tolerance stop.
Krylov is called with `atol = rtol = 0` and a `callback` that returns `‖ws.r‖₂ ≤ atol_k`; `CgWorkspace.r`
is the unpreconditioned residual, updated recursively (`cg.jl:158,240`), so the test costs `O(n)` per
iteration and no products. Krylov also stops by itself when the preconditioned norm satisfies `‖r‖_M + 1 ≤ 1`
(`cg.jl:249`); that stop is treated like the callback's. No residual is recomputed after the solve.

**Miss rule.** A solve is missed when it spends `cg_max_iter` iterations or Krylov throws
(§9.3); `last_reached` is `false` exactly then. A solve that stopped on either test above is
reached, whatever its explicit residual. `IndirectCG` counts its misses over the backend's
life; `InteriorPointWorkspace.cg_misses` counts consecutive misses from `last_solve_converged`, checked
with `finite_residuals` after every step and after the starting-point solve, and is what
`cg_fail_limit` compares against.

**Why the explicit residual is not tested (measured, `ipm_rowtypes_spike.json`).**
- On two-sided rows (spike 2's 18 dense instances with `cg_lagchol3`, 30 sparse with
  `cg_lldl10`, `δ ∈ {1e-8, 1e-6}`, 96 runs) the recursive-residual test gives the same inner
  iteration count on every solve and the same outer count and G1 on every run as spike 2's
  explicit-residual oracle; no solve landed between `atol_k` and `2·atol_k`, and Krylov's own
  stop never fired. Recomputing once per solve and accepting at `2·atol_k` changes nothing
  there except +3 products per solve (1.03–1.13× dense, 1.01–2.0× sparse).
- With equality rows (`w = 1/δ_d`) the explicit reduced residual is not attainable: the
  exact Bunch–Kaufman step exceeds `2·atol_k` on 28 of 272 solves (20% equality rows) and 37 of
  264 (mixed rows), by up to 104×, and those runs converge 18/18. Testing the recomputed
  residual at `2·atol_k` ends 3/18 and 4/18 of the lagged-Cholesky runs `NUMERICAL_ERROR`; the
  miss rule above passes 18/18 and 18/18 with the same outer counts as the exact solver.
  Restarting CG from its current point until the residual passes (spike 1's loop) spent 932
  and 1542 restarts running into the cap. The missed residuals are 1e5–1.6e8 times the rounding
  scale `eps·(‖b‖ + ‖PΔx‖ + ‖A‖₂‖w ⊙ AΔx‖)`, so no cheap floor on the test fixes it.
- The consequence for a caller: a solve is trusted when CG's own recursion says it converged.
  A step that is poor for another reason is not detected by the inner test; it shows as the
  outer stall rule (§8.7) ending the run `NUMERICAL_ERROR`.

**Budget.** `cg_max_iter` per solve (default `500`, §8.10); two solves per outer iteration
(one when `N_s = 0`); no refinement for `IndirectCG`. Products per outer iteration ≈
`3·(inner iterations of both solves)` plus whatever `update_preconditioner!` costs, which the
caller owns.

**Reporting.** `Solution.cg_iters` (total inner iterations of the solve, both algorithms);
the verbose row prints `cg_iters` per outer iteration; the verbose footer prints the total
and the number of missed solves.

**Failure.** `cg_fail_limit` (default `3`) consecutive missed solves end the run
`NUMERICAL_ERROR` with the last point — never `MAX_ITER_REACHED`, which would misreport a
linear-algebra failure as a convergence budget.

### 9.5 Removed

`ProbedWoodburyPreconditioner`, the Hutchinson diagonal estimate, `precond_rows`,
`precond_probes`, `reg_indirect`, and the augmented-system solvers (MINRES, TriMR, TriCG,
GPMR) are not part of the design: each was measured (§9.1) and none reaches G1 broadly
without information a product-only operator cannot supply.

### 9.6 Ship gate (S11, `bench/ipm_matrixfree.jl`)

**Grid.** Dense `make_instance` at `n ∈ {500, 1000, 2000}`, `κ(A) ∈ {1, 1e3, 1e6}`, active
fractions `{0.1, 0.5, 0.9}`, each two-sided and with spike 3's mixed row classes (20% equality,
20% lower-only, 20% upper-only, 10% free), as `LinearMap`s over the dense matrices, with
`LaggedCholesky` (`every = 3`); sparse at `n ∈ {1000, 5000}` with a fixed 10 nonzeros per row
of `A` (plus the identity), with `IncompleteLDL`; the Kronecker family of `spike2.jl` at
sides `{25, 40}` with `LaggedCholesky`; `δ = 1e-8`, `scaling = 0`, `cg_max_iter = 500`, BLAS
single-threaded on neuromancer (timings indicative, §10.3).

**What it records.** Wall clock **including** every `update_preconditioner!` call; products
and factorizations per outer iteration; per solve, the inner iterations, whether Krylov's own
stop fired, and the explicit residual over `atol_k` (bench-side, from the returned `Δx`); the
outer count and G1 of the same instance through `FullKKT` (dense) or `SparseKKT` (sparse), so
the inner rule of §9.4 is compared with an exact solve and no oracle hook is needed in the
package; G1 at both `eps = 1e-6` and the shipped `1e-8`; the ADMM-operator comparison (G3).

**Infeasible instances**, one per family and kind: primal — two rows with the same `aᵢ` and
disjoint intervals (`aᵢᵀx ≤ 0`, `aᵢᵀx ≥ 1`); dual — `P` with one zero eigenvalue along `v`,
`Av = 0`, `qᵀv < 0` (so `P` is not declared `posdef`; direct backend only). Expected: the
matching certificate status, recorded; a `SOLVED` fails the gate.

- **G1 correctness**: referee `≤ 1e-5` at `eps = 1e-6` on every case within the budget.
- **G2 bounded inner work**: with `f` the median inner iterations per solve over the first
  three outer iterations and `t` over the last three, `t ≤ max(10·f, min(100, n/10))`, no solve
  reaching `cg_max_iter`, and the largest inner count below `n`; evaluated at `n ≥ 500` only.
  The absolute term keeps a preconditioner that starts at 1–3 iterations from failing for
  reaching 30; capping it at `n/10` keeps the rule from passing unpreconditioned CG, which
  reaches `n` by construction. At the spikes' `n ≤ 200` the rule is not informative: with
  `max(10f, 100)` spike 3 passes `cg_lagchol3` 18/18 and `cg_lldl10` 23/30, with
  `min(100, n/10)` 18/18 and 19/30, and in spike 2 unpreconditioned CG on Kronecker `n = 100`
  passes the floor-100 form at `t = 88.5`.
- **G3 speed against ADMM-operator**: measured and published; decides nothing (§10.3).

**Ship verdict (S11).** The grid that decided it is decision 1(a)'s dense `LaggedCholesky`
family, reduced to `n ∈ {500, 1000, 2000}`, `κ(A) ∈ {1, 1e6}`, active fractions `{0.1, 0.9}`,
two-sided and mixed rows, G1 evaluated at `eps = 1e-6` only: 24 instances, G1 24/24 and G2
24/24, outer counts equal to `FullKKT` on all 24. The sparse family with `IncompleteLDL`
includes one instance that ends `NUMERICAL_ERROR`, capped at outer iteration 8. Not run: the
full sweep's `κ = 1e3`, fraction `0.5`, the Kronecker family, sparse `n = 5000`, G1 at `1e-8`,
and the per-solve explicit residual over `atol_k`; the ADMM-operator comparison (G3) runs at
`n = 500` only. ADMM's own path through `IndirectCG.solve_system!` runs the same calls in the
same order regardless of the preconditioner keyword, confirmed by the unchanged S0 snapshot and
`indirect_tests`.

Verdict: **SHIP** — G1 and G2 pass on the dense `LaggedCholesky` grid, two-sided and mixed rows,
so the `:indirect` path of §9.2 is documented as supported "with a caller-supplied
preconditioner, measured on these families" (§10.2). The sparse and Kronecker rows are
published per (family, preconditioner), each labeled with its own G1/G2 result; how they enter
the verdict is §10.8 question 1.

### 9.7 Ordering

The spikes have run; their verdicts shaped §8.2, §8.4, §8.5 and this section. S11 is the next
measurement that can change the design (§7.3, §7.4) and precedes S12 and S13. HSD (S9b) is
decided after S9's infeasible cases, before S11, because its cost lands on the operator case.

---

## 10. Decisions (taken with the user, 2026-09-15)

1. **Regularization model:** primal–dual proximal regularization for every backend (§8.5).
2. **Operator IPM in v1:** implemented in S11 (commit 2951256); G1 and G2 pass on the reduced
   grid of decision 1(a) (§9.6), so `:ipm` supports `linsys = :indirect` with a caller-supplied
   preconditioner. No automatic choice selects the IPM for operators in v1.
3. **Gates:** G1 as stated; G2 at `10×`; G3 measured and published, not a gate. The spike and
   the S11 benchmark use the §9.6 sweep (LinearMap over dense, `n` 200–2000, reduced to decision
   1(a)'s grid for the ship verdict) and run on neuromancer; timings there are indicative.
4. **Infeasibility:** certificate tests on directions with a stall rule (§8.7); HSD only if
   S9's random infeasible cases are not detected.
5. **`NUMERICAL_ERROR`** is added to `Status`, with the MOI mapping and docs changes.
6. **Default tolerances:** `1e-8` for `:ipm`, `1e-3` for `:admm`, documented as differing.
7. **Timing checks during the refactor (S2a):** run on neuromancer, interleaved ABBA with the
   S0 tree; indicative, not a verdict.

Decision 9, on the public API, is recorded in §10.9 below.

### 10.8 Decisions on the operator path and IPM details

The user took the recommended option of every question below; the design text above follows
them.

1. **Which runs decide the ship verdict of §9.6?**
   (a) The dense `LaggedCholesky` grid at `n ≥ 500`, two-sided and mixed rows, must pass G1
   and G2; sparse and Kronecker results are published per (family, preconditioner) with their
   own verdicts and do not block. (b) At least one reference preconditioner per family must
   pass. (c) Every reference preconditioner on every family must pass.
   **Recommended: (a).** Only the dense lagged Cholesky has a record that holds up: G1 18/18
   two-sided, 18/18 with equality and mixed rows, at the exact solver's outer counts. The
   incomplete LDLᵀ passes 30/30 only by running past `n` inner iterations (5/30 runs) and
   falls to 26/30 and 27/30 with equality rows, all at the cap. The Kronecker family has no
   measured reference preconditioner, so (b) would pass or fail it on whatever S11 happens to
   try; (c) would block on the weakest preconditioner rather than on the path.
2. **Where do the reference preconditioners live?**
   (a) `bench/` and `test/` only, `LaggedCholesky` documented as a worked example (under 30
   lines, no dependency). (b) A LimitedLDLFactorizations weak-dependency extension shipping both.
   **Recommended: (a).** The incomplete LDLᵀ is the weakest evidence (question 1) and needed a
   diagonal shift on 1182 of 6801 builds; an extension adds a dependency for a path the ladder
   never chooses.
3. **What does `:ipm` do for `Float32` (any `T` with `eps(T) > eps(Float64)`)?**
   **Decided: the IPM is generic over `T <: Real`; defaults derive from `T`; no element type
   is refused.** `Complex` is excluded by the type bound, since the method orders
   `l ≤ Ax ≤ u` and takes minimum step ratios. The defaults are `ipm_floor(T)` (§8.5).
   Evidence: the S10 `Float32` runs below and the
   `Float32`, `BigFloat` and `ForwardDiff.Dual` test items (§8.10). This supersedes the earlier
   options (a) refuse by name in `setup` and (b) allow with `eps = 1e-4`, `δ = sqrt(eps(T))`
   and `FullKKT` routing. S10 ran the spike dense generator at `n = 200`, `FullKKT`, `eps = 1e-4`,
   `δ = sqrt(eps(Float32))`, over the 18-instance grid (`κ ∈ {1, 1e3, 1e6}`, active fractions
   `{0.1, 0.5, 0.9}`, two-sided and mixed rows): all 18 solve, referee `1.1e-5`–`1.0e-4`,
   iteration count equal to the `Float64` run at the same tolerance on 15/18 and within 2 on
   the rest.
4. **What is G2?**
   (a) `t ≤ max(10·f, min(100, n/10))`, no solve at `cg_max_iter`, largest inner count below
   `n`, evaluated at `n ≥ 500`. (b) `t ≤ max(10·f, 100)` at every size. (c) `t/f ≤ 10` as in
   decision 3.
   **Recommended: (a).** At the spikes' sizes a floor of 100 is at least `n/2`, so (b) cannot
   tell a working preconditioner from none (spike 2: unpreconditioned CG passes it on
   Kronecker `n = 100` at `t = 88.5`). (c) fails preconditioners that start at one iteration.
   Spike 3 at `n ≤ 200`: (a)'s ratio term passes `cg_lagchol3` 18/18 and `cg_lldl10` 19/30,
   (b) 18/18 and 23/30.
5. **Regularization default and its safeguard.**
   (a) `reg_primal = reg_dual = 1e-8` on every backend, no `reg_indirect`, bump only when
   `factorize!` fails; §8.5 lists where it is measured and S10 measures the structured reduced
   backends (with an LP). (b) As (a) plus a second bump trigger when the refinement residual
   exceeds `sqrt(eps)·‖rhs‖`. (c) Per-backend defaults.
   **Recommended: (a).** Measured at `δ = 1e-8`: Bunch–Kaufman, no-pivoting `LDLᵀ` (CHOLMOD and
   LDLFactorizations, identical outer counts on 96/96 runs each with `refine_iter = 1`) and
   reduced Cholesky (96/96) all reproduce the same outer counts, equality rows included; larger
   `δ` costs outer convergence (§8.5). S10 measured the structured reduced backends (diagonal,
   tridiagonal, banded, block, sparse reduced), LPs included: none drifted without failing, so
   (b)'s trigger is still not needed. Its threshold would be one more unmeasured constant, and
   the stall rule already ends such a run `NUMERICAL_ERROR`.
6. **May `:ipm` use `linsys = :indirect` on matrices without a caller preconditioner?**
   (a) No, refused by name, as for operators. (b) Yes, with `JacobiPreconditioner`.
   **Recommended: (a).** Exact Jacobi failed G1 (17/27 dense, 21/30 sparse against exact
   ceilings of 27/27 and 30/30), the same kind of failure that decision 2 refuses for operators.
7. **May a caller preconditioner run with equilibration?**
   (a) No: `scaling = 0` required, documented reads `wt.w` and `wt.sigma` only. (b) Yes, with
   `probe = true` and a documented accessor for `D`, `E`, `c`.
   **Recommended: (a).** The matrix the preconditioner approximates is then the caller's own;
   (b) turns `Problem` internals into public API for a case nobody has measured.
8. **Where does CG start under `:ipm`?**
   (a) From zero. (b) From the previous solve (the predictor's `Δx_a` for the corrector).
   **Recommended: (a).** Measured on the dense and sparse reference runs: (b) gains no G1, costs
   1.02× (dense) and 1.21× (sparse) the median products, and adds five misses through Krylov's
   machine-precision stop.
9. **What does `:ipm` do with no inequality side (`N_s = 0`: equality and free rows only)?**
   (a) The rule of §8.4: `μ ≡ 0`, one solve per iteration, `α = 1`. (b) Refuse by name.
   **Recommended: (a).** Measured: 2 outer iterations on 48/48 instances with exact solves,
   18/18 and 30/30 with the reference preconditioners; it is the same `ipm_step!` with the
   corrector skipped.
10. **Which inner stopping and miss rule does `IndirectCG` use under `:ipm`?**
    (a) Stop on the recursive residual (or Krylov's own stop); miss only at `cg_max_iter` or on
    Krylov's throw; no recomputation. (b) As (a) but recompute once and miss above
    `2·atol_k`. (c) Spike 1's loop: restart CG until the recomputed residual passes `atol_k`.
    **Recommended: (a).** Identical to the explicit-residual oracle on 96 two-sided runs (inner
    and outer counts, G1); with equality rows only (a) keeps the exact solver's G1 (18/18 and
    18/18 against 15/18 and 14/18 for (b) and (c)), because even the exact step misses the
    reduced tolerance by up to 104×; (c) ran 932 and 1542 restarts into the cap. (a) trusts
    CG's recursion; a poor step for another reason reaches the outer stall rule instead of
    being flagged per solve.
11. **If the §9.6 gate fails, what happens to `:indirect` with a caller preconditioner on
    matrices?**
    (a) Refused under `:ipm` along with operators; the preconditioner interface serves ADMM
    only. (b) Kept for matrices, documented as unmeasured at scale.
    **Recommended: (a).** The gate measures exactly this path; matrices always have a direct
    backend under `:ipm`, so (b) keeps an unvalidated path with no case that needs it.

### 10.9 Public API

9. **Public API:** option 5, algorithm objects plus shared `Options`, chosen by the user. The
   `algorithm` keyword and the `Settings`/`IPMSettings` structs are replaced outright (the
   package is unregistered: no aliases, no deprecations). §11 specifies it as built (S12b).

---

## 11. Public API: algorithm objects and shared options

```julia
sol = solve(P, q, A, l, u)                                    # OperatorSplitting() is the default
sol = solve(P, q, A, l, u, OperatorSplitting(rho = 0.2, adaptive_rho = :kkt_error); eps_abs = 1e-6)
sol = solve(P, q, A, l, u, InteriorPoint(reg_primal = 1e-7); eps_abs = 1e-9, max_iter = 50)
ws  = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50); sol = solve!(ws)
ws.algorithm                                                  # InteriorPoint{Float64, Float64, Float64, Int}
ws.options                                                    # Options{Float64}
update_settings!(ws; eps_abs = 1e-10)                         # options, merged into the current ones
update_settings!(ws, InteriorPoint(reg_primal = 1e-6))        # parameters, replaced wholesale
sol = solve(Pop, q, Aop, l, u, InteriorPoint(); linsys = :indirect, preconditioner = M, scaling = 0)
```

**Types.** `abstract type QPAlgorithm end`; `abstract type QPWorkspace{T <: Real} end`.
`OperatorSplittingWorkspace{T,MP,MA,V,VI,LS,AC} <: QPWorkspace{T}` and
`InteriorPointWorkspace{T,MP,MA,V,VI,VB,LS} <: QPWorkspace{T}`: two workspace types named for
their state, not one parametric type, because their fields are disjoint beyond `prob`,
`linsys` and the run-state counters, and one type would need the algorithm as a parameter on
top of the seven each already carries. Both hold `algorithm` (the element-typed parameters)
and `options::Options{T}` as mutable, non-`const` fields that `update_settings!` replaces.
`active_kkt`, `adjoint_derivative` and `forward_derivative` take `QPWorkspace{T}`; nothing
else is shared by dispatch.

**`Options{T}`** (`src/core/options.jl`), one concrete struct of the settings both algorithms
read: `max_iter`, `time_limit`, `eps_abs`, `eps_rel`, `eps_prim_inf`, `eps_dual_inf`, `scaling`,
`check_termination`, `check_dualgap`, `scaled_termination`, `warm_starting`, `linsys`,
`polishing`, `polish_refine_iter`, `delta`, `cg_max_iter`, `cg_tol_fraction`. Its keyword
constructor has no default for the eight options whose defaults differ by algorithm; those come
from `algorithm_defaults(alg, T)`, a `NamedTuple` merged ahead of the caller's keywords, so an
explicit value always wins. `default_options(alg, T)` (exported) is `Options{T}` of those
defaults. `setup` builds `Options{T}(; algorithm_defaults(alg, T)..., linsys = LS, kwargs...)`
inside `setup_backend`, where the old settings struct was built, so the constant propagation
that keeps `--trim` resolving the workspace type is unchanged (the trim item is the detector).

| option | `OperatorSplitting` | `InteriorPoint` |
|---|---|---|
| `max_iter` | 4000 | 100 |
| `eps_abs`, `eps_rel` | `1e-3` | `ipm_floor(T)` |
| `eps_prim_inf`, `eps_dual_inf` | `1e-4` | `ipm_floor(T)` |
| `check_termination` | 25 | 1 |
| `cg_max_iter` | 20 | 500 |
| `cg_tol_fraction` | 0.15 | 0.1 |

Everything else defaults the same under both. Field-by-field decisions:

- `verbose` is an `Options` field: both algorithms print a header, one row per termination
  check, and a footer (§9.4 for the interior-point method's own report). `profile_primdual`
  stays an `OperatorSplitting` parameter, since only ADMM's loop reads a clock to accumulate
  the primal-dual integral it fills; the interior-point method has no counterpart to switch on.
- `cg_max_iter` and `cg_tol_fraction` are read by `IndirectCG` under both algorithms with the
  same meaning (a budget per solve; the fraction of the level `set_tolerance_level!` hands in),
  so they are options. Their validation is one rule for both, ADMM's `0 < cg_tol_fraction ≤ 1`;
  before S12b the IPM accepted any positive value.
- `cg_tol_reduction` (ADMM's idle-solve rule) is an `OperatorSplitting` parameter;
  `cg_fail_limit` an `InteriorPoint` parameter.
- `accelerator` and `preconditioner` stay keywords of `setup`, not fields: they are objects
  with state, typed by the caller. `accelerator` with `InteriorPoint()` throws by name.

**Algorithm objects.** `OperatorSplitting{T}`: `rho sigma alpha adaptive_rho
adaptive_rho_interval adaptive_rho_fraction adaptive_rho_tolerance rho_is_vec cg_tol_reduction
profile_primdual`. Its keyword constructor `OperatorSplitting(; …)` validates and stores
the values in `F = float(promote_type(…))` of the given reals (`Float64` for literals, so a
`BigFloat` value is kept exactly); no default depends on `T`, and `setup` converts with
`OperatorSplitting{T}(alg)`. `InteriorPoint{T, RP, RD, RI}`: `reg_primal reg_dual
max_reg_bumps refine_iter step_fraction cg_fail_limit`. Three parameters default to something
the object cannot know without the solve (`reg_primal`, `reg_dual` are `ipm_floor(T)`;
`refine_iter` is `0` under `linsys = :indirect`, else `1`), so `InteriorPoint(; …)` stores them
as `nothing` and carries one type parameter for each, which makes the seed concrete:
`InteriorPoint{F, Nothing, Nothing, Nothing}` with none of the three given,
`InteriorPoint{F, F, Nothing, Nothing}` with `reg_primal` given. `setup` resolves them with
`InteriorPoint{T}(alg, options.linsys)` into the `InteriorPoint{T, T, T, Int}` the workspace
holds. `RP`, `RD` and `RI` are internal, computed by the constructors; a sentinel number for
"not given" was the alternative and was not taken, since a reader of the object cannot tell a
sentinel from a value. `element_typed(alg, T, options)` is the one conversion entry point for
both. Validation
lives with the field: `Options` validates the options, each algorithm constructor its own
parameters, with the messages of the former settings structs.

**Dispatch.** `setup(T, P, q, A, l, u, alg; linsys, kwargs...)` lifts `linsys` into a `Val`
as before and calls `setup_backend(alg, Val(linsys), T, …)`, dispatching on the algorithm type
(the former `Val(:admm)`/`Val(:ipm)` tags are gone). The selection tags `ADMMSelection` and
`IPMSelection` are kept internally: the rungs dispatch on them in the core and in four
extensions, and they carry no parameters, so threading the element-typed algorithm object
through every rung would add nothing. Both `setup_backend` methods reach the named `linsys`
kinds through one `named_backend(Val{LS}, …)` (§5). `adopt_settings!(ls, alg, options)`
dispatches on the algorithm for `IndirectCG`'s `tol_reduction`. `update_settings!(ws, alg)`
with the other algorithm's object throws.

**Keyword errors.** `check_option_names(kwargs)` runs in `setup` and in
`update_settings!(ws; …)` before `Options` is built: a name that is a field of
`OperatorSplitting` or `InteriorPoint` throws `ArgumentError("rho is a parameter of
OperatorSplitting, not an option: pass it as OperatorSplitting(rho = ...).")`, under either
algorithm. The field-name tuples are constants computed at load time, so the check is
trim-safe. An unknown name is left to the keyword constructor and throws `MethodError`, as does
an option or a foreign parameter passed to an algorithm constructor.

**`update_settings!`.** `update_settings!(ws; kwargs...)` merges into `ws.options` (as before,
`linsys` and `scaling` refused). `update_settings!(ws, alg)` replaces the parameters wholesale:
a parameter left out of `alg` takes its default, not the previous value, since the object is a
value. On `OperatorSplittingWorkspace` a change of `rho`, `sigma` or `rho_is_vec` refactorizes
exactly as before; on `InteriorPointWorkspace` nothing refactorizes.

**MOI.** The raw attribute `"algorithm"` (`"admm"`/`"ipm"`, `String` or `Symbol`) selects the
algorithm; every other name is routed to `Options` or to the selected algorithm's parameters
and validated at `MOI.set` by building both (`_build`). Reading an unset name returns the
resolved default of the selected algorithm, `refine_iter` included. Switching `"algorithm"`
refuses a stored name the new algorithm does not accept and revalidates the rest. `optimize!`
builds the algorithm object from the stored parameters and the options from the stored
options, forcing `verbose = false` into the latter under `MOI.Silent`, for either algorithm
since `verbose` is a shared option.

**ChainRules.** `rrule`/`frule` for `solve` take the algorithm as an optional trailing
positional argument and return `NoTangent()` for it.

**Measured (S12b).** ADMM snapshot 41/41 identical; StrictMode audit unchanged; trim item
green; IPM reproduction counts unchanged. `admm_step!` minimums, A/B/B/A on one core, 100 steps
from a fresh setup: Random QP 4.33/4.29 µs (base/new), SVM 9.47/9.72 µs (+2.6%), Huber
19.31/19.12 µs. `setup` minimums: Random QP 367/376 µs (+2.3%), SVM 445/445 µs, Huber
713/710 µs.
