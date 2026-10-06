# PureQP.jl

Pure-Julia solvers for convex quadratic programs:

```math
\begin{aligned}
\text{minimize}   \quad & \tfrac12 x^\top P x + q^\top x \\
\text{subject to} \quad & l \le A x \le u
\end{aligned}
```

`P` is symmetric positive semidefinite, `A` is `m×n`, and `l`, `u` may contain `∓Inf`. Rows where `l == u` are equality constraints.

!!! note "PureQP.jl is the project, not a package to install"
    There is no `PureQP` package. The name covers four that are installed separately:
    **PureOSQP.jl**, **PureIPM.jl** and **PureDAQP.jl**, the three solvers, and
    **PureQPBase.jl**, which each builds on and each re-exports. Install whichever solver you
    want — one `using` is enough — or several, to have their algorithms side by side.

Three algorithms solve it, sharing the same problem interface.
[`OperatorSplitting`](@ref), which PureOSQP.jl supplies, is OSQP's ADMM iteration, and is at
its best on repeated solves, warm starts and matrix-free operators. [`InteriorPoint`](@ref),
which PureIPM.jl supplies, is a Mehrotra predictor–corrector method that reaches `1e-8` in a
few iterations. [`ActiveSet`](@ref), which PureDAQP.jl supplies, is a dual active-set method
for problems with few rows active at the solution, and it stops at the exact answer
rather than converging toward one. The first two share the matrix support and the
linear-system backends described below; `ActiveSet` works from a Cholesky factor of `P` and
rows of `A`, and maintains its own factorization. [Choosing an algorithm](@ref) compares all three.

```julia
using PureOSQP                                                 # ] add PureOSQP
sol = solve(P, q, A, l, u, OperatorSplitting())

using PureIPM                                                  # ] add PureIPM
sol = solve(P, q, A, l, u, InteriorPoint(); eps_abs = 1e-9)

using PureDAQP                                                 # ] add PureDAQP
sol = solve(P, q, A, l, u, ActiveSet())
```

Loading several gives one `solve` that takes any of their algorithms as its sixth positional
argument.

## Your first solve

Install it, describe the problem with five arrays, and call [`solve`](@ref):

```@example first
using PureOSQP

# minimize  x₁² + x₂² - x₁ - 2x₂     subject to   x₁ + x₂ ≤ 1,  x ≥ 0
P = [2.0 0.0; 0.0 2.0]      # the quadratic term, as ½xᵀPx -- note the 2s
q = [-1.0, -2.0]            # the linear term
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [-Inf, 0.0, 0.0]        # lower bounds
u = [1.0, Inf, Inf]         # upper bounds

sol = solve(P, q, A, l, u, OperatorSplitting())
(sol.status, round.(sol.x; digits = 4))
```

`sol.status` is `SOLVED`, and `sol.x` is the answer. The interface is minimal: there is no model object or configuration.

Three things to know:
* **The `P` matrix:** it carries the `½ xᵀPx` term, so a plain `x₁² + x₂²` objective needs `2`s on the diagonal.
* **Constraints:** every constraint is a row of `l ≤ Ax ≤ u`. Use `l == u` for an equality, and `±Inf` for a one-sided constraint.
* **Setup:** `solve` handles everything at once. For many similar problems, use [`setup`](@ref) to build a workspace once and reuse it.

For more, see [Examples](@ref "Building a workspace once") or the implementation details below.

## What makes this different

Any `AbstractMatrix` — dense, sparse, structured or lazy — over any `Real` element type. `P`
and `A` are held by reference and never copied or modified. Every per-iteration product runs
`mul!` on the matrix you passed, so a `Diagonal`, a `Tridiagonal`, a `SubArray` or a
`SparseMatrixCSC` keeps its own product instead of being flattened into a dense copy.
Equilibration reaches the entries through four overridable column traversals, and a
`SparseArrays` weak dependency specialises them to walk only the stored entries.

One matrix in the solver is always dense: the `n×n` reduced system you get by eliminating `ν`.
That comes from the reduction, not from a limit on your input. If `A` has a dense row, the
solver factors the full KKT system sparsely instead, so it never squares the matrix.

The code uses `LinearAlgebra` and `TypeContracts.jl` for the linear-system backend interface.

## Re-using a workspace

For repeated solves, build the workspace once and reuse it — the factorization and all
buffers are retained, and the previous iterates warm-start the next solve:

```julia
ws = setup(P, q, A, l, u, OperatorSplitting(); eps_abs = 1e-8, eps_rel = 1e-8, polishing = true)
sol = solve!(ws)
sol = solve!(ws)     # warm started from the previous solution
```

To start from a known solution:

```julia
warm_start!(ws; x = x0, y = y0)
solve!(ws)
```

## Re-solving with new data

In a loop like Model Predictive Control, keep `P` and `A` fixed and update `q`, `l` and `u`.
[`update!`](@ref) reuses the workspace: the equilibration, the buffers and the iterates, and it
refactorizes only when it must. Under `InteriorPoint` every outer iteration factors a new system
whatever `update!` did — see [Choosing an algorithm](@ref "Re-solving a sequence") for what each
algorithm gets from it.

```julia
ws = setup(P, q, A, l, u, OperatorSplitting())
for step in 1:horizon
    update!(ws; q = q_k, l = l_k, u = u_k)
    sol = solve!(ws)          # warm started from the previous step
end
```

Updating `q` never refactorizes. Changing `l` or `u` refactorizes only when it moves a row
between the equality, inequality and free classes. Changing `P` or `A` always refactorizes.

`ws.refactor_count` counts every factorization, adaptive `ρ` included. So the count grows under
`update!` too whenever `ρ` changes.

`setup` equilibrates once. If your data changes magnitude by a lot, build a new workspace.

## Accuracy

The default tolerances are `eps_abs = eps_rel = 1e-3` under `OperatorSplitting` and `1e-8` under
`InteriorPoint`. Two ways to get more accuracy:
* Lower `eps_abs` and `eps_rel`. You pay in iterations.
* Set `polishing = true`, which solves the equality-constrained QP at the active set exactly.
  That takes the KKT residuals to machine precision, and it costs one extra factorization.

The solver keeps the polished point only when it improves both residuals, so polishing cannot
make the answer worse.

## Which backend you get

This is a choice `OperatorSplitting` and `InteriorPoint` make; `ActiveSet` has no backend to
choose and refuses any `linsys` but `:auto`.

`linsys = :auto` takes the first backend that fits. For two dense matrices that is an `n×n`
Cholesky of the reduced system. A structured matrix — diagonal, banded and the rest — is caught
earlier. An operator that is never formed goes to the matrix-free backend under
`OperatorSplitting`, and to `product_reduced` under `InteriorPoint`, which assembles the reduced
matrix from products instead of iterating.

If that Cholesky finds the reduced matrix is not positive definite, `setup` throws and names
`linsys = :kkt`. It does not switch backend underneath you, because the backend is fixed at
`setup` so every solve dispatches statically. Pass `linsys = :kkt` yourself to factor the full
`(n+m)×(n+m)` system with Bunch-Kaufman, which does not square the conditioning of `A` and is
the more accurate choice on an ill-conditioned problem. [How a backend is chosen](@ref) has the
full order and the condition each candidate asks.

## Watching a solve

`verbose = true` prints progress under `OperatorSplitting` and `InteriorPoint`: a header, one
line per termination check, and a footer with the status, the iterations and the residuals.
`ActiveSet` keeps no iteration log and prints nothing.

```
 iter      objective      prim res      dual res           rho
   25        1.39217        0.0543       0.00297           0.1
   50        1.45711       0.00245      0.000415           0.1
  ...
  125        1.46211      0.000374      0.000174         0.549
```

The `rho` column shows adaptive `ρ` updates, and each one triggers a refactorization. That is
`OperatorSplitting`'s row. `InteriorPoint` prints the barrier parameter `mu` and the step length
`alpha` in its place:

```
 iter      objective      prim res      dual res            mu         alpha
    1        1.28841        0.4213        0.1882        0.3841        1.0000
    2        1.35207       0.02184       0.00931       0.02033        0.9214
  ...
    7        1.36012      8.14e-09      3.02e-09      6.71e-09        1.0000
```

`InteriorPoint`'s footer also gives the run time. On the matrix-free `linsys = :indirect`
backend its rows gain a `cg iters` column for that iteration's conjugate-gradient count, and its
footer gains the total CG iterations and the number of missed inner solves (see
[`Solution.cg_iters`](@ref PureQPBase.Solution)).

Output goes to `Core.stdout` rather than `Base.stdout`, so that `--trim` compilation works. Use
`redirect_stdout` to capture it.

## What a solve reports

The [`Solution`](@ref) object carries the objective, the duality gap, both residuals,
`rel_kkt_error`, the iteration and `ρ` counts, the polishing result, and four timings.

`rho_updates` counts only adaptive `ρ` changes. `ws.refactor_count` counts every
refactorization, data changes included.

## Status values

[`Status`](@ref PureQPBase.Status) has twelve values. Two things to know:
1. A run that did not converge is never marked `SOLVED`.
2. When there is no meaningful primal-dual point, `x` and `y` are `NaN`.

Use [`has_solution`](@ref PureQPBase.has_solution) to check.

## What is rejected

`setup` throws when:
* `P` is not symmetric. The test is exact: `issymmetric(P)`.
* `P + σI` is not positive definite, where `σ` is `sigma` under [`OperatorSplitting`](@ref) and
  `reg_primal` under [`InteriorPoint`](@ref). See below.
* `q` holds a `NaN` or an `Inf`.
* `l` or `u` holds a `NaN`.
* `l[i] > u[i]` in any row.
* `l[i] == +Inf`, or `u[i] == -Inf`.
* The dimensions do not match.
* A setting is out of range: `sigma <= 0`, `alpha` outside `(0, 2)`, or `max_iter <= 0`.

**The convexity test is shifted, so it is not a test that `P` is positive semidefinite.** A `P`
whose most negative eigenvalue is smaller than `σ` passes it. At the default `sigma = 1e-6`, a
`P` with an eigenvalue of `-1e-9` solves and reports `SOLVED`; the same `P` throws at
`sigma = 1e-12`. Lower `σ` to tighten the test. A clearly indefinite `P` is rejected at any `σ`
you would use.

Both of these are stricter than the C library, which reads only the upper triangle of `P` and
therefore cannot see an asymmetry at all. On convexity the two agree: libosqp also refuses a
clearly negative eigenvalue at setup, and also accepts one of `-1e-9`.

## Citation

Stellato, Banjac, Goulart, Bemporad and Boyd, *OSQP: an operator splitting solver for
quadratic programs*, Mathematical Programming Computation 12(4):637–672, 2020.
