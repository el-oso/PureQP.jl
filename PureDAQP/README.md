# PureDAQP.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)

A pure-Julia dual active-set solver for convex quadratic programs:

```
minimize    ½ xᵀPx + qᵀx
subject to  l ≤ Ax ≤ u
```

It takes the problem type, the solution type and the generic `setup`/`solve`/`solve!` from
[PureQPBase.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureQPBase) and re-exports them,
so `using PureDAQP` is all you need.

```julia
using PureDAQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u, ActiveSet())
sol.status   # SOLVED
sol.x        # the exact solution of the equality QP over the active rows
```

## When to use it

Problems with few rows active at the solution. The method takes a few expensive steps and
stops at the exact point, rather than converging toward it, so the answer carries no tolerance
in it and `polishing` has nothing to add.

It needs a Cholesky factor of `P`, so an operator that supplies products only is refused as `P`.
`A` can be anything that supplies products with itself and its transpose. A dense pair forms
`A R⁻¹` once; a `Diagonal`, `BlockDiagonal` or `KroneckerOperator` on either side, and an
operator `A`, are held as they are and `A R⁻¹` is never formed. A sparse or banded `P` or `A` is read into a dense matrix,
so a large sparse problem is the other two algorithms' case.
[Choosing an algorithm](https://el-oso.github.io/PureQP.jl/dev/algorithms) compares all three.

On random dense problems at `1e-6`, it is faster than the C implementation it follows from
`n = 25` up, by a margin that widens with size:

| | n = 25 | n = 50 | n = 100 | n = 200 |
|---|---|---|---|---|
| PureDAQP / DAQP | 1.11× | 1.43× | 2.12× | 2.51× |

Below that the per-iteration constant dominates and DAQP is ahead, by 1.26× at `n = 10`. Full
tables in [Benchmarks](https://el-oso.github.io/PureQP.jl/dev/benchmarks).

## What it does

- an `LDLᵀ` of the working set's Gram matrix under rank-one updates, so an iteration costs
  `O(k²)` in the size of the working set rather than a fresh factorization
- a proximal-point outer loop, which handles a singular or merely ill-conditioned `P`
- Bland's rule after a bounded number of iterations, so the working set cannot cycle
- warm starts from the previous working set, which is what makes a re-solve after `update!`
  cheap
- a solve that allocates nothing: the reported `Solution` is the workspace's own, refilled

What it does not have: a choice of linear-system backend, equilibration, polishing, an
infeasibility certificate, and a dual-infeasibility test. Each is refused by name rather than
ignored.

## License

**MIT.** The method follows Arnström, Bemporad and Axehill, *A dual active-set solver for
embedded quadratic programming using recursive LDLᵀ updates*, IEEE Transactions on Automatic
Control 67(8):4362–4369, 2022 — Algorithm 1, with that paper's Algorithm 2 as the
proximal-point outer loop. The authors' MIT-licensed reference implementation
([DAQP](https://github.com/darnstrom/daqp)) was read alongside the paper, and checks the
answers here. See
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).
