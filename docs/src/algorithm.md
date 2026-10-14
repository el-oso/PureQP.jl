# Algorithm

This page explains how two of the three algorithms work inside:
[`OperatorSplitting`](@ref), OSQP's ADMM iteration, and [`InteriorPoint`](@ref), a Mehrotra
predictor–corrector method. Both reduce every iteration to one linear system of the same shape,
and both use the same set of backends. [The linear system](@ref) and the way a backend is picked
belong to neither algorithm; they are shared.

[`ActiveSet`](@ref) is not on this page. It forms no such system and has no backend: it reduces
the problem to a least-distance problem once and then maintains an `LDLᵀ` of the working set's
Gram matrix under rank-one updates. Its docstring and [Choosing an algorithm](@ref) describe it.

The algorithms differ in what changes inside that system from one iteration to the next, and in
what the outer loop does with the answer. Each section below covers its own. Equilibration,
termination, adaptive `ρ`, infeasibility and polishing come after both, once, because both use
them the same way except where a section says otherwise.

## Operator splitting (ADMM)

The problem is to minimize an objective and stay inside the bounds. ADMM (*alternating direction
method of multipliers*) keeps two copies of the answer: one that minimizes the objective, one
that satisfies the bounds. A penalty term pulls the two copies together. Each iteration solves
an unconstrained problem for `x`, clips `z` into the box, and updates a multiplier `y`. When the
two copies agree, you have the solution.

The linear solve costs the most. Its matrix changes only when `ρ` changes, so the solver factors
it once and reuses the factor.

Each iteration solves one linear system, then projects:

```math
\begin{aligned}
\begin{bmatrix} P + \sigma I & A^\top \\ A & -\mathrm{diag}(\rho)^{-1} \end{bmatrix}
\begin{bmatrix} \tilde x^{k+1} \\ \nu^{k+1} \end{bmatrix}
&= \begin{bmatrix} \sigma x^k - q \\ z^k - \rho^{-1} \odot y^k \end{bmatrix} \\
\tilde z^{k+1} &= z^k + \rho^{-1} \odot (\nu^{k+1} - y^k) \\
x^{k+1} &= \alpha \tilde x^{k+1} + (1-\alpha) x^k \\
z^{k+1} &= \Pi_{[l,u]}\!\left(\alpha \tilde z^{k+1} + (1-\alpha) z^k + \rho^{-1} \odot y^k\right) \\
y^{k+1} &= y^k + \rho \odot \left(\alpha \tilde z^{k+1} + (1-\alpha) z^k - z^{k+1}\right)
\end{aligned}
```

`σ` regularizes the `(1,1)` block, `α` is the over-relaxation parameter, and `ρ` is a vector with larger values on equality rows.

## The linear system

This section and [Choosing a backend](@ref) below describe parts these two algorithms share. Every
backend named here also serves the interior-point method further down this page, at the row
weights that method hands it in place of ADMM's `ρ`. The reference implementation factors the
`(n+m)×(n+m)` quasi-definite matrix with a sparse pivot-free LDLᵀ. An equivalent `n×n` symmetric
positive definite system works too:

```math
(P + \sigma I + A^\top \mathrm{diag}(\rho) A)\, \tilde x = \mathrm{rhs}_x + A^\top(\rho \odot \mathrm{rhs}_z),
\qquad \tilde z = A \tilde x
```

PureOSQP factors this with `cholesky!`. On dense problems the reduced form is faster.

Here are the two systems for a small sparse pair, with each entry colored by where it comes
from. The reduced matrix is a fraction of the size, and `AᵀρA` fills it in wherever two rows of
`A` share a column.

::: details Code that draws the figure

```@example alg_kkt_figure
using CairoMakie, LinearAlgebra, SparseArrays, Random
Random.seed!(1)

n, m = 10, 16
A = Matrix(sprand(Bool, m, n, 0.15) .| sparse(1:m, mod1.(1:m, n), true, m, n))  # no empty row
Pn = [abs(i - j) <= 1 for i in 1:n, j in 1:n]                                  # tridiagonal P

# Entry codes: 0 empty, 1 from P, 2 from A, 3 both (or the −diag(ρ)⁻¹ block).
kkt = zeros(Int, n + m, n + m)
kkt[1:n, 1:n] .= Pn
kkt[1:n, (n + 1):end] .= 2 .* A'
kkt[(n + 1):end, 1:n] .= 2 .* A
for i in 1:m
    kkt[n + i, n + i] = 3
end
red = Pn .+ 2 .* ((A' * A) .> 0)

wong = cgrad([:white, "#0072B2", "#E69F00", "#CC79A7"]; categorical = true)
pattern!(ax, codes) = heatmap!(ax, 1:size(codes, 2), 1:size(codes, 1), permutedims(codes);
                               colormap = wong, colorrange = (-0.5, 3.5))

fig = Figure(size = (820, 420))
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
           title = "full KKT, (n+m)×(n+m) = $(n + m)×$(n + m), LDLᵀ")
pattern!(ax1, kkt)
lines!(ax1, [n + 0.5, n + 0.5], [0.5, n + m + 0.5]; color = :black, linewidth = 1)
lines!(ax1, [0.5, n + m + 0.5], [n + 0.5, n + 0.5]; color = :black, linewidth = 1)
text!(ax1, n / 2 + 0.5, -0.6; text = "P + σI", align = (:center, :bottom), color = "#0072B2", fontsize = 14)
text!(ax1, n + m / 2 + 0.5, -0.6; text = "Aᵀ", align = (:center, :bottom), color = "#E69F00", fontsize = 14)
text!(ax1, -0.5, n / 2 + 0.5; text = "P + σI", align = (:right, :center), color = "#0072B2", fontsize = 14)
text!(ax1, -0.5, n + m / 2 + 0.5; text = "A", align = (:right, :center), color = "#E69F00", fontsize = 14)
text!(ax1, n + m + 1, n + m / 2 + 0.5; text = "−diag(ρ)⁻¹", align = (:left, :center), color = "#CC79A7", fontsize = 14)
ax2 = Axis(fig[1, 2]; aspect = DataAspect(), yreversed = true,
           title = "reduced, n×n = $n×$n, Cholesky")
pattern!(ax2, red)
text!(ax2, n / 2 + 0.5, -0.6; text = "P + σI + Aᵀdiag(ρ)A", align = (:center, :bottom), fontsize = 14)
Legend(fig[2, 1:2],
    [PolyElement(color = c) for c in ("#0072B2", "#E69F00", "#CC79A7")],
    ["from P", "from A", "both, or −diag(ρ)⁻¹"]; orientation = :horizontal, framevisible = false)
foreach(ax -> (hidedecorations!(ax); hidespines!(ax)), (ax1, ax2))
colsize!(fig.layout, 1, Relative(0.62))
nothing # hide
```

:::

```@example alg_kkt_figure
fig # hide
```

```math
\mathrm{bandwidth}(R) = \max\bigl(\mathrm{bandwidth}(P),\; 2\,\mathrm{bandwidth}(A)\bigr)
```

`choose_backend` picks a backend from the problem's structure:

| `P` | `A` | bandwidth of `R` | backend | solve |
|---|---|---|---|---|
| `Diagonal` | `Diagonal` | 0 | [`DiagonalReduced`](@ref PureQPBase.DiagonalReduced) | `n` divisions |
| `SymTridiagonal` or `Tridiagonal` | `Diagonal` | 1 | [`TridiagonalReduced`](@ref PureQPBase.TridiagonalReduced) | `ldlt`, `O(n)` |
| `Diagonal` | `Bidiagonal` | 1 | [`TridiagonalReduced`](@ref PureQPBase.TridiagonalReduced) | `ldlt`, `O(n)` |
| `SymTridiagonal` or `Tridiagonal` | `Bidiagonal` | 1 | [`TridiagonalReduced`](@ref PureQPBase.TridiagonalReduced) | `ldlt`, `O(n)` |
| banded | banded | ``2 \leq b \leq n/4`` | `BandedReduced` | banded `cholesky`, `O(nb²)` |

The banded backend is a package extension. It needs `BandedMatrices.jl`.

A `Diagonal` `P` with a general `A` gets no special treatment, and its reduced matrix is dense.

Fill-in matters. On random sparse `A` the reduced matrix `R` is much sparser than `A` suggests,
but its Cholesky factor is not.

| n | m | density(A) | density(R) | density of chol(R) | fill |
|---|---|---|---|---|---|
| 200 | 400 | 1% | 6.3% | 47.8% | 7.6× |
| 200 | 400 | 5% | 78.9% | 100% | 1.3× |
| 400 | 800 | 1% | 11.2% | 83.3% | 7.4× |
| 400 | 800 | 5% | 95.4% | 100% | 1.1× |

Here is the first row, drawn. It uses a fresh random pattern at the same `n`, `m` and density,
so its percentages differ a little from the table's. The shape of the result does not. `R` keeps
most of `A`'s sparsity. Its Cholesky factor keeps almost none of it.

::: details Code that draws the figure

```@example alg_fill_figure
using CairoMakie, LinearAlgebra, SparseArrays, Random
Random.seed!(1)

n, m = 200, 400
A = sprand(m, n, 0.01)
R = sparse(1.0I, n, n) + A' * A
L = cholesky(Matrix(Symmetric(R))).L
density(M) = round(100 * count(!iszero, M) / length(M); digits = 1)

fig = Figure(size = (900, 380))
for (i, (name, M)) in enumerate(("A" => Matrix(A), "R = σI + AᵀρA" => Matrix(R), "chol(R)" => Matrix(L)))
    axf = Axis(fig[1, i]; aspect = DataAspect(), yreversed = true,
               title = "$name   $(size(M, 1))×$(size(M, 2)),  $(density(M))% nonzero")
    heatmap!(axf, 1:size(M, 2), 1:size(M, 1), permutedims(M .!= 0);
             colormap = [:white, "#0072B2"], colorrange = (0, 1))
    hidedecorations!(axf); hidespines!(axf)
end
nothing # hide
```

:::

```@example alg_fill_figure
fig # hide
```

## Solving with the inverse

Once `R` is factored, the solver inverts it in place. That beats a Cholesky `ldiv!`, because
each solve is then one symmetric matrix-vector product (`symv`) instead of two triangular
solves.

| kernel | solve, n = 200 | relative error |
|---|---|---|
| `potrs` (two triangular solves) | 11.43 µs | 5.2e-16 |
| `symv` against the inverse | 1.61 µs | 6.8e-16 |
| two `trmv` against `L⁻¹` | 3.82 µs | 5.7e-16 |

At these sizes the factor is a few hundred kilobytes and stays in cache. So what limits the
triangular solve is its sequential dependency, not memory bandwidth. Inverting costs about two
Cholesky factorizations (`potri` on top of `potrf`). You pay that once per `ρ` update and get it
back over the hundreds of iterations between updates.

Forming an explicit inverse is usually poor practice. It is safe here for one reason: the
reduced matrix carries the `σI` regularization, so its conditioning has a bound by construction
instead of inheriting one from the data. The error column above shows that. The inverse is as
accurate as the triangular solve.

Conditioning pushes the other way, because forming `AᵀρA` squares `cond(A)`. Here is the
measured relative error of the inner solve against an extended-precision reference, at `n = 60`,
`m = 200`, `P = 0`, on `A` built with geometrically spread singular values:

| cond(A) | reduced, unscaled | reduced, equilibrated | full KKT |
|---|---|---|---|
| 1e4 | 8.2e-10 | 4.3e-10 | 1.5e-13 |
| 1e6 | 1.1e-05 | 4.5e-09 | 8.8e-12 |
| 1e8 | 9.5e-02 | 8.0e-09 | 3.1e-10 |
| 1e10 | fails | 4.8e-09 | 3.2e-08 |
| 1e14 | fails | 1.3e-08 | 2.0e-04 |
| 1e16 | fails | 1.2e-08 | 3.4e-02 |

Equilibration is what makes the reduced form usable. Without it the Cholesky loses all accuracy
by `cond(A) = 1e8` and fails past that. With it, which is the default, the reduced solve holds
around `1e-8` across the whole range. Past `cond(A) = 1e10` it is *more* accurate than the full
KKT factorization, because equilibration bounds what the reduced matrix inherits while the
quasi-definite matrix keeps the raw conditioning.

The first table's `A` family is ill-conditioned by a spread of singular values, which is close
to what equilibration is built to fix. The harder case is ill-conditioning that no diagonal
scaling can remove: an `A` with unit-norm columns and nearly parallel rows.

| row spread | cond(A) | reduced, equilibrated | Cholesky succeeded | full KKT |
|---|---|---|---|---|
| 1e-2 | 2.7e2 | 1.0e-12 | yes | 4.6e-14 |
| 1e-4 | 3.4e4 | 5.4e-09 | yes | 7.0e-11 |
| 1e-6 | 3.4e6 | 2.5e-08 | yes | 4.3e-10 |
| 1e-8 | 3.4e8 | 2.7e-08 | yes | 3.4e-10 |
| 1e-10 | 3.4e10 | 2.7e-08 | yes | 2.4e-10 |

Here the Cholesky is about a hundred times less accurate than the full KKT factorization. It
still succeeds, and it still levels off near `1e-8`. It gets worse gradually rather than
collapsing without warning. With equilibration on, the Cholesky never failed in any case we
measured.

`PureOSQP/bench/kkt_backend.jl` reproduces both tables.

### Choosing a backend

`linsys = :auto`, the default, walks the candidates in order and takes the first one that serves
the given `P` and `A`. The reduced Cholesky is the last candidate: any pair the solver can
materialize reaches it when nothing cheaper fits. If that Cholesky reports the matrix is not
positive definite, `setup` throws and names `linsys = :kkt`, which factors the full
quasi-definite system with `bunchkaufman!` and does not square the conditioning of `A`. On the
measurements above, that does not happen with equilibration on.

Here is the whole order with the condition each candidate asks, top to bottom. A pair whose
types name a backend outright takes it and skips the rest. Every other pair starts at candidate
1 and stops at the first one that accepts it. The two sparse factorization candidates decide
from the pattern alone — the densest row of `A`, the stored entries of the KKT matrix, and the
symbolic `AᵀA ∪ P` pattern count — and factor nothing to find out. Once a candidate accepts a
pair, the solver builds and factors the backend it named, and `setup` keeps that factorization.

[How a backend is chosen](@ref) has the same order per algorithm, with why the interior-point
method skips three of these candidates.

::: details Code that draws the figure

```@example alg_ladder_figure
using CairoMakie

steps = [
    ("Diagonal P and Diagonal A", ":diagonal — n divisions"),
    ("SymTridiagonal/Tridiagonal P with Diagonal A,\nor Diagonal/tridiagonal P with Bidiagonal A", ":tridiagonal — ldlt, O(n)"),
    ("banded P and A, 2 ≤ bandwidth(R) ≤ n/4\n(BandedMatrices loaded)", ":banded — banded Cholesky, O(nb²)"),
    ("sparse A: a row spans half of n,\nor the KKT pattern stays under n²/4?", ":sparse_kkt — factor the full KKT matrix"),
    ("sparse A: symbolic reduced pattern\nunder 5% of n²?", ":cholmod — factor R sparsely"),
    ("KroneckerOperator A, P = μI,\none ρ for all rows, scaling = 0", ":kronecker — eigenbases of the factors"),
    ("BlockDiagonal P and A, same partition", ":block — one solve per block"),
    ("Diagonal P and RowCoupled A, 10k ≤ n", ":lowrank — Woodbury, O(nk)"),
    ("sparse A: form R from stored entries", ":sparse_formed — dense inverse"),
    ("P and A can be materialized", ":cholesky — dense reduced matrix"),
    ("neither can be (a LinearMap)", ":indirect — conjugate gradients"),
]

fig = Figure(size = (860, 760))
ax = Axis(fig[1, 1])
hidedecorations!(ax); hidespines!(ax)
box!(x, y, w, h, txt; color, textcolor = :black) = begin
    poly!(ax, Rect2f(x, y, w, h); color, strokecolor = :black, strokewidth = 1)
    text!(ax, x + w / 2, y + h / 2; text = txt, align = (:center, :center), fontsize = 12, color = textcolor)
end
down!(x, y0, y1) = begin
    lines!(ax, [x, x], [y0, y1]; color = :gray40)
    scatter!(ax, [x], [y1]; marker = :dtriangle, color = :gray40, markersize = 10)
end
H = length(steps)
for (i, (question, answer)) in enumerate(steps)
    y = H - i
    box!(0, y + 0.1, 5.2, 0.8, question; color = (:gray, 0.12))
    box!(6.4, y + 0.1, 4.6, 0.8, answer; color = ("#0072B2", 0.15), textcolor = "#0072B2")
    lines!(ax, [5.2, 6.4], [y + 0.5, y + 0.5]; color = :gray40)
    scatter!(ax, [6.4], [y + 0.5]; marker = :rtriangle, color = :gray40, markersize = 10)
    text!(ax, 5.8, y + 0.55; text = "yes", align = (:center, :bottom), fontsize = 10, color = :gray40)
    i < H && down!(2.6, y + 0.1, y - 0.1)
end
text!(ax, 2.6, H + 0.25; text = "P, A   (linsys = :auto)", align = (:center, :bottom), fontsize = 13)
down!(2.6, H + 0.2, H - 0.1)
lines!(ax, [-0.25, -0.25], [H - 3 + 0.1, H - 0.1]; color = :gray50)
text!(ax, -0.4, H - 1.5; text = "by type", align = (:center, :bottom), fontsize = 11, color = :gray50, rotation = pi / 2)
lines!(ax, [-0.25, -0.25], [0.1, H - 3 - 0.1]; color = :gray50)
text!(ax, -0.4, (H - 3) / 2; text = "candidates 1–8, in order", align = (:center, :bottom), fontsize = 11, color = :gray50, rotation = pi / 2)
text!(ax, 5.0, -0.5; text = "a :cholesky that finds R not positive definite throws and names linsys = :kkt",
      align = (:center, :top), fontsize = 11, color = :gray30)
limits!(ax, -1.2, 11.2, -1.2, H + 1)
nothing # hide
```

:::

```@example alg_ladder_figure
fig # hide
```

`linsys = :kkt` forces the full quasi-definite factorization on every solve. It is slower, which
is the whole point of the reduced form. But it is the more accurate factorization at moderate
conditioning, and it is the closest match to what the reference implementation does. That makes
it useful when you doubt a result. The whole test corpus runs through both backends.

## The interior-point method

[`InteriorPoint`](@ref) is a Mehrotra predictor–corrector method. It does not run the ADMM
recurrence above. Each outer iteration factors a new Newton system at the current point, takes a
predictor step to estimate how far the barrier parameter `μ` can drop, then takes a corrector
step aimed at that `μ`:

```math
\begin{aligned}
\begin{bmatrix} \tilde P + \delta_p I & \tilde A^\top \\ \tilde A & -\mathrm{diag}(w)^{-1} \end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}
&= \begin{bmatrix} -r_d \\ -\mathrm{diag}(w)^{-1} \odot g \end{bmatrix}
\end{aligned}
```

`w` is the current row weight — `z_l/(s_l + \delta_d z_l) + z_u/(s_u + \delta_d z_u)` on an
inequality row, `1/\delta_d` on an equality row, `\delta_d` on a free row — and it changes
every iteration as the slacks `s` and multipliers `z` move toward the boundary. `δ_p` and `δ_d`
are fixed proximal regularizations. The iteration does not tune them away. Solving this system
is the `LinearSystem` contract above, so every backend that serves ADMM's reduced or KKT system
serves this one too, at the row weights the interior-point method hands it.

A run takes a few dozen iterations. Each one pays for a fresh factorization, and the run reaches
`1e-8` by default rather than ADMM's `1e-3`. Pick it over ADMM when you want more accuracy than
ADMM reaches in a modest iteration count, or when ADMM's residuals fall slowly on your problem.
Pick ADMM when you re-solve the same workspace many times through [`update!`](@ref), or when
your matrices are operators and you have no preconditioner to supply.

### What it accepts, and what throws

- [`update!`](@ref), [`warm_start!`](@ref), [`cold_start!`](@ref) and
  [`update_settings!`](@ref) work as they do for ADMM. A settings change never forces a
  refactorization, because every solve resets its regularization and factors before the first
  iteration anyway.
- You must pass `polishing = true` before [`adjoint_derivative`](@ref) or
  [`forward_derivative`](@ref). An interior-point solution holds its inactive-row multipliers at
  `O(μ_final)`, not near zero, which is what a derivative through the active set needs.
  Polishing cleans that up.
- `PureIPM.Optimizer` is the MathOptInterface wrapper around it, as `PureOSQP.Optimizer` is
  around the operator-splitting method. An optimizer runs the algorithm of the package that
  supplies it. It takes the shared options plus that algorithm's own parameters, and nothing
  else.
- `verbose` prints under either of these two: a header, one line per termination check, and a
  footer. It prints `mu` and `alpha` in place of ADMM's `rho`. The interior-point footer also
  gives the run time. On the matrix-free backend its rows gain a `cg iters` column, and its
  footer gains a line for the total CG iterations and the missed inner solves. Only
  [`OperatorSplitting`](@ref) reads `profile_primdual`. Pass it to `InteriorPoint()` and it
  throws, naming the algorithm that owns it, because only ADMM's loop builds up the primal-dual
  integral it fills.
- `accelerator` is ADMM's fixed-point accelerator. Pass it to `InteriorPoint()` and it throws,
  because the interior-point method has no fixed-point iteration to accelerate. Pass a GPU array
  to `InteriorPoint()` and it throws too: none of its backends has a GPU counterpart.
- `linsys = :indirect` runs only with a `preconditioner` you supply and `scaling = 0`, on
  matrices and operators alike. See [Operators under the interior-point method](@ref).
  `linsys = :kronecker` and `linsys = :lowrank` throw. The Kronecker backend needs one weight
  for every row, and the interior-point method's per-row weights break that. The low-rank
  backend's Woodbury solve misses the tolerance on linear programs (see the table below).
- It takes any `T <: Real`. See the note on element types below.

### Backends under the interior-point method

With [`InteriorPoint`](@ref) the Newton system keeps the same shape, but its row weights change
every iteration and reach `1/reg_dual`, `1e8` by default, on equality rows and on rows whose
bound is active. The reduced form squares those weights into its conditioning. So we ran each
backend on problems of its own structure and compared it with the dense full KKT factorization
(`linsys = :kkt`) on the same problem. The table sums up
`PureIPM/bench/results/ipm_backends.json`, which `PureIPM/bench/ipm_backends.jl` writes.
"Referee" is the largest optimality residual computed from the original data. Iterations are
outer iterations, the same for both columns unless the table says otherwise.

| backend | problems | status | referee | iterations (this backend / `:kkt`) |
|---|---|---|---|---|
| `:diagonal` | diagonal pair, `n = 100`; QP, LP, with equality rows | solved | ≤ 3e-10 | 8–9, equal |
| `:tridiagonal` | two tridiagonal pairs; QP, LP, with equality rows | solved | ≤ 6e-9 | 7–9, equal |
| `:banded` | two banded pairs; QP, LP, with equality rows | solved | ≤ 1.1e-8 | 7–9, equal |
| `:block` | block QP, block LP | solved | ≤ 4.4e-9 | 7–8, equal |
| `:cholesky` (dense reduced) | three structured pairs, QP, LP, with equality rows; a dense QP and LP | solved | ≤ 9.2e-9 | 7–11, equal |
| `:ldlfactorizations`, `:cholmod` (sparse reduced) | `banded_qp(200, 300)`; QP, LP | solved | ≤ 2.4e-9 | 9 and 12, equal |
| `:sparse_kkt`, `:ldl_kkt` (sparse KKT) | `banded_qp(200, 300)`, Portfolio, Lasso, SVM, Huber | solved | ≤ 7.7e-9 | 7–12, equal |
| `:lowrank` | two diagonal-plus-low-rank pairs; QP and with equality rows | solved | ≤ 4.3e-9 | 8–9, equal |
| `:lowrank` | the same pairs as LPs | not solved (2 at 100 iterations, 2 numerical errors) | up to 4.7e7, and no residual at all where the solve ends | 44–100 / 8–11 |

A backend stays among the interior-point candidates when it solves every one of its problems
with a referee below `1e-5`, in at most twice the iterations `:kkt` takes, on the problems in
the table above. Every backend passes except `:lowrank`, which fails on linear programs. When
`P` is zero, a variable that only the dense rows reach keeps nothing but `reg_primal` in the
diagonal core. That puts `1e8` in the core's inverse, and on those problems the Woodbury solve
ends without a solution. So `linsys = :lowrank` throws under `InteriorPoint()`, and `:auto`
gives a diagonal `P` with a `RowCoupled` `A` to `product_reduced`: a `RowCoupled` answers
`holds_structure`, so the rungs that read entries decline it and the one that assembles the
reduced matrix from products takes it. `:sparse_formed` has no interior-point counterpart,
because it holds the inverse of the reduced matrix, and this method refactorizes at every
iteration's weights and then solves twice — so its reduced backends keep the Cholesky factor
instead.

`:kkt` copies `A` into the full matrix one entry at a time at every factorization. We measured
that copy on one core: 3% of an iteration on a dense QP (`n = 200`, `m = 400`), 7% on Lasso
(`n = m = 816`, sparse `A`) and 16% on `banded_qp(200, 300)`. The `bunchkaufman!` call is the
rest of the factorization, and most of the iteration.

The interior-point method is generic over the element type `T <: Real`, as ADMM is. Its four
tolerances, its two regularizations and its short-step threshold default to `1e-8` in `Float64`
and in finer arithmetic such as `BigFloat`, and to `sqrt(eps(T))` in coarser arithmetic — for
`Float32` that is `3.5e-4`. The divergence bound is `1/sqrt(eps(T))` times the size of the data.
A number type that wraps another, such as `ForwardDiff.Dual`, takes the precision of `float(T)`.

On the dense test generator (`n = 200`, 18 instances, two-sided and mixed rows), `:kkt` in
`Float32` at `eps_abs = eps_rel = 1e-4` and `reg_primal = reg_dual = sqrt(eps(Float32))`, with
no equilibration, solves all 18 in 4–8 iterations and never raises its regularization. Referees
run from `1.1e-5` to `1.0e-4`. `Float64` at the same tolerances takes 4–6 iterations, with
referees from `2.4e-6` to `8.1e-5`.

`BigFloat` runs on `:kkt` and on the dense reduced Cholesky. Dual numbers (`ForwardDiff.Dual`)
run on the dense reduced Cholesky, which `:auto` picks for them, but not on `:kkt`, because
`bunchkaufman!` has no method for them. The derivative of the objective through a solve matches
a central difference of the `Float64` objective.

## Equilibration

The solver equilibrates the system with modified Ruiz scaling. It stores the scaling as factors
rather than applying it to the matrices:

```math
\tilde P = c\,D P D, \qquad \tilde A = E A D, \qquad \tilde q = c\,D q, \qquad
\tilde l = E l, \qquad \tilde u = E u
```

::: details Code that draws the figure

```@example alg_equil_figure
using CairoMakie

fig = Figure(size = (760, 330))
ax = Axis(fig[1, 1]; aspect = DataAspect())
hidedecorations!(ax); hidespines!(ax)
# A diagonal factor: an outlined square with its diagonal drawn.
diagbox!(x, y, s, label, color) = begin
    lines!(ax, Rect2f(x, y, s, s); color = :black, linewidth = 1)
    lines!(ax, [x, x + s], [y + s, y]; color, linewidth = 3)
    text!(ax, x + s / 2, y - 0.25; text = label, align = (:center, :top), fontsize = 14, color)
end
# The caller's matrix, held by reference.
refbox!(x, y, w, h, label) = begin
    poly!(ax, Rect2f(x, y, w, h); color = (:gray, 0.25), strokecolor = :black, strokewidth = 1)
    text!(ax, x + w / 2, y + h / 2; text = label, align = (:center, :center), fontsize = 15)
    text!(ax, x + w / 2, y - 0.3; text = "the caller's array, by reference",
          align = (:center, :top), fontsize = 12, color = :gray30)
end
n, m = 3.0, 4.5
y0, y1 = 6.5, 0.5
text!(ax, 0, y0 + n / 2; text = L"\tilde{P} = c\,\cdot", align = (:right, :center), fontsize = 16)
diagbox!(0.5, y0, n, "D", "#E69F00")
refbox!(4.0, y0, n, n, "P")
diagbox!(7.5, y0, n, "D", "#E69F00")
text!(ax, 0, y1 + m / 2; text = L"\tilde{A} =", align = (:right, :center), fontsize = 16)
diagbox!(0.5, y1, m, "E", "#CC79A7")
refbox!(5.5, y1, n, m, "A")
diagbox!(9.0, y1 + (m - n), n, "D", "#E69F00")
text!(ax, 13.5, y0 + n / 2; text = "stored: c and the diagonal of D\n(a scalar and an n-vector)",
      align = (:left, :center), fontsize = 13, color = :gray30)
text!(ax, 13.5, y1 + m / 2; text = "stored: the diagonals of E and D\n(an m-vector, the same n-vector)",
      align = (:left, :center), fontsize = 13, color = :gray30)
limits!(ax, -2.5, 22, -1.2, 10.2)
nothing # hide
```

:::

```@example alg_equil_figure
fig # hide
```

Every per-iteration product calls `mul!` on the matrix you passed, with the factors applied
around it. So a structured or lazy `A` keeps its fast product, and nothing is copied.

## Convergence

The iteration stops when the primal and dual residuals meet the tolerances you gave.

```math
r_{\rm prim} = \|Ax - z\|_\infty,
\qquad
r_{\rm dual} = \|Px + q + A^\top y\|_\infty
```

The solver reports both residuals in **problem space**.

Each tolerance is an absolute part plus a relative part:

```math
\epsilon_{\rm prim} = \epsilon_{\rm abs} + \epsilon_{\rm rel}\max\bigl(\|Ax\|_\infty,\|z\|_\infty\bigr),
\qquad
\epsilon_{\rm dual} = \epsilon_{\rm abs} + \epsilon_{\rm rel}\max\bigl(\|Px\|_\infty,\|q\|_\infty,\|A^\top y\|_\infty\bigr)
```

**The duality gap is a third test, and it can only delay convergence.** With `check_dualgap`,
which is on by default and follows libosqp 1.x, a point must also satisfy `|gap| < ε_gap` before
the solver calls it `SOLVED`. The solver checks the gap *after* the residuals pass, never
instead of them, because a point with a small gap and a large residual is not a solution.

Three more things decide what you see:

- **The check is periodic.** It runs every `check_termination` iterations, 25 by default, so a
  reported iteration count lands on a multiple of that. The solver does not go back and look for
  the first iteration that would have passed.
- **Failing the tolerances starts the infeasibility tests**, not just another iteration. A large
  primal residual is what prompts the primal-infeasibility test below.
- **A relaxed pass says so.** When the loop ends without converging, the solver repeats the
  whole test at ten times the tolerances. Passing that gives `SOLVED_INACCURATE`, which is a
  different answer from `SOLVED` and never shown as one. A residual above `INFTY` gives
  `NON_CONVEX`, because a convex problem's residuals cannot diverge. ADMM never reports
  `NUMERICAL_ERROR`. That status belongs to `InteriorPoint()`.

`scaled_termination` tests the equilibrated residuals instead of the unscaled ones. It is off by
default, because the question you want answered is about your problem, not the solver's internal
one.
## Adaptive ρ

The solver re-estimates `ρ` from the ratio of the primal and dual residuals:

```math
\rho_{\text{new}} = \rho \sqrt{
  \frac{r_{\text{prim}} / \max(\|z\|_\infty, \|Ax\|_\infty)}
       {r_{\text{dual}} / \max(\|q\|_\infty, \|A^\top y\|_\infty, \|Px\|_\infty)}}
```

It takes the new value only when the value moves by more than a factor of
`adaptive_rho_tolerance`, because taking it forces a refactorization.

The reference implementation triggers this on wall-clock time by default, once 0.4 × the setup
time has passed. PureOSQP uses a fixed iteration interval instead, 50 by default, so iteration
counts do not depend on how fast the machine is.

## Infeasibility

The solver detects infeasibility from the differences between iterates, `δx` and `δy`. The
interior-point method has no test of its own. It applies this one to its own last step and its
normalized iterate.

## Polishing

Both of these algorithms call the same `polish_kernel!` on their own `(x, y, z)`; `ActiveSet`,
documented in [Choosing an algorithm](@ref), refuses `polishing = true` because it has nothing
left to polish. It guesses the active
set from the iterate: row `i` is lower-active when `z_i - l_i < -y_i` or `l_i == u_i`, and
upper-active when `u_i - z_i < y_i`. That is why polishing must run before you take a derivative
under `InteriorPoint`. An unpolished row holds its multiplier at the barrier parameter rather
than near zero, and this test would read that wrongly. The equality-constrained QP that comes
out

```math
\begin{bmatrix} P + \delta I & A_{\text{red}}^\top \\ A_{\text{red}} & -\delta I \end{bmatrix}
\begin{bmatrix} x \\ y_{\text{red}} \end{bmatrix}
= \begin{bmatrix} -q \\ b_{\text{red}} \end{bmatrix}
```

is factored with `bunchkaufman!`, then corrected by three steps of iterative refinement against
the unregularized operator, which removes the error `δ` introduces. The polished point replaces
the ADMM answer only if both residuals improve.

## The name Cholesky


**André-Louis Cholesky** (born 15 October 1875 in Montguyon; died 31 August 1918 of wounds
received in northern France) was a French army officer and geodesist who ended as head of the
Topographical Service of Tunisia. He did not publish the method himself. It appeared
posthumously in 1924, when a fellow officer, Commandant Benoît, wrote it up in the
*Bulletin géodésique* as *"Note sur une méthode de résolution des équations normales…
(Procédé du Commandant Cholesky)"*.

**Say it `/ʃəˈlɛski/` — *shə-LES-kee*.** The first sound is the *sh* of *shoe*.

He was French, and French ⟨ch⟩ is /ʃ/. A second reading is defensible, from the family's
origins. His paternal line came from the **Cholewski** family, which left Poland during the
Great Emigration, and Polish ⟨ch⟩ is /x/: the fricative in *Bach*, in Greek χ, in Russian х, in
Spanish *j*. That gives *kho-LES-kee*. The field's own literature carries this reading. A 1990
NA Digest exchange set out three candidates and concluded that "all three current pronunciations
seem acceptable" until someone found evidence of the name's origin, noting that a Polish origin
would make *Kholesky* correct.

**A hard English /k/ — "koh-LES-kee", the *k* of *kiosk* — has no basis.** It is neither the
French /ʃ/ nor the Polish /x/. Those two are different sounds: /x/ is a fricative, with the air
still flowing, and /k/ is a plosive, stopped and released. The /k/ reading most likely comes
from the English habit of saying ⟨ch⟩ as /k/ in words taken from Greek — *chorus*, *chaos*,
*character*. This name is not Greek.

References: the pronunciation `/ʃəˈlɛski/` is given by
[Wikipedia's article on the decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition);
the Cholewski descent by
[its biography of Cholesky](https://en.wikipedia.org/wiki/Andr%C3%A9-Louis_Cholesky); the
dates, rank and the Benoît publication by the
[MacTutor biography](https://mathshistory.st-andrews.ac.uk/Biographies/Cholesky/); and the
three-way discussion by [NA Digest, Volume 90 Issue 11 (18 March 1990)](https://www.netlib.org/na-digest-html/90/v90n11.html).
