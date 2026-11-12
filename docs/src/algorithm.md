# Algorithm

This page explains how the three algorithms work inside: [`OperatorSplitting`](@ref), OSQP's
ADMM iteration; [`InteriorPoint`](@ref), a Mehrotra predictor–corrector method; and
[`ActiveSet`](@ref), a dual active-set method.

`OperatorSplitting` and `InteriorPoint` reduce every iteration to one linear system of the same
shape and use the same set of backends. [The linear system](@ref) and the way a backend is picked
belong to neither algorithm; they are shared. They differ in what changes inside that system
from one iteration to the next, and in what the outer loop does with the answer.

`ActiveSet` forms no such system and has no backend. It reduces the problem to a least-distance
problem once and then maintains a factorization of the rows held at a bound, one row at a time.

Each algorithm has its own section below. Equilibration, termination, adaptive `ρ`,
infeasibility and polishing come after the first two, once, because both use them the same way
except where a section says otherwise. `ActiveSet` uses none of them.

## The linear solver each algorithm uses

| | what is solved | what solves it by default |
|---|---|---|
| [`OperatorSplitting`](@ref) | one system per iteration, with a matrix that changes only when `ρ` does | the first candidate that serves the pair: a structured backend, a sparse factorization, or the dense reduced Cholesky (`ReducedCholesky`, which a dense `P` and `A` reach), with conjugate gradients last |
| [`InteriorPoint`](@ref) | one system per solve, with a matrix refactored at every iteration | the same list without the candidates that need fixed weights; a dense `P` and `A` reach `ReducedCholesky` from `n/4` rows, and `FullKKT`, the dense augmented system factored with `bunchkaufman!`, below that |
| [`ActiveSet`](@ref) | no system with a fixed shape: the Cholesky factor `R` of `P`, once, and a factorization of the working set | a triangular solve against `R`, and either a `QR` of the working set's rows or an `LDLᵀ` of their Gram matrix, chosen by `working_set` |

`linsys` picks the first two by name and `:auto` takes the default above; [How a backend is
chosen](@ref) gives the order of the candidates. `ActiveSet` refuses any `linsys` except `:auto`
and `:dense`. Two settings of `ActiveSet` decide the rest, and [the dual active-set
method](@ref "The dual active-set method") describes both:

- `working_set` is what it factors. `:rows` (default) factors the held rows themselves, as a
  `QR`. `:gram` factors their Gram matrix, as an `LDLᵀ`: cheaper per pass, and it squares the
  conditioning of the rows.
- `scan` is how many rows a pass prices to choose the one that enters. `:all` (default) prices
  every row and takes the worst violator. `:window` prices a window of them and takes the worst
  violator in it.

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

### The iteration

Each row `i` of `l ≤ Ãx ≤ ũ` has a lower slack `s_l = Ãx − l̃` and an upper slack
`s_u = ũ − Ãx`, each with a multiplier `z_l`, `z_u` that must stay nonnegative. `μ` is the mean
of the products `s·z` over every inequality side, and it measures how far the point is from
the boundary where a bound is active. A solve goes through these steps.

1. **Starting point.** Unless a point is supplied, `x` solves
   `(P̃ + δ_p I + ÃᵀÃ) x = −q̃ + Ãᵀt`, where `t` is the midpoint of a two-sided row, the finite
   bound of a one-sided row, the bound of an equality row and zero for a free row. The slacks
   are the distances of `Ãx` to the bounds, shifted up until all are positive, and the
   multipliers start at one, then both take Mehrotra's balancing shift. A warm start seeds `x`
   and `y` and rebuilds the slacks and multipliers from them.
2. **Weights and factorization.** `w` is computed from the current slacks and multipliers, as
   above, and the Newton system is factored at it.
3. **Predictor.** One solve gives the direction that aims `μ` at zero. The longest step along
   it that keeps every slack and multiplier nonnegative gives the `μ_a` the step would reach.
4. **Centering.** The centering parameter is `σ = max((μ_a/μ)³, min(1, 0.1‖r‖∞/μ))`, where `r`
   is the largest Newton residual. A predictor that makes little progress gives `σ` near one
   and the corrector re-centers. The second term stops `μ` from falling while the residuals are
   still large.
5. **Corrector.** A second solve, with the factorization from step 2, aims at `σμ` and adds the
   second-order term `Δs_a ∘ Δz_a` from the predictor.
6. **Step.** The step length is `step_fraction`, `0.99` by default, of the way to the boundary,
   capped at one, and primal and dual variables share it. The Newton solves are refined
   `refine_iter` times against the regularized system.

A problem with no inequality side has no complementarity: each iteration takes one solve and a
full step. An equality row has no slacks and its weight is `1/δ_d`; a free row has weight `δ_d`.

### When it stops

`check_termination = 1` by default, so the residual and gap tests run at every iteration, at
the tolerances `eps_abs` and `eps_rel` (`1e-8`). The run ends in one of these ways.

| status | when |
|---|---|
| `SOLVED` | the primal and dual residuals, and the duality gap if `check_dualgap` is set, meet the tolerances |
| `PRIMAL_INFEASIBLE`, `DUAL_INFEASIBLE` | the normalized iterate, or the last step, passes the certificate test ADMM uses |
| `SOLVED_INACCURATE` and the `*_INFEASIBLE_INACCURATE` statuses | the tests pass only at ten times the tolerances, and only when the run stalls or reaches `max_iter` |
| `NUMERICAL_ERROR` | the Newton system cannot be factored after `max_reg_bumps` increases of the regularization, a residual is not finite, conjugate gradients miss `cg_fail_limit` solves in a row, the iterate diverges past a ceiling set from `1/sqrt(eps)` times the size of the data and no certificate passes, or the run stalls and neither test passes |
| `MAX_ITER_REACHED` | `max_iter` iterations (`100`) and neither test passes |
| `TIME_LIMIT_REACHED`, `INTERRUPTED` | `time_limit` or an interrupt, with the point reached |

Two guards watch for trouble. A step shorter than a small threshold for several iterations in a
row stalls the run: the tests run at once on the last step and the normalized iterate, then at
ten times the tolerances, and the run ends `NUMERICAL_ERROR` if neither passes. A `μ` that has
stopped falling, or an iterate past the bound, raises an alert instead: the tests then run at
every iteration whatever `check_termination` says, and the run continues. An infeasible
problem's multipliers grow without bound, and that growth is the evidence the tests read.

When the Newton system cannot be factored, the solver multiplies both regularizations by ten
and tries again, at most `max_reg_bumps` times in one solve. Every solve starts from `reg_primal`
and `reg_dual` again.

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

## The dual active-set method

[`ActiveSet`](@ref) is the dual active-set method of Arnström, Bemporad and Axehill (*A dual
active-set solver for embedded quadratic programming using recursive LDLᵀ updates*, IEEE
Transactions on Automatic Control, 2022). It solves the dual of the QP, which is a
least-distance problem, and it solves it exactly: the answer is the solution of an
equality-constrained QP over the rows held at a bound. It forms no linear system that is
factored once and reused. It keeps a factorization of the held rows and changes it by one row
at a time.

### The reduction

With `P = RᵀR` and `v = R⁻ᵀq`, the substitution `x = R⁻¹(−u − v)` turns the QP into

```math
\min_u \tfrac12\|u\|^2 \quad \text{s.t.} \quad
-(b_u + Mv) \le Mu \le -(b_l + Mv), \qquad M = A R^{-1},
```

where `b_l ≤ Ax ≤ b_u` are the caller's bounds. Each row of `M` is divided by its norm, with
the bounds divided the same way, so a tolerance means the same thing on every row. `R` is the
Cholesky factor of `P`, or of `P + εI` when `eps_prox > 0`, kept in `P`'s own form: a dense
triangle, a diagonal, a block-diagonal or a Kronecker factor. The reduction is built once in
[`setup`](@ref). `M` is formed and stored when `A` and `R` are both dense. Otherwise the
reduction holds `A` and `R` and derives a row of `M` as one row of `A` and one triangular
solve, and `Mu` as one triangular solve and one product with `A`, so storage grows with `m`
only through vectors of length `m`. A sparse or banded `P` or `A` is read into a dense matrix,
since there is no sparse factor and no row read in `O(nnz)`.

### The iteration

The **working set** is the set of rows of `M` held at a bound. `Mₐ` denotes its rows, and `tₐ`
the bounds they are held at. Each pass does one of these things.

1. **The set is singular.** A row of the set is a combination of the others. The next section
   covers this.
2. **Some multiplier has the wrong sign.** The multipliers `μ` solve `Mₐ Mₐᵀ μ = tₐ`. An
   inequality row needs a multiplier whose sign matches the side it is held at, up to a noise
   floor that scales with the conditioning of the set. When one is wrong, the pass moves the
   multipliers from their current values toward the solved ones until the first reaches zero,
   and drops that row.
3. **The multipliers are feasible.** The point `u = Mₐᵀ μ` is dual feasible. Every row is
   priced, `Mu` against its bounds in the caller's units. If no row is violated by more than
   `primal_tol` the run ends. Otherwise the worst violator enters the set, held at the bound it
   violates.

Each pass adds or drops one row. Equality rows are in the set from the first pass, and a step
never drops one, because an equality's multiplier is free in sign. The set holds at most `n`
equality rows: any beyond the `n`th is priced as an inequality, since `n` rows span everything
the others can say.

### Dependent rows and infeasibility

A row that enters the set can be a combination of the rows already held. Then `Mₐ Mₐᵀ` is
singular and the pass walks the null direction `p` of the held rows instead of solving for
multipliers. Moving the multipliers along `p` leaves `Mₐᵀ μ` unchanged, and it ends in one of
three ways.

- **A multiplier reaches zero.** That inequality row drops and the run continues.
- **Nothing blocks, and the dependent row's own bound agrees with what the rows it depends on
  imply.** The row is redundant and is dropped. Two equality rows that state the same
  constraint, or one that is a multiple of another, take this path. "Agrees" means the two
  differ by at most `primal_tol`, in the units pricing reads in.
- **Nothing blocks, and the bound disagrees.** `p` is then an unbounded ray of the dual, and the
  dual of a convex QP is unbounded exactly when the primal is infeasible. The method does not
  report this until it has checked `p`, written back to one entry per row of the caller's
  problem as `y`, against the caller's own `A`, `l` and `u`. After the projection onto the polar
  of the recession cone of `[l, u]`, the support function of `[l, u]` at `y` must be negative, `Aᵀy`
  must be at rounding level against the terms that cancel in it, `|yᵢ|‖aᵢ‖`, and the radius
  `ρ = −support/‖Aᵀy‖₁` within which no `x` satisfies the rows must be at least
  `infeasibility_radius`, `primal_tol^(-1/2)` or about `8e3` by default. Both sums are floating
  point, so each side of the radius test is moved against the claim by `(m + 2)·eps` times the
  sum of its absolute terms. A ray that passes gives `PRIMAL_INFEASIBLE`. One that fails means
  the rank decision came from rounding, or that the rows are feasible only at a point larger
  than the radius, and the run reports `NUMERICAL_ERROR`.

A row dropped as redundant that enters and is dropped again, or more such drops than there are
rows, ends the run as a cycle, `NUMERICAL_ERROR`.

### What the working set factors: `working_set`

The set changes by one row per pass, and neither factorization is ever recomputed from
scratch. A row entering or leaving updates it in place.

| `working_set` | what is factored | the rank test | at most |
|---|---|---|---|
| `:rows` (default) | the `QR` of `Mₐᵀ`, an `n × k` matrix, updated by appending and deleting a column | `|R_ii|`, the part of the entering row orthogonal to the held rows | `n` rows |
| `:gram` | the `LDLᵀ` of `Mₐ Mₐᵀ`, a `k × k` matrix, extended by one row on entry and repaired by a rank-one update on deletion | the diagonal entry of `D`, recomputed from an explicit residual when it is small | `min(m, n) + 1` rows |

`Mₐ Mₐᵀ` squares the condition number of `Mₐ`. That makes `:gram` cheaper per pass, and it
squares the conditioning of the multipliers, which solve `Mₐ Mₐᵀ μ = tₐ` in either
representation. The diagonal entry of `D` for a row that is nearly dependent on the held rows is
a difference of two numbers near one, so rounding can dominate it. When it is below
`100·eps/min(D)`, which includes one the plain formula rounded to zero, the factorization
recomputes it as `‖m − Mₐᵀc‖²` for the projection
coefficients `c` of the entering row `m` on the held rows, refined once, and stores an entry at
or below `eps` as zero, which marks the row dependent. `:rows` reads `|R_ii|` directly. Both give
the same answer wherever both reach one. [Choosing a working set for `ActiveSet`](@ref) measures
the cost and says how to pick.

The `:rows` set holds at most `n` rows, so a row that would be the `(n+1)`th is dependent by
counting. It is not added. The pass computes its coefficients against the rows held, which is
the null direction, and steps along that.

### How many rows a pass prices: `scan`

| `scan` | rows priced per pass | the row that enters |
|---|---|---|
| `:all` (default) | every row, with one product | the worst violator |
| `:window` | a window of `max(256, m ÷ 8)` rows, resumed after the row that entered last | the worst violator in the window |

A window that holds no violated row prices the remaining rows before the run ends, so
`:window` never stops early or misses a violated row. It makes a pass cheaper and takes more
passes, and which effect wins does not follow from the problem's dimensions. When `M` is not
stored, one product prices every row and a window saves nothing. After `4(m + 1)` passes either
setting prices every row and enters the lowest-indexed violated row, Bland's rule, which
terminates finitely. With the default `max_iter = 1000` that happens only when `m < 249`.
[`faster_scan`](@ref) times both settings on a problem of yours.

### The proximal-point loop

With `eps_prox = 0` the method needs `P ≻ 0` and one pass of the iteration above is the whole
solve. With `eps_prox > 0` it solves `P + εI` instead, and wraps the iteration in an outer
loop: each outer pass sets `v = R⁻ᵀ(q − εx)` at the current `x`, rebuilds the bounds, and runs
the inner iteration starting from the working set the last pass ended on. This accepts a
singular `P`, a linear program being the extreme case. The loop stops when `x` moves by less
than `eta_prox` times `max(1, ‖x‖∞)` in the ∞-norm, or after `max_prox` outer passes.

A problem that is unbounded below keeps the loop from settling: every pass moves `x` along a
direction the cost does not resist, and the components that curvature resists shrink
geometrically. When the loop ends on `max_prox` passes, its last step `d` is tested as a
certificate. The components of `d` below `sqrt(eps)/16` of its largest are set to zero, which
is rounding by then. It certifies when `qᵀd < 0`, the floating-point product `Pd` is zero, and
no row moves toward a bound that is present: `Ad` has no positive entry at a row with a finite
upper bound and no negative entry at one with a finite lower bound. The status is then
`DUAL_INFEASIBLE`, and `d` scaled to unit ∞-norm is in `Solution.dual_inf_cert`. A floating-point
zero is not an exact one, so the residual of the certificate in exact arithmetic is rounding.

The conditions hold exactly or not at all, because a tolerance cannot tell an unbounded problem
from a bounded one with a minimizer beyond what the passes reach. Curvature of `1e-9` along a
variable, a cost scaled down by `1e-9`, or a row tilted by `1e-8` toward the direction all make
a bounded problem that a tolerance of `1.5e-8` would call unbounded. A step that is a direction
only to rounding, which includes the bundle master problem of darnstrom/daqp#222, ends the run
at `MAX_ITER_REACHED` with the point reached. `DUAL_INFEASIBLE` therefore misses some unbounded
problems and is not reported for a bounded one. An exact direction can exist on an infeasible
problem too, where `PRIMAL_INFEASIBLE` is the sharper verdict. The loop also stops when the step
is small against `‖x‖∞`, so an unbounded problem with columns scaled by large factors can end
`SOLVED` at a point the passes had not left.

### What a solve returns

| status | when |
|---|---|
| `SOLVED` | no row prices in by more than `primal_tol`, and the point formed from the working set leaves no present bound by more than `primal_tol·(max(1, ‖aᵢ‖₂‖x‖∞) + |zᵢ|)` |
| `PRIMAL_INFEASIBLE` | a singular step finds nothing to block it, and the ray passes the check against the caller's rows: no `x` with `‖x‖∞` below about `8e3` satisfies them |
| `DUAL_INFEASIBLE` | the proximal-point loop ends on `max_prox` passes and its last step is an exact recession direction of descent; `x` and `y` are `NaN` |
| `MAX_ITER_REACHED` | `max_iter` passes (`1000`) without ending, or `max_prox` outer passes without converging and without a certificate; the point reached is returned |
| `NUMERICAL_ERROR` | the infeasibility ray fails its check, a multiplier step finds no row to drop, the pass cycles, or the point formed from the working set violates a row; `x` and `y` are `NaN` |

After a `NUMERICAL_ERROR` or a `PRIMAL_INFEASIBLE` the solver empties the working set to the
equality rows, because it holds the dependent rows of the failed pass and the next solve, with
other bounds, would read that dependency again. That holds through [`update!`](@ref) and
[`warm_start!`](@ref) as well. `PRIMAL_INFEASIBLE` leaves
`prim_inf_cert` empty, and there are no `*_INACCURATE` statuses. Polishing is refused, because
the answer already solves the equality-constrained QP over the working set that polishing would
compute.

A bound is absent when it is infinite, which `PureQPBase` stores as `-1e30` for a lower bound
and `1e30` for an upper one. Any other bound is a constraint, however large, including a lower
bound of `1e30` or an upper bound of `-1e30`. The reduction keeps an absent bound infinite when
it divides a row and its bounds by the row's norm, so the norm does not turn it into a bound.
A row held at a bound that an [`update!`](@ref) removes leaves the working set. The termination
tests in `PureQPBase` read a bound beyond `1e26` as infinite, which `ActiveSet` does not.

[`update!`](@ref) keeps the reduction when only `q`, `l` or `u` change and rebuilds it when `P`
or `A` change. It keeps the working set too, which is what makes a re-solve cheap, unless the
new bounds move a row between equality and inequality, which resets the set to the equality
rows. [`cold_start!`](@ref) drops the working set.

## Equilibration

This section and the four after it apply to [`OperatorSplitting`](@ref) and
[`InteriorPoint`](@ref). [`ActiveSet`](@ref) does its own row normalization, described above,
and needs none of them.

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
