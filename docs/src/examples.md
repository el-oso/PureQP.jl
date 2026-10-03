# Examples

The first seven sections are the applications from the
[OSQP documentation](https://osqp.org/docs/examples/), rewritten for PureOSQP. The rest cover
the solver's own interface. Every block runs when these docs are built, so every number below is
real output from the code.

Two things differ from the upstream versions. PureOSQP takes the **full symmetric** `P`, not an
upper triangle. And every matrix here is dense. These problems are highly structured and sparse,
so read them as a guide to *formulating* problems, not as a claim about which solver to use on
them. [Matrix types](matrices.md) covers the other storage formats.

## Basic usage

```@example demo
using PureOSQP, PureIPM

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u, OperatorSplitting())
(sol.status, sol.x, sol.obj_val)
```

The default tolerances are `1e-3`, the same as upstream. For a sharper answer, tighten them or
turn on polishing:

```@example demo
sharp = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, polishing = true)
(sharp.x, sharp.obj_val)
```

## Using InteriorPoint

The same problem, solved with [`InteriorPoint`](@ref) instead of
[`OperatorSplitting`](@ref). Its default tolerance is `1e-8`, tighter than `1e-3`, and it
reaches it in far fewer iterations:

```@example demo
ipm = solve(P, q, A, l, u, InteriorPoint())
(status = ipm.status, x = round.(ipm.x; digits = 4), admm_iter = sol.iter, ipm_iter = ipm.iter)
```

[Choosing an algorithm](@ref) compares the two: which converges faster at a given tolerance,
what each supports when you re-solve through [`update!`](@ref), and what throws under each.

## Least-squares

Fit `Aₐx ≈ b` as closely as you can, with bounds on `x` that a plain `\` cannot express.
Unconstrained least-squares has a closed form and needs no solver. Adding `0 ≤ x ≤ 1` changes
that. The form below makes `y = Aₐx - b` a variable of its own, which keeps the objective
diagonal and the constraint matrix sparse.

```math
\begin{array}{ll}
  \mbox{minimize}   & \tfrac12 \|A_d x - b\|_2^2 \\
  \mbox{subject to} & 0 \le x \le 1
\end{array}
```

Adding `y = A_d x - b` turns this into a QP in `(x, y)`. The residual carries the whole
objective. The constraint matrix has two row groups: `m` rows that define `y`, and `n` rows that
bound `x`.

::: details Code that draws the figure

```@example ex_blocks
using CairoMakie

# One block-outline drawing of a constraint matrix. `cols` and `rows` are `name => size`
# pairs, one per variable group and per constraint-row group; a block is
# `(rows, cols, label)` with ranges of group indices. Sizes are drawn in proportion, except
# that a group thinner than `minsize` is widened to it so its blocks can hold a label;
# `margin` is the room left for the row labels, as a fraction of the drawn width. A
# label `I` or `−I` is drawn as the identity's diagonal, `0` as an empty cell, anything else
# as a filled block.
function blockfigure(cols, rows, blocks; minsize = 0, margin = 0.45, size = (640, 400), fontsize = 14)
    w = [max(Float64(s), minsize) for (_, s) in cols]
    h = [max(Float64(s), minsize) for (_, s) in rows]
    x, y = [0.0; cumsum(w)], [0.0; cumsum(h)]
    W, H = x[end], y[end]
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    # y runs downward so that row group 1 is on top, as in the matrix literal.
    for (r, c, label) in blocks
        x0, x1 = x[first(c)], x[last(c) + 1]
        y0, y1 = y[first(r)], y[last(r) + 1]
        if occursin(r"^[-−]?I$", label)
            lines!(ax, [x0, x1], [-y0, -y1]; color = "#E69F00", linewidth = 3)
            text!(ax, x0 + 0.72 * (x1 - x0), -(y0 + 0.3 * (y1 - y0)); text = label,
                  align = (:center, :center), fontsize, color = "#E69F00")
        elseif label == "0"
            text!(ax, (x0 + x1) / 2, -(y0 + y1) / 2; text = "0",
                  align = (:center, :center), fontsize, color = :gray60)
        else
            poly!(ax, Rect2f(x0, -y1, x1 - x0, y1 - y0); color = ("#0072B2", 0.25))
            text!(ax, (x0 + x1) / 2, -(y0 + y1) / 2; text = label,
                  align = (:center, :center), fontsize, color = "#0072B2")
        end
    end
    for xi in x[2:(end - 1)]
        lines!(ax, [xi, xi], [0, -H]; color = :gray70, linestyle = :dash, linewidth = 1)
    end
    for yi in y[2:(end - 1)]
        lines!(ax, [0, W], [-yi, -yi]; color = :gray70, linestyle = :dash, linewidth = 1)
    end
    lines!(ax, Rect2f(0, -H, W, H); color = :black, linewidth = 1.5)
    for (i, (name, _)) in enumerate(cols)
        text!(ax, (x[i] + x[i + 1]) / 2, 0.015H; text = name,
              align = (:center, :bottom), fontsize = fontsize - 2)
    end
    for (j, (name, _)) in enumerate(rows)
        text!(ax, -0.015W, -(y[j] + y[j + 1]) / 2; text = name,
              align = (:right, :center), fontsize = fontsize - 2)
    end
    hidedecorations!(ax); hidespines!(ax)
    limits!(ax, -margin * W, 1.02W, -1.02H, 0.12H)
    return fig
end

fig = blockfigure(["x  (n = 20)" => 20, "y  (m = 30)" => 30],
                  ["y = Ad x − b" => 30, "0 ≤ x ≤ 1" => 20],
                  [(1, 1, "Ad"), (1, 2, "−I"), (2, 1, "I"), (2, 2, "0")]; size = (520, 400))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example lsq
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
m, n = 30, 20
Ad = randn(m, n)
b = randn(m)

# variables (x, y) with y = Ad*x - b
P = [zeros(n, n) zeros(n, m); zeros(m, n) Matrix(1.0I, m, m)]
q = zeros(n + m)
A = [Ad              -Matrix(1.0I, m, m);
     Matrix(1.0I, n, n)  zeros(n, m)]
l = [b; zeros(n)]
u = [b; ones(n)]

sol = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, polishing = true, max_iter = 100_000)
x = sol.x[1:n]
(sol.status, residual = norm(Ad * x - b), in_box = all(-1e-7 .<= x .<= 1 + 1e-7))
```

## Lasso

Least-squares that prefers *simple* answers. The `‖x‖₁` term charges for the total size of the
coefficients and drives most of them to exactly zero. The fit then picks a handful of predictors
instead of using all of them a little. `γ` sets how strongly it picks. It becomes a QP when you
split each coefficient into a positive and a negative part, which is what the extra variables
below do.

```math
\begin{array}{ll}
  \mbox{minimize} & \tfrac12 \|A_d x - b\|_2^2 + \gamma \|x\|_1
\end{array}
```

which becomes, in `(x, y, t)`,

```math
\begin{array}{ll}
  \mbox{minimize}   & \tfrac12 y^T y + \gamma \mathbf{1}^T t \\
  \mbox{subject to} & y = A_d x - b \\
                    & -t \le x \le t
\end{array}
```

`γ` enters only through `q`. That is what [`update!`](@ref) is for: the whole regularization path
reuses one workspace, and each solve warm starts from the last.

Three variable groups and three row groups: the residual definition, then the two halves of
`−t ≤ x ≤ t`.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 10)" => 10, "y  (m = 200)" => 200, "t  (n)" => 10],
                  ["y = Ad x − b" => 200, "x − t ≤ 0" => 10, "x + t ≥ 0" => 10],
                  [(1, 1, "Ad"), (1, 2, "−I"), (1, 3, "0"), (2, 1, "I"), (2, 2, "0"), (2, 3, "−I"),
                   (3, 1, "I"), (3, 2, "0"), (3, 3, "I")]; minsize = 40, size = (560, 480))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example lasso
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, m = 10, 200
Ad = randn(m, n)
x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
b = Ad * x_true .+ 0.5 .* randn(m)

nv = 2n + m
P = zeros(nv, nv)
P[n+1:n+m, n+1:n+m] = Matrix(1.0I, m, m)
In, Im = Matrix(1.0I, n, n), Matrix(1.0I, m, m)
A = [Ad  -Im            zeros(m, n);
     In   zeros(n, m)  -In;
     In   zeros(n, m)   In]
l = [b; fill(-Inf, n); zeros(n)]
u = [b; zeros(n); fill(Inf, n)]

ws = setup(P, zeros(nv), A, l, u; eps_abs = 1e-8, eps_rel = 1e-8, max_iter = 100_000)
for γ in (1.0, 3.0, 10.0)
    update!(ws; q = [zeros(n + m); fill(γ, n)])
    sol = solve!(ws)
    nnz = count(>(1e-4), abs.(sol.x[1:n]))
    println("γ = $γ:  $(nnz) nonzeros, ‖Ax−b‖ = $(round(norm(Ad * sol.x[1:n] - b), digits = 3))")
end
```

Sparsity increases with `γ`, as it should.

!!! note "Reading `refactor_count`"
    `update!` with only `q` never refactorizes. `ws.refactor_count` still grows across
    this loop, because **adaptive ρ** refactorizes too. It is a total over the workspace's
    life, not a count of what `update!` did. To check a particular `update!`, read
    `refactor_count` right before and after it.

## Huber fitting

Robust regression. It replaces the squared loss with the Huber penalty, so outliers do not
dominate:

```math
\phi_{\rm hub}(t) = \begin{cases} t^2 & |t| \le 1 \\ 2|t| - 1 & |t| > 1 \end{cases}
```

The equivalent QP, in `(x, u, r, s)`:

```math
\begin{array}{ll}
  \mbox{minimize}   & u^T u + 2\,\mathbf{1}^T (r+s) \\
  \mbox{subject to} & A_d x - b - u = r - s \\
                    & r \ge 0,\quad s \ge 0
\end{array}
```

`u` carries the quadratic part of the loss, and `r − s` the linear part. The first row group is
the residual `Ad x − u − r + s = b`. The second is the identity over `(r, s)`, which keeps them
nonnegative.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 10)" => 10, "u  (m = 100)" => 100, "r  (m)" => 100, "s  (m)" => 100],
                  ["Ad x − u − r + s = b" => 100, "r, s ≥ 0" => 200],
                  [(1, 1, "Ad"), (1, 2, "−I"), (1, 3, "−I"), (1, 4, "I"), (2, 1:2, "0"), (2, 3:4, "I")];
                  minsize = 30, size = (620, 520))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example huber
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, m = 10, 100
Ad = randn(m, n)
x_true = randn(n) ./ sqrt(n)
clean = rand(m) .>= 0.1                      # 10% of the measurements are outliers
b = Ad * x_true .+ 0.5 .* randn(m) .* clean .+ 10.0 .* randn(m) .* .!clean

nv = n + 3m
P = zeros(nv, nv)
P[n+1:n+m, n+1:n+m] = 2 * Matrix(1.0I, m, m)
q = [zeros(n + m); fill(2.0, 2m)]
Im = Matrix(1.0I, m, m)
A = [Ad                -Im  -Im  Im;
     zeros(2m, n + m)   Matrix(1.0I, 2m, 2m)]

sol = solve(P, q, A, [b; zeros(2m)], [b; fill(Inf, 2m)];
            eps_abs = 1e-8, eps_rel = 1e-8, polishing = true, max_iter = 200_000)

x_huber = sol.x[1:n]
x_lsq = Ad \ b
(huber_error = norm(x_huber - x_true), least_squares_error = norm(x_lsq - x_true))
```

The Huber fit recovers `x_true` several times more accurately than least-squares. Over twelve
seeds at this outlier rate, the Huber estimate had the lower error every time, with a median
error of 0.19 against 0.95.

## Support vector machine

Draw the dividing line between two labeled classes, as far from both as you can. The `xᵀx` term
prefers a wide margin. The `max(0, ·)` hinge charges for every point on the wrong side, and `γ`
sets what a misclassification costs against margin width. The hinge is not quadratic, so each
data point gets one slack variable and one extra row.

```math
\begin{array}{ll}
  \mbox{minimize} & \tfrac12 x^T x + \gamma \sum_{i=1}^m \max(0,\; b_i a_i^T x + 1)
\end{array}
```

with the hinge losses lifted into variables `t`:

```math
\begin{array}{ll}
  \mbox{minimize}   & \tfrac12 x^T x + \gamma \mathbf{1}^T t \\
  \mbox{subject to} & t \ge \mathrm{diag}(b) A_d x + 1,\quad t \ge 0
\end{array}
```

One row group per inequality: the hinge, and the identity that keeps `t ≥ 0`.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 10)" => 10, "t  (m = 200)" => 200],
                  ["diag(b) Ad x − t ≤ −1" => 200, "t ≥ 0" => 200],
                  [(1, 1, "diag(b) Ad"), (1, 2, "−I"), (2, 1, "0"), (2, 2, "I")];
                  minsize = 80, margin = 0.9, size = (600, 420), fontsize = 13)
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example svm
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, m = 10, 200
half = m ÷ 2
b = [ones(half); -ones(half)]
Ad = [randn(half, n) ./ sqrt(n) .+ 1 / n;
      randn(half, n) ./ sqrt(n) .- 1 / n]
γ = 1.0

P = [Matrix(1.0I, n, n) zeros(n, m); zeros(m, n) zeros(m, m)]
q = [zeros(n); fill(γ, m)]
Im = Matrix(1.0I, m, m)
A = [Diagonal(b) * Ad  -Im;
     zeros(m, n)        Im]
l = [fill(-Inf, m); zeros(m)]
u = [fill(-1.0, m); fill(Inf, m)]

sol = solve(P, q, A, l, u; eps_abs = 1e-8, eps_rel = 1e-8, polishing = true, max_iter = 200_000)
w = sol.x[1:n]
accuracy = count(i -> sign((Ad*w)[i]) == -b[i], 1:m) / m
(sol.status, weight_norm = norm(w), training_accuracy = accuracy)
```

## Portfolio optimization

Split a budget across assets to earn as much as you can without taking more risk than you want.
`μ` is the expected return of each asset, and `Σ` says how they move together, so `xᵀΣx` is the
variance of the whole portfolio and `γ` is the return you demand per unit of risk. This is the
textbook Markowitz problem, and it is already a QP. The one below is the factor-model form,
which keeps `Σ` as a small factor matrix plus a diagonal instead of a full covariance.

```math
\begin{array}{ll}
  \mbox{maximize}   & \mu^T x - \gamma\, x^T \Sigma x \\
  \mbox{subject to} & \mathbf{1}^T x = 1,\quad x \ge 0
\end{array}
```

with a factor risk model `Σ = F Fᵀ + D`. Adding `y = Fᵀ x` keeps the quadratic term diagonal.
The constraint matrix stacks the definition of `y`, the budget row, and one bound per asset.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 100)" => 100, "y  (k = 10)" => 10],
                  ["y = Fᵀ x" => 10, "1ᵀ x = 1" => 1, "0 ≤ x ≤ 1" => 100],
                  [(1, 1, "Fᵀ"), (1, 2, "−I"), (2, 1, "1ᵀ"), (2, 2, "0"), (3, 1, "I"), (3, 2, "0")];
                  minsize = 14, size = (560, 560))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example portfolio
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, k = 100, 10
F = randn(n, k) .* (rand(n, k) .< 0.7)
D = Diagonal(rand(n) .* sqrt(k))
μ = randn(n)
γ = 1.0

P = [Matrix(D) zeros(n, k); zeros(k, n) Matrix(1.0I, k, k)]
q = [-μ ./ (2γ); zeros(k)]
A = [F'                  -Matrix(1.0I, k, k);
     ones(1, n)           zeros(1, k);
     Matrix(1.0I, n, n)   zeros(n, k)]
l = [zeros(k); 1.0; zeros(n)]
u = [zeros(k); 1.0; ones(n)]

sol = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, polishing = true, max_iter = 100_000)
x = sol.x[1:n]
(budget = sum(x), smallest_weight = minimum(x),
 expected_return = dot(μ, x), risk = dot(x, (F * F' + D) * x))
```

The budget constraint holds exactly, and the most negative weight is around `1e-20`, which is
zero to machine precision. That is what a first-order method gives you on an active bound. If
you need the weights to be non-negative as a hard guarantee, clamp them.

## Model predictive control

The problem [`update!`](@ref) is built for. To drive a quadcopter to a reference height, you
re-solve a finite-horizon optimal control problem at every step. Only the initial-state rows of
`l` and `u` change, so the solver factors once and reuses that factor.

```math
\begin{array}{ll}
  \mbox{minimize}   & (x_N - x_r)^T Q_N (x_N - x_r) + \sum_{k=0}^{N-1} (x_k - x_r)^T Q (x_k - x_r) + u_k^T R u_k \\
  \mbox{subject to} & x_{k+1} = A x_k + B u_k \\
                    & x_{\min} \le x_k \le x_{\max},\quad u_{\min} \le u_k \le u_{\max} \\
                    & x_0 = \bar x
\end{array}
```

Stack the horizon as `z = (x₀, …, x_N, u₀, …, u_{N−1})` and the dynamics rows come out
block-bidiagonal. Row group `k` holds `Ad` under `x_{k−1}`, `−I` under `x_k`, and `Bd` under
`u_{k−1}`. The first row group pins `x₀` to the measured state, and it is the only part that
changes between solves. Below these rows `A` stacks the identity, one bound per variable. The
figure leaves that out. It also draws the `u` columns wider than their true four, so the blocks
fit their labels.

::: details Code that draws the figure

```@example ex_blocks
nx, nu, N = 12, 4, 10
sub(k) = join(Char(0x2080 + d - '0') for d in string(k))
cols = [["x$(sub(k))" => nx for k in 0:N]; ["u$(sub(k))" => nu for k in 0:N-1]]
rows = [["x₀ = x̄" => nx]; ["k = $k" => nx for k in 1:N]]
blocks = [[(k + 1, k + 1, "−I") for k in 0:N];
          [(k + 1, k, "Ad") for k in 1:N];
          [(k + 1, N + 1 + k, "Bd") for k in 1:N]]
fig = blockfigure(cols, rows, blocks; minsize = 8, margin = 0.3, size = (720, 500), fontsize = 10)
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example mpc
using PureOSQP, LinearAlgebra, Printf

Ad = [1.0 0 0 0 0 0 0.1 0 0 0 0 0
      0 1.0 0 0 0 0 0 0.1 0 0 0 0
      0 0 1.0 0 0 0 0 0 0.1 0 0 0
      0.0488 0 0 1.0 0 0 0.0016 0 0 0.0992 0 0
      0 -0.0488 0 0 1.0 0 0 -0.0016 0 0 0.0992 0
      0 0 0 0 0 1.0 0 0 0 0 0 0.0992
      0 0 0 0 0 0 1.0 0 0 0 0 0
      0 0 0 0 0 0 0 1.0 0 0 0 0
      0 0 0 0 0 0 0 0 1.0 0 0 0
      0.9734 0 0 0 0 0 0.0488 0 0 0.9846 0 0
      0 -0.9734 0 0 0 0 0 -0.0488 0 0 0.9846 0
      0 0 0 0 0 0 0 0 0 0 0 0.9846]
Bd = [0 -0.0726 0 0.0726
      -0.0726 0 0.0726 0
      -0.0152 0.0152 -0.0152 0.0152
      0 -0.0006 0 0.0006
      0.0006 0 -0.0006 0
      0.0106 0.0106 0.0106 0.0106
      0 -1.4512 0 1.4512
      -1.4512 0 1.4512 0
      -0.3049 0.3049 -0.3049 0.3049
      0 -0.0236 0 0.0236
      0.0236 0 -0.0236 0
      0.2107 0.2107 0.2107 0.2107]
nx, nu = size(Bd)

u_hover = 10.5916
umin = fill(9.6, nu) .- u_hover
umax = fill(13.0, nu) .- u_hover
xmin = [-pi/6, -pi/6, -Inf, -Inf, -Inf, -1.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]
xmax = [pi/6, pi/6, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]

Q = Diagonal([0, 0, 10.0, 10, 10, 10, 0, 0, 0, 5, 5, 5])
QN = Q
R = 0.1 * Matrix(1.0I, nu, nu)
x0 = zeros(nx)
xr = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # hover one metre up
N = 10

# Stack the horizon into one QP over z = (x_0, …, x_N, u_0, …, u_{N-1}).
nv = (N + 1) * nx + N * nu
P = zeros(nv, nv)
for k in 0:N
    P[k*nx+1:(k+1)*nx, k*nx+1:(k+1)*nx] = k == N ? QN : Q
end
for k in 0:N-1
    o = (N + 1) * nx + k * nu
    P[o+1:o+nu, o+1:o+nu] = R
end
q = [repeat(-Q * xr, N); -QN * xr; zeros(N * nu)]

Ax = zeros((N + 1) * nx, (N + 1) * nx)
for k in 0:N
    Ax[k*nx+1:(k+1)*nx, k*nx+1:(k+1)*nx] = -Matrix(1.0I, nx, nx)
end
for k in 1:N
    Ax[k*nx+1:(k+1)*nx, (k-1)*nx+1:k*nx] = Ad
end
Bu = zeros((N + 1) * nx, N * nu)
for k in 1:N
    Bu[k*nx+1:(k+1)*nx, (k-1)*nu+1:k*nu] = Bd
end
A = [[Ax Bu]; Matrix(1.0I, nv, nv)]
l = [-x0; zeros(N * nx); repeat(xmin, N + 1); repeat(umin, N)]
u = [-x0; zeros(N * nx); repeat(xmax, N + 1); repeat(umax, N)]

ws = setup(P, q, A, l, u; eps_abs = 1e-6, eps_rel = 1e-6, max_iter = 20_000)

x = copy(x0)
for step in 1:15
    sol = solve!(ws)
    sol.status == PureOSQP.SOLVED || error("step $step: $(sol.status)")
    control = sol.x[(N+1)*nx+1:(N+1)*nx+nu]
    global x = Ad * x + Bd * control
    # Only the initial-state rows change, so no refactorization is needed.
    l[1:nx] .= -x
    u[1:nx] .= -x
    update!(ws; l = l, u = u)
    step % 5 == 0 && @printf("step %2d: height = %.4f, ‖x − xr‖ = %.4f\n", step, x[3], norm(x - xr))
end

(final_height = x[3], factorizations = ws.refactor_count)
```

Fifteen closed-loop solves, **one factorization**. That is the whole reason to use `update!`
instead of rebuilding the workspace. The initial-state bounds move every step, but no row
changes constraint class, so the factorization stays valid.

## Building a workspace once

`solve` builds a workspace, solves, and throws the workspace away. [`setup`](@ref) hands it back
instead. The equilibration factors, the buffers and the factorization then survive to the next
[`solve!`](@ref). So do the iterates, which is what makes the second solve short.

```@example workspace
using PureOSQP, LinearAlgebra

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

ws = setup(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
first_solve = solve!(ws)
warm_solve = solve!(ws)
(dimensions(ws), first_solve.iter, warm_solve.iter)
```

[`cold_start!`](@ref) throws the iterates away and touches nothing else.
[`warm_start!`](@ref) seeds them from a point you already have. Neither touches the
factorization, so neither costs a refactorization.

```@example workspace
cold_start!(ws)
cold = solve!(ws)

cold_start!(ws)
warm_start!(ws; x = first_solve.x, y = first_solve.y)
seeded = solve!(ws)

(cold.iter, seeded.iter)
```

The Lasso section updates `q`, and the MPC section updates `l` and `u`. `P` and `A` are the two
that always refactorize.

```@example workspace
before = ws.refactor_count
update!(ws; P = [6.0 1.0; 1.0 3.0], A = [1.0 1.0; 1.0 0.0; 0.0 2.0])
after = ws.refactor_count
resolved = solve!(ws)
(refactorizations = after - before, x = resolved.x)
```

You can change options and algorithm parameters afterwards. Change an option by keyword. Change
the parameters by passing a new [`OperatorSplitting`](@ref); any parameter you leave out takes
its default. `rho`, `sigma` and `rho_is_vec` are built into the factorization, so changing one
of those refactorizes. The rest are free.

```@example workspace
update_settings!(ws; eps_abs = 1e-6, polishing = true)
update_settings!(ws, OperatorSplitting(alpha = 1.5))
update_rho!(ws, 1.0)
(ws.options.eps_abs, ws.options.polishing, ws.algorithm.alpha, live_rho = ws.rho, setting_rho = ws.algorithm.rho)
```

`update_rho!` sets the value the solver runs with. `ws.algorithm.rho` keeps the one you gave
`setup`. Two keywords throw, because the workspace cannot act on them:

```@example workspace
try
    update_settings!(ws; linsys = :kkt)
catch err
    println(sprint(showerror, err))
end
```

[`capabilities`](@ref) reports what this build supports, for the packages currently loaded:

```@example workspace
capabilities()
```

## What a solve reports

The default tolerances leave a residual around `1e-3`. Polishing guesses the active set at the
ADMM solution, then solves the equality-constrained QP that comes out of it exactly. That
usually takes the KKT error to machine precision, and it costs one factorization.

```@example report
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

plain = solve(P, q, A, l, u, OperatorSplitting())
polished = solve(P, q, A, l, u; polishing = true)
(plain.rel_kkt_error, plain.status_polish, polished.rel_kkt_error, polished.status_polish)
```

`status_polish` names which of the five outcomes you got. `polished` answers the narrower
question of whether it was `POLISH_SUCCESS`. The solver takes the polished point only when it
improves both residuals, so `POLISH_FAILED` means you have the unpolished answer.

Everything else a [`Solution`](@ref) carries:

```@example report
(status = polished.status, iter = polished.iter,
 obj_val = polished.obj_val, dual_obj_val = polished.dual_obj_val,
 duality_gap = polished.duality_gap, rel_kkt_error = polished.rel_kkt_error,
 rho_estimate = polished.rho_estimate, rho_updates = polished.rho_updates)
```

The timings are in seconds, on whatever machine built these docs. `run_time` charges
`setup_time` to the first solve only, so a re-solve reports what that re-solve cost:

```@example report
map(t -> round(t; digits = 6),
    (; polished.setup_time, polished.solve_time, polished.polish_time, polished.run_time))
```

## Measuring how fast a solve converges

`sol.iter` tells you how many iterations a solve took, but not *how* it got there. Two runs can
take the same number of iterations while one spends most of them near the answer and the other
only arrives at the end. `OperatorSplitting(profile_primdual = true)` measures that difference:

```@example primdual
using PureOSQP, LinearAlgebra

n = 30
M = randn(n, n)
P = Matrix(Symmetric(M'M)) + n * I
A = Matrix(1.0I, n, n)
q = randn(n)
l, u = fill(-0.5, n), fill(0.5, n)

sol = solve(P, q, A, l, u, OperatorSplitting(profile_primdual = true); eps_abs = 1e-9, eps_rel = 1e-9)
(iter = sol.iter, trapezoid = sol.primdual_int, logmean = sol.primdual_int_log)
```

**What the number is.** The area under the duality-gap curve over the solve, in
**gap × seconds**. Smaller means the gap shrank sooner. It is a *relative* measure. Use it to
compare two runs of the same problem. On its own it means nothing, and it does not carry across
machines: a different CPU gives a different number for the same solve.

**Why there are two.** They are two estimates of one quantity. The solver knows the gap only
where it refreshes residuals, which is every `check_termination` iterations, 25 by default.
`primdual_int` assumes a straight line between samples. `primdual_int_log` assumes the
exponential decay a converging gap follows. A straight line drawn over a decaying curve sits
above it, so the trapezoid reads high and the truth lies between the two.

**Which to use.** Take `primdual_int_log`, and read the ratio between the two as its error bar.
When they are close, either number is sound. When they are far apart, neither is. At the default
interval the trapezoid runs about 3.4× high and the log-mean about 16% low. Sample every
iteration and the ratio comes to 0.93 ([Benchmarks](@ref "The primal-dual integral")).

```@example primdual
dense = solve(
    P, q, A, l, u, OperatorSplitting(profile_primdual = true); eps_abs = 1e-9, eps_rel = 1e-9,
    check_termination = 1,
)
(ratio_default = sol.primdual_int_log / sol.primdual_int,
 ratio_dense = dense.primdual_int_log / dense.primdual_int)
```

Lower `check_termination` to sample more often. You then test termination more often too.
Profiling itself costs under 1%, and it changes neither the answer nor the iteration count.

## Choosing the linear system

`linsys = :auto` picks a backend from how `P` and `A` are stored, as above. `linsys = :kkt`
overrides that and factors the full `(n+m)×(n+m)` quasi-definite system with Bunch-Kaufman, as
the reference implementation does. It is slower, but it does not square the conditioning of `A`,
so use it when you doubt a result.

```@example report
kkt = setup(P, q, A, l, u; linsys = :kkt, eps_abs = 1e-9, eps_rel = 1e-9)
auto = setup(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
(PureOSQP.backend_name(kkt.linsys), PureOSQP.backend_name(auto.linsys),
 solve!(kkt).x, solve!(auto).x)
```

`linsys = :indirect` is the third. It runs preconditioned conjugate gradients on the reduced
system, which it never forms. Use it for a matrix that can only supply products, or one large
and sparse enough that forming an `n×n` inverse costs the most. It lives in a package extension
over Krylov.jl, so it exists only once you load Krylov. Without it, `setup` says so instead of
falling back.

```julia
using PureOSQP, Krylov          # Krylov.jl is a weak dependency; add it yourself

capabilities().indirect_solver  # true only with Krylov loaded
ws = setup(P, q, A, l, u; linsys = :indirect, cg_max_iter = 20)
solve!(ws)
```

The inner solve is inexact, because its tolerance follows the ADMM residuals. So its iterates
differ from the direct backends' in the last digits, even though both converge to the same
point.

## Solution derivatives

[`adjoint_derivative`](@ref) differentiates the KKT conditions at the solution the workspace
holds. From one factorization it gives you the gradients of a scalar loss against all five
pieces of problem data. Give it `∂L/∂x` and `∂L/∂y`, and it returns `∂L/∂P`, `∂L/∂q`, `∂L/∂A`,
`∂L/∂l` and `∂L/∂u`. Every algorithm supports it. On an `InteriorPointWorkspace` the solve that
produced the solution must have run with `polishing = true`, or it throws and names the
workspace ([Choosing an algorithm](@ref "Polishing, derivatives and infeasibility")).

Here `L = x₁`, with the budget row `x₁ + x₂ = 1` the only active constraint:

```@example deriv
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 1.0, 1.0]

ws = setup(P, q, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10, polishing = true)
sol = solve!(ws)
grad = adjoint_derivative(ws, [1.0, 0.0], zeros(3))
(sol.x, grad.dq, grad.dl)
```

Against central differences on `q`:

```@example deriv
h = 1e-6
loss(qq) = solve(P, qq, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10, polishing = true).x[1]
fd = [(loss(q .+ h .* e) - loss(q .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])]
(grad.dq, fd)
```

[`forward_derivative`](@ref) goes the other way. It gives the derivative of the solution along a
change in the data. Widen the budget from `1` to `1 + t` and both variables move, and the two
moves add up to the extra budget:

```@example deriv
dx, dy = forward_derivative(ws; dl = [1.0, 0.0, 0.0], du = [1.0, 0.0, 0.0])
(dx, sum(dx))
```

The derivative exists only where the active set is stable. A row that rests on its bound with a
zero multiplier makes the solution map non-differentiable, and so does an active-set KKT matrix
that is singular or nearly singular. Both functions throw there. Neither returns a regularized
number that would look like an answer.

### A QP as a differentiable layer

You call the derivatives above by hand. Load
[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) instead and [`solve`](@ref)
becomes differentiable to any AD package that reads ChainRules, Zygote among them. A QP can then
sit inside a loss and you can train through it, with no gradient code of your own.

Here we fit a QP to a target. `q` is the parameter, and the loss is how far the solution lands
from the target.

```@example layer
using PureOSQP, ChainRulesCore, Zygote, LinearAlgebra

P = [4.0 1.0; 1.0 2.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [-1.0, -0.6, -0.6]
u = [1.0, 0.6, 0.6]
target = [0.3, -0.2]

# `q` is what we are fitting. Everything inside the loss is an ordinary solve.
loss(q) = sum(abs2, solve(P, q, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10).x .- target)

q = foldl(1:60; init = [0.5, 0.5]) do qk, _
    qk .- 0.5 .* only(Zygote.gradient(loss, qk))
end
(fitted_q = round.(q; digits = 5),
 x = round.(solve(P, q, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10).x; digits = 5),
 target, loss = round(loss(q); digits = 12))
```

Gradient descent drives the solution onto the target, and it differentiates through the solver
at every step.

**This differentiates the solution, not the iteration.** The rules call
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref), which differentiate the KKT
conditions at the active set. That is one linear solve. It reuses a factorization the solve
already produced, and it does not care how many iterations the solve took.

Two things follow. Both throw rather than return an approximation:

- **The solve must converge.** The KKT conditions hold only at the solution, so a run that
  stopped at `max_iter` raises rather than returning the gradient of a point that is not
  the answer.
- **The active set must be non-degenerate**, which `adjoint_derivative` already requires. A
  least-squares answer there would have the right shape and units but be a different
  quantity, and nothing downstream could tell.

The rules set `polishing = true` for you unless you ask otherwise. The derivative is taken at
the active set, and polishing finds that set exactly.

One limit on reach: `rrule` and `frule` cover every AD backend that reads ChainRules, Zygote
included. Mooncake needs an explicit `Mooncake.@from_rrule`, and Enzyme needs its own
`EnzymeRules` shim.

## Infeasible problems

A problem with no feasible point stops at `PRIMAL_INFEASIBLE` and returns a certificate `v` that
satisfies `Aᵀv = 0` with a negative support value. Here the two rows ask for `x ≥ 1` and
`x ≤ 0`:

```@example infeasible
using PureOSQP, LinearAlgebra

A = [1.0; 1.0;;]
sol = solve([1.0;;], [0.0], A, [1.0, -Inf], [Inf, 0.0])
(sol.status, sol.prim_inf_cert, residual = norm(A' * sol.prim_inf_cert, Inf), sol.x)
```

An unbounded problem stops at `DUAL_INFEASIBLE` and returns a certificate `d`, a direction along
which the objective falls without limit. This example minimizes `-x` over the whole line:

```@example infeasible
unbounded = solve(zeros(1, 1), [-1.0], [1.0;;], [-Inf], [Inf])
(unbounded.status, unbounded.dual_inf_cert, unbounded.obj_val, unbounded.x)
```

Neither carries a primal-dual point, so `x` and `y` come back as `NaN` rather than as the last
iterate. `obj_val` is `Inf` for a primal infeasibility and `-Inf` for a dual one. The
certificate that does not apply comes back as an empty vector:

```@example infeasible
(length(sol.dual_inf_cert), length(unbounded.prim_inf_cert))
```

## JuMP and MathOptInterface

The MathOptInterface wrapper is a package extension. It loads when MathOptInterface does. Every
field of [`Options`](@ref) and every parameter of that optimizer's algorithm is a raw optimizer
attribute of the same name. `PureIPM.Optimizer` is the interior-point counterpart of the one
below. We do not run this block here, because the docs do not depend on JuMP:

```julia
using JuMP, PureOSQP

model = Model(PureOSQP.Optimizer)
set_optimizer_attribute(model, "polishing", true)
set_optimizer_attribute(model, "eps_abs", 1e-9)

@variable(model, 0 <= x[1:2] <= 0.7)
@constraint(model, sum(x) == 1)
@objective(model, Min, 2x[1]^2 + x[1] * x[2] + x[2]^2 + x[1] + x[2])

optimize!(model)
value.(x)        # [0.3, 0.7]
```

`PureOSQP.Optimizer` is the only name the core package owns. The wrapper itself lives in the
extension, so a caller who does not use MathOptInterface pays nothing for it.

## Unmaterialized operators

An `A` given as an operator is never formed: the solver asks it for products and, where it has
structure, for the structure itself. Each solver reaches such an `A` on two paths — one that
iterates the linear system, one that factors it — and this section sets up both for each of the
three algorithms. [What unmaterialized operators cost](@ref) has their timings.

The six paths here are the ones the benchmark times. It runs them on fixed data from
`bench/unmaterialized_problems.jl`, so its numbers are reproducible; these examples draw their
matrices at random instead, to keep each one readable on its own.

### Conjugate gradients

A `LinearMap` reaches the matrix-free backend, which needs nothing but products:

```@example unmat
using PureOSQP, PureQPBase, LinearMaps, LinearAlgebra, Krylov, Random

blur(k) = diagm(0 => fill(0.6, k), 1 => fill(0.2, k - 1), -1 => fill(0.2, k - 1))

Random.seed!(7201)
A = kron(LinearMap(blur(24)), LinearMap(blur(24)))     # 576×576, held as two 24×24 factors
n, m = size(A, 2), size(A, 1)
ws = setup(Diagonal(fill(2.0, n)), randn(n), A, fill(-1.0, m), fill(1.0, m),
           OperatorSplitting(); linsys = :indirect, scaling = 0)
sol = solve!(ws)
(backend_name(ws.linsys), sol.status, sol.iter)
```

### A factorization from the factors

The same shape, named onto the Kronecker backend instead. It diagonalizes the reduced matrix by
the factors' own eigenvectors, so two 24×24 eigenproblems replace one 576×576 factorization:

```@example unmat
Random.seed!(7202)
A = kron(LinearMap(randn(24, 24) ./ 5), LinearMap(randn(24, 24) ./ 5))
n, m = size(A, 2), size(A, 1)
ws = setup(Diagonal(fill(2.0, n)), randn(n), A, fill(-1.0, m), fill(1.0, m),
           OperatorSplitting(); linsys = :kronecker, scaling = 0)
sol = solve!(ws)
(backend_name(ws.linsys), sol.status, sol.iter)
```

### Preconditioned conjugate gradients, interior point

The interior-point method takes its conjugate-gradient path only with a preconditioner: measured
without one, or on the Jacobi diagonal, it does not reach its tolerance on most problems, and it
says so rather than returning a loose answer. When both `P` and `A` are Kronecker products, one
preconditioner diagonalizes the whole reduced matrix — `U₁ ⊗ U₂` makes `P` the identity and
`ÃᵀWÃ` diagonal at the same time:

```@example unmat
using PureIPM

pd(k) = (S = randn(k, k); Matrix(Symmetric(S'S ./ k + 2I)))

Random.seed!(7203)
P = PureQPBase.KroneckerOperator(pd(12), pd(12))
A = PureQPBase.KroneckerOperator(randn(12, 12) ./ 4, randn(12, 12) ./ 4)
n, m = size(A, 2), size(A, 1)
ws = setup(P, randn(n), A, fill(-1.0, m), fill(1.0, m), InteriorPoint();
           linsys = :indirect, scaling = 0,
           preconditioner = PureQPBase.KroneckerPreconditioner(P, A))
sol = solve!(ws)
(backend_name(ws.linsys), sol.status, sol.iter, sol.cg_iters)
```

### The reduced matrix from products

The interior-point method's direct path needs the reduced matrix itself, which an operator with no
entries supplies through products: column `j` of `Aᵀ W A` is `Aᵀ(W(A eⱼ))`, so `2n` products give
the whole matrix and `n²` bounds what is held. An `A` with structure does better than that — a
Kronecker product contracts its factors into the matrix without forming either the product or its
columns. No backend is named here; the ladder reaches it:

```@example unmat
Random.seed!(7204)
A = kron(LinearMap(randn(24, 24) ./ 5), LinearMap(randn(24, 24) ./ 5))
n, m = size(A, 2), size(A, 1)
ws = setup(Diagonal(fill(2.0, n)), randn(n), A, fill(-1.0, m), fill(1.0, m),
           InteriorPoint(); scaling = 0)
sol = solve!(ws)
(backend_name(ws.linsys), sol.status, sol.iter)
```

### A working set read from the factors

The dual active-set method has no linear-system backend. It reads one row of `A` each time its
working set changes, which a Kronecker operator answers from its factors, and it keeps one of two
representations of the set. Both are direct; they differ in what they factor.
[`ActiveSet`](@ref) carries the choice, since it is a property of the method rather than an option:

```@example unmat
using PureDAQP

Random.seed!(7205)
A = PureQPBase.KroneckerOperator(randn(16, 16) ./ 3, randn(16, 16) ./ 3)
n, m = size(A, 2), size(A, 1)
prob = (Diagonal(fill(2.0, n)), randn(n), A, fill(-0.05, m), fill(0.05, m))

qr_ws = setup(prob..., ActiveSet(working_set = :rows); scaling = 0)
qr_sol = solve!(qr_ws)
(qr_sol.status, qr_sol.iter)
```

The same problem with the Gram matrix and an `LDLᵀ` instead of the rows and a `QR`:

```@example unmat
gram_ws = setup(prob..., ActiveSet(working_set = :gram); scaling = 0)
gram_sol = solve!(gram_ws)
(gram_sol.status, gram_sol.iter, isapprox(gram_sol.x, qr_sol.x; atol = 1e-6))
```
