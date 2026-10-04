# Operators from functions

A [`LinearMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) is a matrix you never
store. You give it two functions instead of entries: one that multiplies by it, one that
multiplies by its transpose. The solver takes it anywhere it takes a matrix.

A map built from functions reaches the solver as an operator with no entries. A map over a
matrix does not: `LinearMap(B)` arrives as `B`, `kron` of two such maps as a
[`PureQPBase.KroneckerOperator`](@ref), `blockdiag` of them as a
[`PureQPBase.BlockDiagonal`](@ref), and a real multiple of any of these as the same type with
the scalar folded in. Every other composition, and every map built from functions, is an
operator. [What a composed map becomes](@ref) has the full list.

Three worked problems follow. Each has a constraint matrix that would be large, dense, or that
nobody ever assembles, and each one is solved without building it.

All three need the two things listed in [An operator from LinearMaps.jl](@ref): load `Krylov`
and pass `scaling = 0`. The objective is a `LinearMap` over a `Diagonal`, which arrives as that
`Diagonal`, so it needs no declaration of `issymmetric` or `isposdef`; a `P` built from a
function would. Each example states the problem, draws the operator, then solves it twice — once
with the map, once with every matrix written out — and prints the difference between the two
answers.

All three use [`OperatorSplitting`](@ref), the default, which needs nothing more.
[`InteriorPoint`](@ref) solves an operator too, but it wants a preconditioner you supply
instead of the automatic one `scaling = 0` gives up. See
[Operators under the interior-point method](@ref).

## 1. Fitting to sensor readings

You have a signal of 240 points but only measured every fifth one. You want the closest signal
to a target that still agrees with all 48 readings to within 0.05.

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x\|_2^2 + q^T x \\
  \mbox{subject to} & b - 0.05 \le A x \le b + 0.05
\end{array}
```

Here `q = -2t + noise` for the target `t`. So the objective is `‖x - x₀‖²` up to a constant,
where `x₀ = -q/2` is the noisy target, and `b` holds the 48 readings. The constraint matrix
picks out the measured entries. Its transpose puts them back where they came from:

```math
(Ax)_j = x_{5j-4},
\qquad
(A^{\top}y)_i = \begin{cases} y_j & i = 5j-4 \\ 0 & \text{otherwise} \end{cases}
```

Written out, `A` is a 48×240 array with a single 1 in each row. Written as code, it is one
indexing step in each direction, and it stores only the index range:

```math
x \in \mathbb{R}^{240}
\;\xrightarrow{\;\;[\,1{:}5{:}240\,]\;\;}\;
Ax \in \mathbb{R}^{48}
\qquad\qquad
y \in \mathbb{R}^{48}
\;\xrightarrow{\;\;\text{scatter to } 1{:}5{:}240\;\;}\;
A^{\top}y \in \mathbb{R}^{240}
```

```math
A \;=\; \underbrace{I_{240}[\,1{:}5{:}240,\;:\,]}_{48 \times 240,\ \text{48 ones}}
\qquad \text{stored: the range } 1{:}5{:}240 \text{, and nothing else}
```

```@example lm_sensors
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(1)

n, idx = 240, 1:5:240
m = length(idx)

A = LinearMap{Float64}(
    (y, x) -> (@views y .= x[idx]),                      # keep the measured entries
    (x, y) -> (fill!(x, 0.0); @views x[idx] .= y; x),    # scatter them back
    m, n,
)

truth = [sin(2pi * k / n) for k in 1:n]
b = truth[idx]                                           # what the sensors read

P = LinearMap(Diagonal(fill(2.0, n)))
q = -2 .* truth .+ 0.05 .* randn(n)

sol = PureOSQP.solve(P, q, A, b .- 0.05, b .+ 0.05; scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9)
(sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with `A` written out as its 48×240 array, and `P` as a `Diagonal`:

```@example lm_sensors
Ad = Matrix(1.0I, n, n)[idx, :]
dense = PureOSQP.solve(Diagonal(fill(2.0, n)), q, Ad, b .- 0.05, b .+ 0.05; eps_abs = 1e-9, eps_rel = 1e-9)
(dense.status, maximum(abs, sol.x .- dense.x))
```

This is the easiest case to start with. A measurement operator that selects, masks or reorders
is only bookkeeping, and storing it as a matrix buys nothing.

## 2. A constraint on a 2-D grid

A 14×14 image, with a limit on how fast it may change from one pixel to the next, in both
directions at once. The image is 196 variables, so a constraint matrix would be 392×196.

Call the image `X` and let `x = vec(X)` stack its columns into one vector. The problem is to
stay close to a target image `X₀` and keep every neighbor difference within 0.25:

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x - \mathrm{vec}(X_0)\|_2^2 \\
  \mbox{subject to} & -0.25 \le (I \otimes D)\,x \le 0.25 \\
                    & -0.25 \le (D \otimes I)\,x \le 0.25
\end{array}
```

`D` is the 14×14 difference operator along one axis, lower bidiagonal with 1 on the diagonal
and −1 below it:

```math
(Dv)_1 = v_1, \qquad (Dv)_i = v_i - v_{i-1} \quad (i = 2, \dots, 14)
```

To see which pixels each block ties together, use a 4×4 image instead of the 14×14 one. `vec`
stacks the columns, so `x` holds column 1, then column 2, and so on. Neighbors down a column sit
next to each other in `x`. Neighbors along a row sit four places apart. The top row below is the
image, with each pixel's position in `x` in its corner. The strip underneath is `x` itself, and
the arcs show the pairs each block couples.

::: details Code that draws the figure

```@example lm_grid_figure
using CairoMakie

fig = Figure(size = (900, 500))
titles = ("vec order", "I ⊗ D: pairs down each column", "D ⊗ I: pairs along each row")
axs = [Axis(fig[1, c]; title = titles[c], aspect = DataAspect()) for c in 1:3]

# The image: cell (i, j) is pixel xᵢⱼ, entry i + 4(j - 1) of x.
center(i, j) = Point2f(j - 0.5, 4.5 - i)
for ax in axs, j in 1:4, i in 1:4
    poly!(ax, Rect2f(j - 1, 4 - i, 1, 1); color = (:steelblue, 0.12), strokecolor = :black, strokewidth = 1)
end
lines!(axs[1], [center(i, j) for j in 1:4 for i in 1:4]; color = ("#CC79A7", 0.9), linewidth = 2)
linesegments!(axs[2], [center(i, j) => center(i - 1, j) for j in 1:4 for i in 2:4]; color = ("#0072B2", 0.6), linewidth = 6)
linesegments!(axs[3], [center(i, j) => center(i, j - 1) for j in 2:4 for i in 1:4]; color = ("#E69F00", 0.6), linewidth = 6)
for ax in axs, j in 1:4, i in 1:4
    text!(ax, center(i, j); text = L"x_{%$i%$j}", align = (:center, :center), fontsize = 16)
    text!(ax, j - 0.95, 4.97 - i; text = string(i + 4(j - 1)), align = (:left, :top), fontsize = 10, color = :gray40)
end
foreach(ax -> (hidedecorations!(ax); hidespines!(ax)), axs)

# The vector x = vec(X), one cell per entry, in the order the path above visits them.
axv = Axis(fig[2, 1:3]; aspect = DataAspect())
for k in 1:16
    i, j = (k - 1) % 4 + 1, (k - 1) ÷ 4 + 1
    poly!(axv, Rect2f(k - 1, 0, 1, 1); color = (:steelblue, 0.12), strokecolor = :black, strokewidth = 1)
    text!(axv, k - 0.5, 0.5; text = L"x_{%$i%$j}", align = (:center, :center), fontsize = 14)
    text!(axv, k - 0.95, 0.97; text = string(k), align = (:left, :top), fontsize = 9, color = :gray40)
end
for j in 1:4
    linesegments!(axv, [Point2f(4j - 3.9, 1.7) => Point2f(4j - 0.1, 1.7)]; color = :black)
    text!(axv, 4j - 2, 1.8; text = "column $j", align = (:center, :bottom), fontsize = 12)
end
arc(a, b, y0, s) = [Point2f((a + b) / 2 + (b - a) / 2 * cos(t), y0 + s * (b - a) / 2 * sin(t)) for t in range(0, π; length = 40)]
for k in 2:16
    (k - 1) % 4 == 0 || lines!(axv, arc(k - 1.5, k - 0.5, 1, 1); color = "#0072B2", linewidth = 2)
end
for k in 5:16
    lines!(axv, arc(k - 4.5, k - 0.5, 0, -0.6); color = "#E69F00", linewidth = 2)
end
text!(axv, -0.3, 1.3; text = "I ⊗ D: adjacent entries", align = (:right, :center), fontsize = 13, color = "#0072B2")
text!(axv, -0.3, -0.6; text = "D ⊗ I: entries 4 apart", align = (:right, :center), fontsize = 13, color = "#E69F00")
hidedecorations!(axv); hidespines!(axv)
limits!(axv, -6.5, 16.5, -1.5, 2.4)
nothing # hide
```

:::

```@example lm_grid_figure
fig # hide
```

Pixel ``x_{ij}`` is entry ``i + 4(j-1)`` of `x`. On this ordering the two Kronecker products
are

```math
I_4 \otimes D = \begin{pmatrix} D & & & \\ & D & & \\ & & D & \\ & & & D \end{pmatrix}
\qquad\qquad
D \otimes I_4 = \begin{pmatrix} I & & & \\ -I & I & & \\ & -I & I & \\ & & -I & I \end{pmatrix}
```

Each `D` block in `I ⊗ D` acts on the four entries that make up one column, so it subtracts the
pixel above. Each `−I, I` pair in `D ⊗ I` acts on entries four apart, so it subtracts the pixel
to the left:

```math
\big((I \otimes D)\,x\big)_{i + 4(j-1)} = x_{ij} - x_{i-1,\,j}
\qquad\qquad
\big((D \otimes I)\,x\big)_{i + 4(j-1)} = x_{ij} - x_{i,\,j-1}
```

The first row and first column pass through unchanged, as `D` does. On the 14×14 image the
blocks are 14×14 and the row neighbor sits 14 places back. Nothing else changes.

A 2-D operation built from a 1-D one along each axis is a **Kronecker product**. `I ⊗ D` applies
`D` down every column of `X`, and `D ⊗ I` applies it along every row.

```math
X \in \mathbb{R}^{14 \times 14}
\;\xrightarrow{\;\;D \text{ down each column}\;\;}\;
DX = \mathrm{unvec}\big((I \otimes D)\,x\big)
\qquad\qquad
X
\;\xrightarrow{\;\;D \text{ along each row}\;\;}\;
XD^{\top} = \mathrm{unvec}\big((D \otimes I)\,x\big)
```

`kron` combines maps without forming anything. So the constraint matrix is the two blocks
stacked, and only the 14×14 factors exist:

```math
A \;=\; \begin{pmatrix} I_{14} \otimes D \\ D \otimes I_{14} \end{pmatrix}
\qquad 392 \times 196,
\qquad \text{stored: } D \text{ and } I_{14} \text{, both } 14 \times 14
```

One small difference operator serves both axes:

```@example lm_grid
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(2)

k = 14
n = k * k

# Difference between neighbours, along one axis. Both directions supplied.
D = LinearMap{Float64}(
    (y, x) -> (y[1] = x[1]; @views y[2:k] .= x[2:k] .- x[1:(k - 1)]; y),
    (x, y) -> (@views x[1:(k - 1)] .= y[1:(k - 1)] .- y[2:k]; x[k] = y[k]; x),
    k, k,
)
Ik = LinearMap(Matrix(1.0I, k, k))

A = [kron(Ik, D); kron(D, Ik)]     # down the columns, then across the rows

img = [exp(-((i - 7)^2 + (j - 7)^2) / 18) for i in 1:k, j in 1:k]
P = LinearMap(Diagonal(fill(2.0, n)))
q = -2 .* vec(img)

sol = PureOSQP.solve(
    P, q, A, fill(-0.25, 2n), fill(0.25, 2n);
    scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
)
(size(A), sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with `D` written out and both Kronecker products formed as 196-column arrays:

```@example lm_grid
Dd = Bidiagonal(ones(k), -ones(k - 1), :L)
Ad = [kron(I(k), Dd); kron(Dd, I(k))]
dense = PureOSQP.solve(
    Diagonal(fill(2.0, n)), q, Ad, fill(-0.25, 2n), fill(0.25, 2n);
    eps_abs = 1e-9, eps_rel = 1e-9,
)
(size(Ad), dense.status, maximum(abs, sol.x .- dense.x))
```

Only the 14×14 factor is ever stored. The saving grows fast. The same construction on a 256×256
image gives a constraint matrix with `2.1e10` entries, built from a 256×256 one.

## 3. Reusing a model you already have

Sometimes the operator is code somebody already wrote: a filter, a simulator, a forward model.
You can call it, but nobody ever assembled it, and assembling it would take one call per column.

Here it is a first-order recursion, `yₖ = 0.6·yₖ₋₁ + xₖ`, the discrete form of a system that
carries part of its state forward. The problem is to stay close to a random target `x₀` and keep
the system's output within ±1.5 at every step:

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x - x_0\|_2^2 \\
  \mbox{subject to} & -1.5 \le A x \le 1.5
\end{array}
```

Unroll the recursion from `y₀ = 0` and every output becomes a decaying sum of the inputs before
it. That is `A` as a matrix: lower triangular, with `a = 0.6`:

```math
y_k = \sum_{j \le k} a^{\,k-j} x_j
\qquad\Longrightarrow\qquad
A = \begin{pmatrix} 1 & & & \\ a & 1 & & \\ a^2 & a & 1 & \\ \vdots & \ddots & \ddots & \ddots \end{pmatrix}
\qquad \text{stored: } a \text{, and nothing else}
```

Its transpose is the same recursion run backwards, so each direction is one loop:

```math
x
\;\xrightarrow{\;\;y_k = a\,y_{k-1} + x_k,\ \ k = 1, \dots, n\;\;}\;
Ax
\qquad\qquad
y
\;\xrightarrow{\;\;z_k = a\,z_{k+1} + y_k,\ \ k = n, \dots, 1\;\;}\;
A^{\top}y
```

```@example lm_model
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(3)

n, a = 150, 0.6

function forward!(y, x)
    acc = 0.0
    for i in eachindex(x)
        acc = a * acc + x[i]
        y[i] = acc
    end
    return y
end

function adjoint!(x, y)                 # the same recursion, backwards
    acc = 0.0
    for i in reverse(eachindex(y))
        acc = a * acc + y[i]
        x[i] = acc
    end
    return x
end

A = LinearMap{Float64}(forward!, adjoint!, n, n)

P = LinearMap(Diagonal(fill(2.0, n)))
q = -2 .* randn(n)

sol = PureOSQP.solve(
    P, q, A, fill(-1.5, n), fill(1.5, n);
    scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
)
(sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with `A` written out from its closed form, `Aᵢⱼ = aⁱ⁻ʲ` for `i ≥ j`:

```@example lm_model
Ad = [i >= j ? a^(i - j) : 0.0 for i in 1:n, j in 1:n]
dense = PureOSQP.solve(
    Diagonal(fill(2.0, n)), q, Ad, fill(-1.5, n), fill(1.5, n);
    eps_abs = 1e-9, eps_rel = 1e-9,
)
(dense.status, maximum(abs, sol.x .- dense.x))
```

As a matrix this is dense and lower triangular: 11 325 nonzeros here, and `n(n+1)/2` in general.
As a map it is two loops over `n` values.

## Choosing between a map and a matrix

A map wins when applying it costs **asymptotically less** than multiplying by its dense form.
All three examples above do: `O(n)` work against an `O(n²)` matrix.

A map is the wrong choice when the product costs the same either way. A map has no entries to
factor, so the solver runs conjugate gradients, which pays every iteration for work a matrix
pays for once. [When it is the wrong tool](@ref) has the measured comparison.
[An operator is not always solved with CG](@ref) has the conditioning limit, where a bare map
does not converge at all.

## What each composition becomes, and which algorithms take it

A `LinearMap` that is built from matrices is recognized rather than treated as opaque: the
composition becomes the base's own representation of it, and the parts stay available to the
reduction. Every form below solves on all three algorithms.

| `A` as a `LinearMap` | becomes | OperatorSplitting | InteriorPoint | ActiveSet |
|---|---|---|---|---|
| `LinearMap(B)` | the matrix itself | `cholesky` | `bunchkaufman` | `rows` |
| `c * M` | the matrix, scaled | `cholesky` | `bunchkaufman` | `rows` |
| `kron(M₁, …, M_k)` | [`PureQPBase.KroneckerOperator`](@ref) | `kronecker` | `product_reduced` | `rows` |
| `LinearMaps.UniformScalingMap(c, n)` | a `Diagonal` of a `FillArrays.Fill` | `cholesky` | `bunchkaufman` | `rows` |
| `cat(M₁, M₂; dims = (1, 2))` | [`PureQPBase.BlockDiagonal`](@ref) | `indirect` | `product_reduced` | `rows` |
| `vcat(M₁, M₂)` | [`PureQPBase.StackedOperator`](@ref) | `indirect` | `product_reduced` | `rows` |
| `hcat(M₁, M₂)` | [`PureQPBase.JoinedOperator`](@ref) | `cholesky` | `bunchkaufman` | `rows` |
| `hvcat((2, 2), M₁, …, M₄)` | a `StackedOperator` of `JoinedOperator`s | `indirect` | `product_reduced` | `rows` |
| `FillMap(c, (m, n))` | a `FillArrays.Fill` | `cholesky` | `bunchkaufman` | `rows` |
| `M₁ + M₂` | [`PureQPBase.SumOperator`](@ref) | `indirect` | `product_reduced` | `rows` |
| `M₁ * M₂` | [`PureQPBase.ComposedOperator`](@ref) | `indirect` | `product_reduced` | `rows` |
| a `FunctionMap` | [`PureQPBase.ProductOperator`](@ref) | `indirect` | `product_reduced` | `rows` |
| `kron(fn, M)` | [`PureQPBase.ProductOperator`](@ref) | `indirect` | `product_reduced` | `rows` |
| `vcat(kron(…), fn)` | [`PureQPBase.StackedOperator`](@ref) | `indirect` | `product_reduced` | `rows` |

Three things the table is worth reading carefully for. A composition whose parts are not all
recognized keeps the composition: the last row is a stack of a Kronecker block and an opaque one,
and the Kronecker block still contracts its factors inside it. An `hcat` reaches the dense
backends where a `vcat` does not: a join's blocks each own a slice of the columns and the join
holds no structure a dense factor would lose, so it declares none, while a stack keeps its
blocks' rows apart and is served by the rungs that work through products. And an operator with no
entries still answers a *row*, as `Aᵀ eᵢ`, which is why the dual active-set method takes every row
here — entries and rows are different questions, and only equilibration needs the first.

The backends are what these problems reach with a `Diagonal` `P` and `scaling = 0`. A different
`P` moves some of them: a block-diagonal pair partitioned alike reaches `block` rather than
`indirect`, and the Kronecker rung needs `P` a scalar multiple of the identity.

Equilibration is the one real restriction, and it is orthogonal to the three columns: column and
row norms are entries, so an operator that supplies only products needs `scaling = 0`, `probe =
true`, or a [`PureQPBase.structural_rows`](@ref) method.

## What a map supplying only products reaches

A map the table above does not cover arrives as a [`PureQPBase.ProductOperator`](@ref): it
answers products and rows, and has no entries. One backend serves it, and the rest say why not.

| asked for | answer |
|---|---|
| `OperatorSplitting`, `linsys = :indirect` or `:auto` | served — conjugate gradients need no entries |
| `OperatorSplitting` or `InteriorPoint`, `linsys = :dense` or `:kkt` | refused: both assemble their matrix from the entries of `P` and `A` |
| `InteriorPoint`, `linsys = :indirect` | refused: it uses conjugate gradients only with a `preconditioner` of your own |
| `ActiveSet` | refused: the reduction forms `A R⁻¹` for the Cholesky factor of `P`, which products do not give |

Where it is served, leaving the operator alone is also the cheaper setup: nothing is copied and
no entry is read. On a pair with `n = 629` and `m = 2396`, `setup` for the matrix-free backend
takes 10.6 ms from the map against 21 ms from a dense copy of it.

A `P` you can materialize is better passed as `Symmetric`, which is the form `setup` wants
anyway: [`PureQPBase.is_symmetric`](@ref) then answers from the type instead of scanning all `n²`
entries, and the reduction skips a transposing copy. It is worth a few percent of `setup` and
nothing of the solve.

## Two packages supply operators

The examples above use [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl).
[SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) is the other choice. The solver
takes either.

| | LinearMaps | SciMLOperators |
|---|---|---|
| how you build one | `LinearMap{T}(forward, adjoint, m, n)` | `FunctionOperator(op, input, output; op_adjoint)` |
| the function you write | `f(y, x)`, output first | `op(w, v, u, p, t)`, output first, then input |
| preparation before first use | none | `cache_operator` for one built with `*` or `inv` |
| a single operator, per product | 0 B | 0 B |
| a sum, `B + C` | 0 B | 0 B |
| a composition, `B * C` | about 1.7 kB | **0 B** |
| stacking, `[B; C]` | yes | not supported |
| values carried with the operator | no | `p`, replaced by `update_coefficients` |
| `issymmetric`, `isposdef` | declared at construction, for a map built from functions | declared at construction |
| a transpose | required | required |

**Use LinearMaps unless `A` is a product of operators.** It is simpler: two functions and no
preparation. It also stacks blocks, which SciMLOperators cannot. The one thing it cannot do is
apply a product for free.

The allocation rows decide it. We measured them at `n = 200`. A LinearMap product allocates
scratch for the intermediate on every application, and stacking blocks does the same. The solver
applies `A` and `Aᵀ` every iteration, so that cost repeats. SciMLOperators asks for the scratch
once, through `cache_operator`, and then applies for free.

SciMLOperators has nothing that matches stacking. `[L; D]` on two of them gives a plain array
holding them, not an operator. So a constraint made of stacked blocks is either a LinearMap, as
in example 2, or a single `FunctionOperator` whose function fills each block's rows of `w`
itself.

### Writing one

Two things differ from LinearMaps.

**The in-place signature is `op(w, v, u, p, t)`** — `w` receives the result, `v` is the vector
being multiplied, and `u`, `p`, `t` are the state, parameters and time that a differential
equation would supply and a quadratic program does not use. A four-argument function is the
*out-of-place* form, where the first argument is the input; writing into it silently
overwrites the caller's vector.

**An operator built with `*` needs `cache_operator` before it can multiply**, as do `kron` and
`inv`. A single operator and a sum do not. `setup` throws for an uncached one, naming the fix,
rather than failing partway into a solve. It also throws for an operator with no transpose:
`Aᵀ` runs every iteration, so a `FunctionOperator` needs either an `op_adjoint` or
`issymmetric = true`.

```@example sciml
using PureOSQP, SciMLOperators, LinearAlgebra, Krylov, Random
Random.seed!(3)

n = 150

# The recursion from example 3, as a SciMLOperator. Note the five arguments, and that the
# decay travels as the operator's `p` rather than as a captured variable: a captured global
# would make every product allocate.
fwd!(w, v, a) = (acc = 0.0; for i in eachindex(v); acc = a * acc + v[i]; w[i] = acc; end; w)
adj!(w, v, a) = (acc = 0.0; for i in Iterators.reverse(eachindex(v)); acc = a * acc + v[i]; w[i] = acc; end; w)

A = FunctionOperator(
    (w, v, u, p, t) -> fwd!(w, v, p), zeros(n), zeros(n);
    op_adjoint = (w, v, u, p, t) -> adj!(w, v, p), islinear = true, p = 0.6,
)
P = FunctionOperator(
    (w, v, u, p, t) -> (w .= 2 .* v), zeros(n), zeros(n);
    op_adjoint = (w, v, u, p, t) -> (w .= 2 .* v),
    islinear = true, issymmetric = true, isposdef = true,
)

q = -2 .* randn(n)
sol = PureOSQP.solve(
    P, q, A, fill(-1.5, n), fill(1.5, n);
    scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
)
(sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with every matrix written out:

```@example sciml
Ad = [i >= j ? 0.6^(i - j) : 0.0 for i in 1:n, j in 1:n]
dense = PureOSQP.solve(
    Diagonal(fill(2.0, n)), q, Ad, fill(-1.5, n), fill(1.5, n);
    eps_abs = 1e-9, eps_rel = 1e-9,
)
(dense.status, maximum(abs, sol.x .- dense.x))
```

### A composed operator

The same system, observed only at every fifth step, as the sensors of example 1 would see it.
The constraint is on what is observed: the output at the 30 sampled instants must stay within
±1.5, and the other 120 are free.

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x - x_0\|_2^2 \\
  \mbox{subject to} & -1.5 \le S L x \le 1.5
\end{array}
```

`L` is the recursion and `S` the selection, so `A = S L` is 30×150 and a vector passes
through the two in turn. Its transpose passes through them in reverse order. Either way the
value in between has to live somewhere:

```math
x \in \mathbb{R}^{150}
\;\xrightarrow{\;\;y_k = a\,y_{k-1} + x_k\;\;}\;
Lx \in \mathbb{R}^{150}
\;\xrightarrow{\;\;[\,1{:}5{:}150\,]\;\;}\;
SLx \in \mathbb{R}^{30}
```

```math
A \;=\; S L
\qquad \text{stored: } a \text{, the range } 1{:}5{:}150 \text{, and one 150-vector for } Lx
```

That 150-vector is what `cache_operator` allocates. Until it has, the product cannot run, and
`setup` says so rather than letting the first iteration fail:

```@example sciml_composed
using PureOSQP, SciMLOperators, LinearAlgebra, Krylov, Random
Random.seed!(4)

n, a = 150, 0.6
idx = 1:5:n
m = length(idx)

fwd!(w, v, a) = (acc = 0.0; for i in eachindex(v); acc = a * acc + v[i]; w[i] = acc; end; w)
adj!(w, v, a) = (acc = 0.0; for i in Iterators.reverse(eachindex(v)); acc = a * acc + v[i]; w[i] = acc; end; w)

L = FunctionOperator(
    (w, v, u, p, t) -> fwd!(w, v, p), zeros(n), zeros(n);
    op_adjoint = (w, v, u, p, t) -> adj!(w, v, p), islinear = true, p = a,
)
# The range travels as `p` for the same reason the decay does above.
S = FunctionOperator(
    (w, v, u, p, t) -> (@views w .= v[p]), zeros(n), zeros(m);
    op_adjoint = (w, v, u, p, t) -> (fill!(w, 0.0); @views w[p] .= v; w), islinear = true, p = idx,
)
A = S * L                                  # the system's output, where it is observed

P = FunctionOperator(
    (w, v, u, p, t) -> (w .= 2 .* v), zeros(n), zeros(n);
    op_adjoint = (w, v, u, p, t) -> (w .= 2 .* v),
    islinear = true, issymmetric = true, isposdef = true,
)
q = -2 .* randn(n)

try
    PureOSQP.setup(P, q, A, fill(-1.5, m), fill(1.5, m); scaling = 0)
catch err
    showerror(stdout, err)
end
```

`cache_operator` takes a vector the length of the operator's input and returns the operator
with its scratch attached:

```@example sciml_composed
A = cache_operator(A, zeros(n))
sol = PureOSQP.solve(
    P, q, A, fill(-1.5, m), fill(1.5, m);
    scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
)
(sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with the recursion written out and the 30 observed rows taken from it:

```@example sciml_composed
Ld = [i >= j ? a^(i - j) : 0.0 for i in 1:n, j in 1:n]
Ad = Ld[idx, :]
dense = PureOSQP.solve(
    Diagonal(fill(2.0, n)), q, Ad, fill(-1.5, m), fill(1.5, m);
    eps_abs = 1e-9, eps_rel = 1e-9,
)
(dense.status, maximum(abs, sol.x .- dense.x))
```

### One operator, several parameters

The decay is a model parameter, and the operator `A` of the first example carries it as `p`.
To solve for another value, ask for a copy of the operator with that `p`: the out-of-place
`update_coefficients` returns one and leaves `A` as it was. Each copy gets its own `setup`,
which takes the operator and its transpose together.

The problem is the one the first example solves, for three decays:

```@example sciml
decays = (0.4, 0.6, 0.8)
sols = map(decays) do a
    Aa = update_coefficients(A, nothing, a, 0.0)      # a copy of `A` with `p = a`
    PureOSQP.solve(
        P, q, Aa, fill(-1.5, n), fill(1.5, n);
        scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
    )
end
[(a, s.status, s.iter, round(s.obj_val; digits = 8)) for (a, s) in zip(decays, sols)]
```

The same three problems with each matrix written out from its closed form:

```@example sciml
map(decays, sols) do a, s
    Ada = [i >= j ? a^(i - j) : 0.0 for i in 1:n, j in 1:n]
    d = PureOSQP.solve(
        Diagonal(fill(2.0, n)), q, Ada, fill(-1.5, n), fill(1.5, n);
        eps_abs = 1e-9, eps_rel = 1e-9,
    )
    (a, d.status, maximum(abs, s.x .- d.x))
end
```

An operator that already holds a matrix — `MatrixOperator`, and the `DiagonalOperator` built
from one — is unwrapped to that matrix instead of being wrapped. Its entries are what
equilibration and the factoring backends need, and hiding them would force a matrix-free solve
on a problem that does not need one. The type survives, so a `DiagonalOperator` still reaches
the diagonal backend.

`setup` takes the operator and its transpose once and holds both. Updating an operator in
place afterwards — through `update_coefficients!`, or a `MatrixOperator`'s `update_func!` —
reaches `A` but not `Aᵀ`, which is a wrong answer rather than an error. Build a new workspace
after changing an operator; the out-of-place `update_coefficients` above does so by
construction, since what it returns is a new operator.
