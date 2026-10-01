# Matrix types

How you store `P` and `A` changes how much work the solver does. Sometimes by a factor of
hundreds. You control it by passing a different matrix type, not by setting an option.

This page says which type suits which problem, what backend each one reaches, and how to add
a type of your own. For worked problems see [Examples](@ref). For the numbers behind the
advice, see [Benchmarks](@ref).

## Matrix types

Dense matrices always work. If your problems are small, that is all you need. Stop here and
come back when one gets slow.

Two words appear throughout:

- The **reduced matrix** is the `n×n` matrix the solver solves against every iteration.
  Nearly all the time goes there, so the rest of this page is about keeping it small or
  cheap. It is `R = cDPD + σI + Ãᵀdiag(ρ)Ã`, but you do not need the formula to use any of
  this.
- A **backend** is the code that solves against that matrix. There are about ten. `setup`
  picks one from the types of `P` and `A`. You do not pick it. Ask which one you got with
  `PureQPBase.backend_name(ws.linsys)`.

The steps are always the same. Pass a matrix type that describes your problem. Then check
which backend you got. If it is the one you expected, the solver used your structure.

### Which type to use

There are four answers. Which one is right depends on your problem, not on taste:

| your problem | use | why |
|---|---|---|
| small enough to fit in cache | **dense** | nothing beats a contiguous array the CPU keeps close. Structure costs indirection and buys nothing at this size. |
| large, and mostly zeros | **sparse** | you pay for the nonzeros, not for `n²`. `SparseMatrixCSC` covers this. |
| you know more than where the zeros are | **a structured type** | block-diagonal, low-rank, Kronecker. The solver then skips work no sparsity pattern shows. It solves a `BlockDiagonal` as `K` small systems, never as one big one. |
| few zeros, a fast product, and too big for cache | **unmaterialized** | past cache, a dense product waits on memory, not on arithmetic. An operator that builds its product from `O(n)` numbers moves almost nothing and can win outright. |

The last row is easy to miss. What decides it is not size and not sparsity, but whether
**applying** the operator costs less than the dense product. If it does, the operator wins, and
it wins by more once the matrix leaves cache — up to 5.3× on a moving-average operator at
`n = 4000`, where the dense `A` alone is 122 MiB. If it does not, no size will save it: an
operator that is cheap while the rest of the problem is not runs 0.65× of dense at `n = 200`,
and only overtakes it at `n = 500`. The measurements are under
[Choosing a representation](@ref).

### Every type the solver takes

The list above gives the categories. This one gives the types: what each looks like, and what
it buys. `•` is a stored entry. A blank is a zero the type knows about.

**Dense `Matrix`.** Every entry stored. Use it when the problem is small, or when it has no
structure to declare.

```math
\begin{pmatrix} • & • & • & • \\ • & • & • & • \\ • & • & • & • \\ • & • & • & • \end{pmatrix}
```

**`Diagonal`** (LinearAlgebra). One entry per row. Give it a `Diagonal` `A` as well and the
reduced matrix is diagonal too. A solve is then `n` divisions, with nothing factored.

```math
\begin{pmatrix} • & & & \\ & • & & \\ & & • & \\ & & & • \end{pmatrix}
```

**`Bidiagonal`**, **`SymTridiagonal`** and **`Tridiagonal`** (LinearAlgebra). Bandwidth 1.
Smoothing, trend filtering and differencing constraints all land here. The backend is an
`ldlt` and costs `O(n)`.

```math
\begin{pmatrix} • & • & & \\ • & • & • & \\ & • & • & • \\ & & • & • \end{pmatrix}
```

**`BandedMatrix`** ([BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl)).
Bandwidth 2 and up. LinearAlgebra has no symmetric type for those. The backend factors it as
a banded Cholesky and costs `O(n b²)`.

```math
\begin{pmatrix} • & • & • & & \\ • & • & • & • & \\ • & • & • & • & • \\ & • & • & • & • \\ & & • & • & • \end{pmatrix}
```

**`SparseMatrixCSC`** (SparseArrays). Entries wherever you put them, stored by column. Use
it when the pattern is irregular and mostly empty.

```math
\begin{pmatrix} • & & • & \\ & • & & \\ • & & & • \\ & & • & • \end{pmatrix}
```

**[`PureQPBase.BlockDiagonal`](@ref).** Independent blocks, stored as the blocks. You get `K`
systems of size `n/K` instead of one of size `n`. That is `n³/K²` work and `1/K` the memory.

```math
\begin{pmatrix} • & • & & & & \\ • & • & & & & \\ & & • & • & & \\ & & • & • & & \\ & & & & • & • \\ & & & & • & • \end{pmatrix}
```

**[`PureQPBase.RowCoupled`](@ref).** A few dense rows on top of rows that hold one entry each.
A bound per variable, plus a budget or a total. The Woodbury identity solves it in `O(nk)`
and never forms the `n×n` matrix.

```math
\begin{pmatrix} • & • & • & • \\ • & • & • & • \\ • & & & \\ & • & & \\ & & • & \\ & & & • \end{pmatrix}
```

**[`PureQPBase.KroneckerOperator`](@ref).** `A₁ ⊗ A₂`, held as its two factors. Use it for a
constraint that acts across two dimensions at once. The `6×6` below costs `4 + 9` numbers to
store.

```math
A_1 \otimes A_2 = \begin{pmatrix} a_{11}A_2 & a_{12}A_2 \\ a_{21}A_2 & a_{22}A_2 \end{pmatrix}
```

**[`PureQPBase.ProductOperator`](@ref) and `LinearMaps.LinearMap`.** No entries at all, for a
map built from functions. The matrix is a *program*: a chain of cheap steps you apply to `x`.
Nothing is assembled. (A map over a matrix, or a `kron` or `blockdiag` of two, arrives as that
matrix instead: [What a composed map becomes](@ref).) The running-sum constraint further down
this page takes three steps:

```math
x \in \mathbb{R}^{n}
\;\xrightarrow{\;\;\odot\, w\;\;}\;
\;\xrightarrow{\;\;\mathrm{cumsum}\;\;}\;
\;\xrightarrow{\;\;[\,1{:}m\,]\;\;}\;
Ax \in \mathbb{R}^{m}
```

As a matrix that is lower triangular and `m×n`. As a program it is `O(n)` work and three
lines. The composition *is* the representation:

```math
A \;=\; \underbrace{S}_{\text{keep } 1{:}m} \; \underbrace{C}_{\text{cumsum}} \; \underbrace{W}_{\mathrm{diag}(w)}
\qquad \text{stored: } w \text{, and nothing else}
```

LinearMaps composes these lazily, so you can build an operator from others: `B*C`, `B + C`,
`B'`, `kron(B, C)`. It forms no product in that expression:

```math
\mathcal{A} \;=\; B\,C \;+\; D^{\top}E
\qquad\Longrightarrow\qquad
\mathcal{A}x \;=\; B(Cx) \;+\; D^{\top}(Ex)
```

The solver gets everything it needs by running that program on a vector.

Views (`SubArray`) and `Symmetric` wrappers work too. They say how you store a matrix, not
what shape it has, so they bring no backend of their own.

| type | reduced matrix | backend |
|---|---|---|
| `Matrix` | dense | `cholesky` |
| `Diagonal` with `Diagonal` | diagonal | `diagonal` |
| tri/bidiagonal with `Diagonal` | bandwidth 1 | `tridiagonal` |
| banded with banded | bandwidth `2 ≤ b ≤ n/4` | `banded` |
| `SparseMatrixCSC` | sparse, or dense once it fills | `cholmod`, `ldlfactorizations`, `sparse_formed` |
| `BlockDiagonal` pair | `K` blocks | `block` |
| `Diagonal` with `RowCoupled` | diagonal plus rank `k` | `lowrank` |
| `μI` with `KroneckerOperator` | diagonal in the factors' eigenbasis | `kronecker` |

`PureQPBase.backend_name(ws.linsys)` reports which one you got. The table covers
[`OperatorSplitting`](@ref) and [`InteriorPoint`](@ref), which solve against the reduced matrix.
[`ActiveSet`](@ref) has no backend and does not form that matrix. The next section says what
each of the three does with each kind of type, an operator included.

### What each algorithm does with each type

Each cell says what the algorithm does with a `P` and `A` of that kind. The `indirect` backend
is conjugate gradients, which needs Krylov.jl.

| | dense | sparse | structured | unmaterialized |
|---|---|---|---|---|
| [`OperatorSplitting`](@ref) | `cholesky` | `cholmod`, `ldlfactorizations` or `sparse_formed`, by the pattern | the structured backend its conditions admit, else `indirect`: a type that holds its structure is never formed | `indirect` with `scaling = 0`; a direct backend when the type unwraps to a structured one ([What a composed map becomes](@ref)) |
| [`InteriorPoint`](@ref) | the full KKT system, `bunchkaufman` | the sparse KKT family, `cholmod` | `diagonal`, `tridiagonal`, `banded` or `block`; a Kronecker or low-rank pair is refused, since the only rung left would form the matrix | only with a caller-supplied preconditioner, `linsys = :indirect` and `scaling = 0` |
| [`ActiveSet`](@ref) | `A R⁻¹` formed once and stored | read into a dense matrix first | `Diagonal`, `BlockDiagonal` and `KroneckerOperator` are held as they are and `A R⁻¹` is not formed: implicit `A R⁻¹`, `R` in `P`'s form. Banded, tridiagonal and `RowCoupled` are read into a dense matrix first | `A`: implicit `A R⁻¹`, each row read as one adjoint product. `P`: refused, since `P` must have a Cholesky factor |

Three things in that table are conditions on your problem, not choices.

**`ActiveSet` needs a factor of `P`, and nothing of `A` but its products.** The method solves
against `M = A R⁻¹`, where `R` is the Cholesky factor of `P`. `A` is multiplied and read one row
at a time, so every kind of `A` works, an operator included, as long as it can multiply by its
transpose. `P` has a Cholesky factor when it is a dense matrix, a `Diagonal`, a `BlockDiagonal`
of dense blocks, or a `KroneckerOperator` of two dense matrices with `eps_prox = 0`. A `P` that
supplies products only has none, so `setup` throws and names the types that do.

When `A` and the factor of `P` are both dense, `M` is formed once and stored. In every other
case the workspace holds `A` and `R`, derives a row of `M` from them when the working set asks
for it, and computes the product `Mu` as one solve and one product with `A`.

**`InteriorPoint` on an operator needs a preconditioner of your own, and a factor of `P` is not
one.** On the problem measured below, with `cholesky(P)` as the preconditioner the reduced
matrix `P + δI + Aᵀ diag(w) A` has condition number `3.5e14`, against `6.6e14` without it. That
matrix is dominated by `Aᵀ diag(w) A`, whose weights `w` span many orders of magnitude, and a
factor of `P` does nothing about that term. A factorization of the whole reduced matrix,
refreshed as the weights change, is a preconditioner that works
([Operators under the interior-point method](@ref)).

**A Kronecker `P` under `ActiveSet` needs `eps_prox = 0`.** `P₁ ⊗ P₂ + εI` is not a Kronecker
product, so it has no Kronecker factor, and a positive `eps_prox` is refused by name. A singular
Kronecker `P` therefore has no `ActiveSet` path in that form. Passing it as a dense `Matrix`
factors it densely.

#### Unmaterialized is smaller and faster

Holding `A` and `R` instead of `M` is not a trade of speed for memory. Both go down. The
problem has `n = 625` variables and `m = 2208` rows, with `P = P₁ ⊗ P₂`, `A = A₁ ⊗ A₂`,
`cond(P) = 8.3e8` and `cond(A) = 2.4e11`. The first column holds both as
[`PureQPBase.KroneckerOperator`](@ref)s, the second as dense matrices:

| | held as factors | dense |
|---|---|---|
| workspace | 9.363 MiB | 36.270 MiB |
| cold solve | 78.9 ms | 194.8 ms |
| pricing all 2208 rows | 6.33 µs | 97.30 µs |

The whole solve is 2.47× faster and pricing is 15.4× faster. The two solves take the same
number of iterations and agree to `3.7e-13` relative. A warm `solve!` on the factors allocates
0 bytes. `kron_problem(11)` in `PureDAQP/test/helpers.jl` builds the problem.

What the workspace stores grows with `m` through vectors of length `m` only. Adding 6624 rows
at the same `n` adds 812,736 B, where a reduction that formed `M` adds 33,120,000 B. The growth
is not zero because 8.970 MiB of the 9.363 MiB is the working set's own factor, which has
`n × (min(m, n) + 1)` entries and so depends on `n`, not on `m`.

### What `P` has to be

How you store a matrix is one question. What it has to be numerically is another. The
solver checks three things before a solve starts, not during one.

**`P` must be symmetric, and you must pass all of it.** `P` is the matrix in
`½xᵀPx`, so only its symmetric part means anything, and [`setup`](@ref) throws if `issymmetric`
fails. **Pass the full matrix or a `Symmetric` wrapper. Never pass a stored triangle.** A
triangle is a different matrix, and it quietly halves every off-diagonal term:

```math
P = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
\quad\text{is not}\quad
\begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
```

**`P + σI` must be positive definite, not `P` itself.** That is what makes the reduced matrix
factorable. Since `σ > 0`, a merely positive *semi*definite `P`
always passes — including `P = 0`, a feasibility problem — so this rejects only genuine
indefiniteness. [`PureQPBase.is_convex`](@ref) is the test, and a type can answer it cheaply: a
`Diagonal` scans its entries, a `SparseMatrixCSC` factors sparsely, an operator reports what
it was told.

**An indefinite `P` is refused at setup.** It makes the problem non-convex, and then a local
answer is not a global one. `setup` throws and names the remedy
(raise `σ` if `P + σI` can be made definite). Without the check the reduced matrix would often
factor anyway and return a stationary point that is not a minimum. The `NON_CONVEX` status is
a different event: residuals diverging *during* a solve.

**Bad conditioning is the one with no yes-or-no answer.** Eliminating `ν` builds
`Ãᵀdiag(ρ)Ã`, which squares `A`'s conditioning, so the reduced matrix carries `κ(A)²`. Two
things keep that usable, and one limit remains:

- **Equilibration** (`scaling = 10` by default) is what makes the reduced form viable at all.
  Without it the Cholesky loses all accuracy by `κ(A) = 1e8`; with it the inner solve holds
  around `1e-8` across the whole range, and past `κ(A) = 1e10` it is *more* accurate than
  factoring the full KKT matrix.
- **`linsys = :kkt`** does not square the conditioning, and is the thing to reach for when a
  result is in question — though on the sweep in [Benchmarks](@ref "Conditioning") it does not
  extend the range over which ADMM converges.
- **Past about `κ = 1e9` the algorithm, not the arithmetic, runs out.** More precision does not
  help: the same problem takes the same iterations in `Float64` and in 256-bit `BigFloat`,
  because the iteration count follows the problem's geometry. What does help is structure — a
  block-diagonal problem converges at `κ = 1e10` where the dense one does not, and a Kronecker
  operator at `κ = 1e12` in 900 iterations, because `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)` puts only the
  square root of the conditioning in each factor it actually solves with.

### An operator is not always solved with CG

An operator with no structure the solver recognizes is solved with *conjugate gradients* (CG),
which only multiplies by the operator and whose convergence depends on conditioning. An
operator with its own **direct** backend is solved by factoring instead, and conditioning then
affects it only through the structure. That is what [`OperatorSplitting`](@ref)'s
`linsys = :auto` does. Under [`InteriorPoint`](@ref) an operator needs `linsys = :indirect`
named explicitly, with a preconditioner you supply ([Choosing an algorithm](@ref "What each algorithm throws on")).

The Kronecker type is an example. `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so an operator with `κ = 1e12` is
built from two factors with `κ = 1e6` each, and the backend eigendecomposes the factors without
forming the product:

| n | κ(A) | iterations | Kronecker | dense | speedup | CG on the same problem |
|---|---|---|---|---|---|---|
| 400 | 1e12 | 625 / 625 | 2.3 ms | 27.9 ms | **12.1×** | `SOLVED` in 625 iterations |
| 1600 | 1e12 | 1100 / 1100 | 21.8 ms | 1566 ms | **71.8×** | `SOLVED` in 850 iterations |

The Kronecker backend and the dense path reach the same objective, and CG solves both
problems too. The Kronecker backend is the fast one because its cost follows the structure,
not the conditioning.

The problems below are all the same QP, written five ways.

```@example storage
using PureOSQP, LinearAlgebra, SparseArrays

n = 6
Pdiag = Diagonal(2.0 .+ (1:n) ./ n)
Aband = Bidiagonal(fill(1.0, n), fill(-1.0, n - 1), :U)
q = collect(range(-1.0, 1.0; length = n))
l = fill(-0.5, n)
u = fill(0.5, n)

Pd, Ad = Matrix(Pdiag), Matrix(Aband)
reps = [
    "Matrix"                      => (Pd, Ad),
    "Diagonal, Bidiagonal"        => (Pdiag, Aband),
    "SymTridiagonal, Tridiagonal" => (SymTridiagonal(diag(Pd), zeros(n - 1)), Tridiagonal(Ad)),
    "Symmetric, SubArray"         => (Symmetric(Pd), view(Ad, :, :)),
    "SparseMatrixCSC"             => (sparse(Pd), sparse(Ad)),
]

P_before, A_before = copy(Pd), copy(Ad)
reference = setup(Pd, q, Ad, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
ref = solve!(reference)
ref_backend = PureOSQP.backend_name(reference.linsys)
for (name, (Pr, Ar)) in reps
    ws = setup(Pr, q, Ar, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
    @assert ws.prob.P === Pr && ws.prob.A === Ar          # held by reference, not copied
    sol = solve!(ws)
    backend = PureOSQP.backend_name(ws.linsys)
    @assert sol.iter == ref.iter                       # same trajectory
    @assert isapprox(sol.x, ref.x; rtol = 1e-8)        # same answer
    # Bit-exact only where the same factorization ran; see below.
    backend == ref_backend && @assert sol.x == ref.x && sol.y == ref.y
    println(rpad(name, 30), "iter = ", sol.iter, ",  backend = ", backend)
end
@assert Pd == P_before && Ad == A_before        # the caller's arrays are never written to
```

Two things happen here:

**Change only how the entries are reached and you get identical answers, bit for bit.**
`Symmetric`, a `SubArray` and a `SparseMatrixCSC` all feed the same numbers into the same
arithmetic, so `==` holds against the dense reference.

**Change which backend runs and you change the arithmetic.** A `Diagonal` `P` with a
`Bidiagonal` `A` makes the reduced matrix tridiagonal. An `ldlt` on two bands solves it, in
place of a dense inverse and a `symv`. That is a different factorization, so the answers
agree to about `1e-16` rather than bit for bit. The iteration count and the answer match. The
last digits do not. See
[Which backend a structured matrix gets](@ref "Which backend a structured matrix gets").

`P` has to be symmetric *as you store it*. The solver refuses a lower triangle with zeros
above the diagonal rather than mirroring it, because that matrix is a different problem and
is not symmetric. Wrap it in `Symmetric` to say which triangle is the real one.

### Which backend a sparse matrix gets

Eliminate the dual variable from either backend-using algorithm's system and you get an `n×n` reduced
matrix. Whether that matrix is worth keeping sparse depends on its pattern, not on how dense
your input was. `linsys = :auto` reads four things from the pattern: the densest row of `A`,
the size of the symbolic `AᵀA ∪ P` pattern, `n`, and `m`. It factors nothing to decide, so
the factorization `setup` pays for is the one the solve uses. Where that decision sits
among the others is drawn in [Choosing a backend](@ref); the same selection serves
[`InteriorPoint`](@ref) at its own row weights.

```@example storage
band = 200
banded = setup(
    sparse(SymTridiagonal(fill(2.0, band), fill(0.3, band - 1))),
    collect(range(-1.0, 1.0; length = band)),
    sparse(Bidiagonal(fill(1.0, band), fill(-1.0, band - 1), :U)),
    fill(-1.0, band), fill(1.0, band),
)
PureOSQP.backend_name(banded.linsys)
```

A banded `A` gives a banded reduced matrix, so the solver builds and factors it sparsely. A
scattered pattern fills in, and then a sparse factor costs as much as the dense inverse. The
solver still builds that reduced matrix from the stored entries, but factors it densely.

```@example storage
using Random
Random.seed!(1)
ns, ms = 80, 160
scattered = setup(
    sparse(1.0I, ns, ns), collect(range(-1.0, 1.0; length = ns)),
    sprandn(ms, ns, 0.05), fill(-1.0, ms), fill(1.0, ms),
)
PureOSQP.backend_name(scattered.linsys)
```

`PureQPBase.backend_name(ws.linsys)` names whichever backend the workspace ended up with.
The dense default is `:cholesky`. The full KKT factorization is `:bunchkaufman`.

#### Measuring the choice on your own problem

That rule is fitted to a benchmark suite. It is right about a class of problems, not about
yours in particular. [`recommend_linsys`](@ref) measures instead of guessing. It builds every
backend your pair allows, times `setup` and a few iterations on each, and ranks them by what
a whole solve costs: setup once, plus the per-iteration cost over the iterations one full run
takes.

```julia
julia> advice = recommend_linsys(P, q, A, l, u, InteriorPoint())
LinsysAdvice: linsys = :sparse, over a solve of 10 iterations
  linsys       backend               total ms  setup ms  solve ms   ms/iter      fill   iter  status
  :auto        ldlfactorizations        1.603     0.495     1.108    0.1108    0.0015     10  solved
  :sparse      ldlfactorizations        2.236     0.646      1.59     0.159    0.0015     10  solved
  :dense       cholesky               128.206     1.881   126.325   12.6325   0.50062     10  solved
  :kkt         bunchkaufman           792.617    17.187    775.43    77.543   4.44263     10  solved

julia> ws = setup(P, q, A, l, u, InteriorPoint(); linsys = advice.linsys);
```

It is a tool for you, not a step inside `setup`. Nothing on the solve path calls it. It costs
a short solve per candidate, plus one full solve to get the iteration count. Run it once for
a problem shape you solve often, then pin the `linsys` it names.

### Which backend a structured matrix gets

A structured `P` and `A` do more than read cheaply. They can make the reduced matrix itself
narrow, and then there is far less to factor. Eliminating `ν` gives

```math
R = c D P D + \sigma I + \tilde A^\top \mathrm{diag}(\rho) \tilde A
```

Diagonal scaling preserves a bandwidth and `ÃᵀρÃ` doubles `A`'s, so
`bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`. `linsys = :auto` dispatches on the pair of
types. No setting and no density gate comes into it. These pairs are the first candidates in
[Choosing a backend](@ref).

```@example structured
using PureOSQP, LinearAlgebra
n = 200
q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, n), fill(1.0, n)

# A separable objective under box constraints: R is diagonal, so nothing is factored.
box = setup(Diagonal(fill(2.0, n)), q, Diagonal(ones(n)), l, u)

# A tridiagonal objective under box constraints: R stays tridiagonal.
smooth = setup(SymTridiagonal(fill(2.0, n), fill(0.3, n - 1)), q, Diagonal(ones(n)), l, u)

(PureOSQP.backend_name(box.linsys), PureOSQP.backend_name(smooth.linsys))
```

The first has nothing to factor at all — a solve is `n` divisions — and the second is an
`ldlt` that costs `O(n)`. Against the dense path the same problems would otherwise take, that
is worth a great deal at any size; see
[Structured backends](@ref "Structured backends") for the measurements.

Widening `A` widens `R` faster than widening `P` does, the practical consequence of the rule
above. A `Tridiagonal` `A` squares to bandwidth 2, past what `SymTridiagonal` stores, and is
served by a banded Cholesky once BandedMatrices.jl is loaded:

```@example structured
using BandedMatrices
diff = setup(
    SymTridiagonal(fill(4.0, n), fill(0.3, n - 1)), q,
    Tridiagonal(fill(-0.25, n - 1), ones(n), fill(-0.25, n - 1)), l, u,
)
(PureOSQP.backend_name(diff.linsys), diff.linsys.bw)
```

Without BandedMatrices loaded that problem takes the dense path instead — correctly, just not
cheaply. Structure in `P` alone never survives: `ÃᵀρÃ` is dense for a general `A` whatever
`P` looked like, so a `Diagonal` `P` with a dense `A` is a dense reduced matrix and gets the
dense backend.

```@example structured
using Random
Random.seed!(2)
PureOSQP.backend_name(setup(Diagonal(fill(2.0, 40)), q[1:40], randn(60, 40),
                            fill(-1.0, 60), fill(1.0, 60)).linsys)
```

### Supplying a matrix type of your own

`P` and `A` are held by reference and reached through a small set of functions, so a
representation the package has never heard of works by declaring itself
`<: AbstractMatrix{T}` and supplying `size`, `mul!`, and `mul!` against its adjoint. That much
is enough to solve. Everything below is optional; each override replaces one generic walk over
entries with whatever the representation can answer more cheaply.

There are two seam levels, and which one a representation wants depends on whether it can
enumerate a column.

**Per column.** [`PureQPBase.structural_rows`](@ref)`(M, j)` names the rows column `j` can hold
a nonzero in; the four traversals in `PureQPBase/src/scaling.jl` — `weighted_colmax`,
`weighted_colmax_rowmax!`, `scaled_col!` and `add_scaled_col!` — follow it, so a single
`structural_rows` method makes equilibration and the dense formation cost the column's own
entries rather than all `m` of them. A representation whose columns are cheaper to walk than
to index overrides the four traversals directly instead; the sparse extension does that,
because `M[i, j]` on a `SparseMatrixCSC` is a binary search.

**Per sweep.** `column_norms!` and `cost_norms!` are the whole of what equilibration asks
per sweep, so a representation that answers in whole-matrix or closed form overrides those
two and never sees a column index. The GPU extension is the shipped example: it replaces
both with array reductions instead of indexing entries one at a time.

Beyond equilibration there are three more override points, all optional:
`PureQPBase.reduced_diagonal!` for the matrix-free preconditioner,
[`PureQPBase.is_convex`](@ref) for the convexity test `setup` runs before choosing a backend,
and [`PureQPBase.is_symmetric`](@ref) for the symmetry check — the last two both densify or
scan `n²` positions otherwise.

[`PureQPBase.RowCoupled`](@ref) is the worked example in the package itself: a few dense rows
above a block holding one entry per row. It defines `size`, `getindex` and `mul!`, and adds
one `structural_rows` method; that is all it takes for a `Diagonal` `P` with a `RowCoupled`
`A` to reach the low-rank backend and to equilibrate at the cost of its own entries.

An operator that supplies **only** products — nothing to index at all — says so with
[`PureQPBase.is_materializable`](@ref):

```julia
PureQPBase.is_materializable(::MyOperator) = false
```

`linsys = :auto` then skips the dense terminal and lands on the matrix-free backend, which
needs Krylov.jl loaded. `polish!` and the two derivative entry points build a dense matrix
out of `P` and `A` entry by entry, so they throw, naming the operator, rather than failing
inside a factorization: pass `polishing = false`, and differentiate a materialized form of
the problem. Equilibration also walks columns, so an operator that overrides neither seam
level needs `scaling = 0`.

The hot-path guarantees carry a condition here that they do not carry elsewhere. `admm_step!`
allocates nothing and is type-stable for a caller-supplied operator only as far as that
operator's own `mul!` is: a broadcast in it, or a `DimensionMismatch` message built from a
type, is enough to lose both. `PureOSQP/bench/lazy_operator.jl` is written to hold them, and
`bench/strictmode_audit.jl` checks it.

### An operator from LinearMaps.jl

**Use this when your constraint is something you can *do* but would never want to *store*.**

The situation is common in signal and image work. "Take a running total." "Blur this." "Take a
Fourier transform, keep the low frequencies." Each is a perfectly good linear constraint, and
each has a matrix — but for a million-pixel image that matrix has `10¹²` entries and cannot
exist. What you have instead is a function that applies it.

[LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) is the standard Julia
package for exactly that: an object you can multiply by, built from a function. Load it and
this solver accepts one anywhere it accepts a matrix. Nothing else is needed — matrices and
maps can even be mixed in the same call.

The rest of this section covers what a map needs to work. For complete problems solved this
way — a measurement operator, a 2-D grid, and a model reused as-is — see
[Operators from functions](@ref).

#### When this is the right tool

Four situations, in rough order of how often they come up:

1. **The matrix will not fit.** Deblurring a 1000×1000 image is a million variables, so `A` is
   a million by a million: `8` terabytes dense. There is no trade to weigh — an operator
   is the only way the problem exists at all. Same story for 3-D grids, large PDE-constrained
   problems, and anything where `n` runs past `10⁵`.
2. **Applying it is much cheaper than its size suggests.** A convolution or blur is a *dense*
   matrix — every output touches every input — but applying it through an FFT costs
   `O(n log n)` instead of `O(n²)`. Storing it throws that away. The same holds for any
   transform with a fast algorithm: DCT, wavelets, a fast multipole method.
3. **You already have the code, not the entries.** The operator is a simulator, an existing
   forward model, a PDE solve, a linearization somebody else wrote. You can call it; nobody
   ever assembled it, and assembling it would mean `n` separate calls.
4. **Memory is the binding constraint, not time.** The matrix-free path stores vectors where
   the direct path stores an `n×n` inverse — [32× less at `n = 4000`](@ref "The matrix-free
   backend"). If the problem does not fit in RAM, being slower is not the issue.

#### When it is the wrong tool

**For a small problem whose operator is no cheaper than its matrix, use the matrix.** Being
matrix-free replaces a factorization done once with a CG solve in every iteration. On the
operator below, built from `O(n)` stored numbers but paired with a dense `A`, that costs more
than it saves at `n = 200` and less from `n = 500`:

| n | iterations (operator / matrix) | operator | matrix | speedup |
|---|---|---|---|---|
| 200 | 225 / 125 | 3.4 ms | 2.2 ms | **0.65×** |
| 500 | 150 / 125 | 18.6 ms | 20.0 ms | 1.08× |
| 1000 | 175 / 175 | 85.2 ms | 128.8 ms | 1.51× |

In the table in [Which type to use](@ref), where applying the operator is also
cheaper than its dense product, the operator is faster at every size, by 1.23× to 5.26×. Size
matters in both cases: the factorization's cost grows as `n³`, so the matrix-free route gains as
problems grow, and it gains faster when applying the operator is cheap.

The second way to get this wrong is conditioning. A bare `LinearMap` has no structure the
solver can exploit, so it is served by conjugate gradients, which struggles as conditioning
worsens — and this is not a small effect: on the badly conditioned sweep in
[Benchmarks](@ref "Conditioning") the matrix-free backend fails to converge at *every* κ tested,
including mild ones. If your problem is ill-conditioned, a bare map is the wrong shape; give
the solver a structured type with a direct backend instead
([An operator is not always solved with CG](@ref)).

Building one takes two functions: how to apply it, and how to apply its transpose. The
transpose is not optional; the solver needs both directions.

```@example linearmaps
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(4)

n, m = 60, 40
# The constraint: scale each entry by w, take a running total, keep the first m.
# `forward` applies it; `adjoint_` applies its transpose. No m×n array is ever built.
w = 0.5 .+ rand(n)
forward(y, x) = (y .= cumsum(w .* x)[1:m])
function adjoint_(x, y)
    fill!(x, 0.0)
    x[1:m] .= y
    reverse!(x); cumsum!(x, x); reverse!(x)
    x .*= w
    return x
end
A = LinearMap{Float64}((y, x) -> forward(y, x), (x, y) -> adjoint_(x, y), m, n)

# A map built from a function reports only the traits it is told -- see below.
P = LinearMap{Float64}((y, x) -> (y .= 2 .* x), n, n; issymmetric = true, isposdef = true)
q = randn(n)
l, u = fill(-1.0, m), fill(1.0, m)

sol = PureOSQP.solve(P, q, A, l, u; scaling = 0, linsys = :indirect)
(sol.status, sol.iter, round(sol.obj_val; digits = 6))
```

That call needed three things beyond the map itself. Skip any one of them and the solve fails.
Here is each one, and what goes wrong without it.

**1. `using Krylov`.** An operator has no entries, so none of the usual backends can factor
anything. The only one that works is the matrix-free one, which multiplies instead of
factoring — and it lives in Krylov.jl. Without it loaded you get an error naming the remedy.
You do not have to pass `linsys = :indirect`; the solver finds it on its own. It is written
above only to make the requirement visible.

**2. `scaling = 0`.** By default the solver rescales your problem for numerical health, which
means reading down each column of `A` to find its largest entry. A map has no columns to read.
Passing `scaling = 0` turns that step off. If you forget, `setup` throws and says so — it does
not silently skip the rescaling.

**3. Declaring `issymmetric` and `isposdef` on a `P` built from a function.** This one catches
most people. A map built from a function has no entries, and LinearMaps reports only what you
*told* it: ask an undeclared one `isposdef` and it answers `false`. The solver cannot factor an
operator to find out, so it sees an objective that does not claim to be symmetric and convex,
and `setup` throws. Declare both when you build the map, as in the example. You can also build
the wrapper yourself with `ProductOperator{T}(map; symmetric, posdef)` to override what a map
claims.

That is also what LinearMaps gives you over an operator you write by hand. The two declarations
travel with the map, so [`PureQPBase.is_convex`](@ref) reads a flag instead of factoring a
matrix.

A map over a matrix needs neither declaration. `LinearMap(Diagonal(fill(2.0, n)))` reaches the
solver as that `Diagonal`, not as a `ProductOperator`, so it is checked from its entries and
the backends that dispatch on `Diagonal` apply to it. [What a composed map becomes](@ref) lists
which maps arrive as matrices.

**Expect one more thing: a map runs without a preconditioner.** A preconditioner is a cheap
approximation of the problem that makes the iteration converge faster. The one used here comes
from the diagonal of the reduced matrix. A map has no entries, so there is no diagonal to read,
and the solver runs without one. Setup gets *cheaper*, because there is nothing to build, and
each iteration gets **1.33–1.44× more expensive**. We measured that on the same operator written
both ways ([Benchmarks](@ref "An operator that is never materialized")).

Usually you accept that. If the iteration count matters, give your map's type a
`PureQPBase.structural_rows` method. It is one method, described under
[Structured operators](@ref "2. `structural_rows` — setup stops paying for the zeros"). It gets
the preconditioner back *and* lets you drop `scaling = 0`. `probe = true` does not replace it.
Probing answers the rescaling question, not this one.

#### Building `A` by composition

Use LinearMaps rather than a hand-written operator because maps *combine*. A constraint matrix
is usually several blocks stacked together. Sums, products, `kron`, `vcat` and `hcat` all work,
and they mix freely. No product is ever formed:

| you write | you get |
|---|---|
| `B + C` | the sum, applied as `Bx + Cx` |
| `B * C` | the composition, applied as `B(Cx)` |
| `kron(B, C)` | the Kronecker product, applied through the factors |
| `[B; C]` | `vcat` — stack constraint blocks, the common case for `A` |
| `[B C]` | `hcat` — one block per group of variables |
| `[B C; D E]` | `hvcat` — both at once |
| `B'` | the adjoint |

When the pieces are matrices, `kron` and `blockdiag` arrive as structured types, not as a
program. [What a composed map becomes](@ref) says which.

Here `A` is a scaled running sum stacked over a box, built from three maps and never
assembled:

```@example compose
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(11)

n = 6
w = 0.5 .+ rand(n)

# Both directions. The adjoint of a running sum is a reversed running sum.
C = LinearMap{Float64}(
    (y, x) -> (y .= cumsum(x)),
    (x, y) -> (x .= reverse(cumsum(reverse(y)))),
    n, n,
)
W = LinearMap(Diagonal(w))          # scaling; a matrix-backed map knows its own adjoint
B = LinearMap(Matrix(1.0I, n, n))   # a plain box block

A = [C * W; B]                      # vcat of a product over an identity block

P = LinearMap(Diagonal(fill(2.0, n)))   # arrives as the Diagonal itself
q = randn(n)
l = vcat(fill(-3.0, n), fill(-0.4, n))
u = vcat(fill(3.0, n), fill(0.4, n))

sol = PureOSQP.solve(P, q, A, l, u; scaling = 0, eps_abs = 1e-10, eps_rel = 1e-10)

# The same problem with every block written out.
Ad = [Matrix(C) * Matrix(W); Matrix(B)]
dense = PureOSQP.solve(Diagonal(fill(2.0, n)), q, Ad, l, u; eps_abs = 1e-10, eps_rel = 1e-10)
(size(A), sol.status, round.(sol.x; digits = 5), maximum(abs, sol.x .- dense.x))
```

The last value is the largest difference between the two answers.

**Every function-based map you combine needs its adjoint.** The solver applies `Aᵀ` once per
iteration. A `LinearMap` built from a forward function alone fails with
`transpose not implemented`, and it fails partway into the first iteration — not when you build
it, and not on the first product. A map built from a matrix supplies its own adjoint. A map
built from functions does not, which is why `C` above gives both directions.

#### What a composed map becomes

A `LinearMap` reaches the solver as one of this package's own types when it can, and wrapped in
a [`PureQPBase.ProductOperator`](@ref) when it cannot. The difference matters: an unwrapped map
is a matrix the algorithms read through its structure, and a wrapped one has no entries and is
reached only by products. These are the maps that arrive unwrapped:

| you write | the solver receives |
|---|---|
| `LinearMap(B)`, `B` a matrix | `B` itself, so a `Diagonal` stays a `Diagonal` |
| `kron(LinearMap(B₁), LinearMap(B₂))`, exactly two maps | a [`PureQPBase.KroneckerOperator`](@ref) of `B₁` and `B₂` |
| `blockdiag(LinearMap(B₁), …)` | a [`PureQPBase.BlockDiagonal`](@ref) of the `Bᵢ` |
| `c * M`, `c` real, for any `M` above | what `M` becomes, with `c` multiplied into the matrix (a scaled copy), into the first Kronecker factor, or into each block |

Everything else stays a `ProductOperator`: a map built from functions, a sum, a general
product, a `kron` of three maps, a `vcat` or `hcat`. A `KroneckerOperator` can only be built from
dense `Matrix` factors: a `Diagonal` or `Symmetric` factor throws when the map is converted, so
wrap `Matrix` factors.

**`A` composes freely.** Every `LinearMap` composition works as `A`, because `ActiveSet` only
multiplies by `A` and reads it one row at a time, and the other two algorithms multiply by it.
A composition that arrives unwrapped is read through its structure. A wrapped one costs one
product with its adjoint for each row, which is `m` of them at setup and one more for every row
that enters the working set.

**`P` is where `ActiveSet` branches**, because it needs the Cholesky factor of `P`. What a
`LinearMap` `P` can be, and what `ActiveSet` does with it:

| you write | `P` arrives as | `ActiveSet` |
|---|---|---|
| `LinearMap(B)` | `B` | factors it. A sparse or banded `B` is read into a dense matrix first |
| `kron(LinearMap(P₁), LinearMap(P₂))` | `KroneckerOperator` | factors the two factors, never the product, and needs `eps_prox = 0` |
| `blockdiag(LinearMap(P₁), …)` | `BlockDiagonal` | factors each block, when every block is a dense matrix. A `Diagonal` or sparse block is refused |
| `c * P′` | `P′` with `c` folded in | factors it when `P′` is factorable and `c > 0` |
| `B' * B` | `ProductOperator` | refused. Pass `B'B` as a matrix |
| `P₁ + P₂` | `ProductOperator` | refused: a factor of the sum is not a function of the two summands' factors. For `P + εI`, pass `ε` as `eps_prox` |
| `B * C` in general | `ProductOperator` | refused |
| a map built from functions | `ProductOperator` | refused by name |
| `vcat`, `hcat` | `ProductOperator` | not a square matrix, so not a `P` |

A refused `P` is refused by `setup`, and the message names the types that would work. None of
this limits [`OperatorSplitting`](@ref), which takes any `P` that supplies products.

## Structured operators the package ships

`Diagonal` and `Bidiagonal` above are LinearAlgebra's. This package ships three more matrix
types of its own, for three shapes that come up constantly and that LinearAlgebra has no type
for.

**Start here: which one, if any, is yours?**

| if your problem is… | use | typical source |
|---|---|---|
| many small independent sub-problems, side by side | [`PureQPBase.BlockDiagonal`](@ref) | one QP per time step, per asset, per scenario — anything that would be separate problems if they did not share a solve |
| mostly independent, but with a *few* rows tying everything together | [`PureQPBase.RowCoupled`](@ref) | box constraints plus a handful of budget or total-mass rows |
| a constraint applied across two dimensions at once | [`PureQPBase.KroneckerOperator`](@ref) | a 2-D grid, an image, space × time — where the constraint is "this in one direction, that in the other" |
| none of these | nothing to do | pass ordinary matrices; the solver is still fast |

If no row fits, you lose nothing. These are speed-ups, not requirements, and the dense path
gives the same answers.

You use all three the same way. Build it, pass it to [`setup`](@ref) where you would pass a
matrix, then check `backend_name` to confirm the solver picked it up. You configure nothing.

Each one gets its own backend because of what it does to the reduced matrix
`R = cDPD + σI + ÃᵀρÃ`, the matrix every backend must solve with:

::: details Code that draws the figure

```@example structured_figure
using CairoMakie, LinearAlgebra

# Draw the nonzero pattern of `M` as unit cells with its top-left corner at `(x0, y0)`,
# rows running downward.
function cells!(ax, M, x0, y0; color)
    nr, nc = size(M)
    rects = [Rect2f(x0 + j - 1, y0 - i, 1, 1) for i in 1:nr, j in 1:nc if !iszero(M[i, j])]
    isempty(rects) || poly!(ax, rects; color)
    lines!(ax, Rect2f(x0, y0 - nr, nc, nr); color = :black, linewidth = 1)
    return nothing
end
blue, orange, purple = "#0072B2", "#E69F00", "#CC79A7"

fig = Figure(size = (980, 560))
ax = Axis(fig[1, 1]; aspect = DataAspect())
hidedecorations!(ax); hidespines!(ax)
title!(x, s) = text!(ax, x, 3.2; text = s, align = (:center, :bottom), fontsize = 15, font = :bold)
label!(x, y, s) = text!(ax, x, y + 0.3; text = s, align = (:center, :bottom), fontsize = 12)
caption!(x, y, s) = text!(ax, x, y; text = s, align = (:center, :top), fontsize = 11, color = :gray30)
ytop, ybot, ycap = 0, -17, -30.5

# Block-diagonal: P and A share the partition, so R is K small dense blocks.
title!(6, "BlockDiagonal")
Abd = kron(Matrix(1.0I, 4, 4), ones(3, 3))
cells!(ax, Abd, 0, ytop; color = blue)
label!(6, ytop, "A: four 3×3 blocks")
cells!(ax, Abd, 0, ybot; color = purple)
label!(6, ybot, "R: four 3×3 blocks")
caption!(6, ycap, "each block is factored alone;\nR is never assembled whole")

# Kronecker: R = μ′I + ρ (A₁ᵀA₁ ⊗ A₂ᵀA₂), diagonalized by the eigenvectors of the two Grams.
x2 = 17
title!(x2 + 6, "KroneckerOperator")
A1 = [1 1 0; 1 1 1; 0 1 1]
A2 = [1 1 0 0; 1 1 1 0; 0 1 1 1; 0 0 1 1]
cells!(ax, kron(A1, A2), x2, ytop; color = blue)
label!(x2 + 6, ytop, "A = A₁ ⊗ A₂, 3×3 and 4×4")
cells!(ax, A1' * A1, x2 + 1, ybot; color = purple)
text!(ax, x2 + 5, ybot - 2; text = "⊗", align = (:center, :center), fontsize = 18)
cells!(ax, A2' * A2, x2 + 6, ybot; color = purple)
label!(x2 + 6, ybot, "R = μ′I + ρ (A₁ᵀA₁ ⊗ A₂ᵀA₂)")
caption!(x2 + 6, ycap, "eigendecompose the 3×3 and 4×4 factors;\nR is never formed")

# Row-coupled: R = C + VᵀWV with C diagonal and V the k×n coupling rows. Woodbury solves
# through V, so the diagonal and the k×n block are all that is stored.
x3, n, k = 34, 12, 2
title!(x3 + 12, "RowCoupled")
cells!(ax, [ones(k, n); Matrix(1.0I, n, n)], x3, ytop; color = blue)
label!(x3 + 6, ytop, "A: 2 coupling rows, then a bound per variable")
cells!(ax, Matrix(1.0I, n, n), x3, ybot; color = purple)
text!(ax, x3 + 13, ybot - 6; text = "+", align = (:center, :center), fontsize = 18)
cells!(ax, ones(n, k), x3 + 14, ybot; color = orange)
text!(ax, x3 + 17, ybot - 6; text = "W", align = (:center, :center), fontsize = 13)
cells!(ax, ones(k, n), x3 + 18, ybot; color = orange)
label!(x3 + 12, ybot, "R = C + VᵀWV: diagonal C, 2×12 V, 2×2 W")
caption!(x3 + 12, ycap, "stored: the diagonal of C and the block V;\nWoodbury solves through V, and R is never formed")
limits!(ax, -1, 65, -35, 6)
nothing # hide
```

:::

```@example structured_figure
fig # hide
```

A block-diagonal pair keeps `R` block-diagonal, so the solver factors one block at a time. A
Kronecker `A` with `P = μI` makes `R` a Kronecker product of two small Gram matrices, and the
eigenvectors of those factors diagonalize it. A row-coupled `A` makes `R` a diagonal plus a
rank-`k` correction, and Woodbury's identity solves that through the `k×n` coupling block alone.
None of the three ever forms `R`. A dense matrix carrying the same numbers would force it.

### Block-diagonal

**Use it when your problem is really several smaller problems side by side.** Four machines
scheduled independently, twelve months priced independently, a hundred scenarios — anything
where variable 3 never appears in a constraint with variable 40.

The saving is large, and it is why this type exists. Solving one `n×n` system costs about `n³`.
Solving `K` systems of size `n/K` costs `K(n/K)³ = n³/K²`. At `K = 10` that is a hundred times
less work and a tenth of the memory. The solver takes that saving as soon as it can *see* the
blocks, and [`PureQPBase.BlockDiagonal`](@ref) is how it sees them. Give it the same numbers as
one big dense matrix and it sees nothing, so it pays the `n³`.

Store it as a vector of the blocks. `P` and `A` must split at the same places. A block of the
problem is independent only if both halves agree it is.

```@example blocks
using PureOSQP, LinearAlgebra

blocks_P = [Matrix(Symmetric([2.0 0.3; 0.3 2.0])) for _ in 1:4]
blocks_A = [[1.0 -1.0; 0.5 1.0] for _ in 1:4]
P = PureOSQP.BlockDiagonal(blocks_P)
A = PureOSQP.BlockDiagonal(blocks_A)

n, m = size(P, 1), size(A, 1)
q = collect(range(-1.0, 1.0; length = n))
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

`backend_name` returned `:block`, so the solver found the blocks. Here is the saving, counted in
numbers stored:

```@example blocks
(blocks = PureOSQP.backend_info(ws.linsys).factor_nnz, dense = n * (n + 1) ÷ 2)
```

Four blocks of two variables each store 12 numbers. The dense route stores 36. The gap widens
fast: at 20 blocks of 12 it is 1 560 against 28 920. Timings are in
[Benchmarks](@ref "Block-diagonal structure").

**If `backend_name` comes back `:cholesky`**, the solver did not use the blocks. Usually `P` and
`A` split at different places, so the problem does not decouple. The answer is still correct.
The solver just took the slow route to it.

### Kronecker

**Use it when a constraint acts on two dimensions at once.** The clearest case is a 2-D grid:
you want something smoothed along rows *and* along columns. Written out, that constraint matrix
is huge and almost all zeros. Written as `A₁ ⊗ A₂` — "`A₁` across, `A₂` down" — it is two small
matrices.

The saving in storage is immediate. The `6×6` constraint below is stored as `4 + 9 = 13` numbers
instead of 36, and that ratio grows as the square. The saving in time is larger, because the
backend never builds the big matrix.

**The conditions are strict.** This is the fussiest type in the package, so check them before
you use it. All three describe your problem. None of them is a setting you can turn on:

| condition | in plain terms | how to check |
|---|---|---|
| `P` must be `μI` — a single number times the identity | your objective weights every variable equally, or there is no objective at all | `P isa Diagonal && allequal(P.diag)` |
| `ρ` must be one number | every constraint is an inequality — no equalities | you passed no row with `l[i] == u[i]` |
| `scaling = 0` | you turn equilibration off explicitly | pass `scaling = 0` to `setup` |

The second one catches people: **one equality row turns this backend off.** And a Kronecker *`P`*
does not satisfy the first. `P` must be a multiple of the identity.

This backend is [`OperatorSplitting`](@ref) only. [`InteriorPoint`](@ref) throws for
`linsys = :kronecker`, because its row weights are not one number
([Choosing an algorithm](@ref "What each algorithm throws on")).

If a condition fails, the solver uses the dense route instead and says nothing. You get the
right answer either way. That is why every example here checks `backend_name`. It is the only
way to tell whether you got what you asked for.

```@example kron
using PureOSQP, LinearAlgebra

A1 = [1.0 0.5; -0.5 1.0]
A2 = [2.0 0.0 1.0; 0.0 1.5 0.0; 1.0 0.0 2.0]
A = PureOSQP.KroneckerOperator(A1, A2)      # 6×6, stored as 4 + 9 entries

n = size(A, 2)
P = Diagonal(fill(2.0, n))                  # μI, as the backend requires
q = collect(range(-1.0, 1.0; length = n))
ws = setup(P, q, A, fill(-1.0, n), fill(1.0, n); scaling = 0)
PureOSQP.backend_name(ws.linsys)
```

Break one condition and the solver drops the Kronecker backend. It solves the problem densely
instead: slower, and just as correct.

```@example kron
equilibrated = setup(P, q, A, fill(-1.0, n), fill(1.0, n))   # scaling left at its default
nonscalar = setup(Diagonal(1.0:n), q, A, fill(-1.0, n), fill(1.0, n); scaling = 0)
(equilibrated = PureOSQP.backend_name(equilibrated.linsys),
 nonscalar = PureOSQP.backend_name(nonscalar.linsys))
```

#### Ill-conditioned Kronecker problems

You pay for this backend by giving up equilibration, and equilibration is worth most on an
ill-conditioned problem. So that is where you judge the trade. Note that
`κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so each factor carries the square root of the figure below.

**The backend stays sound.** We ran it against a dense path given the same `scaling = 0`, so the
comparison is between the backends and nothing else. It matches iteration for iteration and
agrees on the solution up to `κ(A) ≈ 10¹⁶`
(`PureOSQP/bench/results/kronecker_conditioning.json`):

| κ(A) | kronecker | dense, also unscaled | solutions agree |
|---|---|---|---|
| 1e2 | SOLVED, 175 | SOLVED, 175 | yes |
| 1e8 | SOLVED, 450 | SOLVED, 450 | yes |
| 1e12 | SOLVED, 1100 | SOLVED, 1100 | yes |
| 1e16 | SOLVED, 2525 | SOLVED, 2525 | yes |

Iterations climb steeply with conditioning, from 175 to 2525, because nothing preconditions the
problem. That is the cost, and the structure does not hide it.

**Whether it still wins depends on size.** The backend costs `O(n₁n₂(n₁+n₂))` per iteration
against a dense `O(n₁²n₂²)`, and that saving has to cover the extra iterations. Here it is
against a dense path allowed its equilibration — the real choice a caller faces — at
`κ(A) = 10¹²`:

| n | kronecker (`scaling = 0`) | dense, equilibrated | speedup |
|---|---|---|---|
| 30 | 175 iter, 0.11 ms | 300 iter, 0.19 ms | 1.7× |
| 168 | 350 iter, 0.61 ms | 575 iter, 3.27 ms | 5.4× |
| 480 | 875 iter, 3.98 ms | 500 iter, 27.25 ms | 6.9× |

At `n = 480` the backend takes 1.75× the iterations and still finishes 6.9× sooner. The
iteration counts are noisy in both directions, because ADMM's path is sensitive to scaling. Read
the times, not the ratio of counts.

If your `P` is zero rather than `μI`, equilibration and the structure could work together. A
Kronecker product's row and column ∞-norms are the Kronecker products of the factors' norms, so
equilibrating each factor would keep the diagonalization. We have not built that route.

### Low-rank coupling

**Use it when almost every constraint touches one variable, and only a few touch many.** This
shape is very common and easy to miss. A portfolio with a bound on each holding plus one row
saying "the weights sum to 1". A schedule with a limit per machine plus two rows for total
capacity. A design with a box on each parameter plus a budget.

Write that as an ordinary matrix and the few dense rows make the whole matrix look dense. The
solver then pays as if every constraint coupled everything. [`PureQPBase.RowCoupled`](@ref)
keeps the two kinds apart, so you pay only for the coupling rows you have.

It takes three arguments, in this order:

1. `coupling` — the few dense rows, as a `k×n` matrix. These are the rows that touch many
   variables.
2. `weights` — one number per single-entry row.
3. `cols` — which variable each of those rows refers to.

So `RowCoupled(C, ones(n), 1:n)` means "these `k` dense rows, then a plain bound on each of the
`n` variables".

```@example rowcoupled
using PureOSQP, LinearAlgebra

n = 24
coupling = reshape(collect(range(0.1, 0.8; length = 2n)), 2, n)   # two dense rows
A = PureOSQP.RowCoupled(coupling, ones(n), collect(1:n))          # then a bound per variable
P = Diagonal(fill(1.5, n))
q = collect(range(-1.0, 1.0; length = n))
m = size(A, 1)
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

`:lowrank` means it worked. The cost is `O(nk)` instead of `O(n²)`, so the fewer coupling rows
you have, the more it wins. At one coupling row in 2000 variables it is
[**923× faster**](@ref "Low-rank structure") than the dense route.

**One condition:** the coupling rows must be a small fraction of the variables. The solver drops
this backend once `10k > n`. Two coupling rows therefore need at least 20 variables, which is
why `n = 24` above. Below that point the correction costs more than the dense solve it replaces,
so the dense route is the right answer. `P` must also be `Diagonal`.

Under [`InteriorPoint`](@ref), `linsys = :auto` never picks this backend, and
`linsys = :lowrank` throws. The Woodbury solve misses the tolerance on linear programs
([Backends under the interior-point method](@ref)). A `Diagonal` `P` with a `RowCoupled` `A`
gets the full KKT factorization instead.
