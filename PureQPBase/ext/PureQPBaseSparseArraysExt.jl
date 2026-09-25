"""
    PureQPBaseSparseArraysExt

Sparse-aware traversals and a sparse-forming reduced backend, for `SparseMatrixCSC`.

PureQPBase's per-iteration products already go through `mul!`, which sparse matrices handle
well on their own. Equilibration and the factorization are different: they walk the
caller's matrices entry by entry, and the generic loop visits every structural zero and
reaches each one through `M[i, j]`, which on CSC is a binary search within the column. On a
400×200 matrix at 1% density that is 80 000 searches per sweep where 800 direct reads would
do, and it made equilibration ten times slower on a sparse matrix than on a dense one.

The four column traversals below walk `nzrange` instead. The extension also supplies two
reduced backends — [`SparseFormedInverse`](@ref), which forms `Ãᵀ diag(ρ) Ã` from the stored
entries but still factors densely, and [`SparseCholmod`](@ref), which also factors sparsely
when the factor stays sparse enough to pay — and a convexity test that does not densify `P`.
Nothing else in the solver needs to know the storage.
"""
module PureQPBaseSparseArraysExt

using PureQPBase: PureQPBase
using TypeContracts: TypeContracts, @verify
using LinearAlgebra: Symmetric, Diagonal, LowerTriangular, UpperTriangular,
    UnitLowerTriangular, UnitUpperTriangular, I, diag,
    cholesky, cholesky!, ldlt, ldlt!, issuccess, ldiv!, transpose, transpose!
using SparseArrays: SparseMatrixCSC, nnz, nzrange, rowvals, nonzeros, sparse
using SparseArrays.CHOLMOD: CHOLMOD

"""
    PureQPBase.check_storage(M::SparseMatrixCSC, rows, cols)

Establish that the four column traversals below can index by a row read out of `M` without
checking it, or throw.

Those traversals reach a weight vector, and `dest`, at `rows[k]` — an index the compiler
cannot prove is in range, so it checks every stored entry. On equilibration that check is
most of the per-entry work: for the benchmark suite's Eq QP, whose `P` holds 39 638 entries in a
200×200 matrix, a sweep costs 144.5 µs checked and 18.7 µs unchecked.

Called once from [`validate`](@ref), over `nnz(M)` entries, against ten sweeps that each
traverse the same entries. `SparseMatrixCSC`'s own constructor already enforces most of this;
checking again is cheap and is what makes dropping the per-entry check defensible rather than
assumed.
"""
function PureQPBase.check_storage(M::SparseMatrixCSC, rows::Integer, cols::Integer)
    size(M) == (rows, cols) || throw(
        ArgumentError("expected a $(rows)×$(cols) matrix, got $(size(M))")
    )
    colptr, rv = M.colptr, rowvals(M)
    nz = length(rv)
    (length(colptr) == cols + 1 && colptr[1] == 1 && colptr[cols + 1] == nz + 1) || throw(
        ArgumentError("malformed column pointer for a $(rows)×$(cols) matrix")
    )
    for j in 1:cols
        colptr[j] <= colptr[j + 1] || throw(
            ArgumentError("column pointer decreases at column $j")
        )
    end
    for k in 1:nz
        1 <= rv[k] <= rows || throw(
            ArgumentError("row index $(rv[k]) at position $k is outside 1:$rows")
        )
    end
    return nothing
end

# The four traversals run unchecked, which `check_storage` is what makes safe. `w`, `e` and
# `dest` are indexed at a stored row, and every caller passes one sized to the matrix's rows:
# `validate` checks the dimensions and the workspace buffers are built from them.

@inline function PureQPBase.weighted_colmax(
        ::Type{T}, M::SparseMatrixCSC, j::Integer, w::AbstractVector
    ) where {T}
    r = zero(T)
    rows, vals = rowvals(M), nonzeros(M)
    @inbounds for k in nzrange(M, j)
        r = max(r, w[rows[k]] * abs(T(vals[k])))
    end
    return r
end

@inline function PureQPBase.weighted_colmax_rowmax!(
        ::Type{T}, e::AbstractVector, M::SparseMatrixCSC, j::Integer,
        w::AbstractVector, s
    ) where {T}
    r = zero(T)
    rows, vals = rowvals(M), nonzeros(M)
    @inbounds for k in nzrange(M, j)
        i = rows[k]
        v = abs(T(vals[k]))
        r = max(r, w[i] * v)
        e[i] = max(e[i], s * v)
    end
    return r
end

@inline function PureQPBase.scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::SparseMatrixCSC, j::Integer, f::F
    ) where {T, F}
    rows, vals = rowvals(M), nonzeros(M)
    @inbounds for k in nzrange(M, j)
        i = rows[k]
        dest[i, j] = f(T(vals[k]), i)
    end
    return dest
end

@inline function PureQPBase.add_scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::SparseMatrixCSC, j::Integer, f::F
    ) where {T, F}
    rows, vals = rowvals(M), nonzeros(M)
    @inbounds for k in nzrange(M, j)
        i = rows[k]
        dest[i, j] += f(T(vals[k]), i)
    end
    return dest
end


"""
    SparseFormedInverse{T,M} <: PureQPBase.ReducedInverse

The reduced backend for a sparse `A`, which forms `Ãᵀ diag(ρ) Ã` from the stored entries
instead of through a dense product.

[`PureQPBase.ReducedCholesky`](@ref) writes the scaled `Ã` into an `m×n` dense buffer so that
one `syrk` produces the reduced matrix. That buffer is mostly zeros when `A` is sparse, and
the `syrk` does `mn²` flops to multiply them: on a 2000×4000 problem at 0.25% density it is
about 63% of a refactorization and 65% of the workspace. Accumulating over the stored
entries instead costs `Σᵢ nnzᵢ²` and needs no buffer, so this backend holds only the
inverse.

The reduced matrix itself is still dense, and everything after it is forming — the
Cholesky, the inversion, the per-iteration `symv` — is exactly what the dense backend does.
"""
struct SparseFormedInverse{T <: Real, M <: AbstractMatrix{T}} <: PureQPBase.ReducedInverse
    Rinv::M
    # `A` by rows, which `gram_upper!` reads and `A`'s own column-major storage cannot give.
    # It depends on `A` and on nothing else, so it is filled once where the backend is built
    # and again in [`check_update`](@ref) when `A` is replaced, and every refactorization the
    # weights ask for reads it without touching it. Vectors, so this stays immutable.
    rowptr::Vector{Int}
    colind::Vector{Int}
    nzval::Vector{T}
end

"A row of `A` touching this fraction of the columns fills the reduced matrix on its own."
const DENSE_ROW_FRACTION = 0.5

"""
Fraction of the `(n+m)²` entries of the interior-point method's dense KKT terminal that the
KKT pattern may store before no sparse form can pay.
"""
const KKT_PATTERN_DENSITY = 0.25

"Fraction of the `n×n` reduced matrix the reduced pattern fills before it counts as full."
const FULL_REDUCED_PATTERN = 0.9

"""
Multiple of the reduced matrix's `n²` entries that `Σᵢ nnzᵢ²` may reach before rebuilding it
each outer iteration costs more than the KKT form's `O(nnz)` refill saves.
"""
const REDUCED_ASSEMBLY_LIMIT = 1.5

"""
Fraction of the `n×n` matrix ADMM would otherwise invert that the reduced pattern may fill and
still be worth factoring, given that ADMM factors once and solves against that factorization
for the rest of the run.

Measured in `PureIPM/bench/results/ipm_selection.json`: a pattern at 0.06 loses to the dense inverse
by 4.7× and one at 0.083 by 3.0×, because the factor of a scattered pattern fills in far past
the pattern itself. Every pattern below this limit either wins or ties.
"""
const SPARSE_PATTERN_FRACTION = 0.05

"""
    row_pattern(A) -> (densest, sumsq)

The most nonzeros any row of `A` holds, and `Σᵢ nnzᵢ²` over its rows, in one pass.

`Ãᵀ W Ã` gives each row of `A` an outer product with itself, so `densest²` is a lower bound
on `nnz(R)` and `sumsq` is the work of accumulating `R` from the stored entries.
"""
function row_pattern(A::SparseMatrixCSC)
    counts = zeros(Int, size(A, 1))
    for i in rowvals(A)
        counts[i] += 1
    end
    densest, sumsq = 0, 0
    for c in counts
        densest = max(densest, c)
        sumsq += c * c
    end
    return (densest, sumsq)
end

"""
    sparse_form(P, A, n, m, sel) -> Symbol

Which sparse form of the system `sel`'s algorithm solves this pattern is served by: `:kkt`
for the `(n+m)×(n+m)` quasi-definite matrix, `:reduced` for the `n×n` `P̃ + σI + Ãᵀ W Ã`, or
`:none` for neither, which leaves the pair to the algorithm's terminal rung.

The thresholds are measured per algorithm — ADMM factors once and pays the pattern back over
thousands of solves, the interior-point method rebuilds and refactors every outer iteration —
so there is no generic method and a new [`PureQPBase.SelectionFor`](@ref) reaching this is told
to define one.

The answer is read from the patterns of `P` and `A` alone — the densest row, `Σᵢ nnzᵢ²`, the
stored entries of the KKT matrix, and [`reduced_nnz`](@ref)'s count of the symbolic
`AᵀA ∪ P ∪ I` pattern. Nothing is factored, so the backend the ladder then builds is the
only factorization `setup` pays for.

The thresholds come from `PureIPM/bench/results/ipm_selection.json`, which records every backend each
of 71 problems admits, timed under both algorithms, alongside the backend this rule picked.
Under `InteriorPoint` the pick is within 1.3× of the fastest measured backend on 69 of the 71,
worst case 1.52×; under `OperatorSplitting` it is within 1.03× on all 71. Those figures are
in-sample: the rule was fitted to this set and scored on it. The `OperatorSplitting` column
compares cost per iteration, since 10 of the 71 run to the iteration cap at the sweep's
tolerance rather than converging.
"""
sparse_form(P, A, n::Integer, m::Integer, sel::PureQPBase.SelectionFor) =
    PureQPBase.refuse_selection("PureQPBaseSparseArraysExt.sparse_form", sel)

function sparse_form(P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer, m::Integer, ::PureQPBase.IPMSelection)
    # The terminal is the dense `(n+m)` KKT factorization, so what decides against a sparse
    # form is how much of that matrix is stored to begin with: past a quarter there is too
    # little structure left for any ordering to exploit, and `syrk`-backed dense arithmetic
    # wins. This is what sends the random-`P` families to the terminal and keeps the sparse
    # suite classes off it.
    kkt_entries = nnz(P) + 2 * nnz(A) + n + m
    kkt_entries >= KKT_PATTERN_DENSITY * (n + m)^2 && return :none
    densest, sumsq = row_pattern(A)
    # One row of `A` spanning the variables fills `R` by itself, whatever the rest of the
    # pattern looks like; the KKT form never squares it. The Portfolio class's budget row
    # `1ᵀx = 1` is this case, and the reduced form is 139× slower there.
    densest >= DENSE_ROW_FRACTION * n && return :kkt
    # With few rows the KKT matrix is barely wider than the reduced one, so keeping `A`
    # unsquared costs nothing.
    2 * m <= n && return :kkt
    # `reduced_nnz` stops counting once the pattern passes `limit`, so this costs a fraction
    # of a full count on exactly the patterns that would be most expensive to count.
    limit = cld(n^2, 2) * FULL_REDUCED_PATTERN
    # A reduced pattern this full has no sparsity to lose: factoring it is dense work at `n`,
    # which beats a sparse factorization of a matrix `n + m` wide.
    reduced_nnz(P, A, n, floor(Int, limit)) > limit && return :reduced
    # The reduced matrix is rebuilt and refactored every outer iteration. Accumulating it
    # costs `Σᵢ nnzᵢ²` where refilling the KKT matrix costs `O(nnz)`, so a wide `A` is
    # cheaper to keep unsquared even when its reduced pattern stays sparse.
    return sumsq > REDUCED_ASSEMBLY_LIMIT * n^2 ? :kkt : :reduced
end

function sparse_form(P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer, m::Integer, ::PureQPBase.ADMMSelection)
    # ADMM factors once and solves against that factorization for the rest of the run, so
    # what decides is the per-iteration solve: a pair of sparse triangular solves against a
    # `symv` on the `n×n` inverse the terminal holds. Both sparse forms are therefore
    # measured against `n²`, the terminal's size, and not against their own.
    densest, _ = row_pattern(A)
    if densest >= DENSE_ROW_FRACTION * n
        # The reduced form is out — one row fills it — so the KKT form is the only sparse
        # candidate, and it pays while its whole pattern is smaller than half the matrix the
        # terminal inverts. Portfolio-shaped pairs sit two orders of magnitude inside that;
        # a pair whose `A` is both wide and dense sits outside it.
        return 2 * (nnz(P) + 2 * nnz(A) + n + m) <= n^2 ? :kkt : :none
    end
    limit = SPARSE_PATTERN_FRACTION * n^2
    return reduced_nnz(P, A, n, floor(Int, limit / 2)) * 2 <= limit ? :reduced : :none
end

# Both rungs consult the same rule, so the form it names is the one built and the other
# declines. `gated = false` is what a caller who named `linsys = :sparse` gets: the rule is
# skipped and only a representation mismatch or a factorization failure can still refuse.
function PureQPBase.kkt_rung(
        P, A::SparseMatrixCSC, prob, wt, sel::PureQPBase.SelectionFor; gated::Bool = true
    )
    P isa SparseMatrixCSC || return nothing
    gated && sparse_form(P, A, prob.n, prob.m, sel) !== :kkt && return nothing
    ls = factored_kkt_backend(P, A, prob, wt)
    return isnothing(ls) ? nothing : (ls, true)
end

function PureQPBase.reduced_rung(
        P, A::SparseMatrixCSC, prob, wt, sel::PureQPBase.SelectionFor; gated::Bool = true
    )
    P isa SparseMatrixCSC || return nothing
    gated && sparse_form(P, A, prob.n, prob.m, sel) !== :reduced && return nothing
    ls = cholmod_backend(P, A, prob, wt)
    return isnothing(ls) ? nothing : (ls, true)
end

function PureQPBase.formed_rung(
        P, A::SparseMatrixCSC, prob::PureQPBase.Problem{T}, sel::PureQPBase.ADMMSelection
    ) where {T <: Real}
    # `SparseFormedInverse.factorize!` accumulates `P`'s columns through `add_scaled_col!`,
    # which indexes. `A` is a `SparseMatrixCSC` here and so always readable; `P` is not
    # constrained by the signature.
    PureQPBase.is_materializable(P) || return nothing
    n = prob.n
    Rinv = similar(prob.q0, T, n, n)
    # Grouped here rather than on the first `factorize!`: building it there would leave a
    # `resize!` on a path every refactorization takes, and a guarantee that holds only after
    # the first call is not one the analysis can state.
    ls = SparseFormedInverse{T, typeof(Rinv)}(Rinv, Int[], Int[], T[])
    csr_rows!(ls, A)
    return (ls, false)
end

PureQPBase.backend_name(::SparseFormedInverse) = :sparse_formed

function PureQPBase.backend_info(ls::SparseFormedInverse)
    dim = size(ls.Rinv, 1)
    return PureQPBase.BackendInfo(
        PureQPBase.backend_name(ls), true, :reduced, dim, PureQPBase.dense_triangle(dim)
    )
end

"""
    csr_rows(A) -> (rowptr, colind, nzval)

Group `A`'s stored entries by row, as `A[i, colind[p]] == nzval[p]` for
`p in rowptr[i]:(rowptr[i + 1] - 1)`, with `colind` ascending within each row.

The accumulation needs each row's nonzeros together and CSC stores columns, so something
has to transpose. `copy(transpose(A))` would, but it goes through the `SparseMatrixCSC`
constructor, whose dimension validation raises through a closure that formats its message —
enough runtime dispatch on a branch that never fires to cost `factorize!` the type-stability
guarantee. Three plain vectors carry everything the accumulation reads.
"""
function csr_rows(A::SparseMatrixCSC{Tv}) where {Tv}
    m, n = size(A)
    rows, vals = rowvals(A), nonzeros(A)
    rowptr = Vector{Int}(undef, m + 1)
    fill!(rowptr, 0)
    for k in eachindex(rows)
        rowptr[rows[k]] += 1
    end
    total = 1
    for i in 1:m
        count = rowptr[i]
        rowptr[i] = total
        total += count
    end
    rowptr[m + 1] = total
    colind = Vector{Int}(undef, length(rows))
    nzval = Vector{Tv}(undef, length(vals))
    # Columns are visited in ascending order, which is what leaves `colind` ascending
    # within each row and lets `gram_upper!` take the upper triangle as `k >= j`.
    pos = copy(rowptr)
    for j in 1:n
        for k in nzrange(A, j)
            i = rows[k]
            p = pos[i]
            colind[p] = j
            nzval[p] = vals[k]
            pos[i] = p + 1
        end
    end
    return (rowptr, colind, nzval)
end

"""
    csr_rows!(ls, A) -> ls

Fill `ls`'s row grouping from `A`, growing its buffers to fit and allocating nothing once they
already do.

Separate from [`csr_rows`](@ref) because this one runs on the refactorization path: a `ρ`
retune re-forms the Gram, and rebuilding the grouping there would allocate `O(nnz)` every time
`ρ` moves.
"""
function csr_rows!(ls::SparseFormedInverse{Tv}, A::SparseMatrixCSC) where {Tv}
    m, n = size(A)
    rows, vals = rowvals(A), nonzeros(A)
    nnzA = length(rows)
    length(ls.rowptr) == m + 1 || resize!(ls.rowptr, m + 1)
    length(ls.colind) == nnzA || resize!(ls.colind, nnzA)
    length(ls.nzval) == nnzA || resize!(ls.nzval, nnzA)
    rowptr, colind, nzval = ls.rowptr, ls.colind, ls.nzval
    fill!(rowptr, 0)
    for k in eachindex(rows)
        rowptr[rows[k]] += 1
    end
    total = 1
    for i in 1:m
        count = rowptr[i]
        rowptr[i] = total
        total += count
    end
    rowptr[m + 1] = total
    # `rowptr` doubles as the running cursor and is shifted back afterwards. That is what the
    # `copy` in `csr_rows` buys, and the copy is the allocation this path exists to avoid.
    for j in 1:n
        for k in nzrange(A, j)
            i = rows[k]
            p = rowptr[i]
            colind[p] = j
            nzval[p] = vals[k]
            rowptr[i] = p + 1
        end
    end
    for i in m:-1:2
        rowptr[i] = rowptr[i - 1]
    end
    rowptr[1] = 1
    return ls
end

# `A` is being replaced, so the row grouping no longer describes the matrix the solver holds.
# Rebuilt here, on the `update!` path, rather than left for the next `factorize!`: that one
# runs whenever `ρ` moves and is where the grouping exists to not be rebuilt.
function PureQPBase.check_update(ls::SparseFormedInverse, P, A::SparseMatrixCSC)
    csr_rows!(ls, A)
    return nothing
end

"""
    gram_upper!(R, rowptr, colind, nzval, rho, E, D)

Add `Ãᵀ diag(ρ) Ã` to the upper triangle of `R`, where `Ã = diag(E) A diag(D)`.

Each row of `A` contributes the outer product of its own nonzeros, so the work is
`Σᵢ nnzᵢ²` rather than the `mn²` of a dense product against a mostly-zero buffer.

Only the upper triangle is written, which is all that is ever read: `cholesky!` is handed a
`Symmetric(R, :U)`, `potri!` writes the inverse into the same triangle, and the solve
multiplies by `Symmetric(Rinv, :U)`. The inner loop starts at `p` rather than at the row's
first entry because [`csr_rows`](@ref) leaves each row's columns ascending.
"""
function gram_upper!(
        R::AbstractMatrix{T}, rowptr::Vector{Int}, colind::Vector{Int},
        nzval::AbstractVector, rho::AbstractVector, E::AbstractVector, D::AbstractVector
    ) where {T}
    for i in eachindex(rho, E)
        ei = E[i]
        w = rho[i] * ei * ei
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop
            j = colind[p]
            wvj = w * T(nzval[p]) * D[j]
            for q in p:stop
                k = colind[q]
                R[j, k] += wvj * T(nzval[q]) * D[k]
            end
        end
    end
    return R
end

function PureQPBase.factorize!(ls::SparseFormedInverse{T}, prob, wt)::Bool where {T}
    P, A, D, E, c, n, m = prob.P, prob.A, prob.D, prob.E, prob.c, prob.n, prob.m
    R = ls.Rinv
    fill!(R, zero(T))
    if m > 0
        # Built once and kept: it describes `A`, which only `update!` can replace, and that
        # clears the flag through `check_update`. A `ρ` retune reaches here too, and rebuilding
        # the grouping for one of those would allocate `O(nnz)` every time `ρ` moves.
        gram_upper!(R, ls.rowptr, ls.colind, ls.nzval, wt.w, E, D)
    end
    for j in 1:n
        dj = D[j]
        PureQPBase.add_scaled_col!(T, R, P, j, (p, i) -> c * D[i] * p * dj)
    end
    for i in 1:n
        R[i, i] += wt.sigma
    end
    F = cholesky!(Symmetric(R); check = false)
    issuccess(F) || return false
    PureQPBase.invert_spd!(R, F)
    return true
end


"""
    KKTGram{T}

The upper triangle of the quasi-definite KKT matrix, with the map that refills its values.

    K = ⎡P̃ + σI    Ãᵀ  ⎤
        ⎣Ã      −diag(ρ⁻¹)⎦

Every stored entry comes from one place: an entry of `P`'s upper triangle, an entry of `A`
transposed into the `(1,2)` block, the `σ` on a leading diagonal entry, or a `−ρ⁻¹` on a
trailing one. None of that moves when `ρ`, `D`, `E`, `c` or `σ` do, so the slots are found
once and [`refill_kkt!`](@ref) is a pass over them.

[`kkt_sparse`](@ref) builds the same matrix through five sparse products and as many
intermediate matrices. That is how the pattern is established; it is not how it should be
rebuilt, and a refactorization rebuilds it every time `ρ` is retuned.
"""
struct KKTGram{T}
    K::SparseMatrixCSC{T, Int}
    acolptr::Vector{Int}
    arowval::Vector{Int}
    pcolptr::Vector{Int}
    prowval::Vector{Int}
    arow::Vector{Int}
    acol::Vector{Int}
    aperm::Vector{Int}
    aslot::Vector{Int}
    prow::Vector{Int}
    pcol::Vector{Int}
    pperm::Vector{Int}
    pslot::Vector{Int}
    sslot::Vector{Int}
    rslot::Vector{Int}
end

"""
    SparseKKT{T,V,F} <: PureQPBase.LinearSystem

Factors the full `(n+m)×(n+m)` quasi-definite system sparsely, with CHOLMOD's `ldlt`.

The reduced form squares `A`, so one dense row makes `R` dense however sparse the rest of it
is. The full system does not: a dense row of `A` stays one sparse row of

    K = ⎡P̃ + σI    Ãᵀ  ⎤
        ⎣Ã      −diag(ρ⁻¹)⎦

On the benchmark suite's Portfolio class — 0.9% dense `A`, one row touching every column — `R`
comes out 99% dense while `K`'s factor is 0.3% dense, and factoring `K` is 7.5× faster than
factoring `R`.

`K` is quasi-definite: `P̃ + σI` is positive definite and `−diag(ρ⁻¹)` negative definite. A
quasi-definite matrix has an `LDLᵀ` factorization under *any* symmetric permutation, which
is why a fill-reducing ordering chosen once can be reused without pivoting for stability —
the property upstream's own solver rests on.

`L`, `D⁻¹` and the permutation are extracted rather than solved through, because CHOLMOD's
`ldiv!` allocates on every call and the hot path may not. See
[`PureQPBase.reduced_rhs!`](@ref) for the reduced backends' equivalent.
"""
mutable struct SparseKKT{T <: Real, V <: AbstractVector{T}, F} <: PureQPBase.LinearSystem
    gram::KKTGram{T}
    fact::F
    L::SparseMatrixCSC{T, Int}
    dinv::Vector{T}
    perm::Vector{Int}
    rhs::V
    work::V
end

PureQPBase.backend_name(::SparseKKT) = :sparse_kkt

PureQPBase.backend_info(ls::SparseKKT) = PureQPBase.BackendInfo(
    PureQPBase.backend_name(ls), true, :kkt, size(ls.L, 1), nnz(ls.L)
)

"""
    kkt_sparse(T, P, A, rho_inv, E, D, c, sigma) -> SparseMatrixCSC

The scaled quasi-definite KKT matrix, kept sparse.

Equilibration comes out of the blocks the same way it does for the reduced matrix:
`P̃ = c·diag(D) P diag(D)` and `Ã = diag(E) A diag(D)`, so nothing is formed scaled.
"""
function kkt_sparse(::Type{T}, P, A, rho_inv, E, D, c, sigma) where {T}
    Dg = Diagonal(D)
    Pt = SparseMatrixCSC{T, Int}(c * (Dg * sparse(Symmetric(P)) * Dg) + sigma * I)
    At = SparseMatrixCSC{T, Int}(Diagonal(E) * A * Dg)
    m = size(A, 1)
    return SparseMatrixCSC{T, Int}(
        Symmetric([Pt transpose(At); At sparse(-Diagonal(rho_inv))], :L)
    )
end


"Build the KKT matrix's pattern and the slot map that refills it."
function kkt_gram(::Type{T}, P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer, m::Integer) where {T}
    N = n + m
    prows, arows = rowvals(P), rowvals(A)
    nup = 0
    for j in 1:n, k in nzrange(P, j)
        nup += prows[k] <= j
    end
    na = nnz(A)
    total = nup + na + n + m
    crow = Vector{Int}(undef, total)
    ccol = Vector{Int}(undef, total)
    prow = Vector{Int}(undef, nup)
    pcol = Vector{Int}(undef, nup)
    pperm = Vector{Int}(undef, nup)
    t = 0
    s = 0
    for j in 1:n, k in nzrange(P, j)
        i = prows[k]
        i <= j || continue
        s += 1
        prow[s] = i; pcol[s] = j; pperm[s] = k
        t += 1
        crow[t] = i; ccol[t] = j
    end
    # `Ã` sits in the (1,2) block: entry (i, j) of `A` is entry (j, n + i) of `K`, which is
    # above the diagonal because `j <= n < n + i`.
    arow = Vector{Int}(undef, na)
    acol = Vector{Int}(undef, na)
    aperm = Vector{Int}(undef, na)
    s = 0
    for j in 1:n, k in nzrange(A, j)
        i = arows[k]
        s += 1
        arow[s] = i; acol[s] = j; aperm[s] = k
        t += 1
        crow[t] = j; ccol[t] = n + i
    end
    for j in 1:n
        t += 1
        crow[t] = j; ccol[t] = j
    end
    for i in 1:m
        t += 1
        crow[t] = n + i; ccol[t] = n + i
    end
    K, slot = pattern_from(T, crow, ccol, N)
    return KKTGram{T}(
        K, copy(A.colptr), copy(arows), copy(P.colptr), copy(prows),
        arow, acol, aperm, slot[(nup + 1):(nup + na)],
        prow, pcol, pperm, slot[1:nup],
        slot[(nup + na + 1):(nup + na + n)], slot[(nup + na + n + 1):total],
    )
end

"Whether `g`'s slots still describe these matrices, which they do unless a pattern changed."
function describes(g::KKTGram, P::SparseMatrixCSC, A::SparseMatrixCSC)
    return g.acolptr == A.colptr && g.arowval == rowvals(A) &&
        g.pcolptr == P.colptr && g.prowval == rowvals(P)
end

"""
    refill_kkt!(g, P, A, rho_inv, E, D, c, sigma) -> SparseMatrixCSC

Rebuild `g.K` for the current data, in one pass over the recorded slots, without allocating.
"""
function refill_kkt!(
        g::KKTGram{T}, P::SparseMatrixCSC, A::SparseMatrixCSC, rho_inv, E, D, c, sigma
    ) where {T}
    nz = nonzeros(g.K)
    fill!(nz, zero(T))
    pvals, avals = nonzeros(P), nonzeros(A)
    for k in eachindex(g.pslot)
        nz[g.pslot[k]] += c * D[g.prow[k]] * T(pvals[g.pperm[k]]) * D[g.pcol[k]]
    end
    for k in eachindex(g.aslot)
        nz[g.aslot[k]] += E[g.arow[k]] * T(avals[g.aperm[k]]) * D[g.acol[k]]
    end
    for j in eachindex(g.sslot)
        nz[g.sslot[j]] += sigma
    end
    for i in eachindex(g.rslot)
        nz[g.rslot[i]] -= rho_inv[i]
    end
    return g.K
end

function PureQPBase.factorize!(ls::SparseKKT{T}, prob, wt)::Bool where {T}
    P, A = prob.P, prob.A
    if !describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries elsewhere, so the slot map and
        # the analysis built on its pattern are both stale.
        ls.gram = kkt_gram(T, P, A, prob.n, prob.m)
        K = refill_kkt!(ls.gram, P, A, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
        ls.fact = ldlt(Symmetric(K, :U); check = false)
        # The ordering belongs to the symbolic analysis, so it is reread exactly when a new
        # one is done.
        ls.perm = ls.fact.p::Vector{Int}
    else
        # The pattern does not depend on ρ or the equilibration factors, so every
        # refactorization after the first reuses the ordering and the symbolic phase.
        K = refill_kkt!(ls.gram, P, A, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
        ldlt!(ls.fact, Symmetric(K, :U); check = false)
    end
    issuccess(ls.fact) || return false
    N = prob.n + prob.m
    # `ls.L` packs `D` on the diagonal of a unit-triangular `L`. The substitutions skip that
    # stored diagonal and `D⁻¹` is applied between them, so `dinv` is read from it here.
    ls.L = factor_csc!(ls.L, ls.fact, N, false)
    length(ls.dinv) == N || resize!(ls.dinv, N)
    return ldl_dinv!(ls.dinv, ls.L, N)
end

"""
    ldl_forward!(x, L, N)
    ldl_backward!(x, L, N)

Substitution against the unit-lower factor of an `LDLᵀ`, in place.

`L` is CHOLMOD's packed `LD`: column `j` begins with `D[j]` and its subdiagonal entries
follow, so both loops start one past `colptr[j]` and the diagonal is applied separately.

Written out rather than left to `ldiv!(UnitLowerTriangular(L), x)`, which is the exception
rather than the rule here. Measured on this backend's own factor, resetting the vector every
sample because these solves are in place, `solve_system!` costs 9.06 µs through `ldiv!` and
5.78 µs this way. [`SparseCholmod`](@ref) does the same for its `L Lᵀ` factor, and
[`llt_forward!`](@ref) records where the two paths cross over.
"""
function ldl_forward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in 1:N
        xj = x[j]
        for p in (colptr[j] + 1):(colptr[j + 1] - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function ldl_backward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in N:-1:1
        s = x[j]
        for p in (colptr[j] + 1):(colptr[j + 1] - 1)
            s -= vals[p] * x[rows[p]]
        end
        x[j] = s
    end
    return x
end

function PureQPBase.solve_system!(ls::SparseKKT{T}, prob, wt, rhs_x, rhs_z, x, z)::Nothing where {T}
    n, m = prob.n, prob.m
    N = n + m
    perm, work = ls.perm, ls.work
    # Permute straight out of the two right-hand sides: `K[perm, perm] = L D Lᵀ`, and the
    # assembled vector is never needed in its own order.
    for i in 1:N
        p = perm[i]
        work[i] = p <= n ? rhs_x[p] : rhs_z[p - n]
    end
    ldl_forward!(work, ls.L, N)
    PureQPBase.scale_by!(work, ls.dinv)
    ldl_backward!(work, ls.L, N)
    # And scatter straight into the outputs. The eliminated multiplier gives `z̃` without
    # another product with `A`.
    for i in 1:N
        p = perm[i]
        if p <= n
            x[p] = work[i]
        else
            z[p - n] = work[i]
        end
    end
    w_inv = wt.w_inv
    for i in 1:m
        z[i] = rhs_z[i] + w_inv[i] * z[i]
    end
    return nothing
end

"""
    PureQPBase.solve_multiplier!(ls::SparseKKT, prob, wt, rhs_x, rhs_z, x, nu) -> Nothing

The forward-backward solve already leaves `ν` in `work` before the eliminated multiplier
would be turned into `z̃ = rhs_z + w_inv ⊙ ν`, so this is [`PureQPBase.solve_system!`](@ref)
minus that last loop.
"""
function PureQPBase.solve_multiplier!(
        ls::SparseKKT{T}, prob, wt, rhs_x, rhs_z, x, nu
    )::Nothing where {T}
    n, m = prob.n, prob.m
    N = n + m
    perm, work = ls.perm, ls.work
    for i in 1:N
        p = perm[i]
        work[i] = p <= n ? rhs_x[p] : rhs_z[p - n]
    end
    ldl_forward!(work, ls.L, N)
    PureQPBase.scale_by!(work, ls.dinv)
    ldl_backward!(work, ls.L, N)
    for i in 1:N
        p = perm[i]
        if p <= n
            x[p] = work[i]
        else
            nu[p - n] = work[i]
        end
    end
    return nothing
end

"""
    factored_kkt_backend(P, A, prob, wt) -> SparseKKT or nothing

Assemble the full KKT matrix from the equilibrated data and factor it sparsely.

Whether this form suits the pair is [`sparse_form`](@ref)'s question and is already settled
before this is called, so the factorization produced here is the one the solver goes on to
use — a backend returned from here needs no further `factorize!`. `nothing` means the
representation does not match or the matrix does not factor at this regularization.
"""
function factored_kkt_backend(
        P, A, prob::PureQPBase.Problem{T}, wt::PureQPBase.SystemWeights{T}
    ) where {T <: Real}
    n, m = prob.n, prob.m
    # The concrete type, not `issparse`: everything downstream of here — `kkt_gram`,
    # `refill_kkt!` — is written against `SparseMatrixCSC`'s stored columns, and a
    # `Symmetric` wrapper over one answers `issparse` while matching none of it.
    (P isa SparseMatrixCSC && n > 0 && m > 0) || return nothing
    gram = kkt_gram(T, P, A, n, m)
    K = refill_kkt!(gram, P, A, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
    # As for the reduced matrix: a pure-Julia LDLᵀ, where one is loaded, factors this faster
    # and needs nothing extracted from a foreign factor afterwards.
    alt = PureQPBase.ldl_kkt_backend(gram, prob.q0, n, m)
    isnothing(alt) || return alt
    F = ldlt(Symmetric(K, :U); check = false)
    issuccess(F) || return nothing
    LD = sparse(F.LD)::SparseMatrixCSC{T, Int}
    d = diag(LD)
    any(iszero, d) && return nothing
    check_factor(LD, n + m)
    v = similar(prob.q0, T, n + m)
    return SparseKKT{T, typeof(v), typeof(F)}(
        gram, F, LD, inv.(d), F.p::Vector{Int}, v, similar(v)
    )
end

@verify SparseFormedInverse trim_compat = true


"""
    ReducedGram{T}

The reduced matrix's upper triangle, with the map that rebuilds its values in one pass.

Every off-diagonal entry of `Ãᵀ diag(ρ) Ã` comes from a row of `A` that stores both of its
columns, so one traversal of `A` by rows enumerates every contribution; `P`'s stored entries
and `σ` supply the rest. Which slot of `nonzeros(R)` a contribution lands in depends only on
where `P` and `A` store entries — never on what they store, nor on `ρ`, `D`, `E`, `c` or `σ`
— so the slots are found once and [`refill!`](@ref) is a pass over them.

That matters because a refactorization happens every time `ρ` moves. Rebuilding the matrix
through chained sparse products instead allocates four intermediate matrices: on the
benchmark suite's Huber problem, 2.19 MB and 120 µs against 16.6 µs and nothing here.

`aperm` and `pperm` index into `nonzeros(A)` and `nonzeros(P)` rather than holding copies of
them, so a refill always reads the values the workspace currently holds. The patterns are
kept alongside, because a `P` or `A` whose pattern has changed invalidates every slot.
"""
struct ReducedGram{T}
    R::SparseMatrixCSC{T, Int}
    acolptr::Vector{Int}
    arowval::Vector{Int}
    pcolptr::Vector{Int}
    prowval::Vector{Int}
    rowptr::Vector{Int}
    colind::Vector{Int}
    aperm::Vector{Int}
    aslot::Vector{Int}
    prow::Vector{Int}
    pcol::Vector{Int}
    pperm::Vector{Int}
    pslot::Vector{Int}
    dslot::Vector{Int}
end

"""
    SparseCholmod{T,V,F} <: PureQPBase.LinearSystem

Forms the reduced matrix sparsely *and* factors it sparsely, through SparseArrays.

`cholesky(Symmetric(R))` on a `SparseMatrixCSC` is CHOLMOD, and `cholesky!(F, R)` reuses the
symbolic analysis it already did, so every refactorization after the first pays only for the
numeric phase. The factorization is `L Lᵀ`, not `L D Lᵀ`; the solve below depends on that and
[`cholmod_backend`](@ref) checks it before selecting this backend.

This cannot be a [`PureQPBase.ReducedInverse`](@ref): that stores `R⁻¹` and solves with one
`symv`, and the inverse of a sparse matrix is dense. The Cholesky factor is kept instead and
each solve is a pair of sparse triangular solves. On a banded `R` that is the better trade by
a wide margin — at `n = 4000` a refactorization goes from 1063 ms to 0.66 ms and a solve from
1430 µs to 71 µs — and on a filled-in `R` it is worse, which [`sparse_form`](@ref) decides
from the pattern.

`L` and `perm` are extracted from the factorization rather than solved through it, because
CHOLMOD's `ldiv!` allocates a result and workspace on every call — 64 KB per solve at
`n = 2000`, which the hot path may not do. Applying the permutation and the two triangular
solves over preallocated buffers allocates nothing and measures slightly faster besides.

`Lt` holds the same factor transposed, so that back-substitution can scatter rather than
gather — see [`llt_backward!`](@ref). It costs one transpose per factorization, which is
`O(nnz(L))` against the factorization's own cost.
"""
mutable struct SparseCholmod{T <: Real, V <: AbstractVector{T}, F} <: PureQPBase.LinearSystem
    gram::ReducedGram{T}
    fact::F
    L::SparseMatrixCSC{T, Int}
    Lt::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    permuted::V
end

PureQPBase.backend_name(::SparseCholmod) = :cholmod

PureQPBase.backend_info(ls::SparseCholmod) = PureQPBase.BackendInfo(
    PureQPBase.backend_name(ls), true, :reduced, size(ls.L, 1), nnz(ls.L)
)


"""
    csr_order(A) -> (rowptr, colind, aperm)

`A`'s stored entries grouped by row: entry `p` of row `i` sits at column `colind[p]` and
holds `nonzeros(A)[aperm[p]]`, for `p in rowptr[i]:(rowptr[i + 1] - 1)`, with `colind`
ascending within each row.

The positions rather than the values, so a caller can reread `A` after its numbers change.
"""
function csr_order(A::SparseMatrixCSC)
    m, n = size(A)
    rows = rowvals(A)
    rowptr = zeros(Int, m + 1)
    for k in eachindex(rows)
        rowptr[rows[k]] += 1
    end
    total = 1
    for i in 1:m
        count = rowptr[i]
        rowptr[i] = total
        total += count
    end
    rowptr[m + 1] = total
    colind = Vector{Int}(undef, length(rows))
    aperm = Vector{Int}(undef, length(rows))
    pos = copy(rowptr)
    # Columns visited in ascending order is what leaves `colind` ascending within each row,
    # which is what lets the pair loop take `q >= p` as the upper triangle.
    for j in 1:n
        for k in nzrange(A, j)
            p = pos[rows[k]]
            colind[p] = j
            aperm[p] = k
            pos[rows[k]] = p + 1
        end
    end
    return (rowptr, colind, aperm)
end

"""
    reduced_nnz(P, A, n) -> Int

How many entries the upper triangle of the reduced matrix stores, counted from the patterns
of `P` and `A` without building it.

This is what the fill gate needs, and forming the matrix to ask is the expensive way round:
on the benchmark suite this counts in 10–92 µs where forming it through sparse products took
42–247, and on Huber it is 21.7 µs against 128.2.

Column `j` of `AᵀA` holds a row for every column any row of `A` shares with `j`, so one pass
over `A`'s columns and the rows behind them enumerates the column, and `mark` deduplicates it
in `O(1)` per candidate. `P`'s upper triangle and the diagonal `σ` lands on complete it.

Structural, so it agrees with [`reduced_gram`](@ref) exactly. Counting the formed matrix
instead would undercount: a sparse product drops an entry that cancels to zero, and whether
one cancels depends on `ρ`.
"""
function reduced_nnz(P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer, limit::Integer = typemax(Int))
    rowptr, colind, _ = csr_order(A)
    arows, prows = rowvals(A), rowvals(P)
    mark = zeros(Int, n)
    total = 0
    @inbounds for j in 1:n
        for p in nzrange(A, j)
            k = arows[p]
            for t in rowptr[k]:(rowptr[k + 1] - 1)
                i = colind[t]
                i <= j || continue
                if mark[i] != j
                    mark[i] = j
                    total += 1
                end
            end
        end
        for t in nzrange(P, j)
            i = prows[t]
            i <= j || continue
            if mark[i] != j
                mark[i] = j
                total += 1
            end
        end
        if mark[j] != j
            mark[j] = j
            total += 1
        end
        # The caller compares against a limit, so counting past it answers a question nobody
        # asked. A problem this backend refuses is exactly the one with the most to count:
        # on the benchmark suite's Control class the count reaches the limit after a fraction of
        # the columns, and stopping there is the difference between 92 µs and a few.
        total > limit && return total
    end
    return total
end

"""
Build the reduced matrix's pattern and the slot map that refills it.

Every array is sized before it is filled: the number of contributions is
`Σᵢ nnzᵢ(nnzᵢ + 1) / 2` over `A`'s rows, plus `P`'s upper triangle, plus the `n` diagonal
entries `σ` lands on. Growing them instead would dominate, because for a problem this
backend goes on to refuse there can be an entry per pair of columns sharing a row.
"""
function reduced_gram(::Type{T}, P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer) where {T}
    rowptr, colind, aperm = csr_order(A)
    m = length(rowptr) - 1
    npair = 0
    for i in 1:m
        k = rowptr[i + 1] - rowptr[i]
        npair += k * (k + 1) ÷ 2
    end
    prows = rowvals(P)
    nup = 0
    for j in 1:n, k in nzrange(P, j)
        nup += prows[k] <= j
    end
    N = npair + nup + n
    crow = Vector{Int}(undef, N)
    ccol = Vector{Int}(undef, N)
    t = 0
    for i in 1:m
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop, q in p:stop
            t += 1
            crow[t] = colind[p]
            ccol[t] = colind[q]
        end
    end
    prow = Vector{Int}(undef, nup)
    pcol = Vector{Int}(undef, nup)
    pperm = Vector{Int}(undef, nup)
    s = 0
    for j in 1:n, k in nzrange(P, j)
        i = prows[k]
        i <= j || continue
        s += 1
        prow[s] = i
        pcol[s] = j
        pperm[s] = k
        t += 1
        crow[t] = i
        ccol[t] = j
    end
    for j in 1:n
        t += 1
        crow[t] = j
        ccol[t] = j
    end
    R, slot = pattern_from(T, crow, ccol, n)
    return ReducedGram{T}(
        R, copy(A.colptr), copy(rowvals(A)), copy(P.colptr), copy(prows),
        rowptr, colind, aperm, slot[1:npair],
        prow, pcol, pperm, slot[(npair + 1):(npair + nup)],
        slot[(npair + nup + 1):end],
    )
end

"""
    pattern_from(T, crow, ccol, n) -> (R, slot)

An `n×n` sparse matrix holding each `(crow[t], ccol[t])` once, and for each `t` the index
into `nonzeros(R)` where that contribution accumulates.

CHOLMOD requires row indices ascending within a column, so the contributions are ordered by
row and then, stably, by column. Both passes are counting sorts over keys already known to
lie in `1:n`, which is `O(N + n)` — where a comparison sort per column costs `O(N log N)` and
measured 8× the sparse products this map replaces.
"""
function pattern_from(::Type{T}, crow::Vector{Int}, ccol::Vector{Int}, n::Integer) where {T}
    N = length(crow)
    order = counting_order(ccol, n, counting_order(crow, n, 1:N))
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, N)
    slot = Vector{Int}(undef, N)
    nz = 0
    p = 1
    for j in 1:n
        colptr[j] = nz + 1
        last_row = 0
        while p <= N && ccol[order[p]] == j
            t = order[p]
            i = crow[t]
            if i != last_row
                nz += 1
                rowval[nz] = i
                last_row = i
            end
            slot[t] = nz
            p += 1
        end
    end
    colptr[n + 1] = nz + 1
    resize!(rowval, nz)
    return (SparseMatrixCSC(n, n, colptr, rowval, zeros(T, nz)), slot)
end


"""
    counting_order(key, n, idx) -> Vector{Int}

The indices in `idx` reordered by `key`, stably, for keys in `1:n`.

Stability is what lets two passes sort by a pair of keys: ordering by row first and by column
second leaves the contributions in column-major order with rows ascending.
"""
function counting_order(key::Vector{Int}, n::Integer, idx)
    counts = zeros(Int, n + 1)
    for t in idx
        counts[key[t] + 1] += 1
    end
    counts[1] = 1
    for j in 1:n
        counts[j + 1] += counts[j]
    end
    out = Vector{Int}(undef, length(idx))
    for t in idx
        k = key[t]
        out[counts[k]] = t
        counts[k] += 1
    end
    return out
end

"Whether `g`'s slots still describe these matrices, which they do unless a pattern changed."
function describes(g::ReducedGram, P::SparseMatrixCSC, A::SparseMatrixCSC)
    return g.acolptr == A.colptr && g.arowval == rowvals(A) &&
        g.pcolptr == P.colptr && g.prowval == rowvals(P)
end

"""
    refill!(g, P, A, rho, E, D, c, sigma) -> SparseMatrixCSC

Rebuild `g.R` for the current data, in one pass over the recorded slots and without
allocating.
"""
function refill!(
        g::ReducedGram{T}, P::SparseMatrixCSC, A::SparseMatrixCSC, rho, E, D, c, sigma
    ) where {T}
    nz = nonzeros(g.R)
    fill!(nz, zero(T))
    avals = nonzeros(A)
    rowptr, colind, aperm, aslot = g.rowptr, g.colind, g.aperm, g.aslot
    t = 0
    for i in eachindex(rho, E)
        ei = E[i]
        w = rho[i] * ei * ei
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop
            wvj = w * T(avals[aperm[p]]) * D[colind[p]]
            for q in p:stop
                t += 1
                nz[aslot[t]] += wvj * T(avals[aperm[q]]) * D[colind[q]]
            end
        end
    end
    pvals = nonzeros(P)
    for k in eachindex(g.pslot)
        nz[g.pslot[k]] += c * D[g.prow[k]] * T(pvals[g.pperm[k]]) * D[g.pcol[k]]
    end
    for k in eachindex(g.dslot)
        nz[g.dslot[k]] += sigma
    end
    return g.R
end

function PureQPBase.factorize!(ls::SparseCholmod{T}, prob, wt)::Bool where {T}
    P, A = prob.P, prob.A
    if !describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries somewhere else, so every slot
        # the map holds is stale.
        ls.gram = reduced_gram(T, P, A, prob.n)
        R = refill!(ls.gram, P, A, wt.w, prob.E, prob.D, prob.c, wt.sigma)
        ls.fact = cholesky(Symmetric(R, :U); check = false)
        # The ordering belongs to the symbolic analysis, so it is reread exactly when a new
        # one is done.
        ls.perm = ls.fact.p::Vector{Int}
    else
        # The pattern is unchanged, so the symbolic factorization still describes it and
        # only the values need redoing. This is the case every time `ρ` moves.
        R = refill!(ls.gram, P, A, wt.w, prob.E, prob.D, prob.c, wt.sigma)
        cholesky!(ls.fact, Symmetric(R, :U); check = false)
    end
    issuccess(ls.fact) || return false
    # `choose_backend` selects this backend only after checking that CHOLMOD produces an
    # `L Lᵀ` for this pattern, and the pattern is what decides it, so that holds for the
    # workspace's life; `factor_csc!` refuses anything else rather than reading `D` as ones.
    ls.L = factor_csc!(ls.L, ls.fact, prob.n, true)
    transpose!(ls.Lt, ls.L)
    check_factor(ls.Lt, prob.n)
    return true
end

"""
    check_factor(L, N)

Establish that every index the substitutions will use is in range, or throw.

The substitutions index `x` by a row index read out of `L`, which no compiler can prove is
in bounds, so they are checked once here instead of on every one of the `nnz(L)` accesses —
[`unit_forward!`](@ref) and [`unit_backward!`](@ref) then run unchecked. That is worth 1.18×
to 1.32× on the benchmark suite's factors, which hold two to three nonzeros per column, where the
check is a large fraction of the work done per entry.

Called once per factorization, over `nnz(L)` entries, against a factorization that costs far
more; the guard is not on the per-iteration path.

It throws rather than returning `false` because an out-of-range index is a broken
factorization, not an unfactorizable matrix — a distinction `factorize!`'s `Bool` cannot
carry, and one a caller can do nothing about.
"""
function check_factor(L::SparseMatrixCSC, N::Integer)
    colptr, rows = L.colptr, rowvals(L)
    nz = length(rows)
    (length(colptr) > N && colptr[1] == 1 && colptr[N + 1] == nz + 1) || throw(
        ArgumentError(
            "factor has a malformed column pointer for an order-$N system: " *
                "colptr spans $(colptr[1]):$(colptr[min(N + 1, length(colptr))]) over $nz stored entries"
        )
    )
    for j in 1:N
        colptr[j] <= colptr[j + 1] || throw(
            ArgumentError("factor's column pointer decreases at column $j")
        )
    end
    for p in 1:nz
        1 <= rows[p] <= N || throw(
            ArgumentError("factor stores row index $(rows[p]) at position $p, outside 1:$N")
        )
    end
    return nothing
end

"""
    factor_csc!(L, F, N, ll) -> SparseMatrixCSC

`F`'s numeric factor as an order-`N` `SparseMatrixCSC`, written into `L`'s own buffers.

A simplicial CHOLMOD factor already holds its values column by column: column `j` stores
`nz[j]` entries from `p[j]` onwards, with row indices ascending and the diagonal first.
The columns sit in no particular order and carry slack between them, so the arrays are not a
column pointer and a row vector as they stand, but gathering them into one is a single pass
over the stored entries and reuses the buffers `L` already owns. `sparse(F.LD)` answers the
same question by copying the whole factor inside CHOLMOD, converting the copy, and
allocating three fresh arrays from it; a refactorization happens every outer iteration, and
on the benchmark suite's smallest Random QP going through CHOLMOD costs `factorize!` 5.46 µs
against 3.43 µs here.

`ll` is the form the caller's substitutions read — `true` for `L Lᵀ`, `false` for the `LD` of
an `L D Lᵀ`, which packs `D` where `L Lᵀ` keeps ones. A factor of the other form throws,
because the two put different numbers in the same places.

Nothing here assumes the pattern is unchanged: the column pointer and the row indices are
rebuilt from what `F` currently holds, and the buffers are resized when the count moves. Row
indices are bounded as [`check_factor`](@ref) bounds them and the column pointer is
increasing by construction, which is what the substitutions need to run unchecked.
"""
function factor_csc!(L::SparseMatrixCSC{T, Int}, F, N::Integer, ll::Bool) where {T}
    s = unsafe_load(CHOLMOD.typedpointer(F))
    is_ll = !iszero(s.is_ll)
    is_ll == ll || throw(
        ArgumentError(
            "CHOLMOD returned an $(is_ll ? "L Lᵀ" : "L D Lᵀ") factorization " *
                "where the backend reads an $(ll ? "L Lᵀ" : "L D Lᵀ") one"
        )
    )
    if !iszero(s.is_super) || s.nz == C_NULL
        # A supernodal factor stores its values by supernode rather than by column, so there
        # is no column-wise pattern to read and CHOLMOD converts a copy of the factor.
        # Supernodal factors are `L Lᵀ`, and `ldlt` asks for a simplicial factorization, so
        # only the `L Lᵀ` backend reaches this.
        G = sparse(F.L)::SparseMatrixCSC{T, Int}
        check_factor(G, N)
        return G
    end
    n = Int(s.n)
    n == N || throw(ArgumentError("factor is order $n for an order-$N system"))
    colstart = unsafe_wrap(Array, s.p, (n + 1,); own = false)
    colcount = unsafe_wrap(Array, s.nz, (n,); own = false)
    rows = unsafe_wrap(Array, s.i, (Int(s.nzmax),); own = false)
    vals = unsafe_wrap(Array, Ptr{T}(s.x), (Int(s.nzmax),); own = false)
    total = 0
    for j in 1:n
        total += Int(colcount[j])
    end
    colptr, rowval, nzval = L.colptr, rowvals(L), nonzeros(L)
    if length(rowval) != total
        resize!(rowval, total)
        resize!(nzval, total)
    end
    t = 1
    for j in 1:n
        colptr[j] = t
        base = Int(colstart[j])
        for k in 1:Int(colcount[j])
            i = Int(rows[base + k]) + 1
            1 <= i <= N || throw(
                ArgumentError("factor stores row index $i in column $j, outside 1:$N")
            )
            rowval[t] = i
            nzval[t] = vals[base + k]
            t += 1
        end
    end
    colptr[n + 1] = t
    return L
end

"""
    ldl_dinv!(dinv, L, N) -> Bool

Write `1 / D[j]` into `dinv` for each column of CHOLMOD's packed `LD`, or `false` at the
first column whose pivot is missing or zero.

`D[j]` is the first value stored in column `j`: row indices ascend within a column and `L` is
lower triangular, so the diagonal entry is the one at `colptr[j]` when it is stored at all.
"""
function ldl_dinv!(dinv::Vector{T}, L::SparseMatrixCSC{T, Int}, N::Integer) where {T}
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    for j in 1:N
        k = colptr[j]
        (k < colptr[j + 1] && rows[k] == j) || return false
        d = vals[k]
        iszero(d) && return false
        dinv[j] = inv(d)
    end
    return true
end

"""
    llt_forward!(x, L, N)
    llt_backward!(x, Lt, N)

Substitution against the two factors of an `L Lᵀ`, in place.

CHOLMOD stores each column's diagonal entry first, so [`llt_forward!`](@ref) takes `L[j,j]`
from `nonzeros(L)[colptr[j]]` and runs the off-diagonal entries from one past it.

[`llt_backward!`](@ref) takes `Lᵀ` rather than `L`. Against `L` the loop has to gather —
each column accumulates a dot product into a scalar, which is a serial dependency — where
against `Lᵀ` it scatters, exactly as the forward solve does. On the benchmark suite's Huber
factor, 5199 nonzeros over 1806 columns, that is 5.12 µs against 3.88 µs. Column `j` of
`Lᵀ` is row `j` of `L`, so the diagonal is its *last* entry.

Written out rather than left to `ldiv!(LowerTriangular(L), x)` for the same reason as
[`ldl_forward!`](@ref), and the reason is the factor's shape rather than the wrapper.
Measured on the benchmark suite's own factors, resetting the vector every sample because these
solves are in place: at 2.2 nonzeros per column (Lasso) the pair costs 5.00 µs through
`ldiv!` and 2.91 µs here, and at 2.9 (Huber) 11.59 µs against 9.21 µs. On a factor with 9
nonzeros per column the ordering reverses and `ldiv!` is the faster of the two — at that
density the arithmetic dominates, where here the per-column bookkeeping does, and the generic
path carries more of it.
"""
function llt_forward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in 1:N
        top = colptr[j]
        xj = x[j] / vals[top]
        x[j] = xj
        for p in (top + 1):(colptr[j + 1] - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function llt_backward!(x::AbstractVector, Lt::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = Lt.colptr, rowvals(Lt), nonzeros(Lt)
    @inbounds for j in N:-1:1
        bot = colptr[j + 1] - 1
        xj = x[j] / vals[bot]
        x[j] = xj
        for p in colptr[j]:(bot - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function PureQPBase.solve_system!(ls::SparseCholmod{T}, prob, wt, rhs_x, rhs_z, x, z)::Nothing where {T}
    rhs = PureQPBase.reduced_rhs!(prob, wt, rhs_x, rhs_z)
    perm, work, n = ls.perm, ls.permuted, prob.n
    # R[perm, perm] = L Lᵀ, so the solve is a permutation, two triangular solves, and the
    # inverse permutation -- all over buffers this backend owns.
    for i in 1:n
        work[i] = rhs[perm[i]]
    end
    llt_forward!(work, ls.L, n)
    llt_backward!(work, ls.Lt, n)
    for i in 1:n
        x[perm[i]] = work[i]
    end
    prob.m > 0 && PureQPBase.mul_A!(z, prob, x)
    return nothing
end

"""
    cholmod_backend(P, A, prob, wt) -> SparseCholmod or nothing

Form the reduced matrix from the equilibrated data and factor it sparsely.

Whether this form suits the pair is [`sparse_form`](@ref)'s question and is already settled
before this is called, so the factor produced here is the one the solver goes on to solve
against and its symbolic part is what every later refactorization reuses. `nothing` means the
representation does not match or the matrix does not factor at this regularization.
"""
function cholmod_backend(
        P, A, prob::PureQPBase.Problem{T}, wt::PureQPBase.SystemWeights{T}
    ) where {T <: Real}
    n = prob.n
    proto = prob.q0
    # As in `factored_kkt_backend`: `reduced_gram` and `refill!` need `SparseMatrixCSC`'s
    # stored columns, which a `Symmetric` wrapper over one does not present.
    (P isa SparseMatrixCSC && n > 0) || return nothing
    gram = reduced_gram(T, P, A, n)
    R = refill!(gram, P, A, wt.w, prob.E, prob.D, prob.c, wt.sigma)
    # A pure-Julia LDLᵀ, if one is loaded, factors this faster than CHOLMOD does and hands
    # back `L` and `D` as plain arrays, so nothing has to be extracted from a foreign factor.
    alt = PureQPBase.ldl_backend(gram, proto, n)
    isnothing(alt) || return alt
    F = cholesky(Symmetric(R, :U); check = false)
    issuccess(F) || return nothing
    # An LDLᵀ factorization has no `F.L`, and this backend's solve assumes `L Lᵀ`.
    L = try
        sparse(F.L)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
    Lt = SparseMatrixCSC(transpose(L))
    check_factor(L, n)
    check_factor(Lt, n)
    return SparseCholmod{T, typeof(proto), typeof(F)}(
        gram, F, L, Lt, F.p, similar(proto, T, n)
    )
end

# No `trim_compat` claim: the solve reaches CHOLMOD through `ccall`, and the trim entry
# points cover the dense path.
@verify SparseCholmod
@verify SparseKKT


"Whether every stored entry of `P` lies on the diagonal, in one pass over its columns."
function is_diagonal(P::SparseMatrixCSC)
    rows = rowvals(P)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            rows[k] == j || return false
        end
    end
    return true
end

"""
    check_finite(M::SparseMatrixCSC, rows, cols, name)

Check the stored entries only. The generic method reads every `(i, j)` that
[`PureQPBase.structural_rows`](@ref) names, which for a `SparseMatrixCSC` is every row, and each
read is a search through the column: `m × n` searches for a matrix with `nnz` entries.
"""
function PureQPBase.check_finite(M::SparseMatrixCSC, rows::Integer, cols::Integer, name::String)
    rv, nz = rowvals(M), nonzeros(M)
    for j in 1:cols, k in nzrange(M, j)
        isfinite(nz[k]) || throw(ArgumentError("$name is not finite at entry ($(rv[k]), $j)"))
    end
    return nothing
end

"""
    is_convex(T, P::SparseMatrixCSC, sigma) -> Bool

The convexity test without densifying `P`. `cholesky` on a sparse matrix is CHOLMOD, which
costs `O(nnz(L))` where the generic dense test costs `O(n³)` — 0.24 ms against 22.6 ms on a
tridiagonal `P` at `n = 2000`, which was half of `setup` on a banded problem. A diagonal `P`
skips the factorization entirely.
"""
function PureQPBase.is_convex(::Type{T}, P::SparseMatrixCSC, sigma) where {T}
    isempty(P) && return true
    # A diagonal `P` needs no factorization: `P + σI` is diagonal, so it is positive definite
    # exactly when every entry clears `-σ`. This is not a corner case — an epigraph
    # reformulation leaves the objective diagonal, which is what four of the benchmark suite's
    # seven classes look like.
    if is_diagonal(P)
        vals = nonzeros(P)
        for k in eachindex(vals)
            T(vals[k]) + sigma > zero(T) || return false
        end
        return true
    end
    # An `LDLᵀ` from outside SuiteSparse answers this when one is available, and dispatch
    # settles which it is, so the CHOLMOD factorization below is not merely unused then but
    # unreachable — which is what keeps a sparse `P` inside `juliac --trim`. The bindings
    # reached from `cholesky` are not resolvable statically, so leaving them on a live branch
    # would cost the guarantee whether or not they ever run.
    answer = PureQPBase.ldl_posdef(P, sigma)
    isnothing(answer) || return answer
    return issuccess(cholesky(Symmetric(SparseMatrixCSC{T, Int}(P) + sigma * I); check = false))
end

end # module PureQPBaseSparseArraysExt
