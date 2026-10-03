@inline function limit_scaling(v::T) where {T}
    return v < MIN_SCALING(T) ? one(T) : min(v, MAX_SCALING(T))
end

"How close to one the equilibration updates must be for [`equilibrate!`](@ref) to stop
sweeping early. Below the resolution of a type's arithmetic the test never passes and the
sweep count is exactly the requested one."
@inline RUIZ_TOL(::Type{T}) where {T} = T(1.0e-12)

"""
    check_finite(M, rows, cols, name) -> nothing

Throw unless every entry `M` can be read is finite, naming the first that is not.

Walks [`structural_rows`](@ref) column by column, so a structured representation pays only
for its own entries. A `SparseMatrixCSC` has its own method in the SparseArrays extension that
reads only the stored entries. A stray `NaN` or `Inf` would not be reliably reported by a
factorization run with `check = false`, which is why this runs first. A representation that
cannot be indexed is the caller's to skip.
"""
function check_finite(M, rows::Integer, cols::Integer, name::String)
    for j in 1:cols
        for i in structural_rows(M, j)
            v = M[i, j]
            isfinite(v) || throw(ArgumentError("$name is not finite at entry ($i, $j)"))
        end
    end
    return nothing
end

"""
    check_finite(M::KroneckerOperator, rows, cols, name)

Check the two factors rather than the product they stand for.

`A₁ ⊗ A₂` has `size(A₁) .* size(A₂)` entries and stores only the factors; walking the
product would read every one of them through the operator's `getindex`, which costs more
than the whole of [`setup`](@ref) and is the work the type exists to avoid. An entry of the
product is a product of one entry from each factor, so it is finite exactly when both are.
"""
function check_finite(M::KroneckerOperator, rows::Integer, cols::Integer, name::String)
    # The entry reported is the factor's own, so the message names the factor.
    check_finite(M.A1, size(M.A1, 1), size(M.A1, 2), name * "'s first Kronecker factor")
    check_finite(M.A2, size(M.A2, 1), size(M.A2, 2), name * "'s second Kronecker factor")
    return nothing
end

# ── column traversals ───────────────────────────────────────────────────────────────────
# Equilibration and the dense formation both walk the caller's matrices one column at a
# time. These four functions are the only places that do, so a matrix type that can
# enumerate a column faster than by index needs to override just them, or just
# `structural_rows`. `PureQPBase/ext/PureQPBaseSparseArraysExt.jl` overrides all four for
# `SparseMatrixCSC`, where indexing `M[i, j]` is a binary search and the generic loop visits
# every structural zero.
#
# All four skip entries the matrix cannot have, which assumes `f(0, i) == 0` for the
# functions passed to `scaled_col!` and `add_scaled_col!`, and that the running maxima start
# at zero. Both hold for every caller here, and the sparse overrides already depend on it.

"""
    structural_rows(M, j) -> iterable of row indices

The rows in which column `j` of `M` can hold a nonzero, in increasing order and without
repeats.

Every row, in general. A banded matrix has a much shorter answer, and taking the generic one
is what makes a `Diagonal` `P` cost `O(n)` per column rather than `O(1)` — enough on its own
to make structured storage slower than dense.

The iterator's state must be concretely typed: this runs on the equilibration path, where a
dynamic dispatch costs an order of magnitude and fails `--trim`.

A range for every representation whose nonzeros are contiguous down a column, which is most
of them; [`RowCoupled`](@ref) is the exception, since its dense rows sit above scattered
single-entry ones.
"""
@inline structural_rows(M::AbstractMatrix, j::Integer) = axes(M, 1)
@inline structural_rows(M::Diagonal, j::Integer) = j:j
@inline function structural_rows(M::Bidiagonal, j::Integer)
    lo, hi = firstindex(M, 1), lastindex(M, 1)
    return M.uplo == 'U' ? (max(lo, j - 1):min(hi, j)) : (max(lo, j):min(hi, j + 1))
end
@inline function structural_rows(M::Union{Tridiagonal, SymTridiagonal}, j::Integer)
    lo, hi = firstindex(M, 1), lastindex(M, 1)
    return max(lo, j - 1):min(hi, j + 1)
end

"`max(w[i] * |M[i, j]|)` over the column."
@inline function weighted_colmax(::Type{T}, M::AbstractMatrix, j::Integer, w::AbstractVector) where {T}
    r = zero(T)
    for i in structural_rows(M, j)
        r = max(r, w[i] * abs(T(M[i, j])))
    end
    return r
end

"""
    weighted_colmax_rowmax!(T, e, M, j, w, s) -> max(w[i] * |M[i,j]|)

The column maximum, and at the same time `e[i] = max(e[i], s * |M[i,j]|)` — the row norms
accumulated in the same pass rather than gathered by a second traversal.
"""
@inline function weighted_colmax_rowmax!(
        ::Type{T}, e::AbstractVector, M::AbstractMatrix, j::Integer,
        w::AbstractVector, s
    ) where {T}
    r = zero(T)
    for i in structural_rows(M, j)
        v = abs(T(M[i, j]))
        r = max(r, w[i] * v)
        e[i] = max(e[i], s * v)
    end
    return r
end

"`dest[i, j] = f(M[i, j], i)` over the column, with `dest` already zeroed."
@inline function scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::AbstractMatrix, j::Integer, f::F
    ) where {T, F}
    for i in structural_rows(M, j)
        dest[i, j] = f(T(M[i, j]), i)
    end
    return dest
end

"`dest[i, j] += f(M[i, j], i)` over the column."
@inline function add_scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::AbstractMatrix, j::Integer, f::F
    ) where {T, F}
    for i in structural_rows(M, j)
        dest[i, j] += f(T(M[i, j]), i)
    end
    return dest
end

# The same column, read from the triangle `S` names. Indexing `S` costs a branch per entry and
# reads the half outside the triangle at stride `size(S, 1)`; splitting the column at the
# diagonal gives two branch-free runs, the longer of which is contiguous in the parent. The
# parent is an argument rather than a field read inside the loop, which is what lets the runs
# vectorize.
@inline add_scaled_col!(
    ::Type{T}, dest::AbstractMatrix, S::SymmetricFactorable, j::Integer, f::F
) where {T, F} = add_scaled_col_triangle!(T, dest, parent(S), j, f, S.uplo == 'U')

@inline function add_scaled_col_triangle!(
        ::Type{T}, dest::AbstractMatrix, A::AbstractMatrix, j::Integer, f::F, upper::Bool
    ) where {T, F}
    n = size(A, 1)
    if upper
        for i in 1:j
            dest[i, j] += f(T(A[i, j]), i)
        end
        for i in (j + 1):n
            dest[i, j] += f(T(A[j, i]), i)
        end
    else
        for i in 1:(j - 1)
            dest[i, j] += f(T(A[j, i]), i)
        end
        for i in j:n
            dest[i, j] += f(T(A[i, j]), i)
        end
    end
    return dest
end

"""
    column_norms!(d, e, T, pcol, A, D, E, c)

One Ruiz sweep's column measurements: `d[j]` becomes
`max(c·D[j]·pcol[j], D[j]·‖E ⊙ A[:,j]‖∞)` and `e` accumulates the row norms of the scaled
`A` in the same pass.

`pcol` holds `‖D ⊙ P[:,j]‖∞` for the current `D`, computed by [`cost_norms!`](@ref) at the
end of the previous sweep. `D` does not change between the two, so recomputing it here would
be a second full pass over `P` for the same numbers.

The row norms are accumulated here rather than gathered in a second loop over `A[i, j]` with
`j` innermost: that walks a column-major matrix across its rows, and on a 400×200 problem
the strided reads cost more than everything else in setup put together.
"""
function column_norms!(d, e, ::Type{T}, pcol, A, D, E, c) where {T}
    fill!(e, zero(T))
    for j in eachindex(d)
        dj = D[j]
        aj = weighted_colmax_rowmax!(T, e, A, j, E, dj)
        d[j] = limit_scaling(max(c * dj * pcol[j], dj * aj))
    end
    return d
end

"""
    cost_norms!(pcol, T, P, D, c, n) -> mean column norm

Fill `pcol[j]` with `‖D ⊙ P[:,j]‖∞` and return their `c`-weighted mean, which is what the
cost normalization compares against `‖q̃‖∞`.

One pass serves two purposes: the mean, and the column norms the next sweep needs.

A representation that cannot be indexed overrides this with whole-matrix reductions — see
`PureQPBase/ext/PureQPBaseGPUArraysCoreExt.jl`.
"""
function cost_norms!(pcol, ::Type{T}, P, D, c, n) where {T}
    acc = zero(T)
    for j in 1:n
        pj = weighted_colmax(T, P, j, D)
        pcol[j] = pj
        acc += c * D[j] * pj
    end
    return acc / n
end

"""
    cost_norms!(pcol, T, P::SymmetricFactorable, D, c, n) -> mean

The same column norms, read from the triangle `P` names.

Indexing a `Symmetric` column by column would read the half outside its triangle as
`parent[j, i]` with `i` advancing — across the rows of a column-major array, one cache line per
element, behind a branch that stops the reduction vectorizing. Reading the stored triangle
instead visits each entry once, contiguously: entry `(i, j)` is the matrix's `(i, j)` and its
`(j, i)`, so it bounds column `j` by `D[i]·|v|` and column `i` by `D[j]·|v|`, and one pass over
half the entries fills every column's norm.

`pcol` is accumulated rather than assigned, so it is zeroed first: a column's norm is complete
only once every column holding one of its mirrored entries has been visited.
"""
function cost_norms!(pcol, ::Type{T}, P::SymmetricFactorable, D, c, n) where {T}
    # The parent is passed to the kernel rather than read from `P` inside it. A matrix loaded
    # from a struct field reaches the loop without the `nonnull` and alignment facts a matrix
    # that arrives as an argument carries, and the reduction then does not vectorize: measured
    # at n = 500, 0.475 ms against 0.058 ms for the same loop over the same entries.
    triangle_colmax!(pcol, T, parent(P), D, n, P.uplo == 'U')
    acc = zero(T)
    for j in 1:n
        acc += c * D[j] * pcol[j]
    end
    return acc / n
end

"""
    triangle_colmax!(pcol, T, A, D, n, upper) -> pcol

`pcol[j] = max(D[i] * |M[i, j]|)` for the symmetric `M` whose `upper` triangle (or lower) is
stored in `A`, reading each stored entry once.

Entry `(i, j)` of `A` is both `M[i, j]` and `M[j, i]`, so it bounds column `j` by `D[i]·|v|`
and column `i` by `D[j]·|v|`. `pcol` is accumulated, so it is zeroed first: a column's norm is
complete only once every column holding one of its mirrored entries has been visited.
"""
function triangle_colmax!(
        pcol::AbstractVector{T}, ::Type{T}, A::AbstractMatrix, D::AbstractVector, n, upper::Bool
    ) where {T}
    fill!(pcol, zero(T))
    # The stored part of column `j` is rows `1:j` of an upper parent and `j:n` of a lower one.
    # The two cases are separate loops so the test is not repeated for every column.
    if upper
        for j in 1:n
            triangle_colmax_column!(pcol, T, A, D, j, 1:j)
        end
    else
        for j in 1:n
            triangle_colmax_column!(pcol, T, A, D, j, j:n)
        end
    end
    return pcol
end

@inline function triangle_colmax_column!(
        pcol::AbstractVector{T}, ::Type{T}, A::AbstractMatrix, D::AbstractVector,
        j::Integer, rows::UnitRange{Int}
    ) where {T}
    dj = D[j]
    pj = zero(T)
    for i in rows
        v = abs(T(A[i, j]))
        pj = max(pj, D[i] * v)
        # At `i == j` this repeats the line above with the same value, which `max` absorbs.
        pcol[i] = max(pcol[i], dj * v)
    end
    pcol[j] = max(pcol[j], pj)
    return pcol
end

"""
    equilibrate!(T, P, A, q0, l0, u0, q, l, u, D, E, d, e, pcol, n, sweeps) -> c

Run modified Ruiz equilibration: `D` and `E` become the column and row factors, `q`, `l` and
`u` the scaled data, and the returned `c` the cost factor.

Takes the arrays rather than a [`QPWorkspace`](@ref) because the factors must exist before one
does. The backend is part of the workspace's type, so it is chosen first, and it is chosen by
building and factoring the reduced matrix the solver will actually use — which needs `D`, `E`
and `c`. `d`, `e` and `pcol` are scratch of length `n`, `m` and `n`.
"""
function equilibrate!(
        ::Type{T}, P, A, q0, l0, u0, q, l, u, D, E, d, e, pcol, n, sweeps
    ) where {T}
    fill!(D, one(T))
    fill!(E, one(T))
    c = one(T)
    copyto!(q, q0)
    copyto!(l, l0)
    copyto!(u, u0)
    if sweeps <= 0
        return c
    end
    # Seeds `pcol` for the first sweep; every later one gets it from the cost normalization
    # at the end of the sweep before, which reads `P` with the same `D`.
    cost_norms!(pcol, T, P, D, c, n)
    for sweep in 1:sweeps
        column_norms!(d, e, T, pcol, A, D, E, c)
        e .= limit_scaling.(E .* e)
        d .= inv.(sqrt.(d))
        e .= inv.(sqrt.(e))
        D .*= d
        E .*= e
        q .*= d
        # Cost normalization: average column ∞-norm of the scaled P, against ‖q̃‖∞.
        ct = max(
            cost_norms!(pcol, T, P, D, c, n),
            limit_scaling(maximum(abs, q; init = zero(T))),
        )
        ct = inv(limit_scaling(ct))
        q .*= ct
        c *= ct
        # Early exit: once the multiplicative updates and the cost factor all sit within
        # RUIZ_TOL of one, the sweep has reached the fixed point and the remaining sweeps
        # would move `D`, `E` and `c` by O(tol) at most. The factors then differ from the
        # full-sweep result by less than that, which is what keeps the iterates inside the
        # C-suite tolerance. For a type whose arithmetic cannot resolve the tolerance the
        # test never passes and the sweep count is exactly the requested one.
        moved = max(
            maximum(abs, d .- one(T); init = zero(T)),
            maximum(abs, e .- one(T); init = zero(T)),
            abs(ct - one(T)),
        )
        sweep < sweeps && moved <= RUIZ_TOL(T) && break
    end
    l .= E .* l0
    u .= E .* u0
    return c
end


"""
    mul_A!(out, prob, x)

`out = Ã x = E ⊙ (A (D ⊙ x))`, using the caller's `A` unchanged.

`out` must not alias `prob.tmp_n`, which is used as scratch.
"""
function mul_A!(out::AbstractVector{T}, prob::Problem{T}, x::AbstractVector{T}) where {T}
    multiply!(prob.tmp_n, prob.D, x)
    mul!(out, prob.A, prob.tmp_n)
    # `scale_by!` rather than `out .*= prob.E`: an in-place broadcast has `out` on both
    # sides, which leaves an `unaliascopy` branch that AllocCheck reports as a possible
    # allocation even though it never fires. See PureQPBase/src/elementwise.jl.
    scale_by!(out, prob.E)
    return out
end

"""
    mul_At!(out, prob, y)

`out = Ãᵀ y = D ⊙ (Aᵀ(E ⊙ y))`, using the caller's `A` unchanged.

`out` must not alias `prob.tmp_m`, which is used as scratch.
"""
function mul_At!(out::AbstractVector{T}, prob::Problem{T}, y::AbstractVector{T}) where {T}
    multiply!(prob.tmp_m, prob.E, y)
    mul!(out, prob.A', prob.tmp_m)
    scale_by!(out, prob.D)
    return out
end

# `mul!` against a `Tridiagonal`'s adjoint allocates in LinearAlgebra, which on a path that
# runs every iteration is the difference between holding the no-allocation guarantee and
# losing it. The three bands give `Aᵀ t` directly: `(Aᵀt)[j] = d[j]t[j] + dl[j]t[j+1] +
# du[j-1]t[j-1]`.
function mul_At!(
        out::AbstractVector{T}, prob::Problem{T, <:AbstractMatrix, <:Tridiagonal},
        y::AbstractVector{T}
    ) where {T}
    multiply!(prob.tmp_m, prob.E, y)
    A, t, n = prob.A, prob.tmp_m, prob.n
    dl, d, du = A.dl, A.d, A.du
    # Summed in ascending `i`, which is the order `mul!` against the adjoint uses. Any other
    # order rounds differently, and a representation is supposed to change how the entries
    # are reached, not what comes out.
    for j in 1:n
        v = j > 1 ? du[j - 1] * t[j - 1] : zero(T)
        v += d[j] * t[j]
        j < n && (v += dl[j] * t[j + 1])
        out[j] = v
    end
    scale_by!(out, prob.D)
    return out
end

"""
    mul_P!(out, prob, x)

`out = P̃ x = c (D ⊙ (P (D ⊙ x)))`, using the caller's `P` unchanged.

`out` must not alias `prob.tmp_n`, which is used as scratch.
"""
function mul_P!(out::AbstractVector{T}, prob::Problem{T}, x::AbstractVector{T}) where {T}
    multiply!(prob.tmp_n, prob.D, x)
    mul!(out, prob.P, prob.tmp_n)
    scale_by!(out, prob.D, prob.c)
    return out
end

"""
    reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c)

The diagonal of the reduced matrix, `c·D[j]²·P[j,j] + σ + Σᵢ ρᵢ(E[i]·A[i,j]·D[j])²`.

The matrix-free backend preconditions with this, and it is the one thing that backend needs
from `P` and `A` other than their products. The row sum follows
[`structural_rows`](@ref)`(A, j)`, so a declared structure costs only its own entries; a
representation that cannot be indexed at all overrides this function with whole-matrix
reductions.
"""
reduced_diagonal!(dest, ::Type{T}, P, A, rho, E, D, sigma, c) where {T} =
    indexed_reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c)

"""
    indexed_reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c) -> dest

The reduced diagonal read from entries, which is what [`reduced_diagonal!`](@ref) does for any
representation that can be indexed.

Named separately so a representation that is sometimes indexable can reach it: a composition
whose parts all have entries falls back to this, where one holding an operator that supplies only
products has no diagonal to give.
"""
function indexed_reduced_diagonal!(dest, ::Type{T}, P, A, rho, E, D, sigma, c) where {T}
    # `rho` and `E` are workspace vectors indexed by the same `i` that indexes `A`'s rows, so
    # a representation whose rows are not counted from one would read the wrong weights.
    Base.require_one_based_indexing(rho, E)
    for j in eachindex(dest)
        dj = D[j]
        d = c * dj * T(P[j, j]) * dj + sigma
        for i in structural_rows(A, j)
            a = E[i] * T(A[i, j]) * dj
            d += rho[i] * a * a
        end
        dest[j] = inv(max(d, sqrt(eps(T))))
    end
    return dest
end
