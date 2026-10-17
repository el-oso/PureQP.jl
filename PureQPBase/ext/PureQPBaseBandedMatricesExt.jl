"""
The backend for a reduced matrix that is banded but wider than tridiagonal.

`bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`, so a `Tridiagonal` `A` already squares
past what `SymTridiagonal` can hold. BandedMatrices.jl stores any bandwidth, and its
`cholesky` is LAPACK's banded factorization: `O(n b²)` to factor, `O(n b)` to solve, against
the `O(n³)` and `O(n²)` the dense backend would spend on the same problem.

Loading BandedMatrices is what makes this reachable. Without it a banded problem is served by
[`PureQPBase.ReducedCholesky`](@ref) as before, correctly but densely.
"""
module PureQPBaseBandedMatricesExt

using PureQPBase
using BandedMatrices
using LinearAlgebra

using PureQPBase: LinearSystem, reduced_rhs!, mul_A!
using TypeContracts: TypeContracts, @verify

"""
    BandedReduced{T,M,F} <: LinearSystem

The reduced system stored as a `Symmetric` `BandedMatrix` and factored by a banded Cholesky.

`R` is the storage the reduced matrix is assembled into, and `cholesky!` then overwrites it
with the factor — so between factorizations `R` holds the factor, not the matrix. Every
refactorization refills the bands from scratch before factoring, which is why that is safe.

Unlike the tridiagonal backend's `ldlt`, a banded Cholesky reports indefiniteness through
`issuccess`, so [`PureQPBase.factorize!`](@ref) needs no separate test of the pivots.
"""
mutable struct BandedReduced{T <: Real, M, F} <: LinearSystem
    R::M
    fact::F
    bw::Int
end

"""
    banded_bandwidth(M) -> Int

The number of off-diagonals `M` can hold a nonzero in, for the representations this backend
accepts. It is a property of the type, not of the values, so a numerically zero band is
still stored — refusing to shrink it keeps the answer independent of the data.
"""
banded_bandwidth(::Diagonal) = 0
banded_bandwidth(::Bidiagonal) = 1
banded_bandwidth(::Tridiagonal) = 1
banded_bandwidth(::SymTridiagonal) = 1
banded_bandwidth(M::BandedMatrix) = max(bandwidth(M, 1), bandwidth(M, 2))
banded_bandwidth(M::Symmetric{<:Any, <:BandedMatrix}) = banded_bandwidth(parent(M))

# Equilibration and the dense formation walk a column at a time through `structural_rows`.
# Without these the generic method reports every row, so a matrix holding `O(nb)` entries is
# visited `O(n^2)` times per Ruiz sweep. Both derive their bounds from the bandwidths stated
# above rather than encoding them again.
@inline PureQPBase.structural_rows(M::BandedMatrix, j::Integer) =
    max(firstindex(M, 1), j - bandwidth(M, 2)):min(lastindex(M, 1), j + bandwidth(M, 1))

# The wrapper mirrors one triangle across the diagonal, so the rows it can hold a nonzero in
# span the wider of the parent's two bandwidths on both sides.
@inline function PureQPBase.structural_rows(M::Symmetric{<:Any, <:BandedMatrix}, j::Integer)
    b = banded_bandwidth(M)
    return max(firstindex(M, 1), j - b):min(lastindex(M, 1), j + b)
end

"""
    reduced_bandwidth(P, A) -> Int

`max(bandwidth(P), 2 bandwidth(A))` — the bandwidth of `c D P D + σI + Ãᵀ diag(ρ) Ã`.
Diagonal scaling preserves a bandwidth and squaring `A` doubles it.
"""
reduced_bandwidth(P, A) = max(banded_bandwidth(P), 2 * banded_bandwidth(A))

const BandedLike = Union{
    Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal,
    BandedMatrix, Symmetric{<:Any, <:BandedMatrix},
}

# The types PureQPBase itself has no backend for. The two methods below split so that neither
# overlaps the other, nor the `(Diagonal, Diagonal)`,
# `(SymTridiagonal|Tridiagonal, Diagonal)` and
# `(Diagonal|SymTridiagonal|Tridiagonal, Bidiagonal)` methods in `PureQPBase/src/linsys.jl`, whose
# reduced matrices are narrow enough for the LinearAlgebra backends. A `Tridiagonal` `A`
# squares past those, which is why it is a `WideBand`; a `Tridiagonal` `P` does not, so the
# second method's `P` position stops at the types BandedMatrices itself supplies.
const WideBand = Union{Tridiagonal, BandedMatrix, Symmetric{<:Any, <:BandedMatrix}}
const NarrowBand = Union{Diagonal, Bidiagonal}

PureQPBase.choose_backend(
    P::BandedLike, A::WideBand, prob, wt, sel::PureQPBase.SelectionFor
) = banded_backend(P, A, prob, sel)

PureQPBase.choose_backend(
    P::Union{BandedMatrix, Symmetric{<:Any, <:BandedMatrix}}, A::NarrowBand,
    prob, wt, sel::PureQPBase.SelectionFor
) = banded_backend(P, A, prob, sel)

function banded_backend(
        P, A, prob::PureQPBase.Problem{T}, sel::PureQPBase.SelectionFor
    ) where {T <: Real}
    n = prob.n
    b = reduced_bandwidth(P, A)
    # Below bandwidth 2 the LinearAlgebra backends already apply and are cheaper than a banded
    # factorization. Above `n/4` the dense path wins per iteration.
    #
    # Both halves of the choice favour the band below that. Its factorization is far cheaper —
    # 445x at `b = 0.01n`, still 4.4x at `b = 0.99n` — and so is its iteration, by 5.6x at
    # `b = 0.05n` and 1.5x at `b = 0.2n` for `n = 1000`, crossing 1 near `b = 0.3n` at every
    # size from 500 to 3000. The limit sits below that crossing so the accepted range wins on
    # the factorization and the iteration both, which keeps the choice from depending on how
    # many iterations a particular problem takes.
    #
    # Measure it with BLAS pinned to one thread, as every benchmark here does. A multithreaded
    # `symv` is some eightfold faster and moves the crossing below `b = 0.05n`, so a sweep run
    # under a different thread count answers a different question.
    #
    # `b <= n/4` also implies the band is the smaller representation — `(2b+1)n` against the
    # dense backend's `mn + n^2` — and implies `b < n - 1`, so neither needs testing separately.
    # See `PureOSQP/bench/gate_band_beyond.jl`.
    (b < 2 || 4b > n) && return PureQPBase.dense_rung(P, A, prob, sel)
    R = BandedMatrix{T}(undef, (n, n), (b, b))
    fill!(R.data, zero(T))
    for i in 1:n
        R[i, i] = one(T)
    end
    fact = cholesky(Symmetric(R))
    return (BandedReduced{T, typeof(R), typeof(fact)}(R, fact, b), false)
end

PureQPBase.backend_name(::BandedReduced) = :banded

# The factor keeps `R`'s lower bandwidth, so one triangle of an `n×n` matrix of bandwidth
# `bw` is `n(bw+1)` entries less the `bw(bw+1)/2` that run off the top-left corner.
function PureQPBase.backend_info(ls::BandedReduced)
    dim, bw = size(ls.R, 1), ls.bw
    return PureQPBase.BackendInfo(
        PureQPBase.backend_name(ls), true, :reduced, dim, dim * (bw + 1) - bw * (bw + 1) ÷ 2
    )
end

function PureQPBase.factorize!(ls::BandedReduced{T}, prob, wt)::Bool where {T}
    n, m, b = prob.n, prob.m, ls.bw
    P, A, D, E, c = prob.P, prob.A, prob.D, prob.E, prob.c
    rho, sigma = wt.w, wt.sigma
    R = ls.R
    fill!(R.data, zero(T))
    # `c D P D + σI`, over the entries the band holds.
    for j in 1:n, i in max(1, j - b):j
        v = c * D[i] * P[i, j] * D[j]
        i == j && (v += sigma)
        iszero(v) || (R[i, j] = v; R[j, i] = v)
    end
    # `Ãᵀ diag(ρ) Ã`, row by row: row `k` of `A` reaches only its own band, so each row
    # contributes to a fixed number of entries however large `n` is.
    ba = banded_bandwidth(A)
    for k in 1:m
        w = rho[k] * E[k] * E[k]
        lo, hi = max(1, k - ba), min(n, k + ba)
        for j in lo:hi
            akj = A[k, j] * D[j]
            iszero(akj) && continue
            for i in lo:j
                aki = A[k, i] * D[i]
                iszero(aki) && continue
                v = R[i, j] + w * aki * akj
                R[i, j] = v
                i == j || (R[j, i] = v)
            end
        end
    end
    ls.fact = cholesky!(Symmetric(R); check = false)
    return issuccess(ls.fact)
end

function PureQPBase.solve_system!(ls::BandedReduced, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    copyto!(x, prob.work_n)
    ldiv!(ls.fact, x)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end

# Every position outside the band holds a structural zero on both sides of the diagonal, so
# only the stored band has to be compared: a pair `(i, j)` with either entry stored is
# reached from column `j` or from column `i`, and a pair with neither is zero against zero.
# The generic entrywise scan is `O(n²)` and is the largest single term in a banded `setup`.
function PureQPBase.is_symmetric(M::BandedMatrix)
    size(M, 1) == size(M, 2) || return false
    for j in axes(M, 2), i in PureQPBase.structural_rows(M, j)
        M[i, j] == M[j, i] || return false
    end
    return true
end

# A banded matrix is factored as one, rather than densified into an `n×n` Cholesky. The
# wrapper is admitted alongside the bare matrix because everything below reads `P` through
# `banded_bandwidth`, `size` and `P[i, j]`, all of which the wrapper answers. The union is
# narrower than `BandedLike`: the other types it names have their own methods in `src/`.
function PureQPBase.is_convex(
        ::Type{T}, P::Union{BandedMatrix, Symmetric{<:Any, <:BandedMatrix}}, sigma
    ) where {T}
    isempty(P) && return true
    b = banded_bandwidth(P)
    S = BandedMatrix{T}(undef, size(P), (b, b))
    fill!(S.data, zero(T))
    for j in axes(P, 2), i in max(1, j - b):min(size(P, 1), j + b)
        S[i, j] = P[i, j]
    end
    for i in axes(P, 1)
        S[i, i] += sigma
    end
    return issuccess(cholesky!(Symmetric(S); check = false))
end

@verify BandedReduced trim_compat = true

# The bare name above checks method existence for every parameter; a declared return type
# resolves only for concrete parameters, so the instantiation `banded_backend` builds is
# checked too.
@verify BandedReduced{
    Float64, BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}},
    Cholesky{Float64, BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}},
} trim_compat = true

end
