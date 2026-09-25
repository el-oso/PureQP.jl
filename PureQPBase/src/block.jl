"""
    BlockReduced{T,M,F} <: LinearSystem

The reduced system when it decouples into independent blocks, which it does for a
[`BlockDiagonal`](@ref) `P` and `A` split at the same columns.

Each block `i` carries its own

    Rᵢ = c Dᵢ Pᵢ Dᵢ + σI + Ãᵢᵀ diag(ρᵢ) Ãᵢ

and nothing couples them, so `K` Cholesky factorizations replace one and each solve is `K`
triangular solves over `Σ nᵢ` entries. Against the dense terminal that is `Σ nᵢ³` against `n³`
to factor and `Σ nᵢ²` against `n²` to store; for `K` equal blocks, `K²` and `K`.

`blocks` holds each `Rᵢ` and is overwritten by that block's *inverse*, so a solve is one `symv`
per block rather than two triangular solves; `scaled` is the scratch each block is assembled
in. Both are rebuilt whenever `ρ` moves.
"""
struct BlockReduced{T <: Real, M <: AbstractMatrix{T}} <: LinearSystem
    blocks::Vector{M}
    scaled::Vector{M}      # block `i` holds `sqrt(ρ) ⊙ E ⊙ Aᵢ ⊙ D`, so `Rᵢ` is one `syrk`
end

"""
    BlockReduced(proto::AbstractVector, sizes)

Build one square buffer per block, as `similar(proto, ...)`, following the array type of the
data it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function BlockReduced(
        proto::AbstractVector{T}, sizes::Vector{Int}, rows::Vector{Int}
    ) where {T <: Real}
    blocks = [similar(proto, T, s, s) for s in sizes]
    scaled = [similar(proto, T, r, s) for (r, s) in zip(rows, sizes)]
    for B in blocks
        fill!(B, zero(T))
    end
    return BlockReduced{T, eltype(blocks)}(blocks, scaled)
end

backend_name(::BlockReduced) = :block

# One triangle per block is what is stored, and the blocks are all there is.
#
# Loops rather than `sum(f, ls.blocks)`: the reduction passes the function to
# `Base.MappingRF`, whose two parameters are both `Function`, and `--trim` refuses a call it
# cannot resolve. `refactored!` names this in the message it throws when a factorization
# fails, so the reduction sits on the error path of every block solve.
function backend_info(ls::BlockReduced)
    n = 0
    stored = 0
    for B in ls.blocks
        k = size(B, 1)
        n += k
        stored += k * (k + 1) ÷ 2
    end
    return BackendInfo(backend_name(ls), true, :reduced, n, stored)
end

"""
    block_rung(P, A, prob, wt, sel; require_multiple = true) -> (LinearSystem, Bool) or nothing

Ladder rung for a reduced matrix that decouples into independent blocks. Declines unless both
operands are [`BlockDiagonal`](@ref) over the same column partition.

`require_multiple` also declines a single block, which is the dense terminal wearing a
wrapper and so never worth the ladder's own time; that is a cost comparison, not a
representation one, so a caller who names `linsys = :block` reaches this with
`require_multiple = false` and gets the wrapper even for one block.
"""
block_rung(P, A, prob, wt, sel::SelectionFor; require_multiple::Bool = true) = nothing

function block_rung(P::BlockDiagonal, A::BlockDiagonal, prob, wt, sel::SelectionFor; require_multiple::Bool = true)
    ((!require_multiple || nblocks(P) > 1) && same_column_partition(P, A)) || return nothing
    return (
        BlockReduced(
            prob.q0,
            [length(colrange(A, i)) for i in 1:nblocks(A)],
            [length(rowrange(A, i)) for i in 1:nblocks(A)],
        ),
        false,
    )
end

# The backend factors over the partition it was built with; a new block run of the same type
# is a different partition and would read the wrong blocks.
function check_update(ls::BlockReduced, P, A)
    same_column_partition(P, A) || throw(
        ArgumentError(
            "P and A must keep the block partition the workspace was built with: the " *
                "block backend factors the reduced matrix over that partition. Rebuild " *
                "the workspace with setup."
        )
    )
    for i in eachindex(ls.blocks)
        size(P.blocks[i]) == (size(ls.blocks[i], 1), size(ls.blocks[i], 1)) ||
            throw(ArgumentError("P and A must keep the block sizes the workspace was built with. Rebuild the workspace with setup."))
        size(A.blocks[i]) == (size(ls.scaled[i], 1), size(ls.blocks[i], 1)) ||
            throw(ArgumentError("P and A must keep the block sizes the workspace was built with. Rebuild the workspace with setup."))
    end
    return nothing
end

function factorize!(ls::BlockReduced{T}, prob, wt)::Bool where {T}
    A, P, D, E, c = prob.A, prob.P, prob.D, prob.E, prob.c
    rho, sigma = wt.w, wt.sigma
    for i in eachindex(ls.blocks)
        R = ls.blocks[i]
        cols, rows = colrange(A, i), rowrange(A, i)
        Ai, Pi = A.blocks[i], P.blocks[i]
        # `sqrt(ρ) ⊙ E ⊙ Aᵢ ⊙ D` first, so the block is one `syrk` rather than a scalar triple
        # loop -- the same shape `ReducedCholesky` uses, and the reason this backend is worth
        # having at all: `K` small BLAS-3 calls, not `K` hand-rolled ones.
        W = ls.scaled[i]
        for (jj, j) in pairs(cols), (ii, i2) in pairs(rows)
            W[ii, jj] = sqrt(rho[i2]) * E[i2] * Ai[ii, jj] * D[j]
        end
        mul!(R, W', W)
        for (jj, j) in pairs(cols), (kk, k) in pairs(cols)
            R[jj, kk] += c * D[j] * Pi[jj, kk] * D[k]
        end
        for jj in axes(R, 1)
            R[jj, jj] += sigma
        end
        f = cholesky!(Symmetric(R); check = false)
        issuccess(f) || return false
        # Inverted, not kept as a factor: each iteration's block solve is then one `symv`
        # rather than two triangular solves. Both cost `2nᵢ²` flops, but a triangular solve
        # computes its entries in sequence and `symv` does not, which is the same reason
        # `ReducedCholesky` inverts. `σI` bounds the conditioning, as it does there.
        invert_spd!(R, f)
    end
    return true
end

function solve_system!(ls::BlockReduced, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    b = prob.work_n
    A = prob.A
    for i in eachindex(ls.blocks)
        cols = colrange(A, i)
        mul!(view(x, cols), Symmetric(ls.blocks[i], :U), view(b, cols))
    end
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end
