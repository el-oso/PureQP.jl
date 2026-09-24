"""
    DiagonalLowRank{T,V,M,F} <: LinearSystem

The reduced system when it is a diagonal plus a rank-`k` correction, which it is for a
`Diagonal` `P` with a [`RowCoupled`](@ref) `A`:

    R = C + Vᵀ W V,    C = c D P D + σI + (the one-entry rows' contribution),
                       V = the scaled coupling rows,  W = diag(ρ) over them

Woodbury solves that without forming `R`:

    R⁻¹b = C⁻¹b − C⁻¹Vᵀ (W⁻¹ + V C⁻¹ Vᵀ)⁻¹ V C⁻¹ b

so an iteration is two `gemv`s against a `k×n` block and one `k×k` solve — `O(nk)` against
the dense backend's `O(n²)`, in `O(nk)` storage rather than `O(n²)`.

`cinv` holds `C⁻¹`, `Y` holds `V C⁻¹`, and `fact` factors the `k×k` capacitance
`W⁻¹ + V C⁻¹ Vᵀ`. Both are rebuilt whenever `ρ` moves, which is `O(nk²)` — cheaper than the
`O(n³)` the dense backend spends on the same event.

Unlike the banded backend there is no serial recurrence here: both halves of the apply are
matrix–vector products, so the correction wins per iteration wherever it wins on flops.
"""
mutable struct DiagonalLowRank{
        T <: Real, V <: AbstractVector{T}, M <: AbstractMatrix{T}, F,
    } <: LinearSystem
    cinv::V          # the inverted diagonal core
    Y::M             # V C⁻¹, k×n
    V::M             # the scaled coupling rows, k×n
    cap::M           # the k×k capacitance, rebuilt each factorization
    fact::F
    ky::V            # k-length scratch
    ty::V            # n-length scratch
end

"""
    DiagonalLowRank(proto::AbstractVector, n, k)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data it
was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function DiagonalLowRank(proto::AbstractVector{T}, n::Integer, k::Integer) where {T <: Real}
    cinv = similar(proto, T, n)
    Y = similar(proto, T, k, n)
    V = similar(proto, T, k, n)
    cap = similar(proto, T, k, k)
    fill!(cap, zero(T))
    for i in 1:k
        cap[i, i] = one(T)
    end
    fact = cholesky!(Symmetric(copy(cap)))
    return DiagonalLowRank{T, typeof(cinv), typeof(Y), typeof(fact)}(
        cinv, Y, V, cap, fact, similar(proto, T, k), similar(proto, T, n)
    )
end

backend_name(::DiagonalLowRank) = :lowrank

# `Y` and the capacitance are what the backend stores beyond the core, and the core is `n`.
function backend_info(ls::DiagonalLowRank)
    n, k = length(ls.cinv), size(ls.Y, 1)
    return BackendInfo(backend_name(ls), true, :reduced, n, n + k * n + k * (k + 1) ÷ 2)
end

"""
    lowrank_rung(P, A, prob, wt, sel; require_crossover = true) -> (LinearSystem, Bool) or nothing

Ladder rung for a reduced matrix that is a diagonal plus a low-rank correction. Declines
unless `P` is `Diagonal` and `A` is [`RowCoupled`](@ref) with at least one coupling row.

`require_crossover` also declines when the correction is wide enough that `O(nk)` stops
beating the dense `O(n²)` apply — a cost comparison, not a representation one, so a caller who
names `linsys = :lowrank` reaches this with `require_crossover = false` and gets the
correction at any width.
"""
lowrank_rung(P, A, prob, wt, sel::SelectionFor; require_crossover::Bool = true) = nothing

function lowrank_rung(P::Diagonal, A::RowCoupled, prob, wt, sel::SelectionFor; require_crossover::Bool = true)
    k = coupling_rank(A)
    n = prob.n
    k < 1 && return nothing
    # The apply is two `gemv`s against a `k×n` block where the rung below does one `symv`
    # against `n×n`, so on flops alone the correction wins until `k` reaches `n/2`. It does
    # not: `symv` is multithreaded and both `gemv`s here are narrow, so the measured crossing
    # is `k/n ≈ 0.225` at eight BLAS threads and `≈ 0.445` at one
    # (`PureOSQP/bench/results/gate_crossover_lowrank.json`). `src/` and `ext/` pin no threads, so the
    # limit has to hold at the threaded crossing, and it sits below it: at `k = n/10` the
    # solve is 1.78–2.27× ahead and setup 3.0–6.8× at every size measured.
    require_crossover && 10k > n && return nothing
    return (DiagonalLowRank(prob.q0, n, k), false)
end

"""
    scale_coupling!(ls, prob)

Fill `V` with the scaled coupling rows, `E[i]·C[i,j]·D[j]`.

Depends on the data and the equilibration factors, not on the weights, which is why
[`refactor_weights!`](@ref) leaves it alone.
"""
function scale_coupling!(ls::DiagonalLowRank, prob)
    A, D, E, V = prob.A, prob.D, prob.E, ls.V
    for i in axes(V, 1), j in axes(V, 2)
        V[i, j] = E[i] * A.coupling[i, j] * D[j]
    end
    return ls
end

"""
    refresh_core!(ls, prob, wt) -> Bool

Rebuild everything the weights enter: the diagonal core `C⁻¹`, `Y = V C⁻¹`, and the `k×k`
capacitance and its factorization. Reads `V` and does not write it.
"""
function refresh_core!(ls::DiagonalLowRank{T}, prob, wt)::Bool where {T}
    n = prob.n
    A, P, D, E, c = prob.A, prob.P, prob.D, prob.E, prob.c
    rho, sigma = wt.w, wt.sigma
    k = size(ls.V, 1)
    # The core: `P`'s diagonal, `σ`, and the one-entry rows, each of which touches a single
    # column and so contributes only to the diagonal.
    cinv = ls.cinv
    for j in 1:n
        cinv[j] = c * D[j] * P[j, j] * D[j] + sigma
    end
    w, cols = A.weights, A.cols
    for r in eachindex(w, cols)
        j = cols[r]
        i = k + r
        a = E[i] * w[r] * D[j]
        cinv[j] += rho[i] * a * a
    end
    for j in 1:n
        cinv[j] > zero(T) || return false
        cinv[j] = inv(cinv[j])
    end
    V, Y = ls.V, ls.Y
    for i in 1:k, j in 1:n
        Y[i, j] = V[i, j] * cinv[j]
    end
    # Capacitance `W⁻¹ + V C⁻¹ Vᵀ`. `ρ` is positive, so `W⁻¹` is too and the sum stays
    # positive definite whenever the core is.
    mul!(ls.cap, Y, V')
    for i in 1:k
        ls.cap[i, i] += inv(rho[i])
    end
    ls.fact = cholesky!(Symmetric(ls.cap); check = false)
    return issuccess(ls.fact)
end

function factorize!(ls::DiagonalLowRank, prob, wt)::Bool
    scale_coupling!(ls, prob)
    return refresh_core!(ls, prob, wt)
end

# `V` is `E ⊙ C ⊙ D`, so a weight move leaves it current and only the core, `Y` and the `k×k`
# capacitance have to be rebuilt.
refactor_weights!(ls::DiagonalLowRank, prob, wt) = refresh_core!(ls, prob, wt)

# The backend holds storage for the coupling rank it was built with.
function check_update(ls::DiagonalLowRank, P, A)
    coupling_rank(A) != size(ls.V, 1) && throw(
        ArgumentError(
            "the coupling rows of A must stay the number the workspace was built with: " *
                "the low-rank backend holds storage for that rank. Rebuild the " *
                "workspace with setup."
        )
    )
    return nothing
end

function solve_system!(ls::DiagonalLowRank, prob, wt, rhs_x, rhs_z, x, z)::Nothing
    reduced_rhs!(prob, wt, rhs_x, rhs_z)
    b = prob.work_n
    multiply!(x, ls.cinv, b)          # C⁻¹b
    mul!(ls.ky, ls.Y, b)              # V C⁻¹ b
    ldiv!(ls.fact, ls.ky)             # (W⁻¹ + V C⁻¹ Vᵀ)⁻¹ V C⁻¹ b
    mul!(ls.ty, ls.Y', ls.ky)         # C⁻¹ Vᵀ (…)
    subtract!(x, x, ls.ty)
    prob.m > 0 && mul_A!(z, prob, x)
    return nothing
end
