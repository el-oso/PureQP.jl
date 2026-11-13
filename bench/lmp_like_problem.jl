# A random problem with the structure and conditioning of a problem handed over as `LinearMaps`: the
# shape, the block layout and the spectra, with none of its entries. Test items in `PureDAQP` and
# `PureIPM` read it, so it needs only `LinearAlgebra` and `LinearMaps`.
#
#   min ½xᵀHx + fᵀx   s.t.   Ax ≥ b,        x = [y (76); z (553)]
#
#   H = blockdiag(1.625 I₇₆, M)            M dense, symmetric positive definite, ill conditioned
#   A = [ 0 │ I₄₂⊗W⊗c │ 0 ]               |W z_j| ≤ 2 on 42 groups of 10 variables    (420 rows)
#       [ I₇₆⊗1₁₃⊗1₂ │ Gᵀ⊗c ]             y_k ≥ |g_ρ·z − β_ρ|, 13 rows per group k   (1976 rows)
#
# with `c = (1, −1)`. Each group's 13 rows are a smooth curve in ρ, a constant plus a dominant
# cosine mode and much weaker further modes, so they are nearly linearly dependent: the singular
# values of a group are in the ratio 1 : 0.14 : 0.002 or smaller. The group scales have three tiers,
# `β` is nearly in the range of `G`, so a few rows of each curve bind, and the last 19 groups have
# `β = 0`. `M` has eigenvalues on a fixed quantile curve from 2.5e-5 to 2.1e4, and `G`'s left
# singular vectors lie mostly in the span of its 60 largest eigenvectors.
#
# Which numbers are random: every entry of `M`, `G`, `W` and `f`, and the residual added to `β`. The
# stream is a 64-bit linear congruential recurrence written out here, so the problem is the same on
# every Julia version; `qr`, `eigen` and `svd` only supply orthogonal bases and a projection, which
# the problem does not depend on beyond rounding.

"A stream of uniform values in `[0, 1)` from `seed`, by Knuth's MMIX recurrence."
function lmpl_uniform(seed::Integer)
    s = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x165667B19E3779F9
    return function ()
        s = s * 0x5851F42D4C957F2D + 0x14057B7EF767814F
        return (s >> 11) / 2.0^53
    end
end

"A stream of standard normal values from `seed`, by the Box–Muller transform."
function lmpl_normal(seed::Integer)
    uniform = lmpl_uniform(seed)
    return function ()
        u1 = 1.0 - uniform()
        u2 = uniform()
        return sqrt(-2 * log(u1)) * cospi(2 * u2)
    end
end

lmpl_randn(next, dims...) = [next() for _ in CartesianIndices(dims)]

lmpl_orthonormal(next, m, n) = Matrix(qr(lmpl_randn(next, m, n)).Q)[:, 1:n]

"Piecewise-linear interpolation of `ys` over the increasing `xs` at `x`."
function lmpl_interp(xs, ys, x)
    x <= first(xs) && return first(ys)
    x >= last(xs) && return last(ys)
    i = searchsortedlast(xs, x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])
end

# Probabilities and log10 of the eigenvalues of `M` at them.
const LMPL_EIG_P = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
const LMPL_EIG_LOG = log10.([2.5e-5, 2.5e-5, 6.55e-5, 0.00962, 0.585, 2.19, 5.05, 1490.0, 20600.0])

"""
    lmp_like_problem(; seed = 1) -> (H, f, A, b, x0, solution)

The problem above as `LinearMaps`: `H` and `A` are
`BlockMap`s of `WrappedMap`, `FillMap` and `KroneckerMap` blocks, and `x0` and `solution` are
zeros, since the problem has no stored answer. `min ½xᵀHx + fᵀx s.t. Ax ≥ b`, with `H` 629×629 and
`A` 2396×629.
"""
function lmp_like_problem(; seed::Integer = 1)
    next = lmpl_normal(seed)
    uniform = lmpl_uniform(seed + 7919)
    ny, nz, ng, rg = 76, 553, 76, 13
    # M: eigenvalues on the quantile curve, eigenvectors from a badly scaled low-rank Gram matrix.
    S = Diagonal(exp.(1.2 .* lmpl_randn(next, nz)))
    U = lmpl_orthonormal(next, nz, 60)
    M0 = S * U * Diagonal(exp.(range(0, -6; length = 60))) * U' * S + 1.0e-3 * I
    V = reverse(eigen(Symmetric(M0)).vectors; dims = 2)
    lam = [10^lmpl_interp(LMPL_EIG_P, LMPL_EIG_LOG, 1 - (i - 1) / (nz - 1)) for i in 1:nz]
    M = Matrix(Symmetric(V * Diagonal(lam) * V'))
    # G: groups of 13 rows, each a smooth curve of a constant and cosine/sine modes.
    scale = k -> k <= 38 ? 2000.0 : (k <= 57 ? 220.0 : 135.0)
    modes = (t -> 1.0, t -> cospi(2t), t -> sinpi(2t), t -> cospi(4t), t -> sinpi(4t))
    ratio = (1.0, 0.137, 0.002, 0.0018, 0.0005)
    Vtop = V[:, 1:60]
    t = [(r - 1) / (rg - 1) for r in 1:rg]
    Gr = zeros(ng * rg, nz)
    for k in 1:ng
        s = scale(k) * (1 + 0.04 * next())
        for (j, mode) in enumerate(modes)
            w = 0.9 .* (Vtop * lmpl_randn(next, 60)) ./ sqrt(60) .+ 0.1 .* lmpl_randn(next, nz) ./ sqrt(nz)
            w .*= s * ratio[j] / (norm(mode.(t)) * norm(w))
            for r in 1:rg
                Gr[(k - 1) * rg + r, :] .+= mode(t[r]) .* w
            end
        end
    end
    # β: smooth within a group and anchored across groups, then replaced by the part of it that G
    # reproduces plus a small smooth residual per group; zero for the last 19 groups.
    anchors = [1, 2, 5, 10, 20, 30, 40, 50, 57, 58]
    first_b = [0.47, 0.54, 0.78, 1.2, -2.2, -0.58, -0.28, -0.23, -0.17, 0.0]
    last_b = [3.18, 3.14, 3.0, 2.7, -1.5, -0.52, -0.18, -0.13, -0.09, 0.0]
    beta = zeros(ng * rg)
    for k in 1:(ng - 19), r in 1:rg
        f0, f1 = lmpl_interp(anchors, first_b, k), lmpl_interp(anchors, last_b, k)
        beta[(k - 1) * rg + r] = f0 + (f1 - f0) * t[r]
    end
    F = svd(Gr)
    keep = F.S .> 1.0e-6 * F.S[1]
    beta = Gr * (F.V[:, keep] * ((F.U[:, keep]' * beta) ./ F.S[keep]))
    for k in 1:(ng - 19)
        u = 0.05 .* lmpl_randn(next, 3)
        for r in 1:rg
            beta[(k - 1) * rg + r] += u[1] + u[2] * t[r] + u[3] * t[r]^2
        end
    end
    beta[((ng - 19) * rg + 1):end] .= 0.0
    # W: 5×10 with fixed singular values.
    W = lmpl_orthonormal(next, 5, 5) * Diagonal([0.684, 0.664, 0.642, 0.461, 0.46]) * lmpl_orthonormal(next, 10, 5)'
    # f: nonzero on about half of z[1:420].
    f = zeros(ny + nz)
    for i in 1:420
        uniform() < 0.5 && (f[ny + i] = 0.05 * next())
    end
    c = reshape([1.0, -1.0], 2, 1)
    zero_block(m, n) = LinearMaps.FillMap(0.0, (m, n))
    H = hcat(
        vcat(LinearMap(Diagonal(fill(1.625, ny))), zero_block(nz, ny)),
        vcat(zero_block(ny, nz), LinearMap(M)),
    )
    block1 = hcat(zero_block(420, ny), kron(LinearMap(true * I, 42), LinearMap(W), LinearMap(c)), zero_block(420, 133))
    block2 = hcat(
        kron(LinearMap(true * I, ny), LinearMaps.FillMap(1.0, (rg, 1)), LinearMaps.FillMap(1.0, (2, 1))),
        kron(LinearMap(adjoint(Matrix(Gr'))), LinearMap(c)),
    )
    A = vcat(block1, block2)
    b = vcat(fill(-2.0, 420), vec(vcat(beta', -beta')))
    return (; H, f, A, b, x0 = zeros(ny + nz), solution = zeros(ny + nz))
end
