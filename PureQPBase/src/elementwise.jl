# Elementwise work, written once as scalar functions and scheduled two ways.
#
# The `Vector` methods are explicit loops because that is what carries the allocation-free
# proof: AllocCheck reports a broadcast as allocating even when its destination is distinct
# from its sources, since `Base.Broadcast` keeps an `unaliascopy` branch it cannot rule out,
# and it reports a multi-argument `mapreduce` the same way. The generic methods use
# broadcasts and reductions, which is what an array that forbids scalar indexing needs.
#
# Both schedules call the same scalar functions, so they compute the same thing per element.
# Loops index the destination alone rather than `eachindex(dest, srcs...)`: the workspace's
# buffers are all `similar` to one prototype and so share axes by construction, and the
# multi-array form builds its DimensionMismatch message through a `join` that `--trim`
# cannot resolve.
#
# Dispatch is on the destination array rather than on the workspace, because the workspace's
# element type appears in both its first and its vector parameter and the two spellings are
# ambiguous. `bench/strictmode_audit.jl` checks the loops keep their proof.

"""
    paired(a, b) -> indices

The indices of `a`, having checked that `b` shares them.

What `eachindex(a, b)` does, minus the part `--trim` cannot follow: its `DimensionMismatch`
formats the axes through a `join` over a tuple, which juliac's verifier refuses. The message
here is a constant, and the throw is out of line so the check costs a comparison.
"""
@inline function paired(a, b)
    axes(a) == axes(b) || _axes_mismatch()
    return eachindex(a)
end

@inline function paired(a, b, c)
    (axes(a) == axes(b) && axes(a) == axes(c)) || _axes_mismatch()
    return eachindex(a)
end

@noinline _axes_mismatch() = throw(DimensionMismatch("paired arrays must have the same axes"))

"`a x̃ + (1 - a) x`, the relaxation ADMM applies to both `x` and `z`."
@inline relax(a, tilde, prev) = a * tilde + (one(a) - a) * prev

"The `x` half of an ADMM step: the relaxed iterate, and how far it moved."
function update_x!(x::Vector, delta_x, xtilde, x_prev, a)
    for i in eachindex(x)
        xi = relax(a, xtilde[i], x_prev[i])
        x[i] = xi
        delta_x[i] = xi - x_prev[i]
    end
    return x
end

function update_x!(x, delta_x, xtilde, x_prev, a)
    x .= relax.(a, xtilde, x_prev)
    delta_x .= x .- x_prev
    return x
end

"""
The `z` and `y` half of an ADMM step: relax, project onto `[l, u]`, and move the multiplier
by the amount the projection removed.
"""
function update_zy!(
        z::Vector, y, delta_y, ztilde, z_prev, rho_vec, rho_inv_vec, l, u, a, scratch
    )
    # The loop needs no scratch; the argument keeps the two schedules one call.
    for i in eachindex(z)
        relaxed = relax(a, ztilde[i], z_prev[i])
        zi = clamp(relaxed + rho_inv_vec[i] * y[i], l[i], u[i])
        z[i] = zi
        dy = rho_vec[i] * (relaxed - zi)
        delta_y[i] = dy
        y[i] += dy
    end
    return z
end

function update_zy!(
        z, y, delta_y, ztilde, z_prev, rho_vec, rho_inv_vec, l, u, a, scratch
    )
    # `z_prev` is a distinct array from `z` after the step's swap, so writing `z` first and
    # reading it back below gives the new iterate, not a partially updated one. The relaxed
    # point is computed once into `scratch` — a workspace buffer — rather than as a fresh
    # temporary twice, which is what this schedule is for on an array that forbids scalar
    # indexing.
    # `scratch` holds the relaxed point and is not written again: the dual update is
    # `ρ ⊙ (relaxed - z)`, so adding `ρ⁻¹ ⊙ y` into it would carry `y` into its own update.
    # That term belongs only to the projection, where it fuses into the `clamp` broadcast
    # without a temporary of its own.
    scratch .= relax.(a, ztilde, z_prev)
    z .= clamp.(scratch .+ rho_inv_vec .* y, l, u)
    delta_y .= rho_vec .* (scratch .- z)
    y .+= delta_y
    return z
end

"`dest = a - b`, elementwise."
function subtract!(dest::Vector, a, b)
    for i in eachindex(dest)
        dest[i] = a[i] - b[i]
    end
    return dest
end
subtract!(dest, a, b) = (dest .= a .- b)

"`dest = a + b`, elementwise."
function add!(dest::Vector, a, b)
    for i in eachindex(dest)
        dest[i] = a[i] + b[i]
    end
    return dest
end
add!(dest, a, b) = (dest .= a .+ b)

"`dest += a`, elementwise."
function increment!(dest::Vector, a)
    for i in eachindex(dest)
        dest[i] += a[i]
    end
    return dest
end
increment!(dest, a) = (dest .+= a)

"`dest = a * b`, elementwise."
function multiply!(dest::Vector, a, b)
    for i in eachindex(dest)
        dest[i] = a[i] * b[i]
    end
    return dest
end
multiply!(dest, a, b) = (dest .= a .* b)

"`dest = a / b`, elementwise."
function divide!(dest::Vector, a, b)
    for i in eachindex(dest)
        dest[i] = a[i] / b[i]
    end
    return dest
end
divide!(dest, a, b) = (dest .= a ./ b)

"`dest = s * a - b`, for a scalar `s`."
function scale_subtract!(dest::Vector, s, a, b)
    for i in eachindex(dest)
        dest[i] = s * a[i] - b[i]
    end
    return dest
end
scale_subtract!(dest, s, a, b) = (dest .= s .* a .- b)

"`dest = a - s * b`, for an elementwise `s`."
function subtract_scaled!(dest::Vector, a, s, b)
    for i in eachindex(dest)
        dest[i] = a[i] - s[i] * b[i]
    end
    return dest
end
subtract_scaled!(dest, a, s, b) = (dest .= a .- s .* b)

"`dest += a + s * b`, for a scalar `s`."
function add_scaled!(dest::Vector, a, s, b)
    for i in eachindex(dest)
        dest[i] += a[i] + s * b[i]
    end
    return dest
end
add_scaled!(dest, a, s, b) = (dest .+= a .+ s .* b)

"`dest *= a`, elementwise."
function scale_by!(dest::Vector, a)
    for i in eachindex(dest)
        dest[i] *= a[i]
    end
    return dest
end
scale_by!(dest, a) = (dest .*= a)

"`dest *= s * a`, elementwise, for a scalar `s`."
function scale_by!(dest::Vector, a, s)
    for i in eachindex(dest)
        dest[i] *= a[i] * s
    end
    return dest
end
scale_by!(dest, a, s) = (dest .*= a .* s)
