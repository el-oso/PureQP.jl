# The three ways to write `max|v[i]|`, which `PureQPBase.norm_inf` runs in every termination check.
# Writes bench/results/norm_inf_shapes.json.
#
#     julia --project=bench bench/norm_inf_shapes.jl
#
# The reason this exists is that the choice is not between "a reduction" and "a loop": the loop's
# shape decides it. `max` vectorizes; a comparison that branches to assign does not, and is 2-3x
# slower than the reduction at the sizes where the reduction is at its best. Reporting one ratio
# for "the loop" hides a factor of five between the two loops.
#
# The CPU clock on the host this is usually run on is not pinned, so treat the times as indicative
# rather than as a gate. The ratios are what this is for, and they are stable enough to decide the
# question: the orderings below held on 1.12.7 and 1.13.1.

using LinearAlgebra, Printf
using Chairmarks: @be

BLAS.set_num_threads(1)

"The reduction, which is what reaches `Base.MappingRF`."
reduction(v) = maximum(abs, v; init = zero(eltype(v)))

"A loop over `max`: what `norm_inf` uses."
function max_loop(v)
    r = zero(eltype(v))
    for i in eachindex(v)
        r = max(r, abs(v[i]))
    end
    return r
end

"A loop that branches to assign. Written out because it is the obvious hand translation."
function branch_loop(v)
    r = zero(eltype(v))
    for x in v
        a = abs(x)
        a > r && (r = a)
    end
    return r
end

ns(f, v) = minimum(@be f($v) seconds = 2).time * 1.0e9

rows = NamedTuple[]
for n in (16, 64, 256, 576, 2000, 5000, 50_000)
    v = [sinpi(3.0 * j / n) for j in 1:n]
    reduction(v) ≈ max_loop(v) ≈ branch_loop(v) || error("the three forms disagree at n = $n")
    tr, tm, tb = ns(reduction, v), ns(max_loop, v), ns(branch_loop, v)
    push!(rows, (; n, reduction_ns = tr, max_loop_ns = tm, branch_loop_ns = tb))
    @printf(
        "n=%6d  reduction %9.1f ns | max loop %9.1f (%.2fx) | branching loop %9.1f (%.2fx)\n",
        n, tr, tm, tm / tr, tb, tb / tr
    )
    flush(stdout)
end

mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "norm_inf_shapes.json"), "w") do io
    println(io, "{")
    @printf(io, "  \"julia\": \"%s\",\n", VERSION)
    println(io, "  \"shapes\": [")
    for (i, r) in enumerate(rows)
        @printf(
            io,
            "    {\"n\": %d, \"reduction_ns\": %.1f, \"max_loop_ns\": %.1f, \"branch_loop_ns\": %.1f, \"max_over_reduction\": %.4f, \"branch_over_reduction\": %.4f}%s\n",
            r.n, r.reduction_ns, r.max_loop_ns, r.branch_loop_ns,
            r.max_loop_ns / r.reduction_ns, r.branch_loop_ns / r.reduction_ns,
            i == length(rows) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end
println("\nwrote bench/results/norm_inf_shapes.json")
