# Per-test-item line coverage, and the smallest subset of items that covers the same lines.
#
#     julia --project=bench bench/coverage_per_item.jl PureDAQP
#
# Writes bench/results/coverage_per_item_<package>.json.
#
# Julia's coverage counters accumulate for the life of a process and are written out when it
# exits, so attributing lines to one item means running that item in a process of its own. That
# is what dominates the cost here: one Julia start and one compilation of the package per item.
# Nothing in the ecosystem does this — `LocalCoverage.jl` and `CoverageTools.jl` both report a
# whole run, and the per-item question is what deciding redundancy needs.
#
# What it reports:
#
#   * `unique`, the lines an item covers that no other item covers. An item with none is covered
#     entirely by the rest of the suite, which is the criterion for calling it a duplicate.
#   * `cover`, a greedy set cover: the items, in order, that together reach every line the whole
#     suite reaches. The order is by lines-added-per-second, so dropping the tail costs the least
#     coverage per second saved.
#
# Line coverage is what this measures and all it measures. Two items can cover one set of lines
# and assert different things of it — a solve that must succeed and a solve that must be refused
# by name run the same code. The report says which items are redundant *by lines*; whether an
# item asserts something no other does is a reading, not a measurement.

using CoverageTools
using Printf

const PKG = isempty(ARGS) ? "PureDAQP" : ARGS[1]
const ROOT = normpath(joinpath(@__DIR__, ".."))
const SRC = joinpath(ROOT, PKG, "src")

"Every `@testitem` name in `dir`, in file order."
function item_names(dir)
    names = String[]
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jl") || continue
        for line in eachline(joinpath(root, f))
            m = match(r"^@testitem\s+\"((?:[^\"\\]|\\.)*)\"", line)
            isnothing(m) && continue
            push!(names, replace(m.captures[1], "\\\"" => "\""))
        end
    end
    return names
end

"""
    covered_lines(name) -> Set{Tuple{String, Int}}

The `(file, line)` pairs the item called `name` executes, from a process of its own.

The item is selected by an exact-match filter on its name, so an item whose name is a prefix of
another's still runs alone.
"""
function covered_lines(name)
    script = """
    using TestItemRunner
    @run_package_tests filter = ti -> ti.name == $(repr(name))
    """
    cmd = `$(Base.julia_cmd()) --project=$(joinpath(ROOT, PKG, "test")) --code-coverage=user -e $script`
    run(pipeline(Cmd(cmd; dir = joinpath(ROOT, PKG)); stdout = devnull, stderr = devnull))
    covered = Set{Tuple{String, Int}}()
    for fc in CoverageTools.process_folder(SRC)
        rel = relpath(fc.filename, ROOT)
        for (i, hits) in enumerate(fc.coverage)
            isnothing(hits) && continue
            hits > 0 && push!(covered, (rel, i))
        end
    end
    for (root, _, files) in walkdir(SRC), f in files
        endswith(f, ".cov") && rm(joinpath(root, f); force = true)
    end
    return covered
end

names = item_names(joinpath(ROOT, PKG, "test"))
@printf("%s: %d items\n", PKG, length(names))

sets = Dict{String, Set{Tuple{String, Int}}}()
times = Dict{String, Float64}()
for (i, nm) in enumerate(names)
    t = @elapsed s = covered_lines(nm)
    sets[nm] = s
    times[nm] = t
    @printf("  [%3d/%3d] %6d lines %7.1f s  %s\n", i, length(names), length(s), t, first(nm, 56))
    flush(stdout)
end

total = reduce(union, values(sets); init = Set{Tuple{String, Int}}())
@printf("\nunion over all items: %d lines\n", length(total))

# Lines no other item reaches. An item with none adds no line the rest of the suite misses.
uniques = Dict(
    nm => length(setdiff(s, reduce(union, (sets[o] for o in names if o != nm); init = Set{Tuple{String, Int}}())))
        for (nm, s) in sets
)
redundant = sort([nm for nm in names if iszero(uniques[nm])]; by = nm -> -times[nm])
@printf("items covering no line of their own: %d of %d\n", length(redundant), length(names))
@printf(
    "time held by those items: %.1f s of %.1f s\n",
    sum(times[nm] for nm in redundant; init = 0.0), sum(values(times))
)

# Greedy cover, taking the item that adds the most lines per second it costs.
remaining, chosen = copy(total), String[]
while !isempty(remaining)
    best, gain = "", -1.0
    for nm in names
        nm in chosen && continue
        add = length(intersect(sets[nm], remaining))
        iszero(add) && continue
        rate = add / max(times[nm], 1.0e-3)
        # An `if`, not a `&&` chain: `(best, gain = nm, rate)` inside one parses as a named-tuple
        # expression rather than an assignment, so the selection silently never happened.
        if rate > gain
            best = nm
            gain = rate
        end
    end
    isempty(best) && break
    push!(chosen, best)
    setdiff!(remaining, sets[best])
end
@printf(
    "greedy cover: %d items reach all %d lines (%.1f s)\n",
    length(chosen), length(total), sum(times[nm] for nm in chosen; init = 0.0)
)

mkpath(joinpath(@__DIR__, "results"))

# The line sets themselves, so a different question of the same measurements — another cover, a
# pair that covers one set between them — is answered by reading this rather than by spending the
# collection again.
open(joinpath(@__DIR__, "results", "coverage_sets_$(PKG).tsv"), "w") do io
    for nm in names
        println(io, nm, "\t", join(("$(f):$(l)" for (f, l) in sort(collect(sets[nm]))), " "))
    end
end

open(joinpath(@__DIR__, "results", "coverage_per_item_$(PKG).json"), "w") do io
    println(io, "{")
    @printf(io, "  \"package\": \"%s\",\n  \"items\": %d,\n  \"union_lines\": %d,\n", PKG, length(names), length(total))
    println(io, "  \"per_item\": [")
    for (i, nm) in enumerate(names)
        @printf(
            io, "    {\"name\": %s, \"lines\": %d, \"unique\": %d, \"seconds\": %.2f, \"in_cover\": %s}%s\n",
            repr(nm), length(sets[nm]), uniques[nm], times[nm], nm in chosen, i == length(names) ? "" : ","
        )
    end
    println(io, "  ]")
    println(io, "}")
end
println("wrote bench/results/coverage_per_item_$(PKG).json")
