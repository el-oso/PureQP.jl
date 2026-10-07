# One index over the benchmark caches. Each package's benchmarks write their samples into
# that package's own `bench/results`; this reads all of them and writes a single
# `bench/results/index.json`, which `docs/src/benchmarks.md` renders.
#
#     julia --project=bench bench/consolidate.jl
#
# It reads the caches and never runs a benchmark, so it is cheap and its answer does not
# depend on the machine it runs on.
using JSON, Printf

const ROOT = dirname(@__DIR__)
const INDEX = joinpath(@__DIR__, "results", "index.json")

"""
Where the benchmarks and their caches live. `shared` holds what neither package owns alone:
the problem generators, the snapshot gate and the StrictMode audit.
"""
const SOURCES = (
    (package = "PureOSQP", dir = joinpath(ROOT, "PureOSQP", "bench")),
    (package = "PureIPM", dir = joinpath(ROOT, "PureIPM", "bench")),
    (package = "shared", dir = @__DIR__),
)

"""
    writers(dir) -> Dict{String, String}

Which script in `dir` opens each cache for writing by its literal name. This resolves the
caches not named after the script that writes them; a script naming a cache it does not write
is common in a header comment, so the write mode is part of the match.

A script that builds its cache name by interpolation, or through a path bound to a constant,
is not found here. Both are matched by name instead, by the caller.
"""
function writers(dir)
    out = Dict{String, String}()
    for script in sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
        script == basename(@__FILE__) && continue
        src = read(joinpath(dir, script), String)
        for m in eachmatch(r"open\([^\n]*\"([A-Za-z0-9_]+\.json)\"[^\n]*\"w\"", src)
            out[m.captures[1]] = script
        end
    end
    return out
end

"The value the file records under the first of `ks` it carries, and `nothing` when it carries none."
function meta(data, ks...)
    data isa AbstractDict || return nothing
    for k in ks
        haskey(data, k) && return data[k]
    end
    return nothing
end

"""
    describe(package, dir, file) -> Dict

One record: which package measured it, which script writes it, and what the run was. The
script is usually the cache name with a `.jl` extension, which is the convention most
benchmarks here follow; where no such script exists, the one that opens the cache by name is
used. A cache neither names is reported rather than assumed stale.
"""
function describe(package, dir, file, writes)
    name = first(splitext(file))
    # A measured `Inf` or `NaN` — an unbounded objective, a solver that diverged — is written
    # through as a bare literal, which strict JSON has no spelling for.
    data = JSON.parsefile(joinpath(dir, "results", file); allownan = true)
    byname = name * ".jl"
    script = isfile(joinpath(dir, byname)) ? byname : get(writes, file, nothing)
    return Dict(
        "package" => package,
        "name" => name,
        "script" => script,
        "results" => relpath(joinpath(dir, "results", file), ROOT),
        # Both spellings are in use across the caches.
        "julia_version" => meta(data, "julia_version", "julia"),
        "blas_threads" => meta(data, "blas_threads"),
        "keys" => data isa AbstractDict ? sort(string.(collect(keys(data)))) : String[],
        "bytes" => filesize(joinpath(dir, "results", file)),
    )
end

entries = Dict{String, Any}[]
unrun = Dict{String, Any}[]
for (package, dir) in SOURCES
    resdir = joinpath(dir, "results")
    isdir(resdir) || continue
    writes = writers(dir)
    for file in sort(filter(f -> endswith(f, ".json") && f != "index.json", readdir(resdir)))
        push!(entries, describe(package, dir, file, writes))
    end
    # A benchmark that has never been run leaves no cache, so the docs have nothing to show
    # for it. Naming it here is the only way that stays visible. A script that saves no
    # samples is a problem generator or a gate rather than a benchmark, and is not expected
    # to leave one.
    for script in sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
        script == basename(@__FILE__) && continue
        occursin("\"results\"", read(joinpath(dir, script), String)) || continue
        base = first(splitext(script))
        # Either the cache names this script as its writer, or it is one whose name this
        # script builds by interpolation and so carries the script's name as a prefix.
        any(
            e -> e["package"] == package &&
                (e["script"] == script || startswith(e["name"], base)), entries
        ) && continue
        push!(unrun, Dict("package" => package, "script" => script))
    end
end

orphans = [e for e in entries if isnothing(e["script"])]

open(INDEX, "w") do io
    JSON.print(
        io, Dict(
            "entries" => entries,
            "unrun" => unrun,
            "packages" => [s.package for s in SOURCES],
        ), 2
    )
end

@printf("%-10s %-34s %-10s %8s\n", "package", "cache", "julia", "bytes")
println("-"^66)
for e in entries
    @printf(
        "%-10s %-34s %-10s %8d\n",
        e["package"], e["name"], something(e["julia_version"], "-"), e["bytes"]
    )
end
@printf("\n%d caches over %d packages\n", length(entries), length(SOURCES))
isempty(orphans) ||
    println("no script writes: ", join((e["name"] for e in orphans), ", "))
isempty(unrun) ||
    println("never run: ", join((string(e["package"], "/", e["script"]) for e in unrun), ", "))
println("\nwrote ", relpath(INDEX, ROOT))
