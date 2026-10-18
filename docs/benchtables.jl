# Rendering of the benchmark tables on `src/benchmarks.md` from the saved samples.
#
# A page includes this once and then calls `benchtable` per table, naming the cache and the
# columns it wants. The samples are whatever each package's `bench/results/` holds, so a table
# is never transcribed and never drifts from the run that produced it.

using JSON, Markdown, Printf

"The repository root."
const REPO = normpath(joinpath(@__DIR__, ".."))

"""
    trimexp(s) -> String

`s` with its exponent written the way the prose around these tables writes one: no `+`, and no
leading zero on a single digit. `Printf` emits both (`1e+12`, `1e-09`).
"""
trimexp(s) = replace(s, r"e\+?(-?)0*(\d)" => s"e\1\2")

"""
    cache(package, name) -> Dict

The saved samples of one benchmark. `package` is the package whose `bench/results/` holds the
file, or `"shared"` for the top-level `bench/results/`.
"""
function cache(package, name)
    dir = package == "shared" ? joinpath(REPO, "bench", "results") :
        joinpath(REPO, package, "bench", "results")
    return JSON.parsefile(joinpath(dir, name * ".json"); allownan = true)
end

"""
    num(x) -> Float64

`x` as a float. A sample written as a string parses; a missing one is `NaN`, which every
formatter below renders as an em dash.
"""
num(x::Real) = float(x)
num(x::AbstractString) = something(tryparse(Float64, x), NaN)
num(::Nothing) = NaN
num(::Missing) = NaN

"`x` rounded to `digits` significant digits, without an exponent unless the magnitude needs one."
function sigfig(x::Real, digits::Int = 3)
    isfinite(x) || return "—"
    iszero(x) && return "0"
    r = round(x; sigdigits = digits)
    a = abs(r)
    (a ≥ 1.0e5 || a < 1.0e-3) && return trimexp(Printf.format(Printf.Format("%.$(digits - 1)e"), r))
    places = max(0, digits - 1 - floor(Int, log10(a)))
    return Printf.format(Printf.Format("%.$(places)f"), r)
end

"A duration given in seconds, in milliseconds."
ms(seconds) = (t = num(seconds); isfinite(t) ? sigfig(t * 1000) * " ms" : "—")

"A duration already given in milliseconds."
msval(milliseconds) = (t = num(milliseconds); isfinite(t) ? sigfig(t) * " ms" : "—")

"A duration given in seconds, in microseconds."
us(seconds) = (t = num(seconds); isfinite(t) ? sigfig(t * 1.0e6) * " µs" : "—")

"A ratio, as a multiple."
ratio(x) = (r = num(x); isfinite(r) ? Printf.format(Printf.Format("%.2f"), r) * "×" : "—")

"A quantity spanning orders of magnitude, to one decimal."
function sci(x)
    v = num(x)
    isfinite(v) || return "—"
    iszero(v) && return "0.0"
    return trimexp(Printf.format(Printf.Format("%.1e"), v))
end

"A count."
int(x) = (v = num(x); isfinite(v) ? string(round(Int, v)) : "—")

"`x` as it was written, or an em dash when absent."
str(x) = isnothing(x) || x === missing ? "—" : string(x)

"""
    benchtable(rows, columns) -> Markdown.MD

One table. `columns` pairs each header with how to read a row: a function of the row, or a key
to print as written.
"""
function benchtable(rows, columns)
    io = IOBuffer()
    println(io, "| ", join(first.(columns), " | "), " |")
    println(io, "|", repeat("---|", length(columns)))
    for r in rows
        println(io, "| ", join([render(f, r) for (_, f) in columns], " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end

render(f, row) = string(f(row))
render(key::Union{Symbol, AbstractString}, row) = str(get(row, string(key), nothing))
