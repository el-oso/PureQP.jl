# What the convexity test costs for the structured `P` types the solver ships a backend for.
#
# `setup` calls `is_convex` once before any backend is chosen, and `update!` calls it again
# whenever `P` changes. The generic method densifies -- `O(n^3)` to factor and `O(n^2)` in
# memory -- so a type with a cheaper test is measured here against that fallback, reached
# through `invoke` so the number is the generic method itself rather than a re-derivation.
#
# Three sweeps:
#
#  1. `is_convex` for a `Tridiagonal` and a `Symmetric{<:Any,<:BandedMatrix}` `P`, generic
#     against structured.
#  2. `issymmetric` for a bare `BandedMatrix`, which `validate` calls and which has no
#     structure-aware method, against the `setup` it runs inside -- once scaled and once at
#     `scaling = 0`, so the equilibration share separates by subtraction.
#  3. The `is_convex` share of a `JLArray` setup, where `Matrix{T}(P)` copies the matrix back
#     to the host.
#
# Swept at two BLAS thread counts: `src/` and `ext/` pin nothing, so the deployed solver runs
# at whatever the caller has, and a scalar walk timed against a threaded competitor answers a
# different question than one timed alone.
using PureOSQP, BandedMatrices, JLArrays, GPUArraysCore, Krylov
using LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

const SIZES = (500, 1000, 2000)
const BANDWIDTH = 8
const SIGMA = 1.0e-6
const THREADS = (1, min(8, Sys.CPU_THREADS))
const GPU_SIZES = (100, 200)
const RESULTS = joinpath(@__DIR__, "results", "is_convex_coverage.json")

times(x) = [s.time for s in x.samples]

"The median of `f`'s samples, warmed once so no compilation lands in the timings."
function median_time(f, seconds = 0.5)
    f()
    return median(times(@be f() seconds = seconds))
end

"`is_convex` through the generic `AbstractMatrix` method, whatever `P`'s own method is."
generic_is_convex(P, sigma) =
    invoke(PureOSQP.is_convex, Tuple{Type{Float64}, AbstractMatrix, Any}, Float64, P, sigma)

"`is_symmetric` through its generic method, which is `issymmetric`'s entrywise scan."
generic_is_symmetric(P) = invoke(PureOSQP.is_symmetric, Tuple{Any}, P)

"A symmetric positive definite `BandedMatrix` of bandwidth `b`."
function banded(n, b, rng)
    B = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(B.data, 0.0)
    B[band(0)] .= rand(rng, n) .+ 2b
    for k in 1:b
        B[band(k)] .= rand(rng, n - k) ./ (4b)
        B[band(-k)] .= B[band(k)]
    end
    return B
end

"A symmetric positive definite `Tridiagonal`."
function tridiagonal(n, rng)
    ev = rand(rng, n - 1) ./ 8
    return Tridiagonal(copy(ev), rand(rng, n) .+ 2.0, ev)
end

function is_convex_rows()
    rows = Dict{String, Any}[]
    for nthreads in THREADS
        BLAS.set_num_threads(nthreads)
        for n in SIZES
            rng = MersenneTwister(9001 + n)
            for (case, P) in (
                    "Tridiagonal" => tridiagonal(n, rng),
                    "Symmetric{BandedMatrix}" => Symmetric(banded(n, BANDWIDTH, rng)),
                )
                @assert PureOSQP.is_convex(Float64, P, SIGMA) == generic_is_convex(P, SIGMA)
                ts = median_time(() -> PureOSQP.is_convex(Float64, P, SIGMA))
                tg = median_time(() -> generic_is_convex(P, SIGMA))
                push!(
                    rows, Dict{String, Any}(
                        "sweep" => "is_convex", "blas_threads" => nthreads, "n" => n,
                        "case" => case, "structured_s" => ts, "generic_s" => tg,
                        "speedup" => tg / ts,
                    )
                )
                @printf(
                    "is_convex threads=%d n=%5d %-24s structured %9.3f us  generic %9.3f us  %7.1fx\n",
                    nthreads, n, case, 1.0e6ts, 1.0e6tg, tg / ts
                )
                flush(stdout)
            end
        end
    end
    return rows
end

function issymmetric_rows()
    rows = Dict{String, Any}[]
    for nthreads in THREADS
        BLAS.set_num_threads(nthreads)
        for n in SIZES
            rng = MersenneTwister(4703 + n)
            P = banded(n, BANDWIDTH, rng)
            A = Diagonal(rand(rng, n) .+ 0.5)
            q, l, u = randn(rng, n), -rand(rng, n), rand(rng, n)
            ws = setup(P, q, A, l, u)
            name = String(PureOSQP.backend_name(ws.linsys))
            @assert PureOSQP.is_symmetric(P) == generic_is_symmetric(P)
            ti = median_time(() -> PureOSQP.is_symmetric(P))
            tg = median_time(() -> generic_is_symmetric(P))
            tc = median_time(() -> PureOSQP.is_convex(Float64, P, SIGMA))
            ts = median_time(() -> setup(P, q, A, l, u), 1.0)
            t0 = median_time(() -> setup(P, q, A, l, u; scaling = 0), 1.0)
            push!(
                rows, Dict{String, Any}(
                    "sweep" => "issymmetric", "blas_threads" => nthreads, "n" => n,
                    "bandwidth" => BANDWIDTH, "backend" => name,
                    "structured_s" => ti, "generic_s" => tg, "is_convex_s" => tc,
                    "setup_scaled_s" => ts, "setup_unscaled_s" => t0,
                    "equilibration_s" => ts - t0,
                    # What `setup` costs with the generic predicate in place: the same run
                    # with the band comparison exchanged for the entrywise scan.
                    "setup_generic_predicate_s" => ts - ti + tg,
                    "generic_share_of_setup" => tg / (ts - ti + tg),
                )
            )
            @printf(
                "issymm   threads=%d n=%5d %-10s is_symmetric %8.3f ms  generic %8.3f ms  is_convex %8.3f ms  setup %8.3f ms  unscaled %8.3f ms  generic share %5.1f%%\n",
                nthreads, n, name, 1.0e3ti, 1.0e3tg, 1.0e3tc, 1.0e3ts, 1.0e3t0,
                100tg / (ts - ti + tg)
            )
            flush(stdout)
        end
    end
    return rows
end

"""
    gpu_rows() -> Vector{Dict}

The `is_convex` share of a `JLArray` setup.

Returns no rows on a Julia whose LLVM segfaults compiling this path: the loop vectorizer dies in
`VPScalarPHIRecipe::execute` on 1.13 with LLVM 20.1, which takes the whole process with it rather
than raising. The sweep is restored by widening this check once a Julia that compiles it ships.
"""
function gpu_rows()
    VERSION < v"1.13" || return Dict{String, Any}[]
    JLArrays.allowscalar(false)
    rows = Dict{String, Any}[]
    for nthreads in THREADS
        BLAS.set_num_threads(nthreads)
        for n in GPU_SIZES
            rng = MersenneTwister(2207 + n)
            m = 2n
            X = randn(rng, n, n)
            Ph = Matrix(X'X / n + I)
            Ah = randn(rng, m, n)
            b = Ah * randn(rng, n)
            P, q = jl(Ph), jl(randn(rng, n))
            A, l, u = jl(Ah), jl(b .- rand(rng, m)), jl(b .+ rand(rng, m))
            opts = (linsys = :indirect,)
            tc = median_time(() -> PureOSQP.is_convex(Float64, P, SIGMA))
            ts = median_time(() -> setup(P, q, A, l, u; opts...), 1.0)
            push!(
                rows, Dict{String, Any}(
                    "sweep" => "gpu", "blas_threads" => nthreads, "n" => n, "m" => m,
                    "is_convex_s" => tc, "setup_s" => ts, "share_of_setup" => tc / ts,
                )
            )
            @printf(
                "gpu      threads=%d n=%5d is_convex %8.3f ms  setup %8.3f ms  share %5.1f%%\n",
                nthreads, n, 1.0e3tc, 1.0e3ts, 100tc / ts
            )
            flush(stdout)
        end
    end
    return rows
end

function main()
    rows = vcat(is_convex_rows(), issymmetric_rows(), gpu_rows())
    open(RESULTS, "w") do io
        JSON.print(
            io, Dict(
                "julia_version" => string(VERSION),
                "cpu_threads" => Sys.CPU_THREADS,
                "sizes" => collect(SIZES),
                "gpu_sizes" => collect(GPU_SIZES),
                "sigma" => SIGMA,
                "cases" => rows,
            ), 2
        )
    end
    return println("wrote $RESULTS")
end

main()
