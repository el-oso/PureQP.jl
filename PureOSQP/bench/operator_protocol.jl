# What the matrix-free route costs a products-only operator, against the same problem
# materialized, and what expressing that operator through LinearMaps.jl costs over
# implementing the protocol directly.
#
# One operator, `P = Diagonal(d) + α v vᵀ`, written three ways. The columns are named for the
# three:
#
#   protocol   the `LazyPSD` in `lazy_operator.jl`, an `AbstractMatrix` that implements the
#              products-only protocol itself -- `mul!`, `is_materializable`, `is_convex` and
#              the per-column seam -- and so declines the dense terminal
#   linearmap  the same operator as a `LinearMaps.LinearMap`, which reaches the same backend
#              through `ProductOperator`; wrapping is all the LinearMaps extension does, so
#              the gap between these two columns is what that route costs
#   matrix     the `n×n` matrix the operator names, which stops at `ReducedCholesky`
#
# All three are the same problem, so the columns are a price for never forming the matrix
# rather than a comparison of problems.
#
# Setup and one `admm_step!` are timed separately: setup is where the dense route pays its
# `O(n³)` factorization and the lazy one pays nothing, and the step is where the lazy route
# pays for an inexact inner solve every iteration.
#
# Swept at two BLAS thread counts. `ReducedCholesky`'s step is a threaded `symv` and the
# conjugate-gradient inner solve is a chain of products, so a single-threaded sweep answers a
# different question than the one the deployed solver -- which pins no threads -- runs.
#
# Run:  cd bench && jl operator_protocol.jl
using PureOSQP, LinearAlgebra
using Krylov                   # the matrix-free backend, a weak dependency
using LinearMaps               # the wrapper route, also a weak dependency
using Chairmarks, Printf, JSON, Statistics, Random

include(joinpath(@__DIR__, "lazy_operator.jl"))

const SIZES = (500, 1000)
const THREADS = (1, min(8, Sys.CPU_THREADS))
const BUDGET = 0.3
const RESULTS = joinpath(@__DIR__, "results", "operator_protocol.json")

"""
    problem(n) -> (P, q, A, l, u)

A box-constrained QP whose `P` is a positive diagonal plus a nonnegative rank-one term, and
whose `A` is dense. `scaling = 0` throughout: equilibration walks columns, which a
products-only operator cannot answer without the traversal overrides this file does not add.
"""
function problem(n)
    rng = MersenneTwister(4231 + 7 * n)
    d = rand(rng, n) .+ 2.0
    v = randn(rng, n) ./ sqrt(n)
    P = LazyPSD(d, v, 0.5)
    A = randn(rng, n, n) ./ sqrt(n)
    b = A * randn(rng, n)
    return P, randn(rng, n), A, b .- rand(rng, n), b .+ rand(rng, n)
end

const OPTS = (scaling = 0, eps_abs = 1.0e-8, eps_rel = 1.0e-8)

"""
    as_linearmap(P::LazyPSD) -> LinearMap

The same operator expressed through LinearMaps, applying the identical closure.

`issymmetric` and `isposdef` are stated rather than derived: a `LinearMap` reports what its
author declared and infers nothing from what the map computes, and [`PureOSQP.is_convex`](@ref)
reads those declarations for an operator. `alpha >= 0` in [`LazyPSD`](@ref) is what makes both
true here.
"""
as_linearmap(P::LazyPSD{T}) where {T} = LinearMap{T}(
    (y, x) -> P.apply!(y, x), length(P.d);
    ismutating = true, issymmetric = true, isposdef = true
)

med(x) = median(s.time for s in x.samples)

rows = NamedTuple[]
for nthreads in THREADS
    BLAS.set_num_threads(nthreads)
    @printf("\nBLAS threads = %d\n", nthreads)
    @printf(
        "%6s | %10s %10s %10s %8s | %10s %10s %10s %8s\n",
        "n", "proto set", "lmap set", "matrix set", "lmap/pro",
        "proto step", "lmap step", "matrix stp", "lmap/pro"
    )
    println("-"^96)
    for n in SIZES
        P, q, A, l, u = problem(n)
        Pm = materialize(P)
        Plm = as_linearmap(P)

        lazy = PureOSQP.setup(P, q, A, l, u; OPTS...)
        lmap = PureOSQP.setup(Plm, q, A, l, u; OPTS...)
        dense = PureOSQP.setup(Pm, q, A, l, u; OPTS...)
        ln = String(PureOSQP.backend_name(lazy.linsys))
        mn = String(PureOSQP.backend_name(lmap.linsys))
        dn = String(PureOSQP.backend_name(dense.linsys))
        # The times mean nothing unless the three spellings are the same problem, and the
        # wrapper comparison means nothing unless it reached the same backend.
        ln == mn || error("n=$n: LinearMap took $mn where the direct operator took $ln")
        PureOSQP.admm_step!(lazy)
        PureOSQP.admm_step!(lmap)
        PureOSQP.admm_step!(dense)
        isapprox(lazy.x, dense.x; rtol = 1.0e-6) || error("n=$n: the spellings disagree")
        isapprox(lazy.x, lmap.x; rtol = 1.0e-6) || error("n=$n: the wrapper changed the answer")
        # Not the same preconditioner, and the gap between the columns is mostly this. A
        # `LazyPSD` answers the reduced diagonal in closed form; a `ProductOperator` has no
        # entries to answer it from and runs unpreconditioned, which `probe` does not change
        # -- probing serves equilibration's column norms, not this seam.
        # Whether the backend is preconditioned at all: an `IdentityPreconditioner` carries no
        # values, and a Jacobi one built from a diagonal of ones is the identity in effect.
        preconditioned(ls) =
            ls.precond isa PureQPBase.JacobiPreconditioner && !all(isone, ls.precond.dinv)
        lazy_prec = preconditioned(lazy.linsys)
        map_prec = preconditioned(lmap.linsys)

        sl = @be PureOSQP.setup($P, $q, $A, $l, $u; OPTS...) seconds = BUDGET
        sm = @be PureOSQP.setup($Plm, $q, $A, $l, $u; OPTS...) seconds = BUDGET
        sd = @be PureOSQP.setup($Pm, $q, $A, $l, $u; OPTS...) seconds = BUDGET
        vl = @be PureOSQP.admm_step!($lazy) seconds = BUDGET
        vm = @be PureOSQP.admm_step!($lmap) seconds = BUDGET
        vd = @be PureOSQP.admm_step!($dense) seconds = BUDGET
        tsl, tsm, tsd = med(sl), med(sm), med(sd)
        tvl, tvm, tvd = med(vl), med(vm), med(vd)
        push!(
            rows, (;
                n, blas_threads = nthreads,
                protocol_backend = ln, linearmap_backend = mn, matrix_backend = dn,
                protocol_setup = tsl, linearmap_setup = tsm, matrix_setup = tsd,
                matrix_over_protocol_setup = tsd / tsl, linearmap_over_protocol_setup = tsm / tsl,
                protocol_step = tvl, linearmap_step = tvm, matrix_step = tvd,
                matrix_over_protocol_step = tvd / tvl, linearmap_over_protocol_step = tvm / tvl,
                protocol_preconditioned = lazy_prec, linearmap_preconditioned = map_prec,
            )
        )
        @printf(
            "%6d | %7.3f ms %7.3f ms %7.3f ms %6.2fx | %7.3f µs %7.3f µs %7.3f µs %6.2fx\n",
            n, 1.0e3tsl, 1.0e3tsm, 1.0e3tsd, tsm / tsl,
            1.0e6tvl, 1.0e6tvm, 1.0e6tvd, tvm / tvl
        )
        flush(stdout)
    end
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "cpu_threads" => Sys.CPU_THREADS,
            "budget_seconds" => BUDGET,
            "sizes" => collect(SIZES),
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
