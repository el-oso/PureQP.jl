# Every problem of a serialized LinearMaps data set, solved by the three algorithms in three
# representations, with the backend each reached, its status, iterations, time and distance from
# the stored solution. Writes bench/results/lmp_matrix.json.
#
#     julia --project=bench bench/lmp_matrix.jl [datadir] [out.json] [--reps dense,sparse,…]
#     julia --compile=min --project=bench bench/lmp_matrix.jl --summary [results.json]
#
# `--compile=min` on the summary: rendering the table is about ten milliseconds of work, and
# everything else the invocation costs is compiling `JSON`'s parser and this file's own grouping
# for the `Any`-typed values a parsed document holds. Asking for no compilation prints the same
# table in a quarter of the time. It is wrong for a solving run, where the kernels are the work.
#
# `datadir` holds `*.lmp` and `*.lmpr` files, each a `Serialization`-written NamedTuple
# `(H, f, A, b, x0, solution)` standing for `min ½xᵀHx + fᵀx  s.t.  Ax ≥ b`, with `H` and `A`
# as `LinearMaps.LinearMap`s. `x0` is the unconstrained minimizer and `solution` the answer the
# data's own solver reached.
#
# Three representations of each problem, which is what decides the backend a cell reaches:
#
#   unmaterialized   the maps as deserialized, through the LinearMaps extension's `unwrap`
#   dense            `Matrix(H)`, `Matrix(A)`
#   sparse           `sparse(Matrix(H))`, `sparse(Matrix(A))`
#
# Rebuilding the maps into the base's own structured types is not a fourth: `unwrap` routes a map
# to the same backend the rebuilt type reaches.
#
# The settings are an axis rather than the ladder's choice: `OperatorSplitting` and
# `InteriorPoint` take each `linsys` name, `OperatorSplitting` also four tolerance and polish
# variants, and `ActiveSet` each `working_set` crossed with each `scan`. A forced backend the pair
# does not admit is refused by `setup`, and that refusal is the cell's result rather than an
# omission — the message a caller sees is the thing under test.
#
# Every cell records what was asked for beside what was built, and `rel_kkt_error`,
# `active_rows`, `polished` and `status_polish` beside the status. Each carries information the
# status does not: the dual test is relative to the size of the data, so `SOLVED` can hold a
# distant point; a polished point that is no improvement is discarded, leaving a cell that asked
# for polishing indistinguishable from one that did not; and `active_rows` is what sets an
# active-set method's cost. `README.md` in `lmp-portable/` reads the columns.
#
# Each cell first runs a short throwaway solve that pays compilation, which for an operator naming
# its blocks lands once per distinct block structure, and a BLAS library's first calls. It is then
# solved twice, the second time on a workspace built afresh, and the second times are reported. A
# solve that spends `TIME_LIMIT` is not repeated, so such a cell costs one budget rather than two.
# The first timed solve's time is kept, and `cell_s` is the whole wall clock.
# `BLAS_BACKEND` (default `mkl`) and `BLAS_THREADS` (default 4) pick the BLAS library and its threads.
#
# Equilibration reads entries, so a representation supplying products only is set up with
# `scaling = 0` where the default refuses it; the cell records the scaling it ran with.
#
# The CPU clock on the host this usually runs on is not pinned, so the times are indicative
# rather than a gate.

# MKL is loaded before any package so that every BLAS call in the run goes through it;
# `BLAS_BACKEND=openblas` keeps Julia's default library instead.
if lowercase(get(ENV, "BLAS_BACKEND", "mkl")) == "mkl"
    using MKL
end

using PureQPBase, PureOSQP, PureIPM, PureDAQP
using LinearAlgebra, SparseArrays, LinearMaps, Krylov, LDLFactorizations
using Serialization, JSON, Printf

# BLAS threads for every cell, from `BLAS_THREADS` (default 4). The structured products and the
# triangular products thread; one thread measures the arithmetic alone.
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "4")))

# Seconds of iteration a cell is given. `OperatorSplitting` and `InteriorPoint` check the budget
# every iteration and stop with `TIME_LIMIT_REACHED`, so a problem neither can finish costs this
# much and not the full iteration count.
#
# Two parts of a cell are outside it. `setup` is not iteration, so a factorization runs to
# completion however long it takes — `linsys = :kkt` factors an `(n+m)×(n+m)` matrix, which is the
# one to watch as problems grow. And `ActiveSet` has no time-limit check at all, so it runs to its
# own termination; its cost is bounded by the working set instead, which cannot exceed `n` rows.
const TIME_LIMIT = 0.5
# The warm-up solve before each cell's timed ones: enough iterations to reach the factorization,
# the weight update and the termination check, and a budget that compilation cannot exhaust.
const WARMUP_ITERS = 60
const WARMUP_LIMIT = 120.0
# Every name `linsys` accepts, taken from the base rather than listed here, so a backend added
# there is measured without editing this file. A name the pair does not admit is refused by
# `setup`, and that refusal is the cell's result.
#
# `:auto` is run alongside the forced names: it is the ladder's own choice, which is what decides
# whether an unmaterialized input reaches a direct backend wherever its dense form does, and it
# is the only way to reach a backend that has no `linsys` name, such as `product_reduced`.
const LINSYS = PureQPBase.LINSYS_OPTIONS
const WORKING_SETS = (:rows, :gram)

# `scan` crossed with the working set rather than left at its default, because which of the two
# wins is a property of the representation and not of the problem. `:window` prices a window of
# the rows instead of all of them, which makes an iteration cheaper and usually costs
# iterations. A stored reduction can price a window; an implicit one reaches every row with one
# product, so there a window saves nothing and the extra iterations are all that is left. The
# cells are what tells those apart on a given data set.
const SCANS = (:all, :window)

# The polish axis runs at `linsys = :auto` alone, because polishing is orthogonal to the backend:
# `polish_kernel!` builds the full KKT system of the rows it takes as active and factors that,
# rather than going through the workspace's backend. Pairing it with each `linsys` name would
# repeat one measurement ten times.
#
# `OperatorSplitting` defaults to `eps_abs = eps_rel = 1e-3`, so the `linsys=…` cells already
# measure the loose end and these move the other way. `tol=1e-6` alone is the control: it
# separates what the tighter iterate buys from what the polish solve buys. The last two differ
# only in `delta`, which regularizes the polish solve and bounds how close its answer can come.
# `delta`'s default is absolute, so on a problem whose `‖Px‖∞` is large it is a loose bound and
# the polished point can come back no better than the unpolished one, which is discarded.
#
# The tight variants raise `max_iter` past its default of 4000. Polishing runs only from a solve
# that terminated, so a variant that stops at the iteration limit reports
# `POLISH_NOT_PERFORMED` and measures nothing about polishing. `TIME_LIMIT` bounds each
# cell, so a problem the iteration cannot finish costs a known amount rather than the full count.
const OSQP_TIGHT_MAX_ITER = 50_000
# Every `OperatorSplitting` cell runs with `scaling = 0`: row equilibration of a problem whose rows
# span orders of magnitude of scale (`E` from 0.06 to 400 here) can leave it unconverged at any
# `rho`, where the unscaled problem solves in a few hundred iterations. `default scaling` is the
# one cell that keeps the algorithm's own scaling, as the control.
const OSQP_VARIANTS = (
    ("default scaling", (; scaling = default_options(OperatorSplitting(), Float64).scaling)),
    ("tol=1e-6", (; eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = OSQP_TIGHT_MAX_ITER)),
    ("polish", (; polishing = true)),
    (
        "tol=1e-6 polish",
        (; polishing = true, eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = OSQP_TIGHT_MAX_ITER),
    ),
    (
        "tol=1e-6 polish delta=1e-10",
        (;
            polishing = true, eps_abs = 1.0e-6, eps_rel = 1.0e-6, delta = 1.0e-10,
            max_iter = OSQP_TIGHT_MAX_ITER,
        ),
    ),
)

"Every `(algorithm, requested, make_alg, setup_kwargs)` a problem is run through."
const SETTINGS = (
    (
        ("OperatorSplitting", "linsys=$ls", () -> OperatorSplitting(), (; linsys = ls, scaling = 0))
            for ls in LINSYS
    )...,
    (
        ("OperatorSplitting", label, () -> OperatorSplitting(), (; linsys = :auto, scaling = 0, kw...))
            for (label, kw) in OSQP_VARIANTS
    )...,
    (("InteriorPoint", "linsys=$ls", () -> InteriorPoint(), (; linsys = ls)) for ls in LINSYS)...,
    # Looser tolerances at `linsys = :auto`: how many iterations a solve that stops early saves,
    # and how far its answer is from the stored solution.
    (
        ("InteriorPoint", "eps=$tol", () -> InteriorPoint(), (; linsys = :auto, eps_abs = tol, eps_rel = tol))
            for tol in (1.0e-6, 1.0e-4)
    )...,
    (
        (
            "ActiveSet", "working_set=$ws scan=$sc",
            () -> ActiveSet(; working_set = ws, scan = sc), (;),
        )
            for ws in WORKING_SETS, sc in SCANS
    )...,
)

# ── the matrix ─────────────────────────────────────────────────────────────────────────────

"What `ws` reached: a backend name, or for the active-set method the row representation."
reached(ws::PureDAQP.ActiveSetWorkspace) = string(nameof(typeof(ws.red.ws.M)))
reached(ws) = string(backend_name(ws.linsys))

"The first line of an exception's message."
firstline(e) = first(split(sprint(showerror, e), '\n'))

"Whether `e` is the refusal to equilibrate an operator that has no entries to read."
needs_unscaled(e) = e isa ArgumentError && occursin("scaling = 0", e.msg)

function timed(f)
    t = time_ns()
    v = f()
    return v, (time_ns() - t) / 1.0e9
end

"""
    run_cell(P, q, A, l, u, make_alg, kwargs, Pd, solution) -> NamedTuple

One problem in one representation on one algorithm with one forced setting. Never throws: a
refusal or an error is a cell too, carrying the message.
"""
function run_cell(P, q, A, l, u, make_alg, kwargs, Pd, solution)
    scaling = get(kwargs, :scaling, default_options(make_alg(), Float64).scaling)
    build(extra = (;)) = setup(
        P, q, A, l, u, make_alg(); merge((; time_limit = TIME_LIMIT, scaling), kwargs, extra)...
    )
    cell_start = time_ns()
    cell_s() = (time_ns() - cell_start) / 1.0e9
    try
        local ws
        try
            ws = build()
        catch e
            needs_unscaled(e) || rethrow()
            scaling = 0
            ws = build()
        end
        ws, setup_s = timed(build)
        # Compilation and a BLAS library's first calls are paid on a throwaway workspace, outside
        # the budget: a solve that pays them inside `TIME_LIMIT` is cut short or reported at the
        # budget however fast it runs, and on a slower host they alone exceed it.
        solve!(build((; time_limit = WARMUP_LIMIT, max_iter = WARMUP_ITERS)))
        sol, first_solve_s = timed(() -> solve!(ws))
        # A solve that spent its whole budget has nothing to gain from a second pass and would
        # double the cell's wall time, so only a solve that terminated inside the budget is
        # repeated to get a time that is not compilation. The repeat is on a workspace built
        # afresh: `cold_start!` drops the iterates but keeps what the solver adapted, such as
        # `OperatorSplitting`'s `rho` and its factorization, so a second solve on the same
        # workspace is not the solve a new caller gets.
        resolved = first_solve_s < TIME_LIMIT
        solve_s = first_solve_s
        if resolved
            ws = build()
            sol, solve_s = timed(() -> solve!(ws))
        end
        x = sol.x
        obj(v) = dot(v, Pd * v) / 2 + dot(q, v)
        slack = A * x .- l
        # Each row against its own scale: one absolute tolerance across rows whose bounds span
        # orders of magnitude counts either every row or none.
        active_rows = count(i -> slack[i] <= 1.0e-7 * max(1.0, abs(l[i])), eachindex(slack))
        return (;
            backend = reached(ws), status = string(sol.status), iterations = sol.iter, scaling,
            setup_s, first_solve_s, solve_s, resolved, cell_s = cell_s(),
            err_inf = maximum(abs, x .- solution), obj_gap = obj(x) - obj(solution),
            min_slack = minimum(slack), active_rows,
            rel_kkt_error = sol.rel_kkt_error, prim_res = sol.prim_res, dual_res = sol.dual_res,
            polished = sol.polished, status_polish = string(sol.status_polish),
            polish_s = sol.polish_time, error = nothing,
        )
    catch e
        return (;
            backend = nothing, status = "ERROR", iterations = 0, scaling,
            setup_s = NaN, first_solve_s = NaN, solve_s = NaN, resolved = false, cell_s = cell_s(),
            err_inf = NaN, obj_gap = NaN, min_slack = NaN, active_rows = 0,
            rel_kkt_error = NaN, prim_res = NaN, dual_res = NaN,
            polished = false, status_polish = "POLISH_NOT_PERFORMED", polish_s = NaN,
            error = firstline(e),
        )
    end
end

"`v`, with a non-finite float as `nothing`, which JSON can hold."
jsonable(v) = v isa AbstractFloat && !isfinite(v) ? nothing : v

"The `(family, kind)` a file name carries: `test_101.…lmpr` is family 101, kind `lmpr`."
function family_of(name)
    m = match(r"test_(\d+)\..*\.(lmpr?)$", name)
    return isnothing(m) ? ("?", "?") : (m[1], m[2])
end

const REPRESENTATIONS = ("unmaterialized", "dense", "sparse")

"The representations in `reps` of one deserialized problem, as `(label, P, A)`."
function representations(d, reps)
    Hd, Ad = Matrix(d.H), Matrix(d.A)
    all = (
        ("unmaterialized", () -> (d.H, d.A)),
        ("dense", () -> (Hd, Ad)),
        ("sparse", () -> (sparse(Hd), sparse(Ad))),
    )
    return [(label, build()...) for (label, build) in all if label in reps]
end

function run_matrix(datadir, out; reps = REPRESENTATIONS)
    files = sort(filter(f -> endswith(f, ".lmp") || endswith(f, ".lmpr"), readdir(datadir; join = true)))
    isempty(files) && error(lazy"no .lmp or .lmpr files in $datadir")
    cells = Dict{String, Any}[]
    for (k, f) in enumerate(files)
        d = deserialize(f)
        q, l = d.f, d.b
        n, m = length(q), length(l)
        u = fill(Inf, m)
        Pd = Matrix(d.H)
        family, kind = family_of(basename(f))
        @printf("[%2d/%d] %s  n=%d m=%d\n", k, length(files), basename(f), n, m)
        flush(stdout)
        for (rep, P, A) in representations(d, reps), (alg, requested, make_alg, kwargs) in SETTINGS
            c = run_cell(P, q, A, l, u, make_alg, kwargs, Pd, d.solution)
            @printf(
                "   %-14s %-18s %-30s %-16s %-18s it=%4d  solve=%8.4fs  err=%.2e  act=%4d%s%s\n",
                rep, alg, requested, something(c.backend, "-"), c.status, c.iterations, c.solve_s,
                c.err_inf, c.active_rows,
                c.status_polish == "POLISH_NOT_PERFORMED" ? "" : "  " * c.status_polish,
                isnothing(c.error) ? "" : "  ERROR: " * c.error
            )
            flush(stdout)
            push!(
                cells, Dict(
                    "file" => basename(f), "family" => family, "kind" => kind, "n" => n, "m" => m,
                    "representation" => rep, "algorithm" => alg, "requested" => requested,
                    (string(key) => jsonable(v) for (key, v) in pairs(c))...,
                )
            )
        end
    end
    meta = Dict(
        "julia" => string(VERSION), "host" => gethostname(), "datadir" => abspath(datadir),
        "time_limit" => TIME_LIMIT, "generated" => string(now_utc()),
        "blas" => string(BLAS.get_config()), "blas_threads" => BLAS.get_num_threads(),
        "commit" => try
            readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
        catch
            "unknown"
        end,
        "packages" => Dict(
            string(nameof(M)) => string(pkgversion(M)) for M in (PureQPBase, PureOSQP, PureIPM, PureDAQP)
        ),
    )
    mkpath(dirname(out))
    open(out, "w") do io
        JSON.print(io, Dict("meta" => meta, "cells" => cells), 2)
    end
    println("wrote ", out)
    return out
end

now_utc() = Base.Libc.strftime("%Y-%m-%dT%H:%M:%SZ", time())

# ── the summary, from the saved JSON alone ─────────────────────────────────────────────────

"""
Print one line per `(family, kind, representation, algorithm, requested)`: the backends built,
the statuses, the iterations, the median solve time, the largest distance from the stored
solution, and the errors with their first distinct message.
"""
function summarize(path)
    data = JSON.parsefile(path)
    cells = data["cells"]
    key(c) = (c["family"], c["kind"], c["representation"], c["algorithm"], c["requested"])
    groups = Dict{NTuple{5, String}, Vector{Any}}()
    for c in cells
        push!(get!(groups, key(c), Any[]), c)
    end
    println(
        rpad("family", 10), rpad("representation", 16), rpad("algorithm", 19), rpad("requested", 30),
        rpad("backends", 22), rpad("statuses", 24), rpad("iters", 8), rpad("med solve s", 13),
        rpad("max err", 10), rpad("polished", 10), rpad("max relkkt", 12), "errors"
    )
    for k in sort(collect(keys(groups)))
        g = groups[k]
        backends = join(sort(unique(something.(get.(g, "backend", nothing), "-"))), ",")
        statuses = Dict{String, Int}()
        for c in g
            statuses[c["status"]] = get(statuses, c["status"], 0) + 1
        end
        st = join(("$s×$v" for (s, v) in sort(collect(statuses))), ",")
        solved = filter(c -> isnothing(c["error"]), g)
        med = isempty(solved) ? NaN : median_of([something(c["solve_s"], NaN) for c in solved])
        # The iteration count is what a `MAX_ITER_REACHED` cell is reporting, and it is the
        # quantity that moves when a setting changes which row enters, so it is shown beside the
        # time rather than left in the JSON. A range where the repetitions disagree.
        its = sort(unique(something(c["iterations"], -1) for c in solved))
        iters = isempty(its) ? "-" :
            length(its) == 1 ? string(its[1]) : string(its[1], "-", its[end])
        # A non-finite distance was written as `null`; it is reported as NaN.
        err = isempty(solved) ? NaN : maximum(something(c["err_inf"], NaN) for c in solved)
        errors = unique(c["error"] for c in g if !isnothing(c["error"]))
        # A cell that asked for polishing and had it refused returns the unpolished point, so the
        # count is of cells that got one, against the cells that asked.
        asked = count(c -> get(c, "status_polish", "POLISH_NOT_PERFORMED") != "POLISH_NOT_PERFORMED", g)
        got = count(c -> get(c, "polished", false) === true, g)
        relkkt = isempty(solved) ? NaN : maximum(something(get(c, "rel_kkt_error", nothing), NaN) for c in solved)
        println(
            rpad(k[1] * "." * k[2], 10), rpad(k[3], 16), rpad(k[4], 19), rpad(k[5], 30),
            rpad(backends, 22), rpad(st, 24), rpad(iters, 8),
            rpad(@sprintf("%.4f", med), 13), rpad(@sprintf("%.1e", err), 10),
            rpad(iszero(asked) ? "-" : string(got, "/", asked), 10),
            rpad(@sprintf("%.1e", relkkt), 12),
            isempty(errors) ? "0" : string(length(g) - length(solved), ": ", first(errors))
        )
    end
    return nothing
end

function median_of(v)
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

function main(args)
    if !isempty(args) && args[1] == "--summary"
        return summarize(get(args, 2, joinpath(@__DIR__, "results", "lmp_matrix.json")))
    end
    reps = REPRESENTATIONS
    i = findfirst(==("--reps"), args)
    if !isnothing(i)
        reps = Tuple(split(args[i + 1], ','))
        args = args[setdiff(eachindex(args), (i, i + 1))]
    end
    datadir = get(args, 1, "/tmp")
    out = get(args, 2, joinpath(@__DIR__, "results", "lmp_matrix.json"))
    run_matrix(datadir, out; reps)
    return summarize(out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
