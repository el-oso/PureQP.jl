# A real `juliac --trim=safe` build of the active-set path, which is the authoritative answer
# to "does this code trim". TrimCheck and StrictModeTest both run the same reachability
# analysis in-process and are what the test suites gate on; neither compiles a program, and
# neither sees the patches `juliac` applies to Base before its own trim inference. This script
# is the oracle those two approximate.
#
# `juliac` ships as an app rather than in `share/julia`: install it with
# `juliaup add 1.12 && julia +1.12 -e 'using Pkg; Pkg.Apps.add("JuliaC")'`, which puts it in
# `~/.julia/bin`. The Base patches it needs live in `share/julia/juliac` of the Julia it
# builds against, present in 1.12 and absent in 1.13.1.
#
# Measured on 1.12.7, juliac 0.3.8: a 3.4 MB shared library exporting `daqp_solve`.

using Printf

const ENTRY = joinpath(@__DIR__, "results", "juliac_entry.jl")
const OUT = joinpath(@__DIR__, "results", "libpuredaqp_trim.so")

mkpath(dirname(ENTRY))

# `@ccallable` wrappers with concrete signatures. `juliac --trim` analyses from a call, not from
# a module, so the entry point is what fixes the types the whole call graph is compiled for.
# Both wrappers include `setup`, which the test suites do not assert: it reaches `eigen` for a
# Kronecker `P`, and stock Base despecializes `eigen`'s error messages into a call `--trim`
# cannot resolve, so an in-process verifier refuses code this build accepts.
write(
    ENTRY, """
    module DAQPTrim

    using PureDAQP
    using PureQPBase
    using LinearAlgebra

    Base.@ccallable function daqp_solve(reps::Cint)::Cint
        P = Matrix{Float64}(I, 2, 2)
        q = [1.0, 1.0]
        A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
        l = [1.0, 0.0, 0.0]
        u = [1.0, 0.7, 0.7]
        ws = setup(P, q, A, l, u, ActiveSet())
        iters = 0
        for _ in 1:reps
            iters += solve!(ws).iter
        end
        return Cint(iters)
    end

    # A Kronecker pair, which reaches `KroneckerCholesky` or `KroneckerSquareRoot` by whether
    # the factors are positive definite, and `ImplicitRows` for the reduction.
    Base.@ccallable function daqp_solve_kron(reps::Cint)::Cint
        P = PureQPBase.KroneckerOperator([2.0 0.5; 0.5 3.0], [4.0 1.0; 1.0 2.0])
        A = PureQPBase.KroneckerOperator([1.0 0.5; 0.0 1.0; 1.0 1.0], [1.0 0.0; 0.5 1.0])
        q = [1.0, 1.0, 0.5, -0.5]
        l = fill(-1.0, 6)
        u = fill(1.0, 6)
        ws = setup(P, q, A, l, u, ActiveSet())
        iters = 0
        for _ in 1:reps
            iters += solve!(ws).iter
        end
        return Cint(iters)
    end

    end
    """
)

juliac = Sys.which("juliac")
if isnothing(juliac)
    @info "juliac is not on PATH; install it as the JuliaC app and re-run"
    exit(0)
end

cmd = `$juliac --output-lib $OUT --project=$(@__DIR__) --trim=safe --experimental
    --compile-ccallable $ENTRY`
@info "building" cmd
ok = success(pipeline(cmd; stdout = devnull, stderr = devnull))

if ok && isfile(OUT)
    @printf("trimmed library: %s, %.1f MB\n", basename(OUT), filesize(OUT) / 1024^2)
    # The symbol has to be exported for a C caller to reach it, so its absence is a failed
    # build that happened to produce a file.
    syms = read(`nm -D $OUT`, String)
    for s in ("daqp_solve", "daqp_solve_kron")
        @printf("%-16s exported: %s\n", s, occursin(s, syms))
    end
else
    @info "the trimmed build failed; run the command above without the output redirect to see why"
end
