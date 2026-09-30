# StrictMode gate for PureOSQP's hot path.
#
# Run:  julia --project=bench bench/strictmode_audit.jl
# The global Stop hook runs this automatically after any turn that touches src/.
#
# The gate is StrictModeTest's `test_signatures`, which proves each guarantee with AllocCheck
# and JET. StrictMode's own value-free scan agrees with it on this package's hot path, but it
# only reports, so it cannot gate.
using PureOSQP
using PureIPM                  # supplies the interior-point algorithm
using PureDAQP                 # supplies the dual active-set algorithm
using PureQPBase               # holds the backends and their extensions
using Krylov                   # supplies the :indirect backend, a weak dependency
using LDLFactorizations        # supplies the LDLᵀ backends, likewise
using BandedMatrices           # supplies the banded backend, likewise
using StrictMode, StrictModeTest
using LinearAlgebra, SparseArrays, Random
using InteractiveUtils: subtypes

include(joinpath(@__DIR__, "..", "PureOSQP", "bench", "lazy_operator.jl"))

# A disabled audit prints exactly like a clean one. Never report a pass without this.
StrictMode.assert_enabled()

"""
Refuse to run unless every algorithm in the repository is covered below.

The kernel lists in this file are written by hand, so an algorithm added without a section
here would leave the audit printing a clean pass over a hot path nobody checked -- and a
skipped check looks exactly like a passing one from outside. Two things are asserted, because
either alone leaves a way through: every subdirectory package must be loaded, and every
`QPAlgorithm` that loading them defines must appear in `AUDITED`.

This is the same reasoning as `@verify QPAlgorithm subtypes = true` in each package: a
per-algorithm opt-in is only as good as whoever remembered to opt in.
"""
const AUDITED = Set{Type}(
    [PureOSQP.OperatorSplitting, PureIPM.InteriorPoint, PureDAQP.ActiveSet]
)

let root = dirname(@__DIR__)
    # A package, not merely a directory with a Project.toml: `docs/` has one of those, and a
    # `src/` full of Markdown. The entry point `src/<name>.jl` is what distinguishes them.
    packages = filter(readdir(root)) do d
        isfile(joinpath(root, d, "Project.toml")) && isfile(joinpath(root, d, "src", d * ".jl"))
    end
    unloaded = filter(p -> isnothing(Base.find_package(p)) || !isdefined(Main, Symbol(p)), packages)
    isempty(unloaded) || error(
        "strictmode_audit.jl does not load " * join(unloaded, ", ") *
            ": add `using` for it above and give it a section below, or its hot path ships unaudited."
    )
    unaudited = setdiff(Set{Type}(subtypes(PureQPBase.QPAlgorithm)), AUDITED)
    isempty(unaudited) || error(
        "no StrictMode section for " * join(string.(collect(unaudited)), ", ") *
            ": add its kernels below and list it in AUDITED."
    )
end

function example_workspace(backend::Symbol)
    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    if backend === :sparse
        # The sparse backend is chosen by representation and density, not by a setting, so
        # it is reached by handing `setup` sparse matrices sparse enough to clear the gate.
        return example_sparse_workspace(n, m)
    elseif backend === :cholmod
        # Banded, so the reduced matrix's factor stays sparse enough to be worth keeping.
        return example_banded_workspace(200, 400)
    elseif backend === :sparse_kkt
        return example_kkt_workspace(200, 100)
    elseif backend === :banded
        return example_banded_backend_workspace(200)
    elseif backend === :tridiagonal
        return example_tridiagonal_workspace(200)
    elseif backend === :lowrank
        return example_lowrank_workspace(200, 3)
    elseif backend === :kronecker
        return example_kronecker_workspace(12, 10)
    elseif backend === :block
        return example_block_workspace(200, 5)
    elseif backend === :operator
        return example_operator_workspace(200)
    elseif backend === :productoperator
        return example_product_operator_workspace(200)
    elseif backend === :diagonal
        # Chosen by representation, like the sparse backends: no setting reaches it.
        return example_diagonal_workspace(200)
    end
    ws = PureOSQP.setup(P, q, A, l, u; linsys = backend)
    PureOSQP.solve!(ws)        # compile every specialization before analysing it
    return ws
end

function example_banded_backend_workspace(n)
    Random.seed!(10)
    # A `Tridiagonal` A squares to bandwidth 2, past what `SymTridiagonal` holds.
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n))
    PureOSQP.solve!(ws)
    return ws
end

function example_tridiagonal_workspace(n)
    Random.seed!(9)
    P = SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8)
    A = Diagonal(rand(n) .+ 0.5)
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n))
    PureOSQP.solve!(ws)
    return ws
end

function example_kronecker_workspace(n1, n2)
    Random.seed!(15)
    K = PureOSQP.KroneckerOperator(randn(n1, n1), randn(n2, n2))
    n = n1 * n2
    b = Matrix(K) * randn(n)
    ws = PureOSQP.setup(
        Diagonal(fill(2.0, n)), randn(n), K, b .- rand(n), b .+ rand(n); scaling = 0
    )
    PureOSQP.solve!(ws)
    return ws
end

"A random symmetric positive definite block of side `k`."
function spd_block(k)
    S = randn(k, k)
    return Matrix(Symmetric(S'S ./ k + 3I))
end

function example_block_workspace(n, K)
    Random.seed!(14)
    nb = n ÷ K
    P = PureOSQP.BlockDiagonal([spd_block(nb) for _ in 1:K])
    A = PureOSQP.BlockDiagonal([randn(nb, nb) ./ sqrt(nb) for _ in 1:K])
    m = size(A, 1)
    b = randn(m)
    ws = PureOSQP.setup(P, randn(size(A, 2)), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

function example_lowrank_workspace(n, k)
    Random.seed!(11)
    P = Diagonal(rand(n) .+ 0.5)
    A = PureOSQP.RowCoupled(randn(k, n) ./ 4, ones(n - k), collect(1:(n - k)))
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n))
    PureOSQP.solve!(ws)
    return ws
end

"""
A workspace over a caller-supplied operator that stores no matrix at all: `P` applies
`Diagonal(d) + α v vᵀ` through a closure and declares `is_materializable` false, so the
ladder descends past the dense terminal to the matrix-free rung.

What this row measures is narrower than the others. The hot path here runs the caller's
`mul!`, so `noalloc` and `typestable` on `admm_step!` hold only as far as that `mul!` does;
the operator in `lazy_operator.jl` is written to carry them, and this row is what shows that
the surrounding machinery does not take them away.
"""
function example_operator_workspace(n)
    Random.seed!(12)
    P = LazyPSD(rand(n) .+ 2.0, randn(n) ./ sqrt(n), 0.5)
    A = randn(n, n) ./ sqrt(n)
    b = A * randn(n)
    # `scaling = 0`: equilibration walks columns, and this operator supplies products only.
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(n), b .+ rand(n); scaling = 0)
    PureOSQP.solve!(ws)
    return ws
end

# `LazyPSD` above is an operator written as an `AbstractMatrix` directly. `ProductOperator`
# is the other route -- a wrapper around a hierarchy that is not one -- and it reaches the
# solve path through a different concrete `OperatorSplittingWorkspace` type, so it is analysed separately.
function example_product_operator_workspace(n)
    Random.seed!(13)
    S = randn(n, n)
    P = PureOSQP.ProductOperator{Float64}(
        Symmetric(S'S ./ n + 8I); symmetric = true, posdef = true
    )
    A = PureOSQP.ProductOperator{Float64}(randn(n, n) ./ sqrt(n))
    b = randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(n), b .+ rand(n); scaling = 0)
    PureOSQP.solve!(ws)
    return ws
end

function example_diagonal_workspace(n)
    Random.seed!(7)
    P, A = Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5)
    l, u = -rand(n), rand(n)
    ws = PureOSQP.setup(P, randn(n), A, l, u)
    PureOSQP.solve!(ws)
    return ws
end

function example_banded_workspace(n, m)
    Random.seed!(3)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:m, j in max(1, div(i * n, m) - 2):min(n, div(i * n, m) + 2)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn())
    end
    A = sparse(rows, cols, vals, m, n)
    S = spdiagm(-1 => randn(n - 1), 0 => randn(n), 1 => randn(n - 1))
    P = sparse(Symmetric(S'S)) + 3.0I
    b = A * randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

"""
A problem the full quasi-definite KKT backend is chosen for: a budget row touching every
column, which squares into a dense reduced matrix while leaving `K` sparse. This is what the
OSQP suite's Portfolio class looks like.
"""
function example_kkt_workspace(n, m)
    Random.seed!(8)
    A = vcat(sprandn(m - 1, n, 0.02), sparse(ones(1, n)))
    P = sparse(1.0I, n, n)
    b = A * randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

function example_sparse_workspace(n, m)
    Random.seed!(2)
    A = sprandn(m, n, 0.05)
    S = sprandn(n, n, 0.05)
    P = sparse(Symmetric(S'S)) + (n * 0.05 + 1) * I
    b = A * randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

function example_ipm_workspace(backend::Symbol)
    Random.seed!(101)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    if backend === :sparse_kkt
        return example_ipm_kkt_workspace(200, 100)
    elseif backend === :diagonal
        return example_ipm_diagonal_workspace(200)
    elseif backend === :tridiagonal
        return example_ipm_tridiagonal_workspace(200)
    elseif backend === :banded
        return example_ipm_banded_workspace(200)
    elseif backend === :block
        return example_ipm_block_workspace(200, 5)
    elseif backend === :indirect
        return example_ipm_indirect_workspace(200)
    end
    ws = PureOSQP.setup(P, q, A, l, u, PureIPM.InteriorPoint(); linsys = backend)
    PureOSQP.solve!(ws)
    return ws
end

"""
A pair the sparse KKT ladder rung factors, the same shape as `example_kkt_workspace`. Naming
`linsys = :kkt` under the interior-point method always builds `FullKKT` (dense), unlike ADMM,
so `:sparse` is what reaches the sparse KKT rung here.
"""
function example_ipm_kkt_workspace(n, m)
    Random.seed!(108)
    A = vcat(sprandn(m - 1, n, 0.02), sparse(ones(1, n)))
    P = sparse(1.0I, n, n)
    b = A * randn(n)
    ws = PureOSQP.setup(
        P, randn(n), A, b .- rand(m), b .+ rand(m), PureIPM.InteriorPoint(); linsys = :sparse
    )
    PureOSQP.solve!(ws)
    return ws
end

function example_ipm_diagonal_workspace(n)
    Random.seed!(107)
    P, A = Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5)
    l, u = -rand(n), rand(n)
    ws = PureOSQP.setup(P, randn(n), A, l, u, PureIPM.InteriorPoint())
    PureOSQP.solve!(ws)
    return ws
end

function example_ipm_tridiagonal_workspace(n)
    Random.seed!(109)
    P = SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8)
    A = Diagonal(rand(n) .+ 0.5)
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n), PureIPM.InteriorPoint())
    PureOSQP.solve!(ws)
    return ws
end

function example_ipm_banded_workspace(n)
    Random.seed!(110)
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n), PureIPM.InteriorPoint())
    PureOSQP.solve!(ws)
    return ws
end

function example_ipm_block_workspace(n, K)
    Random.seed!(111)
    nb = n ÷ K
    P = PureOSQP.BlockDiagonal([spd_block(nb) for _ in 1:K])
    A = PureOSQP.BlockDiagonal([randn(nb, nb) ./ sqrt(nb) for _ in 1:K])
    m = size(A, 1)
    b = randn(m)
    ws = PureOSQP.setup(
        P, randn(size(A, 2)), A, b .- rand(m), b .+ rand(m), PureIPM.InteriorPoint()
    )
    PureOSQP.solve!(ws)
    return ws
end

"""
The matrix-free IPM backend, run with a caller-supplied Cholesky preconditioner: the IPM has
no automatic route onto `:indirect`, so it is always named together with a preconditioner.
"""
function example_ipm_indirect_workspace(n)
    Random.seed!(112)
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(n, n) ./ sqrt(n)
    b = A * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    ws = PureOSQP.setup(
        P, q, A, l, u, PureIPM.InteriorPoint();
        linsys = :indirect, scaling = 0, preconditioner = cholesky(Symmetric(P + I))
    )
    PureOSQP.solve!(ws)
    return ws
end

const GUARANTEES = Dict(
    :hot => (:typestable, :noalloc),
    :warm => (:typestable,),
    # The matrix-free backend calls Krylov's `cg!`, which AllocCheck cannot clear: it times
    # itself through an opaque `ccall` to `jl_hrtime`, and its verbose-reporting and
    # residual-history branches are guarded by runtime values, so their `Printf` and
    # `resize!` calls are live code as far as static analysis is concerned even though no
    # solve here takes them. Whitelisting those findings one by one would hollow the gate
    # out, so the claim is split instead: what this package owns is proved statically, and
    # what Krylov owns is measured. See `measured_noalloc` and the `ReducedOperator` check.
    :hot_measured => (:typestable,),
    # `SparseCholmod`, `SparseKKT`, `SparseLDL` and `LDLKKT` all build their matrix through
    # `PureQPBaseSparseArraysExt`'s `reduced_gram`/`kkt_gram`, whose constructors validate
    # dimensions and format the message through code a static analyzer cannot see past -- the
    # same never-taken error path that costs any caller of those constructors its
    # inferrability. That reaches `factorize!`, `refactor_weights!` and, through them,
    # `solve!`: not just `noalloc` but `typestable` too fails for these three, confirmed by
    # `test_signatures` reporting JET's `internal instability / runtime dispatch` finding on
    # `factorize!` and `solve!` here (88 reports on the KKT family). It does not reach the hot
    # path, which keeps every guarantee: `admm_step!`, `update_residuals!` and `solve_system!`
    # are checked here exactly as for every other backend, and `check_termination` -- which
    # never touches the gram assembly -- stays on the plain `:warm` claim below. Stated rather
    # than hidden, because the claim really is narrower here.
    :warm_sparse => (),
)

"""
    measured_noalloc(measure)

Assert a hot-path call allocates nothing, by measurement rather than by static analysis.

This is weaker evidence than AllocCheck and is used only where AllocCheck cannot be
applied. It is paired with a static `:noalloc` check on `ReducedOperator`'s `mul!`, which
is the part of the matrix-free path this package actually writes.
"""
function measured_noalloc(measure)
    measure()   # discard the first, which pays for any lingering compilation
    bytes = measure()
    iszero(bytes) || error("measured $bytes bytes at run time")
    return nothing
end

failures = String[]

for backend in (
        :auto, :kkt, :sparse, :sparse_kkt, :cholmod, :diagonal, :tridiagonal, :banded, :lowrank,
        :indirect, :block, :kronecker, :operator, :productoperator,
    )
    ws = example_workspace(backend)
    W = typeof(ws)
    LS = typeof(ws.linsys)
    PB = typeof(ws.prob)
    WT = typeof(ws.weights)
    V = Vector{Float64}
    # `:operator` and `:productoperator` reach the same matrix-free backend as `:indirect`
    # and take the same exemption for Krylov's `cg!`, whose timing and `allocate_if` branches
    # are statically visible and never taken.
    matrix_free = backend in (:indirect, :operator, :productoperator)
    tier = matrix_free ? :hot_measured : :hot
    # See `:warm_sparse`: only the factorization side is affected, never the hot path.
    # `:cholmod` reaches sparse arithmetic whichever engine factors it -- the reduced matrix
    # is assembled the same way before either sees it. `:sparse_kkt` factors the KKT matrix
    # with the same foreign LDLᵀ code and inherits the same exemption; `solve_multiplier!`
    # is checked separately below, at the hot tier, since it is this package's own code.
    warm = backend in (:cholmod, :sparse_kkt) ? :warm_sparse : :warm
    solve_sys() = @allocated PureOSQP.solve_system!(
        ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde
    )
    step() = @allocated PureOSQP.admm_step!(ws)
    checks = Any[
        (PureOSQP.admm_step!, (W,), tier, step),
        (PureOSQP.update_residuals!, (W,), :hot, nothing),
        (PureOSQP.solve_system!, (LS, PB, WT, V, V, V, V), tier, solve_sys),
        (PureOSQP.check_termination, (W, Bool), :warm, nothing),
        (PureOSQP.factorize!, (LS, PB, WT), warm, nothing),
        # Runs every time `ρ` moves, so it sits inside the solve loop rather than at setup.
        (PureOSQP.refactor_weights!, (LS, PB, WT), warm, nothing),
        (PureOSQP.solve!, (W,), warm, nothing),
    ]
    if backend in (:kkt, :sparse_kkt)
        # ADMM never calls this; the KKT backends' own override is what the guarantee
        # applies to, at the same tier as `solve_system!` since it does the same work.
        solve_mult() = @allocated PureOSQP.solve_multiplier!(
            ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde
        )
        push!(checks, (PureOSQP.solve_multiplier!, (LS, PB, WT, V, V, V, V), tier, solve_mult))
    end
    if matrix_free
        # The operator is this package's own code and gets the full static guarantee, with
        # no exemption: it is where a matrix-free product would allocate if one did.
        op = Base.get_extension(PureQPBase, :PureQPBaseKrylovExt).ReducedOperator(ws.prob, ws.weights)
        push!(checks, (LinearAlgebra.mul!, (V, typeof(op), V), :hot, nothing))
    end
    if backend === :cholmod
        # `factorize!` gets no static claim because it reaches CHOLMOD, but the part this
        # package owns -- rebuilding the reduced matrix -- is plain loops over vectors and
        # carries the full guarantee. A refactorization runs every time `ρ` moves, so an
        # allocation here would land inside the solve loop, not just at setup.
        Ext = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
        G = typeof(ws.linsys.gram)
        M = typeof(ws.prob.P)
        push!(
            checks,
            (Ext.refill!, (G, M, M, V, V, V, Float64, Float64), :hot, nothing)
        )
    end
    for (f, types, tier, measure) in checks
        label = "$(nameof(f))($(join(types, ", "))) [linsys=$backend]"
        try
            isempty(GUARANTEES[tier]) ||
                test_signatures([(f, types)]; guarantees = GUARANTEES[tier])
            extra = ""
            if tier === :hot_measured
                measured_noalloc(measure)
                extra = ", noalloc (measured: 0 bytes)"
            end
            claims = isempty(GUARANTEES[tier]) ? "no static claim (sparse arithmetic)" : join(GUARANTEES[tier], ", ")
            println("  ✓ ", label, "  ", claims, extra)
        catch e
            push!(failures, label)
            println("  ✗ ", label)
            println(sprint(showerror, e))
        end
    end
end
# The interior-point method's own hot kernels, over the backends its ladder actually reaches:
# `FullKKT` (`:auto` on a dense pair), the sparse KKT family (`:kkt` on a sparse pair,
# `LDLKKT` here since `LDLFactorizations` is loaded), the structured reduced backends it
# shares with ADMM, and `:indirect` with a caller preconditioner. `:kronecker` and `:lowrank`
# are absent because the interior-point ladder has no rung for either (uniform weights and,
# respectively, a Woodbury solve that misses the tolerance on linear programs); `:sparse_formed`
# is unreachable under the interior-point method for the same reason, having no `formed_rung`
# method of its own.
for example_kind in (:auto, :sparse_kkt, :diagonal, :tridiagonal, :banded, :block, :indirect)
    ws = example_ipm_workspace(example_kind)
    W = typeof(ws)
    LS = typeof(ws.linsys)
    PB = typeof(ws.prob)
    WT = typeof(ws.weights)
    V = Vector{Float64}
    matrix_free = example_kind === :indirect
    tier = matrix_free ? :hot_measured : :hot
    # The sparse KKT family factors with foreign `LDLᵀ` code, exactly like ADMM's `:sparse_kkt`
    # row above; its own `solve_multiplier!` is this package's code and keeps the full claim.
    warm = example_kind === :sparse_kkt ? :warm_sparse : :warm
    step() = @allocated PureIPM.ipm_step!(ws)
    residuals() = @allocated PureIPM.ipm_residuals!(ws)
    solve_mult() = @allocated PureOSQP.solve_multiplier!(
        ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.dx, ws.dy
    )
    checks = Any[
        (PureIPM.ipm_step!, (W,), tier, step),
        (PureIPM.ipm_residuals!, (W,), tier, residuals),
        (PureOSQP.solve_multiplier!, (LS, PB, WT, V, V, V, V), tier, solve_mult),
        (PureOSQP.check_termination, (W, Bool), :warm, nothing),
        (PureOSQP.factorize!, (LS, PB, WT), warm, nothing),
        (PureOSQP.refactor_weights!, (LS, PB, WT), warm, nothing),
        (PureOSQP.solve!, (W,), warm, nothing),
    ]
    if matrix_free
        op = Base.get_extension(PureQPBase, :PureQPBaseKrylovExt).ReducedOperator(ws.prob, ws.weights)
        push!(checks, (LinearAlgebra.mul!, (V, typeof(op), V), :hot, nothing))
    end
    label_name = PureOSQP.backend_name(ws.linsys)
    for (f, types, tier, measure) in checks
        label = "$(nameof(f))($(join(types, ", "))) [ipm linsys=$label_name]"
        try
            isempty(GUARANTEES[tier]) ||
                test_signatures([(f, types)]; guarantees = GUARANTEES[tier])
            extra = ""
            if tier === :hot_measured
                measured_noalloc(measure)
                extra = ", noalloc (measured: 0 bytes)"
            end
            claims = isempty(GUARANTEES[tier]) ? "no static claim (sparse arithmetic)" : join(GUARANTEES[tier], ", ")
            println("  ✓ ", label, "  ", claims, extra)
        catch e
            push!(failures, label)
            println("  ✗ ", label)
            println(sprint(showerror, e))
        end
    end
end

# ---------------------------------------------------------------- PureDAQP
#
# The dual active-set method has no `LinearSystem` backend, so there is nothing to sweep over:
# it maintains one `LDLᵀ` of the working set's Gram matrix. What runs per iteration is the
# row-update trio and the triangular solves, and those are what carry the guarantee.
#
# `solve_ldp!` is the whole loop, and it is proved here like the rest: it takes its
# tolerances as an `ActiveSet` rather than as keywords, which is a signature
# `test_signatures` can state.
let
    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    b = A * randn(n)
    q = randn(n)
    ws = PureDAQP.setup(P, q, A, b .- rand(m), b .+ rand(m), PureDAQP.ActiveSet())
    PureDAQP.solve!(ws)          # compile every specialization before analysing it

    red = ws.red
    lw = red.ws
    W = lw.W
    LW = typeof(lw)
    WT = typeof(W)
    T = Float64
    # The views the loop actually passes, so the analysed signature is the one that runs.
    PV = typeof(view(lw.p, 1:1))
    MV = typeof(PureDAQP.row(lw, 1))

    AT = typeof(ws.algorithm)

    checks = Any[
        (PureDAQP.activate!, (LW, Int, Int8), :hot, nothing),
        (PureDAQP.deactivate!, (LW, Int), :hot, nothing),
        (PureDAQP.blocking_step, (LW, PV, T), :hot, nothing),
        (PureDAQP.step_and_drop!, (LW, PV, T), :hot, nothing),
        (PureDAQP.add_row!, (WT, MV), :hot, nothing),
        (PureDAQP.remove_row!, (WT, Int), :hot, nothing),
        (PureDAQP.solve_gram!, (WT, PV), :hot, nothing),
        (PureDAQP.null_direction!, (PV, WT, Int), :hot, nothing),
        (PureDAQP.set_targets!, (typeof(red), Vector{T}), :hot, nothing),
        (PureDAQP.first_dependent, (WT, T), :hot, nothing),
        (PureDAQP.singular_step!, (LW, Int, T), :hot, nothing),
        (PureDAQP.full_set_step!, (LW, Int, Int8, T), :hot, nothing),
        (PureDAQP.working_set_multipliers!, (LW,), :hot, nothing),
        (PureDAQP.step_toward_multipliers!, (LW, T), :hot, nothing),
        (PureDAQP.primal_point!, (LW,), :hot, nothing),
        (PureDAQP.entering_row, (LW, T, Bool, UnitRange{Int}), :hot, nothing),
        (PureDAQP.solve_ldp!, (LW, AT, Int), :hot, nothing),
    ]

    println("PureDAQP (dual active set)")
    for (f, types, tier, measure) in checks
        label = "  " * string(nameof(f))
        try
            isnothing(types) ||
                test_signatures([(f, types)]; guarantees = GUARANTEES[tier])
            extra = ""
            if tier === :hot_measured
                measured_noalloc(measure)
                extra = ", noalloc (measured: 0 bytes)"
            end
            claims = isnothing(types) ? "measured only (keyword-only signature)" :
                join(GUARANTEES[tier], ", ")
            println("  ✓ ", label, "  ", claims, extra)
        catch e
            push!(failures, label)
            println("  ✗ ", label)
            println(sprint(showerror, e))
        end
    end
end

if isempty(failures)
    println(
        "\nStrictMode: all guarantees hold (checks_enabled=", StrictMode.checks_enabled(), ")."
    )
else
    println("\nStrictMode: ", length(failures), " failing guarantee(s):")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
