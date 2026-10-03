@testitem "the QPWorkspace contract is enforced, not decorative" begin
    using PureQPBase, PureIPM, TypeContracts
    W = PureOSQP.QPWorkspace
    spec = TypeContracts.list_contract(W)
    required = [s.description for s in spec if !s.optional]
    @test required == [
        "solve!(::Self) :: Solution",
        "warm_start!(::Self) :: Self",
        "cold_start!(::Self) :: Self",
        "update!(::Self) :: Self",
        "update_settings!(::Self) :: Self",
        "update_settings!(::Self, ::QPAlgorithm) :: Self",
        "dimensions(::Self) :: Tuple{Int, Int}",
        "derivative_ready(::Self) :: Nothing",
    ]
    @test sort([nameof(s.f) for s in spec if s.optional]) ==
        [:constraint_violation, :constraint_violation!, :update_rho!]

    # A mutating verb named in a slot's description is itself a slot. `constraint_violation`'s
    # description once ended "`constraint_violation!` writes it in place", and that verb was
    # declared nowhere, defined only by PureOSQP as a generic of its own, and absent from the
    # other two workspaces — a promise the contract made and no checker could see, because a
    # description is prose. Checking the `!` names keeps the prose honest; the rest of a
    # description names arguments and fields, which are not slots and are not checked here.
    slot_names = Set(nameof(s.f) for s in spec)
    for s in spec, m in eachmatch(r"`(\w+!)`", s.description)
        @test Symbol(m.captures[1]) in slot_names
    end

    # `using PureOSQP` alone reaches the whole API, because the module re-exports the base's
    # names rather than listing them again. A list would drift from what it copied, and did:
    # four packages each held their own copy of this surface.
    @test issubset(names(PureQPBase), names(PureOSQP))
    @test setdiff(names(PureOSQP), names(PureQPBase)) ==
        [:OperatorSplitting, :OperatorSplittingWorkspace, :Optimizer, :PureOSQP]
    @test issubset(names(PureQPBase), names(PureIPM))
    @test setdiff(names(PureIPM), names(PureQPBase)) ==
        [:InteriorPoint, :InteriorPointWorkspace, :PureIPM]

    for T in (OperatorSplittingWorkspace, InteriorPointWorkspace)
        @test TypeContracts.satisfies(T, W).satisfied
    end
    @test isempty(TypeContracts.satisfies(OperatorSplittingWorkspace, W).missing_optional)
    # The interior-point workspace implements none of the optional slots, which is what optional
    # means. Recorded here so that implementing one is a deliberate change to this list rather
    # than a silent divergence between the two solvers.
    @test sort(TypeContracts.satisfies(InteriorPointWorkspace, W).missing_optional) == [
        "constraint_violation!(::AbstractVector, ::Self) :: AbstractVector",
        "constraint_violation(::Self) :: AbstractVector",
        "update_rho!(::Self, ::Real) :: Self",
    ]

    # A workspace that declares the supertype and implements nothing inherits only the
    # methods written for every `QPWorkspace`; the rest are reported by name.
    @eval struct IncompleteWorkspace <: PureOSQP.QPWorkspace{Float64} end
    r = TypeContracts.satisfies(IncompleteWorkspace, W)
    @test !r.satisfied
    @test r.missing_methods == [
        "solve!(::Self) :: Solution",
        "warm_start!(::Self) :: Self",
        "cold_start!(::Self) :: Self",
        "update!(::Self) :: Self",
    ]
    @test_throws "solve!(::Self) :: Solution" TypeContracts.check_contract(IncompleteWorkspace, W)
end

@testitem "the QPAlgorithm contract is enforced, not decorative" begin
    using PureIPM, TypeContracts
    Alg = PureOSQP.QPAlgorithm
    spec = TypeContracts.list_contract(Alg)
    @test [nameof(s.f) for s in spec if !s.optional] ==
        [:setup_backend, :algorithm_defaults, :default_options, :element_typed]
    @test [nameof(s.f) for s in spec if s.optional] == [:adopt_settings!]

    for T in (OperatorSplitting, InteriorPoint)
        @test TypeContracts.satisfies(T, Alg).satisfied
    end

    @eval struct IncompleteAlgorithm <: PureOSQP.QPAlgorithm end
    r = TypeContracts.satisfies(IncompleteAlgorithm, Alg)
    @test !r.satisfied
    @test [first(split(m, '(')) for m in r.missing_methods] ==
        ["setup_backend", "algorithm_defaults", "element_typed"]
    @test_throws "setup_backend(::Self" TypeContracts.check_contract(IncompleteAlgorithm, Alg)
end

@testitem "every selection point serves a new SelectionFor or names what to implement" begin
    using PureQPBase
    using LinearAlgebra, SparseArrays, Krylov, LDLFactorizations, GPUArraysCore, JLArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))

    # A third algorithm's selection tag. Nothing is defined for it, which is the point: the
    # ladder must answer every point either with a decline or with the name of the method the
    # algorithm still owes, and never with a `MethodError` from somewhere inside selection.
    @eval struct ThirdSelection <: PureOSQP.SelectionFor end
    sel = ThirdSelection()

    n, m = 12, 15
    P, q, A, l, u = random_qp(n, m; seed = 11)
    prob = PureOSQP.Problem(Float64, P, q, A, l, u; scaling = 0)
    wt = PureOSQP.SystemWeights(ones(m), ones(m), 1.0e-6)
    Ps, As = sparse(P), sparse(A)

    # A rung whose default is to decline needs no method: it already takes any `SelectionFor`.
    @test isnothing(PureOSQP.kkt_rung(P, A, prob, wt, sel))
    @test isnothing(PureOSQP.reduced_rung(P, A, prob, wt, sel))
    @test isnothing(PureOSQP.kronecker_rung(P, A, prob, wt, sel))
    @test isnothing(PureOSQP.block_rung(P, A, prob, wt, sel))
    @test isnothing(PureOSQP.lowrank_rung(P, A, prob, wt, sel))
    @test isnothing(PureOSQP.formed_rung(P, A, prob, sel))

    # So does a `choose_backend` method for a pair whose backend is the same whatever solves it.
    D = Diagonal(fill(2.0, n))
    ls, factored = PureOSQP.choose_backend(D, Diagonal(ones(n)), prob, wt, sel)
    @test ls isa PureOSQP.DiagonalReduced
    @test !factored

    # The four points with no algorithm-independent answer each name themselves.
    @test_throws "PureQPBase.select_backend" PureOSQP.select_backend(P, A, prob, wt, sel)
    @test_throws "PureQPBase.dense_rung" PureOSQP.dense_rung(P, A, prob, sel)
    @test_throws "PureQPBase.indirect_rung" PureOSQP.indirect_rung(P, A, prob, sel)
    # `choose_backend`'s fallback is the ladder, so a pair with no method of its own reports
    # the ladder as what is missing.
    @test_throws "PureQPBase.select_backend" PureOSQP.choose_backend(P, A, prob, wt, sel)

    Ext = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
    @test_throws "sparse_form" Ext.sparse_form(Ps, As, n, m, sel)
    # The two sparse rungs consult that rule, so a gated call reports it rather than the rung.
    @test_throws "sparse_form" PureOSQP.kkt_rung(Ps, As, prob, wt, sel)
    @test_throws "sparse_form" PureOSQP.reduced_rung(Ps, As, prob, wt, sel)
    # Ungated, the same rungs serve any algorithm: the representation is all they need.
    @test first(PureOSQP.kkt_rung(Ps, As, prob, wt, sel; gated = false)) isa PureOSQP.LinearSystem

    # A GPU array has no direct backend under any algorithm, so the refusal is generic.
    @test_throws "only the matrix-free backend has a GPU counterpart" PureOSQP.choose_backend(
        JLArray(P), JLArray(A), prob, wt, sel
    )

    # The matrix-free backend reads settings an algorithm has to hand it. Without a method of
    # its own it would keep `cg_max_iter = 0` and every solve would return its input.
    raw = PureOSQP.indirect_backend(zeros(n), n, m, nothing)
    @test_throws "adopt_settings!" PureOSQP.factorize!(raw, prob, wt)
    PureOSQP.adopt_settings!(raw, OperatorSplitting(), PureOSQP.default_options(OperatorSplitting(), Float64))
    @test PureOSQP.factorize!(raw, prob, wt)
end

@testitem "the Preconditioner contract covers built-in and caller preconditioners" begin
    using PureIPM
    using LinearAlgebra, SparseArrays, Random, Krylov, TypeContracts
    include(joinpath(@__DIR__, "helpers.jl"))
    Pre = PureOSQP.Preconditioner
    @test [s.description for s in TypeContracts.list_contract(Pre)] == [
        "update_preconditioner!(::Self, ::Problem, ::SystemWeights, ::Int) :: Self",
        "LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)",
    ]
    for T in (IdentityPreconditioner, JacobiPreconditioner)
        @test T <: Pre
        @test TypeContracts.satisfies(T, Pre).satisfied
    end

    # Caller preconditioners are not subtypes and satisfy the same contract: a factorization
    # object as it is, and the reference preconditioners of the benchmarks. The benchmark file
    # is loaded without its `using` line, which names a package the tests do not depend on;
    # the preconditioner bodies that need it are not run here.
    P, q, A, l, u = random_qp(10, 15; seed = 5)
    chol = cholesky(Symmetric(P + I))
    @test TypeContracts.check_contract(typeof(chol), Pre).passed
    bench = Module(:BenchPreconditioners)
    Core.eval(bench, :(using PureOSQP, LinearAlgebra, SparseArrays))
    Base.include(
        ex -> Meta.isexpr(ex, :using) ? nothing : ex, bench,
        joinpath(@__DIR__, "..", "..", "PureIPM", "bench", "ipm_preconditioners.jl"),
    )
    @test TypeContracts.check_contract(bench.LaggedCholesky, Pre).passed
    @test TypeContracts.check_contract(bench.IncompleteLDL, Pre).passed

    @eval struct NoApply end
    r = TypeContracts.satisfies(NoApply, Pre)
    @test r.missing_methods == ["LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)"]
    @test_throws "LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)" TypeContracts.check_contract(NoApply, Pre)

    # `setup` refuses it by name under both algorithms, before building anything.
    for alg in (OperatorSplitting(), InteriorPoint())
        @test_throws "has no method LinearAlgebra.ldiv!" setup(
            P, q, A, l, u, alg; linsys = :indirect, scaling = 0, preconditioner = NoApply()
        )
        ws = setup(P, q, A, l, u, alg; linsys = :indirect, scaling = 0, preconditioner = chol)
        @test PureOSQP.backend_name(ws.linsys) === :indirect
    end
end
