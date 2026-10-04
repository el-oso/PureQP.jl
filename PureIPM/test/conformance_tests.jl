@testitem "InteriorPoint honours the Solution and Status contract" begin
    using PureQPBase, PureIPM
    # The assertions live in PureQPBase, which owns `Solution` and `Status`, so both
    # algorithms are held to one statement of what their values mean rather than to
    # whatever each package's own suite happens to check.
    PureQPBase.conforms(InteriorPoint(); eps = 1.0e-8, slow_iters = 2)

    # `using PureIPM` alone reaches the whole API, because the module re-exports the base's
    # names rather than listing them again.
    @test issubset(names(PureQPBase), names(PureIPM))
    @test setdiff(names(PureIPM), names(PureQPBase)) ==
        [:InteriorPoint, :InteriorPointWorkspace, :PureIPM]

    # `Optimizer` is defined and not exported. A caller names it with its package, which is what
    # MathOptInterface expects, and two packages exporting one name would make the unqualified
    # one an `UndefVarError` to anyone who loaded both.
    @test isdefined(PureIPM, :Optimizer)
    @test !(:Optimizer in names(PureIPM))
end
