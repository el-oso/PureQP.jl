# What this package is on its own: the problem and the contracts, with no algorithm. Solving
# is tested where the algorithms are, against a referee this package has no reason to depend on.

@testitem "no algorithm is defined here" begin
    using PureQPBase
    using InteractiveUtils: subtypes
    # Both are declared, so an algorithm package has a supertype to extend and `solve` has a
    # function to add a method to; neither has an instance until one is loaded.
    @test isabstracttype(PureQPBase.QPAlgorithm)
    @test isabstracttype(PureQPBase.QPWorkspace)
    @test isempty(subtypes(PureQPBase.QPAlgorithm))
    @test isempty(subtypes(PureQPBase.QPWorkspace))

    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    # The five-argument forms belong to the package that defines the default algorithm.
    @test_throws MethodError solve(P, q, A, l, u)
    @test_throws MethodError setup(P, q, A, l, u)
end

@testitem "the problem is built and validated" begin
    using PureQPBase
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    prob = PureQPBase.validated_problem(Float64, 2, 3, P, q, A, l, u, 10)
    @test prob.n == 2
    @test prob.m == 3
    @test length(prob.D) == 2
    @test length(prob.E) == 3

    @test_throws "l must be elementwise" PureQPBase.validate(P, q, A, [1.0, 0.0, 1.0], u)
    @test_throws "q" PureQPBase.validate(P, [1.0, NaN], A, l, u)
end

@testitem "a backend factors and solves its system" begin
    using PureQPBase, LinearAlgebra
    include(joinpath(@__DIR__, "helpers.jl"))
    # `ReducedCholesky` is this package's own, and needs no algorithm to drive it: the weights
    # stand in for what one would set.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    prob, wt, ls = backend_for(P, q, A, l, u; scaling = 0)
    @test ls isa PureQPBase.LinearSystem
    @test PureQPBase.backend_name(ls) isa Symbol

    bx, bz = [1.0, 2.0], [0.5, 0.5, 0.5]
    x, z = zeros(2), zeros(3)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, A, wt) \ [bx; bz]
    @test x ≈ ref[1:2] rtol = 1.0e-9
    @test z ≈ A * x rtol = 1.0e-9
end

@testitem "the option names are checked against a backend name" begin
    using PureQPBase
    @test :auto in PureQPBase.LINSYS_OPTIONS
    @test :kkt in PureQPBase.LINSYS_OPTIONS
    @test !(:nonsense in PureQPBase.LINSYS_OPTIONS)
    @test_throws "linsys must be one of" PureQPBase.check_linsys(:nonsense)
    @test PureQPBase.check_linsys(:kkt) === Val(:kkt)
end

@testitem "an operator extension does not claim a pair of plain matrices" begin
    using PureQPBase, LinearMaps, SciMLOperators
    # Each operator extension adds a `setup`/`solve` that wraps its own type and passes a
    # matrix through. A method accepting two plain matrices would wrap neither and forward
    # the same arguments to itself, which recurses until the stack ends. That pair belongs to
    # whichever package defines the default algorithm, and to no extension.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    @test_throws MethodError solve(P, q, A, l, u)
    @test_throws MethodError setup(P, q, A, l, u)
end

@testitem "copyto! between solutions follows a certificate without touching shared vectors" begin
    using PureQPBase: empty_solution, no_certificate, DUAL_INFEASIBLE, SOLVED

    # A workspace's solution points both certificates at one empty vector. A certificate that
    # arrives must not be written into it, or every later result would carry one.
    none = no_certificate(Float64)
    dest = empty_solution(zeros(2), zeros(1), none, none)
    src_cert = empty_solution(fill(NaN, 2), fill(NaN, 1), none, [0.0, 1.0])
    src_cert.status = DUAL_INFEASIBLE
    src_plain = empty_solution([1.0, 2.0], [3.0], none, none)
    src_plain.status = SOLVED

    @test copyto!(dest, src_cert) === dest
    @test dest.status === DUAL_INFEASIBLE
    @test dest.dual_inf_cert == [0.0, 1.0]
    @test isempty(dest.prim_inf_cert)
    @test isempty(none)
    @test dest.dual_inf_cert !== src_cert.dual_inf_cert

    # And back to a plain result: the certificate is empty again, and no vector was resized.
    @test copyto!(dest, src_plain) === dest
    @test dest.status === SOLVED
    @test isempty(dest.dual_inf_cert) && isempty(dest.prim_inf_cert)
    @test dest.x == [1.0, 2.0] && dest.y == [3.0]
    @test isempty(none) && isempty(src_cert.prim_inf_cert)

    # A destination that holds one vector for both certificates keeps the other one intact.
    shared = [0.0, 0.0]
    both = empty_solution(zeros(2), zeros(1), shared, shared)
    both_src = empty_solution(zeros(2), zeros(1), [1.0, 1.0], [2.0, 2.0])
    copyto!(both, both_src)
    @test both.prim_inf_cert == [1.0, 1.0]
    @test both.dual_inf_cert == [2.0, 2.0]

    # A solve that stays at one status allocates nothing.
    copyto!(dest, src_plain)
    @test (@allocated copyto!(dest, src_plain)) == 0

    # The vectors the destination holds for x and y still have to match.
    other = empty_solution(zeros(3), zeros(1), none, none)
    @test_throws "must match the source" copyto!(other, src_plain)
end
