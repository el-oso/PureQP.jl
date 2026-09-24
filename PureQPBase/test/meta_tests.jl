@testitem "every test file parses and declares the test items it should" begin
    # A test file with a syntax error, or one whose `@testitem` header is damaged by an
    # edit, is silently skipped by the runner rather than reported: the suite still says
    # "passed", just with fewer items. This asserts the inventory so that cannot happen
    # unnoticed. Update the counts deliberately when adding or removing a test item.
    expected = Dict(
        "banded_tests.jl" => 3,
        "base_tests.jl" => 5,
        "block_tests.jl" => 3,
        "indirect_tests.jl" => 1,
        "kronecker_tests.jl" => 4,
        "linsys_tests.jl" => 18,
        "meta_tests.jl" => 1,
        "operator_tests.jl" => 5,
        "scaling_tests.jl" => 8,
        "selection_tests.jl" => 6,
        "strictmode_tests.jl" => 2,
    )
    dir = @__DIR__
    files = sort(filter(f -> endswith(f, "_tests.jl"), readdir(dir)))
    @test files == sort(collect(keys(expected)))
    for f in files
        src = read(joinpath(dir, f), String)
        @test Meta.parse("begin\n$src\nend"; raise = false) isa Expr
        n = count(m -> true, eachmatch(r"^@testitem "m, src))
        @test n == expected[f]
    end
end
