# The six unmaterialized problems behind the `Unmaterialized operators` benchmark table and the
# examples of the same name in `docs/src/examples.md`. This file is the only definition of them:
# `bench/unmaterialized_paths.jl` times them, and one test item per algorithm package pins the
# path each one reaches, against `case_fingerprint` so a stale table is a failing test.
#
# Needs `PureOSQP`, `PureIPM`, `PureDAQP`, `PureQPBase`, `LinearMaps`, `LinearAlgebra`, `Krylov`
# and `Random` loaded.

"An `k×k` positive definite matrix, which a Kronecker `P` needs of both its factors."
pd_factor(k) = (S = randn(k, k); Matrix(Symmetric(S'S ./ k + 2I)))

"A `k×k` smoothing stencil: the one-dimensional blur a separable transform applies per axis."
blur_factor(k) = diagm(0 => fill(0.6, k), 1 => fill(0.2, k - 1), -1 => fill(0.2, k - 1))

"""
    unmaterialized_cases() -> Vector{NamedTuple}

One entry per path through the solvers, each with an `A` whose entries are never formed.

`algorithm` names the package and `path` how the linear system is solved. `make_alg()` builds the
algorithm instance, since a working-set representation is a parameter of `ActiveSet` rather than an
option; it is a function so that reading the cases needs none of the three algorithm packages
loaded, and a consumer interested in one of them loads only that one. `options(P, A)` gives the
`setup` keywords, likewise a function because the interior-point method's conjugate-gradient path
takes a preconditioner built from the same factors. `pin` is what the solve must reach: a backend
name, or the algorithm's own parameter where the choice is not a backend.

Each case carries its own seed, so adding or reordering one never changes another's numbers.
"""
function unmaterialized_cases()
    return [
        (
            algorithm = "PureOSQP", path = "CG", seed = 7201,
            name = "separable blur, conjugate gradients",
            pin = :indirect,
            # A separable blur over a 24×24 grid: `A₁ ⊗ A₂` smooths along each axis in turn. The
            # 576×576 operator it stands for is never formed; two 24×24 factors are all that is held.
            build = () -> begin
                A = kron(LinearMap(blur_factor(24)), LinearMap(blur_factor(24)))
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), randn(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureOSQP.OperatorSplitting(),
            options = (P, A) -> (; linsys = :indirect, scaling = 0),
        ),
        (
            algorithm = "PureOSQP", path = "direct", seed = 7202,
            name = "separable transform, Kronecker backend",
            pin = :kronecker,
            # The Kronecker backend diagonalizes the reduced matrix by the factors' own
            # eigenvectors, so it factors two 24×24 problems instead of one 576×576 one.
            build = () -> begin
                A = kron(LinearMap(randn(24, 24) ./ 5), LinearMap(randn(24, 24) ./ 5))
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), randn(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureOSQP.OperatorSplitting(),
            options = (P, A) -> (; linsys = :kronecker, scaling = 0),
        ),
        (
            algorithm = "PureIPM", path = "CG", seed = 7203,
            name = "separable transform, preconditioned conjugate gradients",
            pin = :indirect,
            # Both `P` and `A` are Kronecker products, which is what lets one preconditioner
            # diagonalize the reduced matrix: `U₁ ⊗ U₂` makes `P` the identity and `ÃᵀWÃ`
            # diagonal at once. The interior-point method refuses its CG path without a
            # preconditioner, having been measured not to reach tolerance on the Jacobi diagonal.
            build = () -> begin
                P = PureQPBase.KroneckerOperator(pd_factor(12), pd_factor(12))
                A = PureQPBase.KroneckerOperator(randn(12, 12) ./ 4, randn(12, 12) ./ 4)
                n, m = size(A, 2), size(A, 1)
                (P, randn(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureIPM.InteriorPoint(),
            options = (P, A) -> (;
                linsys = :indirect, scaling = 0,
                preconditioner = PureQPBase.KroneckerPreconditioner(P, A),
            ),
        ),
        (
            algorithm = "PureIPM", path = "direct", seed = 7204,
            name = "separable transform, reduced matrix from products",
            pin = :product_reduced,
            # The interior-point method needs the reduced matrix itself, which an operator with no
            # entries supplies through products: a Kronecker `A` contracts its factors into it.
            build = () -> begin
                A = kron(LinearMap(randn(24, 24) ./ 5), LinearMap(randn(24, 24) ./ 5))
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), randn(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureIPM.InteriorPoint(),
            options = (P, A) -> (; scaling = 0),
        ),
        (
            algorithm = "PureDAQP", path = "direct, QR", seed = 7205,
            name = "separable transform, working set by rows",
            pin = :rows,
            # The dual active-set method reads one row of `A` each time its working set changes,
            # which a Kronecker operator answers from its factors. The bounds are tight enough
            # that rows enter the set, so the representation is doing work.
            build = () -> begin
                A = PureQPBase.KroneckerOperator(randn(16, 16) ./ 3, randn(16, 16) ./ 3)
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), randn(n), A, fill(-0.05, m), fill(0.05, m))
            end,
            make_alg = () -> PureDAQP.ActiveSet(working_set = :rows),
            options = (P, A) -> (; scaling = 0),
        ),
        (
            # The same seed as the case above, so the two representations are two ways through one
            # problem: their iteration counts are comparable and their answers must agree.
            algorithm = "PureDAQP", path = "direct, LDLᵀ", seed = 7205,
            name = "separable transform, working set by Gram matrix",
            pin = :gram,
            build = () -> begin
                A = PureQPBase.KroneckerOperator(randn(16, 16) ./ 3, randn(16, 16) ./ 3)
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), randn(n), A, fill(-0.05, m), fill(0.05, m))
            end,
            make_alg = () -> PureDAQP.ActiveSet(working_set = :gram),
            options = (P, A) -> (; scaling = 0),
        ),
    ]
end

"""
    case_problem(case) -> (P, q, A, l, u)

The problem `case` defines, under its own seed so one case's data never depends on another's.
"""
function case_problem(case)
    Random.seed!(case.seed)
    return case.build()
end

"""
    case_workspace(case) -> QPWorkspace

`case` set up and ready to solve, with the algorithm and options it names.
"""
function case_workspace(case)
    P, q, A, l, u = case_problem(case)
    return setup(P, q, A, l, u, case.make_alg(); case.options(P, A)..., verbose = false)
end

"""
    case_fingerprint(case) -> UInt64

A hash of the problem `case` defines, which changes if any of its data changes.

`P` and `A` are identified by what they do rather than by what they hold: their sizes and their
action on two fixed vectors. Reading their entries to hash them is the one thing these problems
exist to avoid, and an operator holding only its factors has no entries to read without forming
them. The products are rounded, so a fingerprint does not turn over on the last bit of a sum whose
order a refactoring may change.
"""
function case_fingerprint(case)
    P, q, A, l, u = case_problem(case)
    m, n = size(A)
    v = [sinpi(2 * j / n) for j in 1:n]
    w = [cospi(2 * i / m) for i in 1:m]
    # `adjoint`, never `transpose`: these operators define their product for `Adjoint` only, and
    # `transpose` falls back to reading entries one at a time, forming what is held unformed.
    acts = (A * v, adjoint(A) * w, P * v)
    return hash((case.name, m, n, q, l, u, map(a -> round.(a; digits = 9), acts)))
end

"""
    recorded_fingerprints(path) -> Dict{String, String}

The fingerprint each case had when `bench/unmaterialized_paths.jl` last wrote `path`, by case name.

Read with a pattern rather than a JSON parser so that a test asserting the table is current needs
no dependency beyond what it already has.
"""
function recorded_fingerprints(path)
    text = read(path, String)
    pairs = Dict{String, String}()
    for m in eachmatch(r"\"name\": \"([^\"]*)\".*?\"fingerprint\": \"([0-9a-f]+)\"", text)
        pairs[m.captures[1]] = m.captures[2]
    end
    return pairs
end

"The results file `bench/unmaterialized_paths.jl` writes, relative to this file."
results_path() = joinpath(@__DIR__, "results", "unmaterialized_paths.json")
