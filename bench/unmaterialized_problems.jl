# The six unmaterialized problems behind the `Unmaterialized operators` benchmark table and the
# examples of the same name in `docs/src/examples.md`. This file is the only definition of them:
# `bench/unmaterialized_paths.jl` times them, and one test item per algorithm package pins the
# path each one reaches, against `case_fingerprint` so a stale table is a failing test.
#
# Needs `PureOSQP`, `PureIPM`, `PureDAQP`, `PureQPBase`, `LinearMaps`, `LinearAlgebra`, `Krylov`
# and `Random` loaded.

"""
    fixed_stream(seed) -> () -> Float64

A generator of values spread over `[-1, 1)`, from `seed`, by a 64-bit linear congruential
recurrence written out here.

The recurrence is in this file rather than taken from `Random` because the problems below are
fingerprinted, and the fingerprints are compared on whichever Julia version runs the tests:
`randn`'s stream is not stable across versions, so a seeded fixture failed its own comparison on
1.12 after being recorded on 1.13, which says nothing about whether the table is current. The
multipliers are Knuth's MMIX constants.

Spread values rather than a smooth formula: a matrix of samples from `sinpi`/`cospi` is far better
conditioned than one of random entries, and the solvers then converge in a fraction of the
iterations, which measures the fixture instead of the solver. Measured on the dual active-set
method, 20 iterations against 262.
"""
function fixed_stream(seed::Integer)
    s = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x165667B19E3779F9
    return function ()
        s = s * 0x5851F42D4C957F2D + 0x14057B7EF767814F
        return 2.0 * ((s >> 11) / 2.0^53) - 1.0
    end
end

"""
    fixed_matrix(m, n; scale = 1.0, seed = 1) -> Matrix{Float64}

An `m×n` matrix of spread values, the same on every Julia version.

`seed` separates one matrix from another, so the two factors of one Kronecker product differ rather
than making it the square of a single matrix.
"""
function fixed_matrix(m, n; scale = 1.0, seed = 1)
    next = fixed_stream(seed)
    out = Matrix{Float64}(undef, m, n)
    for i in eachindex(out)
        out[i] = scale * next()
    end
    return out
end

"A length-`n` vector of the same values, for a linear term."
function fixed_vector(n; scale = 1.0, seed = 1)
    next = fixed_stream(seed)
    return [scale * next() for _ in 1:n]
end

"An `k×k` positive definite matrix, which a Kronecker `P` needs of both its factors."
pd_factor(k; seed = 1) = (S = fixed_matrix(k, k; seed); Matrix(Symmetric(S'S ./ k + 2I)))

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

Each case builds its own data from a formula, so adding or reordering one never changes another's
numbers, and a fingerprint recorded on one Julia version still matches on the next.
"""
function unmaterialized_cases()
    return [
        (
            algorithm = "PureOSQP", path = "CG",
            name = "separable blur, conjugate gradients",
            pin = :indirect,
            # A separable blur over a 24×24 grid: `A₁ ⊗ A₂` smooths along each axis in turn. The
            # 576×576 operator it stands for is never formed; two 24×24 factors are all that is held.
            build = () -> begin
                A = kron(LinearMap(blur_factor(24)), LinearMap(blur_factor(24)))
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), fixed_vector(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureOSQP.OperatorSplitting(),
            options = (P, A) -> (; linsys = :indirect, scaling = 0),
        ),
        (
            algorithm = "PureOSQP", path = "direct",
            name = "separable transform, Kronecker backend",
            pin = :kronecker,
            # The Kronecker backend diagonalizes the reduced matrix by the factors' own
            # eigenvectors, so it factors two 24×24 problems instead of one 576×576 one.
            build = () -> begin
                A = kron(
                    LinearMap(fixed_matrix(24, 24; scale = 0.2)),
                    LinearMap(fixed_matrix(24, 24; scale = 0.2, seed = 2)),
                )
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), fixed_vector(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureOSQP.OperatorSplitting(),
            options = (P, A) -> (; linsys = :kronecker, scaling = 0),
        ),
        (
            algorithm = "PureIPM", path = "CG",
            name = "separable transform, preconditioned conjugate gradients",
            pin = :indirect,
            # Both `P` and `A` are Kronecker products, which is what lets one preconditioner
            # diagonalize the reduced matrix: `U₁ ⊗ U₂` makes `P` the identity and `ÃᵀWÃ`
            # diagonal at once. The interior-point method refuses its CG path without a
            # preconditioner, having been measured not to reach tolerance on the Jacobi diagonal.
            build = () -> begin
                P = PureQPBase.KroneckerOperator(pd_factor(12), pd_factor(12))
                A = PureQPBase.KroneckerOperator(
                    fixed_matrix(12, 12; scale = 0.25),
                    fixed_matrix(12, 12; scale = 0.25, seed = 2),
                )
                n, m = size(A, 2), size(A, 1)
                (P, fixed_vector(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureIPM.InteriorPoint(),
            options = (P, A) -> (;
                linsys = :indirect, scaling = 0,
                preconditioner = PureQPBase.KroneckerPreconditioner(P, A),
            ),
        ),
        (
            algorithm = "PureIPM", path = "direct",
            name = "separable transform, reduced matrix from products",
            pin = :product_reduced,
            # The interior-point method needs the reduced matrix itself, which an operator with no
            # entries supplies through products: a Kronecker `A` contracts its factors into it.
            build = () -> begin
                A = kron(
                    LinearMap(fixed_matrix(24, 24; scale = 0.2)),
                    LinearMap(fixed_matrix(24, 24; scale = 0.2, seed = 3)),
                )
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), fixed_vector(n), A, fill(-1.0, m), fill(1.0, m))
            end,
            make_alg = () -> PureIPM.InteriorPoint(),
            options = (P, A) -> (; scaling = 0),
        ),
        (
            algorithm = "PureDAQP", path = "direct, QR",
            name = "separable transform, working set by rows",
            pin = :rows,
            # The dual active-set method reads one row of `A` each time its working set changes,
            # which a Kronecker operator answers from its factors. The bounds are tight enough
            # that rows enter the set, so the representation is doing work.
            build = () -> begin
                A = PureQPBase.KroneckerOperator(
                    fixed_matrix(16, 16; scale = 0.33),
                    fixed_matrix(16, 16; scale = 0.33, seed = 4),
                )
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), fixed_vector(n), A, fill(-0.05, m), fill(0.05, m))
            end,
            make_alg = () -> PureDAQP.ActiveSet(working_set = :rows),
            options = (P, A) -> (; scaling = 0),
        ),
        (
            # The same data as the case above, so the two representations are two ways through one
            # problem: their iteration counts are comparable and their answers must agree.
            algorithm = "PureDAQP", path = "direct, LDLᵀ",
            name = "separable transform, working set by Gram matrix",
            pin = :gram,
            build = () -> begin
                A = PureQPBase.KroneckerOperator(
                    fixed_matrix(16, 16; scale = 0.33),
                    fixed_matrix(16, 16; scale = 0.33, seed = 4),
                )
                n, m = size(A, 2), size(A, 1)
                (Diagonal(fill(2.0, n)), fixed_vector(n), A, fill(-0.05, m), fill(0.05, m))
            end,
            make_alg = () -> PureDAQP.ActiveSet(working_set = :gram),
            options = (P, A) -> (; scaling = 0),
        ),
    ]
end

"""
    case_problem(case) -> (P, q, A, l, u)

The problem `case` defines. Each one builds its own data from a formula, so adding or reordering a
case cannot change another's, and the same call gives the same problem on any Julia version.
"""
case_problem(case) = case.build()

"""
    case_workspace(case) -> QPWorkspace

`case` set up and ready to solve, with the algorithm and options it names.
"""
function case_workspace(case)
    P, q, A, l, u = case_problem(case)
    return setup(P, q, A, l, u, case.make_alg(); case.options(P, A)..., verbose = false)
end

"""
    stable_digest(parts...) -> UInt64

A digest of `parts`, the same on every Julia version and every machine.

`Base.hash` is not what this uses, because its result is not stable across Julia versions: a
fingerprint recorded beside the benchmark on one version is compared on whichever version runs the
tests, and a changed hash algorithm would read as a changed problem. This folds the bit patterns
with a fixed multiplier instead, so only the values decide it.
"""
function stable_digest(parts...)
    h = 0xcbf29ce484222325
    for p in parts
        for x in p
            h = (h ⊻ reinterpret(UInt64, Float64(x))) * 0x00000100000001b3
        end
        h = (h ⊻ 0x9E3779B97F4A7C15) * 0x00000100000001b3
    end
    return h
end

"""
    case_fingerprint(case) -> UInt64

A digest of the problem `case` defines, which changes if any of its data changes.

`P` and `A` are identified by what they do rather than by what they hold: their action on the first
unit vector, in both directions. Reading their entries to digest them is the one thing these
problems exist to avoid, and an operator holding only its factors has none to read without forming
them.

`e₁` rather than a spread vector: each entry of `A e₁` is one product of one entry of each factor,
so nothing is summed and no rounding depends on the order a BLAS chooses or on which BLAS it is.
A digest over a spread vector drifts between machines in its last bits, which a comparison for
equality then reports as a changed problem.
"""
function case_fingerprint(case)
    P, q, A, l, u = case_problem(case)
    m, n = size(A)
    e1n = [j == 1 ? 1.0 : 0.0 for j in 1:n]
    e1m = [i == 1 ? 1.0 : 0.0 for i in 1:m]
    # `adjoint`, never `transpose`: these operators define their product for `Adjoint` only, and
    # `transpose` falls back to reading entries one at a time, forming what is held unformed.
    return stable_digest((m, n), q, l, u, A * e1n, adjoint(A) * e1m, P * e1n)
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
