"""
    QPData{T,MP,MA,V}

The problem as the caller gave it: `P` and `A` by reference, the dimensions, and the linear
term and bounds clamped to `±INFTY`. Nothing here is scaled and nothing here is scratch.

An algorithm that works on these directly holds one of these and nothing more — a
[`Problem`](@ref) is what an algorithm asks for when its method works on the
equilibrated problem instead.

Immutable: [`update!`](@ref) writes `q0`, `l0` and `u0` through, and replaces `P` or `A` by
building another `QPData` around the same vectors for the workspace to hold.
"""
struct QPData{T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T}}
    P::MP
    A::MA
    n::Int
    m::Int
    q0::V   # caller's data, clamped to ±INFTY
    l0::V
    u0::V
end

"""
    Problem{T,MP,MA,V}

The equilibrated problem derived from a [`QPData`](@ref), together with the scratch the
scaled products and the [`LinearSystem`](@ref) backends need. Shared by every backend.

The caller's own fields reach through: `prob.P`, `prob.n` and `prob.q0` read the `QPData`
this was built around, so code that needs both sides takes one argument.

An algorithm that never forms the scaled products and has no backend — a dual active-set
method, say — holds the [`QPData`](@ref) alone and never allocates any of this.
"""
struct Problem{T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T}}
    data::QPData{T, MP, MA, V}
    q::V    # equilibrated
    l::V
    u::V
    D::V    # Ruiz factors
    E::V
    c::T
    scaling::Int   # sweeps requested at setup; 0 means D, E and c are identity
    tmp_n::V       # scratch of mul_A!/mul_At!/mul_P!
    tmp_m::V
    work_n::V      # scratch of reduced_rhs! and the backends
    work_m::V
end

# The caller's fields are reached through the scaled problem, so a backend that needs `P` and
# `D` still takes one argument and names both the way it always has. Every call site passes a
# literal, so this folds to the `getfield` it stands for.
@inline function Base.getproperty(prob::Problem, s::Symbol)
    if s === :P || s === :A || s === :n || s === :m ||
            s === :q0 || s === :l0 || s === :u0
        return getproperty(getfield(prob, :data), s)
    end
    return getfield(prob, s)
end

Base.propertynames(::Problem) = (
    :data, :P, :A, :n, :m, :q0, :l0, :u0, :q, :l, :u, :D, :E, :c, :scaling,
    :tmp_n, :tmp_m, :work_n, :work_m,
)

"""
    AnyProblem{T,MP,MA,V}

Either problem type, for the checks that read only what the caller gave and so hold whether
or not the algorithm asking scaled it.
"""
const AnyProblem{T, MP, MA, V} = Union{QPData{T, MP, MA, V}, Problem{T, MP, MA, V}}

"""
    unscaled_iterates(prob, x, y, z) -> (x, y, z)

The iterates an algorithm holds, in the caller's units.

Fresh vectors either way: the caller is handed them to work with, and the workspace's own
must not be what it gets. A [`QPData`](@ref) was never equilibrated, so returning them is all
there is to undo.
"""
function unscaled_iterates(
        prob::Problem{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}
    ) where {T}
    return (prob.D .* x, (prob.E .* y) ./ prob.c, z ./ prob.E)
end

function unscaled_iterates(
        ::QPData{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}
    ) where {T}
    return (copy(x), copy(y), copy(z))
end

"""
    Problem(T, P, q, A, l, u; scaling) -> Problem

Validate `P`, `q`, `A`, `l`, `u`, allocate the buffers `similar` to `q` follows, and run
`scaling` sweeps of Ruiz equilibration into them.

Does not check convexity: whether `P + shift*I` is positive definite is a question for the
algorithm that uses this problem — `σ` for ADMM — not for the data on its own, so the
caller runs [`is_convex`](@ref) itself, with whichever shift it needs.
"""
function Problem(
        ::Type{T}, P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector; scaling::Integer
    ) where {T <: Real}
    n, m = validate(P, q, A, l, u)
    return validated_problem(T, n, m, P, q, A, l, u, scaling)
end

"""
    validated_data(T, n, m, P, q, A, l, u) -> QPData

The caller's data for input `validate` has already accepted, clamped to `±INFTY` and in the
element type the solve runs in. Nothing is scaled and no scratch is allocated, so this is
what an algorithm that works on the caller's problem directly asks for.
"""
function validated_data(
        ::Type{T}, n::Integer, m::Integer, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector
    ) where {T <: Real}
    inf = INFTY(T)
    q0 = copyto!(similar(q, T, n), q)
    l0 = max.(copyto!(similar(l, T, m), l), -inf)
    u0 = min.(copyto!(similar(u, T, m), u), inf)
    return QPData{T, typeof(P), typeof(A), typeof(q0)}(P, A, Int(n), Int(m), q0, l0, u0)
end

"""
    validated_problem(T, n, m, P, q, A, l, u, scaling) -> Problem

`Problem` for data `validate` has already accepted, for a caller that has to
run other checks between validation and equilibration and should not pay for validation twice.
"""
function validated_problem(
        ::Type{T}, n::Integer, m::Integer, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, scaling::Integer
    ) where {T <: Real}
    data = validated_data(T, n, m, P, q, A, l, u)
    return scaled_problem(data, scaling)
end

"""
    scaled_problem(data, scaling) -> Problem

Build the equilibrated problem and the scratch its products and backends need around `data`,
running `scaling` sweeps of Ruiz equilibration.
"""
function scaled_problem(data::QPData{T, MP, MA, V}, scaling::Integer) where {T, MP, MA, V}
    q0, l0, u0 = data.q0, data.l0, data.u0
    n, m = data.n, data.m
    # A single definition, and no default argument: a local function assigned more than once
    # is boxed, which turns every call through it into a dynamic dispatch.
    buf(k, v) = fill!(similar(q0, T, k), v)
    z = zero(T)
    o = one(T)
    q, l, u = copy(q0), copy(l0), copy(u0)
    D, E = buf(n, o), buf(m, o)
    tmp_n, tmp_m, work_n, work_m = buf(n, z), buf(m, z), buf(n, z), buf(m, z)
    c = equilibrate!(
        T, data.P, data.A, q0, l0, u0, q, l, u, D, E, tmp_n, tmp_m, work_n, n, scaling
    )
    return Problem{T, MP, MA, V}(
        data, q, l, u, D, E, c, Int(scaling), tmp_n, tmp_m, work_n, work_m,
    )
end

"""
    validate_update!(prob, ls; P, A, q, l, u) -> Nothing

Throw unless the proposed data can replace what `prob` holds: dimensions, representation,
symmetry, finiteness and bound ordering, plus whatever the backend `ls` needs through
[`check_update`](@ref). Reads the arguments and `prob`, and writes nothing.

Convexity is not checked here, because the shift it is checked at belongs to the algorithm.
"""
function validate_update!(
        prob::AnyProblem{T, MP, MA}, ls; P = nothing, A = nothing, q = nothing, l = nothing,
        u = nothing
    ) where {T, MP, MA}
    n, m = prob.n, prob.m
    if !isnothing(P)
        size(P) == (n, n) || throw(ArgumentError(lazy"P must stay $(n)×$(n), got $(size(P))"))
        P isa MP || throw(
            ArgumentError(
                "P must keep the representation the workspace was built with: its linear-" *
                    "system backend is built for that representation and would read only the " *
                    "structure it implies. Rebuild the workspace with setup to change it."
            )
        )
        is_symmetric(P) || throw(ArgumentError("P must be symmetric"))
        is_materializable(P) || check_symmetric_products(P, prob.q0)
        is_materializable(P) && check_finite(P, n, n, "P")
        check_storage(P, n, n)
    end
    if !isnothing(A)
        size(A) == (m, n) || throw(ArgumentError(lazy"A must stay $(m)×$(n), got $(size(A))"))
        A isa MA || throw(
            ArgumentError(
                "A must keep the representation the workspace was built with: its linear-" *
                    "system backend is built for that representation and would read only the " *
                    "structure it implies. Rebuild the workspace with setup to change it."
            )
        )
        is_materializable(A) && check_finite(A, m, n, "A")
        check_storage(A, m, n)
    end
    if !isnothing(P) || !isnothing(A)
        check_update(ls, isnothing(P) ? prob.P : P, isnothing(A) ? prob.A : A)
    end
    if !isnothing(q)
        length(q) == n || throw(ArgumentError(lazy"length(q) must be $n, got $(length(q))"))
        all(isfinite, q) || throw(ArgumentError("q must be finite, found NaN or Inf"))
    end
    if !isnothing(l) || !isnothing(u)
        # Lengths first: the walks below index every row of both proposals, and a short one
        # would reach the end of a vector rather than this message.
        isnothing(l) || length(l) == m ||
            throw(ArgumentError(lazy"length(l) must be $m, got $(length(l))"))
        isnothing(u) || length(u) == m ||
            throw(ArgumentError(lazy"length(u) must be $m, got $(length(u))"))
        inf = INFTY(T)
        if !isnothing(l)
            any(isnan, l) && throw(ArgumentError("l contains NaN"))
            any(li -> li == Inf, l) && throw(ArgumentError("l may not be +Inf"))
        end
        if !isnothing(u)
            any(isnan, u) && throw(ArgumentError("u contains NaN"))
            any(ui -> ui == -Inf, u) && throw(ArgumentError("u may not be -Inf"))
        end
        # The ordering test runs on the clamped proposals, not on what the workspace holds,
        # so a pair that fails it leaves the old bounds in place.
        for i in 1:m
            li = isnothing(l) ? prob.l0[i] : max(T(l[i]), -inf)
            ui = isnothing(u) ? prob.u0[i] : min(T(u[i]), inf)
            li <= ui ||
                throw(ArgumentError(lazy"l must be elementwise ≤ u, violated at index $i: $li > $ui"))
        end
    end
    return nothing
end

"""
    adopt_update!(prob; P, A, q, l, u) -> prob

Replace what `prob` holds with the data given, and reapply the existing equilibration to it.
Runs only after [`validate_update!`](@ref) and every algorithm-specific check have passed.

`q0`, `l0`, `u0` and the equilibrated copy are written through, so the vectors a workspace
holds stay the vectors it holds. A new `P` or `A` cannot be: the problem is immutable, so
one is returned carrying them, and the caller keeps what it gets —
`ws.prob = adopt_update!(ws.prob; ...)`. It shares every vector with the problem it
replaces.
"""
function adopt_update!(
        data::QPData{T}; P = nothing, A = nothing, q = nothing, l = nothing, u = nothing
    ) where {T}
    isnothing(q) || (data.q0 .= q)
    isnothing(l) || (data.l0 .= max.(T.(l), -INFTY(T)))
    isnothing(u) || (data.u0 .= min.(T.(u), INFTY(T)))
    # Every check reads the arguments, never the problem's own fields, so the matrices are
    # adopted only once none of them can refuse. A refusal that had already replaced `P` or
    # `A` would leave the workspace holding a matrix its factorization and its buffers were
    # not built for, and the next solve reads out of range.
    (isnothing(P) && isnothing(A)) && return data
    newP = isnothing(P) ? data.P : P
    newA = isnothing(A) ? data.A : A
    return QPData{T, typeof(newP), typeof(newA), typeof(data.q0)}(
        newP, newA, data.n, data.m, data.q0, data.l0, data.u0,
    )
end

function adopt_update!(
        prob::Problem{T}; P = nothing, A = nothing, q = nothing, l = nothing,
        u = nothing
    ) where {T}
    data = adopt_update!(getfield(prob, :data); P, A, q, l, u)

    # Reapply the existing equilibration to whatever changed.
    isnothing(q) || (prob.q .= prob.c .* prob.D .* prob.q0)
    if !isnothing(l) || !isnothing(u)
        prob.l .= prob.E .* prob.l0
        prob.u .= prob.E .* prob.u0
    end
    data === getfield(prob, :data) && return prob
    return Problem{T, typeof(data.P), typeof(data.A), typeof(data.q0)}(
        data, prob.q, prob.l, prob.u, prob.D, prob.E, prob.c, prob.scaling,
        prob.tmp_n, prob.tmp_m, prob.work_n, prob.work_m,
    )
end
