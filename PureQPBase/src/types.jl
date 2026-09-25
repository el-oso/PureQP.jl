"""
    Status

How a solve ended, reported as `Solution.status`. Twelve values, in three groups.

**Converged.** [`has_solution`](@ref) is true and `x`, `y` are the answer.

| status | meaning |
|---|---|
| `SOLVED` | primal and dual residuals are under the requested tolerances, and the duality gap too when `check_dualgap` is set |
| `SOLVED_INACCURATE` | the run stopped without meeting those, but the residuals clear ten times them |

**Stopped early.** [`has_solution`](@ref) is true: the iterates reached are a valid point,
simply not a converged one, and are returned rather than discarded.

| status | meaning |
|---|---|
| `MAX_ITER_REACHED` | `max_iter` was spent and even the ten-times check failed |
| `TIME_LIMIT_REACHED` | `time_limit` was spent first; the budget is checked every iteration and the status returned without re-testing the tolerances, so a run can stop at a point that would have passed |
| `INTERRUPTED` | a `Ctrl-C` landed inside the loop |

**No point to return.** [`has_solution`](@ref) is false, and `x` and `y` are filled with
`NaN` rather than a plausible-looking number.

| status | meaning |
|---|---|
| `PRIMAL_INFEASIBLE` | a certificate was found; it is in `Solution.prim_inf_cert` |
| `DUAL_INFEASIBLE` | a certificate was found; it is in `Solution.dual_inf_cert` |
| `PRIMAL_INFEASIBLE_INACCURATE`, `DUAL_INFEASIBLE_INACCURATE` | the same certificates, established only at ten times the requested tolerances |
| `NON_CONVEX` | the residuals diverged, which a convex problem's cannot |
| `NUMERICAL_ERROR` | the interior-point method could not continue: its Newton system stayed unfactorizable after `max_reg_bumps` regularization increases, a residual stopped being finite, conjugate gradients missed `cg_fail_limit` Newton solves in a row (a preconditioner that is not symmetric positive definite is one cause), or the iteration stalled with no certificate. ADMM never returns it |
| `UNSOLVED` | the loop has not run. It is a workspace's state before its first [`solve!`](@ref) and never the status of a completed solve |

An unconverged result is never reported as `SOLVED`.

Failures that are not outcomes of the algorithm — a non-symmetric `P`, a bad setting, a
dimension mismatch — raise instead of returning a status, so there is no error code to
inspect and no way to miss one by not looking.
"""
@enum Status begin
    UNSOLVED
    SOLVED
    SOLVED_INACCURATE
    PRIMAL_INFEASIBLE
    PRIMAL_INFEASIBLE_INACCURATE
    DUAL_INFEASIBLE
    DUAL_INFEASIBLE_INACCURATE
    MAX_ITER_REACHED
    TIME_LIMIT_REACHED
    INTERRUPTED
    NON_CONVEX
    NUMERICAL_ERROR
end

"""
    PolishStatus

Outcome of the polishing step, reported as `Solution.status_polish`. Polishing guesses the
active set, so it can decline for reasons that are not failures.

| status | value | meaning |
|---|---|---|
| `POLISH_LINSYS_ERROR` | -2 | the active-set KKT matrix could not be factored |
| `POLISH_FAILED` | -1 | the polished point did not improve both residuals, so it was discarded |
| `POLISH_NOT_PERFORMED` | 0 | `polishing` was off, which is the default |
| `POLISH_SUCCESS` | 1 | the polished point was accepted and is what `Solution` carries |
| `POLISH_NO_ACTIVE_SET_FOUND` | 2 | no constraint was active, so there was nothing to polish |

Only `POLISH_SUCCESS` changes the answer. `Solution.polished` is the narrower question of
whether that happened; the other four all leave the solver's own point in place, and only the
first two describe something going wrong.
"""
@enum PolishStatus begin
    POLISH_LINSYS_ERROR = -2
    POLISH_FAILED = -1
    POLISH_NOT_PERFORMED = 0
    POLISH_SUCCESS = 1
    POLISH_NO_ACTIVE_SET_FOUND = 2
end

"Statuses that carry a meaningful primal-dual point."
@inline has_solution(s::Status) =
    s === SOLVED || s === SOLVED_INACCURATE || s === MAX_ITER_REACHED ||
    s === TIME_LIMIT_REACHED || s === INTERRUPTED

"Name of a status, for `verbose` output and for `Solution`'s `show` method."
function status_name(s::Status)
    s === SOLVED && return "solved"
    s === SOLVED_INACCURATE && return "solved inaccurate"
    s === PRIMAL_INFEASIBLE && return "primal infeasible"
    s === PRIMAL_INFEASIBLE_INACCURATE && return "primal infeasible inaccurate"
    s === DUAL_INFEASIBLE && return "dual infeasible"
    s === DUAL_INFEASIBLE_INACCURATE && return "dual infeasible inaccurate"
    s === MAX_ITER_REACHED && return "maximum iterations reached"
    s === TIME_LIMIT_REACHED && return "time limit reached"
    s === INTERRUPTED && return "interrupted"
    s === NON_CONVEX && return "problem non convex"
    s === NUMERICAL_ERROR && return "numerical error"
    return "unsolved"
end

"Name of a polishing outcome, for messages."
function polish_status_name(s::PolishStatus)
    s === POLISH_SUCCESS && return "success"
    s === POLISH_FAILED && return "failed"
    s === POLISH_LINSYS_ERROR && return "linear system error"
    s === POLISH_NO_ACTIVE_SET_FOUND && return "no active set found"
    return "not performed"
end

@inline INFTY(::Type{T}) where {T} = min(T(1.0e30), prevfloat(typemax(T)))
@inline MIN_SCALING(::Type{T}) where {T} = T(1.0e-4)
@inline MAX_SCALING(::Type{T}) where {T} = T(1.0e4)
@inline RHO_MIN(::Type{T}) where {T} = T(1.0e-6)
@inline RHO_MAX(::Type{T}) where {T} = T(1.0e6)
@inline RHO_TOL(::Type{T}) where {T} = T(1.0e-4)
@inline RHO_EQ_OVER_INEQ(::Type{T}) where {T} = T(1.0e3)
@inline DIVISION_TOL(::Type{T}) where {T} = one(T) / INFTY(T)
"Multipliers below this are treated as zero in the duality gap, where a 1e-20 `y` against a
large bound would otherwise dominate the sum."
@inline ZERO_DEADZONE(::Type{T}) where {T} = T(1.0e-10)


"""
    Solution{T}

Result of a solve. `x` and `y` are in the caller's problem space. When the status is an
infeasibility, the corresponding certificate is populated and `x`/`y` are filled with
`NaN`; otherwise the certificates are empty. `obj_val` is infinite there rather than `NaN`,
and its sign says which infeasibility it was: `Inf` is the infimum over an empty feasible
set, `-Inf` an objective unbounded below.

`duality_gap` is `xᵀPx + qᵀx + SC(y)`, where `SC` is the support function of `[l, u]`; it
is zero at an exact solution and is reported unscaled. `rel_kkt_error` is the largest of
the two residuals and the gap, so one number bounds how far the point is from optimal.
`rho_updates` counts adaptive-`ρ` changes only, unlike the workspace's `refactor_count`, which
also counts refactorizations forced by new data.

`accel_declined` counts the accelerated steps this solve discarded because they did worse than
the plain step allowed (see `safeguard_tol` in `PureOSQP.anderson`). It is zero without
an accelerator. A count close to `iter` means nearly every proposal was discarded and the
accelerator is only adding work.

`cg_iters` is the number of conjugate-gradient iterations this solve's linear solves took, on
`linsys = :indirect`; it is zero on every direct backend.

`primdual_int` and `primdual_int_log` are the primal-dual integral, `∫|gap| dt` over the
solve, and are zero unless `profile_primdual` was set. They differ only in how the gap is
interpolated between the iterations that sampled it: the first joins samples with a straight
line, the second with an exponential, which is what a geometrically decaying gap does. The
logarithmic mean never exceeds the arithmetic one, so the second is the lower of the two and
the pair brackets the integral; measured, they differ by 1.3× to 4.2×
(`PureOSQP/bench/primdual_integral.jl`).

**Both integrate against wall-clock time and neither is reproducible.** They cannot be
compared across machines, or between runs on a machine whose clock is not pinned, and no
test asserts a value for either — only that they are positive, ordered, and unaffected by
the measuring. They exist to compare convergence *profiles*, not to be an output of the
solve.

Times are in seconds. `setup_time` belongs to the workspace and is reported by every solve
that uses it, but `run_time` counts it only for the first solve — a re-solve did not pay
it again, so adding it in would overstate the cost of the loop that `update!` exists to
make cheap.

`update_time` is the time spent in [`update!`](@ref) since the previous solve, accumulated
across however many calls were made, and it *is* counted in `run_time`: in the loop
`update!` exists for, a cycle is an update followed by a solve, and that pair is what the
caller pays. It resets once reported, so each solve accounts for its own updates and no
others.

!!! warning "An algorithm may hand back the same `Solution` on every solve"
    The type is mutable so that an algorithm can keep one and refill it, which is what lets a
    solve allocate nothing at all — a fresh one costs an allocation for the object itself
    however few of its arrays are new. `ActiveSet` does this: the object
    [`solve!`](@ref) returns is the workspace's own, its `x` and `y` are the workspace's own
    arrays, and the next solve writes through all of it.

    So a `Solution` held across a solve is not a record of the earlier one. To keep an
    answer, copy what you need — `copy(sol.x)`, `sol.obj_val` — before solving again.
    Reading it straight after the solve that produced it, which is what almost every caller
    does, is unaffected.

    `OperatorSplitting` and `InteriorPoint` return a fresh one per solve.
"""
mutable struct Solution{T <: Real}
    x::Vector{T}
    y::Vector{T}
    status::Status
    obj_val::T
    dual_obj_val::T
    duality_gap::T
    prim_res::T
    dual_res::T
    rel_kkt_error::T
    iter::Int
    # Zero unless `profile_primdual` was set. Wall-clock quantities: not reproducible across
    # machines, and not comparable between runs on a machine whose clock is not pinned.
    primdual_int::Float64
    primdual_int_log::Float64
    rho_estimate::T
    rho_updates::Int
    accel_declined::Int
    cg_iters::Int
    polished::Bool
    status_polish::PolishStatus
    setup_time::Float64
    update_time::Float64
    solve_time::Float64
    polish_time::Float64
    run_time::Float64
    prim_inf_cert::Vector{T}
    dual_inf_cert::Vector{T}
end

"""
    reserved(T, n) -> Vector{T}

An empty `Vector{T}` that can grow to `n` elements without allocating.

`resize!` up to `n` and back reuses the memory reserved here, which is what lets a workspace
hand back a field whose length depends on how a run ended without allocating during the run.
"""
reserved(::Type{T}, n::Integer) where {T} = resize!(Vector{T}(undef, n), 0)

"""
    empty_solution(x, y, prim_cert, dual_cert) -> Solution

The [`Solution`](@ref) a workspace keeps and refills, reporting through the arrays it is
given and holding no answer yet.

A solve writes every field before a caller sees it, so the values here are placeholders.
The four arrays belong to the workspace: a caller holding a `Solution` across a second
solve sees the second solve's numbers.
"""
function empty_solution(
        x::Vector{T}, y::Vector{T}, prim_cert::Vector{T}, dual_cert::Vector{T}
    ) where {T}
    return Solution{T}(
        x, y, UNSOLVED, zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), 0,
        0.0, 0.0, zero(T), 0, 0, 0,
        false, POLISH_NOT_PERFORMED,
        0.0, 0.0, 0.0, 0.0, 0.0,
        prim_cert, dual_cert,
    )
end

"""
    copy(sol::Solution) -> Solution

A `Solution` that keeps this run's numbers when the workspace solves again.

A solve refills the workspace's own `Solution` rather than building one, so two results
read from the same workspace are the same object. Copy the one you need to keep before
solving again; comparing a run against a later one otherwise compares it against itself.
"""
function Base.copy(sol::Solution{T}) where {T}
    return Solution{T}(
        copy(sol.x), copy(sol.y), sol.status, sol.obj_val, sol.dual_obj_val,
        sol.duality_gap, sol.prim_res, sol.dual_res, sol.rel_kkt_error, sol.iter,
        sol.primdual_int, sol.primdual_int_log, sol.rho_estimate, sol.rho_updates,
        sol.accel_declined, sol.cg_iters, sol.polished, sol.status_polish,
        sol.setup_time, sol.update_time, sol.solve_time, sol.polish_time, sol.run_time,
        copy(sol.prim_inf_cert), copy(sol.dual_inf_cert),
    )
end

"""
    unit_certificate!(dest, scratch, s, src, scaled) -> dest

Write `s ⊙ src` — or `src` itself when the problem was not equilibrated — into `dest`,
normalized to unit `∞`-norm, resizing `dest` to the length `src` needs.

A certificate proves an infeasibility direction, so only its direction carries meaning and
any positive multiple of it does as well. `dest` is reserved by the workspace, so the
resize reuses memory rather than taking any.

The product runs in `scratch`, a workspace buffer of the solver's own array type, and
`dest` is a `Vector` the caller reads: an array that forbids scalar indexing is scaled
where it lives and copied across once, rather than indexed element by element.
"""
function unit_certificate!(dest::Vector{T}, scratch, s, src, scaled::Bool) where {T}
    n = length(src)
    resize!(dest, n)
    work = view_n(scratch, n)
    if scaled
        multiply!(work, s, src)
    else
        copyto!(work, src)
    end
    copyto!(dest, work)
    nc = zero(T)
    for i in eachindex(dest)
        nc = max(nc, abs(dest[i]))
    end
    nc > zero(T) && (dest ./= nc)
    return dest
end

"The first `n` entries of `v`, as a view, when `v` is longer than the vector being built."
@inline view_n(v, n::Integer) = length(v) == n ? v : view(v, firstindex(v):(firstindex(v) + n - 1))

# The keyword forms (`warm_start!(ws; x, y)`, `update!(ws; q, l, u, P, A)`,
# `update_settings!(ws; kwargs...)`) are checked through their positional signature, which is
# all `hasmethod` sees of a keyword method.
#
# `solve!` returns the workspace's own `Solution`, refilled: a solve allocates nothing, so
# the result is storage the workspace owns rather than a fresh object per run.
@contract QPWorkspace begin
    solve!(::Self)::Solution => "run the algorithm from the workspace's state and refill the workspace's result"
    warm_start!(::Self)::Self => "seed the next solve with `x` and `y`, given as keywords in problem space"
    cold_start!(::Self)::Self => "discard the iterates, so the next solve starts from its own starting point"
    update!(::Self)::Self => "replace `q`, `l`, `u`, `P` or `A`, given as keywords"
    update_settings!(::Self)::Self => "merge the keywords into `ws.options`"
    update_settings!(::Self, ::QPAlgorithm)::Self => "replace `ws.algorithm`; throws for another algorithm's object"
    dimensions(::Self)::Tuple{Int, Int} => "the number of variables and of constraint rows"
    derivative_ready(::Self)::Nothing => "throw unless the iterate's multipliers are ones the active-set test can read"
    :optional
    update_rho!(::Self, ::Real)::Self => "set the ADMM step size and refactorize"
    constraint_violation(::Self)::AbstractVector => "the violation of each row at the current iterate; `constraint_violation!` writes it in place"
end

# `setup_backend` declares no return type: inferred through the abstract data arguments of this
# signature it is `Any`, although every call with concrete arguments returns a concrete workspace.
@contract QPAlgorithm begin
    setup_backend(::Self, ::Val, ::Type{<:Real}, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Options, ::Any, ::Any) => "build and factorize the workspace; `setup` calls it with the backend name lifted into the `Val`, the options it built, and the caller's preconditioner and accelerator"
    algorithm_defaults(::Self, ::Type{<:Real})::NamedTuple => "the `Options` defaults of this algorithm that have no common default"
    default_options(::Self, ::Type{<:Real})::Options => "the `Options` a solve runs with when no keyword is passed"
    element_typed(::Self, ::Type{<:Real}, ::Options)::QPAlgorithm => "the object in the solve's element type, every default resolved"
    :optional
    adopt_settings!(::LinearSystem, ::Self, ::Options)::Nothing => "copy the parameters a backend reads into it; needed for the matrix-free backend"
end

"""
    Solution show, in one line: the status, the iterations, and — when the run carries a
    meaningful point — the objective and the residuals of it.
"""
function Base.show(io::IO, s::Solution{T}) where {T}
    print(io, "Solution{", T, "}: ", status_name(s.status), ", ", s.iter, " iterations")
    if has_solution(s.status)
        print(io, ", objective ", s.obj_val, ", primal residual ", s.prim_res, ", dual residual ", s.dual_res)
        s.polished && print(io, ", polished")
    else
        print(io, ", no point")
    end
    return nothing
end

"""
    check_bounds(l, u)

Throw unless `l ≤ u` elementwise, naming the first index that violates it.
"""
function check_bounds(l::Vector, u::Vector)
    for i in eachindex(l)
        l[i] <= u[i] || throw(ArgumentError("l must be elementwise ≤ u, violated at index $i: $(l[i]) > $(u[i])"))
    end
    return nothing
end

function check_bounds(l, u)
    # A whole-array reduction rather than an indexed loop, so an array that forbids scalar
    # indexing still validates. Naming the offending row needs indexing, so that runs on a
    # host copy and is paid only when the throw is happening anyway.
    all(l .<= u) && return nothing
    return check_bounds(Array(l), Array(u))
end

"""
    is_symmetric(M) -> Bool

Whether `M` equals its transpose, which [`setup`](@ref) requires of `P`.

The generic method is `issymmetric`, an entrywise scan over all `n²` positions. A
representation whose entries are structurally zero outside a known set overrides this and
compares only that set — `PureQPBase/ext/PureQPBaseBandedMatricesExt.jl` does, where the generic scan is
the largest single term in a banded `setup`. It is an override point for the same reason
[`is_convex`](@ref) is: the cost is a property of the representation, not of the problem.
"""
is_symmetric(M) = issymmetric(M)

function validate(P, q, A, l, u)
    n = size(P, 1)
    size(P, 2) == n || throw(ArgumentError("P must be square, got size $(size(P))"))
    size(A, 2) == n || throw(ArgumentError("size(A, 2) = $(size(A, 2)) must equal size(P, 1) = $n"))
    m = size(A, 1)
    length(q) == n || throw(ArgumentError("length(q) = $(length(q)) must equal size(P, 1) = $n"))
    length(l) == m || throw(ArgumentError("length(l) = $(length(l)) must equal size(A, 1) = $m"))
    length(u) == m || throw(ArgumentError("length(u) = $(length(u)) must equal size(A, 1) = $m"))
    # The factorizations run with `check = false` and would not reliably report a non-finite
    # entry, so a stray NaN or Inf is refused here rather than answered with. This precedes
    # the symmetry test because `NaN != NaN`: a `P` holding one is not equal to its own
    # transpose, and would otherwise be refused for the wrong reason.
    is_materializable(P) && check_finite(P, n, n, "P")
    is_materializable(A) && check_finite(A, m, n, "A")
    # An operator has no entries to inspect, so it reports what its author declared and the
    # remedy is the declaration, not the storage.
    is_symmetric(P) || throw(
        ArgumentError(
            is_materializable(P) ?
                "P must be symmetric. Pass the full matrix or a Symmetric wrapper, not a stored triangle." :
                "P must be symmetric, and an operator reports only what it was told: build it with `issymmetric = true`, or wrap it with `ProductOperator{T}(op; symmetric = true)`."
        )
    )
    is_materializable(P) || check_symmetric_products(P, q)
    all(isfinite, q) || throw(ArgumentError("q must be finite, found NaN or Inf"))
    any(isnan, l) && throw(ArgumentError("l contains NaN"))
    any(isnan, u) && throw(ArgumentError("u contains NaN"))
    any(==(Inf), l) && throw(ArgumentError("l may not be +Inf"))
    any(==(-Inf), u) && throw(ArgumentError("u may not be -Inf"))
    check_bounds(l, u)
    check_storage(P, n, n)
    check_storage(A, m, n)
    return (n, m)
end

"""
    check_storage(M, rows, cols)

Establish that traversing `M`'s stored entries stays in range, or throw.

A representation whose column traversals index a weight vector by an index read out of the
matrix overrides this, so that those traversals can drop their per-entry bounds check. That
is worth 7.7× on the equilibration sweeps of a `P` holding 39 638 entries, where the check is
most of the per-entry work.

The generic method has nothing to check: [`structural_rows`](@ref) answers with `axes(M, 1)`,
which the compiler can already prove.
"""
check_storage(M, rows::Integer, cols::Integer) = nothing

# `@constprop :aggressive` because this frame forwards the keywords that the method it calls
# turns into the `Val` naming the backend: constant propagation needs every frame in the
# chain to carry it, and a frame that only forwards is still a frame. The annotation goes on
# the method below; a comment between a docstring and its definition detaches the docstring.
"""
    setup(P, q, A, l, u, alg; kwargs...) -> QPWorkspace

Build a workspace for `min ½xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u`.

`P` must be a full symmetric matrix (or a `Symmetric` wrapper), not a stored triangle.
`P` and `A` may be any `AbstractMatrix` and are not copied or modified. `alg` is the
algorithm and its parameters; the keyword arguments are the fields of [`Options`](@ref), whose
defaults depend on `alg` ([`default_options`](@ref)), plus `preconditioner` and, for an
algorithm that accepts one, `accelerator`. A keyword that is a parameter of an algorithm
(`rho`, `reg_primal`, …) throws, naming the algorithm object it belongs in.

An `OperatorSplitting` algorithm builds an `OperatorSplittingWorkspace`. An `InteriorPoint`
algorithm builds an `InteriorPointWorkspace` instead. It runs on the host for any real
element type and refuses GPU arrays,
`linsys = :kronecker` and `linsys = :lowrank` by name. It solves an operator that supplies
products only, and runs conjugate gradients on any pair, only with `linsys = :indirect`, a
caller-supplied `preconditioner` and `scaling = 0`; without those it throws, and
`linsys = :auto` never chooses that path.

`preconditioner` applies to `linsys = :indirect` only, and is refused with any other
`linsys`. The default, `nothing`, is a [`JacobiPreconditioner`](@ref);
[`IdentityPreconditioner`](@ref) turns preconditioning off. Any other object is used through
`LinearAlgebra.ldiv!` and refreshed by [`update_preconditioner!`](@ref) whenever the weights
change; it approximates the reduced matrix of the `P` and `A` passed here, so it requires
`scaling = 0`.

A `Symmetric` wrapper is accepted over any parent, but it costs something over a
`SparseMatrixCSC`: the sparse-factorization backends are keyed on that concrete type, so a
wrapped one descends past them, and equilibration walks the wrapper entrywise rather than by
stored column. Pass the full `SparseMatrixCSC` to reach those backends.
"""
Base.@constprop :aggressive function setup(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector, alg::QPAlgorithm; kwargs...
    )
    T = float(promote_type(eltype(P), eltype(q), eltype(A), eltype(l), eltype(u)))
    return setup(T, P, q, A, l, u, alg; kwargs...)
end

# `@constprop :aggressive` because the options have to reach inference as constants. A
# non-default keyword — `scaling = 0`, `linsys = :kkt` — otherwise arrives as a non-singleton
# `Pairs`, the compiler's size heuristic refuses to propagate it into a method this large, and
# `options.scaling` stays unknown, leaving every branch below live. The return then merges one
# workspace type per reachable backend, and past `Base.Compiler.MAX_TYPEUNION_LENGTH` (3) the
# union widens to `OperatorSplittingWorkspace{…} where LS`: every later `solve!` is a dynamic dispatch, which `--trim`
# rejects. An absent keyword leaves the empty `Pairs`, a singleton that folds without help.
#
# `linsys` is lifted out of the keywords and into a `Val` because constant propagation is not
# enough for it: naming a backend has to make the other branches *unreachable*, not merely
# narrow the merged return type, since the branch this eliminates is the one that reaches
# `choose_backend` and, through the sparse ladder, CHOLMOD's bindings. Propagation does not
# enter a method this large — the constant arrives as `linsys::Symbol` — so the choice is
# carried as a type parameter and the dead branches are gone by specialization instead.
Base.@constprop :aggressive function setup(
        ::Type{T}, P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector, alg::QPAlgorithm;
        linsys::Symbol = :auto, preconditioner = nothing, accelerator = nothing, kwargs...
    ) where {T <: Real}
    return build_workspace(
        T, alg, check_linsys(linsys), P, q, A, l, u, preconditioner, accelerator; kwargs...
    )
end

"""
    check_linsys(linsys) -> Val{linsys}

Refuse a backend name that is not one, and lift the name into the `Val` that carries it
through `setup_backend`.

The name is checked before it becomes a type parameter: past that point an unusable one
would specialize the whole of `setup_backend` before the options it cannot satisfy are ever
built. [`setup`](@ref) and [`solve`](@ref) each call this in their own frame, where the
caller's `linsys` keyword is still a literal, and pass the `Val` down by position. Deciding
it any deeper would leave the name to reach `setup_backend` by constant propagation, whose
budget a solve with four keywords already exhausts, and an unresolved call is what `--trim`
rejects.
"""
Base.@constprop :aggressive function check_linsys(linsys::Symbol)
    linsys in LINSYS_OPTIONS || throw(
        ArgumentError(lazy"linsys must be one of $LINSYS_LIST, got :$linsys")
    )
    return Val(linsys)
end

"""
    build_workspace(T, alg, Val(linsys), P, q, A, l, u, preconditioner, accelerator; kwargs...)

Build the [`Options`](@ref) and hand everything to the algorithm's `setup_backend`, which
takes it all by position: a keyword call carries a `NamedTuple` whose names inference loses
track of once more than one keyword survives to it.
"""
function build_workspace(
        ::Type{T}, alg::QPAlgorithm, ::Val{LS}, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, preconditioner, accelerator;
        kwargs...
    ) where {T <: Real, LS}
    check_option_names(kwargs, alg)
    options = Options{T}(; algorithm_defaults(alg, T)..., linsys = LS, kwargs...)
    return setup_backend(
        alg, Val(LS), T, P, q, A, l, u, options, preconditioner, accelerator
    )
end
