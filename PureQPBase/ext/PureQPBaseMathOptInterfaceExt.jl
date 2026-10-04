module PureQPBaseMathOptInterfaceExt

"""
    PureQPBaseMathOptInterfaceExt

MathOptInterface wrapper, loaded when MathOptInterface is. Constraints arrive as a
`MatrixOfConstraints` with a `Hyperrectangle`, which is already the `l <= Ax <= u` form the
solver wants, so the only assembly left is appending a row per bounded variable.

One wrapper serves every algorithm: an `Optimizer` carries the algorithm's type, and the
package that defines that algorithm defines the `Optimizer()` a caller reaches it by.
"""

import MathOptInterface as MOI
import PureQPBase
using SparseArrays
using LinearAlgebra: dot

const _SETS{T} = Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}

MOI.Utilities.@product_of_sets(
    _RowSets, MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T},
)

const _Cache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    MOI.Utilities.MatrixOfConstraints{
        T,
        MOI.Utilities.MutableSparseMatrixCSC{T, Int, MOI.Utilities.OneBasedIndexing},
        MOI.Utilities.Hyperrectangle{T},
        _RowSets{T},
    },
}

"""
    Optimizer{T}(algorithm; kwargs...)

The wrapper around one algorithm, named by its type — `OperatorSplitting`, `InteriorPoint`,
or another. The solver name and version it reports are the package that algorithm comes
from, and the raw attributes it accepts are the [`PureQPBase.Options`](@ref) names plus that
algorithm's own parameters.
"""
mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    algorithm::Type
    settings::Dict{Symbol, Any}
    silent::Bool
    sense::MOI.OptimizationSense
    obj_constant::T
    sets::_RowSets{T}
    bound_row::Vector{Int}
    n_affine::Int
    n::Int
    P::SparseMatrixCSC{T, Int}
    q::Vector{T}
    A::SparseMatrixCSC{T, Int}
    l::Vector{T}
    u::Vector{T}
    sol::Union{Nothing, PureQPBase.Solution{T}}
    Ax::Vector{T}
end

function Optimizer{T}(algorithm::Type; kwargs...) where {T}
    o = Optimizer{T}(
        algorithm, Dict{Symbol, Any}(), false, MOI.MIN_SENSE, zero(T),
        _RowSets{T}(), Int[], 0, 0,
        spzeros(T, 0, 0), T[], spzeros(T, 0, 0), T[], T[], nothing, T[]
    )
    for (k, v) in kwargs
        MOI.set(o, MOI.RawOptimizerAttribute(String(k)), v)
    end
    return o
end

"The package an algorithm comes from, which is the solver MathOptInterface is talking to."
_package(o::Optimizer) = parentmodule(o.algorithm)

MOI.get(o::Optimizer, ::MOI.SolverName) = string(nameof(_package(o)))
MOI.get(o::Optimizer, ::MOI.SolverVersion) = string(pkgversion(_package(o)))

MOI.is_empty(o::Optimizer) = o.n == 0 && o.n_affine == 0 && o.sol === nothing

function MOI.empty!(o::Optimizer{T}) where {T}
    o.sense = MOI.MIN_SENSE
    o.obj_constant = zero(T)
    o.sets = _RowSets{T}()
    empty!(o.bound_row); o.n_affine = 0; o.n = 0
    o.sol = nothing; empty!(o.Ax)
    return
end

MOI.supports_incremental_interface(::Optimizer) = false

MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{<:_SETS{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{<:_SETS{T}}) where {T} = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex, MOI.ScalarAffineFunction{T}, MOI.ScalarQuadraticFunction{T}}}) where {T} = true

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(o::Optimizer, ::MOI.Silent, v::Bool) = (o.silent = v; nothing)
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent

# Sets `time_limit`, which bounds the solver's iterations only: setup, polishing and the copy
# from the model are not counted against it.
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, v::Real) = (o.settings[:time_limit] = Float64(v); nothing)
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (delete!(o.settings, :time_limit); nothing)
MOI.get(o::Optimizer, ::MOI.TimeLimitSec) = get(o.settings, :time_limit, nothing)

"The parameter names of this optimizer's algorithm, which its raw attributes may name."
_parameter_names(o::Optimizer) = fieldnames(o.algorithm)

_accepts(o::Optimizer, name::Symbol) =
    name in PureQPBase.OPTION_NAMES || name in _parameter_names(o)

"""
    _build(o, settings) -> (algorithm, options)

The algorithm object and the options, built from the raw settings in `settings`: a name of
[`PureQPBase.Options`](@ref) goes to the options, a parameter of the algorithm to the
algorithm object. Both are validated, and returned element-typed with every default resolved.
"""
function _build(o::Optimizer{T}, settings) where {T}
    names = _parameter_names(o)
    a = o.algorithm(; (k => v for (k, v) in settings if k in names)...)
    opts = (k => v for (k, v) in settings if k in PureQPBase.OPTION_NAMES)
    options = PureQPBase.Options{T}(; PureQPBase.algorithm_defaults(a, T)..., opts...)
    return PureQPBase.element_typed(a, T, options), options
end

MOI.supports(o::Optimizer, a::MOI.RawOptimizerAttribute) = _accepts(o, Symbol(a.name))

# A value is checked by building the options and the algorithm object from it when it is set,
# so a bad one throws here rather than at `optimize!`. Settings that take a Symbol also accept
# its name as a String.
function MOI.set(o::Optimizer{T}, a::MOI.RawOptimizerAttribute, v) where {T}
    MOI.supports(o, a) || throw(MOI.UnsupportedAttribute(a))
    name = Symbol(a.name)
    S = name in PureQPBase.OPTION_NAMES ? PureQPBase.Options{T} : typeof(o.algorithm())
    v isa AbstractString && fieldtype(S, name) === Symbol && (v = Symbol(v))
    _build(o, merge(o.settings, Dict(name => v)))
    o.settings[name] = v
    return
end
function MOI.get(o::Optimizer{T}, a::MOI.RawOptimizerAttribute) where {T}
    MOI.supports(o, a) || throw(MOI.UnsupportedAttribute(a))
    name = Symbol(a.name)
    haskey(o.settings, name) && return o.settings[name]
    algorithm, options = _build(o, o.settings)
    return name in PureQPBase.OPTION_NAMES ? getfield(options, name) : getfield(algorithm, name)
end

_csc(A::MOI.Utilities.MutableSparseMatrixCSC{T, Int, MOI.Utilities.OneBasedIndexing}) where {T} =
    SparseMatrixCSC{T, Int}(A.m, A.n, A.colptr, A.rowval, A.nzval)

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    MOI.empty!(dest)
    cache = _Cache{T}()
    index_map = MOI.copy_to(cache, src)
    A_aff = _csc(cache.constraints.coefficients)
    n = size(A_aff, 2)
    dest.n, dest.n_affine = n, size(A_aff, 1)
    dest.sets = cache.constraints.sets
    l = copy(cache.constraints.constants.lower)
    u = copy(cache.constraints.constants.upper)
    lo, up = cache.variables.lower, cache.variables.upper
    dest.bound_row = zeros(Int, n)
    cols = Int[]
    for j in 1:n
        if lo[j] > typemin(T) || up[j] < typemax(T)
            push!(cols, j); dest.bound_row[j] = length(cols)
            push!(l, lo[j]); push!(u, up[j])
        end
    end
    nb = length(cols)
    dest.A = vcat(A_aff, sparse(1:nb, cols, ones(T, nb), nb, n))
    dest.l, dest.u = l, u
    P, q, c = _objective(cache, T, n)
    dest.sense = MOI.get(cache, MOI.ObjectiveSense())
    if dest.sense == MOI.MAX_SENSE
        P, q, c = -P, -q, -c
    end
    dest.P, dest.q, dest.obj_constant = P, q, c
    return index_map
end

function _objective(cache, ::Type{T}, n) where {T}
    if MOI.get(cache, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        return spzeros(T, n, n), zeros(T, n), zero(T)
    end
    F = MOI.get(cache, MOI.ObjectiveFunctionType())
    g = convert(MOI.ScalarQuadraticFunction{T}, MOI.get(cache, MOI.ObjectiveFunction{F}()))
    I, J, V = Int[], Int[], T[]
    for t in g.quadratic_terms
        i, j = t.variable_1.value, t.variable_2.value
        push!(I, i); push!(J, j); push!(V, t.coefficient)
        i == j || (push!(I, j); push!(J, i); push!(V, t.coefficient))
    end
    q = zeros(T, n)
    for t in g.affine_terms
        q[t.variable.value] += t.coefficient
    end
    return sparse(I, J, V, n, n), q, g.constant
end

function MOI.optimize!(o::Optimizer{T}) where {T}
    names = _parameter_names(o)
    parameters = Dict{Symbol, Any}(k => v for (k, v) in o.settings if k in names)
    options = Dict{Symbol, Any}(k => v for (k, v) in o.settings if k in PureQPBase.OPTION_NAMES)
    o.silent && (options[:verbose] = false)
    ws = PureQPBase.setup(T, o.P, o.q, o.A, o.l, o.u, o.algorithm(; parameters...); options...)
    o.sol = PureQPBase.solve!(ws)
    xr = _is_cert(o.sol.status) ? o.sol.dual_inf_cert : o.sol.x
    o.Ax = isempty(xr) ? fill(T(NaN), length(o.l)) : o.A * xr
    return
end

### results

const _TERMINATION = Dict(
    PureQPBase.UNSOLVED => MOI.OTHER_ERROR,
    PureQPBase.SOLVED => MOI.OPTIMAL,
    PureQPBase.SOLVED_INACCURATE => MOI.ALMOST_OPTIMAL,
    PureQPBase.PRIMAL_INFEASIBLE => MOI.INFEASIBLE,
    PureQPBase.PRIMAL_INFEASIBLE_INACCURATE => MOI.ALMOST_INFEASIBLE,
    PureQPBase.DUAL_INFEASIBLE => MOI.DUAL_INFEASIBLE,
    PureQPBase.DUAL_INFEASIBLE_INACCURATE => MOI.ALMOST_DUAL_INFEASIBLE,
    PureQPBase.MAX_ITER_REACHED => MOI.ITERATION_LIMIT,
    PureQPBase.TIME_LIMIT_REACHED => MOI.TIME_LIMIT,
    PureQPBase.INTERRUPTED => MOI.INTERRUPTED,
    PureQPBase.NON_CONVEX => MOI.INVALID_MODEL,
    PureQPBase.NUMERICAL_ERROR => MOI.NUMERICAL_ERROR,
)

function MOI.get(o::Optimizer, ::MOI.TerminationStatus)
    o.sol === nothing && return MOI.OPTIMIZE_NOT_CALLED
    return _TERMINATION[o.sol.status]
end

MOI.get(o::Optimizer, ::MOI.RawStatusString) =
    o.sol === nothing ? "optimize! not called" : string(o.sol.status)

function MOI.get(o::Optimizer, ::MOI.ResultCount)
    o.sol === nothing && return 0
    s = o.sol.status
    return s == PureQPBase.NON_CONVEX || s == PureQPBase.NUMERICAL_ERROR ? 0 : 1
end

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    (o.sol === nothing || attr.result_index != 1) && return MOI.NO_SOLUTION
    s = o.sol.status
    s == PureQPBase.SOLVED && return MOI.FEASIBLE_POINT
    s == PureQPBase.SOLVED_INACCURATE && return MOI.NEARLY_FEASIBLE_POINT
    s == PureQPBase.DUAL_INFEASIBLE && return MOI.INFEASIBILITY_CERTIFICATE
    s == PureQPBase.DUAL_INFEASIBLE_INACCURATE && return MOI.NEARLY_INFEASIBILITY_CERTIFICATE
    _stopped_early(s) && return MOI.UNKNOWN_RESULT_STATUS
    return MOI.NO_SOLUTION
end

function MOI.get(o::Optimizer, attr::MOI.DualStatus)
    (o.sol === nothing || attr.result_index != 1) && return MOI.NO_SOLUTION
    s = o.sol.status
    s == PureQPBase.SOLVED && return MOI.FEASIBLE_POINT
    s == PureQPBase.SOLVED_INACCURATE && return MOI.NEARLY_FEASIBLE_POINT
    # A certificate is promised only when there is one to read. An algorithm that proves
    # infeasibility without producing a separating vector leaves `prim_inf_cert` empty, and
    # claiming a certificate here is what sends `ConstraintDual` past the end of it.
    if s == PureQPBase.PRIMAL_INFEASIBLE || s == PureQPBase.PRIMAL_INFEASIBLE_INACCURATE
        isempty(o.sol.prim_inf_cert) && return MOI.NO_SOLUTION
        return s == PureQPBase.PRIMAL_INFEASIBLE ? MOI.INFEASIBILITY_CERTIFICATE :
            MOI.NEARLY_INFEASIBILITY_CERTIFICATE
    end
    _stopped_early(s) && return MOI.UNKNOWN_RESULT_STATUS
    return MOI.NO_SOLUTION
end

_flip(o::Optimizer, v) = o.sense == MOI.MAX_SENSE ? -v : v

_is_cert(s) = s == PureQPBase.DUAL_INFEASIBLE || s == PureQPBase.DUAL_INFEASIBLE_INACCURATE

"Statuses that stopped on a budget rather than on the residuals, so the point is real but
its status as a solution is unknown."
_stopped_early(s) = s == PureQPBase.MAX_ITER_REACHED || s == PureQPBase.TIME_LIMIT_REACHED ||
    s == PureQPBase.INTERRUPTED
_is_pinf(s) = s == PureQPBase.PRIMAL_INFEASIBLE || s == PureQPBase.PRIMAL_INFEASIBLE_INACCURATE

function MOI.get(o::Optimizer{T}, attr::MOI.ObjectiveValue) where {T}
    MOI.check_result_index_bounds(o, attr)
    if _is_cert(o.sol.status)
        return _flip(o, dot(o.q, o.sol.dual_inf_cert))
    end
    return _flip(o, o.sol.obj_val + o.obj_constant)
end

function _support(o::Optimizer{T}, y) where {T}
    s = zero(T)
    for i in eachindex(y)
        yi = y[i]
        abs(yi) <= 1.0e-10 && continue
        s -= yi > 0 ? o.u[i] * yi : o.l[i] * yi
    end
    return s
end

function MOI.get(o::Optimizer{T}, attr::MOI.DualObjectiveValue) where {T}
    MOI.check_result_index_bounds(o, attr)
    _is_pinf(o.sol.status) && return _flip(o, _support(o, o.sol.prim_inf_cert))
    return _flip(o, o.sol.dual_obj_val + o.obj_constant)
end

MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.sol.run_time
MOI.get(o::Optimizer, ::MOI.BarrierIterations) = o.sol.iter

function MOI.get(o::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(o, attr)
    _is_cert(o.sol.status) && return o.sol.dual_inf_cert[vi.value]
    return o.sol.x[vi.value]
end

_row(o::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex}) = o.n_affine + o.bound_row[ci.value]
_row(o::Optimizer, ci::MOI.ConstraintIndex{<:MOI.ScalarAffineFunction}) = MOI.Utilities.rows(o.sets, ci)

function MOI.get(o::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(o, attr)
    return o.Ax[_row(o, ci)]
end

_split(::Type{<:MOI.GreaterThan}, d) = max(d, zero(d))
_split(::Type{<:MOI.LessThan}, d) = min(d, zero(d))
_split(::Type{<:Union{MOI.EqualTo, MOI.Interval}}, d) = d

function MOI.get(o::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    MOI.check_result_index_bounds(o, attr)
    y = _is_pinf(o.sol.status) ? o.sol.prim_inf_cert : o.sol.y
    d = -y[_row(o, ci)]
    return F === MOI.VariableIndex ? _split(S, d) : d
end

MOI.get(o::Optimizer, ::MOI.NumberOfVariables) = o.n
MOI.get(o::Optimizer, ::MOI.ListOfVariableIndices) = MOI.VariableIndex.(1:o.n)

"""
    warm_up(factory)

Solve one small model through `factory`, which compiles the bridge and caching-optimizer code
specialized on the optimizer it builds. That compilation costs far more than the model it
solves, and a caller pays it on their first solve otherwise.

A package that supplies an algorithm calls this from its own test or benchmark setup. It is
not run while this extension is precompiled: the code being compiled is specialized on the
algorithm, which this package does not have.
"""
function warm_up(factory)
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(factory; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.EqualTo(1.0))
    MOI.add_constraint(model, 1.0 * x[1] - 1.0 * x[2], MOI.LessThan(0.5))
    MOI.add_constraint(model, 1.0 * x[2], MOI.GreaterThan(0.0))
    MOI.add_constraint(model, x[1], MOI.Interval(-1.0, 1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        1.0 * x[1] * x[1] + 1.0 * x[2] * x[2] + 1.0 * x[1],
    )
    MOI.optimize!(model)
    MOI.get(model, MOI.TerminationStatus())
    MOI.get(model, MOI.VariablePrimal(), x)
    MOI.get(model, MOI.ObjectiveValue())
    return nothing
end

end
