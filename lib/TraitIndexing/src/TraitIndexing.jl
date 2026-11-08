"""
    TraitIndexing

Symbolic access as traits (SciML's move 4, E-160; open issue 118): a stand-in for SymbolicIndexingInterface.
- `IndexProvider{I}`: a system maps indices of its own type `I` to a state variable's or a parameter's position,
  or to the getter of an observed value (`observed`).
- `HasSystem`: a value that carries a system (a problem, a solution, an integrator, a callback's view of it).
- `ValueProvider{U, P}`: a state's, the parameters' and the time's values; `SeriesProvider{U, P}`: those values at
  saved times.
- `getu(x, idx)`: the getter of `idx`, read through the system `x` carries, from any value or series provider.
The getters `getstate` and `getparam` by name are written once against the traits, so they are compiled per
index shape, not per provider. `SymbolCache` is SciML's record of names.
"""
module TraitIndexing
using TraitWorld
@trait_module
export IndexProvider, variable_index, parameter_index, observed, ValueProvider, state_values, parameter_values,
       current_time, SeriesProvider, series_times, series_state, series_parameters, HasSystem, system, getsystem,
       SymbolCache, getstate, getparam, ParameterView, getu, getter, Getter
"""
    IndexProvider{I}

A system's indices, of its own index type `I` (`Symbol` for names; a library's own structured indices, such as
NetworkDynamics' `VIndex`/`EIndex` and vectors of them):
- `variable_index(ip, idx)`: the position of a state variable (`nothing`: not one);
- `parameter_index(ip, idx)`: the position of a parameter (`nothing`: not one);
- `observed(ip, idx)`: the getter of a value computed from the state, the parameters and the time, a function
  `g(u, p, t)` kept before the state's types are known (a `Callable`, traits §9); `nothing`
  where the system computes none (the default).
`I` may be abstract or a union: an index is taken where it is an `I`.
"""
@trait IndexProvider{I} begin
    @default_dynamic
    variable_index(ip::IndexProvider{I}, idx::I)::Union{Nothing,Int}
    parameter_index(ip::IndexProvider{I}, idx::I)::Union{Nothing,Int}
    observed(ip::IndexProvider{I}, idx::I)::Union{Nothing,Callable} = nothing
end
"""
    HasSystem

A value that carries a system object (SciML's `f.sys`, `prob.f.sys`, `sol.prob.f.sys`, `integ.f.sys`;
SymbolicIndexingInterface's `symbolic_container`): the model's own description, kept and given back unchanged
(`getsystem`). A problem's function type implements it (a callable model returns itself); a problem, its
solution and its integrator carry the one their function gave. Where the system is an `IndexProvider{I}`, its
indices serve `getu`.
"""
@trait HasSystem begin
    system(x::HasSystem)::Any
end
"""
    getsystem(x)

The system object `x` carries (`HasSystem`: a problem, a solution, an integrator, a callback's view of it, a model),
or `nothing`.
"""
getsystem(x::HasSystem) = system(x)
getsystem(x) = nothing
"A state's, the parameters' and the time's values (SymbolicIndexingInterface's value provider)."
@trait ValueProvider{U, P} begin
    state_values(v::ValueProvider{U, P})::U
    parameter_values(v::ValueProvider{U, P})::P
    current_time(v::ValueProvider{U, P})::Float64
end
"Values at saved times (SymbolicIndexingInterface's time series): the times, and the state and the parameters in force at each."
@trait SeriesProvider{U, P} begin
    series_times(s::SeriesProvider{U, P})::Vector{Float64}
    series_state(s::SeriesProvider{U, P}, k::Int)::U
    series_parameters(s::SeriesProvider{U, P}, k::Int)::P
end
"SciML's record of names: the state's variables and the parameters, in order."
struct SymbolCache
    variables::Vector{Symbol}
    parameters::Vector{Symbol}
end
SymbolCache(vars::AbstractVector) = SymbolCache(collect(Symbol, vars), Symbol[])
@implement IndexProvider{Symbol} SymbolCache begin
    variable_index(ip, sym) = findfirst(==(sym), ip.variables)
    parameter_index(ip, sym) = findfirst(==(sym), ip.parameters)
end
@noinline _noname(kind::String, sym::Symbol) = throw(ArgumentError("no $kind named :$sym"))
"The value of state variable `sym`."
function getstate(v::All{ValueProvider{U, P}, IndexProvider{Symbol}}, sym::Symbol) where {U, P}
    i = variable_index(v, sym)
    i === nothing && _noname("state variable", sym)
    return state_values(v)[i]
end
"The value of parameter `sym`."
function getparam(v::All{ValueProvider{U, P}, IndexProvider{Symbol}}, sym::Symbol) where {U, P}
    i = parameter_index(v, sym)
    i === nothing && _noname("parameter", sym)
    return parameter_values(v)[i]
end
"The parameters `p` by name, `ps[:σ]` (SciML's `integ.ps`): the names from `ip`, or none."
struct ParameterView{P}
    p::P
    ip::Union{Nothing, IndexProvider{Symbol}}
end
function Base.getindex(v::ParameterView, sym::Symbol)
    i = v.ip === nothing ? nothing : parameter_index(v.ip, sym)
    i === nothing && _noname("parameter", sym)
    return v.p[i]
end

# --- getters (open issue 118) ------------------------------------------------------------------------------
"A state variable's getter: `u[i]`."
struct StateGetter
    i::Int
end
(g::StateGetter)(u, p, t) = u[g.i]
"A parameter's getter: `p[i]`."
struct ParamGetter
    i::Int
end
(g::ParamGetter)(u, p, t) = p[g.i]
"The getter of a vector of indices: each index's getter, their values in a vector."
struct VectorGetter
    gs::Vector{Callable}
end
(g::VectorGetter)(u, p, t) = [f(u, p, t) for f in g.gs]
"""
    getter(sys, idx) -> Callable

The getter of index `idx` in system `sys` (an `IndexProvider{I}`): a function `g(u, p, t)`. A state variable's
or a parameter's by its position; otherwise the system's `observed` getter. A vector of indices that is not
itself an `I` gives the vector of its indices' values.
"""
function getter(ip::IndexProvider{I}, idx) where {I}
    idx isa I || return _getter_other(ip, idx)
    i = variable_index(ip, idx)
    i === nothing || return StateGetter(i)
    j = parameter_index(ip, idx)
    j === nothing || return ParamGetter(j)
    g = observed(ip, idx)
    g === nothing && throw(ArgumentError(string("the system has no variable, parameter or observed value ", repr(idx))))
    return g
end
function _getter_other(ip::IndexProvider{I}, idx) where {I}
    if idx isa AbstractVector
        gs = Callable[]                  # a loop, not a comprehension: a closure would capture `ip` (traits §7.3)
        for i in idx
            push!(gs, getter(ip, i))
        end
        return VectorGetter(gs)
    end
    throw(ArgumentError(string("an index of type ", typeof(idx), " is not one of the system's (", I, ")")))
end
getter(x, idx) = throw(ArgumentError(string("a value of type ", typeof(x), " provides no indices (`IndexProvider`): ",
    "no getter for ", repr(idx))))

"""
    Getter

The getter `getu` returns. `g(v)`: the value at a value provider's state, parameters and time (an integrator, a
callback's view of it); at a series provider (a solution), the values at each saved time, a vector. `g(u, p, t)`:
the value at that state, those parameters and that time.
"""
struct Getter
    f::Callable                      # g(u, p, t), kept before the state's types are known
end
(g::Getter)(u, p, t) = g.f(u, p, t)
(g::Getter)(v) = _read(v, g.f)
_read(v::ValueProvider{U, P}, f) where {U, P} = f(state_values(v), parameter_values(v), current_time(v))
_read(s::SeriesProvider{U, P}, f) where {U, P} = _series(s, f)
# a value of both (a solution: its final values, and its saved ones) is read as a series, as SymbolicIndexingInterface reads a time series
_read(s::All{SeriesProvider{U, P}, ValueProvider{U, P}}, f) where {U, P} = _series(s, f)
# a loop, not a comprehension: a closure would capture `s` (traits §7.3); the element type as `collect` gives it
function _series(s, f)
    ts = series_times(s)
    out = Vector{Any}(undef, length(ts))
    for k in eachindex(ts)
        out[k] = f(series_state(s, k), series_parameters(s, k), ts[k])
    end
    return [x for x in out]
end
_read(v, f) = throw(ArgumentError(string("a value of type ", typeof(v), " provides no values (`ValueProvider`, `SeriesProvider`)")))

"The system whose indices `x` is read by: the one it carries (`HasSystem`), or `x` itself."
_system_of(x::HasSystem) = system(x)
_system_of(x) = x
"""
    getu(x, idx) -> Getter

The getter of `idx` (SymbolicIndexingInterface's `getu`): `x` is a system, or a value that carries one
(`HasSystem`: a problem, a solution, an integrator, a callback's view of it), and `idx` an index of the system's
own type, or a vector of them. `getu(sol, idx)(sol)` gives the values at the solution's saved times;
`getu(integ, idx)(integ)` the value now.
"""
getu(x, idx) = Getter(getter(_system_of(x), idx))
end
