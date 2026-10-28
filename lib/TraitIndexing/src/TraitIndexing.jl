"""
    TraitIndexing

Symbolic access as traits (SciML's move 4, E-160): a stand-in for SymbolicIndexingInterface. `IndexProvider`
maps names to indices; `ValueProvider{U, P}` gives the state's and the parameters' values; the getters are
written once against them, so they are compiled per index shape, not per provider. `SymbolCache` is SciML's
record of names.
"""
module TraitIndexing
using TraitWorld
export IndexProvider, variable_index, parameter_index, ValueProvider, state_values, parameter_values,
       SymbolCache, getstate, getparam
@trait_module begin
"Names to indices: of the state's variables and of the parameters (`nothing`: no such name)."
@trait IndexProvider begin
    @default_with_dynamic_self
    variable_index(ip::IndexProvider, sym::Symbol)::Union{Nothing,Int}
    parameter_index(ip::IndexProvider, sym::Symbol)::Union{Nothing,Int}
end
"A state's and parameters' values, by index."
@trait ValueProvider{U, P} begin
    state_values(v::ValueProvider{U, P})::U
    parameter_values(v::ValueProvider{U, P})::P
end
"SciML's record of names: the state's variables and the parameters, in order."
struct SymbolCache
    variables::Vector{Symbol}
    parameters::Vector{Symbol}
end
SymbolCache(vars::AbstractVector) = SymbolCache(collect(Symbol, vars), Symbol[])
@implement IndexProvider SymbolCache begin
    variable_index(ip, sym) = findfirst(==(sym), ip.variables)
    parameter_index(ip, sym) = findfirst(==(sym), ip.parameters)
end
@noinline _noname(kind::String, sym::Symbol) = throw(ArgumentError("no $kind named :$sym"))
"The value of state variable `sym`."
function getstate(v::(ValueProvider{U, P}, IndexProvider), sym::Symbol) where {U, P}
    i = variable_index(v, sym)
    i === nothing && _noname("state variable", sym)
    return state_values(v)[i]
end
"The value of parameter `sym`."
function getparam(v::(ValueProvider{U, P}, IndexProvider), sym::Symbol) where {U, P}
    i = parameter_index(v, sym)
    i === nothing && _noname("parameter", sym)
    return parameter_values(v)[i]
end
end
end
