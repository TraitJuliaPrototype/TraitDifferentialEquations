# The front end of the problem types, in their owner (SciML's move 3; E-159, stage B): `ODEFunction` and the
# constructor of `ODEProblem` from a plain function, as SciMLBase holds them. Part of the trait module.
using LinearAlgebra: I, UniformScaling

"""
    ODEFunction(f; jac = nothing, mass_matrix = I)

SciML's record of the user's functions: the right-hand side `f(du, u, p, t)`, optionally its
Jacobian `jac(J, u, p, t)` in `u`, and a mass matrix. A plain record: it has no parameter for the
state's or the parameters' type, so it keeps the functions as given; `ODEProblem` enters them at
its bounded parameters, where those types are known. Casts reach through it:
`ODEFunction(compiled_for(f); jac = compiled_for(j))`. The solvers take the identity mass matrix
only (`ArgumentError` otherwise).
"""
struct ODEFunction{F,TJ,TMM}
    f::F
    jac::TJ
    mass_matrix::TMM
    sys::Any                         # an `IndexProvider` (a `SymbolCache`), or `nothing`
end
ODEFunction(f; jac = nothing, mass_matrix = I, sys = nothing) = ODEFunction(f, jac, mass_matrix, sys)

"""
    ODEProblem(f, u0, tspan, p = nothing)
    ODEProblem(ODEFunction(f; jac), u0, tspan, p = nothing)

A problem in SciML's form: `f(du, u, p, t)` in place, an initial state, a time span, parameters.
The state keeps its element type (`Float32` stays `Float32`), or takes the parameters' dual type
for a gradient in `p`, as SciML promotes it. The function enters at its declared signature
`(Vector{T}, Vector{T}, P, Float64) -> Nothing`, and at the library's dual numbers for the
Jacobian: `Dynamic` by default (`Callable`'s), so the solver's code is compiled once per state and
parameter type, never per function; `ODEProblem(compiled_for(f), …)` keeps the function itself, so
the solver is compiled for it, with `f` inferred into it and its result checked at every call. A
Jacobian given in an `ODEFunction` is entered at `(Matrix{T}, Vector{T}, P, Float64) -> Nothing`
the same way, and Rodas5 calls it instead of differentiating `f`.
"""
# neither specialized nor inferred on the function: only the function itself is compiled, at its signature
Base.@nospecializeinfer function ODEProblem(@nospecialize(f), u0::AbstractVector, tspan, p = nothing)
    u = _promote_state(_state(u0), p)
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    if f isa ODEFunction
        f.mass_matrix isa UniformScaling && f.mass_matrix == I ||
            throw(ArgumentError("TraitDifferentialEquations' solvers take the identity mass matrix only"))
        return _problem(f.f, f.jac, u, ts, p, f.sys)
    end
    return _problem(f, nothing, u, ts, p, nothing)
end
# the functions partially evaluated on the parameters (W-63) at the problem's bounds, which have no parameter
# for the parameters' type: the attribute decides each one's form (traits reference §7.4); the partially
# evaluated function beside them, for a solver that enters it at its own chunk
function _problem(@nospecialize(f), @nospecialize(jac), u0::Vector{T}, tspan::Tuple{Float64,Float64}, p::P, @nospecialize(sys)) where {T,P}
    r = _partial(f, p)
    # a value of the trait world is entered at each signature on its own (the same function at the duals)
    tw = f isa Union{Dynamic,Kept,Compiled}
    # the Jacobian by duals: the one function, seen at the duals where it is needed (traits reference §5.8, §6.5)
    ad = T === Float64 && jac === nothing
    jk = jac === nothing ? compiled_for(core_no_jacobian) : _partial(jac, p)
    return ODEProblem(r, u0, tspan, p, jk, ad, UserFunction(jac === nothing && !tw ? r : nothing),
                      sys === nothing ? nothing : Dynamic{IndexProvider}(sys))
end
"""
A user's function partially evaluated on the parameters, with its attribute (`rewrap`, traits reference
§6.4): `compiled_for(f)` gives a compiled record, `with_dynamic(f)` a dynamic one. A kept function (a
`Dynamic` callable, perhaps made at another signature) gives the user's function itself, partially
evaluated, so that the one function is seen at each signature, the duals too (§6.5).
"""
Base.@nospecializeinfer _partial(@nospecialize(f), p) = rewrap(Base.Fix2(PartialFn, p), f)

_state(u0::AbstractVector{<:ForwardDiff.Dual}) = Vector(u0)
_state(u0::AbstractVector{T}) where {T<:AbstractFloat} = Vector{T}(u0)      # Float32 states stay Float32 (row 6)
_state(u0::AbstractVector) = Vector{Float64}(u0)
"The state at the parameters' dual type, as SciML promotes `u0` for a gradient in `p`."
_promote_state(u0::Vector{Float64}, p::AbstractArray{D}) where {D<:ForwardDiff.Dual} = Vector{D}(u0)
_promote_state(u0::Vector, p) = u0

