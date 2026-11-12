# The front end of the problem types, in their owner (SciML's move 3; E-159, stage B): `ODEFunction` and the
# constructor of `ODEProblem` from a plain function, as SciMLBase holds them. Part of the trait module.
using LinearAlgebra: I, UniformScaling

"""
    ODEFunction(f; jac = nothing, mass_matrix, sys, jac_prototype)

SciML's record of the user's functions: the right-hand side `f(du, u, p, t)`, optionally its
Jacobian `jac(J, u, p, t)` in `u`, and the function's optional properties: a mass matrix, the system
object, a Jacobian prototype. Each defaults to what `f` carries (`HasMassMatrix`, `HasSystem`,
`HasJacobianPrototype`; the identity, `nothing`, `nothing` otherwise), so `ODEFunction(model)` keeps a
callable model's properties. A plain record: it has no parameter for the state's or the parameters'
type, so it keeps the functions as given; `ODEProblem` enters them at its bounded parameters, where those
types are known, at `Callable`'s default, `Dynamic` (a cast on a function inside the record does not reach
there: open issue 159). It is callable as `f` is (SciML's is), and carries its properties (it implements the
three traits).
"""
struct ODEFunction{F,TJ,TMM}
    f::F
    jac::TJ
    mass_matrix::TMM
    sys::Any                         # the system object (an `IndexProvider` such as a `SymbolCache`, a model), or `nothing`
    jac_prototype::Any               # a sparsity prototype, or `nothing`
end
Base.@nospecializeinfer ODEFunction(@nospecialize(f); @nospecialize(jac = nothing), @nospecialize(mass_matrix = _mass_matrix(f)),
                                    @nospecialize(sys = _system(f)), @nospecialize(jac_prototype = _jac_prototype(f))) =
    ODEFunction(f, jac, mass_matrix, sys, jac_prototype)
(f::ODEFunction)(du, u, p, t) = f.f(du, u, p, t)
@implement HasMassMatrix ODEFunction{F,TJ,TMM} where {F,TJ,TMM} begin
    mass_matrix(f) = f.mass_matrix
end
@implement HasSystem ODEFunction{F,TJ,TMM} where {F,TJ,TMM} begin
    system(f) = f.sys
end
@implement HasJacobianPrototype ODEFunction{F,TJ,TMM} where {F,TJ,TMM} begin
    jac_prototype(f) = f.jac_prototype
end

"""
    ODEProblem(f, u0, tspan, p = nothing; callback = nothing)
    ODEProblem(ODEFunction(f; jac), u0, tspan, p = nothing; callback = nothing)

A problem in SciML's form: `f(du, u, p, t)` in place, an initial state, a time span, parameters.
The state keeps its element type (`Float32` stays `Float32`), or takes the parameters' dual type
for a gradient in `p`, as SciML promotes it. The function is a trait argument of this
constructor, `Callable` at `(Vector{T}, Vector{T}, P, Float64) -> Nothing` (traits §8.2), and is kept
with its attribute (`keep`): `Dynamic` by default (`Callable`'s), so the solver's code is compiled once
per state and parameter type, never per function; `ODEProblem(compiled_for(f), …)` keeps the function
itself, so the solver is compiled for it, with `f` inferred into it and its result checked at every call.
The problem keeps it partially evaluated on the parameters, with that attribute. A Jacobian given in an
`ODEFunction` is entered at `(Matrix{T}, Vector{T}, P, Float64) -> Nothing`, `Dynamic`, and Rodas5 calls
it instead of differentiating `f`.

The function's optional properties are read here, from the `ODEFunction` or from the function itself
(open issue 119): its mass matrix, system object and Jacobian prototype. The problem keeps them;
`solve` and `init` refuse a mass matrix other than the identity for an algorithm that takes none. The
callbacks given (one, a tuple or a vector) are kept: `solve` and `init` take them before their own.
"""
# the constructor with a trait argument (traits §8.2): a cast at this call decides the function's attribute,
# else `Callable`'s default; the signature's variables come from `u0` and `p`. Open on purpose (§7.3): an
# adapter, it stores the function with its attribute
@unbounded function ODEProblem(f::Callable{RHSSig{P,T},Nothing}, u0::AbstractVector{T}, tspan, p::P;
                    @nospecialize(callback = nothing)) where {T,P}
    k = keep(f)
    return _front(k isa Dynamic ? unwrap(k) : k, k isa Dynamic, u0, tspan, p, callback)
end
# without parameters: the function passed on with its attribute (traits §7.5)
ODEProblem(f::Callable{RHSSig{Nothing,T},Nothing}, u0::AbstractVector{T}, tspan; @nospecialize(callback = nothing)) where {T} =
    ODEProblem(f, u0, tspan, nothing; callback)
# a function that is no `Callable` at the state's type (its methods take other types): kept `Dynamic`, as by default
Base.@nospecializeinfer ODEProblem(@nospecialize(f), u0::AbstractVector, tspan, p = nothing; @nospecialize(callback = nothing)) =
    _front(f, true, u0, tspan, p, callback)
# neither specialized nor inferred on the function: only the function itself is compiled, at its signature
Base.@nospecializeinfer function _front(@nospecialize(f), dyn::Bool, u0::AbstractVector, tspan, p, @nospecialize(callback))
    u = _promote_state(_state(u0), p)
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    cbs = _callbacks(callback)
    # a record's functions at `Callable`'s default (open issue 159)
    f isa ODEFunction && return _problem(f.f, true, f.jac, u, ts, p, f.sys, f.mass_matrix, f.jac_prototype, cbs)
    # a function or a model: its properties as `ODEFunction(f)` reads them, without a record per function type
    return _problem(f, dyn, nothing, u, ts, p, _system(f), _mass_matrix(f), _jac_prototype(f), cbs)
end
"The callbacks given to a problem, as a vector."
Base.@nospecializeinfer _callbacks(@nospecialize(cb)) = cb === nothing ? Any[] : cb isa Union{Tuple,AbstractVector} ? Any[c for c in cb] : Any[cb]
# the functions partially evaluated on the parameters (W-63) at the problem's bounds, which have no parameter
# for the parameters' type, each with its attribute (`dyn`: `Dynamic`, else the function itself); the partially
# evaluated function beside them, for a solver that enters it at its own chunk
function _problem(@nospecialize(f), dyn::Bool, @nospecialize(jac), u0::Vector{T}, tspan::Tuple{Float64,Float64}, p::P,
                  @nospecialize(sys), @nospecialize(mm), @nospecialize(jp), cbs::Vector{Any}) where {T,P}
    # the Jacobian by duals: the one function, seen at the duals where it is needed, so a `Dynamic` one is kept at
    # both signatures (traits §7.7, §9)
    ad = T === Float64 && jac === nothing
    X = ad ? All{Callable{CoreRHSSig{T},Nothing}, Callable{CoreDualSig{DUAL_CHUNK},Nothing}} : Callable{CoreRHSSig{T},Nothing}
    r = _partial(f, p, dyn, X)
    jk = jac === nothing ? core_no_jacobian : _partial(jac, p, true, Callable{CoreJacSig{T},Nothing})
    # for a solver that picks its own dual chunk, the function at `Callable`'s default: a plain value
    return ODEProblem(r, u0, tspan, p, jk, ad, UserFunction(jac === nothing && dyn ? PartialFn(f, p) : nothing),
                      sys, _names(sys), mm, jp, cbs)
end
"""
A user's function partially evaluated on the parameters, with its attribute: kept `Dynamic` at the core's
trait `X` (a signature, or two: `All{…}`) where `dyn`, else the record itself, so the solver is compiled for it.
"""
Base.@nospecializeinfer _partial(@nospecialize(f), p, dyn::Bool, ::Type{X}) where {X} =
    dyn ? Dynamic{X}(PartialFn(f, p)) : PartialFn(f, p)

_state(u0::AbstractVector{<:ForwardDiff.Dual}) = Vector(u0)
_state(u0::AbstractVector{T}) where {T<:AbstractFloat} = Vector{T}(u0)      # Float32 states stay Float32 (row 6)
_state(u0::AbstractVector) = Vector{Float64}(u0)
"The state at the parameters' dual type, as SciML promotes `u0` for a gradient in `p`."
_promote_state(u0::Vector{Float64}, p::AbstractArray{D}) where {D<:ForwardDiff.Dual} = Vector{D}(u0)
_promote_state(u0::Vector, p) = u0


"""
    remake(prob::ODEProblem; u0 = prob.u0, tspan = prob.tspan, p = prob.p, callback)

SciML's `remake`: a problem like `prob` with the fields given changed. The user's functions are taken back from
the problem with their attributes (a compiled function stays compiled) and entered again at the new problem's
types, partially evaluated on the new parameters; the system object, the mass matrix, the Jacobian prototype and
(unless `callback` is given) the callbacks are kept. The state takes the parameters' dual type, as in
`ODEProblem`.
"""
Base.@nospecializeinfer function remake(prob::ODEProblem; @nospecialize(u0 = prob.u0), @nospecialize(tspan = prob.tspan),
                                        @nospecialize(p = prob.p), @nospecialize(callback = missing))
    # the user's functions back from their partially evaluated forms, with their attributes
    f, dyn = _user_function(getfield(prob, :f))   # the field as stored, not as a bound views it
    jac = _jac_given(prob) ? _user_function(getfield(prob, :jac))[1] : nothing
    cbs = callback === missing ? copy(prob.callbacks) : _callbacks(callback)
    u = _promote_state(_state(u0), p)
    return _problem(f, dyn, jac, u, (Float64(tspan[1]), Float64(tspan[2])), p, prob.sys, prob.mass_matrix, prob.jac_prototype, cbs)
end
"Whether a problem was given a Jacobian (its kept Jacobian is not the library's `core_no_jacobian`)."
_jac_given(::ODEProblem{T,P,F,J}) where {T,P,F,J} = J !== CoreNoJacobian
"The user's function back from its partially evaluated form, and whether it was kept `Dynamic`."
Base.@nospecializeinfer _user_function(@nospecialize(k)) = k isa Dynamic ? (unwrap(k).f, true) : (k.f, false)
