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
# The front end is typed: each method's result is a concrete `ODEProblem{T, P, F, J}` that its caller's code knows,
# so a program that makes a problem and solves it is fully typed (traits §14). What is compiled once is decided by
# the function's attribute, not by keeping the compiler from the code: under `Dynamic` the problem's type holds
# `Dynamic{Callable{…}}`, one type for every function; under `compiled_for(f)` it holds the function's type.

# the constructor with a trait argument (traits §8.2): a cast at this call decides the function's attribute,
# else `Callable`'s default; the signature's variables come from `u0` and `p`. Unbounded on purpose (§7.3): an
# adapter, it stores the function with its attribute
@unbounded function ODEProblem(f::Callable{RHSSig{P,T},Nothing}, u0::AbstractVector{T}, tspan, p::P; callback = nothing) where {T,P}
    return _problem_of(f, keep(f), _promote_state(_state(u0), p), _span(tspan), p, _callbacks(callback))
end
# without parameters: the function passed on with its attribute (traits §7.5)
ODEProblem(f::Callable{RHSSig{Nothing,T},Nothing}, u0::AbstractVector{T}, tspan; callback = nothing) where {T} =
    ODEProblem(f, u0, tspan, nothing; callback)
# a function that is no `Callable` at the state's type (its methods take other types): kept `Dynamic`, as by default.
# Not specialized on the function: the problem's type does not depend on it
ODEProblem(@nospecialize(f), u0::AbstractVector, tspan, p = nothing; callback = nothing) =
    _problem_of(f, Erased(f), _promote_state(_state(u0), p), _span(tspan), p, _callbacks(callback))
"A function to keep `Dynamic` whatever the call: one that is no `Callable` at the state's type."
struct Erased
    f::Any
    Erased(@nospecialize(f)) = new(f)
end
# a function or a model, with its attribute `k` (`keep`'s value): its properties are its own
_problem_of(f, k, u::Vector, ts::Tuple{Float64,Float64}, p, cbs::Vector{Any}) =
    _problem(_core_function(k, p, u, nothing), _plain_function(k, p), core_no_jacobian, u, ts, p,
             _system(f), _mass_matrix(f), _jac_prototype(f), cbs)
# SciML's record: its functions at `Callable`'s default, `Dynamic`, whatever the call (a cast on a function inside
# the record does not reach here: open issue 159); its properties are the record's
_problem_of(f::ODEFunction, k, u::Vector, ts::Tuple{Float64,Float64}, p, cbs::Vector{Any}) =
    _problem(_erased_function(f.f, p, u, f.jac), _plain_function(f.f, p, f.jac), _core_jacobian(f.jac, p, u), u, ts, p,
             f.sys, f.mass_matrix, f.jac_prototype, cbs)
_span(tspan) = (Float64(tspan[1]), Float64(tspan[2]))
"The callbacks given to a problem, as a vector."
_callbacks(::Nothing) = Any[]
_callbacks(cbs::Union{Tuple,AbstractVector}) = Any[c for c in cbs]
_callbacks(@nospecialize(cb)) = Any[cb]

# The core's trait for the right-hand side at a state type: with the Jacobian by duals (a `Float64` state, no
# Jacobian given) the one function is seen at the duals too, so a `Dynamic` one is kept at both signatures
# (traits §7.7, §9)
_core_trait(::Vector{Float64}, ::Nothing) = All{Callable{CoreRHSSig{Float64},Nothing}, Callable{CoreDualSig{DUAL_CHUNK},Nothing}}
_core_trait(::Vector{T}, jac) where {T} = Callable{CoreRHSSig{T},Nothing}
_by_duals(::Vector{Float64}, ::CoreNoJacobian) = true
_by_duals(::Vector, jk) = false
# The function partially evaluated on the parameters (W-63) at the problem's bound, which has no parameter for the
# parameters' type, with its attribute: a kept `Dynamic` is kept `Dynamic` at the core's trait, one type for every
# function; a function kept compiled is the record itself, so the solver is compiled for it
_core_function(k::Dynamic, p, u::Vector, jac) = _erased_function(unwrap(k), p, u, jac)
_core_function(k::Erased, p, u::Vector, jac) = _erased_function(k.f, p, u, jac)
_core_function(k, p, u::Vector, jac) = PartialFn(k, p)
_erased_function(@nospecialize(f), p, u::Vector, jac) = Dynamic{_core_trait(u, jac)}(PartialFn(f, p))
# for a solver that picks its own dual chunk, the function at `Callable`'s default as a plain value; `nothing`
# where the function is kept compiled or a Jacobian is given
_plain_function(k::Dynamic, p) = UserFunction(PartialFn(unwrap(k), p))
_plain_function(k::Erased, p) = UserFunction(PartialFn(k.f, p))
_plain_function(k, p) = UserFunction(nothing)
_plain_function(@nospecialize(f), p, ::Nothing) = UserFunction(PartialFn(f, p))
_plain_function(@nospecialize(f), p, jac) = UserFunction(nothing)
# a given Jacobian, entered `Dynamic` at the core's signature; without one, the library's `core_no_jacobian`
_core_jacobian(::Nothing, p, u::Vector) = core_no_jacobian
_core_jacobian(@nospecialize(jac), p, u::Vector{T}) where {T} = Dynamic{Callable{CoreJacSig{T},Nothing}}(PartialFn(jac, p))
# the problem, from its functions as it keeps them. The type's parameters are written out (traits §8.2): the
# constructor that infers them takes its functions at `Callable`'s default
_problem(r, fplain::UserFunction, jk, u0::Vector{T}, tspan::Tuple{Float64,Float64}, p::P,
         @nospecialize(sys), mm, @nospecialize(jp), cbs::Vector{Any}) where {T,P} =
    ODEProblem{T,P,typeof(r),typeof(jk)}(r, u0, tspan, p, jk, _by_duals(u0, jk), fplain, sys,
                                         _names(sys)::Union{Nothing,Dynamic{IndexProvider{Symbol}}}, mm, _is_identity(mm)::Bool, jp, cbs)

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
function remake(prob::ODEProblem; u0 = prob.u0, tspan = prob.tspan, p = prob.p, callback = missing)
    cbs = callback === missing ? copy(prob.callbacks) : _callbacks(callback)
    u = _promote_state(_state(u0), p)
    k = getfield(prob, :f); kj = getfield(prob, :jac)     # the fields as stored, not as a bound views them
    jac = _user_jacobian(kj)
    return _problem(_again(k, p, u, jac), _plain_again(k, p, jac), _core_jacobian(jac, p, u), u, _span(tspan), p,
                    prob.sys, prob.mass_matrix, prob.jac_prototype, cbs)
end
# the user's functions back from their partially evaluated forms, entered again with their attributes
_again(k::Dynamic, p, u::Vector, jac) = _erased_function(unwrap(k).f, p, u, jac)
_again(k::PartialFn, p, u::Vector, jac) = PartialFn(k.f, p)
_plain_again(k::Dynamic, p, jac) = _plain_function(unwrap(k).f, p, jac)
_plain_again(k::PartialFn, p, jac) = UserFunction(nothing)
_user_jacobian(::CoreNoJacobian) = nothing
_user_jacobian(kj::Dynamic) = unwrap(kj).f
"Whether a problem was given a Jacobian (its kept Jacobian is not the library's `core_no_jacobian`)."
_jac_given(::ODEProblem{T,P,F,J}) where {T,P,F,J} = J !== CoreNoJacobian
