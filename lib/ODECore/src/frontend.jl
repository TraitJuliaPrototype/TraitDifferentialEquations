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
types are known, in the form the record holds them: a plain function boxed, `Callable`'s default, a wrapper
kept (open issue 159). It is callable as `f` is (SciML's is), and carries its properties (it implements the
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
# a callable type that is no `Function` joins `Callable` by its declaration (traits §9): `ODEProblem` takes it at
# its `Callable` argument, as it takes a function
@implement Callable ODEFunction{F,TJ,TMM} where {F,TJ,TMM}
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
constructor, `Callable` at `(Vector{T}, Vector{T}, P, Float64) -> Nothing` (traits §8.2), whose default
erases: a plain function is boxed, `Dynamic{Callable{…}}`, so the solver's code is compiled once per state and
parameter type, never per function; `ODEProblem(compiled_for(f), …)` passes the wrapper, so the solver is
compiled for the function, with `f` inferred into it. The problem keeps it partially evaluated on the parameters,
in the form it was given. A Jacobian given in an `ODEFunction` is entered at `(Matrix{T}, Vector{T}, P, Float64) ->
Nothing` in the form the record holds it, and Rodas5 calls it instead of differentiating `f`.

The function's optional properties are read here, from the `ODEFunction` or from the function itself
(open issue 119): its mass matrix, system object and Jacobian prototype. The problem keeps them;
`solve` and `init` refuse a mass matrix other than the identity for an algorithm that takes none. The
callbacks given (one, a tuple or a vector) are entered at the problem's types and kept (`enter_callbacks`):
`solve` and `init` take them before their own.
"""
# The front end is typed: each method's result is a concrete `ODEProblem{T, P, F, J}` that its caller's code knows,
# so a program that makes a problem and solves it is fully typed (traits §14). What is compiled once is decided by
# the value: a box, `Dynamic{Callable{…}}`, one type for every function; a wrapper, `Compiled{…}`, the function's type.

# the constructor with a trait argument (traits §8.2): the definition makes the method at the kept forms and the
# forwarder that boxes a plain function (W-160); the signature's variables come from `u0` and `p`. Unbounded on purpose (§7.3):
# an adapter, it stores the function in the form it is given
@unbounded function ODEProblem(f::Callable{RHSSig{P,T},Nothing}, u0::AbstractVector{T}, tspan, p::P; callback = nothing) where {T,P}
    u = _promote_state(_state(u0), p)
    return _problem_of(f, unwrap(f), u, _span(tspan), p, enter_callbacks(callback, typeof(u), typeof(p)))
end
# without parameters: the kept form passed on
ODEProblem(f::Callable{RHSSig{Nothing,T},Nothing}, u0::AbstractVector{T}, tspan; callback = nothing) where {T} =
    ODEProblem(f, u0, tspan, nothing; callback)
"The object a kept form holds, or the value itself."
_object(@nospecialize(f)) = f
_object(f::Union{Dynamic,Compiled}) = unwrap(f)
# a function or a model in the form `k` it was given (a box, a wrapper): its properties are the object's own
_problem_of(k, f, u::Vector, ts::Tuple{Float64,Float64}, p, cbs::EnteredCallbacks) =
    _problem(_core_function(k, p, u, nothing), _plain_function(k, p), core_no_jacobian, u, ts, p,
             _system(f), _mass_matrix(f), _jac_prototype(f), cbs)
# SciML's record (a `Callable` by its declaration), boxed or wrapped as a whole: its functions in the forms the record holds
# them (open issue 159: `ODEFunction(f; jac = compiled_for(j))` keeps `j` compiled); its properties are the record's
_problem_of(k, f::ODEFunction, u::Vector, ts::Tuple{Float64,Float64}, p, cbs::EnteredCallbacks) =
    _problem(_core_function(f.f, p, u, f.jac), _plain_function(f.f, p, f.jac), _core_jacobian(f.jac, p, u), u, ts, p,
             f.sys, f.mass_matrix, f.jac_prototype, cbs)
_span(tspan) = (Float64(tspan[1]), Float64(tspan[2]))

# The core's trait for the right-hand side at a state type: with the Jacobian by duals (a `Float64` state, no
# Jacobian given) the one function is seen at the duals too, so a box is made at both signatures (traits §7.7, §9)
_core_trait(::Vector{Float64}, ::Nothing) = All{Callable{CoreRHSSig{Float64},Nothing}, Callable{CoreDualSig{DUAL_CHUNK},Nothing}}
_core_trait(::Vector{T}, jac) where {T} = Callable{CoreRHSSig{T},Nothing}
_by_duals(::Vector{Float64}, ::CoreNoJacobian) = true
_by_duals(::Vector, jk) = false
# The function partially evaluated on the parameters (W-63) at the problem's bound, which has no parameter for the
# parameters' type, in the form it was given: a box is boxed again at the core's trait, one type for every function;
# a wrapper wraps the partially evaluated function, so the solver is compiled for it; a plain function (a record's)
# is boxed
_core_function(k::Compiled, p, u::Vector, jac) = Compiled(PartialFn(unwrap(k), p))
_core_function(@nospecialize(f), p, u::Vector, jac) = _erased_function(_object(f), p, u, jac)
_erased_function(@nospecialize(f), p, u::Vector, jac) = Dynamic{_core_trait(u, jac)}(PartialFn(f, p))
# for a solver that picks its own dual chunk, the function as a plain value; `nothing` where the function is
# compiled or a Jacobian is given
_plain_function(k::Compiled, p) = UserFunction(nothing)
_plain_function(@nospecialize(f), p) = UserFunction(PartialFn(_object(f), p))
_plain_function(@nospecialize(f), p, ::Nothing) = _plain_function(f, p)
_plain_function(@nospecialize(f), p, jac) = UserFunction(nothing)
# a given Jacobian at the core's signature, in its form; without one, the library's `core_no_jacobian`
_core_jacobian(::Nothing, p, u::Vector) = core_no_jacobian
_core_jacobian(jac::Compiled, p, u::Vector) = Compiled(PartialFn(unwrap(jac), p))
_core_jacobian(@nospecialize(jac), p, u::Vector{T}) where {T} = Dynamic{Callable{CoreJacSig{T},Nothing}}(PartialFn(_object(jac), p))
# the problem, from its functions as it keeps them. The type's parameters are written out (traits §8.2): the
# constructor that infers them takes its functions at `Callable`'s default
_problem(r, fplain::UserFunction, jk, u0::Vector{T}, tspan::Tuple{Float64,Float64}, p::P,
         @nospecialize(sys), mm, @nospecialize(jp), cbs::EnteredCallbacks{Vector{T},Float64,P}) where {T,P} =
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
the problem in the forms it keeps them (a wrapped function stays wrapped) and entered again at the new problem's
types, partially evaluated on the new parameters; the system object, the mass matrix, the Jacobian prototype and
(unless `callback` is given) the callbacks are kept. The state takes the parameters' dual type, as in
`ODEProblem`.
"""
function remake(prob::ODEProblem; u0 = prob.u0, tspan = prob.tspan, p = prob.p, callback = missing)
    u = _promote_state(_state(u0), p)
    cbs = enter_callbacks(callback === missing ? prob.callbacks : callback, typeof(u), typeof(p))
    k = getfield(prob, :f); kj = getfield(prob, :jac)     # the fields as stored, not as a bound views them
    jac = _user_jacobian(kj)
    return _problem(_again(k, p, u, jac), _plain_again(k, p, jac), _core_jacobian(jac, p, u), u, _span(tspan), p,
                    prob.sys, prob.mass_matrix, prob.jac_prototype, cbs)
end
# the user's functions back from their partially evaluated forms, entered again in their forms
_again(k::Dynamic, p, u::Vector, jac) = _erased_function(unwrap(k).f, p, u, jac)
_again(k::Compiled, p, u::Vector, jac) = Compiled(PartialFn(unwrap(k).f, p))
_plain_again(k::Dynamic, p, jac) = _plain_function(unwrap(k).f, p, jac)
_plain_again(k::Compiled, p, jac) = UserFunction(nothing)
_user_jacobian(::CoreNoJacobian) = nothing
_user_jacobian(kj::Dynamic) = unwrap(kj).f
_user_jacobian(kj::Compiled) = Compiled(unwrap(kj).f)
"Whether a problem was given a Jacobian (its kept Jacobian is not the library's `core_no_jacobian`)."
_jac_given(::ODEProblem{T,P,F,J}) where {T,P,F,J} = J !== CoreNoJacobian
