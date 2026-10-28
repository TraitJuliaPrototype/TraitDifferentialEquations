# ODECore: the trait module. A problem, callbacks and the integrator as callbacks see it are
# traits; the solvers' code is compiled for what these bounds can see (note "Keys"):
#   - a problem type that implements `rhs!` itself is the receiver type, so the solver compiles per
#     problem type;
#   - TDE's own `ODEProblem` keeps its function at parameters bounded by `Callable`: a plain function
#     as a `Dynamic{Callable{…}}` (`Callable`'s default), so the solver compiles once per element and
#     parameter type, whatever the function; under `compiled_for(f)` the function itself, and the
#     solver compiles per function;
#   - callbacks are entered at the problem's types at a solver's `init` and kept with their attribute
#     (`Vector{Callback{U, T, P}}`); they see the integrator as a `Dynamic{IntegratorView{U, T, P}}`,
#     so a new callback compiles nothing in the solver unless the caller casts it with `compiled_for`,
#     and a callback reads `integ.p` at the problem's parameter type;
#   - a solution is a `Trajectory{U, T}`: code over solutions is compiled per state and time type,
#     whatever the model;
#   - the parameters' type stays in the outermost layer (W-63): `solve_ode` partially evaluates the
#     problem's functions on its parameters (`CoreProblem`) and adapts the callbacks (`ParamCallback`),
#     so the solvers' code is compiled per state type, whatever the parameters.
# (the body of the trait module `ODECore`, src/ODECore.jl)
using TraitWorld
using LinearAlgebra: lu!, ldiv!
import LinearAlgebra
import ForwardDiff
export ODEProblemTrait, rhs!, jacobian!, rhs_function, Trajectory, Callback, condition, affect!, IntegratorView, Tsit5, ODESolution, SolverOptions,
       solve_ode, ODEProblem, FunctionCallback, RHSSig, DualSig, JacSig, DUAL_CHUNK, CondSig, AffectSig, FDual, ODETag,
       no_jacobian, UserFunction, add_tstop!, terminate!, interp!, EventCallback, event_condition, event_affect!,
       DiscreteCallback, ContinuousCallback, FunctionEventCallback, enter_callbacks,
       CoreProblem, core_problem, adapt_callbacks, ODEAlgorithm, solve_core, LinearSolver, LinearCache, lincache, linfactor!, linsolve!, DenseLU, StepView, StepCallback, StepEvent, ParamIntegrator,
       ODEFunction

# --- the traits --------------------------------------------------------------------------

"""
What the solver needs from a problem: its initial state, time span and parameters, the right-hand
side in place at element type `T`, and its Jacobian in `u` (for implicit methods). The default
Jacobian is by finite differences of `rhs!`; a problem that can do better implements it (TDE's
`ODEProblem` does: the given Jacobian, else ForwardDiff's duals). `p` is what callbacks see as
`integ.p`.

`rhs_function(prob)` is the user's right-hand side partially evaluated on the parameters, `(du, u, t)`,
as a plain value (W-63), for a solver that enters it at a signature of its own choosing (Rodas5 at the duals of a chunk picked from the
state's length, at its `init`); `nothing` (the default) when the problem has no such function or
brings its own Jacobian, and the solver calls `jacobian!`.
"""
@trait ODEProblemTrait{T, P} begin
    @const u0::Vector{T}
    @const tspan::Tuple{Float64, Float64}
    @const p::P
    rhs!(prob::ODEProblemTrait{T, P}, du::Vector{T}, u::Vector{T}, t::Float64)::Nothing
    jacobian!(prob::ODEProblemTrait{T, P}, J::Matrix{T}, u::Vector{T}, t::Float64)::Nothing = fd_jacobian!(prob, J, u, t)
    rhs_function(prob::ODEProblemTrait{T, P})::Any = nothing
end

"The default Jacobian: forward differences of `rhs!` (an internal function: `prob` is a trait object, a problem itself, or the core's `CoreProblem`)."
function fd_jacobian!(prob, J::Matrix{T}, u::Vector{T}, t::Float64) where {T}
    n = length(u)
    f0 = zeros(T, n); f1 = zeros(T, n); up = copy(u)
    _rhs!(prob, f0, u, t)
    for j in 1:n
        h = sqrt(eps(Float64)) * max(abs(primal(u[j])), 1.0)
        up[j] = u[j] + h
        _rhs!(prob, f1, up, t)
        for i in 1:n
            J[i, j] = (f1[i] - f0[i]) / h
        end
        up[j] = u[j]
    end
    return nothing
end

"""
    IntegratorView{U, T, P}

What a callback sees of the integrator: the state `u::U` (its contents writable), the time `t::T`,
the last step's size `dt::T`, the parameters `p::P`, two flags, and three operations: `add_tstop!`
(a time the solver must step to), `terminate!`, and `interp!(i, out, τ)`, the state at `τ` within
the last accepted step by the solver's interpolant. `U`, `T` and `P` are defined by the fields, so
a callback reads `integ.p` at the problem's parameter type. Each solver's integrator conforms as a
family, by its own methods.
"""
@trait IntegratorView{U, T, P} begin
    @const u::U
    @const t::T
    @const dt::T
    @const p::P
    u_modified::Bool
    terminated::Bool
    add_tstop!(i::IntegratorView{U, T, P}, τ::T)::Nothing
    terminate!(i::IntegratorView{U, T, P})::Nothing
    interp!(i::IntegratorView{U, T, P}, out::U, τ::T)::Nothing
end

"""
    Callback{U, T, P}

A discrete callback, checked after every accepted step: `condition` decides, `affect!` acts on the
integrator, which it sees as a `Dynamic{IntegratorView{U, T, P}}`. The integrator keeps its callbacks
as `Vector{Callback{U, T, P}}`, so their types reach neither the integrator's type nor the driver.
A callback type implements it at the problem's types: `@implement Callback{Vector{Float64}, Float64, P} MyCallback`.
"""
@trait Callback{U, T, P} begin
    condition(cb::Callback{U, T, P}, u::U, t::T, integ::Dynamic{IntegratorView{U, T, P}})::Bool
    affect!(cb::Callback{U, T, P}, integ::Dynamic{IntegratorView{U, T, P}})::Nothing
end

"""
    EventCallback{U, T, P}

A continuous callback: an event where the scalar `event_condition` changes sign within a step; the
solver locates it by bisection on the interpolant (`interp!`), moves the integrator to it, and calls
`event_affect!`.
"""
@trait EventCallback{U, T, P} begin
    event_condition(cb::EventCallback{U, T, P}, u::U, t::T, integ::Dynamic{IntegratorView{U, T, P}})::Float64
    event_affect!(cb::EventCallback{U, T, P}, integ::Dynamic{IntegratorView{U, T, P}})::Nothing
end

"""
    StepView{U, T}

The integrator as the solver core's callbacks see it: `IntegratorView` without the parameters, so
that the core's code is compiled per state type, whatever the parameters (W-63). The outer layer's
`ParamIntegrator` gives the user's callbacks an `IntegratorView{U, T, P}` over it.
"""
@trait StepView{U, T} begin
    @const u::U
    @const t::T
    @const dt::T
    u_modified::Bool
    terminated::Bool
    step_tstop!(i::StepView{U, T}, τ::T)::Nothing
    step_terminate!(i::StepView{U, T})::Nothing
    step_interp!(i::StepView{U, T}, out::U, τ::T)::Nothing
end

"A discrete callback as the solver core calls it: over `StepView{U, T}`, without the parameters."
@trait StepCallback{U, T} begin
    step_condition(cb::StepCallback{U, T}, u::U, t::T, sv::Dynamic{StepView{U, T}})::Bool
    step_affect!(cb::StepCallback{U, T}, sv::Dynamic{StepView{U, T}})::Nothing
end

"A continuous callback as the solver core calls it."
@trait StepEvent{U, T} begin
    step_event_condition(cb::StepEvent{U, T}, u::U, t::T, sv::Dynamic{StepView{U, T}})::Float64
    step_event_affect!(cb::StepEvent{U, T}, sv::Dynamic{StepView{U, T}})::Nothing
end

# --- the signatures of plain functions --------------------------------------------------------

"The signature of a plain right-hand side `f(du, u, p, t)`, at element type `T` (`Float64`, or dual numbers for gradients through `solve`)."
const RHSSig{P,T} = Tuple{Vector{T},Vector{T},P,Float64}
"The signature of a plain Jacobian `jac(J, u, p, t)` (SciML's `ODEFunction(f; jac)`), at element type `T`."
const JacSig{P,T} = Tuple{Matrix{T},Vector{T},P,Float64}
"The signature of a plain callback condition `condition(u, t, integ)`, at the problem's types."
const CondSig{U,T,P} = Tuple{U,T,Dynamic{IntegratorView{U,T,P}}}
"The signature of a plain `affect!(integ)`."
const AffectSig{U,T,P} = Tuple{Dynamic{IntegratorView{U,T,P}}}

# --- ForwardDiff's dual numbers, tagged by the library ----------------------------------------
# (wrapping by facts: the trait module calls ForwardDiff directly; session 23 used its own duals)

"The library's tag for ForwardDiff's duals: they are the library's, whatever the model function."
struct ODETag end
"ForwardDiff's dual number with the library's tag and chunk size `N`."
const FDual{N} = ForwardDiff.Dual{ODETag,Float64,N}

"The signature of a plain right-hand side at the library's duals."
const DualSig{P,N} = Tuple{Vector{FDual{N}},Vector{FDual{N}},P,Float64}

"The chunk of the Jacobian's dual numbers: fixed, so that it is no part of a problem's type (a last chunk may be partial)."
const DUAL_CHUNK = 8

# --- the solver core: the functions partially evaluated on the parameters (W-63) ------------------

"The right-hand side as the solver core calls it, partially evaluated on the parameters: `f(du, u, t)`, at element type `T`."
const CoreRHSSig{T} = Tuple{Vector{T},Vector{T},Float64}
"The Jacobian as the solver core calls it: `jac(J, u, t)`."
const CoreJacSig{T} = Tuple{Matrix{T},Vector{T},Float64}
"The right-hand side at the library's duals of chunk `N`, as the solver core calls it."
const CoreDualSig{N} = Tuple{Vector{FDual{N}},Vector{FDual{N}},Float64}

"""
    PartialFn(f, p)

A function of `(a, u, p, t)` partially evaluated on the parameters: `(a, u, t) -> f(a, u, p, t)`. The
parameters' type stays in this record, which the solver core keeps at a signature without it.
"""
struct PartialFn{F,P}
    f::F
    p::P
end
(r::PartialFn)(a, u, t) = r.f(a, u, r.p, t)
"A problem type's own `rhs!`, partially evaluated on the problem (a problem that implements `ODEProblemTrait` itself)."
struct ProblemRHS{M}
    m::M
end
(r::ProblemRHS)(du, u, t) = rhs!(r.m, du, u, t)
"A problem type's own `jacobian!` (its default: finite differences of its `rhs!`), partially evaluated on the problem."
struct ProblemJac{M}
    m::M
end
(r::ProblemJac)(J, u, t) = jacobian!(r.m, J, u, t)
"No Jacobian given at the core: by the duals, else by finite differences (never called)."
core_no_jacobian(J, u, t) = nothing
const CoreNoJacobian = typeof(core_no_jacobian)

"The user's function as given, its type hidden: what a problem keeps beside the kept forms."
struct UserFunction
    f::Any
    UserFunction(@nospecialize(f)) = new(f)
end

"""
    ODEProblem{T, P, F, D, J}

TDE's own problem, the canonical implementation of `ODEProblemTrait`: a plain right-hand side
`f(du, u, p, t)`, an initial state, a time span, parameters, and optionally a Jacobian
`jac(J, u, p, t)` (SciML's split: the user's functions come in a plain record, `ODEFunction`, which
has no parameter for `u`'s or `p`'s type; the problem has them). The problem keeps each function
partially evaluated on its parameters (`PartialFn(f, p)`, W-63), at a parameter bounded by `Callable`
at a signature without the parameters (traits reference §7.4), as its bound gives it: the record
itself under `compiled_for(f)`, so the solver compiles per function; a `Dynamic{Callable{…}}`
otherwise (`Callable`'s default, and `with_dynamic`), so the solver compiles once per state type,
whatever the function and the parameters:
- `f::F` at `CoreRHSSig{T}`;
- `ad`: whether the Jacobian is by dual numbers (a `Float64` state, no Jacobian given): then `f` is seen
  at the library's duals where the Jacobian needs it (traits reference §5.8, §6.5);
- `jac::J` at `CoreJacSig{T}`, the given Jacobian; without one, the library's `core_no_jacobian`,
  kept compiled, so `jacobian!` folds on `J`.
- `fplain`, the partially evaluated function as a plain value, beside the kept ones (`rhs_function`):
  a solver that picks its own dual chunk enters it at its `init` (Rodas5's `chunk_size`). Held in a
  `UserFunction`, so that no constructor is specialized on the function's type.
`p` is what callbacks see as `integ.p`, the same object the kept functions hold. Made by the front end
`ODEProblem(f, u0, tspan, p)`.
"""
struct ODEProblem{T,P,F<:Callable{CoreRHSSig{T},Nothing},J<:Callable{CoreJacSig{T},Nothing}}
    f::F
    u0::Vector{T}
    tspan::Tuple{Float64,Float64}
    p::P
    jac::J
    ad::Bool
    fplain::UserFunction
end
"No Jacobian given (the four-argument form, for a user's `ODEFunction`; never called)."
no_jacobian(J, u, p, t) = nothing
rhs!(prob::ODEProblem{T}, du::Vector{T}, u::Vector{T}, t::Float64) where {T} = (prob.f(du, u, t)::Nothing; nothing)
"The Jacobian: the given one; else by ForwardDiff's duals for a `Float64` state, by finite differences otherwise."
jacobian!(prob::ODEProblem{T,P,F,J}, Jm::Matrix{T}, u::Vector{T}, t::Float64) where {T,P,F,J} =
    J === CoreNoJacobian ? (prob.ad ? _ad_jacobian!(prob.f, Jm, u, t, Val(DUAL_CHUNK)) : fd_jacobian!(prob, Jm, u, t)) :
    (prob.jac(Jm, u, t)::Nothing; nothing)
"The user's function partially evaluated, for a solver that picks its own dual chunk; `nothing` when a Jacobian was given."
rhs_function(prob::ODEProblem) = prob.fplain.f
"The Jacobian by duals: one call of the right-hand side, seen at the library's duals of chunk `N`, per chunk of `N` columns."
function _ad_jacobian!(fd::Callable{CoreDualSig{N},Nothing}, J::Matrix{Float64}, u::Vector{Float64}, t::Float64, ::Val{N}) where {N}
    n = length(u)
    ud = Vector{FDual{N}}(undef, n)
    dd = Vector{FDual{N}}(undef, n)
    for c0 in 1:N:n
        for j in 1:n
            ud[j] = FDual{N}(u[j], ForwardDiff.Partials(ntuple(k -> j == c0 + k - 1 ? 1.0 : 0.0, Val(N))))
        end
        fd(dd, ud, t)::Nothing
        for k in 1:N
            col = c0 + k - 1
            col > n && break
            for i in 1:n
                J[i, col] = ForwardDiff.partials(dd[i], k)
            end
        end
    end
    return nothing
end
@implement ODEProblemTrait{T,P} ODEProblem{T,P,F,J} where {T,P,F,J}

"""
    CoreProblem{T, F, D, J}

The problem as the solver core sees it (W-63): the initial state, the time span, and the functions
partially evaluated on the parameters, each kept at a parameter bounded by `Callable` at a signature
without them (traits reference §7.4):
- `f::F` at `CoreRHSSig{T}`;
- `ad`: whether the Jacobian is by duals, `f` seen there;
- `jac::J` at `CoreJacSig{T}` (`core_no_jacobian`: by the duals, else by finite differences);
Its type names no parameter type, so the solvers' code is compiled per state type. Made by
`core_problem`, the outer layer.
"""
struct CoreProblem{T,F<:Callable{CoreRHSSig{T},Nothing},J<:Callable{CoreJacSig{T},Nothing}}
    f::F
    u0::Vector{T}
    tspan::Tuple{Float64,Float64}
    jac::J
    ad::Bool
end
core_rhs!(m::CoreProblem{T}, du::Vector{T}, u::Vector{T}, t::Float64) where {T} = (m.f(du, u, t)::Nothing; nothing)
"The core's Jacobian: the given one; else by the duals for a `Float64` state; else by finite differences."
core_jacobian!(m::CoreProblem{T,F,J}, Jm::Matrix{T}, u::Vector{T}, t::Float64) where {T,F,J} =
    J === CoreNoJacobian ? (m.ad ? _ad_jacobian!(m.f, Jm, u, t, Val(DUAL_CHUNK)) : fd_jacobian!(m, Jm, u, t)) :
    (m.jac(Jm, u, t)::Nothing; nothing)
"The problem's right-hand side, whatever the problem: its `rhs!`, or the core's partially evaluated function."
_rhs!(prob, du, u, t) = rhs!(prob, du, u, t)
_rhs!(m::CoreProblem, du, u, t) = core_rhs!(m, du, u, t)
"""
    core_problem(prob) -> CoreProblem

The outer layer's step (W-63): the problem's functions partially evaluated on its parameters. TDE's
`ODEProblem` keeps them so already (`PartialFn`): they pass as they are kept. For a problem type that
implements `ODEProblemTrait` itself, its `rhs!` and `jacobian!` partially evaluated on the problem,
`Dynamic`.
"""
function core_problem(prob::ODEProblem{T,P,F,J}) where {T,P,F,J}
    # a compiled function: the problem type is a member of the problem family, checked at its first use (traits §4.7)
    F <: Dynamic || _member_checked(prob)
    return CoreProblem(_kept(prob.f), prob.u0, prob.tspan, _kept(prob.jac), prob.ad)
end
"A kept function passed on with its attribute: a `Dynamic` value as it is, a compiled one cast again."
_kept(f::Dynamic) = f
_kept(f) = compiled_for(f)
"Nothing: the bound's first-use check of the problem type, as a member of the problem family."
_member_checked(prob::ODEProblemTrait) = nothing
# a bounded function: `prob` is a trait object, so the trait's defaults serve what the type does not implement
core_problem(prob::ODEProblemTrait) =
    CoreProblem(ProblemRHS(prob), prob.u0, prob.tspan, ProblemJac(prob), false)

"""
    DiscreteCallback(condition, affect!)
    ContinuousCallback(condition, affect!)

The user's callback from two plain functions (SciML's): `condition(u, t, integ)::Bool` (discrete) or
`::Float64` (continuous: an event where it changes sign), and `affect!(integ)`. Plain records: they
keep the functions as given, before the problem's types are known; a solver's `init` enters them at
`(U, T, P)` (`enter_callbacks`).
"""
struct DiscreteCallback
    condition::Any
    affect!::Any
    Base.@nospecializeinfer DiscreteCallback(@nospecialize(c), @nospecialize(a)) = new(c, a)
end
struct ContinuousCallback
    condition::Any
    affect!::Any
    Base.@nospecializeinfer ContinuousCallback(@nospecialize(c), @nospecialize(a)) = new(c, a)
end

"A discrete callback as the integrator keeps it: two plain functions entered at the problem's types, each kept with its attribute (`Kept{Callable{…}}`)."
struct FunctionCallback{U,T,P}
    cond::Callable{CondSig{U,T,P},Bool}
    aff::Callable{AffectSig{U,T,P},Nothing}
end
condition(cb::FunctionCallback{U,T,P}, u::U, t::T, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = cb.cond(u, t, integ)
affect!(cb::FunctionCallback{U,T,P}, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.aff(integ); nothing)
@implement Callback{U,T,P} FunctionCallback{U,T,P} where {U,T,P}

"A continuous callback as the integrator keeps it."
struct FunctionEventCallback{U,T,P}
    cond::Callable{CondSig{U,T,P},Float64}
    aff::Callable{AffectSig{U,T,P},Nothing}
end
event_condition(cb::FunctionEventCallback{U,T,P}, u::U, t::T, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = cb.cond(u, t, integ)
event_affect!(cb::FunctionEventCallback{U,T,P}, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.aff(integ); nothing)
@implement EventCallback{U,T,P} FunctionEventCallback{U,T,P} where {U,T,P}

"A plain function kept at a signature (`K`, a `Kept{Callable{…}}`), neither specialized nor inferred on it: entering a new function compiles none of the library's code."
Base.@nospecializeinfer _keep_fn(::Type{K}, @nospecialize(f)) where {K} = K(f)::K

"""
    enter_callbacks(cbs, U, P) -> (discrete, continuous)

A solver's `init` step for callbacks: the user's callbacks entered at the problem's types `(U, Float64, P)`.
A `DiscreteCallback` or `ContinuousCallback` becomes a record of its functions kept at the signatures
(`FunctionCallback`, `FunctionEventCallback`); a value of a callback type is kept as it is, refused
if it does not conform at these types. The integrator keeps the two vectors.
"""
function enter_callbacks(cbs::Vector{Any}, ::Type{U}, ::Type{P}) where {U,P}
    T = Float64
    d = Kept{Callback{U,T,P}}[]                  # `Callback{U,T,P}[]` is not taken as `Kept{…}` here (E-158 finding)
    c = Kept{EventCallback{U,T,P}}[]
    for cb in cbs
        if cb isa DiscreteCallback
            push!(d, FunctionCallback{U,T,P}(_keep_fn(Kept{Callable{CondSig{U,T,P},Bool}}, cb.condition),
                                             _keep_fn(Kept{Callable{AffectSig{U,T,P},Nothing}}, cb.affect!)))
        elseif cb isa ContinuousCallback
            push!(c, FunctionEventCallback{U,T,P}(_keep_fn(Kept{Callable{CondSig{U,T,P},Float64}}, cb.condition),
                                                  _keep_fn(Kept{Callable{AffectSig{U,T,P},Nothing}}, cb.affect!)))
        elseif _is_event_callback(cb, EventCallback{U,T,P})
            push!(c, _keep_fn(Kept{EventCallback{U,T,P}}, cb))
        else
            push!(d, _keep_fn(Kept{Callback{U,T,P}}, cb))
        end
    end
    return d, c
end
"Whether a callback value is a continuous one: a kept value of `EventCallback`, a cast of one, or a value of a type that conforms."
Base.@nospecializeinfer _is_event_callback(@nospecialize(cb), @nospecialize(E)) =
    cb isa Kept{E} || cb isa Dynamic{E} || cb isa Compiled{E} || conforms(typeof(_uncast(cb)), E)
Base.@nospecializeinfer _uncast(@nospecialize(cb)) =
    cb isa TraitWorld.CompiledFor ? getfield(cb, :x) : cb isa TraitWorld.WithDynamic ? getfield(cb, :x) : cb

"""
    ParamIntegrator{U, T, P}

The integrator as the user's callbacks see it (`IntegratorView{U, T, P}`), over the solver core's
`StepView{U, T}` (W-63): the core's state (the same array), copies of its time, step and flags, and
the parameters. The adapters copy the time, step and flags in before each of the user's functions
and the flags back after it (a declared field is a stored field, traits reference §3.3); the
operations go to the core's view. One per solve, shared by its callbacks.
"""
mutable struct ParamIntegrator{U,T,P}
    u::U
    t::T
    dt::T
    const p::P
    u_modified::Bool
    terminated::Bool
    sv::Union{Nothing,Dynamic{StepView{U,T}}}
end
add_tstop!(i::ParamIntegrator{U,T}, τ::T) where {U,T} = (step_tstop!(i.sv::Dynamic{StepView{U,T}}, τ); nothing)
terminate!(i::ParamIntegrator) = (i.terminated = true; nothing)
interp!(i::ParamIntegrator{U,T}, out::U, τ::T) where {U,T} = (step_interp!(i.sv::Dynamic{StepView{U,T}}, out, τ); nothing)
@implement IntegratorView ParamIntegrator{U,T,P} where {U,T,P}

function _sync_in!(pi::ParamIntegrator{U,T}, sv::Dynamic{StepView{U,T}}) where {U,T}
    pi.sv = sv
    pi.u = sv.u; pi.t = sv.t; pi.dt = sv.dt
    pi.u_modified = sv.u_modified; pi.terminated = sv.terminated
    return nothing
end
_sync_out!(pi::ParamIntegrator{U,T}, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (sv.u_modified = pi.u_modified; sv.terminated = pi.terminated; nothing)

"A user's discrete callback, kept at the problem's types, adapted to the solver core (W-63): the outer layer's, compiled per parameter type."
struct ParamCallback{U,T,P}
    cb::Kept{Callback{U,T,P}}
    pi::ParamIntegrator{U,T,P}
    view::Dynamic{IntegratorView{U,T,P}}
end
step_condition(c::ParamCallback{U,T}, u::U, t::T, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); condition(c.cb, u, t, c.view))
step_affect!(c::ParamCallback{U,T}, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); affect!(c.cb, c.view); _sync_out!(c.pi, sv); nothing)
@implement StepCallback{U,T} ParamCallback{U,T,P} where {U,T,P}

"A user's continuous callback, adapted to the solver core."
struct ParamEvent{U,T,P}
    cb::Kept{EventCallback{U,T,P}}
    pi::ParamIntegrator{U,T,P}
    view::Dynamic{IntegratorView{U,T,P}}
end
step_event_condition(c::ParamEvent{U,T}, u::U, t::T, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); event_condition(c.cb, u, t, c.view))
step_event_affect!(c::ParamEvent{U,T}, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); event_affect!(c.cb, c.view); _sync_out!(c.pi, sv); nothing)
@implement StepEvent{U,T} ParamEvent{U,T,P} where {U,T,P}

"""
    adapt_callbacks(cbs, U, p) -> Vector{Any}

The outer layer's step for callbacks (W-63): the user's callbacks entered at the problem's types
`(U, Float64, P)` (`enter_callbacks`), each adapted to the solver core, which sees them as
`StepCallback{U, Float64}` and `StepEvent{U, Float64}`, without the parameters.
"""
function adapt_callbacks(cbs::Vector{Any}, ::Type{U}, p::P) where {U,P}
    T = Float64
    d, c = enter_callbacks(cbs, U, P)
    pi = ParamIntegrator{U,T,P}(U(), 0.0, 0.0, p, false, false, nothing)
    view = Dynamic{IntegratorView{U,T,P}}(pi)
    out = Any[]
    for k in d; push!(out, ParamCallback{U,T,P}(k, pi, view)); end
    for k in c; push!(out, ParamEvent{U,T,P}(k, pi, view)); end
    return out
end

"""
The solver core's `init` step for callbacks: values of `StepCallback{U, Float64}` and
`StepEvent{U, Float64}` (the adapted user's callbacks), kept in two vectors.
"""
function enter_step_callbacks(cbs::Vector{Any}, ::Type{U}) where {U}
    T = Float64
    d = Kept{StepCallback{U,T}}[]
    c = Kept{StepEvent{U,T}}[]
    for cb in cbs
        if _is_event_callback(cb, StepEvent{U,T})
            push!(c, _keep_fn(Kept{StepEvent{U,T}}, cb))
        else
            push!(d, _keep_fn(Kept{StepCallback{U,T}}, cb))
        end
    end
    return d, c
end

# --- options, solution, the integrator ---------------------------------------------------------

struct Tsit5 end

struct SolverOptions
    abstol::Float64
    reltol::Float64
    dtmax::Float64
    maxiters::Int
    saveat::Vector{Float64}          # empty: save every step
end

struct ODESolution{T}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    retcode::Symbol
    naccept::Int
    nreject::Int
    nf::Int
end

"""
    Trajectory{U, T}

A solution as code over solutions sees it (plotting, sampling, events after the fact): the saved
times, the state at the `k`-th of them, and the state at any time in the span (`at`). The key is the
state type `U` and the time type `T`, never the model: TDE's `ODESolution{T}` is a
`Trajectory{Vector{T}, Float64}` for every problem.
"""
@trait Trajectory{U, T} begin
    times(s::Trajectory{U, T})::Vector{T}
    state(s::Trajectory{U, T}, k::Int)::U
    at(s::Trajectory{U, T}, τ::T)::U
end
times(s::ODESolution) = s.t
state(s::ODESolution{T}, k::Int) where {T} = s.u[k]
"The state at `τ`, linear between the saved states (the solution keeps no interpolant)."
function at(s::ODESolution{T}, τ::Float64) where {T}
    ts = s.t
    (isempty(ts) || τ < ts[1] || τ > ts[end]) && throw(DomainError(τ, "outside the saved times $(isempty(ts) ? "(none)" : (ts[1], ts[end]))"))
    k = searchsortedlast(ts, τ)
    k == length(ts) && return copy(s.u[k])
    θ = T((τ - ts[k]) / (ts[k+1] - ts[k]))            # at the state's element type: a Float32 state stays Float32
    return (1 - θ) .* s.u[k] .+ θ .* s.u[k+1]
end
@implement Trajectory{Vector{T},Float64} ODESolution{T} where {T}

"""
An ODE solver: SciML's algorithms, an open set (move 3). A solver package implements it for its own
algorithm type. `solve_core(a, m, callbacks, opts)` is the solver's driver on the core's problem `m`
(its parameters partially evaluated, W-63), with the core's step callbacks.
"""
@trait ODEAlgorithm begin
    solve_core(a::ODEAlgorithm, m::CoreProblem, callbacks::Vector{Any}, opts::SolverOptions)::ODESolution
end

"""
A linear solver's state for the systems `W x = b` of size `n` an implicit method solves at each step:
`linfactor!(c, W)` takes the matrix (and may overwrite it), `linsolve!(c, x, b)` solves for the last
matrix taken. Made by a `LinearSolver`'s `lincache`.
"""
@trait LinearCache begin
    linfactor!(c::LinearCache, W::Matrix{Float64})::Nothing
    linsolve!(c::LinearCache, x::Vector{Float64}, b::Vector{Float64})::Nothing
end
"""
A linear solver an implicit method takes (LinearSolve.jl's algorithms: an open set inside the open set of
algorithms; stage C of move 3). `lincache(ls, n)` makes its state for systems of size `n`.
"""
@trait LinearSolver begin
    lincache(ls::LinearSolver, n::Int)::LinearCache
end
"Dense LU with partial pivoting, LAPACK's: the default linear solver."
struct DenseLU end
mutable struct DenseLUCache
    F::LinearAlgebra.LU{Float64,Matrix{Float64},Vector{Int}}
end
@implement LinearCache DenseLUCache begin
    linfactor!(c, W) = (c.F = lu!(W); nothing)
    linsolve!(c, x, b) = (ldiv!(x, c.F, b); nothing)
end
@implement LinearSolver DenseLU begin
    lincache(ls, n) = DenseLUCache(lu!(Matrix{Float64}(LinearAlgebra.I, n, n)))
end

"""
Tsit5's state between steps. Its type holds the element type, nothing else: the parameters stay in
the outer layer (W-63), and the callbacks are kept with their attributes (`Vector{StepCallback{…}}`),
so their types do not reach it. `uprev`, `kprev` and `tprev` are the last accepted step's start, for `interp!`.
"""
mutable struct Tsit5Integrator{T}
    const u::Vector{T}
    const uprev::Vector{T}
    t::Float64
    dt::Float64
    u_modified::Bool
    terminated::Bool
    const k1::Vector{T}
    const k2::Vector{T}
    const k3::Vector{T}
    const k4::Vector{T}
    const k5::Vector{T}
    const k6::Vector{T}
    const k7::Vector{T}
    const tmp::Vector{T}
    const unew::Vector{T}
    const utilde::Vector{T}
    qold::Float64
    naccept::Int
    nreject::Int
    nf::Int
    tprev::Float64
    const kprev::Vector{T}
    const tstops::Vector{Float64}
    const callbacks::Vector{StepCallback{Vector{T},Float64}}
    const events::Vector{StepEvent{Vector{T},Float64}}
    const evprev::Vector{Float64}
end
step_tstop!(i::Tsit5Integrator, τ::Float64) = (_add_tstop!(i.tstops, i.t, τ); nothing)
step_terminate!(i::Tsit5Integrator) = (i.terminated = true; nothing)
"The state at `τ` within the last accepted step, by the cubic Hermite interpolant (as for `saveat`)."
step_interp!(i::Tsit5Integrator{T}, out::Vector{T}, τ::Float64) where {T} =
    (hermite!(out, (τ - i.tprev) / i.dt, i.dt, i.uprev, i.kprev, i.unew, i.k7); nothing)
@implement StepView Tsit5Integrator{T} where {T}

"Insert `τ` into the sorted times the solver must step to; a time not ahead of `t` is ignored."
function _add_tstop!(tstops::Vector{Float64}, t::Float64, τ::Float64)
    τ > t || return nothing
    k = searchsortedfirst(tstops, τ)
    (k <= length(tstops) && tstops[k] == τ) || insert!(tstops, k, τ)
    return nothing
end

# --- Tsit5 (tableau from OrdinaryDiffEqTsit5 2.0.0, Tsitouras 2011) -------------------------------

const c1, c2, c3, c4 = 0.161, 0.327, 0.9, 0.9800255409045097
const a21 = 0.161
const a31, a32 = -0.008480655492356989, 0.335480655492357
const a41, a42, a43 = 2.8971530571054935, -6.359448489975075, 4.3622954328695815
const a51, a52, a53, a54 = 5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525
const a61, a62, a63, a64, a65 = 5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383
const a71, a72, a73, a74, a75, a76 = 0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774
const bt1, bt2, bt3, bt4 = -0.00178001105222577714, -0.0008164344596567469, 0.007880878010261995, -0.1447110071732629
const bt5, bt6, bt7 = 0.5823571654525552, -0.45808210592918697, 0.015151515151515152

"One Tsit5 step of size `dt` from `i.u` (with `i.k1 = f(u, t)`): the candidate in `i.unew`, `i.k7 = f(unew, t + dt)`; returns the error estimate."
function tsit5_step!(m, i::Tsit5Integrator{T}, dt::Float64, abstol::Float64, reltol::Float64) where {T}
    u, t = i.u, i.t
    k1, k2, k3, k4, k5, k6, k7, tmp, unew = i.k1, i.k2, i.k3, i.k4, i.k5, i.k6, i.k7, i.tmp, i.unew
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * a21 * k1[j]; end
    core_rhs!(m, k2, tmp, t + c1 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a31 * k1[j] + a32 * k2[j]); end
    core_rhs!(m, k3, tmp, t + c2 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j]); end
    core_rhs!(m, k4, tmp, t + c3 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j]); end
    core_rhs!(m, k5, tmp, t + c4 * dt)
    @inbounds for j in eachindex(u)
        tmp[j] = u[j] + dt * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j] + a65 * k5[j])
    end
    core_rhs!(m, k6, tmp, t + dt)
    @inbounds for j in eachindex(u)
        unew[j] = u[j] + dt * (a71 * k1[j] + a72 * k2[j] + a73 * k3[j] + a74 * k4[j] + a75 * k5[j] + a76 * k6[j])
    end
    core_rhs!(m, k7, unew, t + dt)
    i.nf += 6
    # the embedded error, as OrdinaryDiffEq's calculate_residuals with the RMS norm
    acc = 0.0
    @inbounds for j in eachindex(u)
        e = dt * (bt1 * k1[j] + bt2 * k2[j] + bt3 * k3[j] + bt4 * k4[j] + bt5 * k5[j] + bt6 * k6[j] + bt7 * k7[j])
        sc = abstol + max(abs(primal(u[j])), abs(primal(unew[j]))) * reltol
        acc += abs2(primal(e) / sc)
    end
    return sqrt(acc / length(u))
end

"Hairer's initial step size (Solving ODEs I, II.4), as OrdinaryDiffEq's `ode_determine_initdt` in its simple form."
function initdt(m, i::Tsit5Integrator{T}, tdir::Float64, opts::SolverOptions) where {T}
    u, f0, t = i.u, i.k1, i.t
    sc(x) = opts.abstol + abs(primal(x)) * opts.reltol
    d0 = sqrt(sum(abs2(primal(u[j]) / sc(u[j])) for j in eachindex(u)) / length(u))
    d1 = sqrt(sum(abs2(primal(f0[j]) / sc(u[j])) for j in eachindex(u)) / length(u))
    h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1
    h0 = min(h0, opts.dtmax)
    tmp = i.tmp
    @inbounds for j in eachindex(u); tmp[j] = u[j] + h0 * f0[j]; end
    core_rhs!(m, i.k2, tmp, t + h0)
    i.nf += 1
    d2 = sqrt(sum(abs2(primal(i.k2[j] - f0[j]) / sc(u[j])) for j in eachindex(u)) / length(u)) / h0
    h1 = max(d1, d2) <= 1e-15 ? max(1e-6, h0 * 1e-3) : (0.01 / max(d1, d2))^(1 / 5)
    return min(100 * h0, h1, opts.dtmax)
end

"The cubic Hermite interpolant between two accepted states, for `saveat`."
function hermite!(out::Vector{T}, θ::Float64, dt::Float64, u0::Vector{T}, f0::Vector{T}, u1::Vector{T}, f1::Vector{T}) where {T}
    h00 = (1 + 2θ) * (1 - θ)^2
    h10 = θ * (1 - θ)^2
    h01 = θ^2 * (3 - 2θ)
    h11 = θ^2 * (θ - 1)
    @inbounds for j in eachindex(out)
        out[j] = h00 * u0[j] + h10 * dt * f0[j] + h01 * u1[j] + h11 * dt * f1[j]
    end
    return out
end

# --- the driver ---------------------------------------------------------------------------------

"""
    solve_ode(prob, alg, callbacks, opts) -> ODESolution

Solve a problem of any type that implements `ODEProblemTrait`, the outer layer (W-63): the problem's
functions partially evaluated on its parameters (`core_problem`), the user's callbacks (`DiscreteCallback`,
`ContinuousCallback`, values of callback types) entered at the problem's types and adapted to the core
(`adapt_callbacks`), then the algorithm's driver (`solve_core`), which never sees the parameters' type. What
depends on the parameters' type is this function and the adapters: a few constructors and closures.
"""
Base.@nospecializeinfer function solve_ode(@nospecialize(prob), alg::ODEAlgorithm, callbacks::Vector{Any}, opts::SolverOptions)
    m = core_problem(prob)
    cbs = isempty(callbacks) ? callbacks : adapt_callbacks(callbacks, typeof(prob.u0), prob.p)
    return solve_core(alg, m, cbs, opts)
end

#     solve_core(Tsit5(), m::CoreProblem, callbacks, opts) -> ODESolution
#
# Integrate the core's problem from `m.u0` over `m.tspan` with Tsit5 and a PI step-size controller;
# `callbacks` are values of `StepCallback` and `StepEvent` (the adapted user's callbacks). The element
# type `T` is the state's: `Float64`, `Float32`, or a dual number for a gradient through the solve; the
# step-size control reads primal values, so a gradient follows the primal solve's steps. Callbacks take
# `Float64` states.
@implement ODEAlgorithm Tsit5 begin
    solve_core(a, m, callbacks, opts) = tsit5_solve(m, m.u0, m.tspan, callbacks, opts)
end

"Tsit5 for the core's problem `m`, at its state's element type `T`."
function tsit5_solve(m, u0::Vector{T}, tspan::Tuple{Float64,Float64},
                     callbacks::Vector{Any}, opts::SolverOptions) where {T}
    i = tsit5_init(m, u0, tspan, callbacks)
    return tsit5_run!(m, i, tspan[2], opts)
end

"""
Tsit5's `init`: the integrator at `T`, the core's callbacks kept (`enter_step_callbacks`), and
`f(u0, t0)`. Its type is `Tsit5Integrator{T}` whatever the callbacks and the parameters.
"""
function tsit5_init(m, u0::Vector{T}, tspan::Tuple{Float64,Float64}, callbacks::Vector{Any}) where {T}
    t0 = tspan[1]
    n = length(u0)
    T === Float64 || isempty(callbacks) || throw(ArgumentError("callbacks take Float64 states; this solve has $T"))
    d, c = enter_step_callbacks(callbacks, Vector{T})
    z() = zeros(T, n)
    i = Tsit5Integrator{T}(copy(u0), copy(u0), t0, 0.0, false, false,
                           z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), 1e-4, 0, 0, 0,
                           t0, z(), Float64[], d, c, zeros(Float64, length(c)))
    core_rhs!(m, i.k1, i.u, t0)
    i.nf += 1
    return i
end

"""
The time of a sign change of a continuous callback's condition within `(tl, tr]`, `gl` its value at
`tl`: bisection on the solver's interpolant, to the resolution of `Float64`; returns the last time
found on `tl`'s side, so the state there has not crossed. (The same text as PlainODE's.)
"""
function event_time(cb, iv, out, tl::Float64, tr::Float64, gl::Float64)
    for _ in 1:100
        tm = (tl + tr) / 2
        (tm <= tl || tm >= tr) && break
        step_interp!(iv, out, tm)
        gm = step_event_condition(cb, out, tm, iv)
        if gm != 0 && (gm > 0) == (gl > 0)
            tl = tm
        else
            tr = tm
        end
    end
    return tl
end

"Tsit5's driver: steps the integrator `i` to `tf` (or until a callback terminates it)."
function tsit5_run!(m, i::Tsit5Integrator{T}, tf::Float64, opts::SolverOptions) where {T}
    t0 = i.t
    n = length(i.u)
    dt = initdt(m, i, 1.0, opts)
    iv = Dynamic{StepView{Vector{T},Float64}}(i)
    ts = Float64[t0]
    us = Vector{T}[copy(i.u)]
    saveat = opts.saveat
    isave = 1
    if !isempty(saveat) && saveat[1] == t0
        isave = 2
    elseif !isempty(saveat)
        empty!(ts); empty!(us)
    end
    events = i.events
    for k in eachindex(events)
        i.evprev[k] = step_event_condition(events[k], i.u, i.t, iv)
    end
    # PI controller, OrdinaryDiffEq's defaults for a 5th-order method
    beta1, beta2, gamma, qmin, qmax = 7 / 50, 2 / 25, 0.9, 0.2, 10.0
    retcode = :Success
    iters = 0
    while i.t < tf && !i.terminated
        iters += 1
        if iters > opts.maxiters
            retcode = :MaxIters
            break
        end
        dt = min(dt, tf - i.t)
        tstop_hit = false
        if !isempty(i.tstops) && i.tstops[1] - i.t <= dt
            dt = i.tstops[1] - i.t
            tstop_hit = true
        end
        EEst = tsit5_step!(m, i, dt, opts.abstol, opts.reltol)
        q11 = EEst^beta1
        if EEst <= 1.0
            # accept: advance, the continuous callbacks' events, save, then the discrete callbacks
            tnew = tstop_hit ? i.tstops[1] : i.t + dt
            copyto!(i.uprev, i.u)
            copyto!(i.kprev, i.k1)
            copyto!(i.u, i.unew)
            copyto!(i.k1, i.k7)
            i.tprev = i.t
            i.t = tnew
            i.dt = dt
            i.naccept += 1
            ke = 0
            if T === Float64 && !isempty(events)
                te = i.t
                for k in eachindex(events)
                    g = step_event_condition(events[k], i.u, i.t, iv)
                    g0 = i.evprev[k]
                    tl = i.tprev
                    if isnan(g0)                         # the step after this callback's event: its value a
                        tl = i.tprev + i.dt / 100        # little into the step (SciML's repeat nudge)
                        step_interp!(iv, i.utilde, tl)
                        g0 = step_event_condition(events[k], i.utilde, tl, iv)
                    end
                    if g0 != 0 && (g == 0 || (g > 0) != (g0 > 0))
                        tk = event_time(events[k], iv, i.utilde, tl, i.t, g0)
                        if ke == 0 || tk < te
                            ke, te = k, tk
                        end
                    end
                    i.evprev[k] = g
                end
                if ke != 0                                   # the integrator moves to the earliest event
                    step_interp!(iv, i.utilde, te)
                    copyto!(i.u, i.utilde)
                    i.t = te
                end
            end
            tstop_hit && ke == 0 && popfirst!(i.tstops)      # a tstop behind an event stays
            if isempty(saveat)
                push!(ts, i.t); push!(us, copy(i.u))
            else
                while isave <= length(saveat) && saveat[isave] <= i.t
                    θ = (saveat[isave] - i.tprev) / dt
                    push!(ts, saveat[isave])
                    push!(us, hermite!(zeros(T, n), θ, dt, i.uprev, i.kprev, i.unew, i.k7))
                    isave += 1
                end
            end
            if ke != 0
                i.u_modified = true
                step_event_affect!(events[ke], iv)
                core_rhs!(m, i.k1, i.u, i.t)                     # the time moved back: f(u, t) again
                i.nf += 1
                i.u_modified = false
                for k in eachindex(events)                   # the conditions again at the event's state;
                    i.evprev[k] = k == ke ? NaN : step_event_condition(events[k], i.u, i.t, iv)
                end                                          # the event's own a little into the next step
            end
            T === Float64 && for cb in i.callbacks
                if step_condition(cb, i.u, i.t, iv)
                    i.u_modified = true
                    step_affect!(cb, iv)
                    if i.u_modified
                        core_rhs!(m, i.k1, i.u, i.t)          # the state changed: f(u, t) again
                        i.nf += 1
                        i.u_modified = false
                    end
                end
                i.terminated && break
            end
            q = clamp(q11 / i.qold^beta2 / gamma, 1 / qmax, 1 / qmin)
            dt = min(dt / q, opts.dtmax)
            i.qold = max(EEst, 1e-4)
        else
            i.nreject += 1
            dt = dt / min(1 / qmin, q11 / gamma)
        end
    end
    i.terminated && (retcode = :Terminated)
    return ODESolution{T}(ts, us, retcode, i.naccept, i.nreject, i.nf)
end


"The value a step-size controller sees: a dual number's primal value, so that gradients follow the primal's steps."
primal(x::Float64) = x
primal(x::AbstractFloat) = Float64(x)
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))
