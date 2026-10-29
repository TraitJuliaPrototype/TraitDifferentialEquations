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
#   - a solution is a `Trajectory{U, T}` (TraitArrayTools') and keeps its problem as a kept value: code over
#     solutions is compiled per state and time type, whatever the model, and the solution's type has no
#     axis for the problem's function or type;
#   - the parameters' type stays in the outermost layer (W-63): `solve_ode` partially evaluates the
#     problem's functions on its parameters (`CoreProblem`) and adapts the callbacks (`ParamCallback`),
#     so the solvers' code is compiled per state type, whatever the parameters.
# (the body of the trait module `ODECore`, src/ODECore.jl)
using TraitWorld
import TraitWorld: solve!          # BaseTraits' `solve!` (a `Solvable`'s), extended for `init`'s integrator
using LinearAlgebra: lu!, I, UniformScaling
using TraitIndexing                  # symbolic access (move 4): SymbolicIndexingInterface's part
using TraitArrayTools                # the solution as a trajectory (move 5): RecursiveArrayTools' part
import TraitArrayTools: times, state, at
import TraitIndexing: variable_index, parameter_index, state_values, parameter_values
import LinearAlgebra
import ForwardDiff
export ODEProblemTrait, rhs!, jacobian!, rhs_function, Trajectory, Callback, condition, affect!, IntegratorView, Tsit5, ODESolution, CoreSolution, SolverOptions,
       solve_ode, ODEProblem, FunctionCallback, RHSSig, DualSig, JacSig, DUAL_CHUNK, CondSig, AffectSig, FDual, ODETag,
       no_jacobian, UserFunction, add_tstop!, terminate!, interp!, EventCallback, event_condition, event_affect!,
       DiscreteCallback, ContinuousCallback, FunctionEventCallback, enter_callbacks,
       CoreProblem, core_problem, adapt_callbacks, ODEAlgorithm, solve_core, LinearSolver, linfactor, DenseLU,
       NonlinearSolver, NLCache, nlcache, nlstage!, Newton, newton_stage!, getseries, StepView, StepCallback, StepEvent, ParamIntegrator,
       ODEFunction, HasMassMatrix, mass_matrix, HasJacobianPrototype, jac_prototype, HasSystem, system, getsystem,
       HasCallbacks, problem_callbacks, takes_mass_matrix, auto_dt_reset!, save_discretes!, parameters_at,
       initialize!, event_initialize!, event_affect_neg!, no_initialize,
       CoreIntegrator, advance!, core_solution, ODEStepper, init_core, init_ode, step!

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
the last step's size `dt::T`, the parameters `p::P` (by name: `ps[:σ]`, a `ParameterView{P}`), two flags, and three operations: `add_tstop!`
(a time the solver must step to), `terminate!`, and `interp!(i, out, τ)`, the state at `τ` within
the last accepted step by the solver's interpolant; `auto_dt_reset!`, the next step's size estimated again
from the state (after a callback changed the state or the parameters), and `save_discretes!`, the
parameters saved at the current time into the solution (after a callback changed them: SciML's
`save_discretes!`; `parameters_at(sol, τ)` reads them back). `U`, `T` and `P` are defined by the fields, so
a callback reads `integ.p` at the problem's parameter type. The outer layer's `ParamIntegrator` conforms;
it is also what `init` returns.
"""
@trait IntegratorView{U, T, P} <: HasProperty{:ps, ParameterView{P}} begin
    @const u::U
    @const t::T
    @const dt::T
    @const p::P
    u_modified::Bool
    terminated::Bool
    add_tstop!(i::IntegratorView{U, T, P}, τ::T)::Nothing
    terminate!(i::IntegratorView{U, T, P})::Nothing
    interp!(i::IntegratorView{U, T, P}, out::U, τ::T)::Nothing
    auto_dt_reset!(i::IntegratorView{U, T, P})::Nothing
    save_discretes!(i::IntegratorView{U, T, P})::Nothing
end

"""
    Callback{U, T, P}

A discrete callback, checked after every accepted step: `condition` decides, `affect!` acts on the
integrator, which it sees as a `Dynamic{IntegratorView{U, T, P}}`; `initialize!` runs once at the solver's
`init`, at the initial state (SciML's `initialize`: for example, times to stop at). The integrator keeps its
callbacks as `Vector{Callback{U, T, P}}`, so their types reach neither the integrator's type nor the driver.
A callback type implements it at the problem's types: `@implement Callback{Vector{Float64}, Float64, P} MyCallback`.
"""
@trait Callback{U, T, P} begin
    condition(cb::Callback{U, T, P}, u::U, t::T, integ::Dynamic{IntegratorView{U, T, P}})::Bool
    affect!(cb::Callback{U, T, P}, integ::Dynamic{IntegratorView{U, T, P}})::Nothing
    initialize!(cb::Callback{U, T, P}, u::U, t::T, integ::Dynamic{IntegratorView{U, T, P}})::Nothing = nothing
end

"""
    EventCallback{U, T, P}

A continuous callback: an event where the scalar `event_condition` changes sign within a step; the
solver locates it by bisection on the interpolant (`interp!`), moves the integrator to it, and calls
`event_affect!` where the condition rises through zero (from negative), `event_affect_neg!` where it
falls (SciML's `affect!` and `affect_neg!`; by default the same). Every crossing is an event.
`event_initialize!` runs once at the solver's `init`.
"""
@trait EventCallback{U, T, P} begin
    event_condition(cb::EventCallback{U, T, P}, u::U, t::T, integ::Dynamic{IntegratorView{U, T, P}})::Float64
    event_affect!(cb::EventCallback{U, T, P}, integ::Dynamic{IntegratorView{U, T, P}})::Nothing
    event_affect_neg!(cb::EventCallback{U, T, P}, integ::Dynamic{IntegratorView{U, T, P}})::Nothing = event_affect!(cb, integ)
    event_initialize!(cb::EventCallback{U, T, P}, u::U, t::T, integ::Dynamic{IntegratorView{U, T, P}})::Nothing = nothing
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
    step_dt_reset!(i::StepView{U, T})::Nothing
end

"A discrete callback as the solver core calls it: over `StepView{U, T}`, without the parameters."
@trait StepCallback{U, T} begin
    step_condition(cb::StepCallback{U, T}, u::U, t::T, sv::Dynamic{StepView{U, T}})::Bool
    step_affect!(cb::StepCallback{U, T}, sv::Dynamic{StepView{U, T}})::Nothing
    step_initialize!(cb::StepCallback{U, T}, u::U, t::T, sv::Dynamic{StepView{U, T}})::Nothing = nothing
end

"A continuous callback as the solver core calls it: `step_event_affect!` on a rising crossing, `step_event_affect_neg!` on a falling one."
@trait StepEvent{U, T} begin
    step_event_condition(cb::StepEvent{U, T}, u::U, t::T, sv::Dynamic{StepView{U, T}})::Float64
    step_event_affect!(cb::StepEvent{U, T}, sv::Dynamic{StepView{U, T}})::Nothing
    step_event_affect_neg!(cb::StepEvent{U, T}, sv::Dynamic{StepView{U, T}})::Nothing = step_event_affect!(cb, sv)
    step_event_initialize!(cb::StepEvent{U, T}, u::U, t::T, sv::Dynamic{StepView{U, T}})::Nothing = nothing
end

"""
What a solver returns (the core's solution, W-63): the saved times and states, the return code and the
counts. It has no problem: the core never sees the parameters. `solve_ode` makes the user's `ODESolution`
from it.
"""
struct CoreSolution{T}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    retcode::Symbol
    naccept::Int
    nreject::Int
    nf::Int
end

"""
    CoreIntegrator{U, T}

A solver's integrator as the outer layer drives it (SciML's `init`, `step!`, `solve!`): `advance!` takes one
accepted step, with its events, saves and callbacks, and returns `false` when it cannot step (at the end of
the span, terminated, or a failure); `core_solution` is what it saved so far, with the return code. The same
value is the callbacks' `StepView{U, T}`. Made by a steppable algorithm's `init_core` (`ODEStepper`).
"""
@trait CoreIntegrator{U, T} begin
    advance!(i::CoreIntegrator{U, T})::Bool
    core_solution(i::CoreIntegrator{U, T})::CoreSolution
end

# --- optional properties of a problem's function, and of a problem (open issue 119) -------------

"""
    HasMassMatrix

A value with a mass matrix `M` (SciML's `f.mass_matrix`): the problem `M du/dt = f(u, p, t)`. Implemented by
a problem's function type (a callable model, `ODEFunction`) or a problem type. The front end reads it at the
problem's construction (`ODEProblem`), and `solve` and `init` refuse a mass matrix other than the identity
for an algorithm that does not take one (`takes_mass_matrix`).
"""
@trait HasMassMatrix begin
    mass_matrix(x::HasMassMatrix)::Any
end

"""
    HasJacobianPrototype

A value with a Jacobian's sparsity prototype (SciML's `f.jac_prototype`), or `nothing`. A hint: the problem
keeps it for the solvers that use one; a solver that does not computes the dense Jacobian, with the same
results.
"""
@trait HasJacobianPrototype begin
    jac_prototype(x::HasJacobianPrototype)::Any
end

"""
    HasSystem

A value that carries a system object (SciML's `f.sys`, `prob.f.sys`, `sol.prob.f.sys`, `integ.f.sys`;
SymbolicIndexingInterface's `symbolic_container`): the model's own description, which TDE keeps and gives
back unchanged (`getsystem`). A problem's function type implements it (a callable model returns itself);
`ODEFunction`, TDE's problem, its solution and its integrator carry the one their function gave. Where the
system is an `IndexProvider`, its names serve `sol[:x]` and `getstate`.
"""
@trait HasSystem begin
    system(x::HasSystem)::Any
end

"""
    HasCallbacks

A problem that keeps callbacks (SciML's `ODEProblem(…; callback)`): `solve` and `init` take them before the
ones they are given.
"""
@trait HasCallbacks begin
    problem_callbacks(x::HasCallbacks)::Vector{Any}
end

"""
    getsystem(x)

The system object `x` carries (`HasSystem`: a problem, a solution, an integrator, a callback's view of it, a model),
or `nothing`.
"""
getsystem(x::HasSystem) = system(x)
getsystem(x) = nothing

# The front end's reads of a function's or a problem's optional properties: the trait's, or the default.
# - TDE's own records (`ODEProblem`, `ODEFunction`) are read by their fields, and a function (a closure, a generic
#   function) that declares none of the traits takes the default without entering a bound: a bound's entry is
#   compiled per argument type, which would compile the front end again for every new function.
# - A kept value (a function kept at `Callable`) is read through its kept trait only: at a bound its held object does
#   not conform to, TraitWorld raises `TraitBoundError` where a plain value reaches the ordinary fallback (traits §5.5,
#   §6.5; a candidate open issue, E-239). Casts (`compiled_for`, `with_dynamic`) are seen through, at the bound.
"Whether `x` is a kept value (`Dynamic{X}`, `Compiled{X}`, `Kept{X}`)."
_iskept(@nospecialize(x)) = x isa Union{Dynamic,Compiled,Kept}
"Whether the front end reads `x`'s property `X` by the bound: not for a function that does not declare `X`, nor a kept value of another trait."
_by_bound(@nospecialize(x), @nospecialize(X::Type)) =
    !((x isa Function || _iskept(x)) && !conforms(typeof(x), X))
"The system object `x` carries, or `nothing`."
Base.@nospecializeinfer function _system(@nospecialize(x))
    x === nothing && return nothing
    x isa ODEProblem && return getfield(x, :sys)
    x isa ODEFunction && return getfield(x, :sys)
    return _by_bound(x, HasSystem) ? getsystem(x) : nothing
end
"""
The mass matrix `x` carries (`HasMassMatrix`), or the identity. A callable object (not a function: a closure's
captured variables are its fields) or a problem with a `mass_matrix` property other than the identity that does
not declare the trait (SciML's convention reads the property) is refused: solving it as an ODE would ignore the
mass matrix.
"""
Base.@nospecializeinfer function _mass_matrix(@nospecialize(x))
    x isa ODEProblem && return getfield(x, :mass_matrix)
    x isa ODEFunction && return getfield(x, :mass_matrix)
    return _by_bound(x, HasMassMatrix) ? _mass_matrix_of(x) : I
end
_mass_matrix_of(x::HasMassMatrix) = mass_matrix(x)
Base.@nospecializeinfer function _mass_matrix_of(@nospecialize(x))
    !(x isa Function) && hasproperty(x, :mass_matrix) && !_is_identity(getproperty(x, :mass_matrix)) && throw(ArgumentError(string(
        "a value of type ", nameof(typeof(x)), " has a property `mass_matrix` other than the identity but does not ",
        "implement `HasMassMatrix`: declare it (`@implement HasMassMatrix …`), so that the solvers see it")))
    return I
end
"Whether a mass matrix is the identity: then the problem is an ODE."
_is_identity(M::UniformScaling) = M == I
_is_identity(M::AbstractMatrix) = M == I
_is_identity(@nospecialize(M)) = false
"The Jacobian prototype `x` carries (`HasJacobianPrototype`), or `nothing`."
Base.@nospecializeinfer function _jac_prototype(@nospecialize(x))
    x isa ODEProblem && return getfield(x, :jac_prototype)
    x isa ODEFunction && return getfield(x, :jac_prototype)
    return _by_bound(x, HasJacobianPrototype) ? _jac_prototype_of(x) : nothing
end
_jac_prototype_of(x::HasJacobianPrototype) = jac_prototype(x)
_jac_prototype_of(x) = nothing
"The callbacks a problem keeps (`HasCallbacks`), or none."
Base.@nospecializeinfer function _problem_callbacks(@nospecialize(x))
    x isa ODEProblem && return getfield(x, :callbacks)
    return _by_bound(x, HasCallbacks) ? _problem_callbacks_of(x) : Any[]
end
_problem_callbacks_of(x::HasCallbacks) = problem_callbacks(x)
_problem_callbacks_of(x) = Any[]
"A system's names, where it provides them (`IndexProvider`), kept `Dynamic`; else `nothing`."
Base.@nospecializeinfer _names(@nospecialize(s)) = s === nothing || !_by_bound(s, IndexProvider) ? nothing : _names_of(s)
_names_of(s::IndexProvider) = Dynamic{IndexProvider}(s)
_names_of(s) = nothing

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
`p` is what callbacks see as `integ.p`, the same object the kept functions hold. The optional properties
its function carried (open issue 119), each read at the front end and kept as given: the system object
(`HasSystem`) with its names where it provides them, the mass matrix (`HasMassMatrix`), the Jacobian
prototype (`HasJacobianPrototype`); and the callbacks it keeps (`HasCallbacks`). Made by the front end
`ODEProblem(f, u0, tspan, p; callback)`.
"""
struct ODEProblem{T,P,F<:Callable{CoreRHSSig{T},Nothing},J<:Callable{CoreJacSig{T},Nothing}}
    f::F
    u0::Vector{T}
    tspan::Tuple{Float64,Float64}
    p::P
    jac::J
    ad::Bool
    fplain::UserFunction
    sys::Any                                       # the system object, or nothing
    names::Union{Nothing,Dynamic{IndexProvider}}   # its names (move 4), where it provides them
    mass_matrix::Any                               # the identity, or the function's
    jac_prototype::Any                             # nothing, or the function's
    callbacks::Vector{Any}                         # the callbacks it keeps
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
"""
The Jacobian by duals: one call of the right-hand side, seen at the library's duals of chunk `N`, per chunk of `N`
columns. Its declared result reaches the callers compiled by facts, where the entry sees a function kept at
another signature at `Any` (move 5: the problem kept by its solution is checked as a member of its family).
"""
function _ad_jacobian!(fd::Callable{CoreDualSig{N},Nothing}, J::Matrix{Float64}, u::Vector{Float64}, t::Float64, ::Val{N})::Nothing where {N}
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
@implement HasSystem ODEProblem{T,P,F,J} where {T,P,F,J} begin
    system(x) = getfield(x, :sys)
end
@implement HasMassMatrix ODEProblem{T,P,F,J} where {T,P,F,J} begin
    mass_matrix(x) = getfield(x, :mass_matrix)
end
@implement HasJacobianPrototype ODEProblem{T,P,F,J} where {T,P,F,J} begin
    jac_prototype(x) = getfield(x, :jac_prototype)
end
@implement HasCallbacks ODEProblem{T,P,F,J} where {T,P,F,J} begin
    problem_callbacks(x) = getfield(x, :callbacks)
end

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
`ODEProblem` keeps them so already (`PartialFn`): they pass as they are kept, each read back with its
attribute (traits reference §7.4). For a problem type that implements `ODEProblemTrait` itself, its
`rhs!` and `jacobian!` partially evaluated on the problem, `Dynamic`.
"""
function core_problem(prob::ODEProblem{T,P,F,J}) where {T,P,F,J}
    # a compiled function: the problem type is a member of the problem family, checked at its first use (traits §4.7)
    F <: Dynamic || _member_checked(prob)
    return CoreProblem(prob.f, prob.u0, prob.tspan, prob.jac, prob.ad)
end
"Nothing: the bound's first-use check of the problem type, as a member of the problem family."
_member_checked(prob::ODEProblemTrait) = nothing
# a bounded function: `prob` is a trait object, so the trait's defaults serve what the type does not implement
core_problem(prob::ODEProblemTrait) =
    CoreProblem(ProblemRHS(prob), prob.u0, prob.tspan, ProblemJac(prob), false)

"""
    DiscreteCallback(condition, affect!; initialize = no_initialize)
    ContinuousCallback(condition, affect!, affect_neg! = affect!; initialize = no_initialize)

The user's callback from plain functions (SciML's): `condition(u, t, integ)::Bool` (discrete) or
`::Float64` (continuous: an event where it changes sign), `affect!(integ)`; for a continuous callback
`affect_neg!(integ)` where the condition falls through zero (`affect!` where it rises); `initialize(u, t, integ)`,
called once at the solver's `init` (SciML's `initialize`, without the callback argument). They keep the functions
before the problem's types are known, each with its attribute (`Kept{Callable}`, traits reference §7.5);
a solver's `init` enters them at `(U, T, P)` (`enter_callbacks`). Every crossing is an event: an affect
of `nothing` (SciML's "not an event in this direction") is refused.
"""
struct DiscreteCallback
    condition::Callable              # Kept{Callable}: a function kept without a signature
    affect!::Callable
    initialize::Callable
    Base.@nospecializeinfer DiscreteCallback(@nospecialize(c), @nospecialize(a); @nospecialize(initialize = no_initialize)) =
        new(Kept{Callable}(c), Kept{Callable}(_affect(a)), Kept{Callable}(initialize))
end
struct ContinuousCallback
    condition::Callable
    affect!::Callable
    affect_neg!::Callable
    initialize::Callable
    Base.@nospecializeinfer ContinuousCallback(@nospecialize(c), @nospecialize(a), @nospecialize(an = a); @nospecialize(initialize = no_initialize)) =
        new(Kept{Callable}(c), Kept{Callable}(_affect(a)), Kept{Callable}(_affect(an)), Kept{Callable}(initialize))
end
"A callback's `initialize` that does nothing (the default)."
no_initialize(u, t, integ) = nothing
Base.@nospecializeinfer _affect(@nospecialize(a)) = a === nothing ?
    throw(ArgumentError("an affect of `nothing` is not taken: every crossing is an event (pass a function that does nothing)")) : a

"A discrete callback as the integrator keeps it: its plain functions entered at the problem's types, each kept with its attribute (`Kept{Callable{…}}`)."
struct FunctionCallback{U,T,P}
    cond::Callable{CondSig{U,T,P},Bool}
    aff::Callable{AffectSig{U,T,P},Nothing}
    init::Callable{CondSig{U,T,P},Nothing}
end
condition(cb::FunctionCallback{U,T,P}, u::U, t::T, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = cb.cond(u, t, integ)
affect!(cb::FunctionCallback{U,T,P}, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.aff(integ); nothing)
initialize!(cb::FunctionCallback{U,T,P}, u::U, t::T, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.init(u, t, integ); nothing)
@implement Callback{U,T,P} FunctionCallback{U,T,P} where {U,T,P}

"A continuous callback as the integrator keeps it."
struct FunctionEventCallback{U,T,P}
    cond::Callable{CondSig{U,T,P},Float64}
    aff::Callable{AffectSig{U,T,P},Nothing}
    affneg::Callable{AffectSig{U,T,P},Nothing}
    init::Callable{CondSig{U,T,P},Nothing}
end
event_condition(cb::FunctionEventCallback{U,T,P}, u::U, t::T, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = cb.cond(u, t, integ)
event_affect!(cb::FunctionEventCallback{U,T,P}, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.aff(integ); nothing)
event_affect_neg!(cb::FunctionEventCallback{U,T,P}, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.affneg(integ); nothing)
event_initialize!(cb::FunctionEventCallback{U,T,P}, u::U, t::T, integ::Dynamic{IntegratorView{U,T,P}}) where {U,T,P} = (cb.init(u, t, integ); nothing)
@implement EventCallback{U,T,P} FunctionEventCallback{U,T,P} where {U,T,P}

"""
    enter_callbacks(cbs, U, P) -> (discrete, continuous)

A solver's `init` step for callbacks: the user's callbacks entered at the problem's types `(U, Float64, P)`.
A `DiscreteCallback` or `ContinuousCallback` becomes a record of its functions kept at the signatures
(`FunctionCallback`, `FunctionEventCallback`), each with its attribute; a value of a callback type is kept as it is, refused
if it does not conform at these types. The integrator keeps the two vectors.
"""
function enter_callbacks(cbs::Vector{Any}, ::Type{U}, ::Type{P}) where {U,P}
    T = Float64
    d = Kept{Callback{U,T,P}}[]                  # `Callback{U,T,P}[]` is not taken as `Kept{…}` here (E-158 finding)
    c = Kept{EventCallback{U,T,P}}[]
    for cb in cbs
        if cb isa DiscreteCallback
            push!(d, FunctionCallback{U,T,P}(cb.condition, cb.affect!, cb.initialize))
        elseif cb isa ContinuousCallback
            push!(c, FunctionEventCallback{U,T,P}(cb.condition, cb.affect!, cb.affect_neg!, cb.initialize))
        else
            _enter_callback!(d, c, cb)
        end
    end
    return d, c
end
# a callback value goes to the vector of its trait, chosen by its bound (traits reference §5.5), which sees
# through casts and kept values; a value of both traits is a continuous one
_enter_callback!(d::Vector{Kept{Callback{U,T,P}}}, c::Vector{Kept{EventCallback{U,T,P}}}, cb::EventCallback{U,T,P}) where {U,T,P} =
    (push!(c, Kept{EventCallback{U,T,P}}(cb)); nothing)
_enter_callback!(d::Vector{Kept{Callback{U,T,P}}}, c::Vector{Kept{EventCallback{U,T,P}}}, cb::Callback{U,T,P}) where {U,T,P} =
    (push!(d, Kept{Callback{U,T,P}}(cb)); nothing)
_enter_callback!(d::Vector{Kept{Callback{U,T,P}}}, c::Vector{Kept{EventCallback{U,T,P}}}, cb::(EventCallback{U,T,P}, Callback{U,T,P})) where {U,T,P} =
    (push!(c, Kept{EventCallback{U,T,P}}(cb)); nothing)

"""
    ParamIntegrator{U, T, P}

The integrator as the user sees it (`IntegratorView{U, T, P}`): what callbacks see, and what `init` returns.
Over the solver core's `StepView{U, T}` (W-63): the core's state (the same array), copies of its time, step
and flags, and the parameters. The adapters copy the time, step and flags in before each of the user's
functions and the flags back after it (a declared field is a stored field, traits reference §3.3); the
operations go to the core's view. One per solve, shared by its callbacks. It also keeps:
- the problem's system object and names (`HasSystem`, move 4);
- the parameters saved by `save_discretes!` (times `pt`, values `pu`; the problem's at the start), for the solution;
- for `init`, the solver's integrator (`core`, a `Dynamic{CoreIntegrator{U, T}}`, so its type has no axis for the
  algorithm) and the problem as given, for `solve!`'s solution.
"""
mutable struct ParamIntegrator{U,T,P}
    u::U
    t::T
    dt::T
    const p::P
    u_modified::Bool
    terminated::Bool
    sv::Union{Nothing,Dynamic{StepView{U,T}}}
    const sys::Any                                       # the problem's system object, or nothing
    const names::Union{Nothing,Dynamic{IndexProvider}}   # its names (move 4), or none
    const pt::Vector{Float64}                            # the parameters saved: times
    const pu::Vector{P}                                  # and values
    core::Union{Nothing,Dynamic{CoreIntegrator{U,T}}}   # the solver's integrator (`init`), or nothing (`solve`)
    prob::Any                                            # the problem as given (`init`), or nothing
end
add_tstop!(i::ParamIntegrator{U,T}, τ::T) where {U,T} = (step_tstop!(i.sv::Dynamic{StepView{U,T}}, τ); nothing)
terminate!(i::ParamIntegrator) = (i.terminated = true; nothing)
interp!(i::ParamIntegrator{U,T}, out::U, τ::T) where {U,T} = (step_interp!(i.sv::Dynamic{StepView{U,T}}, out, τ); nothing)
auto_dt_reset!(i::ParamIntegrator{U,T}) where {U,T} = (step_dt_reset!(i.sv::Dynamic{StepView{U,T}}); nothing)
save_discretes!(i::ParamIntegrator) = (push!(i.pt, i.t); push!(i.pu, _saved(i.p)); nothing)
"A copy of the parameters to save: the solve goes on changing the integrator's."
_saved(p::AbstractArray) = copy(p)
_saved(p) = deepcopy(p)
@implement HasProperty{:ps, ParameterView{P}} ParamIntegrator{U,T,P} where {U,T,P} begin
    property(i, ::Val{:ps}) = ParameterView(getfield(i, :p), getfield(i, :names))
end
@implement IntegratorView ParamIntegrator{U,T,P} where {U,T,P}
# the integrator a callback sees, by name: `getstate(integ, :x)`, `getparam(integ, :σ)` (SciML's `integ[:x]`,
# `integ.ps[:σ]`), through the traits the integrator conforms to (traits §6.5)
@implement IndexProvider ParamIntegrator{U,T,P} where {U,T,P} begin
    variable_index(i, sym) = (sys = getfield(i, :names); sys === nothing ? nothing : variable_index(sys, sym))
    parameter_index(i, sym) = (sys = getfield(i, :names); sys === nothing ? nothing : parameter_index(sys, sym))
end
@implement ValueProvider{U,P} ParamIntegrator{U,T,P} where {U,T,P} begin
    state_values(i) = getfield(i, :u)
    parameter_values(i) = getfield(i, :p)
end
@implement HasSystem ParamIntegrator{U,T,P} where {U,T,P} begin
    system(i) = getfield(i, :sys)
end

function _sync_in!(pi::ParamIntegrator{U,T}, sv::StepView{U,T}) where {U,T}
    pi.sv = sv
    pi.u = sv.u; pi.t = sv.t; pi.dt = sv.dt
    pi.u_modified = sv.u_modified; pi.terminated = sv.terminated
    return nothing
end
_sync_out!(pi::ParamIntegrator{U,T}, sv::StepView{U,T}) where {U,T} =
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
step_initialize!(c::ParamCallback{U,T}, u::U, t::T, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); initialize!(c.cb, u, t, c.view); _sync_out!(c.pi, sv); nothing)
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
step_event_affect_neg!(c::ParamEvent{U,T}, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); event_affect_neg!(c.cb, c.view); _sync_out!(c.pi, sv); nothing)
step_event_initialize!(c::ParamEvent{U,T}, u::U, t::T, sv::Dynamic{StepView{U,T}}) where {U,T} =
    (_sync_in!(c.pi, sv); event_initialize!(c.cb, u, t, c.view); _sync_out!(c.pi, sv); nothing)
@implement StepEvent{U,T} ParamEvent{U,T,P} where {U,T,P}

"The outer layer's integrator for a problem's types: its parameters (saved at `t0` when callbacks may change them), system and names."
function param_integrator(::Type{U}, p::P, t0::Float64, sys, names, save::Bool) where {U,P}
    pi = ParamIntegrator{U,Float64,P}(U(), t0, 0.0, p, false, false, nothing, sys, names, Float64[], P[], nothing, nothing)
    save && save_discretes!(pi)
    return pi
end

"""
    adapt_callbacks(cbs, U, p) -> Vector{Any}
    adapt_callbacks!(pi, cbs) -> Vector{Any}

The outer layer's step for callbacks (W-63): the user's callbacks entered at the problem's types
`(U, Float64, P)` (`enter_callbacks`), each adapted to the solver core, which sees them as
`StepCallback{U, Float64}` and `StepEvent{U, Float64}`, without the parameters. They share the integrator `pi`.
"""
adapt_callbacks(cbs::Vector{Any}, ::Type{U}, p::P) where {U,P} =
    adapt_callbacks!(param_integrator(U, p, 0.0, nothing, nothing, false), cbs)
function adapt_callbacks!(pi::ParamIntegrator{U,T,P}, cbs::Vector{Any}) where {U,T,P}
    d, c = enter_callbacks(cbs, U, P)
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
        _enter_step_callback!(d, c, cb)
    end
    return d, c
end

_enter_step_callback!(d::Vector{Kept{StepCallback{U,T}}}, c::Vector{Kept{StepEvent{U,T}}}, cb::StepEvent{U,T}) where {U,T} =
    (push!(c, Kept{StepEvent{U,T}}(cb)); nothing)
_enter_step_callback!(d::Vector{Kept{StepCallback{U,T}}}, c::Vector{Kept{StepEvent{U,T}}}, cb::StepCallback{U,T}) where {U,T} =
    (push!(d, Kept{StepCallback{U,T}}(cb)); nothing)
_enter_step_callback!(d::Vector{Kept{StepCallback{U,T}}}, c::Vector{Kept{StepEvent{U,T}}}, cb::(StepEvent{U,T}, StepCallback{U,T})) where {U,T} =
    (push!(c, Kept{StepEvent{U,T}}(cb)); nothing)

"The callbacks' `initialize` at a solver's `init`, at the initial state (SciML's `initialize_callbacks!`): the continuous ones, then the discrete ones."
function initialize_step_callbacks!(d::Vector{Kept{StepCallback{U,T}}}, c::Vector{Kept{StepEvent{U,T}}}, u::U, t::T, sv::StepView{U,T}) where {U,T}
    for cb in c; step_event_initialize!(cb, u, t, sv); end
    for cb in d; step_initialize!(cb, u, t, sv); end
    return nothing
end

# --- options, solution, the integrator ---------------------------------------------------------

struct Tsit5 end

struct SolverOptions
    abstol::Float64
    reltol::Float64
    dtmax::Float64
    maxiters::Int
    saveat::Vector{Float64}          # empty: save every step
    tstops::Vector{Float64}          # times the solver must step to (SciML's `tstops`)
end
SolverOptions(abstol, reltol, dtmax, maxiters, saveat) = SolverOptions(abstol, reltol, dtmax, maxiters, saveat, Float64[])
"The front end's defaults (`solve`'s keywords)."
const DEFAULT_OPTIONS = SolverOptions(1e-6, 1e-3, Inf, 100_000, Float64[], Float64[])

"""
    ODESolution{T, P}

A solution as the user sees it (SciML's move 5): the saved times `t` and states `u`, the return code, the
counts, and the problem it solves, kept with its attribute (`sol.prob`, a `Kept{ODEProblemTrait{T, P}}`). Its
type names the state's element type and the parameters' type, never the model or the function: one solution
type for every function, model and problem type of those types. A `Trajectory{Vector{T}, Float64}`;
`sol(τ)` is `at(sol, τ)`.
"""
struct ODESolution{T,P}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    retcode::Symbol
    naccept::Int
    nreject::Int
    nf::Int
    prob::Kept{ODEProblemTrait{T,P}}            # the problem, kept: no axis for its function or its type
    sys::Any                                     # the problem's system object, or nothing
    names::Union{Nothing,Dynamic{IndexProvider}} # its names, or none
    pt::Vector{Float64}                          # the parameters saved by callbacks (`save_discretes!`): times,
    pu::Vector{P}                                # and values; empty when no callback could change them
end

# the solution as a trajectory (move 5): `Trajectory` is TraitArrayTools', RecursiveArrayTools' stand-in,
# so code over trajectories needs no solver
@implement Trajectory{Vector{T},Float64} ODESolution{T,P} where {T,P} begin
    times(s) = getfield(s, :t)
    state(s, k) = getfield(s, :u)[k]
    at(s, τ) = linear_at(getfield(s, :t), getfield(s, :u), τ)       # the solution keeps no interpolant
end
(s::ODESolution)(τ::Real) = at(s, Float64(τ))

# --- symbolic access (move 4, E-160): the solution's names are its problem's ---------------------------------
# the operations on the kept names themselves
@implement IndexProvider ODESolution{T,P} where {T,P} begin
    variable_index(s, sym) = (sys = getfield(s, :names); sys === nothing ? nothing : variable_index(sys, sym))
    parameter_index(s, sym) = (sys = getfield(s, :names); sys === nothing ? nothing : parameter_index(sys, sym))
end
@implement HasSystem ODESolution{T,P} where {T,P} begin
    system(s) = getfield(s, :sys)
end
"""
    parameters_at(sol, τ)

The parameters in force at `τ`: the last values saved at or before `τ` (`save_discretes!` in a callback; the
problem's at the start), or the problem's parameters where no callback could change them.
"""
function parameters_at(s::ODESolution{T,P}, τ::Real) where {T,P}
    pt = getfield(s, :pt)
    isempty(pt) && return getfield(s, :prob).p
    k = searchsortedlast(pt, Float64(τ))
    return getfield(s, :pu)[max(k, 1)]
end
# a solution's values: its final state, and its problem's parameters
@implement ValueProvider{Vector{T},P} ODESolution{T,P} where {T,P} begin
    state_values(s) = last(getfield(s, :u))
    parameter_values(s) = getfield(s, :prob).p
end
"The time series of state variable `sym` (SciML's `sol[:x]`): one getter per element type, whatever the names' provider."
function getseries(s::(Trajectory{U,Float64}, IndexProvider), sym::Symbol) where {U}
    i = variable_index(s, sym)
    i === nothing && throw(ArgumentError("no state variable named :$sym"))
    ts = times(s)
    return eltype(U)[state(s, k)[i] for k in eachindex(ts)]
end
"`sol.ps[:σ]`: the solution's parameters by name."
struct ParamView{S}
    s::S
end
Base.getindex(v::ParamView, sym::Symbol) = getparam(v.s, sym)
# the sugar, by the owner of the type (move 4: for scripts; trait code reads `getseries`, `getparam`)
Base.getindex(s::ODESolution, sym::Symbol) = getseries(s, sym)
Base.getproperty(s::ODESolution, name::Symbol) = name === :ps ? ParamView(s) : getfield(s, name)
"The solver's solution given its problem, its system, names and saved parameters (the outer layer, W-63)."
with_problem(s::CoreSolution{T}, prob::ODEProblemTrait{T,P}, sys, names, pt::Vector{Float64}, pu::Vector{P}) where {T,P} =
    ODESolution{T,P}(s.t, s.u, s.retcode, s.naccept, s.nreject, s.nf, prob, sys, names, pt, pu)

"""
An ODE solver: SciML's algorithms, an open set (move 3). A solver package implements it for its own
algorithm type. `solve_core(a, m, callbacks, opts)` is the solver's driver on the core's problem `m`
(its parameters partially evaluated, W-63), with the core's step callbacks. `takes_mass_matrix(a)`: whether
it solves `M du/dt = f` for a mass matrix `M` other than the identity (default: no; `solve` and `init`
refuse such a problem for it).
"""
@trait ODEAlgorithm begin
    solve_core(a::ODEAlgorithm, m::CoreProblem, callbacks::Vector{Any}, opts::SolverOptions)::CoreSolution
    takes_mass_matrix(a::ODEAlgorithm)::Bool = false
end

"""
A solver whose integrator user code can step (SciML's `init`, `step!`, `solve!`): `init_core(a, m, callbacks,
opts)` returns its integrator at the start of the span, a `CoreIntegrator{U, Float64}` and `StepView{U, Float64}`
for the state type `U`. A capability of the algorithm's type: `init` takes only these.
"""
@trait ODEStepper <: ODEAlgorithm begin
    init_core(a::ODEStepper, m::CoreProblem, callbacks::Vector{Any}, opts::SolverOptions)::Any
end

"Run a solver's integrator to the end of its span (the solvers' `solve_core` by steps)."
function run_to_end!(i)
    while advance!(i)
    end
    return core_solution(i)
end

"""
A linear solver an implicit method takes (LinearSolve.jl's algorithms: an open set inside the open set of
algorithms; stage C of move 3). `linfactor(ls, W)` factorizes the matrix of the systems `W x = b` an
implicit method solves at each step (it may overwrite `W`) and returns the factorization, a `Solvable`
(BaseTraits): `solve!(F, x, b)` solves for `x`. LinearAlgebra's factorizations are `Solvable`.
"""
@trait LinearSolver begin
    linfactor(ls::LinearSolver, W::Matrix{Float64})::Solvable{Float64}
end
"Dense LU with partial pivoting, LAPACK's: the default linear solver."
struct DenseLU end
@implement LinearSolver DenseLU begin
    linfactor(ls, W) = lu!(W)
end

"""
A nonlinear solver's state for an implicit stage of size `n`: `nlstage!(c, m, x, z, t, γdt)` solves
`x = z + γdt f(x, t)` for the core's problem `m`, from the iterate in `x`, and returns whether it converged
(OrdinaryDiffEq's NLsolve form). Made by a `NonlinearSolver`'s `nlcache`.
"""
@trait NLCache begin
    nlstage!(c::NLCache, m::CoreProblem, x::Vector{Float64}, z::Vector{Float64}, t::Float64, γdt::Float64)::Bool
end
"""
A nonlinear solver the implicit methods take (OrdinaryDiffEq's NLNewton, NLFunctional, …: an open set inside
the implicit methods, itself taking a linear solver; stage D of move 3).
"""
@trait NonlinearSolver begin
    nlcache(nl::NonlinearSolver, n::Int)::NLCache
end
"Newton's method, the problem's Jacobian at every iterate, its linear systems by `linsolve`: the default."
struct Newton{L <: LinearSolver}
    linsolve::L
    maxiters::Int
    tol::Float64
end
Newton(; linsolve = DenseLU(), maxiters = 10, tol = 1e-10) = Newton(linsolve, maxiters, tol)
struct NewtonCache
    ls::LinearSolver                 # Kept{LinearSolver}
    maxiters::Int
    tol::Float64
    J::Matrix{Float64}
    r::Vector{Float64}
    dx::Vector{Float64}
    fx::Vector{Float64}
end
"""
Newton on `x = z + γdt f(x, t)`: the residual `r = x - z - γdt f(x, t)`, the matrix `W = I - γdt J`
(factorized at every iterate, or with `frozen` only at the first), `x -= W \\ r` until `|r| ≤ tol`.
A helper for nonlinear solvers of the Newton kind, in any package.
"""
function newton_stage!(ls::LinearSolver, J::Matrix{Float64}, r::Vector{Float64}, dx::Vector{Float64}, fx::Vector{Float64},
                       maxiters::Int, tol::Float64, frozen::Bool, m::CoreProblem, x::Vector{Float64},
                       z::Vector{Float64}, t::Float64, γdt::Float64)::Bool
    n = length(x)
    local F                          # the factorization of W, a Solvable
    for k in 1:maxiters
        core_rhs!(m, fx, x, t)
        res = 0.0
        @inbounds for i in 1:n
            r[i] = x[i] - z[i] - γdt * fx[i]
            res = max(res, abs(r[i]))
        end
        res <= tol && return true
        if k == 1 || !frozen
            core_jacobian!(m, J, x, t)
            @inbounds for j in 1:n, i in 1:n
                J[i, j] = (i == j ? 1.0 : 0.0) - γdt * J[i, j]
            end
            F = linfactor(ls, J)
        end
        solve!(F, dx, r)
        @inbounds for i in 1:n
            x[i] -= dx[i]
        end
    end
    return false
end
@implement NLCache NewtonCache begin
    nlstage!(c, m, x, z, t, γdt) = newton_stage!(c.ls, c.J, c.r, c.dx, c.fx, c.maxiters, c.tol, false, m, x, z, t, γdt)
end
@implement NonlinearSolver Newton{L} where {L} begin
    nlcache(nl, n) = NewtonCache(nl.linsolve, nl.maxiters, nl.tol, zeros(n, n), zeros(n), zeros(n), zeros(n))
end

"""
Tsit5's state between steps, with its problem and the driver's state (SciML's integrator keeps both). Its type
holds the element type and the core's problem type, nothing else: the parameters stay in the outer layer
(W-63), and the callbacks are kept with their attributes (`Vector{StepCallback{…}}`), so their types do not
reach it. `uprev`, `kprev` and `tprev` are the last accepted step's start, for `interp!`.
"""
mutable struct Tsit5Integrator{T,M}
    const m::M
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
    # the driver's state between steps
    iv::Union{Nothing,Dynamic{StepView{Vector{T},Float64}}}   # the integrator as its callbacks see it
    const opts::SolverOptions
    const tf::Float64
    dtnext::Float64                  # the step to try next
    const ts::Vector{Float64}        # the saved times and states
    const us::Vector{Vector{T}}
    isave::Int                       # the next of `opts.saveat`
    iters::Int
    retcode::Symbol                  # :Default while it runs; :MaxIters
    dt_reset::Bool                   # `auto_dt_reset!`: the next step's size estimated again
end
step_tstop!(i::Tsit5Integrator, τ::Float64) = (_add_tstop!(i.tstops, i.t, τ); nothing)
step_terminate!(i::Tsit5Integrator) = (i.terminated = true; nothing)
"The state at `τ` within the last accepted step, by the cubic Hermite interpolant (as for `saveat`)."
step_interp!(i::Tsit5Integrator{T}, out::Vector{T}, τ::Float64) where {T} =
    (hermite!(out, (τ - i.tprev) / i.dt, i.dt, i.uprev, i.kprev, i.unew, i.k7); nothing)
step_dt_reset!(i::Tsit5Integrator) = (i.dt_reset = true; nothing)
@implement StepView Tsit5Integrator{T,M} where {T,M}
@implement CoreIntegrator{Vector{T},Float64} Tsit5Integrator{T,M} where {T,M} begin
    advance!(i) = tsit5_advance!(i)
    core_solution(i) = CoreSolution{T}(i.ts, i.us, _retcode(i), i.naccept, i.nreject, i.nf)
end
"A driver's return code: `:Terminated`, `:MaxIters`, `:Success` at the span's end, `:Default` while it runs."
_retcode(i) = i.terminated ? :Terminated : i.retcode !== :Default ? i.retcode : i.t >= i.tf ? :Success : :Default

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
    check_properties(prob, alg)

The front end's check of a problem's optional properties against the algorithm (open issue 119): a mass
matrix other than the identity (`HasMassMatrix`, of the problem or of its function) is refused unless the
algorithm takes one (`takes_mass_matrix`); solving the problem as an ODE would ignore it. A Jacobian
prototype is a hint, not checked: a solver that does not use one computes the same Jacobian dense.
"""
Base.@nospecializeinfer function check_properties(@nospecialize(prob), alg::ODEAlgorithm)
    M = _mass_matrix(prob)
    _is_identity(M) || takes_mass_matrix(alg) || throw(ArgumentError(string(
        "the algorithm takes no mass matrix, and this problem has one other than the identity (",
        summary(M), ", from its function's or its own `HasMassMatrix`): a DAE, which solving as an ODE would ",
        "get wrong. Use an algorithm that takes mass matrices (`takes_mass_matrix`).")))
    return nothing
end
"The callbacks of a solve: the problem's (`HasCallbacks`), then the ones given (SciML's `CallbackSet(prob.callback, callback)`)."
Base.@nospecializeinfer function _all_callbacks(@nospecialize(prob), callbacks::Vector{Any})
    pc = _problem_callbacks(prob)
    return isempty(pc) ? callbacks : Any[pc..., callbacks...]
end
_no_saved(p::P) where {P} = P[]

"""
    solve_ode(prob, alg, callbacks, opts) -> ODESolution

Solve a problem of any type that implements `ODEProblemTrait`, the outer layer (W-63): the problem's
properties checked against the algorithm (`check_properties`), its functions partially evaluated on its
parameters (`core_problem`), its callbacks and the user's (`DiscreteCallback`, `ContinuousCallback`, values
of callback types) entered at the problem's types and adapted to the core (`adapt_callbacks!`), then the
algorithm's driver (`solve_core`), which never sees the parameters' type. What depends on the parameters'
type is this function and the adapters: a few constructors and closures. With callbacks, the parameters are
saved at the start and wherever a callback calls `save_discretes!`.
"""
Base.@nospecializeinfer function solve_ode(@nospecialize(prob), alg::ODEAlgorithm, callbacks::Vector{Any}, opts::SolverOptions)
    check_properties(prob, alg)
    m = core_problem(prob)
    cbs = _all_callbacks(prob, callbacks)
    sys = _system(prob)
    names = _names(sys)
    isempty(cbs) && return with_problem(solve_core(alg, m, cbs, opts), prob, sys, names, Float64[], _no_saved(prob.p))
    pi = param_integrator(typeof(prob.u0), prob.p, prob.tspan[1], sys, names, true)
    return with_problem(solve_core(alg, m, adapt_callbacks!(pi, cbs), opts), prob, sys, names, pi.pt, pi.pu)
end

"""
    init_ode(prob, alg, callbacks, opts) -> ParamIntegrator

SciML's `init`: as `solve_ode`, but the solver's integrator at the start of the span (`init_core`, an
`ODEStepper`'s) is returned in the outer layer's integrator, for `step!` and `solve!`. The integrator's type
is `ParamIntegrator{Vector{T}, Float64, P}`, whatever the algorithm, the function and the callbacks.
"""
Base.@nospecializeinfer function init_ode(@nospecialize(prob), alg::ODEStepper, callbacks::Vector{Any}, opts::SolverOptions)
    check_properties(prob, alg)
    m = core_problem(prob)
    cbs = _all_callbacks(prob, callbacks)
    sys = _system(prob)
    pi = param_integrator(typeof(prob.u0), prob.p, prob.tspan[1], sys, _names(sys), true)
    core = init_core(alg, m, isempty(cbs) ? cbs : adapt_callbacks!(pi, cbs), opts)
    return attach!(pi, core, prob)
end
"The solver's integrator and the problem, in the outer layer's integrator."
function attach!(pi::ParamIntegrator{U,T,P}, @nospecialize(core), @nospecialize(prob)) where {U,T,P}
    pi.core = Dynamic{CoreIntegrator{U,T}}(core)
    pi.prob = prob
    _sync_in!(pi, Dynamic{StepView{U,T}}(core))
    return pi
end

"""
    step!(integ)
    step!(integ, dt, stop_at_tdt = false)

SciML's `step!` on an integrator from `init`: one accepted step, with its events, saves and callbacks; or
steps until `integ.t` reaches `integ.t + dt`, exactly when `stop_at_tdt` (a time to stop at). At the end of
the span, or once terminated, it does nothing.
"""
step!(i::ParamIntegrator) = (_advance!(i); nothing)
function step!(i::ParamIntegrator{U,T}, dt::Real, stop_at_tdt::Bool = false) where {U,T}
    dt > 0 || throw(ArgumentError("step!: dt must be positive, got $dt"))
    τ = i.t + T(dt)
    stop_at_tdt && add_tstop!(i, τ)
    while i.t < τ && _advance!(i)
    end
    return nothing
end
function _advance!(i::ParamIntegrator{U,T}) where {U,T}
    core = i.core
    core === nothing && throw(ArgumentError("step!: an integrator made by `init`"))
    sv = i.sv::Dynamic{StepView{U,T}}
    _sync_out!(i, sv)                      # the flags the user set (`u_modified`, `terminated`)
    ok = advance!(core::Dynamic{CoreIntegrator{U,T}})
    _sync_in!(i, sv)
    return ok
end
"""
    solve!(integ) -> ODESolution

SciML's `solve!`: steps the integrator from `init` to the end of its span (or until terminated) and returns
the solution, saved from the start.
"""
function solve!(i::ParamIntegrator{U,T}) where {U,T}
    while _advance!(i)
    end
    core = i.core::Dynamic{CoreIntegrator{U,T}}
    return with_problem(core_solution(core), i.prob, i.sys, i.names, i.pt, i.pu)
end

#     solve_core(Tsit5(), m::CoreProblem, callbacks, opts) -> CoreSolution
#
# Integrate the core's problem from `m.u0` over `m.tspan` with Tsit5 and a PI step-size controller;
# `callbacks` are values of `StepCallback` and `StepEvent` (the adapted user's callbacks). The element
# type `T` is the state's: `Float64`, `Float32`, or a dual number for a gradient through the solve; the
# step-size control reads primal values, so a gradient follows the primal solve's steps. Callbacks take
# `Float64` states. The driver is its integrator's `advance!`, one accepted step at a time: `solve` runs it
# to the end, `init` returns it.
@implement ODEAlgorithm Tsit5 begin
    solve_core(a, m, callbacks, opts) = run_to_end!(tsit5_init(m, m.u0, m.tspan, callbacks, opts))
end
@implement ODEStepper Tsit5 begin
    init_core(a, m, callbacks, opts) = tsit5_init(m, m.u0, m.tspan, callbacks, opts)
end

"""
Tsit5's `init`: the integrator at `T` with its problem, the core's callbacks kept (`enter_step_callbacks`), the
times to stop at, the callbacks' `initialize`, `f(u0, t0)`, the first step's size, the first save and the
continuous callbacks' conditions at the start. Its type is `Tsit5Integrator{T, M}` whatever the callbacks and the
parameters.
"""
function tsit5_init(m, u0::Vector{T}, tspan::Tuple{Float64,Float64}, callbacks::Vector{Any},
                    opts::SolverOptions = DEFAULT_OPTIONS) where {T}
    t0, tf = tspan
    n = length(u0)
    T === Float64 || isempty(callbacks) || throw(ArgumentError("callbacks take Float64 states; this solve has $T"))
    d, c = enter_step_callbacks(callbacks, Vector{T})
    z() = zeros(T, n)
    i = Tsit5Integrator{T,typeof(m)}(m, copy(u0), copy(u0), t0, 0.0, false, false,
                                     z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), 1e-4, 0, 0, 0,
                                     t0, z(), Float64[], d, c, zeros(Float64, length(c)),
                                     nothing, opts, tf, 0.0, Float64[], Vector{T}[], 1, 0, :Default, false)
    iv = Dynamic{StepView{Vector{T},Float64}}(i)
    i.iv = iv
    for τ in opts.tstops
        _add_tstop!(i.tstops, t0, τ)
    end
    isempty(callbacks) || initialize_step_callbacks!(i.callbacks, i.events, i.u, i.t, iv)
    i.u_modified = false
    core_rhs!(m, i.k1, i.u, t0)
    i.nf += 1
    i.dtnext = initdt(m, i, 1.0, opts)
    _save_start!(i.ts, i.us, i.u, t0, opts.saveat) && (i.isave = 2)
    for k in eachindex(i.events)
        i.evprev[k] = step_event_condition(i.events[k], i.u, i.t, iv)
    end
    return i
end
"The first save: the state at `t0`, unless `saveat` starts later; `true` when `saveat`'s first time is `t0`."
function _save_start!(ts::Vector{Float64}, us::Vector{Vector{T}}, u::Vector{T}, t0::Float64, saveat::Vector{Float64}) where {T}
    if isempty(saveat) || saveat[1] == t0
        push!(ts, t0); push!(us, copy(u))
    end
    return !isempty(saveat) && saveat[1] == t0
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

"""
The continuous callbacks' events within the step just accepted (`[tprev, t]`): each condition's sign change,
located by bisection on the interpolant (`event_time`); the integrator is moved to the earliest. Returns the
event's callback (`0`: none) and whether its condition rose through zero. A callback's condition just after
its own event (`NaN` in `evprev`) is read a little into the step (SciML's repeat nudge). Shared by the solvers.
"""
function locate_events!(events::Vector{Kept{StepEvent{Vector{Float64},Float64}}}, evprev::Vector{Float64},
                        iv::StepView{Vector{Float64},Float64}, u::Vector{Float64}, utilde::Vector{Float64},
                        tprev::Float64, t::Float64, dt::Float64)
    ke, te, up = 0, t, false
    for k in eachindex(events)
        g = step_event_condition(events[k], u, t, iv)
        g0 = evprev[k]
        tl = tprev
        if isnan(g0)                         # the step after this callback's event: its value a
            tl = tprev + dt / 100            # little into the step (SciML's repeat nudge)
            step_interp!(iv, utilde, tl)
            g0 = step_event_condition(events[k], utilde, tl, iv)
        end
        if g0 != 0 && (g == 0 || (g > 0) != (g0 > 0))
            tk = event_time(events[k], iv, utilde, tl, t, g0)
            if ke == 0 || tk < te
                ke, te, up = k, tk, g0 < 0
            end
        end
        evprev[k] = g
    end
    return ke, te, up
end

"The event's affect: `step_event_affect!` where the condition rose, `step_event_affect_neg!` where it fell."
event_affect_dir!(cb, iv, up::Bool) = up ? step_event_affect!(cb, iv) : step_event_affect_neg!(cb, iv)

"""
Tsit5's `advance!`: one accepted step (rejected attempts within it), its continuous callbacks' events, the saves,
then the discrete callbacks; `false` when it cannot step (the span's end, terminated, `maxiters`). A state the
user changed between steps (`u_modified`) and `auto_dt_reset!` are taken first.
"""
function tsit5_advance!(i::Tsit5Integrator{T}) where {T}
    m, opts, tf = i.m, i.opts, i.tf
    iv = i.iv::Dynamic{StepView{Vector{T},Float64}}
    events = i.events
    if i.u_modified                                  # changed between steps: f(u, t) and the conditions again
        core_rhs!(m, i.k1, i.u, i.t)
        i.nf += 1
        for k in eachindex(events)
            i.evprev[k] = step_event_condition(events[k], i.u, i.t, iv)
        end
        i.u_modified = false
    end
    if i.dt_reset                                    # `auto_dt_reset!`: the step's size from the state
        i.dtnext = initdt(m, i, 1.0, opts)
        i.dt_reset = false
    end
    n = length(i.u)
    saveat = opts.saveat
    dt = i.dtnext
    # PI controller, OrdinaryDiffEq's defaults for a 5th-order method
    beta1, beta2, gamma, qmin, qmax = 7 / 50, 2 / 25, 0.9, 0.2, 10.0
    while i.t < tf && !i.terminated
        i.iters += 1
        if i.iters > opts.maxiters
            i.retcode = :MaxIters
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
            ke, up = 0, false
            if T === Float64 && !isempty(events)
                ke, te, up = locate_events!(events, i.evprev, iv, i.u, i.utilde, i.tprev, i.t, i.dt)
                if ke != 0                                   # the integrator moves to the earliest event
                    step_interp!(iv, i.utilde, te)
                    copyto!(i.u, i.utilde)
                    i.t = te
                end
            end
            tstop_hit && ke == 0 && popfirst!(i.tstops)      # a tstop behind an event stays
            if isempty(saveat)
                push!(i.ts, i.t); push!(i.us, copy(i.u))
            else
                while i.isave <= length(saveat) && saveat[i.isave] <= i.t
                    θ = (saveat[i.isave] - i.tprev) / dt
                    push!(i.ts, saveat[i.isave])
                    push!(i.us, hermite!(zeros(T, n), θ, dt, i.uprev, i.kprev, i.unew, i.k7))
                    i.isave += 1
                end
            end
            if ke != 0
                i.u_modified = true
                event_affect_dir!(events[ke], iv, up)
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
            i.dtnext = dt
            return true
        else
            i.nreject += 1
            dt = dt / min(1 / qmin, q11 / gamma)
        end
    end
    i.dtnext = dt
    return false
end


"The value a step-size controller sees: a dual number's primal value, so that gradients follow the primal's steps."
primal(x::Float64) = x
primal(x::AbstractFloat) = Float64(x)
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))
