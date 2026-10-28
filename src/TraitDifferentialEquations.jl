"""
    TraitDifferentialEquations

ODE solvers written as a trait module (`ODECore`), with an ordinary layer that offers
SciML's interface: `ODEFunction(f; jac)`, `ODEProblem(f, u0, tspan, p)`, `solve(prob, Tsit5(); callback, saveat)`,
`DiscreteCallback(condition, affect!)`, `ContinuousCallback(condition, affect!)`. A problem is any type that implements `ODEProblemTrait`;
TDE's own `ODEProblem` is the canonical one. The first demonstrator of the trait world's
viability; see README.md for the measurements.

The umbrella over the core (`ODECore`, a package of its own) and the solver packages, as
DifferentialEquations.jl is: it re-exports them, keeps callbacks and options, and calls into `ODECore`.
"""
module TraitDifferentialEquations
using TraitWorld
import ForwardDiff
import ODECore                       # the core, a package of its own (move 3): `TraitDifferentialEquations.ODECore`
using ODECore
using TraitRodas                     # a solver package: `Rodas5`
using TraitIndexing                  # symbolic access (move 4)
using ODECore: RHSSig, DualSig, JacSig, DUAL_CHUNK, CondSig, AffectSig, FunctionCallback, solve_ode, SolverOptions, FDual,
                no_jacobian, UserFunction, ODEAlgorithm, solve_core, getseries
export ODEFunction, ODEProblem, solve, gradient, Tsit5, Rodas5, ODEAlgorithm, solve_core,
       LinearSolver, linfactor, DenseLU,
       SymbolCache, IndexProvider, ValueProvider, variable_index, parameter_index, getstate, getparam, getseries, DiscreteCallback, ContinuousCallback, ODESolution, ODEProblemTrait, rhs!, jacobian!,
       rhs_function, Callback, condition, affect!, IntegratorView, Trajectory, EventCallback, event_condition, event_affect!,
       add_tstop!, terminate!, interp!

# the user's callbacks as given (`DiscreteCallback`, `ContinuousCallback`, values of callback types):
# a solver's `init` enters them at the problem's types
Base.@nospecializeinfer _cbs(@nospecialize(cb)) = cb === nothing ? Any[] : Any[cb]
Base.@nospecializeinfer _cbs(@nospecialize(cbs::Union{Tuple,AbstractVector})) = Any[cb for cb in cbs]

_options(; abstol = 1e-6, reltol = 1e-3, dtmax = Inf, maxiters = 100_000, saveat = Float64[]) =
    SolverOptions(Float64(abstol), Float64(reltol), Float64(dtmax), maxiters, Vector{Float64}(collect(saveat)))

"""
    solve(prob, Tsit5(); callback = nothing, saveat = [], abstol, reltol, dtmax, maxiters)
    solve(prob, Rodas5(); …)

Solve a problem of any type that implements `ODEProblemTrait`: TDE's `ODEProblem`, or a problem
type of its own (`@implement ODEProblemTrait MyProblem`), whose solver code is compiled for that
type. Rodas5 takes `Float64` states and the problem's `jacobian!`, or with `Rodas5(chunk_size = 0)` the problem's
function at duals of a chunk picked from the state's length.

This front end is neither specialized nor inferred on the problem's type, so a caller's code does
not infer the solver again; behind it, a function barrier (`_solve`) is compiled per problem type,
which calls the solver at that type.
"""
Base.@nospecializeinfer solve(@nospecialize(prob), @nospecialize(alg); @nospecialize(callback = nothing), kw...) =   # any `ODEAlgorithm` (move 3)
    _solve(prob, alg, _cbs(callback), _options(; kw...))
_solve(prob, alg, cbs, opts) = solve_ode(prob, alg, cbs, opts)
"The default algorithm (move 3, check 4): the umbrella's own `solve`, Tsit5."
Base.@nospecializeinfer solve(@nospecialize(prob); kw...) = solve(prob, Tsit5(); kw...)

# --- gradients through `solve` (row 5) --------------------------------------------------------------

"A function's primal signature, as an automatic-differentiation tag: arguments `A`, result `S`."
struct Sig{A,S} end

"Whether a value's type holds a dual number anywhere: a closure that captured an outer level's duals."
_holds_duals(@nospecialize(T::Type)) = T <: ForwardDiff.Dual ||
    (T isa Union && any(_holds_duals, Base.uniontypes(T))) ||
    (T isa DataType && (any(_holds_duals, T.parameters) || (!isabstracttype(T) && any(_holds_duals, fieldtypes(T)))))
_holds_duals(@nospecialize(x)) = false                  # a type variable, or a parameter that is a value (`1` in `Array{T, 1}`)

"""
    gradient(loss, p::Vector{Float64}) -> Vector{Float64}

The gradient of `loss(p)::Real` by ForwardDiff, for losses that solve ODEs. The duals' tag is the
loss's primal signature, `Tag{Sig{Tuple{Vector{Float64}}, Float64}, Float64}`, not the loss's type,
so every loss of that signature runs the same solver code: a second loss compiles only itself
(the decision of session 23). A loss that captured dual numbers (nested differentiation) gets
ForwardDiff's own tag, its type, so that the levels cannot be confused.
"""
function gradient(@nospecialize(loss), p::Vector{Float64})
    N = ForwardDiff.pickchunksize(length(p))
    if _holds_duals(typeof(loss))
        return ForwardDiff.gradient(loss, p, ForwardDiff.GradientConfig(loss, p, ForwardDiff.Chunk{N}()))
    end
    return _gradient_sig(loss, p, Val(N))
end
function _gradient_sig(@nospecialize(loss), p::Vector{Float64}, ::Val{N}) where {N}
    Tg = ForwardDiff.Tag{Sig{Tuple{Vector{Float64}},Float64},Float64}
    D = ForwardDiff.Dual{Tg,Float64,N}
    # the loss as a dynamic call at its dual signature: ForwardDiff's driver compiles once per (N, signature)
    f = Dynamic{Callable{Tuple{Vector{D}},D}}(loss)
    cfg = ForwardDiff.GradientConfig(f, p, ForwardDiff.Chunk{N}(), Tg())
    return ForwardDiff.gradient(f, p, cfg, Val(false))      # the tag check off: the tag is the signature's
end

# The precompile workload: the solver for plain functions with `Vector{Float64}` and no parameters,
# with and without callbacks. Any other function of these signatures runs this code: only the
# function itself is compiled, at its declared signature, when the problem is solved.
_wl_rhs!(du, u, p, t) = (du[1] = 10.0 * (u[2] - u[1]); du[2] = u[1] * (28.0 - u[3]) - u[2]; du[3] = u[1] * u[2] - 8 / 3 * u[3]; nothing)
_wl_rhs0!(du, u, p, t) = (du .= -u; nothing)
_wl_rhsp!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
TraitWorld.@workload begin
    prob = ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1), [10.0, 28.0, 8 / 3])
    solve(prob, Tsit5())
    solve(prob, Tsit5(); saveat = [0.05, 0.1])
    solve(prob, Tsit5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    # the integrator's writable fields, as callbacks write them
    solve(prob, Tsit5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.terminated = true; nothing)))
    solve(prob, Tsit5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u_modified = false; nothing)))
    # the view's operations, and a continuous callback (its bisection on interp!)
    solve(prob, Tsit5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (add_tstop!(i, 0.08); terminate!(i); nothing)))
    solve(prob, Tsit5(); callback = ContinuousCallback((u, t, i) -> u[1] - 1.5, i -> (i.u[1] = 1.0; nothing)))
    solve(ODEProblem(_wl_rhs0!, [1.0], (0.0, 0.1)), Tsit5())
    solve(prob, Rodas5())
    solve(ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1)), Rodas5())   # no parameters (p::Nothing)
    solve(prob, Rodas5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    solve(prob, Rodas5(chunk_size = 0))                                    # the chunk picked at init (3)
    # gradients through Tsit5 for three parameters (chunk 3): every loss of this signature runs this code
    gradient(p -> sum(solve(ODEProblem(_wl_rhsp!, [1.0, 0.0, 0.0], (0.0, 0.1), p), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
