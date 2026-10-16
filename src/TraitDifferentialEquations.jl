"""
    TraitDifferentialEquations

ODE solvers written as a trait module (`ODECore`), with an ordinary layer that offers
SciML's interface: `ODEProblem(f, u0, tspan, p)`, `solve(prob, Tsit5(); callback, saveat)`,
`DiscreteCallback(condition, affect!)`. The first demonstrator of the trait world's
viability; see README.md for the measurements.

A package pairs a trait module with an ordinary module that uses both (rule 1): this
module is the ordinary one. It turns plain functions into dynamic calls at the declared
signatures (the boundary's translation for a `Callable` in `DynamicCall` mode) and
calls into `ODECore`.
"""
module TraitDifferentialEquations
using TraitWorld
include("ODECore.jl")
using .ODECore
using .ODECore: RHSSig, CondSig, AffectSig, FunctionModel, FunctionCallback, keep_callback, solve_ode, SolverOptions
export ODEProblem, solve, Tsit5, DiscreteCallback, ODESolution, ODEModel, rhs!, Callback, condition, affect!, IntegratorView

"A problem in SciML's form: `f(du, u, p, t)` in place, an initial state, a time span, parameters."
struct ODEProblem{F,P}
    f::F
    u0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    p::P
end
ODEProblem(f, u0::AbstractVector, tspan, p = nothing) =
    ODEProblem(f, Vector{Float64}(u0), (Float64(tspan[1]), Float64(tspan[2])), p)

"A discrete callback from two plain functions: `condition(u, t, integ)::Bool` and `affect!(integ)`."
DiscreteCallback(condition, affect!) =
    FunctionCallback(dynamic_fn(CondSig, Bool, condition), dynamic_fn(AffectSig, Nothing, affect!))

_kept(::Nothing) = Dynamic{Callback}[]
_kept(cb) = Dynamic{Callback}[keep_callback(cb)]
_kept(cbs::Union{Tuple,AbstractVector}) = Dynamic{Callback}[keep_callback(cb) for cb in cbs]

_options(; abstol = 1e-6, reltol = 1e-3, dtmax = Inf, maxiters = 100_000, saveat = Float64[]) =
    SolverOptions(Float64(abstol), Float64(reltol), Float64(dtmax), maxiters, Vector{Float64}(collect(saveat)))

"""
    solve(prob::ODEProblem, Tsit5(); callback = nothing, saveat = [], abstol, reltol)

The plain function becomes a dynamic call at `(Vector{Float64}, Vector{Float64}, P, Float64)`:
the solver's code is compiled once per parameter type `P`, never per function.
"""
function solve(@nospecialize(prob::ODEProblem), alg::Tsit5; callback = nothing, kw...)
    # not specialized on the function's type: only the function itself is compiled, at its signature
    return _solve_fm(_function_model(prob.f, prob.p), prob.u0, prob.tspan, alg, prob.p, _kept(callback), _options(; kw...))
end
_function_model(@nospecialize(f), p::P) where {P} = FunctionModel{P}(dynamic_fn(RHSSig{P}, Nothing, f), p)
_solve_fm(m::FunctionModel{P}, u0, tspan, alg, p::P, cbs, opts) where {P} = solve_ode(m, u0, tspan, alg, p, cbs, opts)

"""
    solve(model, u0, tspan, Tsit5(); p = nothing, callback = nothing, kw...)

A model of a type that conforms to `ODEModel` (it implements `rhs!` itself): the solver's
code is compiled for that type, in the conformer's image or the session.
"""
solve(model, u0::AbstractVector, tspan, alg::Tsit5; p = nothing, callback = nothing, kw...) =
    solve_ode(model, Vector{Float64}(u0), (Float64(tspan[1]), Float64(tspan[2])), alg, p, _kept(callback), _options(; kw...))

# The precompile workload: the solver for plain functions with `Vector{Float64}` and no parameters,
# with and without callbacks. Any other function of these signatures runs this code: only the
# function itself is compiled, at its declared signature, when the problem is solved.
_wl_rhs!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
_wl_rhs0!(du, u, p, t) = (du .= -u; nothing)
TraitWorld.@workload begin
    prob = ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1), [10.0, 28.0, 8 / 3])
    solve(prob, Tsit5())
    solve(prob, Tsit5(); saveat = [0.05, 0.1])
    solve(prob, Tsit5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    solve(ODEProblem(_wl_rhs0!, [1.0], (0.0, 0.1)), Tsit5())
end

end
