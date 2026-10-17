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
import ForwardDiff
include("ODECore.jl")
using .ODECore
using .ODECore: RHSSig, DualSig, CondSig, AffectSig, FunctionModel, FunctionModelAD, FunctionCallback, keep_callback,
                solve_ode, SolverOptions, FDual
export ODEProblem, solve, gradient, Tsit5, Rodas5, DiscreteCallback, ODESolution, ODEModel, rhs!, jacobian!, Callback, condition,
       affect!, IntegratorView

"A problem in SciML's form: `f(du, u, p, t)` in place, an initial state, a time span, parameters."
struct ODEProblem{F,U,P}
    f::F
    u0::Vector{U}
    tspan::Tuple{Float64,Float64}
    p::P
end
ODEProblem(f, u0::AbstractVector, tspan, p = nothing) =
    ODEProblem(f, _state(u0), (Float64(tspan[1]), Float64(tspan[2])), p)
_state(u0::AbstractVector{<:ForwardDiff.Dual}) = Vector(u0)
_state(u0::AbstractVector) = Vector{Float64}(u0)

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
    u0 = _promote_state(prob.u0, prob.p)
    return _solve_fm(_function_model(prob.f, prob.p, u0), u0, prob.tspan, alg, prob.p, _kept(callback), _options(; kw...))
end
_function_model(@nospecialize(f), p::P, ::Vector{T}) where {P,T} = FunctionModel{P,T}(dynamic_fn(RHSSig{P,T}, Nothing, f), p)
"The state at the parameters' dual type, as SciML promotes `u0` for a gradient in `p`."
_promote_state(u0::Vector{Float64}, p::AbstractArray{D}) where {D<:ForwardDiff.Dual} = Vector{D}(u0)
_promote_state(u0::Vector, p) = u0

"""
    solve(prob::ODEProblem, Rodas5(); callback, saveat, abstol, reltol)

The plain function becomes two dynamic calls: at the plain signature, and at the library's dual
numbers (ForwardDiff's `Dual` with the library's tag, `FDual{N}`) for the Jacobian (`N`, the chunk, is the state's length up to 8). The solver's
code is compiled per `(P, N)`; a new function compiles only itself, at both signatures.
"""
function solve(@nospecialize(prob::ODEProblem), alg::Rodas5; callback = nothing, kw...)
    N = min(length(prob.u0), 8)
    return _solve_ad(_function_model_ad(prob.f, prob.p, Val(N)), prob.u0, prob.tspan, alg, prob.p, _kept(callback), _options(; kw...))
end
_function_model_ad(@nospecialize(f), p::P, ::Val{N}) where {P,N} =
    FunctionModelAD{P,N}(dynamic_fn(RHSSig{P,Float64}, Nothing, f), dynamic_fn(DualSig{P,N}, Nothing, f), p)
_solve_ad(m::FunctionModelAD{P,N}, u0, tspan, alg, p::P, cbs, opts) where {P,N} = solve_ode(m, u0, tspan, alg, p, cbs, opts)
_solve_fm(m::FunctionModel{P,T}, u0::Vector{T}, tspan, alg, p::P, cbs, opts) where {P,T} = solve_ode(m, u0, tspan, alg, p, cbs, opts)

# --- gradients through `solve` (row 5) --------------------------------------------------------------

"A function's primal signature, as an automatic-differentiation tag: arguments `A`, result `S`."
struct Sig{A,S} end

"Whether a value's type holds a dual number anywhere: a closure that captured an outer level's duals."
_holds_duals(@nospecialize(T)) = T <: ForwardDiff.Dual ||
    (T isa DataType && (any(_holds_duals, T.parameters) || (!isabstracttype(T) && any(_holds_duals, fieldtypes(T)))))
_holds_duals(@nospecialize(T::TypeVar)) = false

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
    f = dynamic_fn(Tuple{Vector{D}}, D, loss)
    cfg = ForwardDiff.GradientConfig(f, p, ForwardDiff.Chunk{N}(), Tg())
    return ForwardDiff.gradient(f, p, cfg, Val(false))      # the tag check off: the tag is the signature's
end

"""
    solve(model, u0, tspan, Tsit5(); p = nothing, callback = nothing, kw...)

A model of a type that conforms to `ODEModel` (it implements `rhs!` itself): the solver's
code is compiled for that type, in the conformer's image or the session.
"""
solve(model, u0::AbstractVector, tspan, alg::Union{Tsit5,Rodas5}; p = nothing, callback = nothing, kw...) =
    solve_ode(model, Vector{Float64}(u0), (Float64(tspan[1]), Float64(tspan[2])), alg, p, _kept(callback), _options(; kw...))

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
    solve(ODEProblem(_wl_rhs0!, [1.0], (0.0, 0.1)), Tsit5())
    solve(prob, Rodas5())
    solve(ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1)), Rodas5())   # no parameters (p::Nothing)
    solve(prob, Rodas5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    # gradients through Tsit5 for three parameters (chunk 3): every loss of this signature runs this code
    gradient(p -> sum(solve(ODEProblem(_wl_rhsp!, [1.0, 0.0, 0.0], (0.0, 0.1), p), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
