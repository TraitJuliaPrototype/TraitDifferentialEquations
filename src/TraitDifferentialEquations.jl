"""
    TraitDifferentialEquations

ODE solvers written as a trait module (`ODECore`), with an ordinary layer that offers
SciML's interface: `ODEFunction(f; jac)`, `ODEProblem(f, u0, tspan, p)`, `solve(prob, Tsit5(); callback, saveat)`,
`DiscreteCallback(condition, affect!)`. A problem is any type that implements `ODEProblemTrait`;
TDE's own `ODEProblem` is the canonical one. The first demonstrator of the trait world's
viability; see README.md for the measurements.

A package pairs a trait module with an ordinary module that uses both (rule 1): this
module is the ordinary one. It makes problems from plain functions, keeps callbacks, and
calls into `ODECore`.
"""
module TraitDifferentialEquations
using TraitWorld
import ForwardDiff
using LinearAlgebra: I, UniformScaling
include("ODECore.jl")
using .ODECore
using .ODECore: RHSSig, DualSig, JacSig, DUAL_CHUNK, CondSig, AffectSig, FunctionCallback, keep_callback, solve_ode, SolverOptions, FDual,
                no_jacobian
export ODEFunction, ODEProblem, solve, gradient, Tsit5, Rodas5, DiscreteCallback, ODESolution, ODEProblemTrait, rhs!, jacobian!,
       rhs_function, Callback, condition, affect!, IntegratorView, Trajectory

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
end
ODEFunction(f; jac = nothing, mass_matrix = I) = ODEFunction(f, jac, mass_matrix)

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
Base.@nospecializeinfer function ODECore.ODEProblem(@nospecialize(f), u0::AbstractVector, tspan, p = nothing)
    u = _promote_state(_state(u0), p)
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    if f isa ODEFunction
        f.mass_matrix isa UniformScaling && f.mass_matrix == I ||
            throw(ArgumentError("TraitDifferentialEquations' solvers take the identity mass matrix only"))
        return _problem(f.f, f.jac, u, ts, p)
    end
    return _problem(f, nothing, u, ts, p)
end
# the function at the problem's bounds: its attribute decides its form at each (traits reference §7.4);
# the function as given beside them, for a solver that enters it at its own chunk
_problem(@nospecialize(f), @nospecialize(jac), u0::Vector, tspan::Tuple{Float64,Float64}, p) =
    jac === nothing ? ODECore.ODEProblem(f, u0, tspan, p, f, compiled_for(no_jacobian), f) :
                      ODECore.ODEProblem(f, u0, tspan, p, f, jac, f)
_state(u0::AbstractVector{<:ForwardDiff.Dual}) = Vector(u0)
_state(u0::AbstractVector{T}) where {T<:AbstractFloat} = Vector{T}(u0)      # Float32 states stay Float32 (row 6)
_state(u0::AbstractVector) = Vector{Float64}(u0)
"The state at the parameters' dual type, as SciML promotes `u0` for a gradient in `p`."
_promote_state(u0::Vector{Float64}, p::AbstractArray{D}) where {D<:ForwardDiff.Dual} = Vector{D}(u0)
_promote_state(u0::Vector, p) = u0

"A discrete callback from two plain functions: `condition(u, t, integ)::Bool` and `affect!(integ)`."
DiscreteCallback(condition, affect!) =
    FunctionCallback(Kept{Callable{CondSig,Bool}}(condition), Kept{Callable{AffectSig,Nothing}}(affect!))

_kept(::Nothing) = Kept{Callback}[]
_kept(cb) = Kept{Callback}[keep_callback(cb)]
_kept(cbs::Union{Tuple,AbstractVector}) = Kept{Callback}[keep_callback(cb) for cb in cbs]

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
Base.@nospecializeinfer solve(@nospecialize(prob), alg::Union{Tsit5,Rodas5}; callback = nothing, kw...) =
    _solve(prob, alg, _kept(callback), _options(; kw...))
_solve(prob, alg, cbs, opts) = solve_ode(prob, alg, cbs, opts)

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
    solve(ODEProblem(_wl_rhs0!, [1.0], (0.0, 0.1)), Tsit5())
    solve(prob, Rodas5())
    solve(ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1)), Rodas5())   # no parameters (p::Nothing)
    solve(prob, Rodas5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    solve(prob, Rodas5(chunk_size = 0))                                    # the chunk picked at init (3)
    # gradients through Tsit5 for three parameters (chunk 3): every loss of this signature runs this code
    gradient(p -> sum(solve(ODEProblem(_wl_rhsp!, [1.0, 0.0, 0.0], (0.0, 0.1), p), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
