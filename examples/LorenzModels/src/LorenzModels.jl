"""
    LorenzModels

A user package: a problem type and two callbacks written as trait-world code, conforming to
TraitDifferentialEquations' traits `ODEProblemTrait` and `Callback`. The solver is compiled for
these types in this package's image (the workload below); the user's session compiles nothing
for them. The same package without traits is `baselines/PlainLorenzModels`.

A package is an ordinary module around its trait module (`Models`): `@trait_module` is not in
scope at a package's top level, before `using TraitWorld`.
"""
module LorenzModels
using TraitWorld
using TraitDifferentialEquations
export LorenzProblem, LorenzParams, Bounce, RaiseSigma

"The problem types and callbacks, as trait-world code."
@trait_module module Models
using TraitWorld
using TraitDifferentialEquations
import TraitDifferentialEquations: rhs!, jacobian!, condition, affect!
export LorenzProblem, LorenzParams, Bounce, RaiseSigma

"The Lorenz system's parameters, at the element type `T`. Mutable, so that a callback can change one."
mutable struct LorenzParams{T}
    σ::T
    ρ::T
    β::T
end

"""
    LorenzProblem(u0, tspan, σ, ρ, β)

The Lorenz system as a problem type: its initial state, time span and parameters, and its own work
buffer, all at the element type `T`: `Float64`, `Float32`, or dual numbers when a loss builds the
problem from its parameters under `gradient`.
"""
struct LorenzProblem{T}
    u0::Vector{T}
    tspan::Tuple{Float64,Float64}
    p::LorenzParams{T}
    cache::Vector{T}
end
LorenzProblem(u0::Vector{T}, tspan, σ, ρ, β) where {T} =
    LorenzProblem{T}(u0, (Float64(tspan[1]), Float64(tspan[2])), LorenzParams{T}(σ, ρ, β), zeros(T, 3))

function rhs!(m::LorenzProblem{T}, du::Vector{T}, u::Vector{T}, t::Float64) where {T}
    c, p = m.cache, m.p
    c .= u                                   # the buffer is at T: it holds duals under a gradient
    du[1] = p.σ * (c[2] - c[1])
    du[2] = c[1] * (p.ρ - c[3]) - c[2]
    du[3] = c[1] * c[2] - p.β * c[3]
    return nothing
end

"The Jacobian in `u`, by hand: Rodas5 calls it instead of the trait's default (finite differences)."
function jacobian!(m::LorenzProblem{T}, J::Matrix{T}, u::Vector{T}, t::Float64) where {T}
    p = m.p
    J[1, 1] = -p.σ;        J[1, 2] = p.σ;   J[1, 3] = zero(T)
    J[2, 1] = p.ρ - u[3];  J[2, 2] = -one(T); J[2, 3] = -u[1]
    J[3, 1] = u[2];        J[3, 2] = u[1];  J[3, 3] = -p.β
    return nothing
end
@implement ODEProblemTrait{T, LorenzParams{T}} LorenzProblem{T} where {T}

"""
    Bounce(wall)

A wall at `x = wall`: a state past it is reflected back (a callback that writes the state).
"""
struct Bounce
    wall::Float64
end
condition(cb::Bounce, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView}) = u[1] > cb.wall
function affect!(cb::Bounce, integ::Dynamic{IntegratorView})
    u = integ.u
    u[1] = 2 * cb.wall - u[1]
    return nothing
end
@implement Callback Bounce

"""
    RaiseSigma(at, σ)

From time `at` on, the problem's σ is `σ` (a callback that writes a parameter: it opens the
integrator's `p`, the problem's parameters, at the type it expects, `LorenzParams{Float64}`).
"""
struct RaiseSigma
    at::Float64
    σ::Float64
end
condition(cb::RaiseSigma, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView}) =
    t >= cb.at && (integ.p::LorenzParams{Float64}).σ != cb.σ
function affect!(cb::RaiseSigma, integ::Dynamic{IntegratorView})
    p = integ.p::LorenzParams{Float64}
    p.σ = cb.σ                               # the state is unchanged, f(u, t) is not: the driver evaluates it again
    return nothing
end
@implement Callback RaiseSigma

end
using .Models

TraitWorld.@workload begin
    u0 = [1.0, 0.0, 0.0]
    prob = LorenzProblem(u0, (0.0, 0.1), 10.0, 28.0, 8 / 3)
    solve(prob, Tsit5())
    solve(prob, Tsit5(); callback = (Bounce(10.0), RaiseSigma(0.05, 12.0)))
    solve(LorenzProblem(u0, (0.0, 0.1), 10.0, 28.0, 8 / 3), Rodas5())
    solve(LorenzProblem(u0, (0.0, 0.1), 10.0, 28.0, 8 / 3), Rodas5(); callback = Bounce(10.0))
    # gradients through Tsit5 for three parameters (chunk 3): every loss of this signature runs this code
    gradient(p -> sum(solve(LorenzProblem(eltype(p).(u0), (0.0, 0.1), p[1], p[2], p[3]), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
