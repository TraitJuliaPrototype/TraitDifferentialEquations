"""
    LorenzModels

A user package of route A: a model and two callbacks written as trait-world code, conforming
to TraitDifferentialEquations' traits `ODEModel` and `Callback`. The solver is compiled for
these types in this package's image (the workload below); the user's session compiles nothing
for them. The same package without traits is `baselines/PlainLorenzModels`.

A package is an ordinary module around its trait module (`Models`): `@trait_module` is not in
scope at a package's top level, before `using TraitWorld`.
"""
module LorenzModels
using TraitWorld
using TraitDifferentialEquations
export Lorenz, Bounce, RaiseSigma

"The models and callbacks, as trait-world code."
@trait_module module Models
using TraitWorld
using TraitDifferentialEquations
import TraitDifferentialEquations: rhs!, jacobian!, condition, affect!
export Lorenz, Bounce, RaiseSigma

"""
    Lorenz(σ, ρ, β)

The Lorenz system. The parameters are fields, and the model owns its work buffer, all at the
element type `T`: `Float64`, `Float32`, or dual numbers when a loss builds the model from its
parameters under `gradient`. Mutable, so that a callback can change a parameter.
"""
mutable struct Lorenz{T}
    σ::T
    ρ::T
    β::T
    const cache::Vector{T}
end
Lorenz(σ::T, ρ::T, β::T) where {T} = Lorenz{T}(σ, ρ, β, zeros(T, 3))
Lorenz(σ, ρ, β) = Lorenz(promote(σ, ρ, β)...)

function rhs!(m::Lorenz{T}, du::Vector{T}, u::Vector{T}, t::Float64) where {T}
    c = m.cache
    c .= u                                   # the buffer is at T: it holds duals under a gradient
    du[1] = m.σ * (c[2] - c[1])
    du[2] = c[1] * (m.ρ - c[3]) - c[2]
    du[3] = c[1] * c[2] - m.β * c[3]
    return nothing
end

"The Jacobian in `u`, by hand: Rodas5 calls it instead of the trait's default (finite differences)."
function jacobian!(m::Lorenz{T}, J::Matrix{T}, u::Vector{T}, t::Float64) where {T}
    J[1, 1] = -m.σ;        J[1, 2] = m.σ;   J[1, 3] = zero(T)
    J[2, 1] = m.ρ - u[3];  J[2, 2] = -one(T); J[2, 3] = -u[1]
    J[3, 1] = u[2];        J[3, 2] = u[1];  J[3, 3] = -m.β
    return nothing
end
@implement ODEModel{T} Lorenz{T} where {T}

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

From time `at` on, the model's σ is `σ` (a callback that writes a parameter: it opens the
integrator's `p`, the model, at the type it expects, `Lorenz{Float64}`).
"""
struct RaiseSigma
    at::Float64
    σ::Float64
end
condition(cb::RaiseSigma, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView}) =
    t >= cb.at && (integ.p::Lorenz{Float64}).σ != cb.σ
function affect!(cb::RaiseSigma, integ::Dynamic{IntegratorView})
    m = integ.p::Lorenz{Float64}
    m.σ = cb.σ                               # the state is unchanged, f(u, t) is not: the driver evaluates it again
    return nothing
end
@implement Callback RaiseSigma

end
using .Models

TraitWorld.@workload begin
    u0 = [1.0, 0.0, 0.0]
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Tsit5())
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Tsit5(); callback = (Bounce(10.0), RaiseSigma(0.05, 12.0)))
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Rodas5())
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Rodas5(); callback = Bounce(10.0))
    # gradients through Tsit5 for three parameters (chunk 3): every loss of this signature runs this code
    gradient(p -> sum(solve(Lorenz(p[1], p[2], p[3]), eltype(p).(u0), (0.0, 0.1), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
