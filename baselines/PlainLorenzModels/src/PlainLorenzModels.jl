"""
    PlainLorenzModels

`examples/LorenzModels` without traits (baseline (b)): the same model and callbacks as methods
of PlainODE's generic functions `rhs!`, `jacobian!`, `condition` and `affect!`, with the same
workload. Nothing checks them where they are defined; the solver specializes on their types.
"""
module PlainLorenzModels
using PlainODE
import PlainODE: rhs!, jacobian!, condition, affect!
using PrecompileTools: @compile_workload
export Lorenz, Bounce, RaiseSigma

"The Lorenz system with its parameters and its own work buffer at the element type `T`."
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
    c .= u
    du[1] = m.σ * (c[2] - c[1])
    du[2] = c[1] * (m.ρ - c[3]) - c[2]
    du[3] = c[1] * c[2] - m.β * c[3]
    return nothing
end
function jacobian!(m::Lorenz{T}, J::Matrix{T}, u::Vector{T}, t::Float64) where {T}
    J[1, 1] = -m.σ;        J[1, 2] = m.σ;   J[1, 3] = zero(T)
    J[2, 1] = m.ρ - u[3];  J[2, 2] = -one(T); J[2, 3] = -u[1]
    J[3, 1] = u[2];        J[3, 2] = u[1];  J[3, 3] = -m.β
    return nothing
end

"A wall at `x = wall`: a state past it is reflected back."
struct Bounce
    wall::Float64
end
condition(cb::Bounce, u, t, integ) = u[1] > cb.wall
affect!(cb::Bounce, integ) = (integ.u[1] = 2 * cb.wall - integ.u[1]; nothing)

"From time `at` on, the model's σ is `σ`."
struct RaiseSigma
    at::Float64
    σ::Float64
end
condition(cb::RaiseSigma, u, t, integ) = t >= cb.at && (integ.p::Lorenz{Float64}).σ != cb.σ
affect!(cb::RaiseSigma, integ) = ((integ.p::Lorenz{Float64}).σ = cb.σ; nothing)

@compile_workload begin
    u0 = [1.0, 0.0, 0.0]
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Tsit5())
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Tsit5(); callback = (Bounce(10.0), RaiseSigma(0.05, 12.0)))
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Rodas5())
    solve(Lorenz(10.0, 28.0, 8 / 3), u0, (0.0, 0.1), Rodas5(); callback = Bounce(10.0))
    gradient(p -> sum(solve(Lorenz(p[1], p[2], p[3]), eltype(p).(u0), (0.0, 0.1), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
