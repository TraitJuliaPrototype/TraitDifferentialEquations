"""
    PlainLorenzModels

`examples/LorenzModels` without traits (baseline (b)): the same problem type and callbacks as methods
of PlainODE's generic functions `rhs!`, `jacobian!`, `condition` and `affect!`, with the same
workload. Nothing checks them where they are defined; the solver specializes on their types.
"""
module PlainLorenzModels
using PlainODE
import PlainODE: rhs!, jacobian!, condition, affect!
using PrecompileTools: @compile_workload
export LorenzProblem, LorenzParams, Bounce, RaiseSigma

"The Lorenz system's parameters, at the element type `T`."
mutable struct LorenzParams{T}
    σ::T
    ρ::T
    β::T
end
"The Lorenz system as a problem type, with its own work buffer at the element type `T`."
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
    c .= u
    du[1] = p.σ * (c[2] - c[1])
    du[2] = c[1] * (p.ρ - c[3]) - c[2]
    du[3] = c[1] * c[2] - p.β * c[3]
    return nothing
end
function jacobian!(m::LorenzProblem{T}, J::Matrix{T}, u::Vector{T}, t::Float64) where {T}
    p = m.p
    J[1, 1] = -p.σ;        J[1, 2] = p.σ;   J[1, 3] = zero(T)
    J[2, 1] = p.ρ - u[3];  J[2, 2] = -one(T); J[2, 3] = -u[1]
    J[3, 1] = u[2];        J[3, 2] = u[1];  J[3, 3] = -p.β
    return nothing
end

"A wall at `x = wall`: a state past it is reflected back."
struct Bounce
    wall::Float64
end
condition(cb::Bounce, u, t, integ) = u[1] > cb.wall
affect!(cb::Bounce, integ) = (integ.u[1] = 2 * cb.wall - integ.u[1]; nothing)

"From time `at` on, the problem's σ is `σ`."
struct RaiseSigma
    at::Float64
    σ::Float64
end
condition(cb::RaiseSigma, u, t, integ) = t >= cb.at && (integ.p::LorenzParams{Float64}).σ != cb.σ
affect!(cb::RaiseSigma, integ) = ((integ.p::LorenzParams{Float64}).σ = cb.σ; nothing)

@compile_workload begin
    u0 = [1.0, 0.0, 0.0]
    prob = LorenzProblem(u0, (0.0, 0.1), 10.0, 28.0, 8 / 3)
    solve(prob, Tsit5())
    solve(prob, Tsit5(); callback = (Bounce(10.0), RaiseSigma(0.05, 12.0)))
    solve(LorenzProblem(u0, (0.0, 0.1), 10.0, 28.0, 8 / 3), Rodas5())
    solve(LorenzProblem(u0, (0.0, 0.1), 10.0, 28.0, 8 / 3), Rodas5(); callback = Bounce(10.0))
    gradient(p -> sum(solve(LorenzProblem(eltype(p).(u0), (0.0, 0.1), p[1], p[2], p[3]), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
