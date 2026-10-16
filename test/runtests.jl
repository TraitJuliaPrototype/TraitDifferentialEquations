using Test, TraitDifferentialEquations, TraitWorld

lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
const P0 = [10.0, 28.0, 8 / 3]
const U0 = [1.0, 0.0, 0.0]
# u(1) of Lorenz from these values, by stock OrdinaryDiffEq's Tsit5 at abstol = reltol = 1e-12
const REF = [-9.408450567056128, -9.096199071186515, 28.581627624392613]

# a model as trait-world code: the type implements `rhs!` (route A, compiled for the type)
struct Lorenz
    σ::Float64; ρ::Float64; β::Float64
end
TraitDifferentialEquations.rhs!(m::Lorenz, du::Vector{Float64}, u::Vector{Float64}, t::Float64) =
    (du[1] = m.σ * (u[2] - u[1]); du[2] = u[1] * (m.ρ - u[3]) - u[2]; du[3] = u[1] * u[2] - m.β * u[3]; nothing)
@implement ODEModel{Float64} Lorenz

@testset "TraitDifferentialEquations: Tsit5" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    sol = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
    @test sol.retcode === :Success
    @test isapprox(sol.u[end], REF; rtol = 1e-10)
    s = solve(prob, Tsit5())
    @test isapprox(s.u[end], REF; rtol = 1e-2) && 20 <= length(s.t) <= 40
    s2 = solve(prob, Tsit5(); saveat = 0.0:0.25:1.0)
    @test s2.t == collect(0.0:0.25:1.0) && isapprox(s2.u[end], s.u[end]; rtol = 1e-8)
    # route A: the same model as a type that implements rhs!
    sA = solve(Lorenz(10.0, 28.0, 8 / 3), U0, (0.0, 1.0), Tsit5(); abstol = 1e-12, reltol = 1e-12)
    @test isapprox(sA.u[end], REF; rtol = 1e-10)
    # a parameter-free problem
    s0 = solve(ODEProblem((du, u, p, t) -> (du .= -u; nothing), [1.0], (0.0, 1.0)), Tsit5(); abstol = 1e-10, reltol = 1e-10)
    @test isapprox(s0.u[end][1], exp(-1); rtol = 1e-8)
end

@testset "TraitDifferentialEquations: callbacks" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), copy(P0))
    kick = DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.6, i -> (i.u[1] += 1.0; nothing))
    s = solve(prob, Tsit5(); callback = kick)
    @test s.retcode === :Success && s.u[end] != solve(prob, Tsit5()).u[end]
    stop = DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.terminated = true; nothing))
    s2 = solve(prob, Tsit5(); callback = stop)
    @test s2.retcode === :Terminated && s2.t[end] < 1.0 && s2.u[end][1] > 5.0
    # a callback that changes the parameters, which the integrator holds (the user's vector)
    p = copy(P0)
    slow = DiscreteCallback((u, t, i) -> t > 0.5, i -> (i.p[1] = 1.0; nothing))
    s3 = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); callback = slow)
    @test p[1] == 1.0 && s3.u[end] != s.u[end]
    # two callbacks at once, and their types do not reach the integrator's type
    s4 = solve(prob, Tsit5(); callback = (kick, stop))
    @test s4.retcode === :Terminated
    I = TraitDifferentialEquations.ODECore.Tsit5Integrator{Float64,Vector{Float64}}
    @test length(string(nameof(I))) < 100
end
