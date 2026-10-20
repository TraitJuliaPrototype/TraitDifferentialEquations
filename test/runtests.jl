using Test, TraitDifferentialEquations, TraitWorld
const ForwardDiff = TraitDifferentialEquations.ForwardDiff

lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
const P0 = [10.0, 28.0, 8 / 3]
const U0 = [1.0, 0.0, 0.0]
# u(1) of Lorenz from these values, by stock OrdinaryDiffEq's Tsit5 at abstol = reltol = 1e-12
const REF = [-9.408450567056128, -9.096199071186515, 28.581627624392613]

# a problem type of its own: it implements `rhs!` (compiled for the type)
struct Lorenz3
    u0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    p::Vector{Float64}
end
TraitDifferentialEquations.rhs!(m::Lorenz3, du::Vector{Float64}, u::Vector{Float64}, t::Float64) =
    (du[1] = m.p[1] * (u[2] - u[1]); du[2] = u[1] * (m.p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - m.p[3] * u[3]; nothing)
@implement ODEProblemTrait Lorenz3

@testset "TraitDifferentialEquations: Tsit5" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    sol = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
    @test sol.retcode === :Success
    @test isapprox(sol.u[end], REF; rtol = 1e-10)
    s = solve(prob, Tsit5())
    @test isapprox(s.u[end], REF; rtol = 1e-2) && 20 <= length(s.t) <= 40
    s2 = solve(prob, Tsit5(); saveat = 0.0:0.25:1.0)
    @test s2.t == collect(0.0:0.25:1.0) && isapprox(s2.u[end], s.u[end]; rtol = 1e-8)
    # the same system as a problem type that implements rhs!
    sA = solve(Lorenz3(U0, (0.0, 1.0), P0), Tsit5(); abstol = 1e-12, reltol = 1e-12)
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

@testset "TraitDifferentialEquations: Rodas5 with the library's duals" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    st = solve(prob, Rodas5(); abstol = 1e-10, reltol = 1e-10)
    @test st.retcode === :Success && isapprox(st.u[end], REF; rtol = 1e-9)
    # the library's dual Jacobian equals the analytic one
    J = zeros(3, 3); u = [1.0, 2.0, 3.0]
    jacobian!(prob, J, u, 0.0)
    @test J ≈ [-10.0 10.0 0.0; 28.0-3.0 -1.0 -1.0; 2.0 1.0 -8/3]
    # the same through the compiled function, and by finite differences for a stored function of one signature
    jacobian!(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), P0), J, u, 0.0)
    @test J ≈ [-10.0 10.0 0.0; 28.0-3.0 -1.0 -1.0; 2.0 1.0 -8/3]
    S = TraitDifferentialEquations.RHSSig{Vector{Float64},Float64}
    jacobian!(ODEProblem(Dynamic{Callable{S,Nothing}}(lorenz!), U0, (0.0, 1.0), P0), J, u, 0.0)
    @test isapprox(J, [-10.0 10.0 0.0; 28.0-3.0 -1.0 -1.0; 2.0 1.0 -8/3]; rtol = 1e-6)
    # a stiff problem: Robertson, mass conserved
    rober!(du, u, p, t) = (du[1] = -0.04u[1] + 1e4 * u[2] * u[3]; du[2] = 0.04u[1] - 1e4 * u[2] * u[3] - 3e7 * u[2]^2; du[3] = 3e7 * u[2]^2; nothing)
    sr = solve(ODEProblem(rober!, [1.0, 0.0, 0.0], (0.0, 1e5)), Rodas5(); abstol = 1e-8, reltol = 1e-6)
    @test sr.retcode === :Success && isapprox(sum(sr.u[end]), 1.0; atol = 1e-10) && isapprox(sr.u[end][1], 0.0178; atol = 1e-3)
    # a problem type of its own: its Jacobian by the trait's default (finite differences)
    sA = solve(Lorenz3(U0, (0.0, 1.0), P0), Rodas5(); abstol = 1e-10, reltol = 1e-10)
    @test isapprox(sA.u[end], REF; rtol = 1e-7)
end

@testset "TraitDifferentialEquations: gradients through Tsit5 (row 5)" begin
    loss1(p) = sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end])
    g = gradient(loss1, copy(P0))
    h = 1e-6
    fd = [(loss1(P0 .+ h .* (1:3 .== k)) - loss1(P0 .- h .* (1:3 .== k))) / 2h for k in 1:3]
    @test isapprox(g, fd; rtol = 1e-4)
    # a second loss closure of the same signature: the same dual type, so the solver's code is shared
    w = 2.0
    loss2(p) = w * solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end][1]
    g2 = gradient(loss2, copy(P0))
    fd2 = [(loss2(P0 .+ h .* (1:3 .== k)) - loss2(P0 .- h .* (1:3 .== k))) / 2h for k in 1:3]
    @test isapprox(g2, fd2; rtol = 1e-4)
    # nested: a loss that captures an outer level's duals gets ForwardDiff's own tag (session 23's hazard)
    @test TraitDifferentialEquations._holds_duals(typeof(let d = ForwardDiff.Dual(1.0, 1.0); p -> d * p[1]; end))
    @test !TraitDifferentialEquations._holds_duals(typeof(loss2))
    # a loss that captures a vector (a type parameter that is a value, `Array{Float64, 1}`)
    u0 = copy(U0)
    @test !TraitDifferentialEquations._holds_duals(typeof(p -> sum(u0) * p[1]))
    @test gradient(p -> sum(solve(ODEProblem(lorenz!, u0, (0.0, 1.0), p), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end]), copy(P0)) == g
    outer(a) = gradient(p -> a[1] * p[1]^2, [3.0])[1]                 # d/da (2 a p) at p = 3: 6
    @test ForwardDiff.derivative(x -> outer([x]), 1.0) ≈ 6.0
    # callbacks need Float64 states
    @test_throws ArgumentError gradient(p -> sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5();
                                       callback = DiscreteCallback((u, t, i) -> false, i -> nothing)).u[end]), copy(P0))
end

@testset "TraitDifferentialEquations: a Float32 state (row 6)" begin
    lorenz32!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
    s = solve(ODEProblem(lorenz32!, Float32[1, 0, 0], (0.0, 1.0), Float32[10, 28, 8 / 3]), Tsit5())
    @test s.retcode === :Success && eltype(s.u[end]) === Float32
    s64 = solve(ODEProblem(lorenz32!, U0, (0.0, 1.0), P0), Tsit5())
    @test isapprox(Float64.(s.u[end]), s64.u[end]; rtol = 1e-2)
end


include("routea.jl")
include("compiledcall.jl")
