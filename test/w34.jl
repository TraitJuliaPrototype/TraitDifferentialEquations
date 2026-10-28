# W-34, move 1 in form B (E-157): SciML's split of a plain record (`ODEFunction`) and the problem,
# a given Jacobian, the dual chunk picked per problem at Rodas5's `init`, a callable model from a
# package that knows nothing of the solvers (`examples/ChainModels`), and the solution as a
# `Trajectory`. Runs inside runtests.jl (which defines lorenz!, U0, P0, REF, caught).
include(joinpath(@__DIR__, "..", "examples", "ChainModels", "src", "ChainModels.jl"))
using .ChainModels
const ODECore = TraitDifferentialEquations.ODECore

lorenz_jac!(J, u, p, t) = (J[1, 1] = -p[1]; J[1, 2] = p[1]; J[1, 3] = 0.0;
                           J[2, 1] = p[2] - u[3]; J[2, 2] = -1.0; J[2, 3] = -u[1];
                           J[3, 1] = u[2]; J[3, 2] = u[1]; J[3, 3] = -p[3]; nothing)

@testset "W-34: ODEFunction, a given Jacobian, the chunk at init" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    # the record and the bare function give the same problem type
    @test typeof(ODEProblem(ODEFunction(lorenz!), U0, (0.0, 1.0), P0)) === typeof(prob)
    # a given Jacobian: Rodas5 calls it
    pj = ODEProblem(ODEFunction(lorenz!; jac = lorenz_jac!), U0, (0.0, 1.0), P0)
    J = zeros(3, 3); u = [1.0, 2.0, 3.0]
    jacobian!(pj, J, u, 0.0)
    @test J == [-10.0 10.0 0.0; 25.0 -1.0 -1.0; 2.0 1.0 -8/3]
    # the right-hand side partially evaluated on the parameters (W-63)
    @test rhs_function(pj) === nothing && rhs_function(prob) isa TraitDifferentialEquations.ODECore.PartialFn && rhs_function(prob).f === lorenz!
    @test isapprox(solve(pj, Rodas5(); abstol = 1e-10, reltol = 1e-10).u[end], REF; rtol = 1e-9)
    # a Jacobian that returns J: refused at the bound, as a right-hand side is (traits reference §6.5)
    badjac!(J, u, p, t) = (J .= 0.0)
    e = caught(() -> solve(ODEProblem(ODEFunction(lorenz!; jac = badjac!), U0, (0.0, 1.0), P0), Rodas5()))
    @test e isa TraitBoundError && occursin("Nothing", sprint(showerror, e))
    # the chunk picked at init: the same values as the problem's own Jacobian, in both modes
    r = solve(prob, Rodas5(); abstol = 1e-10, reltol = 1e-10)
    @test solve(prob, Rodas5(chunk_size = 0); abstol = 1e-10, reltol = 1e-10).u == r.u
    @test solve(prob, Rodas5(chunk_size = 2); abstol = 1e-10, reltol = 1e-10).u == r.u
    @test solve(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), P0), Rodas5(chunk_size = 0); abstol = 1e-10, reltol = 1e-10).u == r.u
    # a problem type of its own has no plain function: its jacobian! (finite differences here)
    @test solve(Lorenz3(U0, (0.0, 1.0), P0), Rodas5(chunk_size = 0)).u == solve(Lorenz3(U0, (0.0, 1.0), P0), Rodas5()).u
    # a mass matrix other than I is refused
    @test caught(() -> ODEProblem(ODEFunction(lorenz!; mass_matrix = [1.0 0 0; 0 1 0; 0 0 0]), U0, (0.0, 1.0), P0)) isa ArgumentError
end

@testset "W-34: a callable model enters form B (ChainModels)" begin
    u0 = collect(range(0.0, 1.0; length = 40))
    pa = ODEProblem(ChainModel(PathGraph(40)), u0, (0.0, 1.0), [1.0, 0.1])
    pb = ODEProblem(ChainModel(RingGraph(40)), u0, (0.0, 1.0), [1.0, 0.1])
    # one problem type for every graph type and for a plain function: the solver compiles once per (T, P)
    @test typeof(pa) === typeof(pb) === typeof(ODEProblem(lorenz!, u0, (0.0, 1.0), [1.0, 0.1]))
    sa = solve(pa, Tsit5())
    @test sa.retcode === :Success && isapprox(sum(sa.u[end]), sum(u0); rtol = 0.1)
    # a named tuple for p, and compiled_for: the same values
    pn = ODEProblem(ChainModel(PathGraph(40)), u0, (0.0, 1.0), (k = 1.0, d = 0.1))
    @test solve(pn, Tsit5()).u == sa.u
    @test solve(ODEProblem(compiled_for(ChainModel(PathGraph(40))), u0, (0.0, 1.0), [1.0, 0.1]), Tsit5()).u == sa.u
    @test solve(pn, Rodas5(chunk_size = 0)).u == solve(pa, Rodas5(chunk_size = 0)).u
end

@testset "W-34: the solution as a Trajectory" begin
    s = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5(); saveat = 0.0:0.5:1.0)
    @test ODECore.times(s) == [0.0, 0.5, 1.0] && ODECore.state(s, 3) == s.u[3]
    @test ODECore.at(s, 0.25) ≈ (s.u[1] .+ s.u[2]) ./ 2 && ODECore.at(s, 1.0) == s.u[3]
    @test caught(() -> ODECore.at(s, 2.0)) isa DomainError
    # one solution type for a function, a callable model and a problem type: the state's element type and the
    # parameters' type, never the model (move 4 gives the solution its problem's parameters)
    u0 = collect(range(0.0, 1.0; length = 40))
    sc = solve(ODEProblem(ChainModel(PathGraph(40)), u0, (0.0, 1.0), [1.0, 0.1]), Tsit5())
    @test typeof(s) === typeof(sc) === typeof(solve(Lorenz3(U0, (0.0, 1.0), P0), Tsit5())) === ODESolution{Float64,Vector{Float64}}
    @test Dynamic{Trajectory{Vector{Float64},Float64}}(s) isa Dynamic
    # a Float32 state: `at` stays at Float32 (the member check refuses `Vector{Float64}` there)
    s32 = solve(ODEProblem(lorenz!, Float32.(U0), (0.0, 1.0), Float32.(P0)), Tsit5())
    @test ODECore.at(s32, 0.5) isa Vector{Float32}
end

@testset "symbolic access (move 4): names from the problem's `sys`, by any provider" begin
    f = ODEFunction(lorenz!; sys = SymbolCache([:x, :y, :z], [:σ, :ρ, :β]))
    s = solve(ODEProblem(f, U0, (0.0, 1.0), P0), Tsit5())
    @test s.ps[:ρ] == P0[2] && s[:z] == [u[3] for u in s.u] && getstate(s, :y) == s.u[end][2] && getparam(s, :β) == P0[3]
    @test caught(() -> solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5())[:x]) isa ArgumentError
    @test caught(() -> s[:w]) isa ArgumentError
end

@testset "symbolic access in a callback (move 4): the integrator it sees, by name" begin
    f = ODEFunction(lorenz!; sys = SymbolCache([:x, :y, :z], [:σ, :ρ, :β]))
    seen = Float64[]
    cb = DiscreteCallback((u, t, integ) -> getstate(integ, :x) > 5.0,
                          integ -> (push!(seen, getparam(integ, :ρ)); integ.terminated = true; nothing))
    s = solve(ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 10.0), P0), Tsit5(); callback = cb)
    @test seen == [P0[2]] && s[:x][end] > 5.0 && s.t[end] < 10.0
end
