# W-34, move 2 (E-158): callbacks entered at the problem's types at `init` (`Callback{U, T, P}`,
# `EventCallback{U, T, P}`, kept in vectors of the integrator), the integrator as callbacks see it
# (`IntegratorView{U, T, P}` with `add_tstop!`, `terminate!`, `interp!`). Runs inside runtests.jl.

"Mutable parameters, so that a callback's write reaches the solve."
mutable struct OscP
    k::Float64
    c::Float64
end
osc!(du, u, p, t) = (du[1] = u[2]; du[2] = -p.k * u[1] - p.c * u[2]; nothing)

struct OnlyCondition end
TraitDifferentialEquations.condition(cb::OnlyCondition, u::Vector{Float64}, t::Float64,
                                     integ::Dynamic{IntegratorView{Vector{Float64}, Float64, Vector{Float64}}}) = false

@testset "W-34 move 2: callbacks at the problem's types, the integrator view's operations" begin
    # integ.p is the problem's parameters at their type; a write to it reaches the solve
    OV = Dynamic{IntegratorView{Vector{Float64}, Float64, OscP}}
    @test Base.return_types(i -> i.p.k, (OV,)) == [Float64]
    p = OscP(4.0, 0.1)
    times = Float64[]
    kick = DiscreteCallback((u, t, i) -> t >= 1.0 && isempty(times),
                            i -> (push!(times, i.t); add_tstop!(i, 2.0); nothing))
    at2 = DiscreteCallback((u, t, i) -> t == 2.0, i -> (push!(times, i.t); i.p.c = 0.5; nothing))
    stop = DiscreteCallback((u, t, i) -> t >= 3.0, i -> terminate!(i))
    s = solve(ODEProblem(osc!, [1.0, 0.0], (0.0, 10.0), p), Tsit5(); callback = (kick, at2, stop))
    @test length(times) == 2 && times[2] == 2.0 && p.c == 0.5
    @test s.retcode === :Terminated && 3.0 <= s.t[end] < 3.5
    # one integrator type for two callback sets
    pr = ODEProblem(osc!, [1.0, 0.0], (0.0, 1.0), OscP(4.0, 0.1))
    C = TraitDifferentialEquations.ODECore
    i1 = C.tsit5_init(C.core_problem(pr), pr.u0, pr.tspan, C.adapt_callbacks(Any[kick, stop], Vector{Float64}, pr.p))
    i2 = C.tsit5_init(C.core_problem(pr), pr.u0, pr.tspan, C.adapt_callbacks(Any[ContinuousCallback((u, t, i) -> u[1], i -> nothing)], Vector{Float64}, pr.p))
    @test typeof(i1) === typeof(i2)
    # a continuous callback: the bouncing ball's first contact at sqrt(2/g), the ball above the ground
    ball!(du, u, p, t) = (du[1] = u[2]; du[2] = -9.81; nothing)
    hits = Float64[]
    ground = ContinuousCallback((u, t, i) -> u[1], i -> (push!(hits, i.t); i.u[2] = -0.9 * i.u[2]; nothing))
    sb = solve(ODEProblem(ball!, [1.0, 0.0], (0.0, 3.0)), Tsit5(); callback = ground, abstol = 1e-10, reltol = 1e-10, saveat = 0:0.01:3)
    @test isapprox(hits[1], sqrt(2 / 9.81); atol = 1e-9) && length(hits) >= 3
    @test minimum(u[1] for u in sb.u) >= -1e-9
    # interp! inside an affect!: the state at the step's start equals the saved one
    seen = Float64[]
    probe = DiscreteCallback((u, t, i) -> isempty(seen) && t > 0.1,
                             i -> (out = similar(i.u); interp!(i, out, i.t - i.dt); push!(seen, out[1]); nothing))
    sp = solve(ODEProblem(osc!, [1.0, 0.0], (0.0, 1.0), OscP(4.0, 0.1)), Tsit5(); callback = probe)
    k = findfirst(t -> t > 0.1, sp.t)
    @test seen[1] ≈ sp.u[k-1][1]
    # Rodas5 with the view's operations
    sr = solve(ODEProblem(osc!, [1.0, 0.0], (0.0, 5.0), OscP(4.0, 0.1)), Rodas5(); callback = (stop,))
    @test sr.retcode === :Terminated
    @test_throws ArgumentError solve(ODEProblem(osc!, [1.0, 0.0], (0.0, 1.0), OscP(4.0, 0.1)), Rodas5(); callback = ground)
    # a callback type without affect! is refused at @implement
    e = try
        Core.eval(@__MODULE__, :(@implement Callback{Vector{Float64}, Float64, Vector{Float64}} OnlyCondition)); nothing
    catch err
        err isa LoadError ? err.error : err
    end
    @test e isa ImplementError
end
