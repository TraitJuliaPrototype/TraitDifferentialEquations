# W-32 R2 (E-239), the TDE side of NetworkDynamics' port: optional properties of a problem's function
# (open issue 119), the system object back, a problem's callbacks, `tstops` and the callbacks' `initialize`,
# the crossing's direction, `auto_dt_reset!`, saved parameters, Rodas5's continuous callbacks, and `init`,
# `step!`, `solve!`. Runs inside runtests.jl (which defines lorenz!, U0, P0, REF, caught).

"A callable model that carries its properties, as a library's model does (NetworkDynamics' `Network`)."
struct MModel
    M::Any
end
(m::MModel)(du, u, p, t) = (du .= -u; nothing)
@implement HasMassMatrix MModel begin
    mass_matrix(m) = m.M
end
@implement HasSystem MModel begin
    system(m) = m
end
"An algorithm that solves only whole (no `init`): Tsit5's driver."
struct OnlySolve end
@implement ODEAlgorithm OnlySolve begin
    solve_core(a, m, callbacks, opts) = TraitDifferentialEquations.ODECore.solve_core(Tsit5(), m, callbacks, opts)
end
"A callable with SciML's `mass_matrix` property and no declaration."
struct Undeclared
    mass_matrix::Matrix{Float64}
end
(m::Undeclared)(du, u, p, t) = (du .= -u; nothing)

@testset "R2: optional properties of a problem's function (open issue 119)" begin
    sing = [1.0 0.0; 0.0 0.0]
    m = MModel(sing)
    pm = ODEProblem(m, [1.0, 1.0], (0.0, 1.0))
    # the problem keeps the function's mass matrix; every solver here takes none, and refuses it
    for alg in (Tsit5(), Rodas5())
        e = caught(() -> solve(pm, alg))
        @test e isa ArgumentError && occursin("mass matrix", sprint(showerror, e))
        @test caught(() -> init(pm, alg)) isa ArgumentError
    end
    # through ODEFunction, and through a cast of the model
    @test caught(() -> solve(ODEProblem(ODEFunction(m), [1.0, 1.0], (0.0, 1.0)), Rodas5())) isa ArgumentError
    @test caught(() -> solve(ODEProblem(compiled_for(m), [1.0, 1.0], (0.0, 1.0)), Tsit5())) isa ArgumentError
    # the identity is an ODE: solved
    @test solve(ODEProblem(MModel(TraitDifferentialEquations.ODECore.I), [1.0, 1.0], (0.0, 1.0)), Tsit5()).retcode === :Success
    @test solve(ODEProblem(MModel([1.0 0.0; 0.0 1.0]), [1.0, 1.0], (0.0, 1.0)), Rodas5()).retcode === :Success
    # a mass_matrix property without the trait: refused at the problem, naming the trait
    e = caught(() -> ODEProblem(Undeclared(sing), [1.0, 1.0], (0.0, 1.0)))
    @test e isa ArgumentError && occursin("HasMassMatrix", sprint(showerror, e))
    @test solve(ODEProblem(Undeclared([1.0 0.0; 0.0 1.0]), [1.0, 1.0], (0.0, 1.0)), Tsit5()).retcode === :Success
    # ODEFunction is callable, as SciML's
    du = zeros(3); ODEFunction(lorenz!)(du, U0, P0, 0.0)
    du0 = zeros(3); lorenz!(du0, U0, P0, 0.0)
    @test du == du0
end

@testset "R2: the system object back (census item 12)" begin
    m = MModel(TraitDifferentialEquations.ODECore.I)
    prob = ODEProblem(m, [1.0, 1.0], (0.0, 1.0))
    seen = Any[]
    cb = DiscreteCallback((u, t, i) -> isempty(seen), i -> (push!(seen, getsystem(i)); nothing))
    sol = solve(prob, Tsit5(); callback = cb)
    @test getsystem(prob) === m && getsystem(sol) === m && seen == [m]
    integ = init(prob, Rodas5())
    @test getsystem(integ) === m
    # ODEFunction's `sys` overrides the function's; a `SymbolCache` still gives names
    sc = SymbolCache([:a, :b], Symbol[])
    s2 = solve(ODEProblem(ODEFunction(m; sys = sc), [1.0, 1.0], (0.0, 1.0)), Tsit5())
    @test getsystem(s2) === sc && s2[:b] == [u[2] for u in s2.u]
    # a system that is no index provider is kept; names are refused by name
    @test getsystem(solve(ODEProblem(lorenz!, U0, (0.0, 0.1), P0), Tsit5())) === nothing
    @test caught(() -> sol[:a]) isa ArgumentError
end

@testset "R2: a problem's callbacks, tstops, initialize, the crossing's direction (items 14-16)" begin
    p = [10.0, 28.0, 8 / 3]
    n = Ref(0)
    kick = DiscreteCallback((u, t, i) -> t > 0.5 && n[] == 0, i -> (n[] += 1; i.u[1] += 1.0; nothing))
    pk = ODEProblem(lorenz!, U0, (0.0, 1.0), p; callback = kick)
    s1 = solve(pk, Tsit5())
    n[] = 0
    s2 = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); callback = kick)
    @test s1.u == s2.u && s1.t == s2.t
    # the problem's callbacks come before the ones given
    order = Symbol[]
    a = DiscreteCallback((u, t, i) -> isempty(order), i -> (push!(order, :problem); nothing))
    b = DiscreteCallback((u, t, i) -> length(order) == 1, i -> (push!(order, :given); nothing))
    solve(ODEProblem(lorenz!, U0, (0.0, 0.1), p; callback = a), Tsit5(); callback = b)
    @test order == [:problem, :given]
    # tstops per solve: the solver steps to them
    st = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); tstops = [0.333, 0.777])
    @test 0.333 in st.t && 0.777 in st.t
    # a callback's initialize at each solve's init (DiffEqCallbacks' PresetTimeCallback): the callback is reused
    hit = Float64[]
    preset = DiscreteCallback((u, t, i) -> t == 0.4, i -> (push!(hit, i.t); nothing);
                              initialize = (u, t, i) -> add_tstop!(i, 0.4))
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), p)
    solve(prob, Tsit5(); callback = preset); solve(prob, Rodas5(); callback = preset)
    @test hit == [0.4, 0.4]
    # the crossing's direction: affect! where the condition rises, affect_neg! where it falls
    osc(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
    for alg in (Tsit5(), Rodas5())
        ups, downs = Float64[], Float64[]
        cc = ContinuousCallback((u, t, i) -> u[1], i -> (push!(ups, i.t); nothing), i -> (push!(downs, i.t); nothing))
        solve(ODEProblem(osc, [1.0, 0.0], (0.0, 10.0)), alg; callback = cc, abstol = 1e-10, reltol = 1e-10)
        @test isapprox(downs, [π / 2, 5π / 2]; atol = 1e-6) && isapprox(ups, [3π / 2]; atol = 1e-6)
    end
    @test caught(() -> ContinuousCallback((u, t, i) -> u[1], i -> nothing, nothing)) isa ArgumentError
    # auto_dt_reset! after a change of state: the solve goes on, at the same results as without the reset here
    rs = DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.52, i -> (auto_dt_reset!(i); nothing))
    for alg in (Tsit5(), Rodas5())
        sr = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), alg; callback = rs, abstol = 1e-10, reltol = 1e-10)
        @test sr.retcode === :Success && isapprox(sr.u[end], REF; rtol = 1e-7)
    end
end

@testset "R2: parameters saved by callbacks (item 13)" begin
    p = [10.0, 28.0, 8 / 3]
    change = DiscreteCallback((u, t, i) -> t >= 0.5 && i.p[1] == 10.0, i -> (i.p[1] = 5.0; save_discretes!(i); nothing))
    s = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); callback = change, saveat = [0.25, 0.75])
    @test parameters_at(s, 0.25) == [10.0, 28.0, 8 / 3] && parameters_at(s, 0.75)[1] == 5.0 && p[1] == 5.0
    @test parameters_at(s, 0.0) == [10.0, 28.0, 8 / 3]
    # no callback: the problem's parameters
    s0 = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5())
    @test parameters_at(s0, 0.5) === s0.prob.p
end

@testset "R2: init, step!, solve! (item 2)" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    kick = DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.6, i -> (i.u[1] += 1.0; nothing))
    for alg in (Tsit5(), Rodas5(), Rodas5(chunk_size = 0))
        ref = solve(prob, alg; callback = kick, abstol = 1e-10, reltol = 1e-10)
        integ = init(prob, alg; callback = kick, abstol = 1e-10, reltol = 1e-10)
        @test integ.t == 0.0 && integ.u == U0 && integ.p === prob.p
        step!(integ)
        @test integ.t > 0.0 && integ.dt == integ.t
        sol = solve!(integ)
        # stepping gives the solve's steps, bit for bit
        @test sol.t == ref.t && sol.u == ref.u && sol.retcode === :Success && integ.t == 1.0
    end
    # step!(integ, dt, true) stops at t + dt exactly; the integrator is what callbacks see
    f = ODEFunction(lorenz!; sys = SymbolCache([:x, :y, :z], [:σ, :ρ, :β]))
    integ = init(ODEProblem(f, U0, (0.0, 1.0), P0), Tsit5())
    step!(integ, 0.25, true)
    @test integ.t == 0.25 && getstate(integ, :y) == integ.u[2] && getparam(integ, :ρ) == 28.0
    step!(integ, 0.25)
    @test integ.t >= 0.5
    # a change of state from user code between steps (u_modified), and terminate!
    integ.u[1] += 1.0; integ.u_modified = true
    step!(integ)
    terminate!(integ)
    step!(integ)
    s = solve!(integ)
    @test s.retcode === :Terminated && s.t[end] < 1.0
    # one integrator type for every steppable algorithm, function and callback set
    @test typeof(init(prob, Tsit5())) === typeof(init(prob, Rodas5(); callback = kick)) ===
          typeof(init(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), P0), Rodas5(chunk_size = 0)))
    # an algorithm that cannot be stepped is refused by init
    @test conforms(Tsit5, ODEStepper) && conforms(typeof(Rodas5()), ODEStepper) && !conforms(OnlySolve, ODEStepper)
    @test solve(prob, OnlySolve()).u == solve(prob, Tsit5()).u
    e = caught(() -> init(prob, OnlySolve()))
    @test e isa TraitBoundError && occursin("ODEStepper", sprint(showerror, e))
end
