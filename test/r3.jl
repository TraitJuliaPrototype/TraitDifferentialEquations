# W-32 R3 (E-242): `remake`, dense output (open issue 117), the solution's `idxs` forms.
# Runs inside runtests.jl (which defines lorenz!, U0, P0, REF, caught).

@testset "R3: remake (S7)" begin
    p = [10.0, 28.0, 8 / 3]
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), p)
    p2 = [10.0, 20.0, 8 / 3]
    r = remake(prob; p = p2)
    @test r.p === p2 && r.u0 == U0 && r.tspan == prob.tspan && prob.p === p
    @test solve(r, Tsit5()).u == solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p2), Tsit5()).u
    @test typeof(r) === typeof(prob)
    r2 = remake(prob; u0 = [2.0, 0.0, 0.0], tspan = (0.0, 0.5))
    @test r2.u0 == [2.0, 0.0, 0.0] && r2.tspan == (0.0, 0.5) && r2.p === p
    @test solve(r2, Rodas5()).u[end] ≈ solve(ODEProblem(lorenz!, [2.0, 0.0, 0.0], (0.0, 0.5), p), Rodas5()).u[end]
    # a compiled function stays compiled; a Jacobian given and the callbacks are kept
    pc = ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), p)
    @test typeof(remake(pc; p = p2)) === typeof(pc)
    jl!(J, u, q, t) = (J .= [-q[1] q[1] 0; q[2]-u[3] -1 -u[1]; u[2] u[1] -q[3]]; nothing)
    pj = ODEProblem(ODEFunction(lorenz!; jac = jl!), U0, (0.0, 1.0), p)
    Jm = zeros(3, 3); jacobian!(remake(pj; p = p2), Jm, [1.0, 2.0, 3.0], 0.0)
    @test Jm ≈ [-10.0 10.0 0.0; 20.0-3.0 -1.0 -1.0; 2.0 1.0 -8/3]
    # a record's functions at `Callable`'s default: a cast inside `ODEFunction` reaches neither it nor the problem
    # (open issue 159; lowering L-56 refuses a cast in a call with keyword arguments)
    pcj = ODEProblem(ODEFunction(lorenz!; jac = jl!), U0, (0.0, 1.0), p)
    rcj = remake(pcj; p = p2)
    @test typeof(rcj) === typeof(pcj) && solve(rcj, Rodas5()).u[end] ≈ solve(remake(pj; p = p2), Rodas5()).u[end]
    n = Ref(0)
    cb = DiscreteCallback((u, t, i) -> t > 0.5 && n[] == 0, i -> (n[] += 1; nothing))
    pk = ODEProblem(lorenz!, U0, (0.0, 1.0), p; callback = cb)
    solve(remake(pk; p = p2), Tsit5())
    @test n[] == 1
end

@testset "R3: dense output (open issue 117)" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    for alg in (Tsit5(), Rodas5())
        s = solve(prob, alg; abstol = 1e-10, reltol = 1e-10)
        @test s.interp !== nothing
        # the solver's interpolant: as `saveat` takes it, on the same steps
        ts = [0.137, 0.5, 0.731]
        sa = solve(prob, alg; abstol = 1e-10, reltol = 1e-10, saveat = ts)
        @test sa.interp === nothing
        for (k, τ) in enumerate(ts)
            @test isapprox(s(τ), sa.u[k]; rtol = 1e-13)
        end
        # at the saved times, the saved states; at the end, the reference
        @test s(s.t[7]) ≈ s.u[7] && s(1.0) ≈ s.u[end] && isapprox(s(1.0), REF; rtol = 1e-7)
        # without dense output, linear between the saved states
        sl = solve(prob, alg; abstol = 1e-10, reltol = 1e-10, dense = false)
        τ = (sl.t[3] + sl.t[4]) / 2
        @test sl(τ) ≈ (sl.u[3] .+ sl.u[4]) ./ 2
        @test caught(() -> s(1.5)) isa DomainError
    end
    # the solution's type has no axis for the solver
    @test typeof(solve(prob, Tsit5())) === typeof(solve(prob, Rodas5()))
    # an event: the state at its time is the one saved there, before the affect; after it, the affected state
    bounce = ContinuousCallback((u, t, i) -> u[1] - 0.5, i -> (i.u[1] = 0.0; nothing))
    decay!(du, u, p, t) = (du[1] = 1.0; nothing)
    for alg in (Tsit5(), Rodas5())
        s = solve(ODEProblem(decay!, [0.0], (0.0, 1.0)), alg; callback = bounce)
        k = findfirst(u -> u[1] >= 0.5 - 1e-9, s.u)
        te = s.t[k]
        @test isapprox(te, 0.5; atol = 1e-9) && s(te) ≈ s.u[k]
        @test isapprox(s(0.75)[1], 0.75 - te; atol = 1e-8)
    end
    # init/solve! keep the interpolant too
    integ = init(prob, Tsit5())
    @test solve!(integ)(0.5) ≈ solve(prob, Tsit5())(0.5)
end

@testset "R3: the solution's idxs forms" begin
    sc = SymbolCache([:x, :y, :z], [:σ, :ρ, :β])
    s = solve(ODEProblem(ODEFunction(lorenz!; sys = sc), U0, (0.0, 1.0), P0), Tsit5())
    u = s(0.5)
    @test s(0.5; idxs = 2) == u[2] && s(0.5; idxs = :z) == u[3] && s(0.5; idxs = [:x, 3]) == [u[1], u[3]]
    a = s([0.25, 0.5])
    @test a.t == [0.25, 0.5] && a.u[2] == u
    @test s([0.25, 0.5]; idxs = :y) == [s(0.25)[2], u[2]]
    # an index type of a system's own: test/r118.jl (open issue 118)
    @test caught(() -> s(0.5; idxs = 1.5)) isa ArgumentError
    @test caught(() -> s(0.5; idxs = :w)) isa ArgumentError
end
