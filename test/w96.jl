# W-96 (E-248): Tsit5's own interpolant (dense output, `saveat`, event location), and the states saved before and
# after each event. Runs inside runtests.jl. The upstream numbers are OrdinaryDiffEqTsit5 2.1.5's at abstol = reltol
# = 1e-8 (`experiments/w34-sciml/w96/upstream_ref.jl`, log `upstream-ref-1e-8.log`): there TDE's Tsit5 takes
# upstream's steps (151 and 323 accepted), so the values between steps compare the interpolants.

# Lorenz on (0, 1): upstream's dense output at three times
const W96_TS = [0.137, 0.5, 0.731]
const W96_DENSE = [[1.8953512289070034, 4.032203207326986, 0.28530649649638545],
                   [6.503159951445648, -8.48701603406252, 38.0734777644752],
                   [-7.059711875520795, -8.644540309117616, 25.54990048588804]]
# Lorenz on (0, 3) with z through 25 (either way: u[2] *= 0.9): upstream's event times and saved times with `saveat`
const W96_EVENTS = [0.3414692816088776, 0.6505594646486393, 0.9044151966689878, 1.290673982931957, 1.551477296281431,
                    1.9502763960934364, 2.198734235604787, 2.61266943159187, 2.8480187228514207]
const W96_SAVEAT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
const W96_SAVED = sort([W96_EVENTS; W96_EVENTS; W96_SAVEAT])

@testset "W-96: Tsit5's interpolant" begin
    # fourth order: exact where the solution is a polynomial of degree 4 (the cubic Hermite polynomial is not)
    q = solve(ODEProblem((du, u, p, t) -> (du[1] = t^3; nothing), [0.0], (0.0, 2.0)), Tsit5(); dtmax = 0.5)
    @test q.interp !== nothing && length(q.t) >= 5
    @test maximum(abs(q(τ)[1] - τ^4 / 4) for τ in 0.0:0.01:2.0) < 1e-12
    # upstream's dense output and `saveat`, on upstream's steps
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    s = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8)
    @test s.naccept == 151
    @test maximum(maximum(abs.(s(τ) .- W96_DENSE[k])) for (k, τ) in enumerate(W96_TS)) < 1e-9
    sa = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, saveat = W96_TS)
    @test maximum(maximum(abs.(sa.u[k] .- W96_DENSE[k])) for k in eachindex(W96_TS)) < 1e-9
    # against a tight run: between the steps as accurate as at the steps
    st = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
    at_steps = maximum(maximum(abs.(s.u[k] .- st(s.t[k]))) for k in eachindex(s.t))
    between = maximum(maximum(abs.(s(τ) .- st(τ))) for τ in (s.t[1:end-1] .+ s.t[2:end]) ./ 2)
    @test between <= 2 * at_steps
end

@testset "W-96: events on Tsit5's interpolant, the states saved there" begin
    te = Float64[]
    cb = ContinuousCallback((u, t, i) -> u[3] - 25.0, i -> (push!(te, i.t); i.u[2] *= 0.9; nothing))
    pb = ODEProblem(lorenz!, U0, (0.0, 3.0), P0)
    s = solve(pb, Tsit5(); abstol = 1e-8, reltol = 1e-8, callback = cb)
    # upstream's event times (the cubic Hermite polynomial put them 3e-8 off)
    @test length(te) == length(W96_EVENTS) && maximum(abs.(te .- W96_EVENTS)) < 1e-10
    # every step saved, and each event's state before and after its affect
    @test s.naccept == 323 && length(s.t) == s.naccept + 1 + length(te)
    k = findfirst(==(te[1]), s.t)
    @test s.t[k+1] == te[1] && s.u[k+1][2] == 0.9 * s.u[k][2] && s.u[k+1][[1, 3]] == s.u[k][[1, 3]]
    @test isapprox(s.u[k][3], 25.0; atol = 1e-9)
    # at an event's time the solution is the state before the event, with or without dense output
    @test s(te[1]) ≈ s.u[k]
    sl = solve(pb, Tsit5(); abstol = 1e-8, reltol = 1e-8, callback = cb, dense = false)
    @test sl(te[1]) == sl.u[k] && sl.interp === nothing
    # with `saveat`: its times and the events' pairs, as upstream
    sv = solve(pb, Tsit5(); abstol = 1e-8, reltol = 1e-8, callback = cb, saveat = W96_SAVEAT)
    @test length(sv.t) == length(W96_SAVED) && maximum(abs.(sv.t .- W96_SAVED)) < 1e-10
    # Rodas5: its own interpolant, the same saves
    empty!(te)
    sr = solve(pb, Rodas5(); abstol = 1e-8, reltol = 1e-8, callback = cb)
    @test length(te) == length(W96_EVENTS) && maximum(abs.(te .- W96_EVENTS)) < 1e-8
    @test length(sr.t) == sr.naccept + 1 + length(te)
    empty!(te)
    svr = solve(pb, Rodas5(); abstol = 1e-8, reltol = 1e-8, callback = cb, saveat = W96_SAVEAT)
    @test length(svr.t) == length(W96_SAVED)
end

@testset "W-96: a discrete callback's saves" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    kick = DiscreteCallback((u, t, i) -> t == 0.7, i -> (i.u[1] += 1.0; nothing))
    for alg in (Tsit5(), Rodas5())
        # with `saveat`: before and after the affect where no time of `saveat` is the callback's
        s = solve(prob, alg; callback = kick, tstops = [0.7], saveat = [0.5, 1.0])
        @test s.t == [0.5, 0.7, 0.7, 1.0] && s.u[3][1] == s.u[2][1] + 1.0
        # a time of `saveat` at the callback's: saved once before the affect
        s2 = solve(prob, alg; callback = kick, tstops = [0.7], saveat = [0.5, 0.7, 1.0])
        @test s2.t == [0.5, 0.7, 0.7, 1.0] && s2.u == s.u
        # every step: the step's end, then the state after the affect
        s3 = solve(prob, alg; callback = kick, tstops = [0.7])
        k = findfirst(==(0.7), s3.t)
        @test s3.t[k+1] == 0.7 && s3.u[k+1][1] == s3.u[k][1] + 1.0 && count(==(0.7), s3.t) == 2
    end
end
