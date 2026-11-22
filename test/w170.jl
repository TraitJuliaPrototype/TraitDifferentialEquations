# W-170 (open issue 263): a continuous callback of several conditions, `VectorContinuousCallback` (SciML's): the
# conditions evaluated once per point, each condition's crossing located, the affect told which crossed. The same
# events as the scalar callbacks of the same conditions, in Tsit5 and Rodas5.
@testset "W-170: a continuous callback of several conditions" begin
    lin!(du, u, p, t) = (du[1] = p[1]; du[2] = -p[2]; du[3] = 0.1 * u[3]; nothing)
    prob = ODEProblem(lin!, [0.0, 1.0, 1.0], (0.0, 1.0), [1.0, 2.0])
    for alg in (Tsit5(), Rodas5())
        ts, tv, sv, nv = Float64[], Float64[], Vector{Int8}[], Ref(0)
        c1 = ContinuousCallback((u, t, i) -> u[1] - 0.3, i -> (push!(ts, i.t); i.u[1] = 0.0; nothing))
        c2 = ContinuousCallback((u, t, i) -> u[2], i -> nothing, i -> (push!(ts, i.t); i.u[2] = 1.0; nothing))
        s1 = solve(prob, alg; abstol = 1e-9, reltol = 1e-9, callback = (c1, c2))
        cond!(out, u, t, i) = (nv[] += 1; out[1] = u[1] - 0.3; out[2] = u[2]; nothing)
        function aff!(i, signs)
            push!(tv, i.t); push!(sv, copy(signs))
            signs[1] == 1 && (i.u[1] = 0.0)
            signs[2] == -1 && (i.u[2] = 1.0)
            nothing
        end
        vc = VectorContinuousCallback(cond!, aff!, 2)
        s2 = solve(prob, alg; abstol = 1e-9, reltol = 1e-9, callback = vc)
        # x crosses 0.3 rising at 0.3, 0.6, 0.9; y falls through 0 at 0.5 (and 1.0, the span's end)
        @test length(tv) == length(ts) && maximum(abs.(tv .- ts)) < 1e-12
        @test isapprox(tv[1:4], [0.3, 0.5, 0.6, 0.9]; atol = 1e-8)
        @test sv[1:4] == [Int8[1, 0], Int8[0, -1], Int8[1, 0], Int8[1, 0]]
        @test s2.t == s1.t && s2.u == s1.u && s2.naccept == s1.naccept
        @test nv[] > 0
    end
    # the user's own callback type, by the trait
    struct TwoLevels; hits::Vector{Float64}; end
    TraitDifferentialEquations.vector_event_length(::TwoLevels) = 2
    TraitDifferentialEquations.vector_event_condition!(c::TwoLevels, out::Vector{Float64}, u::Vector{Float64}, t::Float64, i) =
        (out[1] = u[1] - 0.25; out[2] = u[1] - 0.75; nothing)
    TraitDifferentialEquations.vector_event_affect!(c::TwoLevels, i, signs::Vector{Int8}) = (push!(c.hits, i.t); nothing)
    @implement VectorEventCallback{Vector{Float64},Float64,Vector{Float64}} TwoLevels
    c = TwoLevels(Float64[])
    solve(prob, Tsit5(); abstol = 1e-9, reltol = 1e-9, callback = c)
    @test isapprox(c.hits, [0.25, 0.75]; atol = 1e-8)
    @test_throws ArgumentError VectorContinuousCallback(cond -> nothing, (i, s) -> nothing, 0)
end
