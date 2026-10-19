# Route A: models and callbacks as conforming types (examples/LorenzModels), beside the function
# form; and the user's mistakes at the conformance, where each is caught and what the user sees
# (review 2026-10-18, example 5). Runs inside runtests.jl (which defines lorenz!, U0, P0, REF).
isdefined(Main, :toplevel) ||
    (toplevel(ex) = Core.eval(Main, ex isa Expr && ex.head === :block ? Expr(:toplevel, ex.args...) : ex))
"The error `ex` raises at the top level, unwrapped from `LoadError`s, or `nothing`."
function thrown(ex)
    e = try
        toplevel(ex); nothing
    catch err
        err
    end
    while e isa LoadError
        e = e.error
    end
    return e
end
errtext(e) = sprint(showerror, e)

include(joinpath(@__DIR__, "..", "examples", "LorenzModels", "src", "LorenzModels.jl"))
using .LorenzModels

@testset "Route A: a model type with its own cache (LorenzModels)" begin
    m = Lorenz(10.0, 28.0, 8 / 3)
    sA = solve(m, U0, (0.0, 1.0), Tsit5(); abstol = 1e-12, reltol = 1e-12)
    @test isapprox(sA.u[end], REF; rtol = 1e-10)
    # the same arithmetic as the function form: the same steps, the same values
    sB = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5())
    @test solve(m, U0, (0.0, 1.0), Tsit5()).u == sB.u
    # Rodas5 with the model's own Jacobian
    sR = solve(m, U0, (0.0, 1.0), Rodas5(); abstol = 1e-10, reltol = 1e-10)
    @test sR.retcode === :Success && isapprox(sR.u[end], REF; rtol = 1e-8)
    # a Float32 model takes a Float32 state
    s32 = solve(Lorenz(10.0f0, 28.0f0, 8.0f0 / 3), Float32.(U0), (0.0, 1.0), Tsit5())
    @test eltype(s32.u[end]) === Float32 && isapprox(Float64.(s32.u[end]), sB.u[end]; rtol = 1e-2)
    # the integrator's type names the model, nothing else
    I = string(TraitDifferentialEquations.ODECore.Tsit5Integrator{Float64,Lorenz{Float64}})
    @test endswith(I, "Tsit5Integrator{Float64, Lorenz{Float64}}") && length(I) < 100
end

@testset "Route A: callback types (Bounce, RaiseSigma)" begin
    m = Lorenz(10.0, 28.0, 8 / 3)
    sb = solve(m, U0, (0.0, 1.0), Tsit5(); callback = Bounce(5.0))
    # the same callback as two closures (the function route): the same solution
    wall = DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.u[1] = 2 * 5.0 - i.u[1]; nothing))
    sw = solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5(); callback = wall)
    @test sb.u == sw.u && sb.u != solve(m, U0, (0.0, 1.0), Tsit5()).u
    # a callback that writes a parameter of the model (integ.p is the model)
    m2 = Lorenz(10.0, 28.0, 8 / 3)
    sr = solve(m2, U0, (0.0, 1.0), Tsit5(); callback = RaiseSigma(0.5, 12.0))
    @test m2.σ == 12.0
    p = copy(P0)
    raise = DiscreteCallback((u, t, i) -> t >= 0.5 && i.p[1] != 12.0, i -> (i.p[1] = 12.0; nothing))
    @test sr.u == solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); callback = raise).u
    # both, with Rodas5
    m3 = Lorenz(10.0, 28.0, 8 / 3)
    s3 = solve(m3, U0, (0.0, 1.0), Rodas5(); callback = (Bounce(5.0), RaiseSigma(0.5, 12.0)))
    @test s3.retcode === :Success && m3.σ == 12.0
end

@testset "Route A: gradients through Tsit5" begin
    lossA(p) = sum(solve(Lorenz(p[1], p[2], p[3]), eltype(p).(U0), (0.0, 1.0), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end])
    lossB(p) = sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end])
    gA = gradient(lossA, copy(P0))
    @test gA == gradient(lossB, copy(P0))              # the same arithmetic in both routes
    h = 1e-6
    @test isapprox(gA, [(lossA(P0 .+ h .* (1:3 .== k)) - lossA(P0 .- h .* (1:3 .== k))) / 2h for k in 1:3]; rtol = 1e-4)
    # forgetting to promote the state: refused at the operation, naming the declared arguments
    e = try
        gradient(p -> sum(solve(Lorenz(p[1], p[2], p[3]), U0, (0.0, 1.0), Tsit5()).u[end]), copy(P0)); nothing
    catch err
        err
    end
    @test e !== nothing && occursin("is declared for arguments", errtext(e)) && occursin("rhs!", errtext(e))
end

@testset "Route A: errors at the conformance" begin
    # (b) a missing operation: refused at @implement
    e = thrown(quote
        struct NoRhs; k::Float64; end
        @implement ODEModel{Float64} NoRhs
    end)
    @test e isa TraitWorld.ImplementError
    @test occursin("rhs!", errtext(e))          # the conformance error names the operation
    # (b') one operation implemented, the required one missing: refused, naming it
    e = thrown(quote
        struct OnlyJac; k::Float64; end
        TraitDifferentialEquations.jacobian!(m::OnlyJac, J::Matrix{Float64}, u::Vector{Float64}, t::Float64) = (J .= -m.k; nothing)
        @implement ODEModel{Float64} OnlyJac
    end)
    @test e isa TraitWorld.ImplementError && occursin("rhs!", errtext(e))
    # (c) a wrong result type (the last expression returns `du`): refused, naming the declared type
    e = thrown(quote
        struct BadRet; k::Float64; end
        TraitDifferentialEquations.rhs!(m::BadRet, du::Vector{Float64}, u::Vector{Float64}, t::Float64) = (du .= -m.k .* u)
        @implement ODEModel{Float64} BadRet
    end)
    @test e isa TraitWorld.ImplementError
    @test occursin("`rhs!`", errtext(e)) && occursin("returns `Vector{Float64}`, declared `Nothing`", errtext(e))
    # (c) in the block form: the same check
    e = thrown(quote
        struct BlockBad; k::Float64; end
        @implement ODEModel{Float64} BlockBad begin
            rhs!(m, du, u, t) = (du .= -m.k .* u)
        end
    end)
    @test e isa TraitWorld.ImplementError && occursin("declared `Nothing`", errtext(e))
    # (c') a wrong argument type: refused, but the message does not name the declared signature
    e = thrown(quote
        struct BadArg; k::Float64; end
        TraitDifferentialEquations.rhs!(m::BadArg, du::Vector{Float64}, u::Vector{Float32}, t::Float64) = (du .= -m.k .* u; nothing)
        @implement ODEModel{Float64} BadArg
    end)
    @test e isa TraitWorld.ImplementError
    @test occursin("Vector{Float32}", errtext(e))  # and the declared argument types
    # (a) a Float64 cache in a model generic in T: the family conformance is checked at each
    # member's first use, but inference does not prove `Float64(::Dual)` fails inside broadcasting,
    # so the mistake passes the check and surfaces at run time, in Base's broadcasting frames
    toplevel(quote
        struct LorenzF{T}
            σ::T; ρ::T; β::T
            cache::Vector{Float64}
        end
        LorenzF(σ::T, ρ::T, β::T) where {T} = LorenzF{T}(σ, ρ, β, zeros(3))
        function TraitDifferentialEquations.rhs!(m::LorenzF{T}, du::Vector{T}, u::Vector{T}, t::Float64) where {T}
            c = m.cache; c .= u
            du[1] = m.σ * (c[2] - c[1]); du[2] = c[1] * (m.ρ - c[3]) - c[2]; du[3] = c[1] * c[2] - m.β * c[3]
            nothing
        end
        @implement ODEModel{T} LorenzF{T} where {T}
    end)
    @test isapprox(solve(LorenzF(10.0, 28.0, 8 / 3), U0, (0.0, 1.0), Tsit5()).u[end], solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5()).u[end])
    @test_throws MethodError gradient(p -> sum(solve(LorenzF(p[1], p[2], p[3]), eltype(p).(U0), (0.0, 1.0), Tsit5()).u[end]), copy(P0))
end
