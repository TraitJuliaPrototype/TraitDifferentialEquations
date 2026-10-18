# A right-hand side as `Callable{RHSSig{P, Float64}, Nothing}` in its two modes: `DynamicCall`
# (the default: one solver per parameter type) and `CompiledCall` (`compiled_call(f)`: the solver
# compiled for f, the signature still checked), beside the model type (review 2026-10-18,
# example 10; the user: "We need compiled call to have an ODE RHS that has the right type
# signature"). Runs inside runtests.jl (which defines lorenz!, U0, P0, REF).
"The exception `f()` throws, or `nothing`."
caught(f) = try
    f(); nothing
catch err
    err
end

@testset "CompiledCall: a right-hand side with a declared signature" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    sd = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
    sc = solve(ODEProblem(compiled_call(lorenz!), U0, (0.0, 1.0), P0), Tsit5(); abstol = 1e-12, reltol = 1e-12)
    @test sc.u == sd.u && isapprox(sc.u[end], REF; rtol = 1e-10)
    # the long form fixes the signature as well as the mode
    S = TraitDifferentialEquations.RHSSig{Vector{Float64},Float64}
    @test solve(ODEProblem(Callable{S,Nothing,CompiledCall}(lorenz!), U0, (0.0, 1.0), P0), Tsit5()).u == solve(prob, Tsit5()).u
    @test solve(ODEProblem(dynamic_call(lorenz!), U0, (0.0, 1.0), P0), Tsit5()).u == solve(prob, Tsit5()).u
    # a closure with captured state, and a problem without parameters
    k = 0.5
    s0 = solve(ODEProblem(compiled_call((du, u, p, t) -> (du .= -k .* u; nothing)), [1.0], (0.0, 1.0)), Tsit5(); abstol = 1e-10, reltol = 1e-10)
    @test isapprox(s0.u[end][1], exp(-0.5); rtol = 1e-8)
    # with callbacks: the callbacks stay dynamic
    kick = DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.6, i -> (i.u[1] += 1.0; nothing))
    @test solve(ODEProblem(compiled_call(lorenz!), U0, (0.0, 1.0), P0), Tsit5(); callback = kick).u ==
          solve(prob, Tsit5(); callback = kick).u
    # the mode is a choice of speed, never of correctness: where the library serves no compiled
    # entry (Float32 states, Rodas5, gradients), the function runs as a dynamic call, with the same results
    P32, U32 = Float32.(P0), Float32.(U0)
    @test solve(ODEProblem(compiled_call(lorenz!), U32, (0.0, 1.0), P32), Tsit5()).u == solve(ODEProblem(lorenz!, U32, (0.0, 1.0), P32), Tsit5()).u
    @test solve(ODEProblem(compiled_call(lorenz!), U0, (0.0, 1.0), P0), Rodas5()).u == solve(prob, Rodas5()).u
    @test gradient(p -> sum(solve(ODEProblem(compiled_call(lorenz!), U0, (0.0, 1.0), p), Tsit5()).u[end]), copy(P0)) ==
          gradient(p -> sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5()).u[end]), copy(P0))
end

@testset "CompiledCall: a wrong signature is refused" begin
    oop(u, p, t) = -u                       # SciML's out-of-place form: no method at the declared arguments
    ret!(du, u, p, t) = (du .= -u)          # returns du, not nothing
    # a wrapper that names another signature: refused at the entry, before the solve
    e = caught(() -> solve(ODEProblem(Callable{Tuple{Vector{Float64}},Nothing,CompiledCall}(lorenz!), U0, (0.0, 1.0), P0), Tsit5()))
    @test e isa TraitBoundError && occursin("Callable{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, Nothing}", sprint(showerror, e))
    # dynamic mode: refused at the first call, as a MethodError for the declared arguments
    e = caught(() -> solve(ODEProblem(oop, U0, (0.0, 1.0), P0), Tsit5()))
    @test e isa MethodError && e.f === oop && typeof(e.args) == Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Float64}
    # and the declared result, asserted at the call
    e = caught(() -> solve(ODEProblem(ret!, U0, (0.0, 1.0), P0), Tsit5()))
    @test e isa TypeError && e.expected === Nothing
    # compiled mode: the model type for this function is checked at its first use, and the check
    # warns that `rhs!` cannot return; then the first call raises the same errors
    @test_logs (:warn, r"does not conform") match_mode = :any begin
        e = caught(() -> solve(ODEProblem(compiled_call(oop), U0, (0.0, 1.0), P0), Tsit5()))
    end
    @test e isa MethodError && e.f === oop
    @test_logs (:warn, r"does not conform") match_mode = :any begin
        e = caught(() -> solve(ODEProblem(compiled_call(ret!), U0, (0.0, 1.0), P0), Tsit5()))
    end
    @test e isa TypeError && e.expected === Nothing
    # wanted (TraitWorld): both refused at the entry, naming the function and the declared signature
    @test_broken caught(() -> solve(ODEProblem(compiled_call(oop), U0, (0.0, 1.0), P0), Tsit5())) isa TraitBoundError
end
