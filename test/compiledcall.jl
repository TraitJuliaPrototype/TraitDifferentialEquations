# A right-hand side kept at `Callable{RHSSig{P, T}, Nothing}`, `Dynamic` (the default: one solver
# per state and parameter type) or `Compiled` (`compiled_for(f)`: the solver compiled for f, the
# result still checked), beside the problem type (review 2026-10-18,
# example 10; the user: "We need compiled call to have an ODE RHS that has the right type
# signature"). Runs inside runtests.jl (which defines lorenz!, U0, P0, REF).
"The exception `f()` throws, or `nothing`."
caught(f) = try
    f(); nothing
catch err
    err
end

@testset "Compiled: a right-hand side with a declared signature" begin
    prob = ODEProblem(lorenz!, U0, (0.0, 1.0), P0)
    sd = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
    sc = solve(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), P0), Tsit5(); abstol = 1e-12, reltol = 1e-12)
    @test sc.u == sd.u && isapprox(sc.u[end], REF; rtol = 1e-10)
    # a stored function, `Dynamic{Callable{…}}`, and the caller's `with_dynamic`
    S = TraitDifferentialEquations.RHSSig{Vector{Float64},Float64}
    @test solve(ODEProblem(Dynamic{Callable{S,Nothing}}(lorenz!), U0, (0.0, 1.0), P0), Tsit5()).u == solve(prob, Tsit5()).u
    @test solve(ODEProblem(with_dynamic(lorenz!), U0, (0.0, 1.0), P0), Tsit5()).u == solve(prob, Tsit5()).u
    # a closure with captured state, and a problem without parameters
    k = 0.5
    s0 = solve(ODEProblem(compiled_for((du, u, p, t) -> (du .= -k .* u; nothing)), [1.0], (0.0, 1.0)), Tsit5(); abstol = 1e-10, reltol = 1e-10)
    @test isapprox(s0.u[end][1], exp(-0.5); rtol = 1e-8)
    # with callbacks: the callbacks stay dynamic
    kick = DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.6, i -> (i.u[1] += 1.0; nothing))
    @test solve(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), P0), Tsit5(); callback = kick).u ==
          solve(prob, Tsit5(); callback = kick).u
    # the attribute is a choice of speed, never of correctness: Float32 states, Rodas5 and gradients
    # give the same results compiled and dynamic
    P32, U32 = Float32.(P0), Float32.(U0)
    @test solve(ODEProblem(compiled_for(lorenz!), U32, (0.0, 1.0), P32), Tsit5()).u == solve(ODEProblem(lorenz!, U32, (0.0, 1.0), P32), Tsit5()).u
    @test solve(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), P0), Rodas5()).u == solve(prob, Rodas5()).u
    @test gradient(p -> sum(solve(ODEProblem(compiled_for(lorenz!), U0, (0.0, 1.0), p), Tsit5()).u[end]), copy(P0)) ==
          gradient(p -> sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5()).u[end]), copy(P0))
end

@testset "Compiled: a wrong signature is refused" begin
    oop(u, p, t) = -u                       # SciML's out-of-place form: no method at the declared arguments
    ret!(du, u, p, t) = (du .= -u)          # returns du, not nothing
    # a stored function of another signature: refused when the problem is made
    e = caught(() -> solve(ODEProblem(Dynamic{Callable{Tuple{Vector{Float64}},Nothing}}(lorenz!), U0, (0.0, 1.0), P0), Tsit5()))
    @test e isa TraitBoundError && occursin("Callable{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, Nothing}", sprint(showerror, e))
    # `Dynamic`: refused at the first call, as a MethodError for the declared arguments
    e = caught(() -> solve(ODEProblem(oop, U0, (0.0, 1.0), P0), Tsit5()))
    @test e isa MethodError && e.f === oop && typeof(e.args) == Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Float64}
    # and the declared result, asserted at the call
    e = caught(() -> solve(ODEProblem(ret!, U0, (0.0, 1.0), P0), Tsit5()))
    @test e isa TypeError && e.expected === Nothing
    # `Compiled`: the problem type for this function is checked at its first use, and the check
    # warns that `rhs!` cannot return; then the first call raises the same errors
    @test_logs (:warn, r"does not conform") match_mode = :any begin
        e = caught(() -> solve(ODEProblem(compiled_for(oop), U0, (0.0, 1.0), P0), Tsit5()))
    end
    @test e isa MethodError && e.f === oop
    @test_logs (:warn, r"does not conform") match_mode = :any begin
        e = caught(() -> solve(ODEProblem(compiled_for(ret!), U0, (0.0, 1.0), P0), Tsit5()))
    end
    @test e isa TypeError && e.expected === Nothing
    # wanted (TraitWorld): both refused at the entry, naming the function and the declared signature
    @test_broken caught(() -> solve(ODEProblem(compiled_for(oop), U0, (0.0, 1.0), P0), Tsit5())) isa TraitBoundError
end
