# W-32 R3 (E-242), census item 11: nonlinear problems with an open set of solvers (lib/TraitNonlinearSolve).
# Runs inside runtests.jl (which defines lorenz!, U0, P0, caught).
import TraitDifferentialEquations.TraitNonlinearSolve as TNS
using TraitDifferentialEquations.TraitNonlinearSolve: NLCoreProblem, NLCoreSolution, NLOptions, NLStats, nl_residual!, solve_nl

sq!(r, u, p) = (r .= u .^ 2 .- p; nothing)
sqjac!(J, u, p) = (fill!(J, 0.0); for i in eachindex(u); J[i, i] = 2u[i]; end; nothing)
"A problem type of its own: it implements the residual (its Jacobian by the trait's default, finite differences)."
struct Cubic
    u0::Vector{Float64}
    p::Float64
end
TNS.residual!(m::Cubic, r::Vector{Float64}, u::Vector{Float64}) = (r[1] = u[1]^3 - m.p; r[2] = u[2] - u[1]; nothing)
@implement NonlinearProblemTrait Cubic
"An algorithm of the user's own, from outside the package: a damped fixed-point iteration `u ← u - ω r(u)`."
struct Picard
    ω::Float64
end
function picard(a::Picard, m::NLCoreProblem, opts::NLOptions)
    u = copy(m.u0); r = similar(u); nf = 0
    for k in 1:opts.maxiters
        nl_residual!(m, r, u); nf += 1
        maximum(abs, r) <= 1e-12 && return NLCoreSolution{Float64}(u, r, :Success, NLStats(k, nf, 0, 0))
        u .-= a.ω .* r
    end
    return NLCoreSolution{Float64}(u, r, :MaxIters, NLStats(opts.maxiters, nf, 0, 0))
end
@implement NonlinearAlgorithm Picard begin
    solve_nl(a, m, opts) = picard(a, m, opts)
end
"A callable model that carries its system object, as NetworkDynamics' `NetworkFixedT` does."
struct Relax
    target::Vector{Float64}
end
(m::Relax)(du, u, p, t) = (du .= p[1] .* (m.target .- u); nothing)
@implement HasSystem Relax begin
    system(m) = m
end

@testset "Nonlinear problems (census item 11)" begin
    p = [2.0, 3.0, 5.0]
    prob = NonlinearProblem(sq!, [1.0, 1.0, 1.0], p)
    for alg in (NewtonRaphson(), NewtonRaphson(linesearch = true), TrustRegion(), NonlinearPolyalg(), NewtonRaphson(linsolve = TraitDifferentialEquations.DenseLU()))
        s = solve(prob, alg)
        @test s.retcode === :Success && successful_retcode(s) && s.u ≈ sqrt.(p) && maximum(abs, s.resid) <= 1e-12
    end
    # the default algorithm, the package's own front end, and a Jacobian given
    @test solve(prob).u ≈ sqrt.(p)
    @test TNS.solve(prob, NewtonRaphson()).u ≈ sqrt.(p)
    sj = solve(NonlinearProblem(NonlinearFunction(sq!; jac = sqjac!), [1.0, 1.0, 1.0], p), NewtonRaphson())
    @test sj.retcode === :Success && sj.u ≈ sqrt.(p) && sj.stats.njacs >= 1
    # the Jacobian by the library's duals equals the analytic one
    J = zeros(3, 3); nl_jacobian!(prob, J, [1.0, 2.0, 3.0])
    @test J ≈ [2.0 0 0; 0 4.0 0; 0 0 6.0]
    # the solution keeps its problem, with no axis for the function
    @test sj.prob.p === p && typeof(sj) === typeof(solve(prob, NewtonRaphson()))
    # a problem type of its own (finite differences), and an algorithm of the user's own (the open set)
    sc = solve(Cubic([1.0, 0.0], 8.0), NewtonRaphson())
    @test sc.retcode === :Success && sc.u ≈ [2.0, 2.0]
    sp = solve(NonlinearProblem(sq!, [1.0], [1.21]), Picard(0.3))
    @test sp.retcode === :Success && sp.u ≈ [1.1]
    @test NonlinearPolyalg(Picard(0.3), NewtonRaphson()) isa NonlinearPolyalg
    # Newton diverges far from the root where the trust region converges: arctan
    at!(r, u, p) = (r[1] = atan(u[1]); nothing)
    pa = NonlinearProblem(at!, [3.0])
    @test !successful_retcode(solve(pa, NewtonRaphson()))
    st = solve(pa, TrustRegion())
    @test st.retcode === :Success && abs(st.u[1]) < 1e-10
    sp2 = solve(pa, NonlinearPolyalg())
    @test sp2.retcode === :Success && sp2.stats.nsteps > st.stats.nsteps
    # a singular Jacobian: Newton fails cleanly
    z!(r, u, p) = (r[1] = u[1]^2 + 1.0; nothing)
    @test solve(NonlinearProblem(z!, [0.0]), NewtonRaphson()).retcode === :ConvergenceFailure
    @test !successful_retcode(solve(NonlinearProblem(z!, [0.0])))
    # Float64 states only
    @test caught(() -> solve(NonlinearProblem(sq!, Float32[1, 1], Float32[2, 3]), NewtonRaphson())) isa ArgumentError
end

@testset "Steady states (SteadyStateProblem)" begin
    m = Relax([1.0, -2.0])
    ss = SteadyStateProblem(m, [0.0, 0.0], [0.5])
    s = solve(ss, NewtonRaphson())
    @test s.retcode === :Success && s.u ≈ [1.0, -2.0] && getsystem(s) === m && getsystem(ss) === m
    # at a time: a right-hand side that depends on it
    ft!(du, u, p, t) = (du[1] = t - u[1]; nothing)
    @test solve(SteadyStateProblem(ft!, [0.0]; t = 2.5)).u ≈ [2.5]
    # from an ODEFunction (its Jacobian at the time) and from an ODE problem of any type
    jt!(J, u, p, t) = (J[1, 1] = -1.0; nothing)
    @test solve(SteadyStateProblem(ODEFunction(ft!; jac = jt!), [0.0]; t = 1.5), NewtonRaphson()).u ≈ [1.5]
    s2 = solve(SteadyStateProblem(ODEProblem(m, [0.0, 0.0], (0.0, 1.0), [0.5])), NewtonRaphson())
    @test s2.retcode === :Success && s2.u ≈ [1.0, -2.0]
    # Lorenz's fixed point at (√(β(ρ-1)), √(β(ρ-1)), ρ-1)
    sl = solve(SteadyStateProblem(lorenz!, [8.0, 8.0, 27.0], P0), TrustRegion())
    c = sqrt(P0[3] * (P0[2] - 1))
    @test sl.retcode === :Success && sl.u ≈ [c, c, P0[2] - 1]
end
