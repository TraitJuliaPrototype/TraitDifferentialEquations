# A scripted interactive session (a correctness check, not a timing): a user at the prompt with
# a precompiled problem package, then a new right-hand side function, the same in compiled mode, a new
# problem type, a new callback closure and a new callback type, a new loss and a second one, Rodas5, and
# an edit of a function already used. Each entry is evaluated as the REPL evaluates it (soft
# scope, `Core.eval` in `Main`, the latest world), in order. Every result is checked where it is
# made, and printed as a `RESULT` line, so that the trait session and the plain one (PlainODE,
# PlainLorenzModels) can be compared line by line.
#
#     JULIA_LOAD_PATH="scripts/env:@stdlib" julia --startup-file=no scripts/session.jl trait > trait.log
#     JULIA_LOAD_PATH="scripts/env:@stdlib" julia --startup-file=no scripts/session.jl plain > plain.log
#     diff <(grep RESULT trait.log) <(grep RESULT plain.log)        # expected: no difference
using REPL
const LIB = get(ARGS, 1, "trait")
LIB in ("trait", "plain") || error("usage: session.jl trait|plain")

"Each entry: the libraries it is for, and its code. `@lib` stands for the solver package."
const ENTRIES = Tuple{Symbol,String}[
    (:trait, "using TraitDifferentialEquations, TraitWorld, LorenzModels; const Lib = TraitDifferentialEquations"),
    (:plain, "using PlainODE, PlainLorenzModels; const Lib = PlainODE"),
    (:both, """
     const nfail = Ref(0)
     check(name, ok) = (ok || (nfail[] += 1); println(ok ? "CHECK ok   " : "CHECK FAIL ", name))
     result(name, x) = println("RESULT ", name, " ", repr(x))
     fd(loss, p; h = 1e-6) = [(loss(p .+ h .* (1:length(p) .== k)) - loss(p .- h .* (1:length(p) .== k))) / 2h for k in 1:length(p)]
     """),
    # a problem type from the precompiled package, with its callbacks
    (:both, "m = LorenzProblem([1.0, 0.0, 0.0], (0.0, 10.0), 10.0, 28.0, 8 / 3); s1 = solve(m, Tsit5())"),
    (:both, "check(\"precompiled problem type\", s1.retcode === :Success); result(\"s1\", s1.u[end])"),
    (:both, "s1b = solve(LorenzProblem([1.0, 0.0, 0.0], (0.0, 10.0), 10.0, 28.0, 8 / 3), Tsit5(); callback = (Bounce(10.0), RaiseSigma(5.0, 12.0)))"),
    (:both, "check(\"precompiled callbacks\", s1b.retcode === :Success && s1b.u[end] != s1.u[end]); result(\"s1b\", s1b.u[end])"),
    # a new right-hand side function
    (:both, "rossler!(du, u, p, t) = (du[1] = -u[2] - u[3]; du[2] = u[1] + p[1] * u[2]; du[3] = p[2] + u[3] * (u[1] - p[3]); nothing)"),
    (:both, "const r0, rp = [1.0, 0.0, 0.0], [0.2, 0.2, 5.7]; prob = ODEProblem(rossler!, r0, (0.0, 10.0), rp)"),
    (:both, "s2 = solve(prob, Tsit5()); check(\"new function\", s2.retcode === :Success); result(\"s2\", s2.u[end])"),
    # the same function in compiled mode: the same values
    (:both, "s2c = solve(ODEProblem(compiled_for(rossler!), r0, (0.0, 10.0), rp), Tsit5()); check(\"compiled mode\", s2c.u == s2.u)"),
    # a new problem type at the prompt
    (:both, "struct Rossler; u0::Vector{Float64}; tspan::Tuple{Float64,Float64}; p::Vector{Float64}; end"),
    (:both, "Lib.rhs!(m::Rossler, du::Vector{Float64}, u::Vector{Float64}, t::Float64) = (p = m.p; du[1] = -u[2] - u[3]; du[2] = u[1] + p[1] * u[2]; du[3] = p[2] + u[3] * (u[1] - p[3]); nothing)"),
    (:trait, "@implement ODEProblemTrait Rossler"),
    (:both, "s3 = solve(Rossler(r0, (0.0, 10.0), rp), Tsit5()); check(\"new problem type\", s3.u == s2.u)"),
    # a new callback closure
    (:both, "probL = ODEProblem(rossler!, r0, (0.0, 60.0), rp); reset = DiscreteCallback((u, t, i) -> u[3] > 5.0, i -> (i.u[3] = 0.0; nothing))"),
    (:both, "s4 = solve(probL, Tsit5(); callback = reset); check(\"new callback closure\", s4.u[end] != solve(probL, Tsit5()).u[end]); result(\"s4\", s4.u[end])"),
    # a new callback type at the prompt, the same behaviour: the same values
    (:both, "struct ResetZ; lim::Float64; end"),
    (:trait, """
     Lib.condition(cb::ResetZ, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView}) = u[3] > cb.lim
     Lib.affect!(cb::ResetZ, integ::Dynamic{IntegratorView}) = (integ.u[3] = 0.0; nothing)
     """),
    (:plain, """
     Lib.condition(cb::ResetZ, u, t, integ) = u[3] > cb.lim
     Lib.affect!(cb::ResetZ, integ) = (integ.u[3] = 0.0; nothing)
     """),
    (:trait, "@implement Callback ResetZ"),
    (:both, "s5 = solve(probL, Tsit5(); callback = ResetZ(5.0)); check(\"new callback type\", s5.u == s4.u)"),
    (:both, "s5b = solve(Rossler(r0, (0.0, 60.0), rp), Tsit5(); callback = (ResetZ(5.0), reset)); check(\"two callbacks, problem type\", s5b.u == s4.u)"),
    # a new loss, through the new function
    (:both, "loss(p) = sum(abs2, solve(ODEProblem(rossler!, r0, (0.0, 2.0), p), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end])"),
    (:both, "g1 = gradient(loss, copy(rp)); check(\"new loss\", isapprox(g1, fd(loss, rp); rtol = 1e-4)); result(\"g1\", g1)"),
    # a second loss closure, through the precompiled problem type
    (:both, "w = 0.5; loss2(p) = w * solve(LorenzProblem(eltype(p).([1.0, 0.0, 0.0]), (0.0, 1.0), p[1], p[2], p[3]), Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end][1]"),
    (:both, "g2 = gradient(loss2, [10.0, 28.0, 8 / 3]); check(\"second loss\", isapprox(g2, fd(loss2, [10.0, 28.0, 8 / 3]); rtol = 1e-4)); result(\"g2\", g2)"),
    # Rodas5: the function (Jacobian by duals) and the type (by finite differences) agree
    (:both, "s7 = solve(prob, Rodas5(); abstol = 1e-10, reltol = 1e-10); s8 = solve(Rossler(r0, (0.0, 10.0), rp), Rodas5(); abstol = 1e-10, reltol = 1e-10)"),
    (:both, "check(\"Rodas5\", isapprox(s7.u[end], s8.u[end]; rtol = 1e-6) && isapprox(s7.u[end], solve(prob, Tsit5(); abstol = 1e-10, reltol = 1e-10).u[end]; rtol = 1e-6)); result(\"s7\", s7.u[end])"),
    # a Float32 state
    (:both, "s9 = solve(ODEProblem(rossler!, Float32.(r0), (0.0, 10.0), Float32.(rp)), Tsit5()); check(\"Float32\", eltype(s9.u[end]) === Float32); result(\"s9\", s9.u[end])"),
    # an edit of a function already used, in both modes: the next solve uses the new definition
    (:both, "rossler!(du, u, p, t) = (du[1] = -u[2] - u[3]; du[2] = u[1] + p[1] * u[2]; du[3] = p[2] + u[3] * (u[1] - 2 * p[3]); nothing)"),
    (:both, "s10 = solve(prob, Tsit5()); s10c = solve(ODEProblem(compiled_for(rossler!), r0, (0.0, 10.0), rp), Tsit5())"),
    (:both, "s10ref = solve(Rossler(r0, (0.0, 10.0), [0.2, 0.2, 11.4]), Tsit5())"),
    (:both, "check(\"edited function, dynamic\", s10.u == s10ref.u); check(\"edited function, compiled\", s10c.u == s10.u); result(\"s10\", s10.u[end])"),
    (:both, "println(\"SESSION \", nfail[] == 0 ? \"passed\" : \"FAILED (\$(nfail[]))\")"),
]

for (lib, code) in ENTRIES
    lib === :both || lib === Symbol(LIB) || continue
    ex = Meta.parseall(code)
    try
        Base.invokelatest(Core.eval, Main, REPL.softscope(ex))
    catch e
        println("CHECK FAIL entry raised: ", first(split(code, '\n')))
        showerror(stdout, e, catch_backtrace()); println()
        isdefined(Main, :nfail) && (Main.nfail[] += 1)
    end
end
