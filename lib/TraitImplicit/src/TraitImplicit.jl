"""
    TraitImplicit

A fully implicit method as a solver package over `ODECore` (stage D of SciML's move 3, E-221): implicit Euler,
its stages solved by any `NonlinearSolver` (Newton by default), which solves its linear systems by any
`LinearSolver`. Three open sets nested; this package knows only the core's traits.
"""
module TraitImplicit
using TraitWorld
@trait_module
using ODECore: CoreProblem, CoreSolution, SolverOptions, ODEAlgorithm, NonlinearSolver, NLCache, nlcache, nlstage!, Newton
import ODECore: solve_core
export ImplicitEuler
"""
    ImplicitEuler(; dt = 0.01, nlsolve = Newton())

Implicit Euler at a fixed step `dt` (the last step shortened to end at the span's end); each step's stage
`x = u + dt f(x, t + dt)` solved by `nlsolve`, any `NonlinearSolver`.
"""
struct ImplicitEuler{N <: NonlinearSolver}
    dt::Float64
    nlsolve::N
end
ImplicitEuler(; dt = 0.01, nlsolve = Newton()) = ImplicitEuler(dt, nlsolve)
function ie_solve(a::ImplicitEuler, m::CoreProblem, callbacks::Vector{Any}, opts::SolverOptions)::CoreSolution
    isempty(callbacks) || throw(ArgumentError("ImplicitEuler takes no callbacks"))
    u0 = m.u0
    u0 isa Vector{Float64} || throw(ArgumentError("ImplicitEuler takes Float64 states"))
    t0, tf = m.tspan
    n = length(u0)
    c = nlcache(a.nlsolve, n)
    ts = Float64[t0]
    us = Vector{Float64}[copy(u0)]
    u = copy(u0)
    x = copy(u0)
    t = t0
    while t < tf
        dt = min(a.dt, tf - t)
        copyto!(x, u)
        nlstage!(c, m, x, u, t + dt, dt) || return CoreSolution{Float64}(ts, us, :ConvergenceFailure, length(ts) - 1, 0, 0)
        copyto!(u, x)
        t += dt
        push!(ts, t)
        push!(us, copy(u))
    end
    return CoreSolution{Float64}(ts, us, :Success, length(ts) - 1, 0, 0)
end
@implement ODEAlgorithm ImplicitEuler{N} where {N} begin
    solve_core(a, m, callbacks, opts) = ie_solve(a, m, callbacks, opts)
end
end
