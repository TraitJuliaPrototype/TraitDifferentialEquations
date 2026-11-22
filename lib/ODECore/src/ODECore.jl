"""
    ODECore

The core of TraitDifferentialEquations as a package of its own (SciML's move 3, E-159): the traits of
problems, algorithms, callbacks, integrators and solutions; the problem types and their front end; the
step-callback machinery; Tsit5. Solver packages implement `ODEAlgorithm` for their own types; the umbrella,
TraitDifferentialEquations, re-exports them.
"""
module ODECore
using TraitWorld
@trait_module
include("core.jl")
include("frontend.jl")

# The precompile workload of the core's own step machinery: Tsit5 at a `Vector{Float64}` state and parameters with
# one discrete and one continuous callback, the broad typical case. The callbacks' adapters and the events' code are
# this package's methods reached when the solve runs, through the callbacks' traits: an image keeps a package's code
# for another package only where that package's code calls it, so the umbrella's image cannot hold them (open issue
# 228); this package's image can.
_wl_rhs!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
if Base.generating_output(true)
    let prob = ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1), [10.0, 28.0, 8 / 3])
        cbs = (DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)),
               ContinuousCallback((u, t, i) -> u[1] - 1.5, i -> (i.u[2] += 0.5; nothing)))
        solve_ode(prob, Tsit5(), cbs, DEFAULT_OPTIONS)
    end
end
end
