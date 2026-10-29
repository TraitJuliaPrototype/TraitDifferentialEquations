"""
    TraitNonlinearSolve

Nonlinear problems with an open set of solvers (W-32, census item 11; the note
`writeup/plans/2026-10-29-tde-nonlinear-problems.md`), as SciML's NonlinearSolve and SteadyStateDiffEq are:
- `NonlinearProblemTrait{T, P}`, what a solver needs from a problem; TDE's `NonlinearProblem` is the canonical
  one, from `NonlinearProblem(f, u0, p)` with `f(r, u, p)`, or `SteadyStateProblem(f, u0, p; t)` with an ODE
  right-hand side `f(du, u, p, t)` at a fixed time;
- `NonlinearAlgorithm`, an open set: `NewtonRaphson`, `TrustRegion`, and `NonlinearPolyalg`, whose members are
  any algorithms; each takes any `LinearSolver` (the core's open set);
- `solve(prob, alg; abstol, reltol, maxiters)`, the package's front end; the umbrella's `solve` forwards to it.

A package over `ODECore`: it shares the core's optional traits (`HasSystem`, `HasJacobianPrototype`), its linear
solvers and its dual numbers.
"""
module TraitNonlinearSolve
using TraitWorld
using ODECore
using ODECore: LinearSolver, linfactor, DenseLU, FDual, DUAL_CHUNK, HasSystem, HasJacobianPrototype, getsystem,
               ODEFunction, ODEProblemTrait
import ODECore: system, jac_prototype
import ForwardDiff
import LinearAlgebra
export NonlinearProblemTrait, residual!, nl_jacobian!, NonlinearProblem, NonlinearFunction, SteadyStateProblem,
       NonlinearAlgorithm, solve_nl, NLCoreProblem, NLCoreSolution, NLOptions, nl_residual!, nl_core_jacobian!,
       NonlinearSolution, NLStats, NewtonRaphson, TrustRegion, NonlinearPolyalg, successful_retcode
@trait_module begin
include("problems.jl")
include("solvers.jl")
end

# The precompile workload: the solvers for plain functions with `Vector{Float64}` states, whatever the function
# (only the function itself is compiled, at its declared signature, when the problem is solved).
_wl_res!(r, u, p) = (r[1] = u[1]^2 - p[1]; r[2] = u[2] - u[1]; nothing)
_wl_ss!(du, u, p, t) = (du[1] = p[1] - u[1]; du[2] = u[1] - u[2]; nothing)
TraitWorld.@workload begin
    solve(NonlinearProblem(_wl_res!, [1.0, 1.0], [2.0]), NonlinearPolyalg())
    solve(SteadyStateProblem(_wl_ss!, [0.0, 0.0], [1.0]), NewtonRaphson())
    solve(NonlinearProblem(_wl_res!, [1.0, 1.0], [2.0]), TrustRegion())
end
end
