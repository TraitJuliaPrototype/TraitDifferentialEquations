"""
    TraitNLSolvers

A nonlinear solver package over `ODECore` (stage D of SciML's move 3, E-221), as OrdinaryDiffEq's NLsolve
algorithms are: a simplified Newton, the Jacobian taken and factorized once per stage. It knows only the
core's traits; the implicit methods do not know it.
"""
module TraitNLSolvers
using TraitWorld
using ODECore: NonlinearSolver, NLCache, LinearSolver, DenseLU, newton_stage!
import ODECore: nlcache, nlstage!
export SimplifiedNewton
@trait_module begin
"Newton with the Jacobian frozen at the stage's first iterate, its linear systems by `linsolve`."
struct SimplifiedNewton{L <: LinearSolver}
    linsolve::L
    maxiters::Int
    tol::Float64
end
SimplifiedNewton(; linsolve = DenseLU(), maxiters = 20, tol = 1e-10) = SimplifiedNewton(linsolve, maxiters, tol)
struct SimplifiedNewtonCache
    ls::LinearSolver
    maxiters::Int
    tol::Float64
    J::Matrix{Float64}
    r::Vector{Float64}
    dx::Vector{Float64}
    fx::Vector{Float64}
end
@implement NLCache SimplifiedNewtonCache begin
    nlstage!(c, m, x, z, t, γdt) = newton_stage!(c.ls, c.J, c.r, c.dx, c.fx, c.maxiters, c.tol, true, m, x, z, t, γdt)
end
@implement NonlinearSolver SimplifiedNewton{L} where {L} begin
    nlcache(nl, n) = SimplifiedNewtonCache(nl.linsolve, nl.maxiters, nl.tol, zeros(n, n), zeros(n), zeros(n), zeros(n))
end
end
end
