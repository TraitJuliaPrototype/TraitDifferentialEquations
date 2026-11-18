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
end
