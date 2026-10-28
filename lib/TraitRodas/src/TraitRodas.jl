"""
    TraitRodas

Rodas5, a solver package of its own over `ODECore` (SciML's move 3, E-159): it owns `Rodas5`, its integrator
and its Jacobian by duals, and implements the core's `ODEAlgorithm` and `StepView` for them. The core is not
edited for it. What it imports from the core below is what a solver package needs from a core.
"""
module TraitRodas
using TraitWorld
using ODECore
# the solver API: the core's problem and its calls, the dual numbers the core keys its problems' functions
# at, the step-callback machinery, and the step view's operations, which this package implements for its integrator
using ODECore: CoreProblem, CoreDualSig, FDual, ODESolution, SolverOptions, StepView, ODEAlgorithm,
               core_rhs!, core_jacobian!, enter_step_callbacks, _add_tstop!, event_time, step_event_condition,
               step_condition, step_affect!, step_event_affect!,
               LinearSolver, linfactor, DenseLU
import ODECore: solve_core, step_tstop!, step_terminate!, step_interp!
import ForwardDiff
export Rodas5
@trait_module begin
include("rodas5.jl")
end
end
