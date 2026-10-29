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
using ODECore: CoreProblem, CoreDualSig, FDual, CoreSolution, SolverOptions, StepView, ODEAlgorithm, ODEStepper,
               CoreIntegrator, core_rhs!, core_jacobian!, enter_step_callbacks, initialize_step_callbacks!, _add_tstop!,
               step_event_condition, step_condition, step_affect!, locate_events!, event_affect_dir!,
               _save_start!, _retcode, run_to_end!, LinearSolver, linfactor, DenseLU, Interpolant, dense_step,
               kept_interpolant, _steps
import ODECore: solve_core, init_core, advance!, core_solution, step_tstop!, step_terminate!, step_interp!, step_dt_reset!,
                interpolate, push_step!
import ForwardDiff
export Rodas5
@trait_module begin
include("rodas5.jl")
end
end
