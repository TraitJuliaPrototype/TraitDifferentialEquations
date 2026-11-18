# TraitDifferentialEquations

ODE solvers written with traits, with SciML's interface: `ODEFunction`, `ODEProblem`, `solve`, `init`, `remake`, callbacks, symbolic access and gradients. A problem, an algorithm, a callback, an integrator and a solution are traits, so a package adds a problem type, a solver or a callback type by a membership. The solver's code is compiled once per state and parameter type, not per user function, unless the caller asks for code per function.

## The packages

The umbrella `TraitDifferentialEquations` (`src/`) re-exports the packages under `lib/`, as DifferentialEquations.jl does:
- **`ODECore`:** the core. The traits `ODEProblemTrait{T, P}`, `IntegratorView{U, T, P}`, `Callback{U, T, P}`, `EventCallback{U, T, P}`, `StepView`, `StepCallback`, `StepEvent`, `Interpolant{U, T}`, `CoreIntegrator`, `HasMassMatrix`, `HasJacobianPrototype`, `HasCallbacks`, `ODEAlgorithm`, `ODEStepper <: ODEAlgorithm`, `LinearSolver`, `NLCache`, `NonlinearSolver`; the front end (`ODEFunction`, `ODEProblem`, the problem kept by the solution); Tsit5 with its own interpolant; the drivers (`solve_ode`, `init_ode`, `step!`, `solve!`).
- **`TraitRodas`:** Rodas5 with its Jacobian by duals, a solver package over the core.
- **`TraitLinSolvers`, `TraitNLSolvers`, `TraitImplicit`:** a linear solver, a nonlinear solver and an implicit method, each knowing only the core's traits.
- **`TraitIndexing`:** symbolic access, SymbolicIndexingInterface's part: `IndexProvider{I}`, `HasSystem`, `ValueProvider{U, P}`, `SeriesProvider{U, P}`; `getu`, `getstate`, `getparam`, `observed`.
- **`TraitArrayTools`:** `Trajectory{U, T}`, `DiffEqArray`, `plotpoints`, RecursiveArrayTools' part.
- **`TraitNonlinearSolve`:** `NonlinearProblemTrait{T, P}` and `NonlinearAlgorithm`, an open set of solvers (`NewtonRaphson`, `TrustRegion`, `NonlinearPolyalg`); `NonlinearProblem`, `SteadyStateProblem`.

Beside them:
- `examples/LorenzModels`, `examples/ChainModels`: user packages with a problem type, callback types and a callable model of their own.
- `baselines/PlainODE`, `baselines/PlainLorenzModels`: the same code without traits, the control.

## How it uses the design

- **The user's function is erased by default.** `ODEProblem(f, u0, tspan, p)` boxes a plain function, `Dynamic{Callable{…}}`, so the solver is compiled once per state and parameter type. `ODEProblem(compiled_for(f), …)` passes the wrapper, and the solver is compiled for `f`. The problem keeps the form given; so do the callbacks' records (`lib/ODECore/src/frontend.jl`, `src/core.jl`).
- **Boxes where code is compiled once:** a callback sees the integrator as a `Dynamic{IntegratorView{U, T, P}}`; the step callbacks see a `Dynamic{StepView}`; the integrator keeps its core as a `Dynamic{CoreIntegrator}`; a problem's names are a `Dynamic{IndexProvider{Symbol}}`.
- `IndexProvider` erases by default (`@default_dynamic`, `lib/TraitIndexing`).
- **Storage keeps the form given:** a solution holds its solver's steps as the solver made them, a plain member in a field typed by `Interpolant`; `Rodas5(linsolve = ls)` keeps its linear solver as given, so the solver is compiled per linear solver, or once for every linear solver with the box `Dynamic{LinearSolver}(ls)`.

## Running

From the workspace root:

```sh
source env.sh
julia-next/julia --project=TraitDifferentialEquations -e 'using Pkg; Pkg.test()'
```

- `test/runtests.jl` includes the files of the other pieces (callbacks, the integrator view, nonlinear problems, symbolic indices, the interpolant).
- The scripted session, the same steps with the trait library and with the plain one, compared by their `RESULT` lines: `scripts/session.jl trait|plain`, with `JULIA_LOAD_PATH="scripts/env:@stdlib"`, from this directory (the script's head gives the commands).
- `scripts/rows.jl`, `scripts/baseline.jl`, `scripts/errors.jl`, `scripts/pirate.jl`, `scripts/runtime.jl`: the measurements of `HISTORY.md`.

## Results

The current numbers, each against plain Julia or Python, are in the design repository's `writeup/design/evidence.md`. The suite's last result is in its `writeup/state.md` §3; on `julia-next` the suite has 35 testsets.

| Date | Build | Result | Log |
|---|---|---|---|
| 2026-11-15 | `unified` 7b374f39ce, TraitWorld 6f86b4e, TDE 813808a (W-151) | 35 testsets pass | `../experiments/implementation/w151/logs/e406-tde-test.log` |
| 2026-11-08 | `unified` 350c5b1c59, TraitWorld d897223 | 32 testsets pass, 3 checks broken (C-54) | `logs/unified/test-9.log` (each file run on its own), `logs/unified/test-final.log` (`Pkg.test`) |
| 2026-11-08 | `unified` ef93ff7844 (W-128's build B2: C-54 for `X{T}`), TraitWorld c4f9f99 | 32 testsets pass, none broken: the function is kept `Dynamic` by default; Rodas5 with a chunk size enters the problem's plain function (the core problem carries it) | `logs/unified/test-b2.log` (`Pkg.test`) |
| 2026-11-10 | `unified` 36b653bb3b, TraitWorld 048c44e; the top module a trait module (E-357) | 34 testsets pass, none broken | `logs/unified/test-front-trait-module.log` |
| 2026-11-18 | `next` 02139b2168, TraitWorld 3e64ede (W-156) | 35 testsets pass; the REPL session 14 checks ok, its results equal the plain session's | `../experiments/implementation/w156/logs/after-tde.log`, `after-tde-session.log` |

## The design

- The design: the design repository's `writeup/reference/` (traits, facts, compiler, lowering).
- The lessons of the ports: `writeup/intro/porting.md`.
