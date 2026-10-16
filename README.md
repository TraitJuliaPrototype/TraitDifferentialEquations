# TraitDifferentialEquations

ODE solvers written in the trait world: the first demonstrator that the trait world works for a real, large library, measured against stock SciML and against the same code written without traits.

Status: not started. The design is in the design repository (`../writeup/notes/`); this library's design is `../writeup/explorations/solvers.md`.

## What it demonstrates

SciML today recompiles the solver for every new callback closure (0.84–0.96 s each, Julia 1.12.2) and for every new loss closure differentiated through it (2.48 s for the second one). Its integrator types run to 3,553 characters (Tsit5) and 24,426 (Rodas5). The trait design predicts:
- **Callbacks and loss closures cost milliseconds.** Callbacks are cut by a vector of dynamic values, and the automatic-differentiation tag is erased at the projection.
- **No load invalidates the solvers' compiled code, in any load order.**
- **Readable types:** under 100 characters, independent of callbacks.
- **Errors at the model's conformance,** not deep in the solver.
- **A trimmed binary** whose reachable code `verify_main` lists.

## Scope

**Pieces** (about 2,500 lines, mostly numerics reused from OrdinaryDiffEq):
- **`ODECore`,** the trait module:
  - the traits `StateVector{T}`, `ODEModel`, `ODEModelJac`, `StepState`, `IntegratorView`, `Integrator`, `Callback` (with `DiscreteCallback`), `FunctionModel`, `FunctionCallback`;
  - the driver with tstops, `saveat` and dense saving; the PI controller; `ODESolution` with dense output;
  - a compatibility layer: `ODEProblem`, `solve`, `DiscreteCallback`.
- **`ODETsit5`:** the tableau, `perform_step!` and the interpolant, from OrdinaryDiffEqTsit5 2.0.0.
- **`ODERodas`:** Rodas5's tableau and step from OrdinaryDiffEqRosenbrock 2.2.0, `ForwardAD{N}` over ForwardDiff with its own tag, and `DenseLU`.
- **`LorenzModels`,** a user package:
  - `Lorenz{T}` conforming as trait-world code;
  - a `Bounce` callback type;
  - a `RaiseSigma` callback that writes the parameter.
- **Scripts:** a plain `lorenz!` function and closure callbacks at the prompt (the library-generated `FunctionModel` route).

**Baselines:**
- **(a) stock SciML:** OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock and ForwardDiff;
- **(b) the same code without traits:** FunctionWrappers for the model, `Vector{Any}` plus assertions for callbacks. This isolates what the traits add over discipline.

## Measurements

Each in fresh sessions; the stock numbers are from `../experiments/solvers/`.

| # | What | Stock SciML | Prediction |
|---|---|---|---|
| 1 | first Tsit5 solve, precompiled user package | 0.16 s | < 20 ms |
| 2 | first Tsit5 solve, a new model type at the prompt | 0.16 s | 10–50 ms |
| 3 | first Rodas5 solve, new model | 1.43 s | 20–80 ms |
| 4 | a new discrete callback type or closure | 0.84–0.96 s | < 20 ms |
| 5 | gradient through Tsit5: first loss, second loss closure | 1.97–2.73 s, then 2.48 s | < 100 ms, then < 10 ms |
| 6 | `Vector{Float32}` state, not precompiled | 2.06 s | once, then cached |
| 7 | rows 1–5 after CairoMakie, DataFrames, Plots, in all six load orders | to measure | identical; 0 trait-world invalidations |
| 8 | a package defining the `nextfloat(::Dual)` pirate | silent | reported (sealing comes later) |
| 9 | the three error cases of findings §17 | 22–44 frames, longest line up to 33,673 | caught at the conformance; otherwise ≤ 12 frames |
| 10 | `typeof(integrator)`: Tsit5, Tsit5 + callback, Rodas5 | 3,553 / 3,683 / 24,426 characters | < 100 each |
| 11 | run time per step, n = 3 and n = 1,000, by model mode | stock | within 1.3× of compiled at n = 3, 2% at n = 1,000 |
| 12 | trimmed binary solving Lorenz with Tsit5, Rodas5 and `Bounce` | to measure | runs; a few MB |
| 13 | precompile time and image sizes | stock images | to measure: the price of placement |

**Success:**
- rows 4 and 5 improve by an order of magnitude;
- row 7 stays at zero;
- row 9 is caught early for models written as trait-world code;
- row 11 stays within its bounds;
- row 12 runs.

## Order of work

1. Tsit5 with a trait-world model, the driver, and rows 1, 2, 4, 7, 10.
2. The function route and the compatibility layer.
3. Rodas5 with automatic differentiation (rows 3, 9).
4. Tag erasure (row 5).
5. The trimmed binary (row 12).

**Prerequisites in TraitWorld:**
- `Dynamic{X}`;
- lenses and trait fields (properties);
- associated types (`StateVector{T}` families);
- constructor operations and the conformance's inverse;
- definition checking with parameter archetypes;
- the dynamic model mode, pinned per integrator.

**Running on the shared machine:** one Julia job at a time, niced, with full image builds under a memory guard.
