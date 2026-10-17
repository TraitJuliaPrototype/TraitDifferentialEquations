# TraitDifferentialEquations

ODE solvers written in the trait world: the first demonstrator that the trait world works for a real, large library, measured against stock SciML and against the same code written without traits.

Status (session 27, 2026-10-17): steps 1, 3 and 4 done. Step 4 adds gradients through Tsit5: the solver is generic in the element type (the step-size control reads primal values, so a gradient follows the primal solve's steps), `u0` is promoted to the parameters' duals as SciML does, and `gradient(loss, p)` tags the duals by the loss's primal signature (`Tag{Sig{Tuple{Vector{Float64}}, Float64}, Float64}`), with the loss as a dynamic call. **Row 5: first gradient 34–55 ms, a second loss closure 17–21 ms** (SciML 1.97–2.73 s, then 2.48 s); the same gradient with ForwardDiff's own tag 0.67–0.87 s. **Row 1: 6.4–8.0 ms in every load order** (session 26: 32–37 ms), after two TraitWorld fixes found here (policy code no longer runs native code through bare `MethodInstance` invokes; family conformances keep their builders in the declaring package's image). Rows 1–5 measured on the patched build and on the system image with TraitWorld in it (`../experiments/sysimage`). Earlier: steps 1 and 3 (sessions 23–24): Tsit5 with plain functions and model types, discrete callbacks, the SciML-like interface, and Rodas5 with ForwardDiff's own `Dual` under the library's tag, called directly (wrapping by facts). The rows were reviewed for wrapping by facts in session 26 (`../writeup/reviews/2026-10-17-demonstrators-under-facts.md`).

**How it is built (keys):**
- `ODEModel{T}`: `rhs!(m, du, u, t)`. A plain function is held as a `DynamicFn` in the library's `FunctionModel{P}`, so the solver compiles once per parameter type `P`, never per function. A model type that implements `rhs!` is the implementing type, and the solver compiles for it (the comparison).
- `IntegratorView{T, P}`: fields only (`u`, `t`, `p`, `u_modified`, `terminated`), so there is no implementing type; callbacks see the integrator as a `Dynamic{IntegratorView}`.
- `Callback`: kept as `Dynamic{Callback}`; `DiscreteCallback(condition, affect!)` from plain functions is the library's `FunctionCallback` of two `DynamicFn`s.
- Automatic differentiation (steps 3–4): Rodas5's Jacobian with ForwardDiff's `Dual` and a library-owned tag; gradients through `solve` with tags labelled by the loss's primal signature (`Tag{Sig{A, S}, V}`, tested in session 23), with a guard for closures that capture duals.
- ForwardDiff, like Base, is called directly. Its code is compiled under the facts policy (case C of `../writeup/notes/3-wrapping.md`).

## What it demonstrates

SciML today recompiles the solver for every new callback closure (0.84–0.96 s each, Julia 1.12.2) and for every new loss closure differentiated through it (2.48 s for the second one). Its integrator types run to 3,553 characters (Tsit5) and 24,426 (Rodas5). The trait design predicts:
- **Callbacks and loss closures cost milliseconds.** Callbacks are cut by a vector of dynamic values, and the automatic-differentiation tag is labelled by the loss's primal signature, not its type.
- **Loads cannot slow the solvers down, in any load order:** only a stakeholder of one of their dispatch facts, or a pirate that wins at one, can discard their compiled code.
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
- **`ODERodas`:** Rodas5's tableau and step from OrdinaryDiffEqRosenbrock 2.2.0, ForwardDiff's `Dual` with the library's tag (`FDual{N}`), and `DenseLU`.
- **`LorenzModels`,** a user package:
  - `Lorenz{T}` conforming as trait-world code;
  - a `Bounce` callback type;
  - a `RaiseSigma` callback that writes the parameter.
- **Scripts:** a plain `lorenz!` function and closure callbacks at the prompt (the library-generated `FunctionModel` route).

**Baselines:**
- **(a) stock SciML:** OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock and ForwardDiff;
- **(b) the same code without traits:** FunctionWrappers for the model, `Vector{Any}` plus assertions for callbacks. This isolates what the traits add over discipline.

## Measurements

Each in fresh sessions; the stock numbers are from `../experiments/solvers/`. The millisecond predictions are for the patched build: on stock Julia, package images keep no native code for the facts policy's code, so each session compiles it again (piece 5 of `../writeup/explorations/julia-support.md`). Session 26 numbers come from `scripts/rows.jl`, one fresh process per cell, with CSV, DataFrames and CairoMakie loaded before, after or not at all. They are single runs. Rows still marked *remeasure* predate wrapping by facts.

| # | What | Stock SciML | Prediction | Measured (stock SciML on 1.13.2 in brackets) |
|---|---|---|---|---|
| 1 | first Tsit5 solve, precompiled user package | 0.16 s | < 20 ms (patched) | **patched (session 27): 6.4–8.0 ms in every load order, the same on the system image**, from 47 ms before the fixes of session 27 (the family conformance found its builder in each session, 34 ms). Session 26: patched (session 26): 32–37 ms in every load order. Stock: 70–75 ms, 505 ms with the loads before. Session 24: 55–60 ms patched. Before facts (session 23): 73–76 ms (0.14–0.17 s) |
| 2 | first Tsit5 solve, a new model type at the prompt | 0.16 s | 10–50 ms, also after big loads | **patched (session 27), in every order: a second model function 6–8 ms; a new model type `@implement` 0.16–0.23 s, first solve 33–44 ms (113 ms once with the loads after); the same on the system image (`@implement` 0.24–0.26 s).** Session 26: a second model function 7–12 ms in every order (0.10–0.11 s). A new model type: `@implement` 0.73 s, first solve 72 ms; after big loads, before or after, `@implement` 1.45–1.50 s and first solve 480–490 ms. Stock: a second function 7 ms, but 1.39 s after loads; a new type's first solve 4.1–5.1 s. New trait code after big loads is the open problem (open-issues §4.5; Julia piece 12) |
| 3 | first Rodas5 solve, new model | 1.43 s | 20–80 ms | patched (session 27), after a Tsit5 solve: 22–26 ms in every order, also on the system image. Session 26, after a Tsit5 solve in the same session: 23 ms, 27 ms with the loads before, 181 ms with the loads after (to explain). Session 24, Rodas5 first in the session: 72–77 ms in every load order; a second model function 65–92 ms (0.069–0.075 s) |
| 4 | a new discrete callback type or closure | 0.84–0.96 s | < 20 ms | patched (session 27): a new callback closure 13–15 ms without loads, 25–31 ms with them; a second one 25–32 ms; on the system image 13–29 ms and 16 ms. Session 26: a new callback closure 13–14 ms, a second one 25–32 ms, in every order; stock the same (13–16, 31–33 ms). Session 23: 12–14 ms (0.67–0.83 s) |
| 5 | gradient through Tsit5: first loss, second loss closure | 1.97–2.73 s, then 2.48 s | < 100 ms, then < 10 ms (tags labelled by the loss's primal signature) | patched (session 27), in every load order: first gradient 34–55 ms, a second loss closure 17–21 ms, a third loss with another model function 64–72 ms. ForwardDiff's own tag (the loss's type) for comparison: 0.67–0.87 s. The same on the system image. The second loss is above the prediction: the loss closure and `ODEProblem` at the dual types compile per closure |
| 6 | `Vector{Float32}` state, not precompiled | 2.06 s | once, then cached (a view cache; on patched, with its native code) | patched (session 27): first session 0.59 s; later sessions 0.105 s, with the view cache holding the worker (the ordinary layer's methods for `Float32` and the family's builder are found again in each session). Tsit5 is generic in the element type since step 4 |
| 7 | rows 1–5 after CairoMakie, DataFrames, Plots, in all six load orders | to measure | identical; 0 policy CodeInstances lost | patched (session 24, g2all): 0 lost over twelve loads in any order. Session 26: rows 1 and 4 identical in every order; row 2 identical for a new function, not for a new type (new trait code after loads); row 3's 181 ms after the loads to explain. Row 5 not yet |
| 8 | a package defining the `nextfloat(::Dual)` pirate | silent | followed: the solver recompiles exactly where the pirate wins at one of its facts; refused under strict mode (deferred) | first half, patched (session 27, `scripts/pirate.jl`; the pirate is `+` on the gradient's duals, which the solver uses, since it never calls `nextfloat`): defined after a first gradient, it is followed: the next gradient compiles the solver's dual code again (373 ms) and calls the pirate 2,192 times; plain code calls it too; the gradient is unchanged (the pirate returns the same value). Strict mode's refusal: deferred |
| 9 | the three error cases of findings §17 | 22–44 frames, longest line up to 33,673 | caught at the conformance; otherwise ≤ 12 frames | patched (session 27, `scripts/errors.jl`; full message with backtrace, harness frames included as for SciML): (1) parameters used but never passed: 27 lines, **12 frames**, longest line 289 (SciML 71 lines, 29 frames, 3,789); (2) a `Float64` cache under `gradient`: 73 lines, 30 frames, longest line 865 (SciML 101, 44, 5,574); (3) the same under Rodas5's Jacobian: 63 lines, 25 frames, longest line 302 (SciML 81, 22, 33,673). TraitWorld hides its own frames (`JULIA_TW_ALL_FRAMES=1` shows them; 21/42/34 frames before). Cases 2 and 3 keep 12 frames of Base's broadcasting inside the user's function, so they miss the frame bound; none is caught at a conformance, since the model is a plain function |
| 10 | `typeof(integrator)`: Tsit5, Tsit5 + callback, Rodas5 | 3,553 / 3,683 / 24,426 characters | < 100 each | `Tsit5Integrator{Float64, Vector{Float64}}`, with or without callbacks (3,553 / 3,683) |
| 11 | run time per step, n = 3 and n = 1,000, by model mode | stock | within 1.3× of compiled at n = 3, 2% at n = 1,000, including the policy's run-time sites | patched (session 27, `scripts/runtime.jl`, minimum of 20 solves, every step saved, OrdinaryDiffEq without dense output): Lorenz, n = 3: plain function 206 ns per step, model type 173 ns (**1.19×**), OrdinaryDiffEq 355 ns. A diffusion chain, n = 1,000: plain function 7.9 µs, model type 8.0 µs (**within 1%**), OrdinaryDiffEq 6.9 µs: the stage loops here are plain loops, OrdinaryDiffEq's are fused with `@muladd`, 15% faster |
| 12 | trimmed binary solving Lorenz with Tsit5, Rodas5 and `Bounce` | to measure | runs; a few MB | not yet |
| 13 | precompile time, image sizes, image load time (stock and patched) | stock images | to measure: the price of placement and of piece 5 | session 27 (`../experiments/demos/images.jl`, each package compiled again with its dependencies compiled): TraitDifferentialEquations 8.2 s patched, 8.9 s stock; image 9.9 MB native code patched, 7.1 MB stock (piece 5 keeps the policy's code); `using` 0.29 s patched, 0.22 s stock. TraitWorld itself: 30 s and 38 MB patched, 23 s and 28 MB stock; `using` 0.13–0.14 s |

**Success:**
- rows 4 and 5 improve by an order of magnitude;
- row 7 stays at zero policy CodeInstances lost;
- row 9 is caught early for models written as trait-world code;
- row 11 stays within its bounds;
- row 12 runs.

## Order of work

1. Tsit5 with a trait-world model, the driver, and rows 1, 2, 4, 7, 10.
2. The function route and the compatibility layer.
3. Rodas5 with automatic differentiation (rows 3, 9).
4. *(Done, session 27.)* Tag erasure (row 5).
5. The trimmed binary (row 12).

**Prerequisites in TraitWorld:**
- `Dynamic{X}`;
- lenses and trait fields (properties);
- associated types (`StateVector{T}` families);
- constructor operations and the conformance's inverse;
- definition checking with parameter archetypes;
- the dynamic model mode, pinned per integrator (the only one not done).

**Running on the shared machine:** one Julia job at a time, niced, with full image builds under a memory guard.
