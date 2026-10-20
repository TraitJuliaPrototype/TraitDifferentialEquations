# TraitDifferentialEquations

ODE solvers written in the trait world: the first demonstrator that the trait world works for a real, large library, measured against stock SciML and against the same code written without traits.

Status (session 29, 2026-10-19): **built on the facts+traits compiler and measured.**
- Runs on julia-1.13 61852d8fb1 with TraitWorld 375a930: Julia's compiler compiles the trait modules by facts, and `Dynamic` operations are call-site splits (part D).
- The test suite passes there. The scripted session (`scripts/session.jl trait|plain`) gives the same `RESULT` lines for the trait and the plain library.
- Measured in session 29 (the build of 2026-10-19): "Measured in session 29" below, this file's dated results. **The current numbers** are in the design repository's [`../writeup/design/evidence.md`](../writeup/design/evidence.md) §2–3: the first-call rows from the measurement round of 2026-10-20 (julia-1.13 3b8085c222, TraitWorld bed3782, 2 runs per load order); the run time per step still from the build of 2026-10-19.
- Not measured yet: PlainODE as the control, route A, compiled mode, and the rest listed below.
- Built in session 29, correctness first: route A as a user package (`examples/LorenzModels`), a right-hand side as `Callable{…, CompiledCall}`, the user's mistakes at the implementation (example 5 of `../writeup/reviews/2026-10-18-examples-and-demonstrators.md`) and at a wrong signature (example 10), a scripted interactive session (`scripts/session.jl`), and PlainODE equivalents of each (Rodas5, model types, compiled mode, callback types, `baselines/PlainLorenzModels`). Results in "Correctness (session 29)" below.
- The numbers of sessions 23–27 are under "History (before the facts compiler)". They predate parts (B), (C) and (D) of the compiler.

Earlier (session 27): steps 1, 3 and 4 done. Step 4 adds gradients through Tsit5: the solver is generic in the element type (the step-size control reads primal values, so a gradient follows the primal solve's steps), `u0` is promoted to the parameters' duals as SciML does, and `gradient(loss, p)` tags the duals by the loss's primal signature (`Tag{Sig{Tuple{Vector{Float64}}, Float64}, Float64}`), with the loss as a dynamic call. Rows 1–5 measured then on the patched build (see the table under "History (before the facts compiler)"). Steps 1 and 3 (sessions 23–24): Tsit5 with plain functions and model types, discrete callbacks, the SciML-like interface, and Rodas5 with ForwardDiff's own `Dual` under the library's tag.

**How it is built (keys):**
- `ODEModel{T}`: `rhs!(m, du, u, t)`, `jacobian!(m, J, u, t)` (default: finite differences).
- **The function route (B).** A plain function enters `ODECore.solve_rhs` at its declared signature, `f::Callable{RHSSig{P, Float64}, Nothing}`, and is held in the library's `FunctionModel{P, T, F}` as the bound gave it:
  - `DynamicCall`, the default: `F` is a `DynamicFn`, so the solver compiles once per parameter type `P`, never per function;
  - `CompiledCall`, `ODEProblem(compiled_call(f), …)`: `F` names the function, so the solver compiles for `f`, with `f` inferred into it, and the signature still checked at every call.
  - The compiled mode is served for `Float64` states with `Vector{Float64}` or no parameters (a `Callable` bound's signature is a constant, so one entry per served `P`). Elsewhere (`Float32`, gradients, Rodas5) the function runs as a dynamic call, with the same values.
- **Route A.** A model type that implements `rhs!` is the receiver type, and the solver compiles for it, in the image of the package that declares the implementation (`examples/LorenzModels`: `Lorenz{T}` with its own buffer and Jacobian, workload in the package). Its state keeps `u0`'s element type, which must be the model's `T`. Callbacks see the model as `integ.p`.
- `IntegratorView{T, P}`: fields only (`u`, `t`, `p`, `u_modified`, `terminated`), so there is no receiver type; callbacks see the integrator as a `Dynamic{IntegratorView}`.
- `Callback`: kept as `Dynamic{Callback}`; `DiscreteCallback(condition, affect!)` from plain functions is the library's `FunctionCallback` of two `DynamicFn`s; a callback type conforms (`Bounce`, `RaiseSigma` in `LorenzModels`; a new one at the prompt in `scripts/session.jl`).
- Automatic differentiation (steps 3–4): Rodas5's Jacobian with ForwardDiff's `Dual` and a library-owned tag; gradients through `solve` with tags labelled by the loss's primal signature (`Tag{Sig{A, S}, V}`, tested in session 23), with a guard for closures that capture duals.
- ForwardDiff, like Base, is called directly. Its code is compiled under the facts policy (case C of `../writeup/notes/3-wrapping.md`).

## What it demonstrates

SciML today recompiles the solver for every new callback closure (0.84–0.96 s each, Julia 1.12.2) and for every new loss closure differentiated through it (2.48 s for the second one). Its integrator types run to 3,553 characters (Tsit5) and 24,426 (Rodas5). The trait design predicts:
- **Callbacks and loss closures cost milliseconds.** Callbacks are cut by a vector of dynamic values, and the automatic-differentiation tag is labelled by the loss's primal signature, not its type.
- **Loads cannot slow the solvers down, in any load order:** only a stakeholder of one of their dispatch facts, or a pirate that wins at one, can discard their compiled code.
- **Readable types:** under 100 characters, independent of callbacks.
- **Errors at the model's implementation,** not deep in the solver.
- **A trimmed binary** whose reachable code `verify_main` lists.

## Scope

**Pieces** (as built; the plan's split into `ODETsit5`, `ODERodas` was not needed):
- **`src/ODECore.jl`,** the trait module:
  - the traits `ODEModel{T}`, `IntegratorView{T, P}`, `Callback`;
  - the library's conforming types `FunctionModel{P, T, F}`, `FunctionModelAD{P, N}`, `FunctionCallback`;
  - Tsit5 (tableau and step from OrdinaryDiffEqTsit5 2.0.0), Rodas5 (from OrdinaryDiffEqRosenbrock 2.2.0) with ForwardDiff's `Dual` under the library's tag, dense LU, the PI controller, `saveat` by interpolation;
  - `solve_rhs`, the entry with the `Callable` bound.
- **`src/TraitDifferentialEquations.jl`,** the ordinary layer: `ODEProblem`, `solve`, `DiscreteCallback`, `gradient` (tags by the loss's primal signature).
- **`examples/LorenzModels`,** a user package: an ordinary module around its trait module `Models`, with `Lorenz{T}` (mutable, its own buffer, a hand-written Jacobian), `Bounce` (writes the state), `RaiseSigma` (writes the model's σ), and a workload.
- **`scripts/session.jl`:** the interactive session (below). `scripts/rows.jl`, `baseline.jl`, `errors.jl`, `pirate.jl`, `runtime.jl`: the measurements of session 27.

**Baselines:**
- **(a) stock SciML:** OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock and ForwardDiff;
- **(b) the same code without traits** (`baselines/PlainODE`, `baselines/PlainLorenzModels`): FunctionWrappers for a plain function (at the plain and the dual signature for Rodas5), the function itself under `compiled_call`, model types and callback types as methods of PlainODE's generic functions, `Vector{Any}` plus assertions for callbacks, the same gradient tag. Since session 29 it has every feature of the trait form, so each later measurement has its control.

## Measurements

The rows (1–13) are defined in the table under "History (before the facts compiler)", with the numbers of sessions 23–27. The session-29 numbers are in "Measured in session 29".

**Success:**
- rows 4 and 5 improve by an order of magnitude;
- row 7 stays at zero policy CodeInstances lost;
- row 9 is caught early for models written as trait-world code;
- row 11 stays within its bounds;
- row 12 runs.

## Correctness (session 29)

No timings: these rows say what happens, not how fast. Tests: `test/routea.jl`, `test/compiledcall.jl`; frames and lines from a message printed with its backtrace, harness frames included.

**Route A: the user's mistakes at the implementation** (review example 5):

| Mistake | Where caught | What the user sees |
|---|---|---|
| (b) no `rhs!` at all | at `@implement` | `ImplementError: no field of NoRhs conforms to ODEModel and NoRhs has no methods for its operations (fields: k::Float64); name a field `data`, write `project(t) = …`, or implement the operations`. Names neither `rhs!` nor its signature, and suggests projections, which are gone (TraitWorld issue) |
| (b′) `jacobian!` but no `rhs!` | at `@implement` | `OnlyJac has methods for some operations of ODEModel but not for rhs!: define them, …`. Names the operation, not its signature |
| (c) `rhs!` returns `du` (`du .= …` last) | at `@implement`, also in the block form | `` `rhs!` at Tuple{BadRet, Vector{Float64}, Vector{Float64}, Float64} returns `Vector{Float64}`, declared `Nothing` ``: the cause and the fix, 3 lines before the frames |
| (c′) `u::Vector{Float32}` in `rhs!` | at `@implement` | the message of (b): does not name the existing method or the declared signature (TraitWorld issue) |
| (a) a `Float64` buffer in `Lorenz{T}`, under `gradient` | **at run time**: the family member's check at first use passes | `MethodError: no method matching Float64(::Dual{…})`, 69 lines, 28 frames, longest line 685, 12 of the frames Base's broadcasting. Inference does not prove the conversion fails, so no check at `@implement` sees it. The outcome against that the review named |
| a dual model with a `Float64` state (the loss forgot `eltype(p).(u0)`) | at the first operation call | `TraitArgumentError: operation rhs! of trait ODEModel at representation Lorenz{Dual…} is declared for arguments …, got …`, 34 lines, 16 frames |

Since TraitWorld 3ba6c05, the messages of (b) and (c′) name `rhs!` and the declared argument types; the two tests that were broken pass (TraitDifferentialEquations 62e9fa4).

**A right-hand side with a declared signature, in both modes** (review example 10):

| Case | `DynamicCall` (default) | `CompiledCall` (`compiled_call(f)`) |
|---|---|---|
| Lorenz, saved states | identical (`==`) | identical |
| out of place, `f(u, p, t)` | `MethodError` for the declared arguments at the first call, 15 frames | first a warning at the model's first use (`rhs!` … always throws (inferred `Union{}`): a fallback, not an implementation), then the same `MethodError` |
| returns `du` | `TypeError: expected Nothing` at the first call | warning, then the same `TypeError` |
| an explicit wrapper for another signature, `Callable{Tuple{Vector{Float64}}, Nothing, CompiledCall}(f)` | `TraitBoundError` at the entry, before the solve, naming both signatures | the same |
| `Float32` state, Rodas5, gradient | — | no compiled entry: runs dynamic, identical values |
| a function redefined at the prompt | the next solve uses the new definition | the same (the compiled solver is recompiled: the function's owner changed a fact) |

Wanted from TraitWorld: refuse a function without a method at `A`, or with a result disjoint from `S`, at the entry in both modes (a `@test_broken` in `test/compiledcall.jl`).

**The interactive session** (`scripts/session.jl trait|plain`): **passes in both libraries, and the 10 `RESULT` lines of the trait session and the plain one are identical** (16 checks: a precompiled model and its callbacks, a new function, the same in compiled mode (equal values), a new model type (equal values), a new callback closure and a new callback type (equal values), two losses against central differences, Rodas5 by duals against by finite differences, `Float32`, and an edit of a function already used, in both modes). Each entry is evaluated as the REPL evaluates it (soft scope, `Core.eval` in `Main`, in order), not at a real REPL. Found on the way: `gradient` raised a `TypeError` for a loss that captured a `Vector` (`_holds_duals` looked at `1` in `Array{Float64, 1}` as a type); fixed, with a test.

**PlainODE equivalents:** route A model types (`PlainODE.rhs!`, `jacobian!`), `compiled_call` (the function unwrapped, the solver specialized on it), callback types (`PlainODE.condition`, `affect!`), Rodas5 (FunctionWrappers at the plain and the dual signature), `baselines/PlainLorenzModels`. Found on the way: PlainODE's typed default `jacobian!(m, J::Matrix{Float64}, …)` was ambiguous with `jacobian!(m::Lorenz{T}, J::Matrix{T}, …)`; the plain default is now untyped. The trait form has no such ambiguity: a default belongs to the implementation.

## Measured in session 29

A dated table: julia-1.13 61852d8fb1, TraitWorld 375a930 (the build of 2026-10-19), the patched build. `scripts/rows.jl`, fresh processes, *n = 1* per load order; big loads = CSV, DataFrames, CairoMakie. One machine, shared with other jobs. The current numbers, from the round of 2026-10-20: [`../writeup/design/evidence.md`](../writeup/design/evidence.md) §3.

| Row | No loads | Loads before | Loads after |
|---|---|---|---|
| `using TraitDifferentialEquations` | 0.47 s | 7.28 s (one run; 1.29–1.39 s in other runs: a cache rebuilt once, stacked environments) | 0.47 s |
| first Tsit5 solve, a plain function | 10.0 ms | 27.8 ms | 9.9 ms |
| a second model function | 9.8 ms | 11.2 ms | 27.6 ms |
| a new model type: `@implement` / first solve | 142 / 82 ms | 164 / 133 ms | 149 / 89 ms |
| first Rodas5 solve | 16.6 ms | 17.0 ms | 16.8 ms |
| a new callback closure / a second | 12.4 / 5.2 ms | 24.0 / 5.4 ms | **170.7** / 5.2 ms |
| first gradient / second loss / third loss | 29 / 17 / 62 ms | 34 / 18 / 63 ms | 34 / 17 / 62 ms |
| ForwardDiff's own tag, for comparison | 0.70 s | 0.73 s | 0.72 s |
| a `Float32` state, first solve | 0.46 s | 0.45 s | 0.45 s |

- At the start of the phase: a new model type after the loads took 2.0 s and 2.1 s.
- Session 27 (the prototype policy): 0.16–0.23 s and 33–44 ms. The first solve is not back at 33–44 ms yet (the round of 2026-10-20: 60–62 / 44–46 ms without loads; `evidence.md` §3).
- Row 4 (a new callback closure) after the loads: 171 ms against 12 ms without (24 ms in an earlier run): not diagnosed. Not reproduced in the round of 2026-10-20 (julia-1.13 3b8085c222): 24.6–24.7 ms after the loads, against 12.6–12.8 ms without (`evidence.md` §3). The remaining difference is open (the design repository's `writeup/open-issues.md` §4.15).
- The first Tsit5 solve here is a plain function's, not route A's precompiled package (row 1 of the plan).

**Run time per solver step** (row 11; `scripts/runtime.jl`):

| What | Trait form | Baseline | Notes |
|---|---|---|---|
| Lorenz (n = 3) | model type 158–160 ns; plain function 190–199 | OrdinaryDiffEq 338–355 ns | |
| a diffusion chain (n = 1,000) | 7.5–7.9 µs | OrdinaryDiffEq 6.1 µs | OrdinaryDiffEq's stage loops are fused with `@muladd` |

**Not measured yet** (each with PlainODE as the control; in fresh sessions and at a REPL; with the plain baseline under `all` (`JULIA_COMPILE_BY_FACTS=1`, or the facts system image)):
- PlainODE on every row: no PlainODE numbers were taken in session 29, so the session-29 table has no plain control.
- The control under `all` (row 7): compiled code lost per load. Session 29 timed the rows in three load orders but did not count lost CodeInstances.
- Any row at a real REPL: the rows above ran in fresh processes.
- **Row 1 for route A:** the first solve of `LorenzModels.Lorenz` from its image (the plan's "precompiled user package"), against `PlainLorenzModels`.
- **Row 4 for callback types:** a new callback type at the prompt, against a new closure.
- **Example 10, `CompiledCall`:** run time per step at n = 3 and n = 1,000 against `DynamicCall` and the model type (row 11's columns); compile time for a second function in compiled mode; what the image holds.
- **Row 5 for route A:** first gradient and a second loss through `Lorenz{Dual}`.
- **Row 6** from an image that holds the key (the per-session cost, 0.45–0.46 s, is above).
- **Row 9 for route A:** the correctness table below in lines, frames and longest line, beside SciML's.
- **Row 11 with callback types:** the cost of `Dynamic{Callback}` per step for `Bounce`, against a closure and against PlainODE's `Vector{Any}`.
- **The session:** the time per entry of `scripts/session.jl`, both libraries, at a real REPL.
- **Rodas5 in PlainODE:** row 3's control.
- Row 8's strict half and row 12 (the trimmed binary) are not built: strict mode and builds were outside this task.
- Row 13 (precompile time, image sizes, image load time) on the current build.

## Order of work

1. Tsit5 with a trait-world model, the driver, and rows 1, 2, 4, 7, 10.
2. The function route and the compatibility layer.
3. Rodas5 with automatic differentiation (rows 3, 9).
4. *(Done, session 27.)* Tag erasure (row 5).
5. *(Session 29.)* Route A as a user package, `CompiledCall`, the implementation errors, the session, PlainODE complete.
6. The trimmed binary (row 12).

**Prerequisites in TraitWorld:**
- `Dynamic{X}`;
- lenses and trait fields (properties);
- associated types (`StateVector{T}` families);
- constructor operations and the implementation's inverse;
- definition checking with parameter archetypes;
- the dynamic model mode, pinned per integrator: superseded by keys (the function route is the dynamic mode; a model type is compiled for its key);
- bounds that name an associated type (`m::ODEModel{T} where T`) and `Callable` signatures over a method's static parameters: not supported yet (see Correctness).

**Running on the shared machine:** one Julia job at a time, niced, with full image builds under a memory guard.


## History (before the facts compiler)

The measurements of sessions 23–27, under TraitWorld's prototype policy, unchanged.

### Measurements (sessions 23–27)

Each in fresh sessions; the stock numbers are from `../experiments/solvers/`. The millisecond predictions are for the patched build: on stock Julia, package images keep no native code for the facts policy's code, so each session compiles it again (piece 5 of `../writeup/explorations/julia-support.md`). Session 26 numbers come from `scripts/rows.jl`, one fresh process per cell, with CSV, DataFrames and CairoMakie loaded before, after or not at all. They are single runs. Rows still marked *remeasure* predate wrapping by facts.

| # | What | Stock SciML | Prediction | Measured (stock SciML on 1.13.2 in brackets) |
|---|---|---|---|---|
| 1 | first Tsit5 solve, precompiled user package | 0.16 s | < 20 ms (patched) | **patched (session 27): 6.4–8.0 ms in every load order, the same on the system image**, from 47 ms before the fixes of session 27 (the family implementation found its builder in each session, 34 ms). Session 26: patched (session 26): 32–37 ms in every load order. Stock: 70–75 ms, 505 ms with the loads before. Session 24: 55–60 ms patched. Before facts (session 23): 73–76 ms (0.14–0.17 s) |
| 2 | first Tsit5 solve, a new model type at the prompt | 0.16 s | 10–50 ms, also after big loads | **patched (session 27), in every order: a second model function 6–8 ms; a new model type `@implement` 0.16–0.23 s, first solve 33–44 ms (113 ms once with the loads after); the same on the system image (`@implement` 0.24–0.26 s).** Session 26: a second model function 7–12 ms in every order (0.10–0.11 s). A new model type: `@implement` 0.73 s, first solve 72 ms; after big loads, before or after, `@implement` 1.45–1.50 s and first solve 480–490 ms. Stock: a second function 7 ms, but 1.39 s after loads; a new type's first solve 4.1–5.1 s. New trait code after big loads is the open problem (open-issues §4.5; Julia piece 12) |
| 3 | first Rodas5 solve, new model | 1.43 s | 20–80 ms | patched (session 27), after a Tsit5 solve: 22–26 ms in every order, also on the system image. Session 26, after a Tsit5 solve in the same session: 23 ms, 27 ms with the loads before, 181 ms with the loads after (to explain). Session 24, Rodas5 first in the session: 72–77 ms in every load order; a second model function 65–92 ms (0.069–0.075 s) |
| 4 | a new discrete callback type or closure | 0.84–0.96 s | < 20 ms | patched (session 27): a new callback closure 13–15 ms without loads, 25–31 ms with them; a second one 25–32 ms; on the system image 13–29 ms and 16 ms. Session 26: a new callback closure 13–14 ms, a second one 25–32 ms, in every order; stock the same (13–16, 31–33 ms). Session 23: 12–14 ms (0.67–0.83 s) |
| 5 | gradient through Tsit5: first loss, second loss closure | 1.97–2.73 s, then 2.48 s | < 100 ms, then < 10 ms (tags labelled by the loss's primal signature) | patched (session 27), in every load order: first gradient 34–55 ms, a second loss closure 17–21 ms, a third loss with another model function 64–72 ms. ForwardDiff's own tag (the loss's type) for comparison: 0.67–0.87 s. The same on the system image. The second loss is above the prediction: the loss closure and `ODEProblem` at the dual types compile per closure |
| 6 | `Vector{Float32}` state, not precompiled | 2.06 s | once, then cached (a view cache; on patched, with its native code) | patched (session 27): first session 0.59 s; later sessions 0.105 s, with the view cache holding the worker (the ordinary layer's methods for `Float32` and the family's builder are found again in each session). Tsit5 is generic in the element type since step 4 |
| 7 | rows 1–5 after CairoMakie, DataFrames, Plots, in all six load orders | to measure | identical; 0 policy CodeInstances lost | **session 27:** rows 1–5 identical with CSV, DataFrames, CairoMakie before, after or not at all (row 1 6.4–8.0 ms, row 5 30–55 then 17–21 ms); `g2all` 0 policy CodeInstances lost over twelve loads (CSV, DataFrames, CairoMakie, ChurnStorm, BasePirate and seven more), work 9–13 ms. Plots not tested. Earlier: patched (session 24, g2all): 0 lost over twelve loads in any order. Session 26: rows 1 and 4 identical in every order; row 2 identical for a new function, not for a new type (new trait code after loads); row 3's 181 ms after the loads to explain. Row 5 not yet |
| 8 | a package defining the `nextfloat(::Dual)` pirate | silent | followed: the solver recompiles exactly where the pirate wins at one of its facts; refused under strict mode (deferred) | first half, patched (session 27, `scripts/pirate.jl`; the pirate is `+` on the gradient's duals, which the solver uses, since it never calls `nextfloat`): defined after a first gradient, it is followed: the next gradient compiles the solver's dual code again (373 ms) and calls the pirate 2,192 times; plain code calls it too; the gradient is unchanged (the pirate returns the same value). Strict mode's refusal: deferred |
| 9 | the three error cases of findings §17 | 22–44 frames, longest line up to 33,673 | caught at the implementation; otherwise ≤ 12 frames | patched (session 27, `scripts/errors.jl`; full message with backtrace, harness frames included as for SciML): (1) parameters used but never passed: 27 lines, **12 frames**, longest line 289 (SciML 71 lines, 29 frames, 3,789); (2) a `Float64` cache under `gradient`: 73 lines, 30 frames, longest line 865 (SciML 101, 44, 5,574); (3) the same under Rodas5's Jacobian: 63 lines, 25 frames, longest line 302 (SciML 81, 22, 33,673). TraitWorld hides its own frames (`JULIA_TW_ALL_FRAMES=1` shows them; 21/42/34 frames before). Cases 2 and 3 keep 12 frames of Base's broadcasting inside the user's function, so they miss the frame bound; none is caught at an implementation, since the model is a plain function |
| 10 | `typeof(integrator)`: Tsit5, Tsit5 + callback, Rodas5 | 3,553 / 3,683 / 24,426 characters | < 100 each | `Tsit5Integrator{Float64, Vector{Float64}}`, with or without callbacks (3,553 / 3,683) |
| 11 | run time per step, n = 3 and n = 1,000, by model mode | stock | within 1.3× of compiled at n = 3, 2% at n = 1,000, including the policy's run-time sites | patched (session 27, `scripts/runtime.jl`, minimum of 20 solves, every step saved, OrdinaryDiffEq without dense output): Lorenz, n = 3: plain function 206 ns per step, model type 173 ns (**1.19×**), OrdinaryDiffEq 355 ns. A diffusion chain, n = 1,000: plain function 7.9 µs, model type 8.0 µs (**within 1%**), OrdinaryDiffEq 6.9 µs: the stage loops here are plain loops, OrdinaryDiffEq's are fused with `@muladd`, 15% faster |
| 12 | trimmed binary solving Lorenz with Tsit5, Rodas5 and `Bounce` | to measure | runs; a few MB | not yet |
| 13 | precompile time, image sizes, image load time (stock and patched) | stock images | to measure: the price of placement and of piece 5 | session 27 (`../experiments/demos/images.jl`, each package compiled again with its dependencies compiled): TraitDifferentialEquations 8.2 s patched, 8.9 s stock; image 9.9 MB native code patched, 7.1 MB stock (piece 5 keeps the policy's code); `using` 0.29 s patched, 0.22 s stock. TraitWorld itself: 30 s and 38 MB patched, 23 s and 28 MB stock; `using` 0.13–0.14 s |

**Against baseline (b), PlainODE** (session 27; `baselines/PlainODE`, `scripts/baseline.jl`; patched, view caches off, ms; loads = CSV, DataFrames, CairoMakie):

| | TraitDifferentialEquations: no loads / loads before / loads after | PlainODE: no loads / before / after |
|---|---|---|
| `using` | 278–284 / 338 / 278 | 99–112 / 158 / 99 |
| row 1, first Tsit5 solve | 6.5 / 6.5 / 6.8 | 9.9 / **54** / 10.1 |
| row 2, a second model function | 6.0 / 6.2 / 6.3 | 5.0 / 4.8 / 5.1 |
| row 4, a new callback closure | 17 / 29 / 29 | 5.1 / 15 / **144** |
| row 4, a second one | 5.3–5.9 | 4.5–4.8 |
| row 5, first gradient | 30 / 36 / 36 | 37 / 39 / 45 |
| row 5, a second loss closure | 17–20 | 20–22 |

- **What discipline alone gives:** PlainODE (FunctionWrappers, untyped callbacks, the same signature-labelled tag) matches the trait form on rows 2 and 5 in a fresh session, and is faster to load and for a first callback.
- **What the trait form adds:** no load order changes its numbers (row 1 after the loads: 6.5 against 54 ms; a first callback after them: 29 against 144 ms), and its code is checked where it is defined.
- **What it costs:** loading TraitWorld (0.17 s more) and about 12 ms more for a session's first callback: the user's closures see the integrator as a `Dynamic{IntegratorView}`, so their field accesses dispatch, and each closure's pointer wrapper is compiled (about 5 ms per closure, as a FunctionWrapper's would be).
