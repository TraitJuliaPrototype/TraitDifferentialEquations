# TraitDifferentialEquations against PlainODE (baselines/PlainODE: the same Tsit5 without traits,
# with FunctionWrappers and untyped callbacks), one fresh process per library and load order.
#
#     JULIA_LOAD_PATH="scripts/env:../experiments/interfaces/env:@stdlib" julia --startup-file=no scripts/baseline.jl TraitDifferentialEquations|PlainODE none|before|after
const LIB, LOADS = Symbol(ARGS[1]), ARGS[2]
load_big() = @eval Main using CSV, DataFrames, CairoMakie
LOADS == "before" && load_big()
t0 = time()
@eval using $LIB
const P = getfield(Main, LIB)
row(what, t) = println(rpad(string(LIB), 28), rpad(what, 40), round(1000t; digits = 1), " ms")
row("using", time() - t0)
lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
const U0, P0 = [1.0, 0.0, 0.0], [10.0, 28.0, 8 / 3]
row("1 first Tsit5 solve", @elapsed P.solve(P.ODEProblem(lorenz!, U0, (0.0, 1.0), P0), P.Tsit5()))
LOADS == "after" && load_big()
rossler!(du, u, p, t) = (du[1] = -u[2] - u[3]; du[2] = u[1] + p[1] * u[2]; du[3] = p[2] + u[3] * (u[1] - p[3]); nothing)
row("2 a second right-hand side function", @elapsed P.solve(P.ODEProblem(rossler!, U0, (0.0, 1.0), [0.2, 0.2, 5.7]), P.Tsit5()))
kick = P.DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.6, i -> (i.u[1] += 1.0; nothing))
row("4 a new callback closure", @elapsed P.solve(P.ODEProblem(lorenz!, U0, (0.0, 1.0), P0), P.Tsit5(); callback = kick))
stop = P.DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.terminated = true; nothing))
row("4 a second callback closure", @elapsed P.solve(P.ODEProblem(lorenz!, U0, (0.0, 1.0), P0), P.Tsit5(); callback = stop))
loss1(p) = sum(P.solve(P.ODEProblem(lorenz!, U0, (0.0, 1.0), p), P.Tsit5()).u[end])
row("5 first gradient", @elapsed P.gradient(loss1, copy(P0)))
w = 2.0
loss2(p) = w * P.solve(P.ODEProblem(lorenz!, U0, (0.0, 1.0), p), P.Tsit5()).u[end][1]
row("5 a second loss closure", @elapsed P.gradient(loss2, copy(P0)))
