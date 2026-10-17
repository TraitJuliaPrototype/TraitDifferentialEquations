# Rows 1, 2, 3, 4 of the README in one fresh process, with big loads before, after or not at all.
#
#     JULIA_LOAD_PATH="<env with TraitDifferentialEquations>:<env with CSV, DataFrames, CairoMakie>:@stdlib" \
#         julia --startup-file=no scripts/rows.jl none|before|after
#
# "after": CSV, DataFrames and CairoMakie load after the first solve, before rows 2 to 4.
# Each line is one first call: "row <n> <what> <ms>".
const LOADS = ARGS[1]
load_big() = (t = @elapsed(@eval Main using CSV, DataFrames, CairoMakie); println("loads ", round(t; digits = 2), " s"))
LOADS == "before" && load_big()
t = @elapsed @eval using TraitDifferentialEquations, TraitWorld
println("using TraitDifferentialEquations ", round(t; digits = 2), " s")
row(n, what, t) = println("row ", n, " ", what, " ", round(1000t; digits = 1), " ms")

lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
const U0, P0 = [1.0, 0.0, 0.0], [10.0, 28.0, 8 / 3]
row(1, "first Tsit5 solve, a plain function", @elapsed solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5()))
LOADS == "after" && load_big()

rossler!(du, u, p, t) = (du[1] = -u[2] - u[3]; du[2] = u[1] + p[1] * u[2]; du[3] = p[2] + u[3] * (u[1] - p[3]); nothing)
row(2, "a second model function", @elapsed solve(ODEProblem(rossler!, U0, (0.0, 1.0), [0.2, 0.2, 5.7]), Tsit5()))

@eval struct Lorenz; s::Float64; r::Float64; b::Float64; end
@eval TraitDifferentialEquations.rhs!(m::Lorenz, du::Vector{Float64}, u::Vector{Float64}, t::Float64) =
    (du[1] = m.s * (u[2] - u[1]); du[2] = u[1] * (m.r - u[3]) - u[2]; du[3] = u[1] * u[2] - m.b * u[3]; nothing)
t = @elapsed @eval @implement ODEModel{Float64} Lorenz
row(2, "a new model type: @implement", t)
row(2, "a new model type: first solve", @elapsed solve(Lorenz(10.0, 28.0, 8 / 3), U0, (0.0, 1.0), Tsit5()))

row(3, "first Rodas5 solve", @elapsed solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Rodas5()))

kick = DiscreteCallback((u, t, i) -> t > 0.5 && t < 0.6, i -> (i.u[1] += 1.0; nothing))
row(4, "a new callback closure", @elapsed solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5(); callback = kick))
stop = DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.terminated = true; nothing))
row(4, "a second callback closure", @elapsed solve(ODEProblem(lorenz!, U0, (0.0, 1.0), P0), Tsit5(); callback = stop))

# row 5: gradients through Tsit5; tags labelled by the loss's primal signature
import ForwardDiff
loss1(p) = sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5()).u[end])
row(5, "first gradient (Tsit5, 3 parameters)", @elapsed gradient(loss1, copy(P0)))
w = 2.0
loss2(p) = w * solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5()).u[end][1]
row(5, "a second loss closure", @elapsed gradient(loss2, copy(P0)))
loss3(p) = sum(abs2, solve(ODEProblem(rossler!, U0, (0.0, 1.0), p), Tsit5()).u[end])
row(5, "a third loss, another model function", @elapsed gradient(loss3, [0.2, 0.2, 5.7]))
loss4(p) = sum(solve(ODEProblem(lorenz!, U0, (0.0, 1.0), p), Tsit5()).u[end]) / 2
row(5, "ForwardDiff's own tag (the loss's type), for comparison", @elapsed ForwardDiff.gradient(loss4, copy(P0)))
