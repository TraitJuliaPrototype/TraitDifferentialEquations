# Row 11: run time per step, n = 3 (Lorenz) and n = 1,000 (a diffusion chain), by problem form,
# against stock OrdinaryDiffEq with the same method, tolerances and saving (every step, no dense output).
#
#     JULIA_LOAD_PATH="scripts/env:../experiments/solvers/env:@stdlib" julia --startup-file=no scripts/runtime.jl
using TraitDifferentialEquations, TraitWorld
import OrdinaryDiffEq as ODE
const TDE = TraitDifferentialEquations

lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
function chain!(du, u, p, t)
    n = length(u)
    @inbounds for i in 1:n
        l = u[i == 1 ? n : i - 1]; r = u[i == n ? 1 : i + 1]
        du[i] = p[1] * (l - 2u[i] + r) - p[2] * u[i]
    end
    return nothing
end
struct LorenzM; u0::Vector{Float64}; tspan::Tuple{Float64,Float64}; p::Vector{Float64}; end
TDE.rhs!(m::LorenzM, du::Vector{Float64}, u::Vector{Float64}, t::Float64) = lorenz!(du, u, m.p, t)
@implement ODEProblemTrait LorenzM
struct ChainM; u0::Vector{Float64}; tspan::Tuple{Float64,Float64}; p::Vector{Float64}; end
TDE.rhs!(m::ChainM, du::Vector{Float64}, u::Vector{Float64}, t::Float64) = (chain!(du, u, m.p, t); nothing)
@implement ODEProblemTrait ChainM

function best(f; n = 20)
    f()
    return minimum(@elapsed(f()) for _ in 1:n)
end
function row(name, n, tsde, tfm, tmt, steps_t, tode, steps_o)
    ns(t, s) = round(Int, 1e9 * t / s)
    println(rpad(name, 22), "per step: plain function ", ns(tfm, steps_t), " ns, problem type ", ns(tmt, steps_t),
            " ns, OrdinaryDiffEq ", ns(tode, steps_o), " ns; steps ", steps_t, " / ", steps_o,
            "; function/problem type ", round(tfm / tmt; digits = 2), ", problem type/OrdinaryDiffEq ", round((tmt / steps_t) / (tode / steps_o); digits = 2))
end

u3, p3 = [1.0, 0.0, 0.0], [10.0, 28.0, 8 / 3]
s = solve(ODEProblem(lorenz!, u3, (0.0, 10.0), p3), Tsit5())
so = ODE.solve(ODE.ODEProblem(lorenz!, u3, (0.0, 10.0), p3), ODE.Tsit5(); dense = false)
row("Lorenz, n = 3", 3, 0,
    best(() -> solve(ODEProblem(lorenz!, u3, (0.0, 10.0), p3), Tsit5())),
    best(() -> solve(LorenzM(u3, (0.0, 10.0), p3), Tsit5())), s.naccept + s.nreject,
    best(() -> ODE.solve(ODE.ODEProblem(lorenz!, u3, (0.0, 10.0), p3), ODE.Tsit5(); dense = false)), so.stats.naccept + so.stats.nreject)

N = 1000
uN = [exp(-((i - N / 2) / 50)^2) for i in 1:N]
pN = [1.0, 0.01]
s = solve(ODEProblem(chain!, uN, (0.0, 100.0), pN), Tsit5())
so = ODE.solve(ODE.ODEProblem(chain!, uN, (0.0, 100.0), pN), ODE.Tsit5(); dense = false)
row("chain, n = 1,000", N, 0,
    best(() -> solve(ODEProblem(chain!, uN, (0.0, 100.0), pN), Tsit5())),
    best(() -> solve(ChainM(uN, (0.0, 100.0), pN), Tsit5())), s.naccept + s.nreject,
    best(() -> ODE.solve(ODE.ODEProblem(chain!, uN, (0.0, 100.0), pN), ODE.Tsit5(); dense = false)), so.stats.naccept + so.stats.nreject)
