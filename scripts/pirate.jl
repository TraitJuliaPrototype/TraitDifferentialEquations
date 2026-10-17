# Row 8, first half: a pirate that wins at one of the solver's facts is followed, as plain Julia
# follows it (rule P: nothing is skipped). The pirate is `+` on ForwardDiff's duals of the
# gradient's tag, defined after a first gradient by a module that owns neither `+` nor `Dual`. It
# counts its calls and returns the same value; afterwards the solver's dual code calls it (it
# recompiles where the pirate wins), and plain Julia's code does too.
#
#     JULIA_LOAD_PATH="scripts/env:@stdlib" julia --startup-file=no scripts/pirate.jl
using TraitDifferentialEquations
import ForwardDiff
lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
loss(p) = sum(solve(ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0), p), Tsit5()).u[end])
p0 = [10.0, 28.0, 8 / 3]
g1 = gradient(loss, copy(p0))
t1 = @elapsed gradient(loss, copy(p0))
const Tg = ForwardDiff.Tag{TraitDifferentialEquations.Sig{Tuple{Vector{Float64}},Float64},Float64}
const D3 = ForwardDiff.Dual{Tg,Float64,3}
const CALLS = Ref(0)
@eval module Pirate
import ..D3, ..CALLS, ForwardDiff
Base.:+(a::D3, b::D3) = (CALLS[] += 1; D3(ForwardDiff.value(a) + ForwardDiff.value(b), ForwardDiff.partials(a) + ForwardDiff.partials(b)))
end
t2 = @elapsed (g2 = gradient(loss, copy(p0)))
n_solver = CALLS[]
CALLS[] = 0
a = D3(1.0, ForwardDiff.Partials((1.0, 0.0, 0.0)))
plain = a + a
println("before the pirate: gradient ", round(1000t1; digits = 1), " ms (compiled)")
println("after the pirate:  gradient ", round(1000t2; digits = 1), " ms (the solver's dual code compiled again), pirate called ",
        n_solver, " times by the solver, ", CALLS[], " by plain code; gradients equal: ", g1 == g2)
