# Row 9: the three error cases of findings §17, measured as there: the full message with its stack
# trace (`showerror` with the backtrace), its lines, bytes, frames and longest line.
#
#     JULIA_LOAD_PATH="scripts/env:@stdlib" julia --startup-file=no scripts/errors.jl
using TraitDifferentialEquations
const U0 = [1.0, 0.0, 0.0]
function report(name, thunk)
    msg = try
        thunk()
        "no error"
    catch e
        sprint((io, e, bt) -> showerror(IOContext(io, :limit => false), e, bt), e, catch_backtrace())
    end
    lines = split(msg, '\n')
    frames = count(l -> occursin(r"^\s*\[\d+\]", l), lines)
    println(rpad(name, 52), " lines ", length(lines), ", bytes ", sizeof(msg), ", frames ", frames,
            ", longest line ", maximum(length, lines))
    get(ENV, "SHOW", "0") == "1" && println(msg, "\n")
end
# 1. the right-hand side uses parameters, the problem has none
lorenz!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
report("1. parameters used but never passed", () -> solve(ODEProblem(lorenz!, U0, (0.0, 1.0)), Tsit5()))
# 2. a Float64 cache inside the right-hand side, under a gradient (dual numbers)
const cache = zeros(3)
cached!(du, u, p, t) = (cache .= u; du[1] = p[1] * (cache[2] - cache[1]); du[2] = cache[1] * (p[2] - cache[3]) - cache[2]; du[3] = cache[1] * cache[2] - p[3] * cache[3]; nothing)
report("2. a Float64 cache under gradient", () -> gradient(p -> sum(solve(ODEProblem(cached!, U0, (0.0, 1.0), p), Tsit5()).u[end]), [10.0, 28.0, 8 / 3]))
# 3. the same cache under Rodas5's Jacobian (the library's dual numbers)
report("3. the same cache under Rodas5's Jacobian", () -> solve(ODEProblem(cached!, U0, (0.0, 1.0), [10.0, 28.0, 8 / 3]), Rodas5()))
