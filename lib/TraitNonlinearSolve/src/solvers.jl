# The solvers, an open set: the trait, the options, the front end, Newton–Raphson, a trust region, a polyalgorithm.
# (the body of the trait module `TraitNonlinearSolve`)

"""
    NLOptions(abstol, reltol, maxiters)

A solve's options: success when the residual's largest entry is at most `abstol`; `:Stalled` when a step (or a
trust region) shrinks below `reltol` times the solution's size first; at most `maxiters` iterations. The defaults
are NonlinearSolve's, `eps()^(4/5)` (about 3e-13) and 1000.
"""
struct NLOptions
    abstol::Float64
    reltol::Float64
    maxiters::Int
end
const NL_TOL = eps(Float64)^(4 / 5)
_nloptions(; abstol = NL_TOL, reltol = NL_TOL, maxiters = 1000) = NLOptions(Float64(abstol), Float64(reltol), Int(maxiters))

"""
    NonlinearAlgorithm

A nonlinear solver: NonlinearSolve's algorithms, an open set. A solver package implements it for its own algorithm
type. `solve_nl(a, m, opts)` solves the core's problem `m` (its parameters partially evaluated, W-63) through
`nl_residual!` and `nl_core_jacobian!`, and returns an `NLCoreSolution`.
"""
@trait NonlinearAlgorithm begin
    solve_nl(a::NonlinearAlgorithm, m::NLCoreProblem, opts::NLOptions)::NLCoreSolution
end

"""
    solve(prob, alg = NonlinearPolyalg(); abstol, reltol, maxiters) -> NonlinearSolution

Solve a problem of any type that implements `NonlinearProblemTrait` (TDE's `NonlinearProblem`, a
`SteadyStateProblem`, a problem type of its own) with any `NonlinearAlgorithm`. The package's front end; the
umbrella's `solve` forwards a nonlinear algorithm here. Neither specialized nor inferred on the problem's type.
"""
Base.@nospecializeinfer solve(@nospecialize(prob), @nospecialize(alg); kw...) = solve_nonlinear(prob, alg, _nloptions(; kw...))
Base.@nospecializeinfer solve(@nospecialize(prob); kw...) = solve(prob, NonlinearPolyalg(); kw...)
# Whether a value is a nonlinear algorithm or problem, by its trait: bounded methods with an untyped fallback after them
# (traits §5.5; lowering L-3), for a front end that serves several kinds (the umbrella's `solve`). By dispatch, not by
# `conforms`: a caller that infers the front end would otherwise infer TraitWorld's membership lookup (E-242).
"Whether `alg` is a `NonlinearAlgorithm` (an algorithm of other traits too counts as one)."
is_nonlinear_algorithm(alg::NonlinearAlgorithm) = true
is_nonlinear_algorithm(alg) = false
"Whether `prob` is a `NonlinearProblemTrait`."
is_nonlinear_problem(prob::NonlinearProblemTrait) = true
is_nonlinear_problem(prob) = false

"The outer layer: the problem's functions partially evaluated (`nl_core`), the algorithm, the solution with its problem."
Base.@nospecializeinfer solve_nonlinear(@nospecialize(prob), alg::NonlinearAlgorithm, opts::NLOptions) =
    nl_with_problem(solve_nl(alg, nl_core(prob), opts), prob, ODECore._system(prob))

_maxabs(x::Vector{Float64}) = maximum(abs, x; init = 0.0)
_norm2(x::Vector{Float64}) = sqrt(sum(abs2, x))
_dot(x::Vector{Float64}, y::Vector{Float64}) = sum(x[i] * y[i] for i in eachindex(x); init = 0.0)
"`ls`'s factorization of a copy of `J` (`J` is kept), or `nothing` where it is singular."
function _factor(ls::LinearSolver, W::Matrix{Float64}, J::Matrix{Float64})
    copyto!(W, J)
    try
        return linfactor(ls, W)
    catch e
        e isa LinearAlgebra.SingularException && return nothing
        rethrow()
    end
end
_float64_only(m::NLCoreProblem{T}, name::String) where {T} =
    T === Float64 || throw(ArgumentError("$name takes Float64 states; this problem's state has $T"))

"""
    NewtonRaphson(; linsolve = DenseLU(), linesearch = false)

Newton's method: the Jacobian at every iterate, the linear systems by `linsolve` (any `LinearSolver`); with
`linesearch`, the step halved until the residual's squared norm falls by Armijo's condition.
"""
struct NewtonRaphson{L<:LinearSolver}
    linsolve::L
    linesearch::Bool
end
NewtonRaphson(; linsolve = DenseLU(), linesearch = false) = NewtonRaphson(linsolve, linesearch)
@implement NonlinearAlgorithm NewtonRaphson{L} where {L} begin
    solve_nl(a, m, opts) = newton_raphson(a.linsolve, a.linesearch, m, opts)
end

function newton_raphson(ls::LinearSolver, linesearch::Bool, m::NLCoreProblem, opts::NLOptions)::NLCoreSolution
    _float64_only(m, "NewtonRaphson")
    x = copy(m.u0)::Vector{Float64}
    n = length(x)
    r = zeros(n); J = zeros(n, n); W = zeros(n, n); dx = zeros(n); xt = zeros(n); rt = zeros(n)
    nf, nj, nfac = 1, 0, 0
    nl_residual!(m, r, x)
    for k in 0:opts.maxiters
        _maxabs(r) <= opts.abstol && return NLCoreSolution{Float64}(x, r, :Success, NLStats(k, nf, nj, nfac))
        k == opts.maxiters && break
        nl_core_jacobian!(m, J, x); nj += 1
        F = _factor(ls, W, J); nfac += 1
        F === nothing && return NLCoreSolution{Float64}(x, r, :ConvergenceFailure, NLStats(k, nf, nj, nfac))
        solve!(F, dx, r)
        all(isfinite, dx) || return NLCoreSolution{Float64}(x, r, :ConvergenceFailure, NLStats(k, nf, nj, nfac))
        φ0 = sum(abs2, r)
        α = 1.0
        while true
            @inbounds for i in 1:n; xt[i] = x[i] - α * dx[i]; end
            nl_residual!(m, rt, xt); nf += 1
            # Armijo on ‖r‖²: the Newton direction's slope is -2‖r‖²
            (!linesearch || α < 1e-8 || (all(isfinite, rt) && sum(abs2, rt) <= (1 - 2e-4 * α) * φ0)) && break
            α /= 2
        end
        copyto!(x, xt); copyto!(r, rt)
        if α * _maxabs(dx) <= opts.reltol * max(_maxabs(x), 1.0) && _maxabs(r) > opts.abstol
            return NLCoreSolution{Float64}(x, r, :Stalled, NLStats(k + 1, nf, nj, nfac))
        end
    end
    return NLCoreSolution{Float64}(x, r, :MaxIters, NLStats(opts.maxiters, nf, nj, nfac))
end

"""
    TrustRegion(; linsolve = DenseLU(), initial_radius = 1.0, max_radius = 1e4)

Powell's dogleg trust region on `‖r‖²`: the Newton step (by `linsolve`) where it lies within the region, else
the Cauchy step or the dogleg between them; the region shrinks where the model predicts badly and grows where it
predicts well. Robust where Newton's full steps diverge; a singular Jacobian takes the steepest descent.
"""
struct TrustRegion{L<:LinearSolver}
    linsolve::L
    initial_radius::Float64
    max_radius::Float64
end
TrustRegion(; linsolve = DenseLU(), initial_radius = 1.0, max_radius = 1e4) =
    TrustRegion(linsolve, Float64(initial_radius), Float64(max_radius))
@implement NonlinearAlgorithm TrustRegion{L} where {L} begin
    solve_nl(a, m, opts) = trust_region(a.linsolve, a.initial_radius, a.max_radius, m, opts)
end

function trust_region(ls::LinearSolver, Δ0::Float64, Δmax::Float64, m::NLCoreProblem, opts::NLOptions)::NLCoreSolution
    _float64_only(m, "TrustRegion")
    x = copy(m.u0)::Vector{Float64}
    n = length(x)
    r = zeros(n); J = zeros(n, n); W = zeros(n, n); pN = zeros(n); g = zeros(n); Jg = zeros(n)
    p = zeros(n); Jp = zeros(n); xt = zeros(n); rt = zeros(n)
    nf, nj, nfac = 1, 0, 0
    nl_residual!(m, r, x)
    Δ = Δ0
    fresh = true                     # the Jacobian and the steps at `x` are to be computed
    okN = false
    for k in 0:opts.maxiters
        _maxabs(r) <= opts.abstol && return NLCoreSolution{Float64}(x, r, :Success, NLStats(k, nf, nj, nfac))
        k == opts.maxiters && break
        if fresh
            nl_core_jacobian!(m, J, x); nj += 1
            F = _factor(ls, W, J); nfac += 1
            okN = F !== nothing
            if okN
                solve!(F, pN, r)
                pN .*= -1
                okN = all(isfinite, pN)
            end
            LinearAlgebra.mul!(g, transpose(J), r)          # the gradient of ½‖r‖²
            LinearAlgebra.mul!(Jg, J, g)
            fresh = false
        end
        if okN && _norm2(pN) <= Δ
            copyto!(p, pN)
        else
            gg = sum(abs2, g)
            gg == 0 && return NLCoreSolution{Float64}(x, r, :Stalled, NLStats(k, nf, nj, nfac))
            τ = gg / max(sum(abs2, Jg), floatmin(Float64))
            if !okN || τ * sqrt(gg) >= Δ                     # the Cauchy step at the boundary
                @inbounds for i in 1:n; p[i] = -(Δ / sqrt(gg)) * g[i]; end
            else                                             # the dogleg: from the Cauchy point towards Newton's
                a = 0.0; b = 0.0; c = τ^2 * gg - Δ^2
                @inbounds for i in 1:n
                    d = pN[i] + τ * g[i]
                    a += d^2; b += 2 * (-τ * g[i]) * d
                end
                s = (-b + sqrt(max(b^2 - 4a * c, 0.0))) / (2a)
                @inbounds for i in 1:n; p[i] = -τ * g[i] + s * (pN[i] + τ * g[i]); end
            end
        end
        LinearAlgebra.mul!(Jp, J, p)
        φ = sum(abs2, r)
        pred = φ - sum(abs2, r[i] + Jp[i] for i in 1:n)
        @inbounds for i in 1:n; xt[i] = x[i] + p[i]; end
        nl_residual!(m, rt, xt); nf += 1
        φt = all(isfinite, rt) ? sum(abs2, rt) : Inf
        ρ = pred > 0 ? (φ - φt) / pred : -1.0
        np = _norm2(p)
        if ρ < 0.25
            Δ = 0.25 * np
        elseif ρ > 0.75 && np >= 0.99 * Δ
            Δ = min(2Δ, Δmax)
        end
        if ρ > 1e-4
            copyto!(x, xt); copyto!(r, rt)
            fresh = true
        end
        Δ <= opts.reltol * max(_maxabs(x), 1.0) && _maxabs(r) > opts.abstol &&
            return NLCoreSolution{Float64}(x, r, :Stalled, NLStats(k + 1, nf, nj, nfac))
    end
    return NLCoreSolution{Float64}(x, r, :MaxIters, NLStats(opts.maxiters, nf, nj, nfac))
end

"""
    NonlinearPolyalg(algs...)
    NonlinearPolyalg()

A polyalgorithm: its members (any `NonlinearAlgorithm`s, kept with their attributes, so its type has no axis for
them) tried in turn from the guess until one succeeds; else the result with the smallest residual, with its return
code. The default, `NonlinearPolyalg()`, is Newton–Raphson, then Newton–Raphson with a line search, then the
trust region (NonlinearSolve's default polyalgorithm's Jacobian-based members); `solve(prob)` uses it.
"""
struct NonlinearPolyalg
    algs::Vector{Kept{NonlinearAlgorithm}}
end
# (no `@nospecialize` on the vararg: lowering L-46)
function NonlinearPolyalg(a1, rest...)
    v = Kept{NonlinearAlgorithm}[]
    for a in (a1, rest...)
        push!(v, Kept{NonlinearAlgorithm}(a))
    end
    return NonlinearPolyalg(v)
end
NonlinearPolyalg() = NonlinearPolyalg(NewtonRaphson(), NewtonRaphson(linesearch = true), TrustRegion())
@implement NonlinearAlgorithm NonlinearPolyalg begin
    solve_nl(a, m, opts) = polyalg_solve(a.algs, m, opts)
end
function polyalg_solve(algs::Vector{Kept{NonlinearAlgorithm}}, m::NLCoreProblem, opts::NLOptions)::NLCoreSolution
    isempty(algs) && throw(ArgumentError("a polyalgorithm without members"))
    best = nothing
    stats = NLStats(0, 0, 0, 0)
    for alg in algs
        s = solve_nl(alg, m, opts)::NLCoreSolution
        stats = stats + s.stats
        if successful_retcode(s.retcode)
            return typeof(s)(s.u, s.resid, s.retcode, stats)
        end
        (best === nothing || maximum(abs, s.resid; init = 0.0) < maximum(abs, best.resid; init = 0.0)) && (best = s)
    end
    return typeof(best)(best.u, best.resid, best.retcode, stats)
end
