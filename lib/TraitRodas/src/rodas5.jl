"""
    Rodas5(; chunk_size = nothing, linsolve = DenseLU())

Rodas5 with the problem's `jacobian!` (`chunk_size = nothing`: TDE's `ODEProblem` by duals in chunks
of `DUAL_CHUNK`), or with the problem's plain function (`rhs_function`) entered at `init` at the duals
of chunk `chunk_size` (`0`: picked from the state's length, as ForwardDiff's `pickchunksize`). Its linear
systems are solved by `linsolve`, any `LinearSolver` (stage C): kept as the user's object, so the solver is
compiled per linear solver, or `with_dynamic(ls)`, compiled once for every linear solver.
"""
struct Rodas5{L <: LinearSolver}
    chunk_size::Union{Nothing,Int}
    linsolve::L
end
Rodas5(; chunk_size = nothing, linsolve = DenseLU()) = Rodas5(chunk_size, linsolve)

# --- Rodas5 (tableau from OrdinaryDiffEqRosenbrockTableaus 2.0.0; Di Marzo 1993) --------------------

const R5A = [
    0 0 0 0 0 0 0 0
    2.0 0 0 0 0 0 0 0
    3.040894194418781 1.041747909077569 0 0 0 0 0 0
    2.576417536461461 1.62208306077664 -0.9089668560264532 0 0 0 0 0
    2.760842080225597 1.446624659844071 -0.3036980084553738 0.2877498600325443 0 0 0 0
    -14.09640773051259 6.925207756232704 -41.47510893210728 2.343771018586405 24.13215229196062 0 0 0
    -14.09640773051259 6.925207756232704 -41.47510893210728 2.343771018586405 24.13215229196062 1 0 0
    -14.09640773051259 6.925207756232704 -41.47510893210728 2.343771018586405 24.13215229196062 1 1 0
]
const R5C = [
    0 0 0 0 0 0 0
    -10.31323885133993 0 0 0 0 0 0
    -21.04823117650003 -7.234992135176716 0 0 0 0 0
    32.22751541853323 -4.943732386540191 19.44922031041879 0 0 0 0
    -20.69865579590063 -8.816374604402768 1.260436877740897 -0.7495647613787146 0 0 0
    -46.22004352711257 -17.49534862857472 -289.6389582892057 93.60855400400906 318.3822534212147 0 0
    34.20013733472935 -14.1553540271769 57.823356409884 25.83362985412365 1.408950972071624 -6.551835421242162 0
    42.57076742291101 -13.80770672017997 93.98938432427124 18.77919633714503 -31.5835918722337 -6.685968952921985 -5.810979938412932
]
const R5c = [0, 0.38, 0.3878509998321533, 0.483971893787384, 0.457047700881958, 1, 1, 1]
const R5d = [0.19, -0.1823079225333714636, -0.319231832186874912, 0.3449828624725343, -0.377417564392089818, 0, 0, 0]
const R5H = [
    27.354592673333357 -6.925207756232857 26.40037733258859 0.5635230501052979 -4.699151156849391 -1.6008677469422725 -1.5306074446748028 -1.3929872940716344
    44.19024239501722 1.3677947663381929e-13 202.93261852171622 -35.5669339789154 -181.91095152160645 3.4116351403665033 2.5793540257308067 2.2435122582734066
    -44.0988150021747 -5.755396159656812e-13 -181.26175034586677 56.99302194811676 183.21182741427398 -7.480257918273637 -5.792426076169686 -5.32503859794143
]
const R5gamma = 0.19
const R5b = [R5A[8, 1:7]..., 1.0]

"""
    DualJacobian{N, FD}

The core's plain function (partially evaluated on the parameters) entered at Rodas5's `init` at the
library's duals of chunk `N`, picked per problem from the state's length: `fd::FD` is bounded by
`Callable{CoreDualSig{N}, Nothing}` (traits reference §7.4), so `N` is the integrator's, not the
problem's. With the dual buffers.
"""
struct DualJacobian{N,FD<:Callable{CoreDualSig{N},Nothing}}
    fd::FD
    ud::Vector{FDual{N}}
    dd::Vector{FDual{N}}
end
"The Jacobian by the integrator's duals: one call of the function per chunk of `N` columns."
function dual_jacobian!(jc::DualJacobian{N}, J::Matrix{Float64}, u::Vector{Float64}, t::Float64) where {N}
    n = length(u)
    ud, dd = jc.ud, jc.dd
    for c0 in 1:N:n
        for j in 1:n
            ud[j] = FDual{N}(u[j], ForwardDiff.Partials(ntuple(k -> j == c0 + k - 1 ? 1.0 : 0.0, Val(N))))
        end
        jc.fd(dd, ud, t)::Nothing
        for k in 1:N
            col = c0 + k - 1
            col > n && break
            for i in 1:n
                J[i, col] = ForwardDiff.partials(dd[i], k)
            end
        end
    end
    return nothing
end
"Enter the plain function `f` at the duals of chunk `N` (`f` as given: a cast or a kept value keeps its attribute)."
Base.@nospecializeinfer dual_jacobian(@nospecialize(f), n::Int, ::Val{N}) where {N} =
    DualJacobian(f, Vector{FDual{N}}(undef, n), Vector{FDual{N}}(undef, n))

"Rodas5's state between steps: the integrator's view for callbacks, the stages, the Jacobian and W; `jc`, the function at the duals entered at `init` (`DualJacobian`), or `nothing`: the problem's `jacobian!`."
mutable struct Rodas5Integrator{JC}
    const u::Vector{Float64}
    const uprev::Vector{Float64}
    t::Float64
    dt::Float64
    u_modified::Bool
    terminated::Bool
    const unew::Vector{Float64}
    const ks::Vector{Vector{Float64}}
    const kint::Vector{Vector{Float64}}
    const fsal::Vector{Float64}
    const du::Vector{Float64}
    const du1::Vector{Float64}
    const dT::Vector{Float64}
    const ltmp::Vector{Float64}
    const J::Matrix{Float64}
    const W::Matrix{Float64}
    qold::Float64
    naccept::Int
    nreject::Int
    nf::Int
    njac::Int
    const jc::JC
    const ls::LinearSolver           # Kept{LinearSolver}: W's factorization, per step
    tprev::Float64
    const tstops::Vector{Float64}
    const callbacks::Vector{StepCallback{Vector{Float64},Float64}}
end
step_tstop!(i::Rodas5Integrator, τ::Float64) = (_add_tstop!(i.tstops, i.t, τ); nothing)
step_terminate!(i::Rodas5Integrator) = (i.terminated = true; nothing)
"The state at `τ` within the last accepted step, by Rodas5's interpolant."
function step_interp!(i::Rodas5Integrator, out::Vector{Float64}, τ::Float64)
    θ = (τ - i.tprev) / i.dt
    θ1 = 1 - θ
    k1, k2, k3 = i.kint[1], i.kint[2], i.kint[3]
    @inbounds for j in eachindex(out)
        out[j] = θ1 * i.uprev[j] + θ * (i.u[j] + θ1 * (k1[j] + θ * (k2[j] + θ * k3[j])))
    end
    return nothing
end
@implement StepView Rodas5Integrator{JC} where {JC}

"One Rodas5 step of size `dt` from `i.u`: the candidate in `i.unew`, the interpolant's k's in `i.kint`; returns the error estimate."
function rodas5_step!(m, i::Rodas5Integrator, dt::Float64, abstol::Float64, reltol::Float64)
    u, t, n = i.u, i.t, length(i.u)
    ks, du, du1, dT, ltmp, J, W, unew = i.ks, i.du, i.du1, i.dT, i.ltmp, i.J, i.W, i.unew
    core_rhs!(m, i.fsal, u, t)
    i.jc === nothing ? core_jacobian!(m, J, u, t) : dual_jacobian!(i.jc, J, u, t)
    i.njac += 1
    h = sqrt(eps(Float64)) * max(abs(t), 1.0)          # the time derivative, by a forward difference
    core_rhs!(m, du, u, t + h)
    i.nf += 2
    @inbounds for j in 1:n; dT[j] = (du[j] - i.fsal[j]) / h; end
    dtgamma = dt * R5gamma
    copyto!(W, J)                                      # W = J - I / (dt γ), as OrdinaryDiffEq's W_transform
    @inbounds for k in 1:n; W[k, k] -= 1 / dtgamma; end
    F = linfactor(i.ls, W)                             # a Solvable, for the eight stages
    @inbounds for j in 1:n; ltmp[j] = i.fsal[j] + dt * R5d[1] * dT[j]; end
    solve!(F, ks[1], ltmp)
    ks[1] .*= -1
    for stage in 2:8
        @inbounds for j in 1:n
            acc = u[j]
            for l in 1:stage-1; acc += R5A[stage, l] * ks[l][j]; end
            unew[j] = acc
        end
        if !(stage > 2 && R5c[stage] == R5c[stage-1] && all(l -> R5A[stage, l] == R5A[stage-1, l], 1:stage-1))
            core_rhs!(m, du, unew, t + R5c[stage] * dt)
            i.nf += 1
        end
        @inbounds for j in 1:n
            acc = 0.0
            for l in 1:stage-1; acc += R5C[stage, l] / dt * ks[l][j]; end
            ltmp[j] = du[j] + dt * R5d[stage] * dT[j] + acc
        end
        solve!(F, ks[stage], ltmp)
        ks[stage] .*= -1
    end
    acc2 = 0.0
    @inbounds for j in 1:n
        y = u[j]
        for l in 1:8; y += R5b[l] * ks[l][j]; end
        unew[j] = y
        sc = abstol + max(abs(u[j]), abs(y)) * reltol
        acc2 += abs2(ks[8][j] / sc)                    # the embedded error: btilde = (0, …, 0, 1)
    end
    @inbounds for r in 1:3, j in 1:n
        acc = 0.0
        for l in 1:8; acc += R5H[r, l] * ks[l][j]; end
        i.kint[r][j] = acc
    end
    return sqrt(acc2 / n)
end

"Rodas5's interpolant between `u` (at θ = 0) and `unew` (θ = 1)."
function rodas5_interp!(out::Vector{Float64}, θ::Float64, i::Rodas5Integrator)
    θ1 = 1 - θ
    k1, k2, k3 = i.kint[1], i.kint[2], i.kint[3]
    @inbounds for j in eachindex(out)
        out[j] = θ1 * i.u[j] + θ * (i.unew[j] + θ1 * (k1[j] + θ * (k2[j] + θ * k3[j])))
    end
    return out
end

#     solve_core(Rodas5(), m::CoreProblem, callbacks, opts) -> ODESolution
#
# Rodas5 with the core's Jacobian (`core_jacobian!`: the given one, by the library's duals, or by finite
# differences), or with the core's plain function entered at `init` at duals of a chunk picked per
# problem (`Rodas5(chunk_size = 0)`), dense LU, and a PI step-size controller. `Float64` states only.
@implement ODEAlgorithm Rodas5{L} where {L} begin
    solve_core(a, m, callbacks, opts) = rodas5_solve(m, a, m.u0, m.tspan, callbacks, opts)
end
rodas5_solve(m, alg::Rodas5, u0::Vector{T}, tspan, callbacks, opts) where {T} =
    throw(ArgumentError("Rodas5 takes Float64 states; this problem's state has $T"))

"""
Rodas5's `init` for the core's problem `m`: with `alg.chunk_size` an `Int` and a plain function from
the problem (`core_rhs_function`), the function is entered at the duals of chunk `N` here
(`DualJacobian`), and the driver is compiled per `N`; otherwise the core's Jacobian.
"""
function rodas5_solve(m, alg::Rodas5, u0::Vector{Float64}, tspan::Tuple{Float64,Float64},
                      callbacks::Vector{Any}, opts::SolverOptions)
    alg.chunk_size === nothing && return rodas5_run(m, u0, tspan, callbacks, opts, nothing, alg.linsolve)
    m isa CoreProblem && m.ad || return rodas5_run(m, u0, tspan, callbacks, opts, nothing, alg.linsolve)
    n = length(u0)
    N = alg.chunk_size == 0 ? ForwardDiff.pickchunksize(n) : alg.chunk_size
    jc = dual_jacobian(m.f, n, Val(N))         # the problem's one function, seen at the duals of chunk `N`
    return rodas5_run(m, u0, tspan, callbacks, opts, jc, alg.linsolve)::ODESolution{Float64}   # a barrier per N
end

"Rodas5 for the core's problem `m`, its Jacobian by `jc` (`nothing`: the core's)."
function rodas5_run(m, u0::Vector{Float64}, tspan::Tuple{Float64,Float64},
                    callbacks::Vector{Any}, opts::SolverOptions, jc, ls)
    t0, tf = tspan
    n = length(u0)
    z() = zeros(Float64, n)
    d, c = enter_step_callbacks(callbacks, Vector{Float64})
    isempty(c) || throw(ArgumentError("continuous callbacks: Tsit5 only"))
    i = Rodas5Integrator{typeof(jc)}(copy(u0), copy(u0), t0, 0.0, false, false, z(),
                                    [z() for _ in 1:8], [z() for _ in 1:3], z(), z(), z(), z(), z(),
                                    zeros(Float64, n, n), zeros(Float64, n, n), 1e-4, 0, 0, 0, 0, jc, ls,
                                    t0, Float64[], d)
    iv = Dynamic{StepView{Vector{Float64},Float64}}(i)
    # the initial step: as Tsit5's driver, from f(u0)
    core_rhs!(m, i.fsal, i.u, t0)
    i.nf += 1
    sc(x) = opts.abstol + abs(x) * opts.reltol
    d0 = sqrt(sum(abs2(i.u[j] / sc(i.u[j])) for j in 1:n) / n)
    d1 = sqrt(sum(abs2(i.fsal[j] / sc(i.u[j])) for j in 1:n) / n)
    dt = min((d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1, opts.dtmax)
    ts = Float64[t0]
    us = Vector{Float64}[copy(u0)]
    saveat = opts.saveat
    isave = 1
    if !isempty(saveat) && saveat[1] == t0
        isave = 2
    elseif !isempty(saveat)
        empty!(ts); empty!(us)
    end
    beta1, beta2, gamma, qmin, qmax = 7 / 50, 2 / 25, 0.9, 0.2, 10.0
    retcode = :Success
    iters = 0
    while i.t < tf && !i.terminated
        iters += 1
        if iters > opts.maxiters
            retcode = :MaxIters
            break
        end
        dt = min(dt, tf - i.t)
        tstop_hit = false
        if !isempty(i.tstops) && i.tstops[1] - i.t <= dt
            dt = i.tstops[1] - i.t
            tstop_hit = true
        end
        EEst = rodas5_step!(m, i, dt, opts.abstol, opts.reltol)
        q11 = EEst^beta1
        if EEst <= 1.0
            tnew = tstop_hit ? popfirst!(i.tstops) : i.t + dt
            if isempty(saveat)
                push!(ts, tnew); push!(us, copy(i.unew))
            else
                while isave <= length(saveat) && saveat[isave] <= tnew
                    push!(ts, saveat[isave])
                    push!(us, rodas5_interp!(zeros(Float64, n), (saveat[isave] - i.t) / dt, i))
                    isave += 1
                end
            end
            copyto!(i.uprev, i.u)
            copyto!(i.u, i.unew)
            i.tprev = i.t
            i.t = tnew
            i.dt = dt
            i.naccept += 1
            for cb in i.callbacks
                if step_condition(cb, i.u, i.t, iv)
                    i.u_modified = true
                    step_affect!(cb, iv)
                    i.u_modified = false
                end
                i.terminated && break
            end
            q = clamp(q11 / i.qold^beta2 / gamma, 1 / qmax, 1 / qmin)
            dt = min(dt / q, opts.dtmax)
            i.qold = max(EEst, 1e-4)
        else
            i.nreject += 1
            dt = dt / min(1 / qmin, q11 / gamma)
        end
    end
    i.terminated && (retcode = :Terminated)
    return ODESolution{Float64}(ts, us, retcode, i.naccept, i.nreject, i.nf)
end
