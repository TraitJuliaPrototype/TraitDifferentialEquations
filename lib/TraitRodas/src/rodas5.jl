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

"""
    Rodas5Steps

Rodas5's dense output (open issue 117): per accepted step its start `t0`, its end `t1` (an event's time where an
event cut it), its size `dt`, the states at both ends of the full step and the interpolant's three terms; the state
between by Rodas5's own interpolant (`rodas5_interp!`), as its `saveat` and event location take it.
"""
struct Rodas5Steps
    t0::Vector{Float64}
    t1::Vector{Float64}
    dt::Vector{Float64}
    u0::Vector{Vector{Float64}}
    u1::Vector{Vector{Float64}}
    k::Vector{Vector{Vector{Float64}}}
end
Rodas5Steps() = Rodas5Steps(Float64[], Float64[], Float64[], Vector{Float64}[], Vector{Float64}[], Vector{Vector{Float64}}[])
function push_step!(d::Rodas5Steps, t0::Float64, t1::Float64, dt::Float64, u0::Vector{Float64}, u1::Vector{Float64},
                    kint::Vector{Vector{Float64}})
    push!(d.t0, t0); push!(d.t1, t1); push!(d.dt, dt)
    push!(d.u0, copy(u0)); push!(d.u1, copy(u1)); push!(d.k, [copy(k) for k in kint])
    return nothing
end
"""
Rodas5's state between steps, with its problem and the driver's state: the integrator's view for callbacks, the
stages, the Jacobian and W; `jc`, the function at the duals entered at `init` (`DualJacobian`), or `nothing`: the
problem's `jacobian!`. After an accepted step `uprev` is its start, `unew` its end and `kint` its interpolant's
terms, until the next step.
"""
mutable struct Rodas5Integrator{JC,M}
    const m::M
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
    const events::Vector{StepEvent{Vector{Float64},Float64}}
    const evprev::Vector{Float64}
    const utilde::Vector{Float64}
    # the driver's state between steps
    iv::Union{Nothing,Dynamic{StepView{Vector{Float64},Float64}}}
    const opts::SolverOptions
    const tf::Float64
    dtnext::Float64
    const ts::Vector{Float64}
    const us::Vector{Vector{Float64}}
    isave::Int
    iters::Int
    retcode::Symbol
    dt_reset::Bool
    const dense::Union{Nothing,Rodas5Steps}   # the steps, for dense output (`opts.dense`)
end
step_tstop!(i::Rodas5Integrator, τ::Float64) = (_add_tstop!(i.tstops, i.t, τ); nothing)
step_terminate!(i::Rodas5Integrator) = (i.terminated = true; nothing)
"The state at `τ` within the last accepted step, by Rodas5's interpolant (`uprev` to `unew`)."
function step_interp!(i::Rodas5Integrator, out::Vector{Float64}, τ::Float64)
    rodas5_interp!(out, (τ - i.tprev) / i.dt, i.uprev, i.unew, i.kint)
    return nothing
end
step_dt_reset!(i::Rodas5Integrator) = (i.dt_reset = true; nothing)
@implement StepView Rodas5Integrator{JC,M} where {JC,M}
@implement CoreIntegrator{Vector{Float64},Float64} Rodas5Integrator{JC,M} where {JC,M} begin
    advance!(i) = rodas5_advance!(i)
    core_solution(i) = CoreSolution{Float64}(i.ts, i.us, _retcode(i), i.naccept, i.nreject, i.nf,
                                             kept_interpolant(_steps(i.dense)))
end

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

"Rodas5's interpolant between `u0` (at θ = 0) and `u1` (θ = 1), with the step's terms `kint`."
function rodas5_interp!(out::Vector{Float64}, θ::Float64, u0::Vector{Float64}, u1::Vector{Float64}, kint::Vector{Vector{Float64}})
    θ1 = 1 - θ
    k1, k2, k3 = kint[1], kint[2], kint[3]
    @inbounds for j in eachindex(out)
        out[j] = θ1 * u0[j] + θ * (u1[j] + θ1 * (k1[j] + θ * (k2[j] + θ * k3[j])))
    end
    return out
end
# Rodas5's dense output, by its interpolant (after it: the implementation's check infers the result)
@implement Interpolant{Vector{Float64},Float64} Rodas5Steps begin
    interpolate(d, τ) = (k = dense_step(d.t1, τ);
                         rodas5_interp!(similar(d.u0[k]), (τ - d.t0[k]) / d.dt[k], d.u0[k], d.u1[k], d.k[k]))
end

#     solve_core(Rodas5(), m::CoreProblem, callbacks, opts) -> CoreSolution
#
# Rodas5 with the core's Jacobian (`core_jacobian!`: the given one, by the library's duals, or by finite
# differences), or with the core's plain function entered at `init` at duals of a chunk picked per
# problem (`Rodas5(chunk_size = 0)`), dense LU, and a PI step-size controller. `Float64` states only.
# Discrete and continuous callbacks; the driver is the integrator's `advance!`, so `init` steps it too.
@implement ODEAlgorithm Rodas5{L} where {L} begin
    solve_core(a, m, callbacks, opts) = rodas5_solve(m, a, m.u0, m.tspan, callbacks, opts)
end
@implement ODEStepper Rodas5{L} where {L} begin
    init_core(a, m, callbacks, opts) = rodas5_init(m, a, m.u0, m.tspan, callbacks, opts)
end
rodas5_solve(m, alg::Rodas5, u0::Vector{T}, tspan, callbacks, opts) where {T} =
    throw(ArgumentError("Rodas5 takes Float64 states; this problem's state has $T"))
rodas5_init(m, alg::Rodas5, u0::Vector{T}, tspan, callbacks, opts) where {T} =
    throw(ArgumentError("Rodas5 takes Float64 states; this problem's state has $T"))

"""
Rodas5's `init` for the core's problem `m`: with `alg.chunk_size` an `Int` and a plain function from
the problem (`core_rhs_function`), the function is entered at the duals of chunk `N` here
(`DualJacobian`), and the driver is compiled per `N`; otherwise the core's Jacobian.
"""
function rodas5_init(m, alg::Rodas5, u0::Vector{Float64}, tspan::Tuple{Float64,Float64},
                     callbacks::Vector{Any}, opts::SolverOptions)
    alg.chunk_size === nothing && return rodas5_start(m, u0, tspan, callbacks, opts, nothing, alg.linsolve)
    m isa CoreProblem && m.ad || return rodas5_start(m, u0, tspan, callbacks, opts, nothing, alg.linsolve)
    n = length(u0)
    N = alg.chunk_size == 0 ? ForwardDiff.pickchunksize(n) : alg.chunk_size
    jc = dual_jacobian(m.f, n, Val(N))         # the problem's one function, seen at the duals of chunk `N`
    return rodas5_start(m, u0, tspan, callbacks, opts, jc, alg.linsolve)
end
"Rodas5 to the end of the span: its `init`, then `advance!` until it cannot step; a barrier per chunk."
rodas5_solve(m, alg::Rodas5, u0::Vector{Float64}, tspan::Tuple{Float64,Float64}, callbacks::Vector{Any}, opts::SolverOptions) =
    _rodas5_run(rodas5_init(m, alg, u0, tspan, callbacks, opts))::CoreSolution{Float64}
_rodas5_run(i::Rodas5Integrator) = run_to_end!(i)

"""
Rodas5's integrator for the core's problem `m`, its Jacobian by `jc` (`nothing`: the core's): the core's callbacks
kept, the times to stop at, the callbacks' `initialize`, the first step's size from `f(u0)` (as Tsit5's driver),
the first save and the continuous callbacks' conditions at the start.
"""
function rodas5_start(m, u0::Vector{Float64}, tspan::Tuple{Float64,Float64},
                      callbacks::Vector{Any}, opts::SolverOptions, jc, ls)
    t0, tf = tspan
    n = length(u0)
    z() = zeros(Float64, n)
    d, c = enter_step_callbacks(callbacks, Vector{Float64})
    i = Rodas5Integrator{typeof(jc),typeof(m)}(m, copy(u0), copy(u0), t0, 0.0, false, false, z(),
                                    [z() for _ in 1:8], [z() for _ in 1:3], z(), z(), z(), z(), z(),
                                    zeros(Float64, n, n), zeros(Float64, n, n), 1e-4, 0, 0, 0, 0, jc, ls,
                                    t0, Float64[], d, c, zeros(Float64, length(c)), z(),
                                    nothing, opts, tf, 0.0, Float64[], Vector{Float64}[], 1, 0, :Default, false,
                                    opts.dense ? Rodas5Steps() : nothing)
    iv = Dynamic{StepView{Vector{Float64},Float64}}(i)
    i.iv = iv
    for τ in opts.tstops
        _add_tstop!(i.tstops, t0, τ)
    end
    isempty(callbacks) || initialize_step_callbacks!(i.callbacks, i.events, i.u, i.t, iv)
    i.u_modified = false
    i.dtnext = rodas5_initdt!(i)
    _save_start!(i.ts, i.us, i.u, t0, opts.saveat) && (i.isave = 2)
    for k in eachindex(i.events)
        i.evprev[k] = step_event_condition(i.events[k], i.u, i.t, iv)
    end
    return i
end

"The first step's size: as Tsit5's driver's start, from `f(u, t)` (in `fsal`); also `auto_dt_reset!`'s."
function rodas5_initdt!(i::Rodas5Integrator)
    m, opts, n = i.m, i.opts, length(i.u)
    core_rhs!(m, i.fsal, i.u, i.t)
    i.nf += 1
    sc(x) = opts.abstol + abs(x) * opts.reltol
    d0 = sqrt(sum(abs2(i.u[j] / sc(i.u[j])) for j in 1:n) / n)
    d1 = sqrt(sum(abs2(i.fsal[j] / sc(i.u[j])) for j in 1:n) / n)
    return min((d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1, opts.dtmax)
end

"""
Rodas5's `advance!`: one accepted step (rejected attempts within it), its continuous callbacks' events (bisection
on the interpolant, as Tsit5's), the saves (`save_step!`; before and after an event's affect), then the discrete callbacks; `false` when it cannot step.
"""
function rodas5_advance!(i::Rodas5Integrator)
    m, opts, tf = i.m, i.opts, i.tf
    iv = i.iv::Dynamic{StepView{Vector{Float64},Float64}}
    events = i.events
    if i.u_modified                                  # changed between steps: the conditions again
        for k in eachindex(events)
            i.evprev[k] = step_event_condition(events[k], i.u, i.t, iv)
        end
        i.u_modified = false
    end
    if i.dt_reset
        i.dtnext = rodas5_initdt!(i)
        i.dt_reset = false
    end
    dt = i.dtnext
    beta1, beta2, gamma, qmin, qmax = 7 / 50, 2 / 25, 0.9, 0.2, 10.0
    while i.t < tf && !i.terminated
        i.iters += 1
        if i.iters > opts.maxiters
            i.retcode = :MaxIters
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
            tnew = tstop_hit ? i.tstops[1] : i.t + dt
            copyto!(i.uprev, i.u)
            copyto!(i.u, i.unew)
            i.tprev = i.t
            i.t = tnew
            i.dt = dt
            i.naccept += 1
            ke, up = 0, false
            if !isempty(events)
                ke, te, up = locate_events!(events, i.evprev, iv, i.u, i.utilde, i.tprev, i.t, i.dt)
                if ke != 0                                   # the integrator moves to the earliest event
                    step_interp!(iv, i.utilde, te)
                    copyto!(i.u, i.utilde)
                    i.t = te
                end
            end
            i.dense === nothing || push_step!(i.dense, i.tprev, i.t, dt, i.uprev, i.unew, i.kint)
            tstop_hit && ke == 0 && popfirst!(i.tstops)      # a tstop behind an event stays
            saved = save_step!(i)
            if ke != 0
                saved || save_state!(i)                      # the state before the affect
                i.u_modified = true
                event_affect_dir!(events[ke], iv, up)
                i.u_modified && save_state!(i)               # and after it
                saved = false
                i.u_modified = false
                for k in eachindex(events)                   # the conditions again at the event's state;
                    i.evprev[k] = k == ke ? NaN : step_event_condition(events[k], i.u, i.t, iv)
                end                                          # the event's own a little into the next step
            end
            for cb in i.callbacks
                if step_condition(cb, i.u, i.t, iv)
                    saved || save_state!(i)                  # the state before the affect
                    i.u_modified = true
                    step_affect!(cb, iv)
                    i.u_modified = false
                    save_state!(i)                           # and after it
                    saved = false
                end
                i.terminated && break
            end
            q = clamp(q11 / i.qold^beta2 / gamma, 1 / qmax, 1 / qmin)
            dt = ke == 0 ? min(dt / q, opts.dtmax) : dt   # after an event, the step's own size (SciML's `dtrelax = 1`)
            i.qold = max(EEst, 1e-4)
            i.dtnext = dt
            return true
        else
            i.nreject += 1
            dt = dt / min(1 / qmin, q11 / gamma)
        end
    end
    i.dtnext = dt
    return false
end
