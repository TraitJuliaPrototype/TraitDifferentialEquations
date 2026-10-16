# ODECore: the trait module. The model, callbacks and the integrator as callbacks see it are
# traits; the solvers' code is compiled for what these bounds can see (note "Keys"):
#   - a model that implements `rhs!` itself is the implementing type, so the driver compiles
#     per model type ("route A", the comparison);
#   - a plain function is stored as a `DynamicFn` in the library's `FunctionModel{P}`, so the
#     driver compiles once per parameter type P, whatever the function (the default route);
#   - callbacks are kept as `Dynamic{Callback}` and see the integrator as a
#     `Dynamic{IntegratorView}`, so a new callback compiles nothing in the solver.
@trait_module module ODECore
using TraitWorld
export ODEModel, rhs!, Callback, condition, affect!, IntegratorView, Tsit5, ODESolution, SolverOptions,
       solve_ode, FunctionModel, FunctionCallback, keep_callback, RHSSig, CondSig, AffectSig

# --- the traits --------------------------------------------------------------------------

"What the solver needs from a model: the right-hand side, in place, at element type `T`."
@trait ODEModel{T} begin
    rhs!(m::ODEModel{T}, du::Vector{T}, u::Vector{T}, t::Float64)::Nothing
end

"""
What a callback sees of the integrator: the state (its contents writable), the time, the
parameters, and two flags. Fields only, so there is no implementing type: the key is the
fields' types, `T` and `P` defined by matching them.
"""
@trait IntegratorView{T, P} begin
    @const u::Vector{T}
    @const t::Float64
    @const p::P
    u_modified::Bool
    terminated::Bool
end

"A discrete callback: checked after every accepted step."
@trait Callback begin
    condition(cb::Callback, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView})::Bool
    affect!(cb::Callback, integ::Dynamic{IntegratorView})::Nothing
end

# --- the library's own conforming types ---------------------------------------------------

"The signature of a plain right-hand side `f(du, u, p, t)`."
const RHSSig{P} = Tuple{Vector{Float64},Vector{Float64},P,Float64}
const CondSig = Tuple{Vector{Float64},Float64,Dynamic{IntegratorView}}
const AffectSig = Tuple{Dynamic{IntegratorView}}

"A plain function as a model: a dynamic call at the declared signature, and the parameters."
struct FunctionModel{P}
    f::DynamicFn{RHSSig{P},Nothing}
    p::P
end
rhs!(m::FunctionModel, du::Vector{Float64}, u::Vector{Float64}, t::Float64) = (m.f(du, u, m.p, t); nothing)
@implement ODEModel{Float64} FunctionModel{P} where {P}

"A callback made of two plain functions (SciML's `DiscreteCallback(condition, affect!)`)."
struct FunctionCallback
    cond::DynamicFn{CondSig,Bool}
    aff::DynamicFn{AffectSig,Nothing}
end
condition(cb::FunctionCallback, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView}) = cb.cond(u, t, integ)
affect!(cb::FunctionCallback, integ::Dynamic{IntegratorView}) = (cb.aff(integ); nothing)
@implement Callback FunctionCallback

"Keep a callback of any conforming type beyond the call: what `solve` stores."
keep_callback(cb::Callback) = Dynamic{Callback}(cb)

# --- options, solution, the integrator ---------------------------------------------------------

struct Tsit5 end

struct SolverOptions
    abstol::Float64
    reltol::Float64
    dtmax::Float64
    maxiters::Int
    saveat::Vector{Float64}          # empty: save every step
end

struct ODESolution{T}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    retcode::Symbol
    naccept::Int
    nreject::Int
    nf::Int
end

"Tsit5's state between steps. Its type holds the element type and the parameters' type, nothing else."
mutable struct Tsit5Integrator{T,P}
    const u::Vector{T}
    const uprev::Vector{T}
    t::Float64
    dt::Float64
    const p::P
    u_modified::Bool
    terminated::Bool
    const k1::Vector{T}
    const k2::Vector{T}
    const k3::Vector{T}
    const k4::Vector{T}
    const k5::Vector{T}
    const k6::Vector{T}
    const k7::Vector{T}
    const tmp::Vector{T}
    const unew::Vector{T}
    const utilde::Vector{T}
    qold::Float64
    naccept::Int
    nreject::Int
    nf::Int
end
@implement IntegratorView Tsit5Integrator{T,P} where {T,P}

# --- Tsit5 (tableau from OrdinaryDiffEqTsit5 2.0.0, Tsitouras 2011) -------------------------------

const c1, c2, c3, c4 = 0.161, 0.327, 0.9, 0.9800255409045097
const a21 = 0.161
const a31, a32 = -0.008480655492356989, 0.335480655492357
const a41, a42, a43 = 2.8971530571054935, -6.359448489975075, 4.3622954328695815
const a51, a52, a53, a54 = 5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525
const a61, a62, a63, a64, a65 = 5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383
const a71, a72, a73, a74, a75, a76 = 0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774
const bt1, bt2, bt3, bt4 = -0.00178001105222577714, -0.0008164344596567469, 0.007880878010261995, -0.1447110071732629
const bt5, bt6, bt7 = 0.5823571654525552, -0.45808210592918697, 0.015151515151515152

"One Tsit5 step of size `dt` from `i.u` (with `i.k1 = f(u, t)`): the candidate in `i.unew`, `i.k7 = f(unew, t + dt)`; returns the error estimate."
function tsit5_step!(m, i::Tsit5Integrator{T}, dt::Float64, abstol::Float64, reltol::Float64) where {T}
    u, t = i.u, i.t
    k1, k2, k3, k4, k5, k6, k7, tmp, unew = i.k1, i.k2, i.k3, i.k4, i.k5, i.k6, i.k7, i.tmp, i.unew
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * a21 * k1[j]; end
    rhs!(m, k2, tmp, t + c1 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a31 * k1[j] + a32 * k2[j]); end
    rhs!(m, k3, tmp, t + c2 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j]); end
    rhs!(m, k4, tmp, t + c3 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j]); end
    rhs!(m, k5, tmp, t + c4 * dt)
    @inbounds for j in eachindex(u)
        tmp[j] = u[j] + dt * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j] + a65 * k5[j])
    end
    rhs!(m, k6, tmp, t + dt)
    @inbounds for j in eachindex(u)
        unew[j] = u[j] + dt * (a71 * k1[j] + a72 * k2[j] + a73 * k3[j] + a74 * k4[j] + a75 * k5[j] + a76 * k6[j])
    end
    rhs!(m, k7, unew, t + dt)
    i.nf += 6
    # the embedded error, as OrdinaryDiffEq's calculate_residuals with the RMS norm
    acc = 0.0
    @inbounds for j in eachindex(u)
        e = dt * (bt1 * k1[j] + bt2 * k2[j] + bt3 * k3[j] + bt4 * k4[j] + bt5 * k5[j] + bt6 * k6[j] + bt7 * k7[j])
        sc = abstol + max(abs(u[j]), abs(unew[j])) * reltol
        acc += abs2(e / sc)
    end
    return sqrt(acc / length(u))
end

"Hairer's initial step size (Solving ODEs I, II.4), as OrdinaryDiffEq's `ode_determine_initdt` in its simple form."
function initdt(m, i::Tsit5Integrator{T}, tdir::Float64, opts::SolverOptions) where {T}
    u, f0, t = i.u, i.k1, i.t
    sc(x) = opts.abstol + abs(x) * opts.reltol
    d0 = sqrt(sum(abs2(u[j] / sc(u[j])) for j in eachindex(u)) / length(u))
    d1 = sqrt(sum(abs2(f0[j] / sc(u[j])) for j in eachindex(u)) / length(u))
    h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1
    h0 = min(h0, opts.dtmax)
    tmp = i.tmp
    @inbounds for j in eachindex(u); tmp[j] = u[j] + h0 * f0[j]; end
    rhs!(m, i.k2, tmp, t + h0)
    i.nf += 1
    d2 = sqrt(sum(abs2((i.k2[j] - f0[j]) / sc(u[j])) for j in eachindex(u)) / length(u)) / h0
    h1 = max(d1, d2) <= 1e-15 ? max(1e-6, h0 * 1e-3) : (0.01 / max(d1, d2))^(1 / 5)
    return min(100 * h0, h1, opts.dtmax)
end

"The cubic Hermite interpolant between two accepted states, for `saveat`."
function hermite!(out::Vector{T}, θ::Float64, dt::Float64, u0::Vector{T}, f0::Vector{T}, u1::Vector{T}, f1::Vector{T}) where {T}
    h00 = (1 + 2θ) * (1 - θ)^2
    h10 = θ * (1 - θ)^2
    h01 = θ^2 * (3 - 2θ)
    h11 = θ^2 * (θ - 1)
    @inbounds for j in eachindex(out)
        out[j] = h00 * u0[j] + h10 * dt * f0[j] + h01 * u1[j] + h11 * dt * f1[j]
    end
    return out
end

# --- the driver ---------------------------------------------------------------------------------

"""
    solve_ode(m, u0, tspan, Tsit5(), p, callbacks, opts) -> ODESolution

Integrate the model from `u0` over `tspan` with Tsit5 and a PI step-size controller. `p` is
what callbacks see as `integ.p`; `callbacks` are kept callbacks (`keep_callback`).
"""
function solve_ode(m::ODEModel, u0::Vector{Float64}, tspan::Tuple{Float64,Float64}, alg::Tsit5, p,
                   callbacks::Vector{Dynamic{Callback}}, opts::SolverOptions)
    t0, tf = tspan
    n = length(u0)
    z() = zeros(Float64, n)
    i = Tsit5Integrator{Float64,typeof(p)}(copy(u0), copy(u0), t0, 0.0, p, false, false,
                                            z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), 1e-4, 0, 0, 0)
    rhs!(m, i.k1, i.u, t0)
    i.nf += 1
    dt = initdt(m, i, 1.0, opts)
    iv = Dynamic{IntegratorView}(i)
    ts = Float64[t0]
    us = Vector{Float64}[copy(u0)]
    saveat = opts.saveat
    isave = 1
    if !isempty(saveat) && saveat[1] == t0
        isave = 2
    elseif !isempty(saveat)
        empty!(ts); empty!(us)
    end
    # PI controller, OrdinaryDiffEq's defaults for a 5th-order method
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
        EEst = tsit5_step!(m, i, dt, opts.abstol, opts.reltol)
        q11 = EEst^beta1
        if EEst <= 1.0
            # accept: save, advance, then callbacks
            tnew = i.t + dt
            if isempty(saveat)
                push!(ts, tnew); push!(us, copy(i.unew))
            else
                while isave <= length(saveat) && saveat[isave] <= tnew
                    θ = (saveat[isave] - i.t) / dt
                    push!(ts, saveat[isave])
                    push!(us, hermite!(zeros(Float64, n), θ, dt, i.u, i.k1, i.unew, i.k7))
                    isave += 1
                end
            end
            copyto!(i.uprev, i.u)
            copyto!(i.u, i.unew)
            copyto!(i.k1, i.k7)
            i.t = tnew
            i.naccept += 1
            for cb in callbacks
                if condition(cb, i.u, i.t, iv)
                    i.u_modified = true
                    affect!(cb, iv)
                    if i.u_modified
                        rhs!(m, i.k1, i.u, i.t)          # the state changed: f(u, t) again
                        i.nf += 1
                        i.u_modified = false
                    end
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

end
