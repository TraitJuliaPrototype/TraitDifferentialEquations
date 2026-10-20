"""
    PlainODE

TraitDifferentialEquations' Tsit5 without traits (its README's baseline (b)): the same tableau,
step, controller and driver, written with the discipline a careful Julia programmer uses. The model
function is a `FunctionWrapper` at `(Vector{T}, Vector{T}, P, Float64)`, so the solver compiles once
per `(T, P)`, not per function; callbacks are kept in a `Vector{Any}` and called dynamically, with
asserted results; gradients use the same tag labelled by the loss's signature and wrap the loss.

The other routes of TraitDifferentialEquations, written the same way (session 30):
- a model type that adds a method to `PlainODE.rhs!` (and optionally `jacobian!`): the solver is
  specialized on the model's type, as it is compiled for the implementing type there;
- `compiled_for(f)`: the function passed unwrapped, so the solver specializes on it (the plain
  equivalent of `Compiled`; nothing checks the signature but the call itself);
- callback types that add methods to `PlainODE.condition` and `PlainODE.affect!`;
- Rodas5, with the Jacobian by ForwardDiff at a library-owned tag (two FunctionWrappers) for a
  plain function, and the model's `jacobian!` (finite differences by default) for a model type.
"""
module PlainODE
using FunctionWrappers: FunctionWrapper
import ForwardDiff
using PrecompileTools: @compile_workload
using LinearAlgebra: lu!, ldiv!
export ODEProblem, solve, Tsit5, Rodas5, DiscreteCallback, gradient, compiled_for, rhs!, jacobian!, condition, affect!

struct Tsit5 end
struct Rodas5 end
struct SolverOptions
    abstol::Float64
    reltol::Float64
    dtmax::Float64
    maxiters::Int
    saveat::Vector{Float64}
end
struct ODESolution{T}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    retcode::Symbol
    naccept::Int
    nreject::Int
    nf::Int
end
"A discrete callback: two plain functions, kept untyped."
struct DiscreteCallback
    condition::Any
    affect!::Any
end
"""
    condition(cb, u, t, integ)::Bool, affect!(cb, integ)

What the driver calls for a kept callback; a callback type adds methods (the plain form of
conforming to `Callback`). Called dynamically, results asserted.
"""
function condition end
function affect! end
condition(cb::DiscreteCallback, u, t, integ) = cb.condition(u, t, integ)
affect!(cb::DiscreteCallback, integ) = (cb.affect!(integ); nothing)

"""
    rhs!(model, du, u, t), jacobian!(model, J, u, t)

A model type's right-hand side and Jacobian; a model type adds methods (the plain form of
conforming to `ODEModel`). The Jacobian defaults to forward differences of `rhs!`. The default
takes untyped arguments: typed like the trait's declaration, it would be ambiguous with a model's
method generic in its element type (`jacobian!(m::Lorenz{T}, J::Matrix{T}, …)`).
"""
function rhs! end
jacobian!(m, J, u, t) = fd_jacobian!((du, u, p, t) -> rhs!(m, du, u, t), J, u, nothing, t)

"A model type as the solver's function `(du, u, p, t)`: the solver specializes on `M`."
struct ModelRHS{M}
    m::M
end
(r::ModelRHS)(du, u, p, t) = (rhs!(r.m, du, u, t); nothing)

"A function the caller asks to be compiled into the solver (the plain form of `compiled_for`)."
struct Compiled{F}
    f::F
end
compiled_for(f) = Compiled(f)
primal(x::Float64) = x
primal(x::AbstractFloat) = Float64(x)
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))

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
    m(k2, tmp, i.p, t + c1 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a31 * k1[j] + a32 * k2[j]); end
    m(k3, tmp, i.p, t + c2 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j]); end
    m(k4, tmp, i.p, t + c3 * dt)
    @inbounds for j in eachindex(u); tmp[j] = u[j] + dt * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j]); end
    m(k5, tmp, i.p, t + c4 * dt)
    @inbounds for j in eachindex(u)
        tmp[j] = u[j] + dt * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j] + a65 * k5[j])
    end
    m(k6, tmp, i.p, t + dt)
    @inbounds for j in eachindex(u)
        unew[j] = u[j] + dt * (a71 * k1[j] + a72 * k2[j] + a73 * k3[j] + a74 * k4[j] + a75 * k5[j] + a76 * k6[j])
    end
    m(k7, unew, i.p, t + dt)
    i.nf += 6
    # the embedded error, as OrdinaryDiffEq's calculate_residuals with the RMS norm
    acc = 0.0
    @inbounds for j in eachindex(u)
        e = dt * (bt1 * k1[j] + bt2 * k2[j] + bt3 * k3[j] + bt4 * k4[j] + bt5 * k5[j] + bt6 * k6[j] + bt7 * k7[j])
        sc = abstol + max(abs(primal(u[j])), abs(primal(unew[j]))) * reltol
        acc += abs2(primal(e) / sc)
    end
    return sqrt(acc / length(u))
end

"Hairer's initial step size (Solving ODEs I, II.4), as OrdinaryDiffEq's `ode_determine_initdt` in its simple form."
function initdt(m, i::Tsit5Integrator{T}, tdir::Float64, opts::SolverOptions) where {T}
    u, f0, t = i.u, i.k1, i.t
    sc(x) = opts.abstol + abs(primal(x)) * opts.reltol
    d0 = sqrt(sum(abs2(primal(u[j]) / sc(u[j])) for j in eachindex(u)) / length(u))
    d1 = sqrt(sum(abs2(primal(f0[j]) / sc(u[j])) for j in eachindex(u)) / length(u))
    h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1
    h0 = min(h0, opts.dtmax)
    tmp = i.tmp
    @inbounds for j in eachindex(u); tmp[j] = u[j] + h0 * f0[j]; end
    m(i.k2, tmp, i.p, t + h0)
    i.nf += 1
    d2 = sqrt(sum(abs2(primal(i.k2[j] - f0[j]) / sc(u[j])) for j in eachindex(u)) / length(u)) / h0
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

`m` is called as `m(du, u, p, t)`: a FunctionWrapper (compiled once per `(T, P)`), a model
type's `ModelRHS` or a function itself (compiled per type). Integrate the model from `u0` over `tspan` with Tsit5 and a PI step-size controller. `p` is
what callbacks see as `integ.p`; `callbacks` are kept callbacks (`keep_callback`). The element
type `T` is `Float64`, or a dual number for a gradient through the solve; the step-size control
reads primal values, so a gradient follows the primal solve's steps. Callbacks take `Float64` states.
"""
function solve_ode(m, u0::Vector{T}, tspan::Tuple{Float64,Float64}, alg::Tsit5, p::P,
                   callbacks::Vector{Any}, opts::SolverOptions) where {T,P}
    t0, tf = tspan
    n = length(u0)
    T === Float64 || isempty(callbacks) || throw(ArgumentError("callbacks take Float64 states; this solve has $T"))
    z() = zeros(T, n)
    i = Tsit5Integrator{T,typeof(p)}(copy(u0), copy(u0), t0, 0.0, p, false, false,
                                            z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), 1e-4, 0, 0, 0)
    m(i.k1, i.u, i.p, t0)
    i.nf += 1
    dt = initdt(m, i, 1.0, opts)
    ts = Float64[t0]
    us = Vector{T}[copy(u0)]
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
                    push!(us, hermite!(zeros(T, n), θ, dt, i.u, i.k1, i.unew, i.k7))
                    isave += 1
                end
            end
            copyto!(i.uprev, i.u)
            copyto!(i.u, i.unew)
            copyto!(i.k1, i.k7)
            i.t = tnew
            i.naccept += 1
            T === Float64 && for cb in callbacks
                if condition(cb, i.u, i.t, i)::Bool
                    i.u_modified = true
                    affect!(cb, i)
                    if i.u_modified
                        m(i.k1, i.u, i.p, i.t)          # the state changed: f(u, t) again
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
    return ODESolution{T}(ts, us, retcode, i.naccept, i.nreject, i.nf)
end

"Forward differences of `f(du, u, p, t)` in `u`: the default Jacobian."
function fd_jacobian!(f, J::Matrix{Float64}, u::Vector{Float64}, p, t::Float64)
    n = length(u)
    f0 = zeros(Float64, n); f1 = zeros(Float64, n); up = copy(u)
    f(f0, u, p, t)
    for j in 1:n
        h = sqrt(eps(Float64)) * max(abs(u[j]), 1.0)
        up[j] = u[j] + h
        f(f1, up, p, t)
        for i in 1:n
            J[i, j] = (f1[i] - f0[i]) / h
        end
        up[j] = u[j]
    end
    return nothing
end

# --- ForwardDiff's dual numbers, tagged by the library ----------------------------------------

"The library's tag for ForwardDiff's duals."
struct ODETag end
const FDual{N} = ForwardDiff.Dual{ODETag,Float64,N}

"""
A plain function for Rodas5: FunctionWrappers at the plain and at the dual signature (chunk `N`),
so the solver specializes per `(P, N)`, never per function.
"""
struct WrappedAD{P,N}
    f::FunctionWrapper{Nothing,Tuple{Vector{Float64},Vector{Float64},P,Float64}}
    fd::FunctionWrapper{Nothing,Tuple{Vector{FDual{N}},Vector{FDual{N}},P,Float64}}
end
(w::WrappedAD)(du, u, p, t) = w.f(du, u, p, t)
"The Jacobian Rodas5 asks for: by duals for a wrapped function, the model's `jacobian!` for a model type."
function pjacobian!(m::WrappedAD{P,N}, J::Matrix{Float64}, u::Vector{Float64}, p::P, t::Float64) where {P,N}
    n = length(u)
    ud = Vector{FDual{N}}(undef, n)
    dd = Vector{FDual{N}}(undef, n)
    for c0 in 1:N:n
        for j in 1:n
            ud[j] = FDual{N}(u[j], ForwardDiff.Partials(ntuple(k -> j == c0 + k - 1 ? 1.0 : 0.0, Val(N))))
        end
        m.fd(dd, ud, p, t)
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
pjacobian!(r::ModelRHS, J::Matrix{Float64}, u::Vector{Float64}, p, t::Float64) = jacobian!(r.m, J, u, t)
pjacobian!(f, J::Matrix{Float64}, u::Vector{Float64}, p, t::Float64) = fd_jacobian!(f, J, u, p, t)

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

"Rodas5's state between steps: the integrator's view for callbacks, the stages, the Jacobian and W."
mutable struct Rodas5Integrator{P}
    const u::Vector{Float64}
    const uprev::Vector{Float64}
    t::Float64
    dt::Float64
    const p::P
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
end

"One Rodas5 step of size `dt` from `i.u`: the candidate in `i.unew`, the interpolant's k's in `i.kint`; returns the error estimate."
function rodas5_step!(m, i::Rodas5Integrator, dt::Float64, abstol::Float64, reltol::Float64)
    u, t, n = i.u, i.t, length(i.u)
    ks, du, du1, dT, ltmp, J, W, unew = i.ks, i.du, i.du1, i.dT, i.ltmp, i.J, i.W, i.unew
    m(i.fsal, u, i.p, t)
    pjacobian!(m, J, u, i.p, t)
    i.njac += 1
    h = sqrt(eps(Float64)) * max(abs(t), 1.0)          # the time derivative, by a forward difference
    m(du, u, i.p, t + h)
    i.nf += 2
    @inbounds for j in 1:n; dT[j] = (du[j] - i.fsal[j]) / h; end
    dtgamma = dt * R5gamma
    copyto!(W, J)                                      # W = J - I / (dt γ), as OrdinaryDiffEq's W_transform
    @inbounds for k in 1:n; W[k, k] -= 1 / dtgamma; end
    F = lu!(W)
    @inbounds for j in 1:n; ltmp[j] = i.fsal[j] + dt * R5d[1] * dT[j]; end
    ldiv!(ks[1], F, ltmp)
    ks[1] .*= -1
    for stage in 2:8
        @inbounds for j in 1:n
            acc = u[j]
            for l in 1:stage-1; acc += R5A[stage, l] * ks[l][j]; end
            unew[j] = acc
        end
        if !(stage > 2 && R5c[stage] == R5c[stage-1] && all(l -> R5A[stage, l] == R5A[stage-1, l], 1:stage-1))
            m(du, unew, i.p, t + R5c[stage] * dt)
            i.nf += 1
        end
        @inbounds for j in 1:n
            acc = 0.0
            for l in 1:stage-1; acc += R5C[stage, l] / dt * ks[l][j]; end
            ltmp[j] = du[j] + dt * R5d[stage] * dT[j] + acc
        end
        ldiv!(ks[stage], F, ltmp)
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

"""
    solve_ode(m, u0, tspan, Rodas5(), p, callbacks, opts) -> ODESolution

Rodas5 with the Jacobian `pjacobian!(m, J, u, p, t)`: by the library's duals for a plain function,
the model's `jacobian!` for a model type; dense LU, and a PI step-size controller.
"""
function solve_ode(m, u0::Vector{Float64}, tspan::Tuple{Float64,Float64}, alg::Rodas5, p,
                   callbacks::Vector{Any}, opts::SolverOptions)
    t0, tf = tspan
    n = length(u0)
    z() = zeros(Float64, n)
    i = Rodas5Integrator{typeof(p)}(copy(u0), copy(u0), t0, 0.0, p, false, false, z(),
                                    [z() for _ in 1:8], [z() for _ in 1:3], z(), z(), z(), z(), z(),
                                    zeros(Float64, n, n), zeros(Float64, n, n), 1e-4, 0, 0, 0, 0)
    # the initial step: as Tsit5's driver, from f(u0)
    m(i.fsal, i.u, i.p, t0)
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
        EEst = rodas5_step!(m, i, dt, opts.abstol, opts.reltol)
        q11 = EEst^beta1
        if EEst <= 1.0
            tnew = i.t + dt
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
            i.t = tnew
            i.naccept += 1
            for cb in callbacks
                if condition(cb, i.u, i.t, i)::Bool
                    i.u_modified = true
                    affect!(cb, i)
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

# --- the problem, solve, gradient -------------------------------------------------------------

struct ODEProblem{F,U,P}
    f::F
    u0::Vector{U}
    tspan::Tuple{Float64,Float64}
    p::P
end
_state(u0::AbstractVector{<:ForwardDiff.Dual}) = Vector(u0)
_state(u0::AbstractVector{T}) where {T<:AbstractFloat} = Vector{T}(u0)
_state(u0::AbstractVector) = Vector{Float64}(u0)
ODEProblem(f, u0::AbstractVector, tspan, p = nothing) = ODEProblem(f, _state(u0), (Float64(tspan[1]), Float64(tspan[2])), p)
_promote_state(u0::Vector{Float64}, p::AbstractArray{D}) where {D<:ForwardDiff.Dual} = Vector{D}(u0)
_promote_state(u0::Vector, p) = u0
_kept(::Nothing) = Any[]
_kept(cbs::Union{Tuple,AbstractVector}) = Any[cb for cb in cbs]
_kept(cb) = Any[cb]
_options(; abstol = 1e-6, reltol = 1e-3, dtmax = Inf, maxiters = 100_000, saveat = Float64[]) =
    SolverOptions(Float64(abstol), Float64(reltol), Float64(dtmax), maxiters, Vector{Float64}(collect(saveat)))
function solve(@nospecialize(prob::ODEProblem), alg::Tsit5; callback = nothing, kw...)
    u0 = _promote_state(prob.u0, prob.p)
    return _solve(prob.f, u0, prob.tspan, alg, prob.p, _kept(callback), _options(; kw...))
end
function _solve(@nospecialize(f), u0::Vector{T}, tspan, alg, p::P, cbs, opts) where {T,P}
    w = FunctionWrapper{Nothing,Tuple{Vector{T},Vector{T},P,Float64}}(f isa Compiled ? f.f : f)
    return solve_ode(w, u0, tspan, alg, p, cbs, opts)
end
# compiled mode, where TraitDifferentialEquations serves it (Float64 states, Vector{Float64} or no
# parameters): the function itself, so the solver specializes on its type
_solve(f::Compiled, u0::Vector{Float64}, tspan, alg::Tsit5, p::Union{Nothing,Vector{Float64}}, cbs, opts) =
    solve_ode(f.f, u0, tspan, alg, p, cbs, opts)
function solve(@nospecialize(prob::ODEProblem), alg::Rodas5; callback = nothing, kw...)
    N = min(length(prob.u0), 8)
    return _solve_ad(prob.f isa Compiled ? prob.f.f : prob.f, prob.u0, prob.tspan, alg, prob.p, _kept(callback), _options(; kw...), Val(N))
end
_solve_ad(@nospecialize(f), u0, tspan, alg, p::P, cbs, opts, ::Val{N}) where {P,N} =
    solve_ode(WrappedAD{P,N}(f, f), u0, tspan, alg, p, cbs, opts)
"""
    solve(model, u0, tspan, alg; p = model, callback, kw...)

A model type with a `rhs!` method: the solver specializes on it. The state keeps `u0`'s element type.
"""
solve(model, u0::AbstractVector, tspan, alg::Union{Tsit5,Rodas5}; p = model, callback = nothing, kw...) =
    solve_ode(ModelRHS(model), _state(u0), (Float64(tspan[1]), Float64(tspan[2])), alg, p, _kept(callback), _options(; kw...))

struct Sig{A,S} end
function gradient(@nospecialize(loss), p::Vector{Float64})
    N = ForwardDiff.pickchunksize(length(p))
    return _gradient(loss, p, Val(N))
end
function _gradient(@nospecialize(loss), p::Vector{Float64}, ::Val{N}) where {N}
    Tg = ForwardDiff.Tag{Sig{Tuple{Vector{Float64}},Float64},Float64}
    D = ForwardDiff.Dual{Tg,Float64,N}
    f = FunctionWrapper{D,Tuple{Vector{D}}}(loss)
    return ForwardDiff.gradient(f, p, ForwardDiff.GradientConfig(f, p, ForwardDiff.Chunk{N}(), Tg()), Val(false))
end

_wl_rhs!(du, u, p, t) = (du[1] = p[1] * (u[2] - u[1]); du[2] = u[1] * (p[2] - u[3]) - u[2]; du[3] = u[1] * u[2] - p[3] * u[3]; nothing)
_wl_rhs0!(du, u, p, t) = (du[1] = 10.0 * (u[2] - u[1]); du[2] = u[1] * (28.0 - u[3]) - u[2]; du[3] = u[1] * u[2] - 8 / 3 * u[3]; nothing)
@compile_workload begin
    prob = ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1), [10.0, 28.0, 8 / 3])
    solve(prob, Tsit5())
    solve(prob, Tsit5(); saveat = [0.05, 0.1])
    solve(prob, Tsit5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    solve(prob, Rodas5())
    solve(ODEProblem(_wl_rhs0!, [1.0, 0.0, 0.0], (0.0, 0.1)), Rodas5())
    solve(prob, Rodas5(); callback = DiscreteCallback((u, t, i) -> t > 0.05, i -> (i.u[1] += 1.0; nothing)))
    gradient(p -> sum(solve(ODEProblem(_wl_rhs!, [1.0, 0.0, 0.0], (0.0, 0.1), p), Tsit5()).u[end]), [10.0, 28.0, 8 / 3])
end

end
