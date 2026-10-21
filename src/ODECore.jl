# ODECore: the trait module. A problem, callbacks and the integrator as callbacks see it are
# traits; the solvers' code is compiled for what these bounds can see (note "Keys"):
#   - a problem type that implements `rhs!` itself is the receiver type, so the solver compiles per
#     problem type;
#   - TDE's own `ODEProblem` keeps its function at parameters bounded by `Callable`: a plain function
#     as a `Dynamic{Callable{…}}` (`Callable`'s default), so the solver compiles once per element and
#     parameter type, whatever the function; under `compiled_for(f)` the function itself, and the
#     solver compiles per function;
#   - callbacks are kept with their attribute (`Vector{Callback}`) and see the integrator as a
#     `Dynamic{IntegratorView}`, so a new callback compiles nothing in the solver unless the caller
#     casts it with `compiled_for`.
@trait_module module ODECore
using TraitWorld
using LinearAlgebra: lu!, ldiv!
import ForwardDiff
export ODEProblemTrait, rhs!, jacobian!, Callback, condition, affect!, IntegratorView, Tsit5, Rodas5, ODESolution, SolverOptions,
       solve_ode, ODEProblem, FunctionCallback, keep_callback, RHSSig, DualSig, DUAL_CHUNK, CondSig, AffectSig, FDual, ODETag

# --- the traits --------------------------------------------------------------------------

"""
What the solver needs from a problem: its initial state, time span and parameters, the right-hand
side in place at element type `T`, and its Jacobian in `u` (for implicit methods). The default
Jacobian is by finite differences of `rhs!`; a problem that can do better implements it (TDE's
`ODEProblem` does, with ForwardDiff's duals). `p` is what callbacks see as `integ.p`.
"""
@trait ODEProblemTrait{T, P} begin
    @const u0::Vector{T}
    @const tspan::Tuple{Float64, Float64}
    @const p::P
    rhs!(prob::ODEProblemTrait{T, P}, du::Vector{T}, u::Vector{T}, t::Float64)::Nothing
    jacobian!(prob::ODEProblemTrait{T, P}, J::Matrix{T}, u::Vector{T}, t::Float64)::Nothing = fd_jacobian!(prob, J, u, t)
end

"The default Jacobian: forward differences of `rhs!` (an internal function: `prob` is a trait object, or a problem itself)."
function fd_jacobian!(prob, J::Matrix{T}, u::Vector{T}, t::Float64) where {T}
    n = length(u)
    f0 = zeros(T, n); f1 = zeros(T, n); up = copy(u)
    rhs!(prob, f0, u, t)
    for j in 1:n
        h = sqrt(eps(Float64)) * max(abs(primal(u[j])), 1.0)
        up[j] = u[j] + h
        rhs!(prob, f1, up, t)
        for i in 1:n
            J[i, j] = (f1[i] - f0[i]) / h
        end
        up[j] = u[j]
    end
    return nothing
end

"""
What a callback sees of the integrator: the state (its contents writable), the time, the
parameters, and two flags. Fields only, so there is no receiver type: the key is the
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

# --- the signatures of plain functions --------------------------------------------------------

"The signature of a plain right-hand side `f(du, u, p, t)`, at element type `T` (`Float64`, or dual numbers for gradients through `solve`)."
const RHSSig{P,T} = Tuple{Vector{T},Vector{T},P,Float64}
const CondSig = Tuple{Vector{Float64},Float64,Dynamic{IntegratorView}}
const AffectSig = Tuple{Dynamic{IntegratorView}}

# --- ForwardDiff's dual numbers, tagged by the library ----------------------------------------
# (wrapping by facts: the trait module calls ForwardDiff directly; session 23 used its own duals)

"The library's tag for ForwardDiff's duals: they are the library's, whatever the model function."
struct ODETag end
"ForwardDiff's dual number with the library's tag and chunk size `N`."
const FDual{N} = ForwardDiff.Dual{ODETag,Float64,N}

"The signature of a plain right-hand side at the library's duals."
const DualSig{P,N} = Tuple{Vector{FDual{N}},Vector{FDual{N}},P,Float64}

"The chunk of the Jacobian's dual numbers: fixed, so that it is no part of a problem's type (a last chunk may be partial)."
const DUAL_CHUNK = 8

"""
    ODEProblem{T, P, R, D}

TDE's own problem, the canonical implementation of `ODEProblemTrait`: a plain right-hand side
`f(du, u, p, t)`, an initial state, a time span and parameters. The function is kept at two
parameters bounded by `Callable` (traits reference §7.4), each the function as its bound gives it:
the function itself under `compiled_for(f)`, so the solver compiles per function; a
`Dynamic{Callable{…}}` otherwise (`Callable`'s default, and `with_dynamic`), so it compiles once per
`(T, P)`:
- `f::F` at `RHSSig{P, T}`;
- `fd::D` at the library's dual numbers `DualSig{P, DUAL_CHUNK}`, for the Jacobian of a `Float64`
  state (a `Dynamic` one's pointer is made at its first call).
Made by `ODEProblem(f, u0, tspan, p, f)`; the front end `ODEProblem(f, u0, tspan, p)` does that.
"""
struct ODEProblem{T,P,F<:Callable{RHSSig{P,T},Nothing},D<:Callable{DualSig{P,DUAL_CHUNK},Nothing}}
    f::F
    u0::Vector{T}
    tspan::Tuple{Float64,Float64}
    p::P
    fd::D
end
rhs!(prob::ODEProblem{T,P}, du::Vector{T}, u::Vector{T}, t::Float64) where {T,P} = (prob.f(du, u, prob.p, t)::Nothing; nothing)
"The Jacobian: by ForwardDiff's duals for a `Float64` state, by finite differences otherwise."
jacobian!(prob::ODEProblem{T}, J::Matrix{T}, u::Vector{T}, t::Float64) where {T} =
    T === Float64 ? ad_jacobian!(prob, J, u, t) : fd_jacobian!(prob, J, u, t)
"The Jacobian by duals: one call of the function at the duals per chunk of `DUAL_CHUNK` columns."
function ad_jacobian!(prob::ODEProblem{Float64}, J::Matrix{Float64}, u::Vector{Float64}, t::Float64)
    N = DUAL_CHUNK
    n = length(u)
    ud = Vector{FDual{N}}(undef, n)
    dd = Vector{FDual{N}}(undef, n)
    for c0 in 1:N:n
        for j in 1:n
            ud[j] = FDual{N}(u[j], ForwardDiff.Partials(ntuple(k -> j == c0 + k - 1 ? 1.0 : 0.0, Val(N))))
        end
        prob.fd(dd, ud, prob.p, t)::Nothing
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
@implement ODEProblemTrait{T,P} ODEProblem{T,P,F,D} where {T,P,F,D}

"A callback made of two plain functions (SciML's `DiscreteCallback(condition, affect!)`)."
struct FunctionCallback
    cond::Callable{CondSig,Bool}            # a function kept with its attribute (Kept{Callable{…}})
    aff::Callable{AffectSig,Nothing}
end
condition(cb::FunctionCallback, u::Vector{Float64}, t::Float64, integ::Dynamic{IntegratorView}) = cb.cond(u, t, integ)
affect!(cb::FunctionCallback, integ::Dynamic{IntegratorView}) = (cb.aff(integ); nothing)
@implement Callback FunctionCallback

"Keep a callback of any conforming type beyond the call, with its attribute: what `solve` stores."
keep_callback(cb::Callback) = Kept{Callback}(cb)

# --- options, solution, the integrator ---------------------------------------------------------

struct Tsit5 end
struct Rodas5 end

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
    rhs!(m, i.k2, tmp, t + h0)
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
    solve_ode(prob, Tsit5(), callbacks, opts) -> ODESolution

Integrate the problem from `prob.u0` over `prob.tspan` with Tsit5 and a PI step-size controller.
`prob.p` is what callbacks see as `integ.p`; `callbacks` are kept callbacks (`keep_callback`). The
element type `T` is the state's: `Float64`, `Float32`, or a dual number for a gradient through the
solve; the step-size control reads primal values, so a gradient follows the primal solve's steps.
Callbacks take `Float64` states.
"""
solve_ode(prob::ODEProblemTrait, alg::Tsit5, callbacks::Vector{Callback}, opts::SolverOptions) =
    tsit5_solve(prob, prob.u0, prob.tspan, prob.p, callbacks, opts)

"Tsit5 for the problem `m` (a trait object), at its state's element type `T`."
function tsit5_solve(m, u0::Vector{T}, tspan::Tuple{Float64,Float64}, p,
                     callbacks::Vector{Callback}, opts::SolverOptions) where {T}
    t0, tf = tspan
    n = length(u0)
    T === Float64 || isempty(callbacks) || throw(ArgumentError("callbacks take Float64 states; this solve has $T"))
    z() = zeros(T, n)
    i = Tsit5Integrator{T,typeof(p)}(copy(u0), copy(u0), t0, 0.0, p, false, false,
                                            z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), 1e-4, 0, 0, 0)
    rhs!(m, i.k1, i.u, t0)
    i.nf += 1
    dt = initdt(m, i, 1.0, opts)
    iv = Dynamic{IntegratorView}(i)
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
    return ODESolution{T}(ts, us, retcode, i.naccept, i.nreject, i.nf)
end


"The value a step-size controller sees: a dual number's primal value, so that gradients follow the primal's steps."
primal(x::Float64) = x
primal(x::AbstractFloat) = Float64(x)
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))


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
@implement IntegratorView Rodas5Integrator{P} where {P}

"One Rodas5 step of size `dt` from `i.u`: the candidate in `i.unew`, the interpolant's k's in `i.kint`; returns the error estimate."
function rodas5_step!(m, i::Rodas5Integrator, dt::Float64, abstol::Float64, reltol::Float64)
    u, t, n = i.u, i.t, length(i.u)
    ks, du, du1, dT, ltmp, J, W, unew = i.ks, i.du, i.du1, i.dT, i.ltmp, i.J, i.W, i.unew
    rhs!(m, i.fsal, u, t)
    jacobian!(m, J, u, t)
    i.njac += 1
    h = sqrt(eps(Float64)) * max(abs(t), 1.0)          # the time derivative, by a forward difference
    rhs!(m, du, u, t + h)
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
            rhs!(m, du, unew, t + R5c[stage] * dt)
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
    solve_ode(prob, Rodas5(), callbacks, opts) -> ODESolution

Rodas5 with the problem's Jacobian (`jacobian!`: by the library's duals for TDE's `ODEProblem`, by
finite differences or the problem type's own otherwise), dense LU, and a PI step-size controller.
`Float64` states only.
"""
solve_ode(prob::ODEProblemTrait, alg::Rodas5, callbacks::Vector{Callback}, opts::SolverOptions) =
    rodas5_solve(prob, prob.u0, prob.tspan, prob.p, callbacks, opts)
rodas5_solve(m, u0::Vector{T}, tspan, p, callbacks, opts) where {T} =
    throw(ArgumentError("Rodas5 takes Float64 states; this problem's state has $T"))

"Rodas5 for the problem `m` (a trait object)."
function rodas5_solve(m, u0::Vector{Float64}, tspan::Tuple{Float64,Float64}, p,
                      callbacks::Vector{Callback}, opts::SolverOptions)
    t0, tf = tspan
    n = length(u0)
    z() = zeros(Float64, n)
    i = Rodas5Integrator{typeof(p)}(copy(u0), copy(u0), t0, 0.0, p, false, false, z(),
                                    [z() for _ in 1:8], [z() for _ in 1:3], z(), z(), z(), z(), z(),
                                    zeros(Float64, n, n), zeros(Float64, n, n), 1e-4, 0, 0, 0, 0)
    iv = Dynamic{IntegratorView}(i)
    # the initial step: as Tsit5's driver, from f(u0)
    rhs!(m, i.fsal, i.u, t0)
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
                if condition(cb, i.u, i.t, iv)
                    i.u_modified = true
                    affect!(cb, iv)
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

end
