# The problems: the trait, TDE's record and its front end, the core's problem, the solution.
# (the body of the trait module `TraitNonlinearSolve`)

"The signature of a residual partially evaluated on the parameters, `f(r, u)`, at element type `T`."
const NLSig{T} = Tuple{Vector{T},Vector{T}}
"The signature of its Jacobian `jac(J, u)`."
const NLJacSig{T} = Tuple{Matrix{T},Vector{T}}
"The residual at the library's duals of chunk `N`."
const NLDualSig{N} = Tuple{Vector{FDual{N}},Vector{FDual{N}}}

"""
    NonlinearProblemTrait{T, P}

What a nonlinear solver needs from a problem: the initial guess `u0`, the parameters `p`, the residual
`residual!(prob, r, u)` (the problem is `r(u) = 0`) and its Jacobian in `u`, `nl_jacobian!(prob, J, u)`. The
default Jacobian is by finite differences of `residual!`; a problem that can do better implements it (TDE's
`NonlinearProblem` does: the given Jacobian, else ForwardDiff's duals). A problem type of the user's implements
it, as `ODEProblemTrait`.
"""
@trait NonlinearProblemTrait{T, P} begin
    @const u0::Vector{T}
    @const p::P
    residual!(prob::NonlinearProblemTrait{T, P}, r::Vector{T}, u::Vector{T})::Nothing
    nl_jacobian!(prob::NonlinearProblemTrait{T, P}, J::Matrix{T}, u::Vector{T})::Nothing = nl_fd_jacobian!(prob, J, u)
end

"The default Jacobian: forward differences of the residual (`prob` a trait object, a problem itself, or the core's problem)."
function nl_fd_jacobian!(prob, J::Matrix{T}, u::Vector{T}) where {T}
    n = length(u)
    f0 = zeros(T, n); f1 = zeros(T, n); up = copy(u)
    _residual!(prob, f0, u)
    for j in 1:n
        h = sqrt(eps(Float64)) * max(abs(ODECore.primal(u[j])), 1.0)
        up[j] = u[j] + h
        _residual!(prob, f1, up)
        for i in 1:n
            J[i, j] = (f1[i] - f0[i]) / h
        end
        up[j] = u[j]
    end
    return nothing
end

"""
    NonlinearFunction(f; jac = nothing, sys, jac_prototype)

SciML's record of a nonlinear problem's functions: the residual `f(r, u, p)` in place, optionally its Jacobian
`jac(J, u, p)` in `u`, and the function's optional properties, the system object and a Jacobian prototype. Each
property defaults to what `f` carries (`HasSystem`, `HasJacobianPrototype`), as `ODEFunction`'s do. A plain
record: no parameter for the state's or the parameters' type; `NonlinearProblem` enters the functions at its
bounded parameters. Callable as `f` is.
"""
struct NonlinearFunction{F,TJ}
    f::F
    jac::TJ
    sys::Any
    jac_prototype::Any
end
Base.@nospecializeinfer NonlinearFunction(@nospecialize(f); @nospecialize(jac = nothing), @nospecialize(sys = ODECore._system(f)),
                                          @nospecialize(jac_prototype = ODECore._jac_prototype(f))) =
    NonlinearFunction(f, jac, sys, jac_prototype)
(f::NonlinearFunction)(r, u, p) = f.f(r, u, p)
@implement HasSystem NonlinearFunction{F,TJ} where {F,TJ} begin
    system(f) = f.sys
end
@implement HasJacobianPrototype NonlinearFunction{F,TJ} where {F,TJ} begin
    jac_prototype(f) = f.jac_prototype
end

"A residual `f(r, u, p)` partially evaluated on the parameters: `(r, u) -> f(r, u, p)` (W-63)."
struct NLPartial{F,P}
    f::F
    p::P
end
(r::NLPartial)(a, u) = r.f(a, u, r.p)
"An ODE right-hand side `f(du, u, p, t)` (or its Jacobian `jac(J, u, p, t)`) at a fixed time, partially evaluated on the parameters: `(a, u) -> f(a, u, p, t)`."
struct AtTime{F,P}
    f::F
    p::P
    t::Float64
end
(r::AtTime)(a, u) = r.f(a, u, r.p, r.t)
"A problem type's own right-hand side at a fixed time (a problem that implements `ODEProblemTrait`)."
struct ProblemAtTime{M}
    m::M
    t::Float64
end
(r::ProblemAtTime)(a, u) = ODECore.rhs!(r.m, a, u, r.t)
"Its Jacobian at that time."
struct ProblemJacAtTime{M}
    m::M
    t::Float64
end
(r::ProblemJacAtTime)(J, u) = ODECore.jacobian!(r.m, J, u, r.t)
"A problem type's own residual and Jacobian, partially evaluated on the problem (for the core)."
struct ProblemResidual{M}
    m::M
end
(r::ProblemResidual)(a, u) = residual!(r.m, a, u)
struct ProblemNLJac{M}
    m::M
end
(r::ProblemNLJac)(J, u) = nl_jacobian!(r.m, J, u)
"No Jacobian given: by the duals for a `Float64` state, by finite differences otherwise (never called)."
nl_no_jacobian(J, u) = nothing
const NLNoJacobian = typeof(nl_no_jacobian)

# a user's function partially evaluated, with its attribute (traits reference §6.4): `compiled_for(f)` stays compiled,
# a kept function gives the user's function itself, so that the one function is seen at each signature (the duals too)
Base.@nospecializeinfer _nlpartial(@nospecialize(f), p) = rewrap(Base.Fix2(NLPartial, p), f)
Base.@nospecializeinfer _attime(@nospecialize(f), p, t::Float64) = rewrap(g -> AtTime(g, p, t), f)

"""
    NonlinearProblem{T, P, F, J}

TDE's own nonlinear problem, the canonical implementation of `NonlinearProblemTrait`: the residual partially
evaluated on the parameters and kept at a parameter bounded by `Callable{NLSig{T}, Nothing}` (traits reference
§7.4), so the solvers compile once per state type whatever the function and the parameters (`Dynamic`, the
default), or per function under `compiled_for(f)`; the initial guess; the parameters; the Jacobian (given, or the
library's `nl_no_jacobian`, kept compiled, so `nl_jacobian!` folds on `J`); `ad`, whether the Jacobian is by the
library's duals; and the function's optional properties, kept as given (the system object, a Jacobian prototype).
Made by `NonlinearProblem(f, u0, p)` and `SteadyStateProblem(f, u0, p; t)`.
"""
struct NonlinearProblem{T,P,F<:Callable{NLSig{T},Nothing},J<:Callable{NLJacSig{T},Nothing}}
    f::F
    u0::Vector{T}
    p::P
    jac::J
    ad::Bool
    sys::Any                     # the system object, or nothing
    jac_prototype::Any           # nothing, or the function's
end
residual!(prob::NonlinearProblem{T}, r::Vector{T}, u::Vector{T}) where {T} = (prob.f(r, u)::Nothing; nothing)
"The Jacobian: the given one; else by ForwardDiff's duals for a `Float64` state; else by finite differences."
nl_jacobian!(prob::NonlinearProblem{T,P,F,J}, Jm::Matrix{T}, u::Vector{T}) where {T,P,F,J} =
    J === NLNoJacobian ? (prob.ad ? _nl_ad_jacobian!(prob.f, Jm, u, Val(DUAL_CHUNK)) : nl_fd_jacobian!(prob, Jm, u)) :
    (prob.jac(Jm, u)::Nothing; nothing)
@implement NonlinearProblemTrait{T,P} NonlinearProblem{T,P,F,J} where {T,P,F,J}
@implement HasSystem NonlinearProblem{T,P,F,J} where {T,P,F,J} begin
    system(x) = getfield(x, :sys)
end
@implement HasJacobianPrototype NonlinearProblem{T,P,F,J} where {T,P,F,J} begin
    jac_prototype(x) = getfield(x, :jac_prototype)
end

"The Jacobian by duals: one call of the residual, seen at the library's duals of chunk `N`, per chunk of `N` columns."
function _nl_ad_jacobian!(fd::Callable{NLDualSig{N},Nothing}, J::Matrix{Float64}, u::Vector{Float64}, ::Val{N})::Nothing where {N}
    n = length(u)
    ud = Vector{FDual{N}}(undef, n)
    dd = Vector{FDual{N}}(undef, n)
    for c0 in 1:N:n
        for j in 1:n
            ud[j] = FDual{N}(u[j], ForwardDiff.Partials(ntuple(k -> j == c0 + k - 1 ? 1.0 : 0.0, Val(N))))
        end
        fd(dd, ud)::Nothing
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

"""
    NonlinearProblem(f, u0, p = nothing)
    NonlinearProblem(NonlinearFunction(f; jac, sys, jac_prototype), u0, p = nothing)

A nonlinear problem in SciML's form: find `u` with `f(r, u, p)` giving `r = 0`, from the guess `u0`. The function
enters at `(Vector{T}, Vector{T}) -> Nothing`, partially evaluated on the parameters, and at the library's duals
for the Jacobian; a Jacobian given in a `NonlinearFunction` at `(Matrix{T}, Vector{T}) -> Nothing`. The function's
optional properties (`HasSystem`, `HasJacobianPrototype`) are read here and kept.
"""
Base.@nospecializeinfer function NonlinearProblem(@nospecialize(f), u0::AbstractVector, p = nothing)
    u = ODECore._state(u0)
    f isa NonlinearFunction && return _nlproblem(_nlpartial(f.f, p), f.jac === nothing ? nothing : _nlpartial(f.jac, p),
                                                 u, p, f.sys, f.jac_prototype)
    return _nlproblem(_nlpartial(f, p), nothing, u, p, ODECore._system(f), ODECore._jac_prototype(f))
end

"""
    SteadyStateProblem(f, u0, p = nothing; t = Inf)
    SteadyStateProblem(ODEFunction(f; jac, sys, jac_prototype), u0, p = nothing; t = Inf)
    SteadyStateProblem(prob::ODEProblemTrait; t = Inf)

The steady state of an ODE right-hand side `f(du, u, p, t)` at the time `t`: the nonlinear problem `f(u, p, t) = 0`,
a `NonlinearProblem` (SteadyStateDiffEq's `SSRootfind`: any `NonlinearAlgorithm` solves it). A Jacobian given in
an `ODEFunction` is taken at `t`; the function's properties as `NonlinearProblem` reads them. From a problem of any
type that implements `ODEProblemTrait`: its `rhs!` and `jacobian!` at `t`, its guess and parameters.
"""
Base.@nospecializeinfer function SteadyStateProblem(@nospecialize(f), u0::AbstractVector, p = nothing; t::Real = Inf)
    u = ODECore._state(u0)
    tt = Float64(t)
    f isa ODEFunction && return _nlproblem(_attime(f.f, p, tt), f.jac === nothing ? nothing : _attime(f.jac, p, tt),
                                           u, p, f.sys, f.jac_prototype)
    return _nlproblem(_attime(f, p, tt), nothing, u, p, ODECore._system(f), ODECore._jac_prototype(f))
end
# the problem's properties read before it enters the bound (a trait object is passed on only where its trait is expected)
Base.@nospecializeinfer SteadyStateProblem(@nospecialize(prob); t::Real = Inf) =
    _steady(prob, Float64(t), ODECore._system(prob), ODECore._jac_prototype(prob))
_steady(prob::ODEProblemTrait{T,P}, t::Float64, @nospecialize(sys), @nospecialize(jp)) where {T,P} =
    NonlinearProblem(ProblemAtTime(prob, t), prob.u0, prob.p, ProblemJacAtTime(prob, t), false, sys, jp)

function _nlproblem(@nospecialize(r), @nospecialize(jr), u0::Vector{T}, p::P, @nospecialize(sys), @nospecialize(jp)) where {T,P}
    ad = T === Float64 && jr === nothing
    jk = jr === nothing ? compiled_for(nl_no_jacobian) : jr
    return NonlinearProblem(r, u0, p, jk, ad, sys, jp)
end

"""
    NLCoreProblem{T, F, J}

The problem as the solvers see it (W-63): the residual and its Jacobian partially evaluated on the parameters,
each kept at a parameter bounded by `Callable` at a signature without them, the guess, and `ad`. Its type names
no parameter type, so the solvers' code is compiled per state type. Made by `nl_core`, the outer layer.
"""
struct NLCoreProblem{T,F<:Callable{NLSig{T},Nothing},J<:Callable{NLJacSig{T},Nothing}}
    f::F
    u0::Vector{T}
    jac::J
    ad::Bool
end
"The residual at `u`, into `r`."
nl_residual!(m::NLCoreProblem{T}, r::Vector{T}, u::Vector{T}) where {T} = (m.f(r, u)::Nothing; nothing)
"The core's Jacobian at `u`: the given one; else by the duals for a `Float64` state; else by finite differences."
nl_core_jacobian!(m::NLCoreProblem{T,F,J}, Jm::Matrix{T}, u::Vector{T}) where {T,F,J} =
    J === NLNoJacobian ? (m.ad ? _nl_ad_jacobian!(m.f, Jm, u, Val(DUAL_CHUNK)) : nl_fd_jacobian!(m, Jm, u)) :
    (m.jac(Jm, u)::Nothing; nothing)
_residual!(prob, r, u) = residual!(prob, r, u)
_residual!(m::NLCoreProblem, r, u) = nl_residual!(m, r, u)
"""
    nl_core(prob) -> NLCoreProblem

The outer layer's step: TDE's `NonlinearProblem` keeps its functions partially evaluated already; they pass as
kept. A problem type that implements `NonlinearProblemTrait` itself: its `residual!` and `nl_jacobian!`, `Dynamic`.
"""
function nl_core(prob::NonlinearProblem{T,P,F,J}) where {T,P,F,J}
    F <: Dynamic || _member_checked(prob)     # a compiled function: the type checked as a member of the family
    return NLCoreProblem(prob.f, prob.u0, prob.jac, prob.ad)
end
_member_checked(prob::NonlinearProblemTrait) = nothing
nl_core(prob::NonlinearProblemTrait) = NLCoreProblem(ProblemResidual(prob), prob.u0, ProblemNLJac(prob), false)

"A solve's counts (SciML's `sol.stats`): iterations, residual and Jacobian evaluations, factorizations."
struct NLStats
    nsteps::Int
    nf::Int
    njacs::Int
    nfactors::Int
end
Base.:+(a::NLStats, b::NLStats) = NLStats(a.nsteps + b.nsteps, a.nf + b.nf, a.njacs + b.njacs, a.nfactors + b.nfactors)

"""
What a solver returns: the solution `u`, the residual there, the return code (`:Success`; `:MaxIters`; `:Stalled`,
no progress possible with the residual above `abstol`; `:ConvergenceFailure`, a singular Jacobian), the counts. No
problem: the core never sees the parameters.
"""
struct NLCoreSolution{T}
    u::Vector{T}
    resid::Vector{T}
    retcode::Symbol
    stats::NLStats
end

"""
    NonlinearSolution{T, P}

A nonlinear problem's solution as the user sees it: `u`, `resid`, `retcode`, `stats`, the problem kept with its
attribute (`sol.prob`, a `Kept{NonlinearProblemTrait{T, P}}`: no axis for its function or type), and the system
object its function carried (`HasSystem`, `getsystem(sol)`).
"""
struct NonlinearSolution{T,P}
    u::Vector{T}
    resid::Vector{T}
    retcode::Symbol
    stats::NLStats
    prob::Kept{NonlinearProblemTrait{T,P}}
    sys::Any
end
@implement HasSystem NonlinearSolution{T,P} where {T,P} begin
    system(s) = getfield(s, :sys)
end
nl_with_problem(s::NLCoreSolution{T}, prob::NonlinearProblemTrait{T,P}, @nospecialize(sys)) where {T,P} =
    NonlinearSolution{T,P}(s.u, s.resid, s.retcode, s.stats, prob, sys)

"""
    successful_retcode(retcode::Symbol)
    successful_retcode(sol)

Whether a return code (or a solution's) is a success: `:Success` (SciML's `successful_retcode`).
"""
successful_retcode(r::Symbol) = r === :Success
successful_retcode(s::NonlinearSolution) = successful_retcode(s.retcode)
