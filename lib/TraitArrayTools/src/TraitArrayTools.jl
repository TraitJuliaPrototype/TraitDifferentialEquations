"""
    TraitArrayTools

A trajectory as code over solutions sees it (SciML's move 5): a stand-in for RecursiveArrayTools, the
package below SciMLBase that holds `DiffEqArray` and what plotting reads from a solution. `Trajectory{U, T}`
is the trait; `DiffEqArray` conforms here, a solver's solution where its type is defined. Code over
trajectories (`plotpoints`) is written once against the trait, so it is compiled per state and time type,
whatever produced the trajectory. A plotting package needs this package, not the solvers.
"""
module TraitArrayTools
using TraitWorld
export Trajectory, times, state, at, DiffEqArray, plotpoints, linear_at
@trait_module begin
"""
    Trajectory{U, T}

The saved times, the state at the `k`-th of them, and the state at any time in the span (`at`). The key is
the state type `U` and the time type `T`, never the model or the solver.
"""
@trait Trajectory{U, T} begin
    times(s::Trajectory{U, T})::Vector{T}
    state(s::Trajectory{U, T}, k::Int)::U
    at(s::Trajectory{U, T}, τ::T)::U
end

"""
    linear_at(ts, us, τ)

The state at `τ`, linear between the saved states `us` at the times `ts`; `DomainError` outside them. At a
saved time, the first state saved there (at an event's time, the state before the event: SciML's left
continuity). At the state's element type: a `Float32` state stays `Float32`. For trajectories that keep no
interpolant.
"""
function linear_at(ts::Vector{Float64}, us::Vector{Vector{E}}, τ::Float64) where {E}
    (isempty(ts) || τ < ts[1] || τ > ts[end]) &&
        throw(DomainError(τ, "outside the saved times $(isempty(ts) ? "(none)" : (ts[1], ts[end]))"))
    k = searchsortedfirst(ts, τ)
    ts[k] == τ && return copy(us[k])
    θ = E((τ - ts[k-1]) / (ts[k] - ts[k-1]))
    return (1 - θ) .* us[k-1] .+ θ .* us[k]
end

"""
    DiffEqArray(u, t)

RecursiveArrayTools' record of states `u[k]` at times `t[k]`: a trajectory from anywhere (a measurement, a
simulation of another library), interpolated linearly. `A(τ)` is `at(A, τ)`.
"""
struct DiffEqArray{E}
    u::Vector{Vector{E}}
    t::Vector{Float64}
    function DiffEqArray{E}(u::Vector{Vector{E}}, t::Vector{Float64}) where {E}
        length(u) == length(t) || throw(DimensionMismatch("$(length(u)) states, $(length(t)) times"))
        issorted(t) || throw(ArgumentError("the times are not sorted"))
        return new{E}(u, t)
    end
end
DiffEqArray(u::AbstractVector{<:AbstractVector{E}}, t::AbstractVector{<:Real}) where {E} =
    DiffEqArray{E}(Vector{E}[Vector{E}(x) for x in u], Vector{Float64}(t))
@implement Trajectory{Vector{E},Float64} DiffEqArray{E} where {E} begin
    times(s) = s.t
    state(s, k) = s.u[k]
    at(s, τ) = linear_at(s.t, s.u, τ)
end
(s::DiffEqArray)(τ::Real) = at(s, Float64(τ))

"""
    plotpoints(s::Trajectory, i) -> Matrix{Float64}

The points `(t, u[i])` of a trajectory at its saved times, an n × 2 matrix, as plotting libraries take a
line's points (TraitMakie's `Positions`, `lines(points)`). Written against the trait: compiled once per state
and time type, for every solver and model; once for a `Dynamic` trajectory.
"""
function plotpoints(s::Trajectory{U,Float64}, i::Int) where {U}
    ts = times(s)
    n = length(ts)
    out = Matrix{Float64}(undef, n, 2)
    for k in 1:n
        out[k, 1] = ts[k]
        out[k, 2] = Float64(state(s, k)[i])
    end
    return out
end
end
end
