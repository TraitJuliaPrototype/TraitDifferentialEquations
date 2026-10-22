"""
    ChainModels

A user package that knows nothing of TraitDifferentialEquations or TraitWorld: a callable model
`ChainModel{G}`, a diffusion on a graph of type `G` (standing for NetworkDynamics' graph type) with a
cubic damping, called as SciML calls a right-hand side, `m(du, u, p, t)`. It enters
`ODEProblem(m, u0, tspan, p)` as any function does (form B): `Dynamic` by default, so the solvers
compile nothing for a new graph type.

The parameters `p`: a vector `[k, d]`, or a named tuple with fields `k` and `d` (and optionally
`w`, a vector of node weights: a named tuple that is not a bits type).
"""
module ChainModels
export ChainModel, PathGraph, RingGraph, EdgeList, edgepairs

"A path 1–2–…–n."
struct PathGraph
    n::Int
end
"A ring: the path and the edge n–1."
struct RingGraph
    n::Int
end
"Any graph as its list of edges."
struct EdgeList
    n::Int
    edges::Vector{Tuple{Int,Int}}
end
nv(g::Union{PathGraph,RingGraph,EdgeList}) = g.n
edgepairs(g::PathGraph) = ((i, i + 1) for i in 1:g.n-1)
edgepairs(g::RingGraph) = ((i, mod1(i + 1, g.n)) for i in 1:g.n)
edgepairs(g::EdgeList) = g.edges

"A diffusion on the graph `graph`: `du[i] = k Σ_j (u[j] - u[i]) - d w[i] u[i]^3`."
struct ChainModel{G}
    graph::G
end

coupling(p::AbstractVector) = (p[1], p[2])
coupling(p::NamedTuple) = (p.k, p.d)
weight(p, i) = 1.0
weight(p::NamedTuple{(:k, :d, :w)}, i) = p.w[i]

function (m::ChainModel)(du, u, p, t)
    k, d = coupling(p)
    @inbounds for i in eachindex(du)
        du[i] = -d * weight(p, i) * u[i]^3
    end
    @inbounds for (i, j) in edgepairs(m.graph)
        flux = k * (u[j] - u[i])
        du[i] += flux
        du[j] -= flux
    end
    return nothing
end

end
