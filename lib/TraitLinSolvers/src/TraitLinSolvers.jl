"""
    TraitLinSolvers

A linear solver package over `ODECore` (stage C of SciML's move 3, E-220), as LinearSolve.jl's algorithms are
for OrdinaryDiffEq: it implements the core's `LinearSolver` and `LinearCache` for its own types. Neither it nor
any solver package knows the other.
"""
module TraitLinSolvers
using TraitWorld
using ODECore: LinearSolver, LinearCache
import ODECore: lincache, linfactor!, linsolve!
import LinearAlgebra
export DenseQR
@trait_module begin
"Dense QR with column pivoting: slower than LU, more robust for ill-conditioned systems."
struct DenseQR end
mutable struct DenseQRCache
    F::LinearAlgebra.QRPivoted{Float64,Matrix{Float64},Vector{Float64},Vector{Int}}
end
@implement LinearCache DenseQRCache begin
    linfactor!(c, W) = (c.F = LinearAlgebra.qr!(W, LinearAlgebra.ColumnNorm()); nothing)
    linsolve!(c, x, b) = (LinearAlgebra.ldiv!(x, c.F, b); nothing)
end
@implement LinearSolver DenseQR begin
    lincache(ls, n) = DenseQRCache(LinearAlgebra.qr!(Matrix{Float64}(LinearAlgebra.I, n, n), LinearAlgebra.ColumnNorm()))
end
end
end
