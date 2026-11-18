"""
    TraitLinSolvers

A linear solver package over `ODECore` (stage C of SciML's move 3, E-220), as LinearSolve.jl's algorithms are
for OrdinaryDiffEq: it implements the core's `LinearSolver` for its own type; the factorization it returns is LinearAlgebra's, a `Solvable`. Neither it nor
any solver package knows the other.
"""
module TraitLinSolvers
using TraitWorld
@trait_module
using ODECore: LinearSolver
import ODECore: linfactor
import LinearAlgebra
export DenseQR
"Dense QR with column pivoting: slower than LU, more robust for ill-conditioned systems."
struct DenseQR end
@implement LinearSolver DenseQR begin
    linfactor(ls, W) = LinearAlgebra.qr!(W, LinearAlgebra.ColumnNorm())
end
end
