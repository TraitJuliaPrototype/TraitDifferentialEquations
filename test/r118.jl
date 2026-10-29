# Open issue 118 (W-32, E-245): structured indices and observed values by the trait protocol. A system maps
# indices of its own type (`IndexProvider{I}`) to a state variable, a parameter or an observed value's getter;
# `getu` reads them from the system, a solution, an integrator and a callback's view.
# Runs inside runtests.jl (which defines caught).

"A model's own index types (as NetworkDynamics' `VIndex`/`EIndex`)."
abstract type GIdx end
struct Node <: GIdx; i::Int; end
struct Flow <: GIdx; i::Int; end          # observed: the gain times the difference of two neighbours
struct Gain <: GIdx; end                  # the parameter
"A chain of decaying nodes: a callable model that carries itself as its system, with indices of its own."
struct Chain
    n::Int
end
(m::Chain)(du, u, p, t) = (du .= -p[1] .* u; nothing)
@implement HasSystem Chain begin
    system(m) = m
end
@implement IndexProvider{GIdx} Chain begin
    variable_index(m, x) = x isa Node ? x.i : nothing
    parameter_index(m, x) = x isa Gain ? 1 : nothing
    observed(m, x) = x isa Flow ? Kept{Callable}((u, p, t) -> p[1] * (u[x.i + 1] - u[x.i])) : nothing
end
"The same model with vectors of indices as indices of its own: one getter for all of them."
struct VChain
    n::Int
end
(m::VChain)(du, u, p, t) = (du .= -p[1] .* u; nothing)
@implement HasSystem VChain begin
    system(m) = m
end
@implement IndexProvider{Union{GIdx, Vector{Flow}}} VChain begin
    variable_index(m, x) = x isa Node ? x.i : nothing
    parameter_index(m, x) = x isa Gain ? 1 : nothing
    observed(m, x) = x isa Flow ? Kept{Callable}((u, p, t) -> p[1] * (u[x.i + 1] - u[x.i])) :
                     x isa Vector{Flow} ? Kept{Callable}((u, p, t) -> [p[1] * (u[f.i + 1] - u[f.i]) for f in x]) : nothing
end

const U3 = [1.0, 2.0, 4.0]
flow(u, p, i) = p[1] * (u[i + 1] - u[i])

@testset "open issue 118: a system's own indices" begin
    m = Chain(3)
    # the system itself: a getter at (u, p, t)
    @test getu(m, Node(2))(U3, [0.5], 0.0) == 2.0 && getu(m, Gain())(U3, [0.5], 0.0) == 0.5
    @test getu(m, Flow(2))(U3, [0.5], 0.0) == 1.0 && getu(m, [Node(1), Flow(1)])(U3, [0.5], 0.0) == [1.0, 0.5]
    # a solution: the values at its saved times
    s = solve(ODEProblem(m, copy(U3), (0.0, 1.0), [0.5]), Tsit5(); saveat = [0.0, 0.5, 1.0])
    @test getsystem(s) === m
    @test getu(s, Node(3))(s) == [u[3] for u in s.u]
    @test getu(s, Flow(1))(s) == [flow(u, [0.5], 1) for u in s.u]
    @test getu(s, [Flow(1), Node(2)])(s) == [[flow(u, [0.5], 1), u[2]] for u in s.u]
    # the solution between its saved times, by the same getters
    s2 = solve(ODEProblem(m, copy(U3), (0.0, 1.0), [0.5]), Tsit5())
    @test s2(0.3; idxs = Flow(2)) == flow(s2(0.3), [0.5], 2) && s2(0.3; idxs = Node(1)) == s2(0.3)[1]
    @test s2([0.3, 0.6]; idxs = Gain()) == [0.5, 0.5]
    @test s2([0.3, 0.6]; idxs = [Node(1), Flow(1)]) == [[s2(τ)[1], flow(s2(τ), [0.5], 1)] for τ in (0.3, 0.6)]
    # parameters changed by a callback and saved: each time reads the ones in force there (at the callback's time
    # the states before and after its affect, both read with the parameters saved at that time, as
    # SymbolicIndexingInterface reads a time)
    change = DiscreteCallback((u, t, i) -> t >= 0.5 && i.p[1] == 0.5, i -> (i.p[1] = 2.0; save_discretes!(i); nothing))
    sc = solve(ODEProblem(m, copy(U3), (0.0, 1.0), [0.5]), Tsit5(); saveat = [0.25, 0.75], callback = change)
    tc = sc.t[2]
    @test sc.t == [0.25, tc, tc, 0.75] && 0.5 <= tc < 0.75
    @test getu(sc, Gain())(sc) == [0.5, 2.0, 2.0, 2.0] &&
          getu(sc, Flow(1))(sc) == [flow(sc.u[1], [0.5], 1); [flow(sc.u[k], [2.0], 1) for k in 2:4]]
    @test sc(0.75; idxs = Flow(1)) == flow(sc(0.75), [2.0], 1)
    # the integrator, and a callback's view of it
    integ = init(ODEProblem(m, copy(U3), (0.0, 1.0), [0.5]), Tsit5())
    step!(integ, 0.25, true)
    @test getu(integ, Flow(2))(integ) == flow(integ.u, [0.5], 2) && getu(integ, Node(1))(integ) == integ.u[1]
    seen = Float64[]
    look = DiscreteCallback((u, t, i) -> isempty(seen) && t > 0.2, i -> (push!(seen, getu(i, Flow(1))(i), i.u[2] - i.u[1]); nothing))
    solve(ODEProblem(m, copy(U3), (0.0, 1.0), [0.5]), Tsit5(); callback = look)
    @test length(seen) == 2 && seen[1] == 0.5 * seen[2]
    # vectors of indices of the system's own: its getter, not one per index
    v = VChain(3)
    sv = solve(ODEProblem(v, copy(U3), (0.0, 1.0), [0.5]), Tsit5(); saveat = [0.0, 1.0])
    @test getu(sv, [Flow(1), Flow(2)])(sv) == [[flow(u, [0.5], 1), flow(u, [0.5], 2)] for u in sv.u]
    @test getu(sv, [Node(1), Flow(2)])(sv) == [[u[1], flow(u, [0.5], 2)] for u in sv.u]   # not an index of its own: one getter per index
    # names: a SymbolCache's
    sn = solve(ODEProblem(ODEFunction(m; sys = SymbolCache([:a, :b, :c], [:k])), copy(U3), (0.0, 1.0), [0.5]), Tsit5())
    @test getu(sn, :b)(sn) == sn[:b] && getu(sn, :k)(sn) == fill(0.5, length(sn.t)) && getu(sn, [:a, :c])(sn) == [[u[1], u[3]] for u in sn.u]
    # refused: another index type, an unknown index, a problem without a system, a value that provides no values
    @test caught(() -> getu(s, :x)) isa ArgumentError
    @test caught(() -> getu(s, Flow(1))(U3)) isa ArgumentError
    e = caught(() -> getu(m, Flow(7))(U3, [0.5], 0.0))
    @test e isa BoundsError
    @test caught(() -> getu(solve(ODEProblem((du, u, p, t) -> (du .= -u; nothing), [1.0], (0.0, 1.0)), Tsit5()), Node(1))) isa ArgumentError
    @test caught(() -> s(0.5; idxs = 1.5)) isa ArgumentError
end
