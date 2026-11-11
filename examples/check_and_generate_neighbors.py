"""Qiskit-free phase-2 smoke example: python examples/check_and_generate_neighbors.py"""
import networkx as nx
from qubit_reuse_repair import ReusePlan, ReuseValidityChecker, ValidityAwareEngine

n = 4
ins = {q: ('in', q) for q in range(n)}
outs = {q: ('out', q) for q in range(n)}
g = nx.DiGraph()
g.add_edges_from((ins[q],outs[q]) for q in range(n))
checker = ReuseValidityChecker.from_graphs(
    num_qubits=n, active_qubits=range(n),
    full_graph=g, full_inputs=ins, full_outputs=outs)
original = ReusePlan.from_chains([[0,1],[2,3]])
engine = ValidityAwareEngine(checker)
print('Original:', original.chains, checker.validate(original).as_dict())
print('Allowed pair matrix over', engine.index.active, ':')
for row in engine.index.allowed:
    print('  ', [int(x) for x in row])
neighbors, stats = engine.neighbors(original)
print('Valid neighbors:', len(neighbors), stats)
for neighbor in neighbors[:10]:
    print('  ', neighbor.mutation.operator, neighbor.plan.chains, neighbor.plan.signature[:10])
