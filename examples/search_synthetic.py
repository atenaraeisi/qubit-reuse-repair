"""Qiskit-free end-to-end pre-routing Search smoke demonstration.

The 2q pairs below are a synthetic proxy context, NOT a quantum circuit or
measured physical improvement. This shows search working on validated chains.
Run: python examples/search_synthetic.py
"""
from __future__ import annotations
import json
import networkx as nx
from qubit_reuse_repair import (ReusePlan, ReuseValidityChecker, ValidityAwareEngine,
    RepairConfig, SearchEngine, SearchLimits, StructuralScorer)

n = 6
ins = {q: ("in", q) for q in range(n)}
outs = {q: ("out", q) for q in range(n)}
g = nx.DiGraph()
g.add_edges_from((ins[q], outs[q]) for q in range(n))
checker = ReuseValidityChecker.from_graphs(num_qubits=n, active_qubits=range(n),
    full_graph=g, full_inputs=ins, full_outputs=outs)
original = ReusePlan.from_chains([[0,1], [2,3], [4,5]])
synthetic_pairs = ([(0,2)]*6 + [(1,4)]*4 + [(3,5)]*3 + [(0,5)]*2 + [(2,4)]*2)
scorer = StructuralScorer(tuple(synthetic_pairs))
for policy in ("structural_beam", "random_uniform"):
    cfg = RepairConfig(search_policy=policy, model='none', objective='swap_count',
        max_depth=3, beam_width=5, budget=100, search_seed=2026)
    result = SearchEngine(ValidityAwareEngine(checker), cfg,
        scorer if policy == 'structural_beam' else None,
        limits=SearchLimits(max_attempts_per_operator=150, raw_attempt_budget=1000)).run(original)
    print(f"{policy}: {original.chains} -> {result.selected.chains}")
    print(f"  changed={result.changed}; n_scored={result.n_scored}; raw_attempts={result.raw_attempts}")
    print(f"  score={result.selected_score}; physical_gain_measured={result.to_dict()['physical_gain_measured']}")
