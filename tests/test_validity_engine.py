"""Pure graph tests; Qiskit not needed.  Fast filter vs independent graph oracle."""
from __future__ import annotations

import itertools
import random
from collections import Counter

import networkx as nx
import pytest

from qubit_reuse_repair import (ReusePlan, ReuseValidityChecker,
    ValidityAwareEngine, CompatibilityIndex, ValidationResult)
from qubit_reuse_repair.validity import OPERATORS, iter_mutation_deltas


def make_checker(n: int, seed: int, *, inactive=()):
    rng = random.Random(seed)
    g = nx.DiGraph()
    ins = {q: ("in", q) for q in range(n)}
    outs = {q: ("out", q) for q in range(n)}
    last = dict(ins)
    g.add_nodes_from(ins.values())
    g.add_nodes_from(outs.values())
    active = [q for q in range(n) if q not in inactive]
    for t in range(rng.randrange(2, 12)):
        wires = rng.sample(active, rng.randrange(1, min(3, len(active)) + 1))
        gate = ("gate", t)
        for q in wires:
            g.add_edge(last[q], gate)
            last[q] = gate
    for q in range(n):
        g.add_edge(last[q], outs[q])
    return ReuseValidityChecker.from_graphs(
        num_qubits=n, active_qubits=active,
        full_graph=g, full_inputs=ins, full_outputs=outs)


def independent_valid(ck, plan, *, target_width=None):
    """Slow full-event-DAG reference, independent of bitset closure code."""
    if plan.qubits != ck.active_qubits:
        return False
    if target_width is not None and plan.active_width != target_width:
        return False
    for a, b in plan.reuse_edges:
        if a == b or ck.candidate_outputs[a] in nx.descendants(ck.candidate_graph, ck.candidate_inputs[b]):
            return False
    g = ck.original_graph.copy()
    for a, b in plan.reuse_edges:
        g.add_edge(ck.output_nodes[a], ck.input_nodes[b])
    return nx.is_directed_acyclic_graph(g)


def all_plans(n, width):
    """Enumerate ordered chain partitions and canonical-deduplicate."""
    seen = set()
    for perm in itertools.permutations(range(n)):
        for cuts in itertools.combinations(range(1, n), width - 1):
            ch = []
            before = 0
            for cut in cuts + (n,):
                ch.append(perm[before:cut])
                before = cut
            plan = ReusePlan.from_chains(ch)
            if plan not in seen:
                seen.add(plan)
                yield plan


def legacy_mutate(plan, op):
    """Independent, raw direct chain rewrites (no mutation delta shortcuts)."""
    chains = plan.chains
    if op == 'suffix_swap':
        for i, j in itertools.combinations(range(len(chains)), 2):
            l, r = chains[i], chains[j]
            for a in range(1, len(l)+1):
                for b in range(1, len(r)+1):
                    out = list(chains)
                    out[i], out[j] = l[:a] + r[b:], r[:b] + l[a:]
                    yield ReusePlan.from_chains(out)
    elif op == 'chain_transfer':
        for i, src in enumerate(chains):
            if len(src) <= 1:
                continue
            for a, q in enumerate(src):
                for j, dst in enumerate(chains):
                    if i == j:
                        continue
                    for b in range(len(dst)+1):
                        out = list(chains)
                        out[i] = src[:a] + src[a+1:]
                        out[j] = dst[:b] + (q,) + dst[b:]
                        yield ReusePlan.from_chains(out)
    else:
        raise AssertionError(op)


def test_graph_validator_rejection_reasons_and_inactive_width():
    ck = make_checker(5, 12345, inactive=(4,))
    singleton = ReusePlan.from_chains([[0],[1],[2],[3]])
    assert ck.validate(singleton, target_width=4).is_valid
    assert ck.validate(singleton).full_width == 5
    assert ck.validate(singleton, target_width=3).reason == 'width_mismatch'
    assert ck.validate(singleton, target_width=True).reason == 'width_mismatch'
    assert ck.validate([[0, 1], [1], [2], [3]]).reason == 'malformed_chains'
    assert ck.validate([[0], [1], [2]]).reason == 'active_qubit_set_mismatch'
    assert ck.validate([[0], [1], [2], [3], [4]]).reason == 'active_qubit_set_mismatch'
    assert ck.validate([[0], [1], [2], []]).reason == 'malformed_chains'
    assert not ck.is_boundary_legal(4, 0)


def test_multi_qubit_gate_forbids_intra_gate_reuse():
    g = nx.DiGraph()
    ins = {q: ('i',q) for q in range(3)}
    outs = {q: ('o',q) for q in range(3)}
    g.add_nodes_from([*ins.values(),*outs.values()])
    for q in (0,1):
        g.add_edge(ins[q], '2q_gate')
        g.add_edge('2q_gate', outs[q])
    g.add_edge(ins[2],outs[2])
    ck = ReuseValidityChecker.from_graphs(num_qubits=3,active_qubits=range(3),
         full_graph=g,full_inputs=ins,full_outputs=outs)
    assert not ck.is_boundary_legal(0, 1)
    assert not ck.is_boundary_legal(1, 0)
    assert ck.validate([[0,1],[2]]).reason == 'illegal_reuse_boundary'
    assert ck.validate([[0,2],[1]]).is_valid


def test_exhaustive_checker_and_bitset_agreement_on_small_dags():
    comparisons = 0
    for n in (3, 4, 5):
        for seed in range(5):
            ck = make_checker(n, seed+n*100)
            idx = CompatibilityIndex(ck)
            for width in (2, 3):
                if width > n:
                    continue
                for plan in all_plans(n,width):
                    expected = independent_valid(ck, plan, target_width=width)
                    legal = all(idx.legal_boundary(a,b) for a,b in plan.reuse_edges)
                    fast = legal and idx.preserves_acyclicity(plan.reuse_edges)
                    assert fast == expected, (n, seed, plan, fast, expected)
                    assert ck.validate(plan,target_width=width).is_valid == expected
                    comparisons += 1
    assert comparisons > 1000


def test_full_neighbor_parity_and_mandatory_checker_calls(monkeypatch):
    total = Counter()
    for n in (3,4,5,6):
        for seed in range(5):
            ck = make_checker(n, seed+n*200)
            eng = ValidityAwareEngine(ck)
            sources = []
            for width in range(2, min(n, 4)):
                sources.extend(p for p in all_plans(n,width)
                               if independent_valid(ck,p,target_width=width))
            random.Random(seed).shuffle(sources)
            for source in sources[:12]:
                prepared = eng.prepare(source)
                seen = set()
                for op in OPERATORS:
                    expected = []
                    for proposal in legacy_mutate(source,op):
                        if proposal == source or proposal in seen:
                            continue
                        seen.add(proposal)
                        if independent_valid(ck,proposal,target_width=source.active_width):
                            expected.append(proposal)
                    actual, stats = prepared.scan(op)
                    assert [n.plan for n in actual] == expected
                    assert stats.full_reject == 0
                    assert stats.accepted == len(expected)
                    total['cases'] += 1
                    total['valid'] += stats.accepted
                    total['rejected'] += stats.fast_reject_illegal_boundary + stats.fast_reject_dependency_cycle
    assert total['cases'] >= 50
    assert total['valid'] > 0
    assert total['rejected'] > 0

    # Additionally prove that the full checker is invoked on every fast-survivor,
    # not merely in optional debug mode.
    ck = make_checker(5, 501)
    source = ReusePlan.from_chains([[0],[1],[2],[3],[4]])
    # Width==n has no nontrivial operators; find a valid width-3 source.
    source = next(p for p in all_plans(5,3) if ck.validate(p).is_valid)
    eng = ValidityAwareEngine(ck)
    prepared = eng.prepare(source)
    original_validate = ck.validate
    calls = []
    def track(plan, **kw):
        calls.append(plan)
        return original_validate(plan, **kw)
    monkeypatch.setattr(ck, 'validate', track)
    results, stats = prepared.scan_all()
    assert len(calls) == len(results)  # parent was prepared before monkeypatch
    assert all(original_validate(n.plan,target_width=source.active_width).is_valid for n in results)
    assert stats.accepted == len(results)


def test_raw_budget_deterministic_and_no_duplicate_across_operators():
    ck = make_checker(6, 313)
    eng = ValidityAwareEngine(ck)
    source = next((p for w in (3,4,5,6) for p in all_plans(6,w) if ck.validate(p).is_valid))
    first, stats1 = eng.neighbors(source, max_attempts_per_operator=4)
    again, stats2 = eng.neighbors(source, max_attempts_per_operator=4)
    assert first == again and stats1 == stats2
    assert stats1.attempted <= 8
    assert len({n.plan for n in first}) == len(first)
    assert all(n.plan.active_width == source.active_width for n in first)
    with pytest.raises(ValueError,match='Duplicate'):
        eng.neighbors(source,operators=('suffix_swap','suffix_swap'))
    with pytest.raises(ValueError):
        eng.prepare(ReusePlan.from_chains([[0],[1],[2],[3],[4],[5]])).scan('bogus')
    with pytest.raises(ValueError):
        eng.prepare(source).scan('chain_transfer',max_attempts=-1)


def test_unsupported_fail_closed():
    ck = make_checker(3,31)
    unsupported = ValidationResult(False, 'unsupported_classical_condition')
    ck2 = ReuseValidityChecker.from_graphs(num_qubits=3,active_qubits=range(3),
        full_graph=ck.original_graph,full_inputs=ck.input_nodes,full_outputs=ck.output_nodes,
        unsupported=unsupported)
    src = ReusePlan.from_chains([[0],[1],[2]])
    assert ck2.validate(src).reason == 'unsupported_classical_condition'
    with pytest.raises(ValueError,match='Unsupported'):
        ValidityAwareEngine(ck2)


def test_invalid_parent_rejected_before_neighbor_search():
    ck = make_checker(4,0)
    eng = ValidityAwareEngine(ck)
    bad = next(p for p in all_plans(4,2) if not ck.validate(p).is_valid)
    assert not ck.validate(bad).is_valid
    with pytest.raises(ValueError,match='not statically valid'):
        eng.prepare(bad)


def test_pairwise_legal_edges_may_still_form_dependency_cycle():
    """Regression witness: pair matrix alone MUST NOT accept whole plan."""
    ck = make_checker(4, 2)
    plan = ReusePlan.from_chains([[0], [1, 3, 2]])
    idx = CompatibilityIndex(ck)
    assert all(idx.legal_boundary(a,b) for a,b in plan.reuse_edges)
    assert not idx.preserves_acyclicity(plan.reuse_edges)
    assert ck.validate(plan).reason == 'dependency_cycle'


def test_adversarial_malformed_inputs_fail_closed_300_cases():
    """300 deterministic malformed source plans; no accepted invalid mutation."""
    ck = make_checker(5, 0)
    for i in range(300):
        k = i % 6
        if k == 0:
            p = [[0,1],[1],[2],[3],[4]]  # duplicate
        elif k == 1:
            p = [[0],[1],[2],[3]]      # missing active logical qubit
        elif k == 2:
            p = [[0],[1],[2],[3],[4],[5]]  # extra unknown qubit
        elif k == 3:
            p = [[0],[1],[2],[3],[]]   # empty chain
        elif k == 4:
            p = [[0],[1],[2],[3],[True]]  # bool ID
        else:
            p = [[0],[1],[2],[3],[-1]] # negative ID
        assert not ck.validate(p).is_valid
