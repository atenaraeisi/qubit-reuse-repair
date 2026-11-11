"""Use a common-plan JSON and a source QASM with Qiskit 0.45.3 installed.

  python examples/check_qasm_neighbors.py benchmarks/my_circuit.qasm plan.json

Input plan.json: [[0,4],[1],[2,3]] with original logical IDs, *active* only.
"""
import json
import sys
from qiskit import QuantumCircuit
from qubit_reuse_repair import ReusePlan, ReuseValidityChecker, ValidityAwareEngine

if len(sys.argv) != 3:
    raise SystemExit('usage: python examples/check_qasm_neighbors.py circuit.qasm plan.json')
qc = QuantumCircuit.from_qasm_file(sys.argv[1])
plan = ReusePlan.from_json(open(sys.argv[2], encoding='utf-8').read())
checker = ReuseValidityChecker(qc)
verdict = checker.validate(plan, target_width=plan.active_width)
print('Original validation:', verdict.as_dict())
if not verdict.is_valid:
    raise SystemExit('INVALID original plan; fail closed')
engine = ValidityAwareEngine(checker)
neighbors, stats = engine.neighbors(plan)
print('Generation stats:', stats)
print('Candidate plans:', json.dumps([n.plan.to_dict() for n in neighbors], indent=2))
