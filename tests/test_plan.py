import json
import pytest

from qubit_reuse_repair import ReusePlan, CircuitSpec


def test_canonical_order_preserves_each_chain_direction():
    p1 = ReusePlan.from_chains([[2, 4], [0, 3], [1]])
    p2 = ReusePlan.from_chains([[1], [0, 3], [2, 4]])
    assert p1 == p2
    assert p1.signature == p2.signature
    assert p1.chains == ((0, 3), (1,), (2, 4))
    assert p1.reuse_edges == ((0, 3), (2, 4))
    assert ReusePlan.from_chains([[3, 0], [1], [2, 4]]) != p1


def test_width_and_membership():
    p = ReusePlan.from_chains([[1], [0, 2], [3, 4]])
    ctx = CircuitSpec(n_declared_qubits=7, active_qubits=frozenset(range(5)))
    ctx.check_plan_integrity(p, target_active_width=3)
    assert p.active_width == 3
    assert p.n_active_qubits == 5
    assert ctx.full_width(p) == 5  # 3 active chains + 2 declared inactive
    with pytest.raises(ValueError, match="active width changed"):
        ctx.check_plan_integrity(p, target_active_width=2)
    with pytest.raises(ValueError, match="mismatch"):
        CircuitSpec(7, frozenset(range(6))).check_plan_integrity(p)


@pytest.mark.parametrize("chains", [[], [[0], []], [[0, 1], [1]], [[-1]], [[True]], [[0.1]]])
def test_malformed_plans_rejected(chains):
    with pytest.raises((ValueError, TypeError)):
        ReusePlan.from_chains(chains)


def test_roundtrip():
    p = ReusePlan.from_chains([[4, 7, 8], [0, 3], [2]])
    assert ReusePlan.from_json(p.to_json()) == p
    assert ReusePlan.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError):
        ReusePlan.from_json('{"not":"a plan"}')
    with pytest.raises(ValueError):
        ReusePlan.from_dict({"schema_version": 9, "chains": [[0]]})
