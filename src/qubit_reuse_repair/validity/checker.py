"""Generator-independent *static* dependency and reuse-plan validator.

Port of the user's experiments/reuse_audit_fixed/pilot_code/reuse_validity.py,
using the common immutable ReusePlan from phase 1.  No GidNET/CaQR/DQCC imports.

Support: static circuits with Gate instructions, barriers, terminal measurement.
This checker deliberately does not certify dynamic-circuit semantic equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from ..core import CircuitSpec, ReusePlan


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str
    active_width: int | None = None
    full_width: int | None = None
    num_reuse_boundaries: int = 0
    boundaries: tuple[tuple[int, int], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def legal(self) -> bool:
        return self.is_valid

    def as_dict(self) -> dict[str, Any]:
        return {
            "legal": self.legal, "is_valid": self.is_valid,
            "reason": self.reason, "active_width": self.active_width,
            "full_width": self.full_width,
            "num_reuse_boundaries": self.num_reuse_boundaries,
            "boundaries": self.boundaries,
            **self.details,
        }


def active_qubits_from_circuit(circuit: Any) -> frozenset[int]:
    """Same active convention as the audited original: ignore barrier/measure."""
    active: set[int] = set()
    for inst in circuit.data:
        if inst.operation.name in {"measure", "barrier"}:
            continue
        for q in inst.qubits:
            active.add(circuit.find_bit(q).index)
    return frozenset(active)


def _supported_circuit(circuit: Any) -> ValidationResult | None:
    try:
        from qiskit.circuit import Gate
    except ImportError as exc:
        raise ImportError("Qiskit is required for QuantumCircuit inputs; use from_graphs for pure graph tests") from exc

    measured: set[int] = set()
    for idx, inst in enumerate(circuit.data):
        op = inst.operation
        name = op.name
        qs = {circuit.find_bit(q).index for q in inst.qubits}
        if getattr(op, "condition", None) is not None:
            return ValidationResult(False, "unsupported_classical_condition", details={"operation_index": idx, "name": name})
        if name == "barrier":
            continue
        if name == "measure":
            if len(qs) != 1 or len(inst.clbits) != 1:
                return ValidationResult(False, "unsupported_measurement", details={"operation_index": idx})
            measured.update(qs)
            continue
        if qs & measured:
            return ValidationResult(False, "mid_circuit_measurement", details={"operation_index": idx, "qubits": tuple(sorted(qs & measured))})
        if not isinstance(op, Gate):
            return ValidationResult(False, "unsupported_nonunitary_instruction", details={"operation_index": idx, "name": name})
    return None


def _circuit_to_graph(circuit: Any, *, omit_final_measurements: bool):
    try:
        from qiskit.converters import circuit_to_dag
    except ImportError as exc:
        raise ImportError("Install the Qiskit extra (recommended Qiskit 0.45.3/Python 3.10 or 3.11)") from exc
    qc = circuit.copy()
    if omit_final_measurements:
        qc.remove_final_measurements(inplace=True)
    dag = circuit_to_dag(qc)
    for node in tuple(dag.op_nodes()):
        if node.op.name == "barrier":
            dag.remove_op_node(node)
    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes())
    for source, target, _wire in dag.edges():
        g.add_edge(source, target)
    inputs = {i: dag.input_map[q] for i, q in enumerate(dag.qubits)}
    outputs = {i: dag.output_map[q] for i, q in enumerate(dag.qubits)}
    return g, inputs, outputs


class ReuseValidityChecker:
    """Independently validate fixed-*active*-width reuse plans on a circuit DAG.

    Construct once per circuit, reuse across many candidates. Pair legality is
    computed on a graph without terminal measurements; global acyclicity is
    tested on the original graph including terminal measurements.
    """

    def __init__(self, circuit: Any):
        active = active_qubits_from_circuit(circuit)
        if not active:
            raise ValueError("No gate-active logical qubits")
        self.spec = CircuitSpec(int(circuit.num_qubits), active)
        self.source_circuit = circuit
        self.unsupported = _supported_circuit(circuit)
        self.original_graph, self.input_nodes, self.output_nodes = _circuit_to_graph(
            circuit, omit_final_measurements=False)
        self.candidate_graph, self.candidate_inputs, self.candidate_outputs = _circuit_to_graph(
            circuit, omit_final_measurements=True)
        self._finish_initialization()

    @classmethod
    def from_graphs(
        cls, *, num_qubits: int, active_qubits: Iterable[int],
        full_graph: nx.DiGraph, full_inputs: Mapping[int, Any], full_outputs: Mapping[int, Any],
        candidate_graph: nx.DiGraph | None = None,
        candidate_inputs: Mapping[int, Any] | None = None,
        candidate_outputs: Mapping[int, Any] | None = None,
        unsupported: ValidationResult | None = None,
    ) -> "ReuseValidityChecker":
        """Qiskit-free constructor; also useful for a future native DAG adapter."""
        obj = cls.__new__(cls)
        obj.spec = CircuitSpec(num_qubits, frozenset(active_qubits))
        obj.source_circuit = None
        obj.unsupported = unsupported
        obj.original_graph = full_graph.copy()
        obj.input_nodes = dict(full_inputs)
        obj.output_nodes = dict(full_outputs)
        obj.candidate_graph = (candidate_graph if candidate_graph is not None else full_graph).copy()
        obj.candidate_inputs = dict(candidate_inputs if candidate_inputs is not None else full_inputs)
        obj.candidate_outputs = dict(candidate_outputs if candidate_outputs is not None else full_outputs)
        obj._finish_initialization()
        return obj

    @property
    def active_qubits(self) -> frozenset[int]:
        return self.spec.active_qubits

    @property
    def num_qubits(self) -> int:
        return self.spec.n_declared_qubits

    def _finish_initialization(self) -> None:
        expected = set(range(self.num_qubits))
        for mapping in (self.input_nodes, self.output_nodes, self.candidate_inputs, self.candidate_outputs):
            if set(mapping) != expected:
                raise ValueError("Every DAG input/output map must contain all declared logical qubits")
        if not nx.is_directed_acyclic_graph(self.original_graph):
            raise ValueError("Original circuit graph is not acyclic")
        if not nx.is_directed_acyclic_graph(self.candidate_graph):
            raise ValueError("Candidate circuit graph is not acyclic")
        for mapping, graph in ((self.input_nodes, self.original_graph),
                               (self.output_nodes, self.original_graph),
                               (self.candidate_inputs, self.candidate_graph),
                               (self.candidate_outputs, self.candidate_graph)):
            if any(n not in graph for n in mapping.values()):
                raise ValueError("Input/output DAG node missing in dependency graph")
        self._forbidden_pairs = self._compute_forbidden()

    def _compute_forbidden(self) -> frozenset[tuple[int, int]]:
        blocked = set()
        for b in range(self.num_qubits):
            reachable = nx.descendants(self.candidate_graph, self.candidate_inputs[b])
            reachable.add(self.candidate_inputs[b])
            for a in range(self.num_qubits):
                if a == b or self.candidate_outputs[a] in reachable:
                    blocked.add((a, b))
        return frozenset(blocked)

    def is_boundary_legal(self, a: int, b: int) -> bool:
        return a in self.active_qubits and b in self.active_qubits and (a, b) not in self._forbidden_pairs

    def validate(self, plan: ReusePlan | Sequence[Sequence[int]], *,
                 target_width: int | None = None) -> ValidationResult:
        """Return typed result, including reason for invalid input. No semantic proof."""
        try:
            p = plan if isinstance(plan, ReusePlan) else ReusePlan.from_chains(plan)
        except (ValueError, TypeError, OverflowError):
            return ValidationResult(False, "malformed_chains")
        width = p.active_width
        full_width = width + self.spec.n_inactive_qubits
        if p.qubits != self.spec.active_qubits:
            return ValidationResult(False, "active_qubit_set_mismatch", width, full_width,
                                    details={"missing": tuple(sorted(self.spec.active_qubits - p.qubits)),
                                             "extra": tuple(sorted(p.qubits - self.spec.active_qubits))})
        if target_width is not None and (isinstance(target_width, bool)
                                         or not isinstance(target_width, Integral) or target_width != width):
            return ValidationResult(False, "width_mismatch", width, full_width,
                                    details={"target_width": target_width})
        if self.unsupported is not None:
            return ValidationResult(False, self.unsupported.reason, width, full_width,
                                    details=self.unsupported.details)
        edges = p.reuse_edges
        bad = tuple((a, b) for a, b in edges if not self.is_boundary_legal(a, b))
        if bad:
            return ValidationResult(False, "illegal_reuse_boundary", width, full_width,
                                    len(edges), edges, {"illegal_boundaries": bad})
        combined = self.original_graph.copy()
        for a, b in edges:
            combined.add_edge(self.output_nodes[a], self.input_nodes[b])
        if not nx.is_directed_acyclic_graph(combined):
            return ValidationResult(False, "dependency_cycle", width, full_width,
                                    len(edges), edges)
        return ValidationResult(True, "ok_static", width, full_width,
                                len(edges), edges,
                                {"static_validity_only": True,
                                 "inactive_qubits": tuple(sorted(set(range(self.num_qubits)) - self.active_qubits))})
