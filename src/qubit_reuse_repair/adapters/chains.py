"""Explicit chain-format ingress for GidNET / CaQR / DQCC outputs.

An adapter does NOT infer native compiler semantics or qubit-ID remapping.
Callers must export ordered reuse chains and supply native->logical IDs if
necessary. Materialization is delegated to a SAME-PATH compiler callback.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from ..core import CircuitSpec, ReusePlan
from ..validity.checker import active_qubits_from_circuit


@dataclass(frozen=True)
class ExplicitChainsAdapter:
    name: str
    native_to_logical: Mapping[int, int] | None = None
    compile_plan: Callable[[Any, ReusePlan], Any] | None = None

    def circuit_spec(self, source_circuit: Any) -> CircuitSpec:
        return CircuitSpec(source_circuit.num_qubits, active_qubits_from_circuit(source_circuit))

    def to_common(self, source_circuit: Any, native_plan: Any) -> ReusePlan:
        if isinstance(native_plan, ReusePlan):
            candidate = native_plan
        else:
            if isinstance(native_plan, Mapping):
                if set(native_plan) != {'chains'}:
                    raise ValueError('Only an explicit {"chains": ...} representation is supported')
                native_plan = native_plan['chains']
            if isinstance(native_plan, (str, bytes)):
                raise TypeError('Use ReusePlan.from_json for a JSON string')
            if self.native_to_logical is None:
                candidate = ReusePlan.from_chains(native_plan)
            else:
                chains = []
                for chain in native_plan:
                    converted = []
                    for qubit in chain:
                        if qubit not in self.native_to_logical:
                            raise ValueError(f'Missing native-to-logical mapping for {qubit!r}')
                        converted.append(self.native_to_logical[qubit])
                    chains.append(converted)
                candidate = ReusePlan.from_chains(chains)
        self.circuit_spec(source_circuit).check_plan_integrity(candidate)
        return candidate

    def materialize(self, source_circuit: Any, plan: ReusePlan) -> Any:
        if self.compile_plan is None:
            raise NotImplementedError(f'{self.name} native dynamic materializer not configured')
        self.circuit_spec(source_circuit).check_plan_integrity(plan)
        return self.compile_plan(source_circuit, plan)


class GidNETChainsAdapter(ExplicitChainsAdapter):
    def __init__(self, **kwargs): super().__init__('GidNET', **kwargs)


class CaQRChainsAdapter(ExplicitChainsAdapter):
    def __init__(self, **kwargs): super().__init__('CaQR', **kwargs)


class DQCCChainsAdapter(ExplicitChainsAdapter):
    def __init__(self, **kwargs): super().__init__('DQCC', **kwargs)
