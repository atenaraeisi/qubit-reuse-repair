"""Stable extension points for the three external reuse-plan generators."""

from __future__ import annotations
from typing import Any, Protocol
from .context import CircuitSpec
from .plan import ReusePlan


class PlanAdapter(Protocol):
    """Implement separately for GidNET, CaQR, and DQCC.

    `to_common` MUST preserve original logical-qubit IDs. `materialize`
    must compile both original and repaired plans via the SAME path so
    physical comparisons are fair. Neither method ranks/repairs a plan.
    """
    name: str

    def circuit_spec(self, source_circuit: Any) -> CircuitSpec: ...
    def to_common(self, source_circuit: Any, native_plan: Any) -> ReusePlan: ...
    def materialize(self, source_circuit: Any, plan: ReusePlan) -> Any: ...


class PhysicalRouter(Protocol):
    """Separate hardware-routing backend; the router owns the frozen coupling map.

    Detailed result type: routing.backend.RouteObservation. This lightweight
    protocol is retained for existing adapter contracts.
    """
    def route(self, dynamic_circuit: Any, *, seed: int) -> Any: ...
