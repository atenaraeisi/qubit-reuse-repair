"""Reproducible physical-routing backend and observations.

Physical metrics deliberately match the *metric definitions* in the earlier
Experiment-4A router: Qiskit transpile + SABRE layout/routing + opt_level=1;
SWAP is routed.count_ops()['swap'], routed depth is routed.depth().

The Phase-5 generator-neutral materializer is NEW, so matching the router
configuration does NOT reproduce old Experiment-4A cost data or semantics.
No routing information may be consulted during structural search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Protocol


MAX_SEED = 2**32 - 1


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SEED:
        raise ValueError(f"routing seed must be an integer in [0, {MAX_SEED}]")
    return value


@dataclass(frozen=True)
class RoutingSettings:
    """Locked router options, separate from independently chosen search seeds."""

    layout_method: str = "sabre"
    routing_method: str = "sabre"
    optimization_level: int = 1

    def __post_init__(self) -> None:
        if not self.layout_method or not isinstance(self.layout_method, str):
            raise ValueError("layout_method must be a nonempty string")
        if not self.routing_method or not isinstance(self.routing_method, str):
            raise ValueError("routing_method must be a nonempty string")
        if isinstance(self.optimization_level, bool) or self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2 or 3")

    def to_dict(self) -> dict:
        return {"layout_method": self.layout_method, "routing_method": self.routing_method,
                "optimization_level": self.optimization_level,
                "metric_definitions": {
                    "swap_count": "int(transpiled_circuit.count_ops().get('swap', 0))",
                    "routed_depth": "int(transpiled_circuit.depth())",
                }}


@dataclass(frozen=True)
class RouteObservation:
    """One actual routing attempt. The circuit is the routed output itself."""

    circuit: Any
    seed: int
    swap_count: int
    routed_depth: int
    routing_time_sec: float
    operation_counts: dict[str, int]

    def __post_init__(self) -> None:
        _seed(self.seed)
        for name in ("swap_count", "routed_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.routing_time_sec) or self.routing_time_sec < 0:
            raise ValueError("routing_time_sec must be nonnegative and finite")
        if self.circuit is None:
            raise ValueError("routed circuit must not be None")
        if not isinstance(self.operation_counts, dict) or any(
            not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v < 0
            for k, v in self.operation_counts.items()
        ):
            raise ValueError("operation_counts must be a string-to-nonnegative-int mapping")
        if self.operation_counts.get("swap", 0) != self.swap_count:
            raise ValueError("swap_count disagrees with routed operation counts")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "swap_count": self.swap_count,
            "routed_depth": self.routed_depth,
            "routing_time_sec": self.routing_time_sec,
            "operation_counts": dict(sorted(self.operation_counts.items())),
        }


class RoutingBackend(Protocol):
    """Injectable router for Qiskit production routing or deterministic tests."""

    def route(self, circuit: Any, *, seed: int) -> RouteObservation: ...


class RoutingFailure(RuntimeError):
    """Routing was attempted but did not produce a valid physical result."""


class QiskitSabreRouter:
    """Physical routing in Qiskit; no reliance on GidNET/CaQR/DQCC internals.

    The underlying topology is frozen when this instance is made. By default,
    the historical SABRE configuration is used; no explicit basis_gates are
    supplied to reproduce the legacy metric convention, not hardware timing.
    """

    def __init__(self, coupling_map: Any, settings: RoutingSettings | None = None):
        # Qiskit imported lazily, so all other stages still run without it.
        from qiskit.transpiler import CouplingMap
        if not isinstance(coupling_map, CouplingMap):
            raise TypeError("coupling_map must be a qiskit.transpiler.CouplingMap")
        if coupling_map.size() < 1:
            raise ValueError("coupling map has no physical qubits")
        self.coupling_map = coupling_map
        self.settings = settings or RoutingSettings()

    def route(self, circuit: Any, *, seed: int) -> RouteObservation:
        from qiskit import QuantumCircuit, transpile
        seed = _seed(seed)
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("route requires a Qiskit QuantumCircuit")
        if circuit.num_qubits > self.coupling_map.size():
            raise ValueError(f"circuit width {circuit.num_qubits} exceeds hardware width {self.coupling_map.size()}")
        start = time.perf_counter()
        try:
            routed = transpile(
                circuit, coupling_map=self.coupling_map,
                layout_method=self.settings.layout_method,
                routing_method=self.settings.routing_method,
                optimization_level=self.settings.optimization_level,
                seed_transpiler=seed,
            )
        except Exception as exc:
            raise RoutingFailure(f"Qiskit routing failed for seed {seed}: {exc}") from exc
        duration = time.perf_counter() - start
        if routed is None:
            raise RoutingFailure("Qiskit routing returned None")
        counts = {str(name): int(n) for name, n in routed.count_ops().items()}
        depth = routed.depth()
        if depth is None:
            raise RoutingFailure("Qiskit returned an undefined routed depth")
        return RouteObservation(
            circuit=routed, seed=seed,
            swap_count=counts.get("swap", 0), routed_depth=int(depth),
            routing_time_sec=duration, operation_counts=counts,
        )


def make_coupling_map(topology: str, *, physical_qubits: int | None = None,
                      grid_rows: int | None = None, grid_cols: int | None = None):
    """Build a bidirectional line/ring/grid or exact heavy-hex-distance-7 map.

    Exposes topology separately from the reuse-plan's chain width. For line/
    ring, the number of hardware qubits is mandatory. For grid, use rows/cols.
    """
    from qiskit.transpiler import CouplingMap
    if topology == "heavy_hex7":
        if physical_qubits is not None or grid_rows is not None or grid_cols is not None:
            raise ValueError("heavy_hex7 topology size is fixed; omit width/grid options")
        return CouplingMap.from_heavy_hex(7)
    if topology == "grid":
        if physical_qubits is not None or grid_rows is None or grid_cols is None:
            raise ValueError("grid requires --grid-rows and --grid-cols, not --physical-qubits")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 2
               for n in (grid_rows, grid_cols)):
            raise ValueError("grid rows and cols must both be >= 2")
        return CouplingMap.from_grid(grid_rows, grid_cols, bidirectional=True)
    if topology not in ("line", "ring"):
        raise ValueError("topology must be line, ring, grid or heavy_hex7")
    if (isinstance(physical_qubits, bool) or not isinstance(physical_qubits, int)
            or physical_qubits < (3 if topology == "ring" else 2)):
        raise ValueError("line needs >=2 hardware qubits; ring needs >=3")
    if grid_rows is not None or grid_cols is not None:
        raise ValueError("rows/cols only apply to grid")
    constructor = CouplingMap.from_line if topology == "line" else CouplingMap.from_ring
    return constructor(physical_qubits, bidirectional=True)
