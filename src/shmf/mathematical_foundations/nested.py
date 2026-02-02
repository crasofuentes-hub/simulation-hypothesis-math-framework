from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NestedSimulationModel:
    """Nested simulation resource model (branching-aware).

    Parameters
    ----------
    base_compute_budget:
        Normalized compute budget at depth 0 (dimensionless).
    efficiency:
        Fraction of parent compute budget available to the child (0 < efficiency < 1).
    branching_factor:
        Number of child simulations per simulation node (>= 0).

    Notes
    -----
    Single-branch budget at depth d:
        B(d) = B0 * efficiency^d

    Aggregate budget up to depth D:
        Total(D) = B0 * sum_{d=0..D} (branching_factor * efficiency)^d
    """

    base_compute_budget: float = 1.0
    efficiency: float = 0.5
    branching_factor: int = 1

    def __post_init__(self) -> None:
        if not (self.base_compute_budget > 0):
            raise ValueError("base_compute_budget must be > 0.")
        if not (0.0 < self.efficiency < 1.0):
            raise ValueError("efficiency must satisfy 0 < efficiency < 1.")
        if self.branching_factor < 0:
            raise ValueError("branching_factor must be >= 0.")

    def compute_budget_at_depth(self, depth: int) -> float:
        if depth < 0:
            raise ValueError("depth must be >= 0.")
        return self.base_compute_budget * (self.efficiency ** depth)

    def total_budget_up_to_depth(self, max_depth: int) -> float:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        r = float(self.branching_factor) * self.efficiency
        if r == 1.0:
            return self.base_compute_budget * float(max_depth + 1)
        return self.base_compute_budget * ((1.0 - (r ** (max_depth + 1))) / (1.0 - r))