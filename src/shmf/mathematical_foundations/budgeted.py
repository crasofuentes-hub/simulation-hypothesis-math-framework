from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class BudgetedBranchingModel:
    """
    M2: Compute-constrained nesting model.

    Depth d has compute budget:
        C_d = C0 * r^d

    Each child simulation has sustaining cost k (compute units).
    Then the feasible branching at depth d is:
        b_d = floor(C_d / k)

    This prevents runaway branching in regimes where attenuation dominates.

    Parameters
    ----------
    base_compute_budget : float
        C0, normalized compute budget at depth 0.
    efficiency : float
        r in (0, 1], depth attenuation factor.
    child_cost : float
        k > 0, sustaining compute cost per child simulation.
    """

    base_compute_budget: float = 1.0
    efficiency: float = 0.6
    child_cost: float = 0.05

    def __post_init__(self) -> None:
        if self.base_compute_budget <= 0:
            raise ValueError("base_compute_budget must be > 0.")
        if not (0 < self.efficiency <= 1):
            raise ValueError("efficiency must be in (0, 1].")
        if self.child_cost <= 0:
            raise ValueError("child_cost must be > 0.")

    def compute_budget_at_depth(self, depth: int) -> float:
        if depth < 0:
            raise ValueError("depth must be >= 0.")
        return float(self.base_compute_budget * (self.efficiency**depth))

    def branching_at_depth(self, depth: int) -> int:
        c_d = self.compute_budget_at_depth(depth)
        return int(np.floor(c_d / self.child_cost))

    def simulations_per_depth(self, max_depth: int) -> List[int]:
        """
        Returns N_d for d = 0..max_depth, under budgeted branching:

        N_0 = 1
        N_{d+1} = N_d * b_d
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        n: List[int] = [1]
        for d in range(0, max_depth):
            b_d = self.branching_at_depth(d)
            n.append(n[-1] * b_d)
        return n

    def observer_weight_per_depth(self, max_depth: int) -> List[float]:
        """
        A proxy weight per depth:
            W_d = N_d * C_d

        This is a modeling choice (proxy for "observer-mass").
        """
        ns = self.simulations_per_depth(max_depth)
        weights: List[float] = []
        for d, n_d in enumerate(ns):
            c_d = self.compute_budget_at_depth(d)
            weights.append(float(n_d) * c_d)
        return weights

    def probability_depth_distribution(self, max_depth: int) -> np.ndarray:
        """
        P(d) = W_d / sum_i W_i  for d=0..max_depth
        """
        w = np.asarray(self.observer_weight_per_depth(max_depth), dtype=float)
        s = float(w.sum())
        if s <= 0:
            raise ValueError("Total weight is non-positive; check parameters.")
        return w / s
