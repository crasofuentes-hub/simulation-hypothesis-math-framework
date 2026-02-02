from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StochasticBranchingModel:
    """
    M3: Stochastic branching with Monte Carlo, reproducible by seed.

    Budgets:
        C_d = C0 * r^d

    Branching:
        b_d ~ Poisson(lambda_d)
    where lambda_d is drawn from a prior or is a fixed function of budget.

    We provide two commonly useful priors:

    - "fixed_rate": lambda_d = lambda0  (baseline)
    - "budget_scaled": lambda_d = alpha * C_d / k
        where k is a unit cost scaling factor and alpha is an efficiency multiplier.
      This ties branching to compute budget in expectation.

    Notes
    -----
    This model yields distributions over N_d and weight W_d = N_d * C_d.

    Parameters
    ----------
    base_compute_budget : float
        C0
    efficiency : float
        r in (0,1]
    prior_kind : str
        "fixed_rate" or "budget_scaled"
    lambda0 : float
        For fixed_rate prior.
    alpha : float
        For budget_scaled prior (multiplier).
    cost_scale : float
        For budget_scaled prior, k-like scaling > 0.
    """

    base_compute_budget: float = 1.0
    efficiency: float = 0.6
    prior_kind: str = "budget_scaled"
    lambda0: float = 2.0
    alpha: float = 1.0
    cost_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.base_compute_budget <= 0:
            raise ValueError("base_compute_budget must be > 0.")
        if not (0 < self.efficiency <= 1):
            raise ValueError("efficiency must be in (0, 1].")
        if self.prior_kind not in {"fixed_rate", "budget_scaled"}:
            raise ValueError("prior_kind must be 'fixed_rate' or 'budget_scaled'.")
        if self.lambda0 <= 0:
            raise ValueError("lambda0 must be > 0.")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if self.cost_scale <= 0:
            raise ValueError("cost_scale must be > 0.")

    def compute_budget_at_depth(self, depth: int) -> float:
        if depth < 0:
            raise ValueError("depth must be >= 0.")
        return float(self.base_compute_budget * (self.efficiency**depth))

    def expected_lambda_at_depth(self, depth: int) -> float:
        if self.prior_kind == "fixed_rate":
            return float(self.lambda0)
        c_d = self.compute_budget_at_depth(depth)
        return float(self.alpha * c_d / self.cost_scale)

    def simulate_once(self, max_depth: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate one branching tree realization up to max_depth.

        Returns
        -------
        dict with keys:
          - depth: [0..max_depth]
          - budgets: C_d
          - lambdas: lambda_d (expected)
          - branching: b_d sampled for d=0..max_depth-1 (last depth has no outgoing)
          - N: N_d simulations count
          - W: W_d = N_d * C_d (weight proxy)
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        rng = np.random.default_rng(seed)

        depth = np.arange(0, max_depth + 1, dtype=int)
        budgets = np.asarray([self.compute_budget_at_depth(int(d)) for d in depth], dtype=float)
        lambdas = np.asarray([self.expected_lambda_at_depth(int(d)) for d in depth], dtype=float)

        branching = np.zeros(max_depth + 1, dtype=int)
        if max_depth > 0:
            # sample b_d for d=0..max_depth-1
            branching[:-1] = rng.poisson(lambdas[:-1])
            branching[-1] = 0

        N = np.zeros(max_depth + 1, dtype=np.int64)
        N[0] = 1
        for d in range(0, max_depth):
            N[d + 1] = N[d] * int(branching[d])

        W = N.astype(float) * budgets

        return {
            "depth": depth,
            "budgets": budgets,
            "lambdas": lambdas,
            "branching": branching,
            "N": N,
            "W": W,
        }

    def monte_carlo(
        self,
        max_depth: int,
        n_trials: int,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Monte Carlo over branching realizations. Reproducible with seed.

        We use seed+i for each trial to keep determinism while allowing parallelization later.

        Returns summary arrays (mean, std) for N_d and W_d.
        """
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0.")
        Ns = []
        Ws = []
        for i in range(n_trials):
            out = self.simulate_once(max_depth=max_depth, seed=int(seed) + i)
            Ns.append(out["N"].astype(float))
            Ws.append(out["W"].astype(float))
        Nmat = np.vstack(Ns)
        Wmat = np.vstack(Ws)

        return {
            "depth": np.arange(0, max_depth + 1, dtype=int),
            "N_mean": Nmat.mean(axis=0),
            "N_std": Nmat.std(axis=0, ddof=1) if n_trials > 1 else np.zeros(max_depth + 1),
            "W_mean": Wmat.mean(axis=0),
            "W_std": Wmat.std(axis=0, ddof=1) if n_trials > 1 else np.zeros(max_depth + 1),
        }

    def probability_depth_distribution_mc(
        self,
        max_depth: int,
        n_trials: int,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a depth distribution P(d) per trial using W_d / sum W, then return (mean, std).

        Returns
        -------
        (p_mean, p_std) arrays of length max_depth+1
        """
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0.")
        ps = []
        for i in range(n_trials):
            out = self.simulate_once(max_depth=max_depth, seed=int(seed) + i)
            W = out["W"].astype(float)
            s = float(W.sum())
            if s <= 0:
                ps.append(np.zeros_like(W))
            else:
                ps.append(W / s)
        P = np.vstack(ps)
        p_mean = P.mean(axis=0)
        p_std = P.std(axis=0, ddof=1) if n_trials > 1 else np.zeros(max_depth + 1)
        return p_mean, p_std
