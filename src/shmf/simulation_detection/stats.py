from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Hochberg (BH) procedure controlling FDR.

    Returns a boolean mask of discoveries (reject H0).
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1D array.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p)
    p_sorted = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p_sorted <= thresh
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool)

    k_max = int(np.max(np.where(passed)[0]))
    cutoff = p_sorted[k_max]
    return p <= cutoff


def zscore_to_p_two_sided(z: np.ndarray) -> np.ndarray:
    """
    Two-sided p-values from z-scores using normal approximation.

    Uses erfc for numerical stability.
    """
    z = np.asarray(z, dtype=float)
    # p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))
    return np.erfc(np.abs(z) / np.sqrt(2.0))


def spectral_peak_zscore(x: np.ndarray, freq_index: Optional[int] = None) -> float:
    """
    Lightweight spectral-peak z-score (existing-style primitive).

    Computes magnitude spectrum |FFT(x)|. If freq_index is None, uses the max bin (excluding DC).
    z = (peak - mean) / std of magnitudes.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if x.size < 8:
        raise ValueError("x must have length >= 8.")

    spec = np.abs(np.fft.rfft(x))
    if spec.size <= 2:
        raise ValueError("Insufficient spectrum length.")
    # exclude DC bin
    spec_nd = spec.copy()
    spec_nd[0] = 0.0

    if freq_index is None:
        idx = int(np.argmax(spec_nd))
    else:
        if not (0 <= freq_index < spec_nd.size):
            raise ValueError("freq_index out of range.")
        idx = int(freq_index)

    peak = float(spec_nd[idx])
    mu = float(np.mean(spec_nd))
    sigma = float(np.std(spec_nd, ddof=1))
    if sigma == 0.0:
        return 0.0
    return (peak - mu) / sigma


@dataclass(frozen=True)
class NullModel:
    """
    Null model generator for artifact detection.

    kind:
      - "white": Gaussian white noise N(0,1)
      - "phase_randomized": preserves amplitude spectrum of x, randomizes phases
    """

    kind: str = "white"

    def __post_init__(self) -> None:
        if self.kind not in {"white", "phase_randomized"}:
            raise ValueError("kind must be 'white' or 'phase_randomized'.")

    def sample(self, x: np.ndarray, seed: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(seed)
        if self.kind == "white":
            return rng.normal(0.0, 1.0, size=x.size)

        # phase_randomized: keep rfft magnitudes, randomize phases
        X = np.fft.rfft(x)
        mag = np.abs(X)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=mag.size)
        # keep DC and Nyquist (if present) real
        phase[0] = 0.0
        if mag.size > 1:
            phase[-1] = 0.0
        Xn = mag * np.exp(1j * phase)
        xn = np.fft.irfft(Xn, n=x.size)
        return np.asarray(xn, dtype=float)


def empirical_p_value(
    statistic: float,
    null_statistics: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Empirical p-value from null distribution.

    alternative:
      - "greater": p = P(T >= t_obs)
      - "less":    p = P(T <= t_obs)
      - "two-sided": p = 2 * min(P(T>=t), P(T<=t))
    """
    t = float(statistic)
    null = np.asarray(null_statistics, dtype=float)
    if null.ndim != 1 or null.size == 0:
        raise ValueError("null_statistics must be a non-empty 1D array.")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("Invalid alternative.")
    # add +1 smoothing (conservative) to avoid 0 p-values
    n = null.size
    if alternative == "greater":
        return float((np.sum(null >= t) + 1) / (n + 1))
    if alternative == "less":
        return float((np.sum(null <= t) + 1) / (n + 1))
    pg = float((np.sum(null >= t) + 1) / (n + 1))
    pl = float((np.sum(null <= t) + 1) / (n + 1))
    return float(2.0 * min(pg, pl))


def detect_spectral_artifacts(
    x: np.ndarray,
    freq_indices: Iterable[int],
    null_model: NullModel,
    n_null: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute p-values for spectral peak z-scores at specified freq indices under a chosen null model,
    then apply BH-FDR.

    Returns
    -------
    z_obs: observed z-scores per freq index
    pvals: empirical p-values per freq index
    reject: BH discoveries boolean mask
    """
    x = np.asarray(x, dtype=float)
    idxs = np.asarray(list(freq_indices), dtype=int)
    if idxs.ndim != 1 or idxs.size == 0:
        raise ValueError("freq_indices must be non-empty.")
    if n_null <= 0:
        raise ValueError("n_null must be > 0.")

    z_obs = np.asarray([spectral_peak_zscore(x, int(i)) for i in idxs], dtype=float)

    # null distribution per index by resampling
    null_z = np.zeros((idxs.size, n_null), dtype=float)
    for j in range(n_null):
        xn = null_model.sample(x, seed=seed + j)
        null_z[:, j] = np.asarray([spectral_peak_zscore(xn, int(i)) for i in idxs], dtype=float)

    pvals = np.asarray(
        [empirical_p_value(z_obs[i], null_z[i, :], alternative="greater") for i in range(idxs.size)],
        dtype=float,
    )

    reject = benjamini_hochberg(pvals, alpha=alpha)
    return z_obs, pvals, reject


def power_two_sample_normal(
    effect_size: float,
    alpha: float = 0.05,
    n: int = 100,
    two_sided: bool = True,
) -> float:
    """
    Approximate power for detecting a mean shift in Normal data with known variance=1
    using a z-test approximation.

    effect_size: delta / sigma, Cohen's d when sigma=1
    n: sample size

    This is a simple scaffolding utility for planning.
    """
    if n <= 1:
        raise ValueError("n must be > 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    d = float(effect_size)
    # standard error
    se = 1.0 / np.sqrt(n)
    # critical z
    if two_sided:
        zcrit = np.sqrt(2.0) * np.erfcinv(alpha)  # approx inv for two-sided
    else:
        zcrit = np.sqrt(2.0) * np.erfcinv(2.0 * alpha)
    # noncentral shift
    mu = d / se
    # power approx: P(|Z+mu| > zcrit)
    # = P(Z > zcrit - mu) + P(Z < -zcrit - mu)
    # Use erfc for tails
    def tail_gt(a: float) -> float:
        return float(0.5 * np.erfc(a / np.sqrt(2.0)))

    def cdf_lt(a: float) -> float:
        return float(0.5 * np.erfc(-a / np.sqrt(2.0)))

    return tail_gt(zcrit - mu) + cdf_lt(-zcrit - mu)
