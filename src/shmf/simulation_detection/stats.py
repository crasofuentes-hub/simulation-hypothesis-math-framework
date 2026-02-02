from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class NullModel(str, Enum):
    """Null models used for artifact detection."""

    GAUSSIAN = "gaussian"
    PHASE_RANDOMIZED = "phase_randomized"


def spectral_peak_zscore(x: np.ndarray, freq_index: int | None = None) -> float:
    """
    Lightweight spectral-peak z-score.

    We compute power spectrum via rFFT, then z-score either:
    - the chosen bin `freq_index`, or
    - the max over bins 1..end (excluding DC) if freq_index is None.

    Returns a scalar z-score.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be 1D with length >= 8.")

    x_fft = np.fft.rfft(x)
    power = (np.abs(x_fft) ** 2).astype(float)

    if power.size <= 1:
        return 0.0  # degenerate

    # Exclude DC
    bins = power[1:]
    if bins.size == 0:
        return 0.0

    mu = float(bins.mean())
    sd = float(bins.std(ddof=1)) if bins.size > 1 else 0.0
    if sd <= 0:
        return 0.0

    if freq_index is None:
        val = float(bins.max())
    else:
        idx = int(freq_index)
        if idx < 0 or idx >= power.size:
            raise ValueError("freq_index out of range for rFFT bins.")
        if idx == 0:
            # DC explicitly discouraged; treat as not-an-artifact here
            val = float(mu)
        else:
            val = float(power[idx])

    return float((val - mu) / sd)


def zscore_to_p_two_sided(z: float) -> float:
    """Two-sided p-value from z using SciPy's normal survival function."""
    z = float(z)
    p = 2.0 * float(norm.sf(abs(z)))
    # Clamp strictly into [0,1]
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)


def empirical_p_value(
    z_obs: float,
    null_z: np.ndarray,
    *,
    alternative: str = "greater",
) -> float:
    """
    Empirical p-value with +1 smoothing:
        p = (count_more_extreme + 1) / (n + 1)

    Ensures p in (0,1], avoids 0 and avoids >1.
    """
    null_z = np.asarray(null_z, dtype=float)
    if null_z.ndim != 1 or null_z.size == 0:
        raise ValueError("null_z must be a non-empty 1D array.")

    z = float(z_obs)
    alt = str(alternative).lower().strip()

    if alt == "greater":
        k = int(np.sum(null_z >= z))
    elif alt == "less":
        k = int(np.sum(null_z <= z))
    elif alt in ("two-sided", "two_sided", "two sided"):
        k = int(np.sum(np.abs(null_z) >= abs(z)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    n = int(null_z.size)
    p = (k + 1.0) / (n + 1.0)

    # numeric guard
    if p <= 0.0:
        return 1.0 / (n + 1.0)
    if p > 1.0:
        return 1.0
    return float(p)


def benjamini_hochberg(pvals: Iterable[float], alpha: float = 0.05) -> tuple[list[bool], np.ndarray]:
    """
    Benjamini–Hochberg (BH) FDR procedure.

    Returns:
      reject: list[bool]  (python bools to satisfy tests using `is True/False`)
      qvals : np.ndarray  (BH-adjusted q-values, same order as input)
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    p = np.asarray(list(pvals), dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1D.")
    m = int(p.size)
    if m == 0:
        return ([], np.asarray([], dtype=float))

    # clamp p into [0,1]
    p = np.clip(p, 0.0, 1.0)

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # BH threshold line
    thresh = (float(alpha) * ranks) / float(m)

    # Find largest k where p_k <= (k/m)*alpha
    passed = p_sorted <= thresh
    if np.any(passed):
        kmax = int(np.max(np.where(passed)[0]))
        cutoff = float(p_sorted[kmax])
        reject_sorted = p_sorted <= cutoff
    else:
        reject_sorted = np.zeros(m, dtype=bool)

    # q-values: p_i * m / rank_i, then enforce monotonicity from the end
    q_sorted = (p_sorted * float(m)) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # return to original order
    reject = np.zeros(m, dtype=bool)
    qvals = np.zeros(m, dtype=float)
    reject[order] = reject_sorted
    qvals[order] = q_sorted

    # IMPORTANT: list of python bools (not np.bool_)
    return ([bool(x) for x in reject.tolist()], qvals)


def power_two_sample_normal(
    effect: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Power for a two-sample z-test for difference in means, assuming:
      - equal variance,
      - known/normalized sigma = 1,
      - effect = (mu1 - mu2) / sigma.

    Standard error for difference: sqrt(2/n).
    Under H1: Z ~ N(effect / se, 1).

    Returns power in [0,1].
    """
    n = int(n_per_group)
    if n <= 0:
        raise ValueError("n_per_group must be > 0.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    se = np.sqrt(2.0 / float(n))
    mu = float(effect) / float(se)

    if two_sided:
        zcrit = float(norm.isf(float(alpha) / 2.0))
        # Power = P(|Z| >= zcrit | Z ~ N(mu,1))
        power = float(norm.sf(zcrit - mu) + norm.cdf(-zcrit - mu))
    else:
        zcrit = float(norm.isf(float(alpha)))
        power = float(norm.sf(zcrit - mu))

    # clamp
    if power < 0.0:
        return 0.0
    if power > 1.0:
        return 1.0
    return float(power)


@dataclass(frozen=True)
class DetectionResult:
    idxs: np.ndarray
    z_obs: np.ndarray
    pvals: np.ndarray
    qvals: np.ndarray
    reject: list[bool]
    null_z: np.ndarray


def _null_surrogates(
    x: np.ndarray,
    n_null: int,
    null_model: NullModel,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)

    if n_null <= 0:
        raise ValueError("n_null must be > 0.")

    if null_model == NullModel.GAUSSIAN:
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        if sd <= 0.0:
            return np.tile(np.full_like(x, mu, dtype=float), (n_null, 1))
        return rng.normal(loc=mu, scale=sd, size=(n_null, x.size)).astype(float)

    if null_model == NullModel.PHASE_RANDOMIZED:
        x_fft = np.fft.rfft(x)
        mag = np.abs(x_fft)
        out = np.zeros((n_null, x.size), dtype=float)
        for i in range(n_null):
            phase = rng.uniform(0.0, 2.0 * np.pi, size=mag.size)
            phase[0] = 0.0
            if mag.size > 1:
                phase[-1] = 0.0
            x_null_fft = mag * np.exp(1j * phase)
            out[i, :] = np.fft.irfft(x_null_fft, n=x.size)
        return out

    raise ValueError(f"Unknown null_model: {null_model!r}")


def detect_spectral_artifacts(
    x: np.ndarray,
    freq_indices: Iterable[int] | None = None,
    *,
    null_model: NullModel = NullModel.PHASE_RANDOMIZED,
    n_null: int = 256,
    alpha: float = 0.05,
    seed: int = 0,
) -> DetectionResult:
    """
    Detect candidate spectral artifacts with:
      - a chosen null model,
      - empirical p-values (one-sided 'greater'),
      - BH-FDR control.

    Returns DetectionResult.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if x.size < 8:
        raise ValueError("x must have at least 8 samples.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    if freq_indices is None:
        n_bins = np.fft.rfft(x).size
        hi = min(16, n_bins - 1)
        if hi < 1:
            raise ValueError("signal too short for spectral testing.")
        idxs = np.arange(1, hi + 1, dtype=int)
    else:
        idxs = np.asarray(list(freq_indices), dtype=int)
        if idxs.size == 0:
            raise ValueError("freq_indices must be non-empty.")
        if np.any(idxs < 0):
            raise ValueError("freq_indices must be >= 0.")

    z_obs = np.asarray([spectral_peak_zscore(x, int(i)) for i in idxs], dtype=float)

    sur = _null_surrogates(x=x, n_null=int(n_null), null_model=null_model, seed=int(seed))
    null_z = np.zeros((idxs.size, int(n_null)), dtype=float)

    for j in range(int(n_null)):
        null_z[:, j] = np.asarray(
            [spectral_peak_zscore(sur[j, :], int(i)) for i in idxs],
            dtype=float,
        )

    pvals = np.asarray(
        [empirical_p_value(z_obs[i], null_z[i, :], alternative="greater") for i in range(idxs.size)],
        dtype=float,
    )

    reject, qvals = benjamini_hochberg(pvals, alpha=float(alpha))
    return DetectionResult(
        idxs=idxs,
        z_obs=z_obs,
        pvals=pvals,
        qvals=qvals,
        reject=reject,
        null_z=null_z,
    )