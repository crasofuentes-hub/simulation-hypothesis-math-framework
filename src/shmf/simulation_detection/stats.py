from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# -----------------------------
# Public API expected by tests
# -----------------------------

@dataclass(frozen=True)
class NullModel:
    """
    Test-facing null model wrapper.

    Tests expect: NullModel(kind="phase_randomized")
    """
    kind: str = "phase_randomized"

    def __post_init__(self) -> None:
        k = str(self.kind).strip().lower()
        object.__setattr__(self, "kind", k)
        if k not in {"gaussian", "phase_randomized"}:
            raise ValueError("NullModel.kind must be 'gaussian' or 'phase_randomized'.")


def spectral_peak_zscore(x: np.ndarray, freq_index: int | None = None) -> float:
    """
    Lightweight spectral-peak z-score.

    Compute rFFT power spectrum and z-score either:
      - bin `freq_index` (if provided), or
      - the max over bins 1..end (excluding DC) if freq_index is None.

    Returns a scalar float.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be 1D with length >= 8.")

    x_fft = np.fft.rfft(x)
    power = (np.abs(x_fft) ** 2).astype(float)

    if power.size <= 1:
        return 0.0

    bins = power[1:]  # exclude DC
    if bins.size == 0:
        return 0.0

    mu = float(bins.mean())
    sd = float(bins.std(ddof=1)) if bins.size > 1 else 0.0
    if sd <= 0.0:
        return 0.0

    if freq_index is None:
        val = float(bins.max())
    else:
        idx = int(freq_index)
        if idx < 0 or idx >= power.size:
            raise ValueError("freq_index out of range for rFFT bins.")
        if idx == 0:
            val = mu
        else:
            val = float(power[idx])

    return float((val - mu) / sd)


def zscore_to_p_two_sided(z: float | np.ndarray) -> float | np.ndarray:
    """
    Two-sided p-value from z.

    Tests call this with an array, so this is vectorized.
    """
    z_arr = np.asarray(z, dtype=float)
    p = 2.0 * norm.sf(np.abs(z_arr))
    p = np.clip(p, 0.0, 1.0)

    # Preserve scalar-in/scalar-out behavior
    if np.ndim(z) == 0:
        return float(p)
    return np.asarray(p, dtype=float)


def empirical_p_value(
    z_obs: float,
    null_z: np.ndarray,
    *,
    alternative: str = "greater",
) -> float:
    """
    Empirical p-value with +1 smoothing:
      p = (k + 1) / (n + 1)

    Guarantees p in (0, 1].
    """
    null_z = np.asarray(null_z, dtype=float)
    if null_z.ndim != 1 or null_z.size == 0:
        raise ValueError("null_z must be a non-empty 1D array.")

    z = float(z_obs)
    alt = str(alternative).strip().lower()

    if alt == "greater":
        k = int(np.sum(null_z >= z))
    elif alt == "less":
        k = int(np.sum(null_z <= z))
    elif alt in {"two-sided", "two_sided", "two sided"}:
        k = int(np.sum(np.abs(null_z) >= abs(z)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    n = int(null_z.size)
    p = (k + 1.0) / (n + 1.0)

    # Hard clamp for numerical safety
    if p <= 0.0:
        return 1.0 / (n + 1.0)
    if p > 1.0:
        return 1.0
    return float(p)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    BH-FDR rejection mask ONLY.

    Tests expect:
      rej = benjamini_hochberg(...)
      assert rej.dtype == bool
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1D.")

    m = int(p.size)
    if m == 0:
        return np.asarray([], dtype=bool)

    p = np.clip(p, 0.0, 1.0)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    thresh = (float(alpha) * ranks) / float(m)
    passed = p_sorted <= thresh

    reject_sorted = np.zeros(m, dtype=bool)
    if np.any(passed):
        kmax = int(np.max(np.where(passed)[0]))
        cutoff = float(p_sorted[kmax])
        reject_sorted = p_sorted <= cutoff

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.astype(bool)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    BH-adjusted q-values (not part of the public test API, but used by detect_spectral_artifacts).
    """
    p = np.asarray(pvals, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    m = int(p.size)
    if m == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    q_sorted = (p_sorted * float(m)) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.zeros(m, dtype=float)
    q[order] = q_sorted
    return q


def power_two_sample_normal(
    *,
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Power for a two-sample z-test, sigma normalized to 1.

    Tests call:
      power_two_sample_normal(effect_size=..., alpha=..., n=..., two_sided=...)
    """
    nn = int(n)
    if nn <= 0:
        raise ValueError("n must be > 0.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    se = np.sqrt(2.0 / float(nn))
    mu = float(effect_size) / float(se)

    if two_sided:
        zcrit = float(norm.isf(float(alpha) / 2.0))
        power = float(norm.sf(zcrit - mu) + norm.cdf(-zcrit - mu))
    else:
        zcrit = float(norm.isf(float(alpha)))
        power = float(norm.sf(zcrit - mu))

    return float(np.clip(power, 0.0, 1.0))


@dataclass(frozen=True)
class DetectionResult:
    idxs: np.ndarray
    z_obs: np.ndarray
    pvals: np.ndarray
    qvals: np.ndarray
    reject: np.ndarray
    null_z: np.ndarray


def _null_surrogates(x: np.ndarray, n_null: int, null_model: NullModel, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)

    if int(n_null) <= 0:
        raise ValueError("n_null must be > 0.")

    if null_model.kind == "gaussian":
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        if sd <= 0.0:
            return np.tile(np.full_like(x, mu, dtype=float), (int(n_null), 1))
        return rng.normal(loc=mu, scale=sd, size=(int(n_null), x.size)).astype(float)

    if null_model.kind == "phase_randomized":
        x_fft = np.fft.rfft(x)
        mag = np.abs(x_fft)
        out = np.zeros((int(n_null), x.size), dtype=float)

        for i in range(int(n_null)):
            phase = rng.uniform(0.0, 2.0 * np.pi, size=mag.size)
            phase[0] = 0.0
            if mag.size > 1:
                phase[-1] = 0.0
            x_null_fft = mag * np.exp(1j * phase)
            out[i, :] = np.fft.irfft(x_null_fft, n=x.size)

        return out

    raise ValueError(f"Unknown null model kind: {null_model.kind!r}")


def detect_spectral_artifacts(
    x: np.ndarray,
    freq_indices: Iterable[int] | None = None,
    *,
    null_model: NullModel | None = None,
    n_null: int = 256,
    alpha: float = 0.05,
    seed: int = 0,
) -> DetectionResult:
    """
    Orchestrator: null model -> empirical p-values -> BH reject mask (+ q-values).

    Tests require it to run with:
      nm = NullModel(kind="phase_randomized")
      detect_spectral_artifacts(x, null_model=nm, ...)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be 1D with length >= 8.")

    nm = null_model if null_model is not None else NullModel(kind="phase_randomized")

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

    sur = _null_surrogates(x=x, n_null=int(n_null), null_model=nm, seed=int(seed))
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

    reject = benjamini_hochberg(pvals, alpha=float(alpha))
    qvals = _bh_qvalues(pvals)

    return DetectionResult(
        idxs=idxs,
        z_obs=z_obs,
        pvals=pvals,
        qvals=qvals,
        reject=reject,
        null_z=null_z,
    )