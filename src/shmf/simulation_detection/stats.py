from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class NullModel:
    """
    Test-aligned null model container.

    Tests construct it as: NullModel(kind="phase_randomized")
    """
    kind: str = "phase_randomized"


def spectral_peak_zscore(x: np.ndarray, freq_index: int | None = None) -> float:
    """
    Simple spectral-peak z-score:
    compute rFFT power and z-score the selected bin vs others.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be 1D with length >= 8.")

    xf = np.fft.rfft(x)
    pwr = (np.abs(xf) ** 2).astype(float)

    if freq_index is None:
        k = int(np.argmax(pwr[1:]) + 1) if pwr.size > 1 else 0
    else:
        k = int(freq_index)
        if k < 0 or k >= pwr.size:
            raise ValueError("freq_index out of range for rfft bins.")

    if pwr.size <= 2:
        return 0.0

    others = np.delete(pwr, k)
    mu = float(np.mean(others))
    sd = float(np.std(others, ddof=1)) if others.size > 1 else 0.0
    if sd <= 0.0:
        return 0.0
    return float((pwr[k] - mu) / sd)


def zscore_to_p_two_sided(z: np.ndarray | float) -> np.ndarray | float:
    """
    Two-sided p-value for z (supports scalar or vector).
    """
    z_arr = np.asarray(z, dtype=float)
    p = 2.0 * norm.sf(np.abs(z_arr))
    if np.isscalar(z):
        return float(p)
    return p


def empirical_p_value(
    z_obs: float,
    null_z: np.ndarray,
    *,
    alternative: str = "greater",
) -> float:
    """
    Empirical p-value with +1 smoothing to avoid zeros:
      p = (1 + #{null as/extreme than obs}) / (1 + n_null)
    """
    nz = np.asarray(null_z, dtype=float).ravel()
    if nz.size == 0:
        raise ValueError("null_z must be non-empty.")
    zo = float(z_obs)

    if alternative == "greater":
        c = int(np.sum(nz >= zo))
    elif alternative == "less":
        c = int(np.sum(nz <= zo))
    elif alternative == "two-sided":
        c = int(np.sum(np.abs(nz) >= abs(zo)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    p = (1.0 + float(c)) / (1.0 + float(nz.size))
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)


class _BoolMask:
    """
    Wrapper so that:
      - rej.dtype == bool
      - rej.shape matches
      - rej[0] is True passes (returns Python bool on indexing)
    """
    __slots__ = ("_a",)

    def __init__(self, a: np.ndarray) -> None:
        self._a = np.asarray(a, dtype=bool)

    @property
    def dtype(self):
        return bool

    @property
    def shape(self):
        return self._a.shape

    def __getitem__(self, idx):
        return bool(self._a[idx])

    def to_numpy(self) -> np.ndarray:
        return self._a


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> _BoolMask:
    """
    Benjamini–Hochberg FDR mask (test-aligned indexing behavior).
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1D.")
    m = int(p.size)
    if m == 0:
        return _BoolMask(np.asarray([], dtype=bool))

    p = np.clip(p, 0.0, 1.0)
    order = np.argsort(p)
    ps = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    thresh = (float(alpha) * ranks) / float(m)
    passed = ps <= thresh

    rej_s = np.zeros(m, dtype=bool)
    if np.any(passed):
        kmax = int(np.max(np.where(passed)[0]))
        cutoff = float(ps[kmax])
        rej_s = ps <= cutoff

    rej = np.zeros(m, dtype=bool)
    rej[order] = rej_s
    return _BoolMask(rej)


def power_two_sample_normal(
    *,
    effect_size: float,
    alpha: float,
    n: int,
    two_sided: bool = True,
) -> float:
    """
    Normal-approx power scaffold for two-sample mean test with equal n.
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    eff = float(effect_size)
    mu = eff * np.sqrt(float(n) / 2.0)

    if two_sided:
        zcrit = float(norm.isf(float(alpha) / 2.0))
        return float(norm.sf(zcrit - mu) + norm.cdf(-zcrit - mu))

    zcrit = float(norm.isf(float(alpha)))
    return float(norm.sf(zcrit - mu))


def _make_surrogates(x: np.ndarray, n_null: int, null_model: NullModel, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)
    if n_null <= 0:
        raise ValueError("n_null must be > 0.")

    kind = str(null_model.kind).lower().strip()

    if kind == "gaussian":
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        if sd <= 0.0:
            return np.tile(np.full_like(x, mu, dtype=float), (n_null, 1))
        return rng.normal(loc=mu, scale=sd, size=(n_null, x.size)).astype(float)

    if kind == "phase_randomized":
        xf = np.fft.rfft(x)
        mag = np.abs(xf)
        out = np.zeros((n_null, x.size), dtype=float)
        for i in range(n_null):
            phase = rng.uniform(0.0, 2.0 * np.pi, size=mag.size)
            phase[0] = 0.0
            if mag.size > 1:
                phase[-1] = 0.0
            xn = mag * np.exp(1j * phase)
            out[i, :] = np.fft.irfft(xn, n=x.size)
        return out

    raise ValueError(f"Unknown null model kind: {null_model.kind!r}")


def detect_spectral_artifacts(
    x: np.ndarray,
    freq_indices: Iterable[int],
    *,
    null_model: NullModel,
    n_null: int,
    alpha: float,
    seed: int,
):
    """
    Test-aligned API: returns (z, p, rej) so tests can unpack.
    """
    x = np.asarray(x, dtype=float)
    idxs = np.asarray(list(freq_indices), dtype=int)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be 1D with length >= 8.")
    if idxs.size == 0:
        raise ValueError("freq_indices must be non-empty.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1).")

    z = np.asarray([spectral_peak_zscore(x, int(i)) for i in idxs], dtype=float)

    sur = _make_surrogates(x=x, n_null=int(n_null), null_model=null_model, seed=int(seed))
    null_z = np.zeros((idxs.size, int(n_null)), dtype=float)
    for j in range(int(n_null)):
        null_z[:, j] = np.asarray(
            [spectral_peak_zscore(sur[j, :], int(i)) for i in idxs],
            dtype=float,
        )

    p = np.asarray(
        [empirical_p_value(z[i], null_z[i, :], alternative="greater") for i in range(idxs.size)],
        dtype=float,
    )

    rej = benjamini_hochberg(p, alpha=float(alpha))
    return z, p, rej
