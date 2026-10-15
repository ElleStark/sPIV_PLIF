"""Cospectra and coherence of velocity and scalar fluctuations at single points (step 7).

Estimator (Welch): Hann window, segments of ``nperseg`` samples with 50% overlap, each segment linearly detrended
(removes slow drift within a segment; the record mean is removed by the detrend as well), one-sided densities.
    Co_qc(f) = Re S_qc(f),   integral of Co_qc over f = <q'c'> (up to the detrend and window bias)
    gamma^2_qc(f) = |S_qc|^2 / (S_qq S_cc)
Confidence intervals of the cospectrum come from the scatter of the per-segment cospectra:
    CI = mean +/- t * std_seg * sqrt(f_ov / K), K segments, f_ov = 1.056 the variance inflation for Hann windows at
    50% overlap (neighbouring segment estimates are correlated).
Coherence significance (zero true coherence) at level alpha: gamma^2 > 1 - alpha^(1 / (K_eff - 1)), K_eff = K / f_ov.

Gaps (NaN) are filled by linear interpolation before the transform; the fraction filled is returned so heavily
gapped series can be rejected. Sampling is not anti-aliased (PIV/PLIF frames), so energy above Nyquist folds back;
spectra near Nyquist should be read with that in mind.

Frequency-dependent effective diffusivity at a point with mean gradient g:
    K(f) df = -Co_uc(f) df / g, and the cumulative K(< f) = -int_0^f Co_uc / g, which ends at the pointwise K.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats

OVERLAP_VAR_FACTOR = 1.056  # Hann, 50% overlap


def fill_gaps(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    if fin.all() or not fin.any():
        return x, float(1 - fin.mean())
    idx = np.arange(x.size)
    return np.interp(idx, idx[fin], x[fin]), float(1 - fin.mean())


def cospectrum(q: np.ndarray, c: np.ndarray, fs: float, *, nperseg: int = 256, ci: float = 0.95) -> dict[str, np.ndarray]:
    """Welch cospectrum with per-segment CI, quadrature spectrum, auto-spectra and coherence."""
    q, fq = fill_gaps(q)
    c, fc = fill_gaps(c)
    kw = dict(fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="linear")
    f, Sqc = signal.csd(q, c, **kw)
    _, Sqq = signal.welch(q, **kw)
    _, Scc = signal.welch(c, **kw)
    # per-segment cospectra for the CI (same segmentation as Welch)
    step = nperseg // 2
    starts = range(0, q.size - nperseg + 1, step)
    segs = []
    for s0 in starts:
        _, s_qc = signal.csd(q[s0 : s0 + nperseg], c[s0 : s0 + nperseg], fs=fs, window="hann", nperseg=nperseg,
                             noverlap=0, detrend="linear")
        segs.append(s_qc.real)
    segs = np.array(segs)
    K = segs.shape[0]
    t = stats.t.ppf(0.5 + ci / 2, df=max(1, K - 1))
    half = t * segs.std(axis=0, ddof=1) * np.sqrt(OVERLAP_VAR_FACTOR / K)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(Sqc) ** 2 / (Sqq * Scc)
    K_eff = K / OVERLAP_VAR_FACTOR
    coh_sig = 1 - (1 - ci) ** (1 / max(1.0, K_eff - 1))
    return {"f": f, "co": Sqc.real, "quad": Sqc.imag, "co_ci": half, "Sqq": Sqq, "Scc": Scc, "coh": coh,
            "coh_sig": coh_sig, "n_segments": K, "gap_fraction": max(fq, fc)}


def cumulative_K(f: np.ndarray, co: np.ndarray, grad: float) -> np.ndarray:
    """K(< f) = -int_0^f Co(f') df' / grad (trapezoid)."""
    df = np.diff(f)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (co[1:] + co[:-1]) * df)])
    return -cum / grad
