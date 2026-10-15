"""Statistical convergence and uncertainty of time-averaged fluxes (step 2).

Integral time scale
    rho(tau) = autocorrelation coefficient of a fluctuation series, estimated with the biased (1/N) estimator,
    which is the unbiased pair-mean multiplied by the triangular taper (N - k)/N (smoothly tapered to zero at
    the record length, so no post-decay wiggles are amplified). NaN samples are handled with pair counts.
    T_int = integral of rho from 0 to its first zero crossing (crossing linearly interpolated), the same
    definition used for the integral length scales in ILS_mpi_from_npy.py.

Block bootstrap
    The time mean of p_t = q'_t c'_t is resampled with a moving-block bootstrap (blocks of length L_b, drawn
    with replacement with uniformly random start, concatenated to the record length). L_b is chosen from the
    integral time scale (default 2 T_int), so blocks are approximately independent; an iid bootstrap would
    understate the uncertainty for time-correlated data. Block sums are precomputed per base block, so each
    bootstrap replicate is a linear combination of base blocks: F*_r = (M @ P) / (M @ n), M = multiplicities.
    The fixed-frame means Q, C are held at their full-record values (an O(1/N) approximation).

Effective number of independent samples
    N_eff = var(p) / SE^2 (from the bootstrap), and N_eff = T_record / (2 T_int) (from the time scale).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz is deprecated in NumPy 2.x


def autocorrelation(x: np.ndarray, *, max_lag: int | None = None) -> np.ndarray:
    """Autocorrelation coefficient along axis 0 of x (nt, ...), NaN-aware, biased (tapered) estimator.

    rho(k) = [sum_t a_t a_{t+k} / n_pairs(k)] * (N - k)/N / var, a = x - mean, using only valid pairs.
    Returns (max_lag + 1, ...).
    """
    x = np.asarray(x, dtype=np.float64)
    nt = x.shape[0]
    max_lag = nt - 1 if max_lag is None else min(max_lag, nt - 1)
    m = np.isfinite(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(x, axis=0) / m.sum(axis=0)
    a = np.where(m, x - mean, 0.0)
    mf = m.astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(2 * nt - 1)))  # zero padding: linear (not circular) correlation
    fa = np.fft.rfft(a, n=nfft, axis=0)
    fm = np.fft.rfft(mf, n=nfft, axis=0)
    cov = np.fft.irfft(fa * np.conj(fa), n=nfft, axis=0)[: max_lag + 1]
    npairs = np.rint(np.fft.irfft(fm * np.conj(fm), n=nfft, axis=0)[: max_lag + 1])
    taper = ((nt - np.arange(max_lag + 1)) / nt).reshape((-1,) + (1,) * (x.ndim - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(npairs > 0, cov / npairs, np.nan) * taper
        rho = r / r[0]
    return rho


def integral_scale(rho: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate rho (along axis 0) to its first zero crossing (trapezoid, crossing interpolated).

    Returns (T, crossed): T has rho.shape[1:]; crossed is False where rho never reached zero (T is then the
    integral over all available lags, i.e. a lower bound limited by the record).
    """
    rho = np.asarray(rho, dtype=np.float64)
    flat = rho.reshape(rho.shape[0], -1)
    T = np.full(flat.shape[1], np.nan)
    crossed = np.zeros(flat.shape[1], dtype=bool)
    for j in range(flat.shape[1]):
        f = flat[:, j]
        fin = np.isfinite(f)
        if not fin[0]:
            continue
        n_fin = f.size if fin.all() else int(np.argmin(fin))
        f = f[:n_fin]
        below = np.flatnonzero(f <= 0)
        if below.size == 0:
            T[j] = trapz(f, dx=d) if f.size > 1 else 0.0
            continue
        k = below[0]
        frac = f[k - 1] / (f[k - 1] - f[k])
        T[j] = (trapz(f[:k], dx=d) if k > 1 else 0.0) + 0.5 * f[k - 1] * frac * d
        crossed[j] = True
    return T.reshape(rho.shape[1:]), crossed.reshape(rho.shape[1:])


def product_series(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """p_t = (q - Q)(c - C) along axis 0 with pairwise-valid means (NaN where either is invalid)."""
    m = np.isfinite(q) & np.isfinite(c)
    qn = np.where(m, q, np.nan)
    cn = np.where(m, c, np.nan)
    with np.errstate(invalid="ignore"):
        return (qn - np.nanmean(qn, axis=0)) * (cn - np.nanmean(cn, axis=0))


def integral_time_scales(series: np.ndarray, dt: float, *, max_lag: int | None = None, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Integral time scale of each column of series (nt, ...) in seconds; processed in column chunks."""
    nt = series.shape[0]
    flat = series.reshape(nt, -1)
    T = np.full(flat.shape[1], np.nan)
    crossed = np.zeros(flat.shape[1], dtype=bool)
    for s in range(0, flat.shape[1], chunk):
        rho = autocorrelation(flat[:, s : s + chunk], max_lag=max_lag)
        T[s : s + chunk], crossed[s : s + chunk] = integral_scale(rho, dt)
    return T.reshape(series.shape[1:]), crossed.reshape(series.shape[1:])


def choose_block_length(
    T_int: np.ndarray, region: np.ndarray, *, dt: float, base_block_frames: int, n_base_blocks: int,
    factor: float = 2.0, percentile: float = 95.0, min_blocks: int = 10,
) -> tuple[int, float]:
    """Block length (in base blocks) = factor * percentile(T_int over region), rounded up to whole base blocks,
    capped so that at least ``min_blocks`` blocks fit in the record. Returns (k_base_blocks, T_used [s])."""
    vals = T_int[region & np.isfinite(T_int)]
    if vals.size == 0:
        raise ValueError("No finite integral time scales in the selected region")
    T_used = float(np.percentile(vals, percentile))
    k = int(np.ceil(factor * T_used / (dt * base_block_frames)))
    k = max(1, min(k, n_base_blocks // min_blocks))
    return k, T_used


def moving_block_multiplicities(n_base: int, k: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_base) counts of how often each base block appears in each moving-block-bootstrap replicate
    (blocks of k consecutive base blocks, random starts, enough blocks to cover n_base)."""
    n_draw = -(-n_base // k)
    starts = rng.integers(0, n_base - k + 1, size=(n_boot, n_draw))
    M = np.zeros((n_boot, n_base), dtype=np.float64)
    for off in range(k):
        np.add.at(M, (np.repeat(np.arange(n_boot), n_draw), (starts + off).ravel()), 1.0)
    return M


def bootstrap_flux_se(
    block_paths: dict[str, Path], Qs: np.ndarray, Cs: np.ndarray, q: str, M: np.ndarray, *,
    row_chunk: int = 64, ci: tuple[float, float] = (2.5, 97.5),
) -> dict[str, np.ndarray]:
    """Bootstrap standard error (and percentile CI half-width) of <q'c'> per pixel from base-block sums.

    Per base block b (shifted variables, full-record pair means Qs, Cs held fixed):
        P_b = sum (q - Qs)(c - Cs) = Sqc_b - Cs Sq_b - Qs Sc_b + n_b Qs Cs,   F*_r = sum_b M_rb P_b / sum_b M_rb n_b
    """
    mm = {f: np.load(block_paths[f"{q}_{f}"], mmap_mode="r") for f in ("n", "sq", "sc", "sqc")}
    _, ny, nx = mm["n"].shape
    se = np.full((ny, nx), np.nan)
    half_ci = np.full((ny, nx), np.nan)
    for r0 in range(0, ny, row_chunk):
        r1 = min(r0 + row_chunk, ny)
        n = np.asarray(mm["n"][:, r0:r1], dtype=np.float64)
        Q, C = Qs[r0:r1][None], Cs[r0:r1][None]
        P = np.asarray(mm["sqc"][:, r0:r1], dtype=np.float64) - C * mm["sq"][:, r0:r1] - Q * mm["sc"][:, r0:r1] + n * Q * C
        nb = n.shape[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            F = (M @ np.nan_to_num(P.reshape(nb, -1))) / (M @ n.reshape(nb, -1))
        se[r0:r1] = np.nanstd(F, axis=0, ddof=1).reshape(r1 - r0, nx)
        lo, hi = np.nanpercentile(F, ci, axis=0)
        half_ci[r0:r1] = (0.5 * (hi - lo)).reshape(r1 - r0, nx)
    return {"se": se, "ci_halfwidth": half_ci}


def subset_flux_from_blocks(
    block_paths: dict[str, Path], q: str, blocks: slice, *, row_chunk: int = 128
) -> np.ndarray:
    """<q'c'> from a contiguous subset of base blocks (e.g. first/second half of the record), with the means
    recomputed over that subset (an exact fixed-frame decomposition of the sub-record)."""
    mm = {f: np.load(block_paths[f"{q}_{f}"], mmap_mode="r") for f in ("n", "sq", "sc", "sqc")}
    _, ny, nx = mm["n"].shape
    out = np.full((ny, nx), np.nan)
    for r0 in range(0, ny, row_chunk):
        r1 = min(r0 + row_chunk, ny)
        s = {f: np.asarray(mm[f][blocks, r0:r1], dtype=np.float64).sum(axis=0) for f in mm}
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r0:r1] = s["sqc"] / s["n"] - (s["sq"] / s["n"]) * (s["sc"] / s["n"])
    return out


def running_covariance(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Running estimate of <q'c'> using the first n frames (means recomputed for each n), along axis 0."""
    m = np.isfinite(q) & np.isfinite(c)
    with np.errstate(invalid="ignore"):
        # constant shifts leave the covariance unchanged and avoid cancellation (e.g. large mean v)
        qz = np.where(m, q - np.nanmean(np.where(m, q, np.nan), axis=0), 0.0)
        cz = np.where(m, c - np.nanmean(np.where(m, c, np.nan), axis=0), 0.0)
    n = np.cumsum(m, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.cumsum(qz * cz, axis=0) / n - (np.cumsum(qz, axis=0) / n) * (np.cumsum(cz, axis=0) / n)


def subset_moments_from_blocks(
    block_paths: dict[str, Path], q: str, blocks: slice, *, row_chunk: int = 128
) -> dict[str, np.ndarray]:
    """Fixed-frame decomposition of a sub-record (contiguous base blocks): <q'c'>, and the pair-valid means of the
    SHIFTED q and c (add the stream shift K_q, K_c to get physical means), with the means recomputed over the subset."""
    mm = {f: np.load(block_paths[f"{q}_{f}"], mmap_mode="r") for f in ("n", "sq", "sc", "sqc")}
    _, ny, nx = mm["n"].shape
    out = {k: np.full((ny, nx), np.nan) for k in ("flux", "q_mean_shifted", "c_mean_shifted", "n")}
    for r0 in range(0, ny, row_chunk):
        r1 = min(r0 + row_chunk, ny)
        s = {f: np.asarray(mm[f][blocks, r0:r1], dtype=np.float64).sum(axis=0) for f in mm}
        with np.errstate(invalid="ignore", divide="ignore"):
            Q, C = s["sq"] / s["n"], s["sc"] / s["n"]
            out["flux"][r0:r1] = s["sqc"] / s["n"] - Q * C
            out["q_mean_shifted"][r0:r1], out["c_mean_shifted"][r0:r1] = Q, C
        out["n"][r0:r1] = s["n"]
    return out
