"""Streaming Reynolds decomposition and turbulent scalar fluxes (step 1), plus inputs for step 2.

Definitions (per fixed pixel, fixed Eulerian frame):
    q = Q + q',  Q = time mean of q over all valid frames,  q in {u, v, w, c}
    <q'c'> = mean over frames where both q and c are valid of (q - Q_p)(c - C_p)
where Q_p, C_p are the means over that same pairwise-valid set (pairwise-complete covariance). Because the
mean is taken at a fixed pixel, slow plume meander contributes to <q'c'>: these are fixed-frame fluxes, not
plume-relative ones.

All statistics are accumulated in one pass over the (t, y, x) stacks in time chunks, as float64 sums of the
SHIFTED variables q - K_q (K_q = per-pixel mean of the first chunk). Shifting removes the cancellation that
E[qc] - E[q]E[c] would otherwise suffer with a large mean streamwise velocity, and gives exact results.

During the same pass the following are written/kept for the uncertainty analysis:
    * per-base-block sums (n, sum q, sum c, sum qc) for each velocity component, as float32 memmaps, so the
      moving-block bootstrap can be done afterwards for any block length that is a multiple of the base block;
    * full time series on a coarse sub-grid (integral time scales) and at representative points
      (convergence, spectra).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

VEL = ("u", "v", "w")
ALL = ("u", "v", "w", "c")
BLOCK_FIELDS = ("n", "sq", "sc", "sqc")


@dataclass
class StreamSettings:
    chunk_frames: int = 20  # frames read per chunk; must be a multiple of base_block_frames (~2.5 GB peak RAM at 947^2)
    base_block_frames: int = 20  # base block for the bootstrap (block length is a multiple of this)
    coarse_stride: int = 8  # sub-grid stride (pixels) for stored time series
    w_outlier_threshold: float | None = -1.0  # w' < threshold -> NaN (existing repo rule); None disables
    flip_plif_rows: bool = False  # flip the PLIF stack vertically on read (baseline PLIF is stored upside down)
    t_slice: slice = slice(None)

    def __post_init__(self) -> None:
        if self.chunk_frames % self.base_block_frames:
            raise ValueError("chunk_frames must be a multiple of base_block_frames")


@dataclass
class StreamResult:
    sums: dict[str, np.ndarray]  # float64 accumulators of shifted variables (see _accumulate)
    shift: dict[str, np.ndarray]  # K_q per pixel
    n_frames: int
    block_paths: dict[str, Path]  # "<q>_<field>" -> .npy memmap (n_blocks, ny, nx)
    block_frames: np.ndarray  # frames in each base block (last may be short)
    coarse_rows: np.ndarray
    coarse_cols: np.ndarray
    coarse_series: dict[str, np.ndarray]  # q -> (nt, n_rows, n_cols) float32, invalid = NaN
    point_series: dict[str, np.ndarray] = field(default_factory=dict)  # q -> (nt, n_points) float32


def _new_sums(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    z = lambda: np.zeros(shape, dtype=np.float64)  # noqa: E731
    sums = {}
    for q in ALL:  # own (single-variable) moments
        for k in ("n", "s", "ss"):
            sums[f"{q}_{k}"] = z()
    for q in VEL:  # pairwise moments with c, over frames where both q and c are valid
        for k in ("n", "sq", "sc", "sqq", "scc", "sqc", "sq2c", "sqc2", "sq2c2"):
            sums[f"{q}c_{k}"] = z()
    return sums


def apply_w_outlier_rule(w_shifted: np.ndarray, threshold: float | None) -> np.ndarray:
    """Existing repo rule (flow_properties.compute_fluctuating_components): w' < -1 m/s -> NaN.
    Applied to w - K_w with K_w the first-chunk mean, which differs from the full mean by << 1 m/s."""
    if threshold is not None:
        w_shifted = np.where(w_shifted < threshold, np.nan, w_shifted)
    return w_shifted


def _accumulate(sums: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add one chunk of shifted data (each (nt, ny, nx) float64, NaN = invalid) to the accumulators.
    Returns per-frame pair masks (for block sums)."""
    for q in ALL:
        a = data[q]
        m = np.isfinite(a)
        az = np.where(m, a, 0.0)
        sums[f"{q}_n"] += m.sum(axis=0)
        sums[f"{q}_s"] += az.sum(axis=0)
        sums[f"{q}_ss"] += (az * az).sum(axis=0)
    c = data["c"]
    cm = np.isfinite(c)
    masks = {}
    for q in VEL:
        pm = np.isfinite(data[q]) & cm
        qz = np.where(pm, data[q], 0.0)
        cz = np.where(pm, c, 0.0)
        q2, c2, qc = qz * qz, cz * cz, qz * cz
        sums[f"{q}c_n"] += pm.sum(axis=0)
        sums[f"{q}c_sq"] += qz.sum(axis=0)
        sums[f"{q}c_sc"] += cz.sum(axis=0)
        sums[f"{q}c_sqq"] += q2.sum(axis=0)
        sums[f"{q}c_scc"] += c2.sum(axis=0)
        sums[f"{q}c_sqc"] += qc.sum(axis=0)
        sums[f"{q}c_sq2c"] += (q2 * cz).sum(axis=0)
        sums[f"{q}c_sqc2"] += (qz * c2).sum(axis=0)
        sums[f"{q}c_sq2c2"] += (q2 * c2).sum(axis=0)
        masks[q] = (pm, qz, cz, qc)
    return masks


def stream_moments(
    paths: dict[str, Path | str],
    block_dir: Path,
    *,
    settings: StreamSettings = StreamSettings(),
    points: Sequence[tuple[int, int]] = (),
    progress: bool = True,
) -> StreamResult:
    """One pass over the (t, y, x) stacks ``paths = {"u", "v", "w", "c"}``.

    Returns accumulators (see ``finalize_moments``), per-base-block sums written to ``block_dir``, and stored
    time series on the coarse sub-grid and at ``points`` [(row, col), ...].
    """
    arrs = {q: np.load(paths[q], mmap_mode="r") for q in ALL}
    shapes = {a.shape for a in arrs.values()}
    if len(shapes) != 1:
        raise ValueError(f"Stacks have different shapes: { {q: a.shape for q, a in arrs.items()} }")
    nt_total, ny, nx = arrs["c"].shape
    t_idx = range(*settings.t_slice.indices(nt_total))
    if t_idx.step != 1:
        raise ValueError("t_slice must be contiguous")
    nt = len(t_idx)
    b0 = settings.base_block_frames
    n_blocks = -(-nt // b0)
    block_frames = np.full(n_blocks, b0)
    if nt % b0:
        block_frames[-1] = nt % b0

    block_dir.mkdir(parents=True, exist_ok=True)
    block_paths, block_mm = {}, {}
    for q in VEL:
        for f in BLOCK_FIELDS:
            p = block_dir / f"block_{q}_{f}.npy"
            block_paths[f"{q}_{f}"] = p
            block_mm[f"{q}_{f}"] = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n_blocks, ny, nx))

    rows = np.arange(0, ny, settings.coarse_stride)
    cols = np.arange(0, nx, settings.coarse_stride)
    coarse = {q: np.full((nt, rows.size, cols.size), np.nan, dtype=np.float32) for q in ALL}
    pr = np.array([p[0] for p in points], dtype=int)
    pc = np.array([p[1] for p in points], dtype=int)
    if len(points) and ((pr < 0).any() or (pr >= ny).any() or (pc < 0).any() or (pc >= nx).any()):
        raise ValueError(f"Points {list(points)} out of bounds for grid {(ny, nx)}")
    pts = {q: np.full((nt, len(points)), np.nan, dtype=np.float32) for q in ALL}

    sums = _new_sums((ny, nx))
    shift: dict[str, np.ndarray] = {}
    for start in range(0, nt, settings.chunk_frames):
        stop = min(start + settings.chunk_frames, nt)
        sl = slice(t_idx[start], t_idx[stop - 1] + 1)
        data = {}
        for q in ALL:
            raw = np.asarray(arrs[q][sl], dtype=np.float64)
            if q == "c" and settings.flip_plif_rows:
                raw = raw[:, ::-1, :]
            if q not in shift:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels -> NaN -> shift 0
                    k = np.nanmean(raw, axis=0)
                shift[q] = np.nan_to_num(k, nan=0.0)
            data[q] = raw - shift[q][None]
        data["w"] = apply_w_outlier_rule(data["w"], settings.w_outlier_threshold)

        masks = _accumulate(sums, data)

        # Per-base-block sums
        nb_chunk = -(-(stop - start) // b0)
        b_first = start // b0
        for q in VEL:
            pm, qz, cz, qc = masks[q]
            for j in range(nb_chunk):
                fs = slice(j * b0, min((j + 1) * b0, stop - start))
                b = b_first + j
                block_mm[f"{q}_n"][b] = pm[fs].sum(axis=0)
                block_mm[f"{q}_sq"][b] = qz[fs].sum(axis=0)
                block_mm[f"{q}_sc"][b] = cz[fs].sum(axis=0)
                block_mm[f"{q}_sqc"][b] = qc[fs].sum(axis=0)

        # Stored time series (unshifted values; w outliers already NaN)
        for q in ALL:
            vals = data[q] + shift[q][None]
            coarse[q][start:stop] = vals[:, rows][:, :, cols]
            if len(points):
                pts[q][start:stop] = vals[:, pr, pc]
        if progress and (start // settings.chunk_frames) % 25 == 0:
            print(f"  frames {start}-{stop} of {nt}", flush=True)

    for mm in block_mm.values():
        mm.flush()
    del block_mm
    return StreamResult(
        sums=sums, shift=shift, n_frames=nt, block_paths=block_paths, block_frames=block_frames,
        coarse_rows=rows, coarse_cols=cols, coarse_series=coarse, point_series=pts,
    )


def finalize_moments(
    sums: dict[str, np.ndarray],
    shift: dict[str, np.ndarray],
    *,
    min_valid_frames: int = 100,
    rho_min_rms: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Means, rms, fluxes, product-series variance, and correlation coefficients from the accumulators.

    Returns arrays (ny, nx):
        <q>_mean, <q>_rms, <q>_n                    for q in u, v, w, c (own valid samples)
        <q>c_flux = <q'c'>, <q>c_n                  pairwise-valid frames
        <q>c_pvar = var of p_t = q'c' (for N_eff), <q>c_rho = <q'c'> / (q_rms c_rms) (pairwise rms)
    Pixels with fewer than ``min_valid_frames`` samples are NaN. rho is NaN where the pairwise rms of q or c
    is below ``rho_min_rms[q]`` / ``rho_min_rms["c"]``.
    """
    rho_min_rms = rho_min_rms or {}
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for q in ALL:
            n = sums[f"{q}_n"]
            m = sums[f"{q}_s"] / n
            var = np.clip(sums[f"{q}_ss"] / n - m * m, 0.0, None)
            bad = n < min_valid_frames
            out[f"{q}_mean"] = np.where(bad, np.nan, m + shift[q])
            out[f"{q}_rms"] = np.where(bad, np.nan, np.sqrt(var))
            out[f"{q}_n"] = n
        for q in VEL:
            n = sums[f"{q}c_n"]
            Q = sums[f"{q}c_sq"] / n
            C = sums[f"{q}c_sc"] / n
            Eqc = sums[f"{q}c_sqc"] / n
            flux = Eqc - Q * C
            varq = np.clip(sums[f"{q}c_sqq"] / n - Q * Q, 0.0, None)
            varc = np.clip(sums[f"{q}c_scc"] / n - C * C, 0.0, None)
            # E[(q-Q)^2 (c-C)^2] expanded in raw (shifted) moments
            Ea2b2 = (
                sums[f"{q}c_sq2c2"] / n
                - 2 * C * sums[f"{q}c_sq2c"] / n
                - 2 * Q * sums[f"{q}c_sqc2"] / n
                + C * C * sums[f"{q}c_sqq"] / n
                + Q * Q * sums[f"{q}c_scc"] / n
                + 4 * Q * C * Eqc
                - 3 * Q * Q * C * C
            )
            pvar = np.clip(Ea2b2 - flux * flux, 0.0, None)
            rq, rc = np.sqrt(varq), np.sqrt(varc)
            rho = flux / (rq * rc)
            rho[(rq < rho_min_rms.get(q, 0.0)) | (rc < rho_min_rms.get("c", 0.0)) | (rq <= 0) | (rc <= 0)] = np.nan
            bad = n < min_valid_frames
            out[f"{q}c_flux"] = np.where(bad, np.nan, flux)
            out[f"{q}c_pvar"] = np.where(bad, np.nan, pvar)
            out[f"{q}c_rho"] = np.where(bad, np.nan, rho)
            out[f"{q}c_n"] = n
            # pair means (shifted) kept for the bootstrap
            out[f"_{q}c_Qs"] = Q
            out[f"_{q}c_Cs"] = C
    return out


def noise_floor_from_background(
    c_rms: np.ndarray, c_mean: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, strips_mm: Sequence[Sequence[float]],
    y_range_mm: Sequence[float] | None = None,
) -> dict[str, float]:
    """PLIF noise floor from plume-free background strips (cross-stream ranges in mm, as in compute_bg_percentiles).

    sigma_n^2 = median over strip pixels of the temporal variance of c. Measurement noise uncorrelated with
    velocity biases c_rms upward (c_rms^2 = c_true^2 + sigma_n^2) but leaves <q'c'> unbiased to first order.
    """
    xm = np.zeros(x_mm.size, dtype=bool)
    for lo, hi in strips_mm:
        xm |= (x_mm >= lo) & (x_mm <= hi)
    ym = np.ones(y_mm.size, dtype=bool) if y_range_mm is None else (y_mm >= y_range_mm[0]) & (y_mm <= y_range_mm[1])
    sel = np.ix_(ym, xm)
    var = c_rms[sel] ** 2
    return {
        "noise_std": float(np.sqrt(np.nanmedian(var))),
        "noise_std_p90": float(np.sqrt(np.nanpercentile(var, 90))),
        "background_mean": float(np.nanmedian(c_mean[sel])),
        "n_pixels": int(np.isfinite(var).sum()),
    }


def noise_corrected_rms(c_rms: np.ndarray, noise_std: float) -> np.ndarray:
    """c_rms with the noise variance removed: sqrt(max(c_rms^2 - sigma_n^2, 0))."""
    return np.sqrt(np.clip(c_rms**2 - noise_std**2, 0.0, None))
