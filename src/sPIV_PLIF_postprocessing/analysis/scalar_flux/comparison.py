"""Cross-case comparison of effective diffusivities and of the meander -> relative-dispersion transition.

In a fixed (Eulerian) frame, meander and relative dispersion cannot be separated directly (that needs a
plume-following decomposition). The transition is therefore located with three proxies, reported side by side:

    s_sigma_L  first downstream distance where the plume width exceeds the eddy size, sigma_x(s) >= L_u,x(s)
               (L_u,x = cross-stream median over the plume of the longitudinal integral length scale of u).
               Plumes narrower than the energetic eddies are moved as a whole (meander); wider plumes are mixed
               internally (relative dispersion).
    s_rho_50   where the flank correlation rho_flank(s) = mean |rho_uc| over 0.5 <= |eta| <= 1.5, eta = (x - x_c)/sigma_x,
               has fallen halfway from its near-field level to its far-field level (s_rho_25 / s_rho_75 bound the
               transition zone). Whole-plume displacement gives strongly coherent u'-c' coupling on the flanks.
    s_rho_abs_<lvl>  where rho_flank first falls below a common absolute level (default 0.40, 0.35), for comparing
               cases whose rho_flank has not levelled off by the end of the field of view (the relative half-decay
               is then biased upstream).
    s_K_90     where K_x / (u_rms L_u,x) first reaches 90% of its far-field median. This is Taylor's
               ballistic -> diffusive transition (s ~ U T_L), related to but distinct from the meander transition.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import median_filter


def _row_stat(field: np.ndarray, mask: np.ndarray, stat=np.nanmedian) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return stat(np.where(mask & np.isfinite(field), field, np.nan), axis=1)


def smooth_along_s(prof: np.ndarray, s_mm: np.ndarray, window_mm: float) -> np.ndarray:
    """Running median over ``window_mm`` of streamwise distance (NaN-tolerant via edge filling)."""
    n = max(1, int(round(window_mm / np.mean(np.abs(np.diff(s_mm))))))
    p = np.array(prof, dtype=float)
    fin = np.isfinite(p)
    if not fin.any():
        return p
    idx = np.arange(p.size)
    filled = np.interp(idx, idx[fin], p[fin])
    out = median_filter(filled, size=n, mode="nearest")
    out[~fin] = np.nan
    return out


def first_crossing(s_mm: np.ndarray, prof: np.ndarray, level: float, *, direction: str) -> float:
    """First downstream distance (scanning from the source) where ``prof`` crosses ``level`` going
    ``direction`` ("up" or "down"); linear interpolation between samples. NaN if it never crosses."""
    order = np.argsort(s_mm)
    s, p = s_mm[order], prof[order]
    ok = np.isfinite(p)
    s, p = s[ok], p[ok]
    d = p - level
    hits = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) if direction == "up" else np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
    if hits.size == 0:
        return np.nan
    i = hits[0]
    return float(s[i] + (s[i + 1] - s[i]) * (-d[i]) / (d[i + 1] - d[i]))


def transition_metrics(
    s_mm: np.ndarray, x_mm: np.ndarray, sigma: np.ndarray, xc: np.ndarray, rho_uc: np.ndarray, L_ux: np.ndarray,
    Kx_nd: np.ndarray, region: np.ndarray, *, near_mm=(20.0, 60.0), far_mm=(200.0, 280.0), smooth_mm=10.0,
    abs_levels=(0.40, 0.35), abs_start_mm: float = 40.0,
) -> dict:
    """Row-wise diagnostics vs downstream distance and the three transition estimates (see module docstring)."""
    eta = (x_mm[None, :] - xc[:, None]) / sigma[:, None]
    flank = region & (np.abs(eta) >= 0.5) & (np.abs(eta) <= 1.5)
    prof = {
        "sigma_x_mm": smooth_along_s(sigma, s_mm, smooth_mm),
        "L_ux_mm": smooth_along_s(_row_stat(L_ux * 1000, region), s_mm, smooth_mm),
        "rho_flank": smooth_along_s(_row_stat(np.abs(rho_uc), flank, np.nanmean), s_mm, smooth_mm),
        "Kx_nd": smooth_along_s(_row_stat(Kx_nd, region), s_mm, smooth_mm),
    }
    prof["sigma_over_L"] = prof["sigma_x_mm"] / prof["L_ux_mm"]
    near = (s_mm >= near_mm[0]) & (s_mm <= near_mm[1])
    far = (s_mm >= far_mm[0]) & (s_mm <= far_mm[1])
    r_near, r_far = float(np.nanmedian(prof["rho_flank"][near])), float(np.nanmedian(prof["rho_flank"][far]))
    out = {"rho_flank_near": r_near, "rho_flank_far": r_far}
    for q in (25, 50, 75):
        lvl = r_near - q / 100 * (r_near - r_far)
        out[f"s_rho_{q}"] = first_crossing(s_mm, prof["rho_flank"], lvl, direction="down") if r_near > r_far else np.nan
    beyond = np.where(s_mm >= abs_start_mm, prof["rho_flank"], np.nan)  # skip the near-source rise
    for lvl in abs_levels:
        out[f"s_rho_abs_{lvl:.2f}"] = first_crossing(s_mm, beyond, lvl, direction="down")
    out["s_sigma_L"] = first_crossing(s_mm, prof["sigma_over_L"], 1.0, direction="up")
    kfar = float(np.nanmedian(prof["Kx_nd"][s_mm >= 150]))
    out["Kx_nd_far"] = kfar
    out["s_K_90"] = first_crossing(s_mm, prof["Kx_nd"], 0.9 * kfar, direction="up")
    return {"profiles": prof, "metrics": out}
