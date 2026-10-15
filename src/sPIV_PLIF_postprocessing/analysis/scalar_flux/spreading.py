"""Plume spreading and the Taylor-dispersion comparison (step 6).

Widths, per row (downstream station), from the fixed-frame mean concentration C(x) with background b removed:
    * second moment: sigma_x^2 = int (x - x_c)^2 (C - b)_+ dx / int (C - b)_+ dx   (geometry.plume_centroid_and_width)
    * Gaussian fit:  C(x) = A exp(-(x - x0)^2 / (2 sigma_g^2)) + b0, with a free offset b0 (robust to background
      errors and to tails, which dominate the second moment far downstream).

Three estimates of a plume-integrated cross-stream eddy diffusivity (fixed frame, so meander is included):
    K_T     = 1/2 U d(sigma_x^2)/ds                          Taylor / Fickian spreading, U = local advection speed
    K_FM    = +int (x - x_c) <u'c'> dx / int (C - b) dx       flux moment (no gradients needed)
    K_FG(s) = binned flux-gradient regression (step 4)
K_T = K_FM follows from integrating U dC/ds = -d<u'c'>/dx against (x - x_c)^2 across the plume (by parts,
-int x^2 dF/dx dx = +2 int x F dx; for F = -K dC/dx this gives K int C dx), assuming U uniform
across the plume, a conserved scalar mass flux int U (C - b) dx, and negligible streamwise flux divergence
(d<v'c'>/ds) and mean cross-stream advection. In a single plane, int U (C - b) dx is NOT conserved for a 3D plume:
it decays as the plume spreads out of the sheet. For a separable (x, z) plume with the sheet on its centre plane, the
out-of-plane flux-divergence term cancels exactly against the decay of int C dx, so K_T = K_FM still holds; the
in-plane flux is reported for transparency and is not used to infer any z quantity.
U is the concentration-weighted mean streamwise speed |V| across the row.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def _gauss(x, A, x0, sig, b0):
    return A * np.exp(-((x - x0) ** 2) / (2 * sig**2)) + b0


def gaussian_fit_widths(C: np.ndarray, x_mm: np.ndarray, xc0: np.ndarray, sig0: np.ndarray, *, background: float = 0.0,
                        min_peak: float = 0.005, max_sigma_frac: float = 0.25) -> dict[str, np.ndarray]:
    """Per-row Gaussian + offset fit (initial guess from the moments). Returns A, x0, sigma_g, b0. NaN if the fit fails,
    the row has no plume, sigma_g exceeds max_sigma_frac of the field width (the Gaussian is then not resolved inside
    the field of view), or the fitted centre lies outside the field."""
    ny = C.shape[0]
    out = {k: np.full(ny, np.nan) for k in ("A", "x0", "sigma_g", "b0")}
    for r in range(ny):
        y = C[r]
        ok = np.isfinite(y)
        if not ok.any() or not np.isfinite(sig0[r]) or np.nanmax(y) - background < min_peak:
            continue
        p0 = (np.nanmax(y) - background, xc0[r], sig0[r], background)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(_gauss, x_mm[ok], y[ok], p0=p0, maxfev=4000,
                                 bounds=([0, x_mm.min(), 0.1, -np.inf], [np.inf, x_mm.max(), 2 * np.ptp(x_mm), np.inf]))
        except (RuntimeError, ValueError):
            continue
        if p[2] > max_sigma_frac * np.ptp(x_mm) or not (x_mm.min() < p[1] < x_mm.max()):
            continue
        out["A"][r], out["x0"][r], out["sigma_g"][r], out["b0"][r] = p
    return out


def ds_derivative(f: np.ndarray, s_mm: np.ndarray, window_mm: float, polyorder: int = 2) -> np.ndarray:
    """d f / d s [per m] with a Savitzky-Golay derivative over ``window_mm`` (rows ordered as in the grid; s may
    decrease with row). NaNs are interpolated before filtering and restored after."""
    ds = float(np.mean(np.diff(s_mm))) / 1000  # signed, m per row
    win = max(polyorder + 3, int(round(window_mm / abs(ds * 1000))) // 2 * 2 + 1)
    fin = np.isfinite(f)
    if fin.sum() < win:
        return np.full_like(f, np.nan, dtype=float)
    idx = np.arange(f.size)
    g = np.interp(idx, idx[fin], f[fin])
    d = savgol_filter(g, win, polyorder, deriv=1, delta=ds, mode="interp")
    d[~fin] = np.nan
    return d


def advection_speed(V: np.ndarray, C: np.ndarray, background: float) -> np.ndarray:
    """Concentration-weighted mean |V| per row."""
    w = np.clip(np.nan_to_num(C - background), 0, None)
    w = np.where(np.isfinite(V), w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(w * np.abs(np.nan_to_num(V)), axis=1) / w.sum(axis=1)


def scalar_mass_flux(V: np.ndarray, C: np.ndarray, x_mm: np.ndarray, background: float) -> np.ndarray:
    """int |V| (C - b) dx per row [m/s * C * m], over pixels where V is valid."""
    dx = abs(float(np.mean(np.diff(x_mm)))) / 1000
    m = np.isfinite(V)
    return np.sum(np.where(m, np.abs(np.nan_to_num(V)) * (C - background), 0.0), axis=1) * dx


def flux_moment_K(flux_x: np.ndarray, C: np.ndarray, x_mm: np.ndarray, xc_mm: np.ndarray, background: float,
                  band: np.ndarray | None = None) -> np.ndarray:
    """K_FM(s) = +int (x - x_c) <u'c'> dx / int (C - b) dx, per row, over ``band`` (e.g. |x - x_c| <= 3 sigma) and
    finite flux pixels."""
    xr = (x_mm[None, :] - xc_mm[:, None]) / 1000
    m = np.isfinite(flux_x) & (np.ones_like(C, dtype=bool) if band is None else band)
    num = np.sum(np.where(m, xr * np.nan_to_num(flux_x), 0.0), axis=1)
    den = np.sum(np.where(m, C - background, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def taylor_K(sigma_mm: np.ndarray, U: np.ndarray, s_mm: np.ndarray, window_mm: float) -> np.ndarray:
    """K_T = 1/2 U d(sigma^2)/ds [m^2/s], with sigma in mm and the derivative from ``ds_derivative``.
    Uses the derivative with respect to downstream distance s (not the grid y)."""
    sig2 = (sigma_mm / 1000) ** 2
    return 0.5 * U * ds_derivative(sig2, s_mm, window_mm)
