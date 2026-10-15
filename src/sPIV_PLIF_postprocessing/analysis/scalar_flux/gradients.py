"""Mean scalar gradients (step 3).

Only in-plane gradients are computed: dC/dx (cross-stream, along columns) and dC/dy (streamwise grid
coordinate, along rows). No dC/dz exists for a single-plane measurement and none is inferred.

The mean field is smoothed before differentiating:
    * "gaussian": normalized convolution, C_s = G*(C m) / G*m with m the validity mask, so edges and masked
      pixels do not pull the smoothed field toward zero (no edge artifacts from zero padding);
    * "savgol": separable Savitzky-Golay (polynomial order 2), window = smallest odd number of pixels
      >= 2*sigma; requires a fully valid field.
Gradients use second-order central differences (np.gradient, second-order one-sided at the array edges) with the
physical grid spacing. Near the array edge and near invalid pixels the smoothing kernel is truncated, and the
one-sided average biases the gradient of a curved field (about 10% at 2 sigma from the edge for a Gaussian plume
in tests); gradients within ``edge_band`` = max(1, ceil(2 sigma / spacing)) pixels of the array edge or of
invalid data are therefore set to NaN.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def smooth_mean_field(
    C: np.ndarray, sigma_mm: float, dx_mm: float, dy_mm: float, *, method: str = "gaussian",
    valid: np.ndarray | None = None, truncate: float = 3.0, sigma_y_mm: float | None = None,
) -> np.ndarray:
    """Smooth C with a kernel of scale ``sigma_mm`` (Gaussian standard deviation; S-G half-window).
    ``sigma_y_mm`` (optional) sets a different scale along y (rows): anisotropic smoothing, used to suppress the
    stationary PLIF striation ripple along y (~8-17 mm wavelength) while keeping cross-stream gradients sharp."""
    valid = np.isfinite(C) if valid is None else (valid & np.isfinite(C))
    sy = sigma_mm if sigma_y_mm is None else sigma_y_mm
    if sigma_mm <= 0 and sy <= 0:
        return np.where(valid, C, np.nan)
    sig_px = (sy / abs(dy_mm), sigma_mm / abs(dx_mm))
    if method == "gaussian":
        m = valid.astype(np.float64)
        num = ndimage.gaussian_filter(np.where(valid, C, 0.0), sig_px, mode="constant", truncate=truncate)
        den = ndimage.gaussian_filter(m, sig_px, mode="constant", truncate=truncate)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[~valid] = np.nan
        return out
    if method == "savgol":
        if not valid.all():
            raise ValueError("savgol smoothing requires a fully valid field; use method='gaussian' with masks")
        out = np.asarray(C, dtype=np.float64)
        for axis, s in ((0, sig_px[0]), (1, sig_px[1])):
            win = max(3, int(np.ceil(2 * s)) // 2 * 2 + 1)
            out = signal.savgol_filter(out, win, polyorder=2, axis=axis, mode="interp")
        return out
    raise ValueError(f"Unknown smoothing method '{method}'")


def edge_band_px(sigma_mm: float, dx_mm: float, dy_mm: float) -> int:
    """Width (pixels) of the band next to edges/invalid data where the smoothed gradient is masked."""
    return max(1, int(np.ceil(2 * sigma_mm / min(abs(dx_mm), abs(dy_mm)))))


def mean_gradient(Cs: np.ndarray, dx_m: float, dy_m: float, *, edge_band: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(dC/dx, dC/dy) in units of C per metre; x along columns, y along rows (signed grid spacing).
    NaN within ``edge_band`` pixels of the array edge or of invalid pixels."""
    dCdy, dCdx = np.gradient(Cs, dy_m, dx_m, edge_order=2)
    invalid = ~np.isfinite(Cs)
    if invalid.any():
        invalid = ndimage.binary_dilation(invalid, iterations=edge_band)
    if edge_band > 0:
        invalid[:edge_band] = invalid[-edge_band:] = True
        invalid[:, :edge_band] = invalid[:, -edge_band:] = True
    dCdx[invalid] = np.nan
    dCdy[invalid] = np.nan
    return dCdx, dCdy


def gradient_sensitivity(
    C: np.ndarray, sigmas_mm: list[float], dx_mm: float, dy_mm: float, *, method: str = "gaussian",
    valid: np.ndarray | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    """Smoothed field and gradients for several smoothing scales: {sigma: {"C_s", "dCdx", "dCdy"}}."""
    out = {}
    for s in sigmas_mm:
        Cs = smooth_mean_field(C, s, dx_mm, dy_mm, method=method, valid=valid)
        gx, gy = mean_gradient(Cs, dx_mm / 1000, dy_mm / 1000, edge_band=edge_band_px(s, dx_mm, dy_mm))
        out[float(s)] = {"C_s": Cs, "dCdx": gx, "dCdy": gy}
    return out


def anisotropic_gradients(
    C: np.ndarray, sigma_x_mm: float, sigma_y_mm: float, dx_mm: float, dy_mm: float, *, valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Gaussian normalized-convolution smoothing with separate scales along x and y, then central differences.
    The masked edge band follows the larger scale along each direction (conservatively, the larger of the two)."""
    Cs = smooth_mean_field(C, sigma_x_mm, dx_mm, dy_mm, valid=valid, sigma_y_mm=sigma_y_mm)
    band = edge_band_px(max(sigma_x_mm, sigma_y_mm), dx_mm, dy_mm)
    gx, gy = mean_gradient(Cs, dx_mm / 1000, dy_mm / 1000, edge_band=band)
    return {"C_s": Cs, "dCdx": gx, "dCdy": gy}
