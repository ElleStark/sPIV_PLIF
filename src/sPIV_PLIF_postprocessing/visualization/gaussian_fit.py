from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger("sPIV_PLIF.viz")


def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    fit_x_range: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]]:
    """
    Fit a Gaussian curve to (x, y) via nonlinear least squares.

    Returns (x_fit, y_gaussian, (amplitude, mu, sigma)) aligned to the x points used for fitting,
    or None on failure.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape; got {x.shape} vs {y.shape}")

    mask = np.isfinite(x) & np.isfinite(y)
    if fit_x_range is not None:
        xmin, xmax = fit_x_range
        mask &= (x >= xmin) & (x <= xmax)

    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 3:
        logger.warning("Gaussian fit skipped: insufficient points (%d).", x_fit.size)
        return None
    if not np.any(y_fit > 0):
        logger.warning("Gaussian fit skipped: no positive values.")
        return None

    weight_sum = float(np.sum(y_fit))
    mu0 = float(np.sum(x_fit * y_fit) / weight_sum) if weight_sum > 0 else float(np.mean(x_fit))
    variance0 = float(np.sum(y_fit * (x_fit - mu0) ** 2) / weight_sum) if weight_sum > 0 else float(np.var(x_fit))
    sigma0 = max(float(np.sqrt(variance0)), 1e-6)
    amp0 = float(np.nanmax(y_fit))

    try:
        popt, _ = curve_fit(
            _gaussian,
            x_fit,
            y_fit,
            p0=[amp0, mu0, sigma0],
            bounds=([0.0, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
            maxfev=2000,
        )
    except Exception as exc:  # pragma: no cover - external optimizer
        logger.warning("Gaussian fit failed: %s", exc)
        return None

    gaussian_curve = _gaussian(x_fit, *popt)
    amp, mu, sigma = popt
    return x_fit, gaussian_curve, (float(amp), float(mu), float(sigma))
