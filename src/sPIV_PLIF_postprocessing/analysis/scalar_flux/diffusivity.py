"""Gradient-diffusion test (step 4) and effective eddy diffusivity (step 5), in-plane only.

Model (fixed-frame Eulerian means; no z-gradient, no K_z):
    <u'c'> = -K_x dC/dx   (cross-stream, x along columns)
    <v'c'> = -K_y dC/dy   (streamwise grid coordinate, y along rows)
    tensor form in the plane: [<u'c'>, <v'c'>]^T = -K [dC/dx, dC/dy]^T, K a 2x2 tensor.
Because the mean is a fixed-pixel mean, plume meander contributes to the fluxes, so K here is a fixed-frame
("meander-inclusive") eddy diffusivity, not a plume-relative one.

A point is DOWN-GRADIENT when flux * gradient < 0 (i.e. pointwise K > 0) and COUNTER-GRADIENT when > 0.

Estimators
    * per-station regression of flux on gradient over the plume pixels in a streamwise bin: through the origin
      (slope = -K, R^2 uncentered: 1 - SS_res / sum f^2) and with an intercept (ordinary R^2). The formal slope
      SE treats pixels as independent and therefore UNDERSTATES the uncertainty (fluxes and gradients are
      spatially correlated over ~ the integral length scale); robustness checks (step 9) give empirical spread.
    * pointwise K = -flux / gradient, masked where |gradient| < frac * max|grad C| (per row by default) or where
      the flux is not significant (|flux| < k SE).
    * windowed: least squares through the origin over a square window W, K_W = -sum(f g) / sum(g^2), i.e. a
      g^2-weighted average of pointwise K that does not blow up where g -> 0.
    * windowed 2x2 tensor: per window, normal equations for f_i = -(K_i1 gx + K_i2 gy); masked where the window
      gradient covariance is ill-conditioned (gx and gy nearly collinear).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------------------------------ regression
def fit_through_origin(g: np.ndarray, f: np.ndarray) -> dict[str, float]:
    """f = b g. Returns K = -b, its formal SE, uncentered R^2, n."""
    n = g.size
    if n < 3 or np.sum(g * g) == 0:
        return {"K": np.nan, "K_se": np.nan, "r2": np.nan, "n": n}
    b = np.sum(f * g) / np.sum(g * g)
    res = f - b * g
    s2 = np.sum(res**2) / (n - 1)
    return {"K": -b, "K_se": float(np.sqrt(s2 / np.sum(g * g))), "r2": float(1 - np.sum(res**2) / np.sum(f * f)), "n": n}


def fit_with_intercept(g: np.ndarray, f: np.ndarray) -> dict[str, float]:
    """f = a + b g. Returns K = -b, its formal SE, intercept a, R^2, n."""
    n = g.size
    if n < 4 or np.var(g) == 0:
        return {"K": np.nan, "K_se": np.nan, "intercept": np.nan, "r2": np.nan, "n": n}
    gm, fm = g.mean(), f.mean()
    sxx = np.sum((g - gm) ** 2)
    b = np.sum((g - gm) * (f - fm)) / sxx
    a = fm - b * gm
    res = f - a - b * g
    ss_tot = np.sum((f - fm) ** 2)
    return {"K": -b, "K_se": float(np.sqrt(np.sum(res**2) / (n - 2) / sxx)), "intercept": float(a),
            "r2": float(1 - np.sum(res**2) / ss_tot) if ss_tot > 0 else np.nan, "n": n}


def downgradient_fractions(g: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """Fraction of points with f*g < 0 (down-gradient) and f*g > 0 (counter-gradient)."""
    p = f * g
    nz = p != 0
    if not nz.any():
        return np.nan, np.nan
    return float(np.mean(p[nz] < 0)), float(np.mean(p[nz] > 0))


def station_rows(downstream_mm: np.ndarray, centers_mm: list[float], width_mm: float) -> list[np.ndarray]:
    """Row indices of each streamwise bin |s - center| <= width/2."""
    return [np.flatnonzero(np.abs(downstream_mm - c) <= width_mm / 2) for c in centers_mm]


def gradient_diffusion_test(
    flux: np.ndarray, grad: np.ndarray, valid: np.ndarray, rows_per_bin: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-bin regressions and down-gradient fractions over pixels where ``valid`` & both finite."""
    keys = ("n", "K0", "K0_se", "r2_0", "K1", "K1_se", "intercept1", "r2_1", "frac_down", "frac_counter")
    out = {k: np.full(len(rows_per_bin), np.nan) for k in keys}
    for i, rows in enumerate(rows_per_bin):
        f, g, m = flux[rows], grad[rows], valid[rows] & np.isfinite(flux[rows]) & np.isfinite(grad[rows])
        f, g = f[m], g[m]
        o, w = fit_through_origin(g, f), fit_with_intercept(g, f)
        out["n"][i] = g.size
        out["K0"][i], out["K0_se"][i], out["r2_0"][i] = o["K"], o["K_se"], o["r2"]
        out["K1"][i], out["K1_se"][i], out["intercept1"][i], out["r2_1"][i] = w["K"], w["K_se"], w["intercept"], w["r2"]
        out["frac_down"][i], out["frac_counter"][i] = downgradient_fractions(g, f)
    return out


# ------------------------------------------------------------------------------------------ masks
def gradient_mask(gx: np.ndarray, gy: np.ndarray, which: np.ndarray, frac: float, mode: str = "row") -> np.ndarray:
    """True where |which| >= frac * max|grad C|; the max is taken per row (``mode="row"``, so the far field is not
    judged against the much steeper near-source gradients) or over the whole field (``mode="global"``)."""
    gmag = np.hypot(gx, gy)
    with np.errstate(invalid="ignore"):
        if mode == "row":
            ref = np.nanmax(np.where(np.isfinite(gmag), gmag, -np.inf), axis=1, keepdims=True)
        elif mode == "global":
            ref = np.nanmax(gmag)
        else:
            raise ValueError(mode)
        return np.abs(which) >= frac * ref


# ------------------------------------------------------------------------------------------ K estimators
def pointwise_K(flux: np.ndarray, grad: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """K = -flux / grad where ``mask`` (and both finite); NaN elsewhere."""
    with np.errstate(invalid="ignore", divide="ignore"):
        K = -flux / grad
    K[~(mask & np.isfinite(K))] = np.nan
    return K


def _box(a: np.ndarray, w: int) -> np.ndarray:
    return ndimage.uniform_filter(a, size=w, mode="constant") * (w * w)


def windowed_K(
    flux: np.ndarray, grad: np.ndarray, valid: np.ndarray, window_px: int, *, min_frac_valid: float = 0.5,
    grad_energy_min: np.ndarray | float = 0.0,
) -> dict[str, np.ndarray]:
    """Local least squares through the origin over a window_px x window_px window.

    K_W = -sum(f g) / sum(g^2) over valid pixels; r_W = sum(f g) / sqrt(sum f^2 sum g^2) (uncentered correlation;
    r_W = -1 is perfect down-gradient). Masked where fewer than min_frac_valid of the window pixels are valid or the
    mean g^2 in the window is below ``grad_energy_min``.
    """
    m = valid & np.isfinite(flux) & np.isfinite(grad)
    f = np.where(m, flux, 0.0)
    g = np.where(m, grad, 0.0)
    mf = m.astype(np.float64)
    n = _box(mf, window_px)
    sfg, sgg, sff = _box(f * g, window_px), _box(g * g, window_px), _box(f * f, window_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        K = -sfg / sgg
        r = sfg / np.sqrt(sff * sgg)
        mean_g2 = sgg / n
    bad = (n < min_frac_valid * window_px**2) | ~(mean_g2 >= grad_energy_min) | ~np.isfinite(K)
    K[bad] = np.nan
    r[bad] = np.nan
    return {"K": K, "r": r, "n": n}


def windowed_tensor(
    fx: np.ndarray, fy: np.ndarray, gx: np.ndarray, gy: np.ndarray, valid: np.ndarray, window_px: int, *,
    cond_max: float = 30.0, min_frac_valid: float = 0.5,
) -> dict[str, np.ndarray]:
    """2x2 eddy-diffusivity tensor from [fx, fy] = -K [gx, gy] by windowed least squares.

    Normal equations per window, with G = [[Sxx, Sxy], [Sxy, Syy]] (sums of g products):
        [K_i1, K_i2] = -[S(f_i gx), S(f_i gy)] G^-1.
    Masked where cond(G) > cond_max (gradient direction nearly constant within the window, so the tensor
    components are not separately identifiable).
    """
    m = valid & np.isfinite(fx) & np.isfinite(fy) & np.isfinite(gx) & np.isfinite(gy)
    z = lambda a: np.where(m, a, 0.0)  # noqa: E731
    fx0, fy0, gx0, gy0 = z(fx), z(fy), z(gx), z(gy)
    n = _box(m.astype(np.float64), window_px)
    Sxx, Sxy, Syy = _box(gx0 * gx0, window_px), _box(gx0 * gy0, window_px), _box(gy0 * gy0, window_px)
    det = Sxx * Syy - Sxy**2
    tr = Sxx + Syy
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = np.sqrt(np.clip(tr**2 / 4 - det, 0, None))
        cond = (tr / 2 + disc) / (tr / 2 - disc)
        out = {}
        for name, f0 in (("x", fx0), ("y", fy0)):
            a, b = _box(f0 * gx0, window_px), _box(f0 * gy0, window_px)
            # [a, b] G^-1 with G^-1 = [[Syy, -Sxy], [-Sxy, Sxx]] / det
            out[f"K{name}x"] = -(a * Syy - b * Sxy) / det
            out[f"K{name}y"] = -(-a * Sxy + b * Sxx) / det
    bad = (n < min_frac_valid * window_px**2) | ~(cond <= cond_max)
    for k in list(out):
        out[k][bad] = np.nan
    out["cond"] = cond
    return out


def cross_stream_profile(field: np.ndarray, region: np.ndarray, *, stat: str = "median") -> np.ndarray:
    """Per-row median (or mean) of ``field`` over ``region`` pixels; NaN where a row has no valid pixels."""
    vals = np.where(region & np.isfinite(field), field, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(vals, axis=1) if stat == "median" else np.nanmean(vals, axis=1)


def plume_region(C: np.ndarray, x_mm: np.ndarray, *, background: float, c_threshold: float,
                 xc: np.ndarray, sigma: np.ndarray, n_sigma: float = 2.5) -> np.ndarray:
    """Pixels inside the plume: C - background > c_threshold and |x - x_c(y)| <= n_sigma * sigma_x(y)."""
    near = np.abs(x_mm[None, :] - xc[:, None]) <= n_sigma * sigma[:, None]
    return ((C - background) > c_threshold) & near


def flux_gradient_angle(fx: np.ndarray, fy: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Unsigned angle [deg] between the in-plane flux vector (<u'c'>, <v'c'>) and the down-gradient direction -grad C:
    0 = exactly down-gradient, 90 = orthogonal, 180 = exactly counter-gradient. NaN where either vector is zero/NaN."""
    dot = -(fx * gx + fy * gy)
    norm = np.hypot(fx, fy) * np.hypot(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.clip(dot / norm, -1.0, 1.0)
    return np.where(norm > 0, np.degrees(np.arccos(cosang)), np.nan)
