"""Flow property calculations for PIV fields.

The computation formulas mirror the original script; functions simply
package each derived quantity so they can be reused from tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# Default slices and paths replicate the original script behavior.
DEFAULT_BASE_PATH = None
DEFAULT_X_SLICE = None
DEFAULT_Y_SLICE = None
DEFAULT_T_SLICE = None
DEFAULT_DX = 0.0  # m
DEFAULT_DT = 0.0  # sec
DEFAULT_NU = 0  # kinematic viscosity


def load_velocity_components(
    case_name: str,
    *,
    base_path: Path | str = DEFAULT_BASE_PATH,
    x_slice: slice = DEFAULT_X_SLICE,
    y_slice: slice = DEFAULT_Y_SLICE,
    t_slice: slice = DEFAULT_T_SLICE,
    mmap_mode: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load u, v, w components with slicing applied."""
    base = Path(base_path)
    u = np.load(base / "PIV" / "Interpolated_to_PLIF" / f"piv_{case_name}_u.npy", mmap_mode=mmap_mode)[x_slice, y_slice, t_slice]
    v = np.load(base / "PIV" / "Interpolated_to_PLIF" / f"piv_{case_name}_v.npy", mmap_mode=mmap_mode)[x_slice, y_slice, t_slice]
    w = np.load(base / "PIV" / "Interpolated_to_PLIF" / f"piv_{case_name}_w.npy", mmap_mode=mmap_mode)[x_slice, y_slice, t_slice]
    return u, v, w


def load_mean_velocity_components(
    case_name: str,
    *,
    base_path: Path | str = DEFAULT_BASE_PATH,
    mmap_mode: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mean u, v, w components from the mean_fields archive."""
    base = Path(base_path)
    # mean_fields = np.load(base / "mean_fields" / f"mean_fields_{case_name}.npz", mmap_mode=mmap_mode)
    # return mean_fields["u"], mean_fields["v"], mean_fields["w"]
    u_mean_field = np.load(base / "mean_variance_fields" / f"{case_name}_u_mean.npy", mmap_mode=mmap_mode)
    v_mean_field = np.load(base / "mean_variance_fields" / f"{case_name}_v_mean.npy", mmap_mode=mmap_mode)
    w_mean_field = np.load(base / "mean_variance_fields" / f"{case_name}_w_mean.npy", mmap_mode=mmap_mode)

    return u_mean_field, v_mean_field, w_mean_field


def compute_fluctuating_components(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    u_mean: np.ndarray,
    v_mean: np.ndarray,
    w_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose velocity fields into fluctuating components."""
    if u.ndim == 3 and u_mean.ndim == 2:
        u_mean = u_mean[None, :, :]
    if v.ndim == 3 and v_mean.ndim == 2:
        v_mean = v_mean[None, :, :]
    if w.ndim == 3 and w_mean.ndim == 2:
        w_mean = w_mean[None, :, :]

    u_flx = u - u_mean
    v_flx = v - v_mean
    w_flx = w - w_mean
    w_flx[w_flx<(-1)] = np.nan
    return u_flx, v_flx, w_flx


def compute_fluctuating_strain_rates(
    u_flx: np.ndarray,
    v_flx: np.ndarray,
    *,
    dx: float = DEFAULT_DX,
    dt: float = DEFAULT_DT,
    x_axis: int = 2,
    y_axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Longitudinal gradients of the fluctuating velocity: du'/dx, dv'/dy and dw'/dz from continuity.

    Arrays are (t, y, x): u is the cross-stream component (along x = axis 2, columns) and v the streamwise
    component (along y = axis 1, rows). 

    Returns:
        duflx_dx, dvflx_dy, dwflx_dz
    """
    duflx_dx = np.asarray(np.gradient(u_flx, dx, axis=x_axis))
    dvflx_dy = np.asarray(np.gradient(v_flx, dx, axis=y_axis))
    dwflx_dz = -(duflx_dx + dvflx_dy)  # continuity (incompressible fluctuations)
    return duflx_dx, dvflx_dy, dwflx_dz


def compute_viscous_dissipation(
    duflx_dx: np.ndarray,
    dvflx_dy: np.ndarray,
    dwflx_dz: np.ndarray,
    *,
    nu: float = DEFAULT_NU,
) -> np.ndarray:
    """Compute viscous energy dissipation rate."""
    epsilon = 5 * nu * (
        np.mean(duflx_dx**2, axis=0) + np.mean(dvflx_dy**2, axis=0) + np.mean(dwflx_dz**2, axis=0)
    )
    return epsilon


def compute_taylor_scales(
    u_flx: np.ndarray,
    v_flx: np.ndarray,
    w_flx: np.ndarray,
    epsilon: np.ndarray,
    *,
    nu: float = DEFAULT_NU,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Taylor microscale, Kolmogorov scales, and Taylor Reynolds number."""
    avg_rms = np.sqrt((1 / 3) * (np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0) + np.mean(w_flx**2, axis=0)))
    Taylor_microscale = np.sqrt(15 * nu / epsilon) * avg_rms  # homogeneous isotropic turbulence assumption
    kolmogorov_length_scale = (nu**3 / epsilon) ** 0.25
    kolmogorov_time_scale = (nu / epsilon) ** 0.5
    Taylor_Re = avg_rms * Taylor_microscale / nu
    return Taylor_microscale, kolmogorov_length_scale, kolmogorov_time_scale, Taylor_Re


def compute_turbulent_kinetic_energy(
    u_flx: np.ndarray,
    v_flx: np.ndarray,
    w_flx: np.ndarray,
) -> np.ndarray:
    """Compute turbulent kinetic energy."""
    u_mnsq = np.mean(u_flx**2, axis=0)
    v_mnsq = np.mean(v_flx**2, axis=0)
    w_mnsq = np.mean(w_flx**2, axis=0)
    tke = 0.5 * (u_mnsq + v_mnsq + w_mnsq)
    return tke, u_mnsq, v_mnsq, w_mnsq


def compute_turbulence_intensity(
    u_flx: np.ndarray,
    v_flx: np.ndarray,
    w_flx: np.ndarray,
    u_mean: float = 0.30,
) -> np.ndarray:
    """Compute turbulence intensity using RMS components and mean streamwise velocity."""
    t_intensity_avg = np.sqrt((1 / 3) * (np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0) + np.mean(w_flx**2, axis=0))) / u_mean
    return t_intensity_avg


# ----------------------------------------------------------------------------------------------------------------
# Dissipation-rate estimation (Oct 2026 revision)
#
# Velocity-gradient tensor from stereo PIV in one plane: all x- and y-derivatives of u, v, w are measured;
# dw/dz follows from continuity; du/dz and dv/dz are not measured.
#   eps_iso_long = 15 nu <(du_i/dx_i)^2>_avg = 5 nu [<(du/dx)^2> + <(dv/dy)^2> + <(dw/dz)^2>]   (isotropy)
#   eps_hybrid   = nu <du_i/dx_j du_i/dx_j>, using the 7 measured/continuity terms and, for the 2 unmeasured
#                  ones, the isotropic substitutions <(du/dz)^2> = <(du/dy)^2>, <(dv/dz)^2> = <(dv/dx)^2>
#                  (homogeneous turbulence: eps = nu <du_i/dx_j du_i/dx_j>; only 2 of 9 terms assumed).
# Gradients are central differences spanning 2*step grid points, NaN where the stencil leaves the field.
# ----------------------------------------------------------------------------------------------------------------
GRADIENT_TERMS = ("du_dx", "du_dy", "dv_dx", "dv_dy", "dw_dx", "dw_dy", "dw_dz")


def _central_diff(a: np.ndarray, axis: int, step: int, spacing: float) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float64)
    n = a.shape[axis]
    if n <= 2 * step:
        return out
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    mid = [slice(None)] * a.ndim
    hi[axis] = slice(2 * step, n)
    lo[axis] = slice(0, n - 2 * step)
    mid[axis] = slice(step, n - step)
    out[tuple(mid)] = (a[tuple(hi)] - a[tuple(lo)]) / (2 * step * spacing)
    return out


def velocity_gradients(
    u_flx: np.ndarray, v_flx: np.ndarray, w_flx: np.ndarray, *, dx: float, dy: float, step: int = 1,
    x_axis: int = 2, y_axis: int = 1,
) -> dict[str, np.ndarray]:
    """In-plane gradients of the fluctuating velocity (u along x, v along y, w out of plane) plus dw/dz from
    continuity. ``dx``, ``dy`` are the signed grid spacings [m] along the x and y array axes."""
    g = {
        "du_dx": _central_diff(u_flx, x_axis, step, dx), "du_dy": _central_diff(u_flx, y_axis, step, dy),
        "dv_dx": _central_diff(v_flx, x_axis, step, dx), "dv_dy": _central_diff(v_flx, y_axis, step, dy),
        "dw_dx": _central_diff(w_flx, x_axis, step, dx), "dw_dy": _central_diff(w_flx, y_axis, step, dy),
    }
    g["dw_dz"] = -(g["du_dx"] + g["dv_dy"])
    return g


class GradientMoments:
    """NaN-aware running sums of squared gradient terms (and two cross terms) per pixel."""

    def __init__(self, shape: tuple[int, int]):
        self.sums = {k: np.zeros(shape) for k in GRADIENT_TERMS + ("du_dy*dv_dx", "du_dx*dv_dy")}
        self.counts = {k: np.zeros(shape) for k in self.sums}

    def add(self, g: dict[str, np.ndarray]) -> None:
        prods = {k: g[k] ** 2 for k in GRADIENT_TERMS}
        prods["du_dy*dv_dx"] = g["du_dy"] * g["dv_dx"]
        prods["du_dx*dv_dy"] = g["du_dx"] * g["dv_dy"]
        for k, p in prods.items():
            m = np.isfinite(p)
            self.sums[k] += np.where(m, p, 0.0).sum(axis=0)
            self.counts[k] += m.sum(axis=0)

    def means(self, min_count: int = 1) -> dict[str, np.ndarray]:
        with np.errstate(invalid="ignore", divide="ignore"):
            return {k: np.where(self.counts[k] >= min_count, self.sums[k] / self.counts[k], np.nan) for k in self.sums}


def dissipation_estimates(gm: dict[str, np.ndarray], *, nu: float) -> dict[str, np.ndarray]:
    """eps_iso_long and eps_hybrid (see block comment above) from mean squared gradients."""
    eps_iso = 5 * nu * (gm["du_dx"] + gm["dv_dy"] + gm["dw_dz"])
    measured = sum(gm[k] for k in GRADIENT_TERMS)
    eps_hybrid = nu * (measured + gm["du_dy"] + gm["dv_dx"])  # + <(du/dz)^2> + <(dv/dz)^2> by isotropy
    return {"eps_iso_long": eps_iso, "eps_hybrid": eps_hybrid}


def isotropy_ratios(gm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Diagnostics of the isotropy assumptions: transverse/longitudinal (isotropic 2), dw/dz vs longitudinal (1),
    cross-term <du/dx dv/dy> / mean longitudinal (isotropic -1/2)."""
    long_mean = 0.5 * (gm["du_dx"] + gm["dv_dy"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "u_trans_over_long": gm["du_dy"] / gm["du_dx"],
            "v_trans_over_long": gm["dv_dx"] / gm["dv_dy"],
            "dwdz_over_long": gm["dw_dz"] / long_mean,
            "cross_over_long": gm["du_dx*dv_dy"] / long_mean,
        }


def turbulence_scales(epsilon: np.ndarray, k: np.ndarray, *, nu: float) -> dict[str, np.ndarray]:
    """Taylor microscale lambda = sqrt(15 nu u'^2 / eps) with u'^2 = 2k/3, Re_lambda = u' lambda / nu,
    Kolmogorov eta = (nu^3/eps)^(1/4) and tau_eta = (nu/eps)^(1/2). Non-positive eps -> NaN."""
    eps = np.where(epsilon > 0, epsilon, np.nan)
    up = np.sqrt(2.0 * k / 3.0)
    lam = np.sqrt(15 * nu / eps) * up
    return {"taylor_microscale": lam, "taylor_Re": up * lam / nu, "kolmogorov_length": (nu**3 / eps) ** 0.25,
            "kolmogorov_time": (nu / eps) ** 0.5, "u_prime": up}


class StructureFunctionAccumulator:
    """Second-order longitudinal structure functions D_LL(r) = <[q(x + r e_q) - q(x)]^2> for u along x and v along y,
    accumulated per (region, streamwise band) over sampled frames. The pair is assigned to the region/band of its
    first point. Inertial-range estimate: eps(r) = (D_LL / C2)^(3/2) / r, C2 ~ 2.0 (Kolmogorov constant for D_LL)."""

    def __init__(self, masks: dict[str, np.ndarray], r_px: list[int], *, x_axis: int = 2, y_axis: int = 1):
        self.masks, self.r_px = masks, list(r_px)
        self.x_axis, self.y_axis = x_axis, y_axis
        self.sums = {(c, name, r): 0.0 for c in "uv" for name in masks for r in self.r_px}
        self.counts = {key: 0 for key in self.sums}

    def add(self, u_flx: np.ndarray, v_flx: np.ndarray) -> None:
        for comp, arr, axis in (("u", u_flx, self.x_axis), ("v", v_flx, self.y_axis)):
            n = arr.shape[axis]
            for r in self.r_px:
                if r >= n:
                    continue
                a = np.take(arr, np.arange(r, n), axis=axis)
                b = np.take(arr, np.arange(0, n - r), axis=axis)
                d2 = (a - b) ** 2
                for name, mask in self.masks.items():
                    m2 = np.take(mask, np.arange(0, n - r), axis=axis - 1)  # mask is (y, x)
                    sel = d2[:, m2]
                    fin = np.isfinite(sel)
                    self.sums[(comp, name, r)] += float(sel[fin].sum())
                    self.counts[(comp, name, r)] += int(fin.sum())

    def results(self, spacing_m: float, *, c2: float = 2.0) -> dict:
        out = {}
        for name in self.masks:
            for comp in "uv":
                D = np.array([self.sums[(comp, name, r)] / max(self.counts[(comp, name, r)], 1) for r in self.r_px])
                r_m = np.array(self.r_px) * spacing_m
                out[(comp, name)] = {"r_m": r_m, "D_LL": D, "eps_r": (D / c2) ** 1.5 / r_m}
        return out


def save_arrays(targets: Iterable[tuple[Path | str, np.ndarray]]) -> None:
    """Save a collection of arrays to disk."""
    for path_like, arr in targets:
        path = Path(path_like)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
