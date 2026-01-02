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
    u = np.load(base / "PIV" / f"piv_{case_name}_u.npy", mmap_mode=mmap_mode)[x_slice, y_slice, t_slice]
    v = np.load(base / "PIV" / f"piv_{case_name}_v.npy", mmap_mode=mmap_mode)[x_slice, y_slice, t_slice]
    w = np.load(base / "PIV" / f"piv_{case_name}_w.npy", mmap_mode=mmap_mode)[x_slice, y_slice, t_slice]
    return u, v, w


def load_mean_velocity_components(
    case_name: str,
    *,
    base_path: Path | str = DEFAULT_BASE_PATH,
    mmap_mode: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mean u, v, w components from the mean_fields archive."""
    base = Path(base_path)
    mean_fields = np.load(base / "mean_fields" / f"mean_fields_{case_name}.npz", mmap_mode=mmap_mode)
    return mean_fields["u"], mean_fields["v"], mean_fields["w"]


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
        u_mean = u_mean[:, :, None]
    if v.ndim == 3 and v_mean.ndim == 2:
        v_mean = v_mean[:, :, None]
    if w.ndim == 3 and w_mean.ndim == 2:
        w_mean = w_mean[:, :, None]

    u_flx = u - u_mean
    v_flx = v - v_mean
    w_flx = w - w_mean
    return u_flx, v_flx, w_flx


def compute_fluctuating_strain_rates(
    u_flx: np.ndarray,
    v_flx: np.ndarray,
    *,
    dx: float = DEFAULT_DX,
    dt: float = DEFAULT_DT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradients of fluctuating velocity components.

    Returns:
        duflx_dx, dvflx_dy, dwflx_dz
    """
    duflx_dx = np.gradient(u_flx, dx, axis=0)
    duflx_dx = np.asarray(duflx_dx)

    dvflx_dy = np.gradient(v_flx, dx, axis=1)
    dvflx_dy = np.asarray(dvflx_dy)

    dwflx_dz = 0 - duflx_dx - dvflx_dy  # from continuity equation

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
        np.mean(duflx_dx**2, axis=2) + np.mean(dvflx_dy**2, axis=2) + np.mean(dwflx_dz**2, axis=2)
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
    avg_rms = np.sqrt((1 / 3) * (np.mean(u_flx**2, axis=2) + np.mean(v_flx**2, axis=2) + np.mean(w_flx**2, axis=2)))
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
    u_mnsq = np.mean(u_flx**2, axis=2)
    v_mnsq = np.mean(v_flx**2, axis=2)
    w_mnsq = np.mean(w_flx**2, axis=2)
    tke = 0.5 * (u_mnsq + v_mnsq + w_mnsq)
    return tke, u_mnsq, v_mnsq, w_mnsq


def compute_turbulence_intensity(
    u_flx: np.ndarray,
    v_flx: np.ndarray,
    w_flx: np.ndarray,
    u_mean: float = 0.30,
) -> np.ndarray:
    """Compute turbulence intensity using RMS components and mean streamwise velocity."""
    t_intensity_avg = np.sqrt((1 / 3) * (np.mean(u_flx**2, axis=2) + np.mean(v_flx**2, axis=2) + np.mean(w_flx**2, axis=2))) / u_mean
    return t_intensity_avg


def save_arrays(targets: Iterable[tuple[Path | str, np.ndarray]]) -> None:
    """Save a collection of arrays to disk."""
    for path_like, arr in targets:
        path = Path(path_like)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
