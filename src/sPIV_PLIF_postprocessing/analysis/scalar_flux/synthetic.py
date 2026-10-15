"""Synthetic fixed-plane datasets with known answers, for tests and pipeline checks.

Gaussian plume with a known eddy diffusivity
    Mean field (cross-stream x, downstream s = y_max - y, mean advection V in the -y direction):
        C(x, s) = A sigma0 / sigma(s) exp(-x^2 / (2 sigma(s)^2)),   sigma(s)^2 = sigma0^2 + 2 K_x s / U
    which is the Fickian/Taylor solution, so K_x = 1/2 U d(sigma^2)/ds holds exactly.
    Fluctuations follow a mixing-length closure: velocity fluctuations are AR(1) in time (coefficient phi) and
    independent between pixels, and
        c' = -(K_x / sigma_u^2) u' dC/dx - (K_y / sigma_v^2) v' dC/dy + noise
    so <u'c'> = -K_x dC/dx and <v'c'> = -K_y dC/dy exactly in expectation, and <w'c'> = 0.
Pure noise
    c and the velocities are independent noise: all fluxes are zero in expectation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def ar1(shape: tuple[int, ...], phi: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) series along axis 0. Integral time scale (in samples) = (1 + phi) / (2 (1 - phi))
    for the trapezoid-rule integral of rho(k) = phi^k."""
    e = rng.standard_normal(shape)
    x = np.empty(shape)
    x[0] = e[0]
    s = np.sqrt(1 - phi**2)
    for t in range(1, shape[0]):
        x[t] = phi * x[t - 1] + s * e[t]
    return x


def gaussian_plume_mean(x_mm: np.ndarray, s_mm: np.ndarray, *, amp: float, sigma0_mm: float, K: float, U: float):
    """Mean field C (ny, nx) and analytic dC/dx, dC/ds [per m], sigma(s) [mm]."""
    s_m, x_m = s_mm / 1000, x_mm / 1000
    sig2 = (sigma0_mm / 1000) ** 2 + 2 * K * s_m / U  # m^2
    sig = np.sqrt(sig2)[:, None]
    C = amp * (sigma0_mm / 1000) / sig * np.exp(-(x_m[None, :] ** 2) / (2 * sig**2))
    dCdx = -C * x_m[None, :] / sig**2
    # dC/ds = dC/dsig2 * dsig2/ds, dsig2/ds = 2K/U
    dCdsig2 = C * (-0.5 / sig**2 + x_m[None, :] ** 2 / (2 * sig**4))
    dCds = dCdsig2 * 2 * K / U
    return C, dCdx, dCds, np.sqrt(sig2) * 1000


def write_synthetic_case(
    out_dir: Path, *, kind: str = "plume", nt: int = 2400, ny: int = 64, nx: int = 72, d_mm: float = 1.0,
    K_x: float = 2e-4, K_y: float = 1e-4, U: float = 0.25, sigma_u: float = 0.02, sigma_v: float = 0.025,
    sigma_w: float = 0.015, phi: float = 0.8, amp: float = 1.0, sigma0_mm: float = 6.0, noise_c: float = 0.01,
    nan_edge_cols: int = 2, seed: int = 0, name: str = "synth",
) -> dict:
    """Write (t, y, x) float32 stacks piv_<name>_{u,v,w}.npy, <name>_PLIF.npy and grid files to out_dir.
    Returns the truth: {"C", "dCdx", "dCdy", "sigma_mm", "K_x", "K_y", "x_mm", "y_mm", ...}."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_mm = (np.arange(nx) - (nx - 1) / 2) * d_mm
    y_mm = np.arange(ny) * d_mm
    s_mm = y_mm.max() - y_mm  # source at the high-row end
    C, dCdx, dCds, sig_mm = gaussian_plume_mean(x_mm, s_mm, amp=amp, sigma0_mm=sigma0_mm, K=K_x, U=U)
    dCdy = -dCds  # y increases opposite to s

    up = sigma_u * ar1((nt, ny, nx), phi, rng)
    vp = sigma_v * ar1((nt, ny, nx), phi, rng)
    wp = sigma_w * ar1((nt, ny, nx), phi, rng)
    if kind == "plume":
        cp = -(K_x / sigma_u**2) * up * dCdx[None] - (K_y / sigma_v**2) * vp * dCdy[None]
        c = C[None] + cp + noise_c * rng.standard_normal((nt, ny, nx))
    elif kind == "noise":
        c = C[None] + noise_c * ar1((nt, ny, nx), phi, rng)
    else:
        raise ValueError(kind)
    u, v, w = up, -U + vp, wp
    if nan_edge_cols:
        for a in (u, v, w):
            a[:, :, :nan_edge_cols] = np.nan
    for comp, arr in (("u", u), ("v", v), ("w", w)):
        np.save(out_dir / f"piv_{name}_{comp}.npy", arr.astype(np.float32))
    np.save(out_dir / f"{name}_PLIF.npy", c.astype(np.float32))
    xg, yg = np.meshgrid(x_mm, y_mm)
    np.save(out_dir / f"{name}_xgrid.npy", xg.astype(np.float32))
    np.save(out_dir / f"{name}_ygrid.npy", yg.astype(np.float32))
    T_int_samples = (1 + phi) / (2 * (1 - phi))
    return {
        "C": C, "dCdx": dCdx, "dCdy": dCdy, "sigma_mm": sig_mm, "K_x": K_x if kind == "plume" else 0.0,
        "K_y": K_y if kind == "plume" else 0.0, "x_mm": x_mm, "y_mm": y_mm, "U": U, "nan_edge_cols": nan_edge_cols,
        "T_int_samples": T_int_samples, "sigma_u": sigma_u, "sigma_v": sigma_v,
    }
