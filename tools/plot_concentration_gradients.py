"""
Load concentration data, compute spatial/temporal gradients, and plot:
1) time-averaged signed spatial gradients <dc/dx>_t and <dc/dy>_t
2) time-averaged signed products <(dc/dx)*(dc/dt)>_t and <(dc/dy)*(dc/dt)>_t

Run:
    python tools/plot_concentration_gradients.py
"""

from __future__ import annotations

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "baseline"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
OUT_DIR = BASE_PATH / "Plots" / "concentration_gradients"

X_SLICE = slice(0, 600)
Y_SLICE = slice(100, 500)
T_SLICE = slice(0,500)  # e.g., slice(0, 2000)

# Grid/time spacing used in gradient computations
DX = 0.5
DY = 0.5
DT = 0.05

XLABEL = "x index"
YLABEL = "y index"
FIG_DPI = 600

# Plot controls
SPATIAL_CMAP = "RdBu_r"
PRODUCT_CMAP = "RdBu_r"
SPATIAL_LOG_SCALE = False  # signed fields should generally stay linear
SPATIAL_VMIN = None  # None -> symmetric auto range
SPATIAL_VMAX = None  # None -> symmetric auto range
PRODUCT_VMIN = None  # None -> symmetric auto range
PRODUCT_VMAX = None  # None -> symmetric auto range
# -------------------------------------------------------------------


def _load_concentration_stack() -> np.ndarray:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")
    conc = np.load(CONCENTRATION_PATH, mmap_mode="r")
    if conc.ndim != 3:
        raise ValueError(f"Expected 3D concentration stack; got shape {conc.shape}")
    return np.asarray(conc[X_SLICE, Y_SLICE, T_SLICE], dtype=float)


def _compute_gradient_fields(conc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dcdx = np.gradient(conc, DX, axis=0)
    dcdy = np.gradient(conc, DY, axis=1)
    dcdt = np.gradient(conc, DT, axis=2)
    return dcdx, dcdy, dcdt


def _plot_field(
    field_2d: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    fig_path: Path,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
) -> None:
    x_offset = X_SLICE.start or 0
    y_offset = Y_SLICE.start or 0
    x_idx = np.arange(field_2d.shape[0]) + x_offset
    y_idx = np.arange(field_2d.shape[1]) + y_offset

    norm = None
    if log_scale:
        finite_pos = field_2d[np.isfinite(field_2d) & (field_2d > 0)]
        if finite_pos.size == 0:
            raise ValueError("Log scale requested, but field has no positive finite values.")
        vmin_eff = vmin if vmin is not None else float(np.nanmin(finite_pos))
        vmax_eff = vmax if vmax is not None else float(np.nanmax(finite_pos))
        norm = LogNorm(vmin=max(vmin_eff, 1e-12), vmax=vmax_eff)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(
        y_idx,
        x_idx,
        field_2d,
        shading="auto",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h, ax=ax, label=cbar_label)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    conc = _load_concentration_stack()
    dcdx, dcdy, dcdt = _compute_gradient_fields(conc)

    dcdx_mean = np.nanmean(dcdx, axis=2)
    dcdy_mean = np.nanmean(dcdy, axis=2)
    dcdx_dcdt_mean = np.nanmean(dcdx * dcdt, axis=2)
    dcdy_dcdt_mean = np.nanmean(dcdy * dcdt, axis=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"dcdx_mean_{CASE_NAME}.npy", dcdx_mean)
    np.save(OUT_DIR / f"dcdy_mean_{CASE_NAME}.npy", dcdy_mean)
    np.save(OUT_DIR / f"dcdx_times_dcdt_mean_{CASE_NAME}.npy", dcdx_dcdt_mean)
    np.save(OUT_DIR / f"dcdy_times_dcdt_mean_{CASE_NAME}.npy", dcdy_dcdt_mean)

    spatial_vmin = SPATIAL_VMIN
    spatial_vmax = SPATIAL_VMAX
    if spatial_vmin is None or spatial_vmax is None:
        merged = np.concatenate([dcdx_mean.ravel(), dcdy_mean.ravel()])
        lim = float(np.nanmax(np.abs(merged))) if np.any(np.isfinite(merged)) else 1.0
        lim = max(lim, 1e-12)
        spatial_vmin = -lim
        spatial_vmax = lim

    _plot_field(
        dcdx_mean,
        title=f"Time-averaged signed spatial gradient <dc/dx>_t\ncase={CASE_NAME}",
        cbar_label="<dc/dx>_t",
        fig_path=OUT_DIR / f"dcdx_mean_{CASE_NAME}.png",
        cmap=SPATIAL_CMAP,
        vmin=spatial_vmin,
        vmax=spatial_vmax,
        log_scale=SPATIAL_LOG_SCALE,
    )

    _plot_field(
        dcdy_mean,
        title=f"Time-averaged signed spatial gradient <dc/dy>_t\ncase={CASE_NAME}",
        cbar_label="<dc/dy>_t",
        fig_path=OUT_DIR / f"dcdy_mean_{CASE_NAME}.png",
        cmap=SPATIAL_CMAP,
        vmin=spatial_vmin,
        vmax=spatial_vmax,
        log_scale=SPATIAL_LOG_SCALE,
    )

    prod_vmin = PRODUCT_VMIN
    prod_vmax = PRODUCT_VMAX
    if prod_vmin is None or prod_vmax is None:
        merged = np.concatenate([dcdx_dcdt_mean.ravel(), dcdy_dcdt_mean.ravel()])
        lim = float(np.nanmax(np.abs(merged))) if np.any(np.isfinite(merged)) else 1.0
        lim = max(lim, 1e-12)
        prod_vmin = -lim
        prod_vmax = lim

    _plot_field(
        dcdx_dcdt_mean,
        title=f"Time-averaged signed product <(dc/dx)*(dc/dt)>_t\ncase={CASE_NAME}",
        cbar_label="<(dc/dx)*(dc/dt)>_t",
        fig_path=OUT_DIR / f"dcdx_times_dcdt_mean_{CASE_NAME}.png",
        cmap=PRODUCT_CMAP,
        vmin=prod_vmin,
        vmax=prod_vmax,
        log_scale=False,
    )

    _plot_field(
        dcdy_dcdt_mean,
        title=f"Time-averaged signed product <(dc/dy)*(dc/dt)>_t\ncase={CASE_NAME}",
        cbar_label="<(dc/dy)*(dc/dt)>_t",
        fig_path=OUT_DIR / f"dcdy_times_dcdt_mean_{CASE_NAME}.png",
        cmap=PRODUCT_CMAP,
        vmin=prod_vmin,
        vmax=prod_vmax,
        log_scale=False,
    )

    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
