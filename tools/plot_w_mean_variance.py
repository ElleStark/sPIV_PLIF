"""
Plot mean and variance of the vertical (w) velocity field for each case.

Edit the paths/settings below, then run:
    python tools/plot_w_mean_variance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import cmasher as cmr

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CASE_NAMES = ["nearbed", "fractal", "buoyant", "baseline"]
PIV_DIR = BASE_PATH / "PIV"
OUT_DIR = BASE_PATH / "Plots" / "w_stats"
X_COORDS_PATH: Path | None = BASE_PATH / "x_coords.npy"
Y_COORDS_PATH: Path | None = BASE_PATH / "y_coords.npy"
X_SLICE = slice(None)  # applied to axis 1 (x/columns)
Y_SLICE = slice(None)  # applied to axis 0 (y/rows)
T_SLICE = slice(None)
TIME_AXIS = 2  # axis along which mean/variance are computed
USE_MEMMAP = False
SAVE_ARRAYS = True
ARRAY_OUT_DIR = BASE_PATH / "mean_variance_fields"

MEAN_CMAP = cmr.viola_r
VAR_CMAP = cmr.rainforest_r
MEAN_VMIN: float | None = -0.1
MEAN_VMAX: float | None = 0.1
MEAN_SYMMETRIC = True  # if True and vmin/vmax unset, use symmetric limits about 0
VAR_VMIN: float | None = 0
VAR_VMAX: float | None = 0.01
VAR_LOG_SCALE = False
FIG_DPI = 600
XLABEL = "x"
YLABEL = "y"
# -------------------------------------------------------------------


def _load_coords(path: Path | None, slicer: slice) -> np.ndarray | None:
    if path is None:
        return None
    coords = np.load(path)
    return coords[slicer]


def _plot_field(
    field: np.ndarray,
    *,
    title: str,
    out_path: Path,
    cmap,
    vmin: float | None,
    vmax: float | None,
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
    log_scale: bool = False,
    cbar_label: str,
) -> None:
    ny, nx = field.shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x_coords/y_coords length must match the field grid.")

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    if log_scale:
        positive = field[np.isfinite(field) & (field > 0)]
        if positive.size == 0:
            raise ValueError(f"No positive values available for log scale in {title}")
        vmin_eff = vmin if vmin is not None and vmin > 0 else float(np.nanpercentile(positive, 1))
        vmax_eff = vmax if vmax is not None else float(np.nanpercentile(positive, 99))
        norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap, norm=norm)
    fig.colorbar(h, ax=ax, label=cbar_label)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _compute_mean_variance(w_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if w_stack.ndim != 3:
        raise ValueError(f"Expected 3D w stack; got shape {w_stack.shape}")
    w_mean = np.nanmean(w_stack, axis=TIME_AXIS)
    w_var = np.nanvar(w_stack, axis=TIME_AXIS)
    return w_mean, w_var


def main() -> None:
    x_coords_full = _load_coords(X_COORDS_PATH, X_SLICE)
    y_coords_full = _load_coords(Y_COORDS_PATH, Y_SLICE)

    for case in CASE_NAMES:
        w_path = PIV_DIR / f"piv_{case}_w.npy"
        if not w_path.exists():
            print(f"[skip] missing w file for case '{case}': {w_path}")
            continue
        w_stack = np.load(w_path, mmap_mode="r" if USE_MEMMAP else None)
        # w_stack = np.flipud(w_stack)
        # np.save(w_path, w_stack)  # overwrite with flipped version


        w_stack = w_stack[Y_SLICE, X_SLICE, T_SLICE]

        w_mean, w_var = _compute_mean_variance(w_stack)

        if MEAN_SYMMETRIC and MEAN_VMIN is None and MEAN_VMAX is None:
            abs_max = float(np.nanmax(np.abs(w_mean)))
            mean_vmin = -abs_max
            mean_vmax = abs_max
        else:
            mean_vmin = MEAN_VMIN
            mean_vmax = MEAN_VMAX

        mean_out = OUT_DIR / f"w_mean_{case}.png"
        var_out = OUT_DIR / f"w_var_{case}.png"

        _plot_field(
            w_mean,
            title=f"Mean vertical velocity (w): {case}",
            out_path=mean_out,
            cmap=MEAN_CMAP,
            vmin=mean_vmin,
            vmax=mean_vmax,
            x_coords=x_coords_full,
            y_coords=y_coords_full,
            log_scale=False,
            cbar_label="w mean",
        )
        _plot_field(
            w_var,
            title=f"Vertical velocity variance (w): {case}",
            out_path=var_out,
            cmap=VAR_CMAP,
            vmin=VAR_VMIN,
            vmax=VAR_VMAX,
            x_coords=x_coords_full,
            y_coords=y_coords_full,
            log_scale=VAR_LOG_SCALE,
            cbar_label="w variance",
        )

        if SAVE_ARRAYS:
            ARRAY_OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.save(ARRAY_OUT_DIR / f"{case}_w_mean.npy", w_mean)
            np.save(ARRAY_OUT_DIR / f"{case}_w_var.npy", w_var)

        print(f"[ok] saved mean/variance plots for case '{case}' to {OUT_DIR}")


if __name__ == "__main__":
    main()
