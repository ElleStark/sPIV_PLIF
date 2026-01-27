"""
Draw instantaneous overlay: concentration pcolormesh + colored quiver arrows.

Edit the paths/settings below, then run:
    python tools/draw_inst_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.io.load_processed_fields import load_fields
from src.sPIV_PLIF_postprocessing.visualization.viz import save_overlay_contour

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
CASE_NAME = "fractal"  # used to build file paths
U_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PIV/piv_{CASE_NAME}_u.npy")
V_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PIV/piv_{CASE_NAME}_v.npy")
W_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PIV/piv_{CASE_NAME}_w.npy")
C_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/Old/plif_{CASE_NAME}.npy")
FRAME_IDX = 216  # frame index to plot
OUT_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/{CASE_NAME}/frame{FRAME_IDX}_res.png")
CMIN = 0.01
CMAX = 1.0
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
LOG_SCALE = True  # set True to plot concentration on a log scale
CMAP_NAME = cmr.rainforest # jet for concentration
CMAP_SLICE = (0.0, 1)
C_UNDER: str | None = None  # fade in from white
C_UNDER_TRANSITION: float | None = None  # fraction of cmap for white->jet blend
C_UNDER_START: float | None = None
C_UNDER_END: float | None = None
# CMAP_NAME = "jet"  # jet for concentration
# CMAP_SLICE = (0.0, 1.0)
# C_UNDER: str | None = "white"  # fade in from white
# C_UNDER_TRANSITION: float | None = 0.1  # fraction of cmap for white->jet blend
# C_UNDER_START: float | None = 0.01
# C_UNDER_END: float | None = 0.02
PCOLORMESH_ALPHA = 0.85  # reduce saturation/opacity of the concentration field
X_LIMITS: tuple[float, float] | None = (10, 40)
X_SUBSET: tuple[float, float] | None = (10, 40)
Y_SUBSET: tuple[float, float] | None = (85, 115)
# X_LIMITS: tuple[float, float] | None = (-20.0, 0.0)
# X_SUBSET: tuple[float, float] | None = (-20.0, 0.0)
# Y_SUBSET: tuple[float, float] | None = (145.0, 165.0)
CONTOUR_LEVELS: int | list[float] | None = None  # disable contours for snapshots
CONTOUR_COLOR = "#555555"
CONTOUR_WIDTH = 0.75
CONTOUR_BOX_FRACTION: tuple[float, float, float, float] | None = None
CONTOUR_LEVELS_IN_BOX: int | list[float] | None = None
CONTOUR_WIDTH_IN_BOX: float | None = 0.8
CONTOUR_COLOR_IN_BOX: str | None = None
CONTOUR_CMAP: str | None = "cmr.ember"
CONTOUR_CMAP_IN_BOX: str | None = "cmr.ember"
CONTOUR_LABELS = False
CONTOUR_LABELS_IN_BOX: bool | None = False

# Quiver settings (match mean overlay)
SHOW_QUIVER = False
QUIVER_CMAP: str | None = None
QUIVER_COLOR = "#333333"  # medium gray arrows
QUIVER_COLORBAR = False
QUIVER_ALPHA = 1.0
QUIVER_VMIN: float | None = 0.1
QUIVER_VMAX: float | None = 0.5
STRIDE_ROWS = 1
STRIDE_COLS = 1
QUIVER_SCALE = 0.75
QUIVER_HEADWIDTH = 2.5
QUIVER_HEADLENGTH = 3.0
QUIVER_HEADAXISLENGTH = 2.5
QUIVER_TAILWIDTH = 0.003

USE_MEMMAP = False  # set True to load with mmap_mode='r'
LOAD_FRAME_ONLY = True  # True loads just FRAME_IDX; False loads full stacks
USE_DARK_BACKGROUND = True
APPLY_MEDIAN_SMOOTH = True
MEDIAN_WINDOW = 9  # pixels


def _median_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply a kxk median filter (per frame if 3D)."""
    if k % 2 == 0 or k < 1:
        raise ValueError("Median window size must be an odd positive integer.")

    def _smooth2d(a: np.ndarray) -> np.ndarray:
        pad = k // 2
        padded = np.pad(a, pad_width=pad, mode="edge")
        windows = sliding_window_view(padded, (k, k))
        return np.nanmedian(windows, axis=(-2, -1))

    if arr.ndim == 2:
        return _smooth2d(arr)
    if arr.ndim == 3:
        return np.stack([_smooth2d(arr[:, :, i]) for i in range(arr.shape[2])], axis=2)
    raise ValueError(f"Median smooth expects 2D or 3D input; got shape {arr.shape}")


def _subset_xy(
    u: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
    x_subset: tuple[float, float] | None,
    y_subset: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subset arrays and coordinates to requested x/y bounds."""
    ny, nx = u.shape[:2]
    x_coords_arr = np.asarray(x_coords) if x_coords is not None else np.arange(nx)
    y_coords_arr = np.asarray(y_coords) if y_coords is not None else np.arange(ny)

    if x_coords_arr.ndim != 1 or y_coords_arr.ndim != 1:
        raise ValueError("x_coords and y_coords must be 1D when subsetting.")
    if len(x_coords_arr) != nx or len(y_coords_arr) != ny:
        raise ValueError("x_coords/y_coords length must match the data grid dimensions.")

    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    if x_subset is not None:
        x_min, x_max = sorted(x_subset)
        mask_x = (x_coords_arr >= x_min) & (x_coords_arr <= x_max)
        if not np.any(mask_x):
            raise ValueError(f"No x points fall within requested X_SUBSET {x_subset}.")
        x_idx = np.where(mask_x)[0]
    if y_subset is not None:
        y_min, y_max = sorted(y_subset)
        mask_y = (y_coords_arr >= y_min) & (y_coords_arr <= y_max)
        if not np.any(mask_y):
            raise ValueError(f"No y points fall within requested Y_SUBSET {y_subset}.")
        y_idx = np.where(mask_y)[0]

    def _slice(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[np.ix_(y_idx, x_idx)]
        if arr.ndim == 3:
            return arr[np.ix_(y_idx, x_idx, np.arange(arr.shape[2]))]
        raise ValueError(f"Expected 2D or 3D arrays for subsetting; got {arr.shape}")

    return (
        _slice(u),
        _slice(v),
        _slice(c),
        x_coords_arr[x_idx],
        y_coords_arr[y_idx],
    )


def main() -> None:
    if USE_DARK_BACKGROUND:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = "black"
        plt.rcParams["figure.facecolor"] = "black"
    else:
        plt.style.use("default")
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
    load_frame_idx = FRAME_IDX if LOAD_FRAME_ONLY else None
    stacks = load_fields(
        U_PATH,
        V_PATH,
        W_PATH,
        C_PATH,
        enforce_float32=True,
        mmap_mode="r" if USE_MEMMAP else None,
        frame_idx=load_frame_idx,
    )

    x_coords = np.load(X_PATH) if X_PATH else None
    y_coords = np.load(Y_PATH) if Y_PATH else None
    u_plot = stacks.u
    v_plot = stacks.v
    c_plot = _median_smooth(stacks.c, MEDIAN_WINDOW) if APPLY_MEDIAN_SMOOTH else stacks.c

    if X_SUBSET is not None or Y_SUBSET is not None:
        u_plot, v_plot, c_plot, x_coords, y_coords = _subset_xy(
            u_plot,
            v_plot,
            c_plot,
            x_coords,
            y_coords,
            X_SUBSET,
            Y_SUBSET,
        )

    contour_box = None
    if CONTOUR_BOX_FRACTION is not None:
        xmin_f, xmax_f, ymin_f, ymax_f = CONTOUR_BOX_FRACTION
        x_min, x_max = (
            (float(np.min(x_coords)), float(np.max(x_coords))) if x_coords is not None else (0.0, float(u_plot.shape[1] - 1))
        )
        y_min, y_max = (
            (float(np.min(y_coords)), float(np.max(y_coords))) if y_coords is not None else (0.0, float(u_plot.shape[0] - 1))
        )
        contour_box = (
            x_min + xmin_f * (x_max - x_min),
            x_min + xmax_f * (x_max - x_min),
            y_min + ymin_f * (y_max - y_min),
            y_min + ymax_f * (y_max - y_min),
        )

    save_overlay_contour(
        u_plot,
        v_plot,
        c_plot,
        out_path=OUT_PATH,
        frame_idx=None if LOAD_FRAME_ONLY else FRAME_IDX,
        cmin=CMIN,
        cmax=CMAX,
        x_coords=x_coords,
        y_coords=y_coords,
        log_scale=LOG_SCALE,
        title=f"Instantaneous overlay frame {FRAME_IDX}",
        cmap_name=CMAP_NAME,
        cmap_slice=CMAP_SLICE,
        cmap_under=C_UNDER,
        cmap_under_transition=C_UNDER_TRANSITION,
        cmap_under_start=C_UNDER_START,
        cmap_under_end=C_UNDER_END,
        pcolormesh_alpha=PCOLORMESH_ALPHA,
        xlim=X_LIMITS,
        contour_levels=CONTOUR_LEVELS,
        contour_color=CONTOUR_COLOR,
        contour_width=CONTOUR_WIDTH,
        contour_box=contour_box,
        contour_levels_in_box=CONTOUR_LEVELS_IN_BOX,
        contour_color_in_box=CONTOUR_COLOR_IN_BOX,
        contour_width_in_box=CONTOUR_WIDTH_IN_BOX,
        contour_cmap=CONTOUR_CMAP,
        contour_cmap_in_box=CONTOUR_CMAP_IN_BOX,
        contour_labels=CONTOUR_LABELS,
        contour_labels_in_box=CONTOUR_LABELS_IN_BOX,
        show_quiver=SHOW_QUIVER,
        quiver_cmap=QUIVER_CMAP,
        quiver_color=QUIVER_COLOR,
        quiver_colorbar=QUIVER_COLORBAR,
        quiver_alpha=QUIVER_ALPHA,
        quiver_vmin=QUIVER_VMIN,
        quiver_vmax=QUIVER_VMAX,
        stride_rows=STRIDE_ROWS,
        stride_cols=STRIDE_COLS,
        quiver_scale=QUIVER_SCALE,
        quiver_headwidth=QUIVER_HEADWIDTH,
        quiver_headlength=QUIVER_HEADLENGTH,
        quiver_headaxislength=QUIVER_HEADAXISLENGTH,
        quiver_tailwidth=QUIVER_TAILWIDTH,
    )
    print(f"Saved instantaneous overlay to {OUT_PATH}")


if __name__ == "__main__":
    main()
