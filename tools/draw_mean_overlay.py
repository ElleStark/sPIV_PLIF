"""
Draw an overlay of mean velocity/concentration fields from saved mean arrays.

Edit the paths/settings below, then run:
    python tools/draw_mean_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.visualization.viz import save_overlay_contour

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
MEAN_FIELDS_PATH = Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_smSource.npz")
OUT_PATH = Path("E:/sPIV_PLIF_ProcessedData/Plots/Mean/overlay_mean_smSource.png")
CMIN = 0.02
CMAX = 1.0
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
LOG_SCALE = True  # set True to plot concentration on a log scale
CMAP_NAME = "cmr.rainforest_r"
CMAP_SLICE = (0.0, 0.75)
CONTOUR_LEVELS: int | list[float] | None = None  # disable contours
CONTOUR_COLOR = "#555555"
CONTOUR_WIDTH = 0.75
CONTOUR_BOX_FRACTION: tuple[float, float, float, float] | None = None  # (xmin_frac, xmax_frac, ymin_frac, ymax_frac); set None to disable box handling
CONTOUR_LEVELS_IN_BOX: int | list[float] | None = None  # unused when contours disabled
CONTOUR_WIDTH_IN_BOX: float | None = 0.8
CONTOUR_COLOR_IN_BOX: str | None = None  # slightly lighter than main color
CONTOUR_CMAP: str | None = "cmr.ember"  # use cmasher ember gradient
CONTOUR_CMAP_IN_BOX: str | None = "cmr.ember"
CONTOUR_LABELS = True 
CONTOUR_LABELS_IN_BOX: bool | None = False
SHOW_QUIVER = True  # enable arrows
QUIVER_CMAP: str | None = "cmr.ember"
QUIVER_COLORBAR = True
QUIVER_ALPHA = 1.0
QUIVER_VMIN: float | None = 0.1  # set to fix arrow color scale 
QUIVER_VMAX: float | None = 0.45  # set to fix arrow color scale max
stride_rows = 30  # stride along array rows (y dimension)
stride_cols = 20  # stride along array columns (x dimension)
scale = 0.03  # increase to shorten arrows
headwidth = 4.5  # optional, width of the arrow head
headlength = 5  # optional, length of the arrow head
headaxislength = 3.5  # optional, length of the arrow head axis
tailwidth = 0.003  # optional, width of the arrow tail


def main() -> None:
    if not MEAN_FIELDS_PATH.exists():
        raise FileNotFoundError(f"Mean fields file not found: {MEAN_FIELDS_PATH}")

    mean_data = np.load(MEAN_FIELDS_PATH)
    for key in ("u", "v", "c"):
        if key not in mean_data:
            raise KeyError(f"Missing '{key}' in {MEAN_FIELDS_PATH}")

    u_mean = np.array(mean_data["u"], copy=False)
    v_mean = np.array(mean_data["v"], copy=False)
    c_mean = np.array(mean_data["c"], copy=False)

    x_coords = np.load(X_PATH) if X_PATH else None
    y_coords = np.load(Y_PATH) if Y_PATH else None
    contour_box = None
    if CONTOUR_BOX_FRACTION is not None:
        xmin_f, xmax_f, ymin_f, ymax_f = CONTOUR_BOX_FRACTION
        x_min, x_max = (
            (float(np.min(x_coords)), float(np.max(x_coords))) if x_coords is not None else (0.0, float(u_mean.shape[1] - 1))
        )
        y_min, y_max = (
            (float(np.min(y_coords)), float(np.max(y_coords))) if y_coords is not None else (0.0, float(u_mean.shape[0] - 1))
        )
        contour_box = (
            x_min + xmin_f * (x_max - x_min),
            x_min + xmax_f * (x_max - x_min),
            y_min + ymin_f * (y_max - y_min),
            y_min + ymax_f * (y_max - y_min),
        )

    save_overlay_contour(
        u_mean,
        v_mean,
        c_mean,
        out_path=OUT_PATH,
        frame_idx=None,  # 2D arrays already sliced
        cmin=CMIN,
        cmax=CMAX,
        x_coords=x_coords,
        y_coords=y_coords,
        log_scale=LOG_SCALE,
        title="Mean field overlay (contours)",
        cmap_name=CMAP_NAME,
        cmap_slice=CMAP_SLICE,
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
        quiver_colorbar=QUIVER_COLORBAR,
        quiver_alpha=QUIVER_ALPHA,
        quiver_vmin=QUIVER_VMIN,
        quiver_vmax=QUIVER_VMAX,
        stride_rows=stride_rows,
        stride_cols=stride_cols,
        quiver_scale=scale,
        quiver_headwidth=headwidth,
        quiver_headlength=headlength,
        quiver_headaxislength=headaxislength,
        quiver_tailwidth=tailwidth,
    )
    print(f"Saved mean overlay to {OUT_PATH}")


if __name__ == "__main__":
    main()
