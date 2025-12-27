"""
Draw overlay figures: PLIF concentration as a colormap with PIV quiver on top.

Paths and settings are hard-coded below for convenience as a quick tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.io.load_processed_fields import load_fields  
from src.sPIV_PLIF_postprocessing.visualization.viz import save_overlay_quiver

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
U_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/9.26.2025_30cms_smSource_smTG15cm_neuHe0.137_air0.147_PIV0.01_iso_u.npy")
V_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/9.26.2025_30cms_smSource_smTG15cm_neuHe0.137_air0.147_PIV0.01_iso_v.npy")
W_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/9.26.2025_30cms_smSource_smTG15cm_neuHe0.137_air0.147_PIV0.01_iso_w.npy")
C_PATH = Path("E:/sPIV_PLIF_ProcessedData/PLIF/PLIF_small_source.npy")
FRAME_IDX = 46 # frame index to plot
OUT_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/Small_source/overlay_frame{FRAME_IDX}.png")
STRIDE_ROWS = 20  # stride along array rows (y dimension)
STRIDE_COLS = 15  # stride along array columns (x dimension)
SCALE = 0.04  # increase to shorten arrows
HEADWIDTH = 5.5  # optional, width of the arrow head
HEADLENGTH = 6  # optional, length of the arrow head
HEADAXISLENGTH = 4  # optional, length of the arrow head axis
TAILWIDTH = 0.002  # optional, width of the arrow tail
CMIN = 0.02  # minimum concentration for the colormap
CMAX = 1
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
USE_MEMMAP = False  # set True to load with mmap_mode='r'
LOAD_FRAME_ONLY = True  # set True to read just FRAME_IDX instead of full stacks
LOG_SCALE = True  # set True to plot concentration on a log scale
ARROW_COLOR = "k"  # optional, color of the arrows
CMAP_NAME = "cmr.rainforest_r"  # colormap name
CMAP_SLICE = (0.0, 0.75)  # tuple of (start, end) fractions of the colormap


def main() -> None:
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

    save_overlay_quiver(
        stacks.u,
        stacks.v,
        stacks.c,
        out_path=OUT_PATH,
        frame_idx=None if LOAD_FRAME_ONLY else FRAME_IDX,
        stride_rows=STRIDE_ROWS,
        stride_cols=STRIDE_COLS,
        scale=SCALE,
        headwidth=HEADWIDTH,
        headlength=HEADLENGTH,
        headaxislength=HEADAXISLENGTH,
        tailwidth=TAILWIDTH,
        cmin=CMIN,
        cmax=CMAX,
        x_coords=x_coords,
        y_coords=y_coords,
        log_scale=LOG_SCALE,
        title=f"Overlay frame {FRAME_IDX}",
        arrow_color=ARROW_COLOR,
        cmap_name=CMAP_NAME,
        cmap_slice=CMAP_SLICE,
    )
    print(f"Saved overlay to {OUT_PATH}")


if __name__ == "__main__":
    main()
