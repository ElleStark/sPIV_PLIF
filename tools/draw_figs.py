"""
Draw overlay figures: PLIF concentration as a colormap with PIV quiver on top.

Paths and settings are hard-coded below for convenience as a quick tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.io.load_processed_fields import load_fields  

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
U_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_u.npy")
V_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_v.npy")
W_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_w.npy")
C_PATH = Path("E:/sPIV_PLIF_ProcessedData/PLIF/PLIF_baseline.npy")
FRAME_IDX = 0
OUT_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/Baseline/overlay_frame{FRAME_IDX}.png")
STRIDE_X = 20
STRIDE_Y = 15
SCALE = 0.04  # increase to shorten arrows
HEADWIDTH = 5.5  # optional, width of the arrow head
HEADLENGTH = 6  # optional, length of the arrow head
HEADAXISLENGTH = 4  # optional, length of the arrow head axis
TAILWIDTH = 0.002  # optional, width of the arrow tail
CMIN = 0.01
CMAX = 1
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
USE_MEMMAP = False  # set True to load with mmap_mode='r'
LOAD_FRAME_ONLY = True  # set True to read just FRAME_IDX instead of full stacks
LOG_SCALE = True  # set True to plot concentration on a log scale


def main() -> None:
    frame_idx = FRAME_IDX if LOAD_FRAME_ONLY else None
    stacks = load_fields(
        U_PATH,
        V_PATH,
        W_PATH,
        C_PATH,
        enforce_float32=True,
        mmap_mode="r" if USE_MEMMAP else None,
        frame_idx=frame_idx,
    )

    u = stacks.u
    v = stacks.v
    c = stacks.c

    if frame_idx is None:
        if FRAME_IDX < 0 or FRAME_IDX >= u.shape[2]:
            raise IndexError(f"frame {FRAME_IDX} out of range for {u.shape[2]} frames")
        u_f = u[:, :, FRAME_IDX]
        v_f = v[:, :, FRAME_IDX]
        c_f = c[:, :, FRAME_IDX] if c.ndim == 3 else c
    else:
        u_f = u
        v_f = v
        c_f = c

    ny, nx = u_f.shape
    x_coords = np.load(X_PATH) if X_PATH else np.arange(nx)
    y_coords = np.load(Y_PATH) if Y_PATH else np.arange(ny)

    if x_coords.ndim != 1 or y_coords.ndim != 1:
        raise ValueError("x and y coordinate arrays must be 1D.")
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x/y lengths must match data grid dimensions.")

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    # Mask invalids
    mask_vec = np.isfinite(u_f) & np.isfinite(v_f)
    mask_c = np.isfinite(c_f)
    if LOG_SCALE:
        mask_c &= c_f > 0  # LogNorm requires positive values

    # Concentration colormap: first 65% of rainforest_r
    cmap = cmr.get_sub_cmap("cmr.rainforest_r", 0.0, 0.65)

    # Set normalization: log if requested, otherwise linear bounds.
    norm = None
    if LOG_SCALE:
        if not np.any(mask_c):
            raise ValueError("No positive concentration values available for log scale.")
        data_pos = c_f[mask_c]
        vmin = CMIN if CMIN is not None and CMIN > 0 else float(np.nanmin(data_pos))
        vmax = CMAX if CMAX is not None else float(np.nanmax(data_pos))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin_kw = None
        vmax_kw = None
    else:
        vmin_kw = CMIN
        vmax_kw = CMAX

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(
        X,
        Y,
        np.ma.array(c_f, mask=~mask_c),
        cmap=cmap,
        shading="auto",
        vmin=vmin_kw,
        vmax=vmax_kw,
        norm=norm,
    )
    fig.colorbar(h, ax=ax, label="c")

    # Quiver with stride
    Xs = X[::STRIDE_X, ::STRIDE_Y]
    Ys = Y[::STRIDE_X, ::STRIDE_Y]
    us = u_f[::STRIDE_X, ::STRIDE_Y]
    vs = v_f[::STRIDE_X, ::STRIDE_Y]
    ms = mask_vec[::STRIDE_X, ::STRIDE_Y]
    Xs = Xs[ms]
    Ys = Ys[ms]
    us = us[ms]
    vs = vs[ms]

    ax.quiver(
        Xs,
        Ys,
        us,
        vs,
        angles="xy",
        scale_units="xy",
        scale=SCALE,
        headwidth=HEADWIDTH,
        headlength=HEADLENGTH,
        width=TAILWIDTH,
        headaxislength=HEADAXISLENGTH,
        pivot="mid",
        color="k",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Overlay frame {FRAME_IDX}")
    ax.set_aspect("equal", adjustable='box')
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved overlay to {OUT_PATH}")


if __name__ == "__main__":
    main()
