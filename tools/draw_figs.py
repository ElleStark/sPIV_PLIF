"""
Draw overlay figures: PLIF concentration as a colormap with PIV quiver on top.

Paths and settings are hard-coded below for convenience as a quick tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from sPIV_PLIF_postprocessing.io.load_processed_fields import load_fields  # noqa: E402

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
U_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_u.npy")
V_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_v.npy")
W_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_w.npy")
C_PATH = Path("E:/sPIV_PLIF_ProcessedData/PLIF/8.29_30cmsPWM2.25_smTG15cm_noHC_PLIFairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_c.npy")
FRAME_IDX = 0
OUT_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/Baseline/overlay_frame{FRAME_IDX}.png")
STRIDE = 10
SCALE = 0.05  # increase to shorten arrows
CMIN = None
CMAX = None
X_PATH: Path | None = None  # optional npy for x coords (1D)
Y_PATH: Path | None = None  # optional npy for y coords (1D)
USE_MEMMAP = False  # set True to load with mmap_mode='r'


def main() -> None:
    stacks = load_fields(
        U_PATH,
        V_PATH,
        W_PATH,
        C_PATH,
        enforce_float32=True,
        mmap_mode="r" if USE_MEMMAP else None,
    )

    u = stacks.u
    v = stacks.v
    c = stacks.c

    if FRAME_IDX < 0 or FRAME_IDX >= u.shape[2]:
        raise IndexError(f"frame {FRAME_IDX} out of range for {u.shape[2]} frames")

    u_f = u[:, :, FRAME_IDX]
    v_f = v[:, :, FRAME_IDX]
    c_f = c[:, :, FRAME_IDX] if c.ndim == 3 else c

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

    # Concentration colormap (first 0-80% of chroms)
    cmap = cmr.get_sub_cmap("cmr.chroms", 0.0, 0.8)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(
        X,
        Y,
        np.where(mask_c, c_f, np.nan),
        cmap=cmap,
        shading="auto",
        vmin=CMIN,
        vmax=CMAX,
    )
    fig.colorbar(h, ax=ax, label="c")

    # Quiver with stride
    Xs = X[::STRIDE, ::STRIDE]
    Ys = Y[::STRIDE, ::STRIDE]
    us = u_f[::STRIDE, ::STRIDE]
    vs = v_f[::STRIDE, ::STRIDE]
    ms = mask_vec[::STRIDE, ::STRIDE]

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
        width=0.002,
        pivot="mid",
        color="k",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Overlay frame {FRAME_IDX}")
    ax.axis("equal")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved overlay to {OUT_PATH}")


if __name__ == "__main__":
    main()
