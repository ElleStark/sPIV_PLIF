"""
Plot an instantaneous snapshot of the vertical (w) velocity field
or the instantaneous scalar flux w'c'.

Edit the paths/settings below, then run:
    python tools/plot_w_snapshot.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cmasher as cmr

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
PIV_DIR = BASE_PATH / "PIV"
PLIF_DIR = BASE_PATH / "PLIF"
OUT_DIR = BASE_PATH / "Plots" / "Instantaneous"
CASE_NAME = "fractal"
FRAME_IDX = 200
W_PATH = PIV_DIR / f"piv_{CASE_NAME}_w.npy"
C_PATH = PLIF_DIR / f"PLIF_{CASE_NAME}_smoothed.npy"
X_COORDS_PATH: Path | None = BASE_PATH / "x_coords.npy"
Y_COORDS_PATH: Path | None = BASE_PATH / "y_coords.npy"
USE_MEMMAP = True
PLOT_MODE = "wc"  # "w" for w' or "wc" for w'c'
W_CMAP = cmr.viola
W_VMIN: float | None = -0.2
W_VMAX: float | None = 0.2
WC_CMAP = cmr.prinsenvlag_r
WC_VMIN: float | None = -0.005
WC_VMAX: float | None = 0.005
FIG_DPI = 600
XLABEL = "x"
YLABEL = "y"
# -------------------------------------------------------------------


def _load_coords(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path)


def main() -> None:
    if not W_PATH.exists():
        raise FileNotFoundError(f"Missing w file: {W_PATH}")

    w_stack = np.load(W_PATH, mmap_mode="r" if USE_MEMMAP else None)
    if w_stack.ndim != 3:
        raise ValueError(f"Expected 3D w stack; got shape {w_stack.shape}")
    if FRAME_IDX < 0 or FRAME_IDX >= w_stack.shape[2]:
        raise IndexError(f"FRAME_IDX {FRAME_IDX} out of range for {w_stack.shape[2]} frames")

    w_mean = np.nanmean(w_stack, axis=2)
    w_frame = w_stack[:, :, FRAME_IDX] - w_mean

    if PLOT_MODE not in {"w", "wc"}:
        raise ValueError("PLOT_MODE must be 'w' or 'wc'.")

    if PLOT_MODE == "wc":
        if not C_PATH.exists():
            raise FileNotFoundError(f"Missing concentration file: {C_PATH}")
        c_stack = np.load(C_PATH, mmap_mode="r" if USE_MEMMAP else None)
        if c_stack.ndim != 3:
            raise ValueError(f"Expected 3D concentration stack; got shape {c_stack.shape}")
        if c_stack.shape[:2] != w_stack.shape[:2]:
            raise ValueError("w and c stacks must share the same spatial shape.")
        if FRAME_IDX < 0 or FRAME_IDX >= c_stack.shape[2]:
            raise IndexError(f"FRAME_IDX {FRAME_IDX} out of range for {c_stack.shape[2]} frames")
        c_mean = np.nanmean(c_stack, axis=2)
        c_frame = c_stack[:, :, FRAME_IDX] - c_mean
        plot_field = w_frame * c_frame
        cmap = WC_CMAP
        norm = Normalize(vmin=WC_VMIN, vmax=WC_VMAX)
        cbar_label = "w'c'"
        title = f"Instantaneous w'c' snapshot: {CASE_NAME}, frame {FRAME_IDX}"
        out_name = f"wc_snapshot_{CASE_NAME}_frame{FRAME_IDX}.png"
    else:
        plot_field = w_frame
        cmap = W_CMAP
        norm = Normalize(vmin=W_VMIN, vmax=W_VMAX)
        cbar_label = "w'"
        title = f"Instantaneous w' snapshot: {CASE_NAME}, frame {FRAME_IDX}"
        out_name = f"w_snapshot_{CASE_NAME}_frame{FRAME_IDX}.png"
    x_coords = _load_coords(X_COORDS_PATH)
    y_coords = _load_coords(Y_COORDS_PATH)

    ny, nx = w_frame.shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x_coords/y_coords length must match the field grid.")

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(X, Y, plot_field, shading="auto", cmap=cmap, norm=norm)
    fig.colorbar(h, ax=ax, label=cbar_label)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out_path = OUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved w snapshot to {out_path}")


if __name__ == "__main__":
    main()
