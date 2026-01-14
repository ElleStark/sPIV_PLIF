"""
Plot an instantaneous snapshot of the fluctuating concentration field (c').

Edit the paths/settings below, then run:
    python tools/plot_c_snapshot.py
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
PLIF_DIR = BASE_PATH / "PLIF"
OUT_DIR = BASE_PATH / "Plots" / "Instantaneous"
CASE_NAME = "fractal"
FRAME_IDX = 200
C_PATH = PLIF_DIR / f"PLIF_{CASE_NAME}_smoothed.npy"
X_COORDS_PATH: Path | None = BASE_PATH / "x_coords.npy"
Y_COORDS_PATH: Path | None = BASE_PATH / "y_coords.npy"
USE_MEMMAP = True
C_CMAP = cmr.viola
C_VMIN: float | None = -0.5
C_VMAX: float | None = 0.5
FIG_DPI = 600
XLABEL = "x"
YLABEL = "y"
# -------------------------------------------------------------------


def _load_coords(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path)


def main() -> None:
    if not C_PATH.exists():
        raise FileNotFoundError(f"Missing concentration file: {C_PATH}")

    c_stack = np.load(C_PATH, mmap_mode="r" if USE_MEMMAP else None)
    if c_stack.ndim != 3:
        raise ValueError(f"Expected 3D concentration stack; got shape {c_stack.shape}")
    if FRAME_IDX < 0 or FRAME_IDX >= c_stack.shape[2]:
        raise IndexError(f"FRAME_IDX {FRAME_IDX} out of range for {c_stack.shape[2]} frames")

    c_mean = np.nanmean(c_stack, axis=2)
    c_frame = c_stack[:, :, FRAME_IDX] - c_mean

    x_coords = _load_coords(X_COORDS_PATH)
    y_coords = _load_coords(Y_COORDS_PATH)

    ny, nx = c_frame.shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x_coords/y_coords length must match the field grid.")

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")
    norm = Normalize(vmin=C_VMIN, vmax=C_VMAX)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(X, Y, c_frame, shading="auto", cmap=C_CMAP, norm=norm)
    fig.colorbar(h, ax=ax, label="c'")
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Instantaneous c' snapshot: {CASE_NAME}, frame {FRAME_IDX}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out_path = OUT_DIR / f"c_fluct_snapshot_{CASE_NAME}_frame{FRAME_IDX}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved c' snapshot to {out_path}")


if __name__ == "__main__":
    main()
