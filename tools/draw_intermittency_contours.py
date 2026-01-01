"""
Compute intermittency (fraction of time concentration exceeds a threshold) and plot contours.

Edit the paths/settings below, then run:
    python tools/draw_intermittency_contours.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.analysis import compute_intermittency

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
CASE_NAME = "baseline"
CONCENTRATION_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/plif_{CASE_NAME}_smoothed.npy")
X_COORDS_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_COORDS_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
OUT_INTERMITTENCY_PATH: Path | None = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Intermittency/intermittency_{CASE_NAME}.npy")
LOAD_INTERMITTENCY_PATH: Path | None = OUT_INTERMITTENCY_PATH  # set to None to force recompute
OUT_FIG_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Intermittency/intermittency_contours_{CASE_NAME}.png")
THRESHOLD = 0.02  # concentration threshold defining "present" events
FIGSIZE: tuple[float, float] = (7.0, 5.5)
CONTOUR_LEVELS: Sequence[float] | int = np.linspace(0.1, 0.9, 5)
CONTOUR_CMAP = cmr.get_sub_cmap("cmr.ocean_r", 0.15, 1.0)
CONTOUR_LINE_WIDTH = 1.5


def _load_coords(path: Path | None, expected_len: int, name: str) -> np.ndarray:
    """Load 1D coordinate array or fall back to pixel indices."""
    if path is None:
        return np.arange(expected_len)
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name}-coords must be 1D; got shape {arr.shape} from {path}")
    if len(arr) != expected_len:
        raise ValueError(
            f"{name}-coords length {len(arr)} does not match intermittency {name}-dimension {expected_len}"
        )
    return arr


def main() -> None:
    intermittency: np.ndarray
    if LOAD_INTERMITTENCY_PATH is not None and LOAD_INTERMITTENCY_PATH.exists():
        intermittency = np.load(LOAD_INTERMITTENCY_PATH)
        print(f"Loaded intermittency array from {LOAD_INTERMITTENCY_PATH}")
    else:
        if not CONCENTRATION_PATH.exists():
            raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")
        conc_stack = np.load(CONCENTRATION_PATH)
        if conc_stack.ndim != 3:
            raise ValueError(f"Expected concentration stack with shape (y, x, t); got {conc_stack.shape}")
        intermittency = compute_intermittency(conc_stack, THRESHOLD, axis=2, percent=False)

    ny, nx = intermittency.shape
    x_coords = _load_coords(X_COORDS_PATH, nx, "x")
    y_coords = _load_coords(Y_COORDS_PATH, ny, "y")
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    if OUT_INTERMITTENCY_PATH is not None:
        OUT_INTERMITTENCY_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUT_INTERMITTENCY_PATH, intermittency)
        print(f"Saved intermittency array to {OUT_INTERMITTENCY_PATH}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cs = ax.contour(
        X,
        Y,
        intermittency,
        levels=CONTOUR_LEVELS,
        cmap=CONTOUR_CMAP,
        linewidths=CONTOUR_LINE_WIDTH,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Intermittency contours (threshold={THRESHOLD})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    OUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PATH, dpi=600)
    plt.close(fig)
    print(f"Saved intermittency contour plot to {OUT_FIG_PATH}")


if __name__ == "__main__":
    main()
