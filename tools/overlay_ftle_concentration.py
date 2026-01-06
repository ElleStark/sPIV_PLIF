"""
Overlay an FTLE field as contours on top of an instantaneous concentration snapshot.

Edit the settings below, then run:
    python tools/overlay_ftle_concentration.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
FTLE_PATH = Path("data/FTLE_t50.0to50.05s_fractalCasev2.npy")  # 2D or 3D with leading dim to squeeze
CASE_NAME = "fractal"
CONCENTRATION_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/plif_{CASE_NAME}_smoothed.npy")
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
FRAME_IDX = 1000  # concentration frame to overlay
OUT_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/FTLE/ftle_conc_overlay_{CASE_NAME}.png")
CONC_CMAP = cmr.ocean_r
FTLE_CMAP = 'Greys'
FTLE_LEVELS: int | list[float] | None = 40  # e.g., 40 or explicit list; None disables contours
CONC_VMIN: float | None = 0.005  # set to float to fix scale; None auto-scales
CONC_VMAX: float | None = 1
FTLE_VMIN: float | None = -8
FTLE_VMAX: float | None = 1
CONC_LOG_SCALE = True  # set True for LogNorm on concentration
XLIM: tuple[float, float] | None = (-100.0, 100.0)
YLIM: tuple[float, float] | None = (0, 250)
DPI = 600
# -------------------------------------------------------------------


def _load_conc_slice(path: Path, frame_idx: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D concentration stack (y, x, t); got shape {arr.shape}")
    if frame_idx < 0 or frame_idx >= arr.shape[2]:
        raise IndexError(f"FRAME_IDX {frame_idx} out of range (0..{arr.shape[2]-1})")
    return arr[:, :, frame_idx]


def _load_ftle(path: Path) -> np.ndarray:
    ftle = np.load(path)
    # Handle leading singleton dims
    while ftle.ndim > 2 and ftle.shape[0] == 1:
        ftle = np.squeeze(ftle, axis=0)
    if ftle.ndim == 3:
        # If time-resolved FTLE provided, use first frame by default
        ftle = ftle[0, :, :]
    if ftle.ndim != 2:
        raise ValueError(f"FTLE array must be 2D after squeezing; got shape {ftle.shape}")
    return ftle


def main() -> None:
    if not FTLE_PATH.exists():
        raise FileNotFoundError(f"FTLE file not found: {FTLE_PATH}")
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")

    ftle = _load_ftle(FTLE_PATH)
    conc = _load_conc_slice(CONCENTRATION_PATH, FRAME_IDX)
    if ftle.shape != conc.shape:
        raise ValueError(f"FTLE shape {ftle.shape} does not match concentration slice {conc.shape}")

    x_coords = np.load(X_PATH) if X_PATH else np.arange(ftle.shape[1])
    y_coords = np.load(Y_PATH) if Y_PATH else np.arange(ftle.shape[0])
    if len(x_coords) != ftle.shape[1] or len(y_coords) != ftle.shape[0]:
        raise ValueError("x/y coordinate lengths must match FTLE/concentration grid.")
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    conc_vmin = float(np.nanmin(conc)) if CONC_VMIN is None else CONC_VMIN
    conc_vmax = float(np.nanmax(conc)) if CONC_VMAX is None else CONC_VMAX
    ftle_vmin = float(np.nanmin(ftle)) if FTLE_VMIN is None else FTLE_VMIN
    ftle_vmax = float(np.nanmax(ftle)) if FTLE_VMAX is None else FTLE_VMAX
    conc_norm = LogNorm(vmin=max(conc_vmin, np.finfo(float).tiny), vmax=conc_vmax) if CONC_LOG_SCALE else None
    ftle[ftle < ftle_vmin] = ftle_vmin
    conc[conc < conc_vmin] = conc_vmin

    fig, ax = plt.subplots(figsize=(8, 6))

    if FTLE_LEVELS is not None:
        cs = ax.contourf(
            X,
            Y,
            ftle,
            levels=FTLE_LEVELS,
            cmap=FTLE_CMAP,
            vmin=ftle_vmin,
            vmax=ftle_vmax,
            alpha=1.0,
        )

    h = ax.pcolormesh(
        X,
        Y,
        np.ma.array(np.fliplr(conc), mask=~np.isfinite(conc) | (conc <= 0) if CONC_LOG_SCALE else ~np.isfinite(conc)),
        cmap=CONC_CMAP,
        shading="auto",
        vmin=None if CONC_LOG_SCALE else conc_vmin,
        vmax=None if CONC_LOG_SCALE else conc_vmax,
        norm=conc_norm,
        alpha=0.65,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if XLIM is not None:
        ax.set_xlim(XLIM)
    if YLIM is not None:
        ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h, ax=ax, label="Concentration")
    if FTLE_LEVELS is not None:
        sm = plt.cm.ScalarMappable(cmap=FTLE_CMAP)
        sm.set_clim(ftle_vmin, ftle_vmax)
        fig.colorbar(sm, ax=ax, label="FTLE")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI)
    plt.close(fig)

    print(f"Saved FTLE/concentration overlay to {OUT_PATH}")


if __name__ == "__main__":
    main()
