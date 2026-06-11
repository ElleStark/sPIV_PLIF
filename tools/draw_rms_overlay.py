"""
Draw concentration RMS as a pcolormesh.

Edit the paths/settings below, then run:
    python tools/draw_rms_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from numpy.lib.stride_tricks import sliding_window_view

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
CASE_NAME = "baseline"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
C_PATH = BASE_PATH / "PLIF" / f"{CASE_NAME}_PLIF.npy"
C_MEAN_PATH: Path | None = BASE_PATH / "mean_fields" / f"{CASE_NAME}_PLIF_mean.npy"
C_FLUCT_SAVE_PATH: Path | None = BASE_PATH / "mean_fields" / f"{CASE_NAME}_PLIF_flucts.npy"
RMS_OUT_DIR = BASE_PATH / "rms_fields"
C_RMS_PATH = RMS_OUT_DIR / f"{CASE_NAME}_c_rms.npy"
OUT_PATH = BASE_PATH / "Plots" / "RMS" / f"c_rms_{CASE_NAME}_NEWDATA.png"
X_PATH: Path | None = BASE_PATH / "PLIF" / f"{CASE_NAME}_xgrid.npy"
Y_PATH: Path | None = BASE_PATH / "PLIF" / f"{CASE_NAME}_ygrid.npy"

CMIN = 0.01
CMAX = 0.35
LOG_SCALE = True
CMAP_NAME = "jet"
C_UNDER: str | None = "white"
C_UNDER_TRANSITION: float | None = 0.1  # fraction of cmap for white->cmap blend
PCOLORMESH_ALPHA = 0.85
APPLY_MEDIAN_SMOOTH = False
MEDIAN_WINDOW = 3
USE_MEMMAP = False
SAVE_C_FLUCTUATIONS = False
FIG_DPI = 600
FIGSIZE = (8, 6)
X_SUBSET: tuple[float, float] | None = (-100, 100)
Y_SUBSET: tuple[float, float] | None = None
XLABEL = "x"
YLABEL = "y"
COLORBAR_LABEL = "c RMS"


def _load_optional_array(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def _as_1d_coords(coords: np.ndarray | None, axis: int, size: int) -> np.ndarray:
    if coords is None:
        return np.arange(size)

    coords = np.asarray(coords)
    if coords.ndim == 1:
        values = coords
    elif coords.ndim == 2 and axis == 1:
        values = coords[0, :]
    elif coords.ndim == 2 and axis == 0:
        values = coords[:, 0]
    else:
        raise ValueError(f"Expected 1D or 2D coordinate grid; got shape {coords.shape}")

    if len(values) != size:
        raise ValueError(f"Coordinate length {len(values)} does not match field size {size}")
    return values


def _median_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k % 2 == 0 or k < 1:
        raise ValueError("Median window size must be an odd positive integer.")

    pad = k // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    windows = sliding_window_view(padded, (k, k))
    return np.nanmedian(windows, axis=(-2, -1))


def _compute_c_rms() -> np.ndarray:
    if not C_PATH.exists():
        raise FileNotFoundError(f"Missing concentration stack: {C_PATH}")

    c_data = np.load(C_PATH, mmap_mode="r" if USE_MEMMAP else None)
    if c_data.ndim != 3:
        raise ValueError(f"Expected 3D concentration stack; got shape {c_data.shape}")

    if C_MEAN_PATH is not None and C_MEAN_PATH.exists():
        c_mean = np.load(C_MEAN_PATH)
    else:
        c_mean = np.nanmean(c_data, axis=0)
    if c_mean.shape != c_data.shape[1:]:
        raise ValueError(f"Mean concentration shape {c_mean.shape} does not match stack grid {c_data.shape[1:]}")

    fluctuations = c_data - c_mean[np.newaxis, :, :]
    if SAVE_C_FLUCTUATIONS and C_FLUCT_SAVE_PATH is not None:
        C_FLUCT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(C_FLUCT_SAVE_PATH, fluctuations)

    c_rms = np.sqrt(np.nanmean(np.square(fluctuations), axis=0))
    if APPLY_MEDIAN_SMOOTH:
        c_rms = _median_smooth(c_rms, MEDIAN_WINDOW)

    RMS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C_RMS_PATH, c_rms)
    print(f"Saved c RMS field to {C_RMS_PATH}")
    return c_rms


def _load_or_compute_c_rms() -> np.ndarray:
    if C_RMS_PATH.exists():
        c_rms = np.load(C_RMS_PATH)
        print(f"Loaded c RMS field from {C_RMS_PATH}")
    else:
        c_rms = _compute_c_rms()

    if c_rms.ndim != 2:
        raise ValueError(f"Expected 2D c RMS field; got shape {c_rms.shape}")
    return c_rms


def _subset_field(
    field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_idx = np.arange(field.shape[0])
    x_idx = np.arange(field.shape[1])

    if X_SUBSET is not None:
        x_min, x_max = sorted(X_SUBSET)
        x_mask = (x_coords >= x_min) & (x_coords <= x_max)
        if not np.any(x_mask):
            raise ValueError(f"No x points fall within requested X_SUBSET {X_SUBSET}.")
        x_idx = np.where(x_mask)[0]

    if Y_SUBSET is not None:
        y_min, y_max = sorted(Y_SUBSET)
        y_mask = (y_coords >= y_min) & (y_coords <= y_max)
        if not np.any(y_mask):
            raise ValueError(f"No y points fall within requested Y_SUBSET {Y_SUBSET}.")
        y_idx = np.where(y_mask)[0]

    return field[np.ix_(y_idx, x_idx)], x_coords[x_idx], y_coords[y_idx]


def _plot_c_rms(c_rms: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
    cmap = plt.get_cmap(CMAP_NAME).copy()
    if C_UNDER is not None:
        cmap.set_under(C_UNDER)
    if C_UNDER_TRANSITION is not None and C_UNDER_TRANSITION > 0:
        try:
            colors = cmap(np.linspace(0, 1, 256))
            n_under = max(2, int(len(colors) * min(C_UNDER_TRANSITION, 0.5)))
            white = np.array([1.0, 1.0, 1.0, 1.0])
            first_color = colors[0]
            under_grad = np.stack(
                [
                    white * (1 - t) + first_color * t
                    for t in np.linspace(0, 1, n_under, endpoint=True)
                ],
                axis=0,
            )
            colors = np.vstack([under_grad, colors])
            cmap = ListedColormap(colors)
        except Exception:
            pass
    norm: Normalize | LogNorm
    if LOG_SCALE:
        if CMIN <= 0:
            raise ValueError("CMIN must be positive when LOG_SCALE is True.")
        norm = LogNorm(vmin=CMIN, vmax=CMAX)
    else:
        norm = Normalize(vmin=CMIN, vmax=CMAX)

    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    mesh = ax.pcolormesh(
        x_grid,
        y_grid,
        c_rms,
        shading="auto",
        cmap=cmap,
        norm=norm,
        alpha=PCOLORMESH_ALPHA,
    )
    fig.colorbar(mesh, ax=ax, label=COLORBAR_LABEL)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Concentration RMS: {CASE_NAME}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved c RMS plot to {OUT_PATH}")


def main() -> None:
    c_rms = _load_or_compute_c_rms()
    y_size, x_size = c_rms.shape

    x_coords = _as_1d_coords(_load_optional_array(X_PATH), axis=1, size=x_size)
    y_coords = _as_1d_coords(_load_optional_array(Y_PATH), axis=0, size=y_size)
    c_rms, x_coords, y_coords = _subset_field(c_rms, x_coords, y_coords)
    _plot_c_rms(c_rms, x_coords, y_coords)


if __name__ == "__main__":
    main()
