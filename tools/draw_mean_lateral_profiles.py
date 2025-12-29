"""
Plot mean lateral concentration distributions (x-direction) for multiple cases.

Edit the paths/settings below, then run:
    python tools/draw_mean_lateral_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.visualization.viz import plot_lateral_profiles

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
# List of (label, path) pairs for mean field .npz files
MEAN_CASES: list[tuple[str, Path]] = [
    # ("baseline", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_baseline.npz")),
    # ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_buoyant.npz")),
    # ("fractal", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_fractal.npz")),
    # ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_diffusive.npz")),
    # ("smSource", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_smSource.npz")),
    # ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_nearbed.npz")),
]
# List of (label, path) pairs for single .npy mean concentration arrays
MEAN_C_ARRAYS: list[tuple[str, Path]] = [
    # Example:
    ("buoyant_c_mean", Path("E:/sPIV_PLIF_ProcessedData/PLIF/plif_buoyant_smoothed_mean.npy")),
    ("fractal_c_mean", Path("E:/sPIV_PLIF_ProcessedData/PLIF/plif_fractal_smoothed_mean.npy"))
]
X_PATH = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
TARGET_Y_MM = [280.0, 150.0, 20.0]
OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/Plots/Mean/Profiles/Tests")
XLABEL = "x (mm)"
YLABEL = "Mean concentration"
NORMALIZE_TO_MAX = True
LINE_COLOR = "#000000"
LINESTYLES = ["solid", "dashed", "dashdot", "dotted", (0, (3, 1, 1, 1))]
LINE_WIDTH = 1.0
XLIM = (-100, 100)
SET_YLIM_TO_DATA_MAX = True
ROWS_TO_AVERAGE = 40
YLIM = (0.0, 1.0)


def _load_mean_c(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mean fields file not found: {path}")
    data = np.load(path)
    if "c" not in data:
        raise KeyError(f"Missing 'c' in {path}")
    c_mean = np.array(data["c"], copy=False)
    if c_mean.ndim != 2:
        raise ValueError(f"Expected 2D mean concentration in {path}; got shape {c_mean.shape}")
    return c_mean


def _load_mean_c_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mean c array not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mean concentration in {path}; got shape {arr.shape}")
    return arr


def main() -> None:
    x_coords = np.load(X_PATH)
    y_coords = np.load(Y_PATH)

    cases_loaded: list[tuple[str, np.ndarray]] = []
    for label, path in MEAN_CASES:
        c_mean = _load_mean_c(path)
        cases_loaded.append((label, c_mean))
    for label, path in MEAN_C_ARRAYS:
        c_mean = _load_mean_c_npy(path)
        cases_loaded.append((label, c_mean))
    if not cases_loaded:
        raise ValueError("No mean cases provided. Populate MEAN_CASES and/or MEAN_C_ARRAYS.")

    for target_y in TARGET_Y_MM:
        out_path = OUT_DIR / f"lateral_profile_y{300-target_y:g}mm.png"
        title = f"Mean concentration lateral profile at y ≈ {300-target_y:g} mm"
        plot_lateral_profiles(
            cases_loaded,
            x_coords=x_coords,
            y_coords=y_coords,
            target_y=target_y,
            out_path=out_path,
            title=title,
            xlabel=XLABEL,
            ylabel=YLABEL,
            normalize_to_max=NORMALIZE_TO_MAX,
            line_color=LINE_COLOR,
            linestyles=LINESTYLES,
            line_width=LINE_WIDTH,
            xlim=XLIM,
            set_ylim_to_data_max=SET_YLIM_TO_DATA_MAX,
            rows_to_average=ROWS_TO_AVERAGE,
            ylim=YLIM,
        )
        print(f"Saved lateral profile plot for y={300-target_y} mm to {out_path}")


if __name__ == "__main__":
    main()
