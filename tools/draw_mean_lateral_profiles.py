"""
Plot mean lateral concentration distributions (x-direction) for multiple cases.

Edit the paths/settings below, then run:
    python tools/draw_mean_lateral_profiles.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.visualization.viz import (
    compute_gaussian_params_at_y,
    plot_gaussian_param_scatter,
    plot_lateral_profiles,
)

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
# List of (label, path) pairs for mean field .npz files
MEAN_CASES: list[tuple[str, Path]] = [
    ("smSource", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_smSource.npz")),
    ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_nearbed.npz")),
    ("fractal", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_fractal.npz")),
    ("baseline", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_baseline.npz")),
    ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_buoyant.npz")),
    ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_diffusive.npz")),
]
# List of (label, path) pairs for single .npy mean concentration arrays
MEAN_C_ARRAYS: list[tuple[str, Path]] = [
]
X_PATH = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
TARGET_Y_MM = [250.0, 150.0, 50.0]
OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/Plots/Mean/Profiles")
XLABEL = "x (mm)"
YLABEL = "Mean concentration"
NORMALIZE_TO_MAX = True
LINE_COLOR: str | None = None  # Use palette-based colors if None
LINESTYLES = ["solid"]
LINE_WIDTH = 1.0
XLIM = (-100, 100)
SET_YLIM_TO_DATA_MAX = True
ROWS_TO_AVERAGE = 10
YLIM = (0.0, 1.0)
FIT_X_RANGE = (-50.0, 100.0)
GAUSSIAN_MARKERS = ["o", "s", "^", "D", "P", "X"]


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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

    gaussian_results: list[tuple[float, float, list[tuple[str, float, float]]]] = []
    for target_y in TARGET_Y_MM:
        out_path = OUT_DIR / f"lateral_profile_y{300-target_y:g}mm.png"
        title = f"Mean concentration lateral profile at y = {300 - target_y:g} mm"
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
            line_alpha=0.85,
            xlim=XLIM,
            set_ylim_to_data_max=SET_YLIM_TO_DATA_MAX,
            rows_to_average=ROWS_TO_AVERAGE,
            ylim=YLIM,
            fit_x_range=FIT_X_RANGE,
            legend=True,
            grid=False,
        )
        print(f"Saved lateral profile plot for y={300-target_y} mm to {out_path}")
        params_at_y, y_match = compute_gaussian_params_at_y(
            cases_loaded,
            x_coords=x_coords,
            y_coords=y_coords,
            target_y=target_y,
            normalize_to_max=NORMALIZE_TO_MAX,
            xlim=XLIM,
            rows_to_average=ROWS_TO_AVERAGE,
            fit_x_range=FIT_X_RANGE,
        )
        gaussian_results.append((target_y, y_match, params_at_y))

    mu_out = OUT_DIR / "gaussian_mu_across_cases.png"
    sigma_out = OUT_DIR / "gaussian_sigma_across_cases.png"
    plot_gaussian_param_scatter(
        gaussian_results,
        param="mu",
        out_path=mu_out,
        title="Gaussian mu across cases",
        ylabel="mu (mm)",
        markers=GAUSSIAN_MARKERS,
    )
    plot_gaussian_param_scatter(
        gaussian_results,
        param="sigma",
        out_path=sigma_out,
        title="Gaussian sigma across cases",
        ylabel="sigma (mm)",
        markers=GAUSSIAN_MARKERS,
    )
    print(f"Saved Gaussian mu scatter to {mu_out}")
    print(f"Saved Gaussian sigma scatter to {sigma_out}")


if __name__ == "__main__":
    main()
