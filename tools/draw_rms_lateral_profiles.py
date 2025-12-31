"""
Plot RMS lateral concentration distributions (x-direction) for multiple cases.

Edit the paths/settings below, then run:
    python tools/draw_rms_lateral_profiles.py
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

from src.sPIV_PLIF_postprocessing.visualization.viz import plot_lateral_profiles

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
# List of (label, path) pairs for RMS field .npz files
RMS_CASES: list[tuple[str, Path]] = [
    # ("baseline", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/baseline_c_rms.npz")),
    # ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/buoyant_c_rms.npz")),
    # ("fractal", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/fractal_c_rms.npz")),
    # ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/diffusive_c_rms.npz")),
    # ("smSource", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/smSource_c_rms.npz")),
    # ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/nearbed_c_rms.npz")),
]
# List of (label, path) pairs for single .npy RMS concentration arrays
RMS_C_ARRAYS: list[tuple[str, Path]] = [
    ("smSource", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/smSource_c_rms.npy")),
    ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/nearbed_c_rms.npy")),
    ("fractal", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/fractal_c_rms.npy")),
    ("baseline", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/baseline_c_rms.npy")),
    ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/buoyant_c_rms.npy")),
    ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/rms_fields/diffusive_c_rms.npy")),
]

X_PATH = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
TARGET_Y_MM = [250.0, 150.0, 50.0]
OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/Plots/RMS/Profiles")
XLABEL = "x (mm)"
YLABEL = "RMS concentration"
NORMALIZE_TO_MAX = False
LINE_COLOR: str | None = None  # Use palette-based colors if None
LINESTYLES = ["solid"]
LINE_WIDTH = 0.75
XLIM = (-75, 75)
SET_YLIM_TO_DATA_MAX = True
ROWS_TO_AVERAGE = 10
YLIM = (0.0, 0.38)


def _load_rms_c(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"RMS fields file not found: {path}")
    data = np.load(path)
    if "c_rms" in data:
        c_rms = np.array(data["c_rms"], copy=False)
    elif "c" in data:
        c_rms = np.array(data["c"], copy=False)
    else:
        raise KeyError(f"Missing 'c_rms' (or fallback 'c') in {path}")
    if c_rms.ndim != 2:
        raise ValueError(f"Expected 2D RMS concentration in {path}; got shape {c_rms.shape}")
    return c_rms


def _load_rms_c_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"RMS c array not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D RMS concentration in {path}; got shape {arr.shape}")
    return arr


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    x_coords = np.load(X_PATH)
    y_coords = np.load(Y_PATH)

    rms_cases_loaded: list[tuple[str, np.ndarray]] = []
    for label, path in RMS_CASES:
        try:
            c_rms = _load_rms_c(path)
        except FileNotFoundError:
            continue
        rms_cases_loaded.append((label, c_rms))
    for label, path in RMS_C_ARRAYS:
        c_rms = _load_rms_c_npy(path)
        rms_cases_loaded.append((label, c_rms))
    if not rms_cases_loaded:
        raise ValueError("No RMS cases provided. Populate RMS_CASES and/or RMS_C_ARRAYS.")

    for target_y in TARGET_Y_MM:
        out_path = OUT_DIR / f"rms_lateral_profile_y{300 - target_y:g}mm.png"
        title = f"RMS concentration lateral profile at y = {300 - target_y:g} mm"
        plot_lateral_profiles(
            rms_cases_loaded,
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
            fit_x_range=None,
            perform_gaussian_fit=False,
            legend=True,
            grid=False,
        )
        print(f"Saved RMS lateral profile plot for y={300 - target_y} mm to {out_path}")


if __name__ == "__main__":
    main()
