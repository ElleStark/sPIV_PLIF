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
    ("smSource", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/smSource_PLIF_mean.npy")),
    ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/nearbed_PLIF_mean.npy")),
    ("fractal", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/fractal_PLIF_mean.npy")),
    ("baseline", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/baseline_PLIF_mean.npy")),
    ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/buoyant_PLIF_mean.npy")),
    ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/mean_fields/diffusive_PLIF_mean.npy")),
]
# List of (label, path) pairs for single .npy mean concentration arrays
MEAN_C_ARRAYS: list[tuple[str, Path]] = [
]
X_PATHS: list[tuple[str, Path]] = [
    ("smSource", Path("E:/sPIV_PLIF_ProcessedData/PLIF/smSource_xgrid.npy")), 
    ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/PLIF/nearbed_xgrid.npy")),
    ("fractal", Path("E:/sPIV_PLIF_ProcessedData/PLIF/fractal_xgrid.npy")),
    ("baseline", Path("E:/sPIV_PLIF_ProcessedData/PLIF/baseline_xgrid.npy")),
    ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/PLIF/buoyant_xgrid.npy")),       
    ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/PLIF/diffusive_xgrid.npy")),
]
Y_PATHS: list[tuple[str, Path]] = [
    ("smSource", Path("E:/sPIV_PLIF_ProcessedData/PLIF/smSource_ygrid.npy")), 
    ("nearbed", Path("E:/sPIV_PLIF_ProcessedData/PLIF/nearbed_ygrid.npy")),
    ("fractal", Path("E:/sPIV_PLIF_ProcessedData/PLIF/fractal_ygrid.npy")),
    ("baseline", Path("E:/sPIV_PLIF_ProcessedData/PLIF/baseline_ygrid.npy")),
    ("buoyant", Path("E:/sPIV_PLIF_ProcessedData/PLIF/buoyant_ygrid.npy")),       
    ("diffusive", Path("E:/sPIV_PLIF_ProcessedData/PLIF/diffusive_ygrid.npy")),
]
TARGET_Y_MM = [250.0, 150.0, 50.0]
# Dense y locations used only for the Gaussian mu/sigma curves (no profile plots saved).
# Clipped to the y-range shared by all cases.
GAUSSIAN_Y_MM = np.arange(10.0, 291.0, 5.0)
GAUSSIAN_MARKER_EVERY = 2  # draw a marker every N points (fits still use all points)
MEAN_VELOCITY_MM_S = 300.0  # mean advection speed for t = x / U in the diffusivity fit
GAUSSIAN_AXES_SIZE_IN = (3.4, 3.55)  # shared (width, height) of the mu and sigma plot areas
DIFFUSIVITY_FIT_X_MIN_MM = 100.0  # fit sigma only for x >= this downstream distance...
DIFFUSIVITY_FIT_ALL_POINTS = {"diffusive"}  # ...except these cases, which use all points
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
    # data = np.load(path)
    # if "c" not in data:
    #     raise KeyError(f"Missing 'c' in {path}")
    # c_mean = np.array(data["c"], copy=False)
    c_mean = np.load(path)
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


def load_inputs() -> tuple[list[tuple[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load mean concentration fields and 1D x/y coordinates for every case."""
    x_coords_loaded = {}
    for label, path in X_PATHS:
        x_coords = np.load(path)
        x_coords = x_coords[0, :]
        x_coords_loaded[label] = x_coords

    y_coords_loaded = {}
    for label, path in Y_PATHS:
        y_coords = np.load(path)
        y_coords = y_coords[:, 0]
        y_coords_loaded[label] = y_coords

    cases_loaded: list[tuple[str, np.ndarray]] = []
    for label, path in MEAN_CASES:
        c_mean = _load_mean_c(path)
        cases_loaded.append((label, c_mean))
    for label, path in MEAN_C_ARRAYS:
        c_mean = _load_mean_c_npy(path)
        cases_loaded.append((label, c_mean))
    if not cases_loaded:
        raise ValueError("No mean cases provided. Populate MEAN_CASES and/or MEAN_C_ARRAYS.")
    return cases_loaded, x_coords_loaded, y_coords_loaded


def compute_dense_gaussian_results(
    cases_loaded: list[tuple[str, np.ndarray]],
    x_coords_loaded: dict[str, np.ndarray],
    y_coords_loaded: dict[str, np.ndarray],
) -> list[tuple[float, float, list[tuple[str, float, float]]]]:
    """Gaussian (mu, sigma) fits for every case at each GAUSSIAN_Y_MM location."""
    # Restrict dense fit locations to the y-range covered by every case, so the
    # nearest-index lookup doesn't clamp repeatedly to the grid edge.
    y_lo = max(float(np.nanmin(y)) for y in y_coords_loaded.values())
    y_hi = min(float(np.nanmax(y)) for y in y_coords_loaded.values())
    gaussian_y = [float(y) for y in GAUSSIAN_Y_MM if y_lo <= y <= y_hi]
    if not gaussian_y:
        raise ValueError(f"No GAUSSIAN_Y_MM values fall within shared y-range [{y_lo:g}, {y_hi:g}].")

    gaussian_results: list[tuple[float, float, list[tuple[str, float, float]]]] = []
    for target_y in gaussian_y:
        params_at_y, y_match = compute_gaussian_params_at_y(
            cases_loaded,
            x_coords_loaded,
            y_coords_loaded,
            target_y=target_y,
            normalize_to_max=NORMALIZE_TO_MAX,
            xlim=XLIM,
            rows_to_average=ROWS_TO_AVERAGE,
            fit_x_range=FIT_X_RANGE,
        )
        gaussian_results.append((target_y, y_match, params_at_y))
    return gaussian_results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cases_loaded, x_coords_loaded, y_coords_loaded = load_inputs()

    for target_y in TARGET_Y_MM:
        out_path = OUT_DIR / f"lateral_profile_y{300-target_y:g}mm_NEWDATA.png"
        title = f"Mean concentration lateral profile at y = {300 - target_y:g} mm"
        plot_lateral_profiles(
            cases_loaded,
            x_coords_loaded,
            y_coords_loaded,
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

    gaussian_results = compute_dense_gaussian_results(cases_loaded, x_coords_loaded, y_coords_loaded)

    mu_out = OUT_DIR / "gaussian_mu_across_cases_NEWDATA.png"
    sigma_out = OUT_DIR / "gaussian_sigma_across_cases_NEWDATA.png"
    plot_gaussian_param_scatter(
        gaussian_results,
        param="mu",
        out_path=mu_out,
        title="Gaussian mu across cases",
        ylabel="mu (mm)",
        markers=GAUSSIAN_MARKERS,
        marker_every=GAUSSIAN_MARKER_EVERY,
        hollow_markers=True,
        connect_points=False,
        axes_size_in=GAUSSIAN_AXES_SIZE_IN,
    )
    diffusivity_fits = plot_gaussian_param_scatter(
        gaussian_results,
        param="sigma",
        out_path=sigma_out,
        title="Gaussian sigma across cases",
        ylabel="sigma (mm)",
        markers=GAUSSIAN_MARKERS,
        marker_every=GAUSSIAN_MARKER_EVERY,
        fit_diffusivity=True,
        mean_velocity_mm_s=MEAN_VELOCITY_MM_S,
        hollow_markers=True,
        axes_size_in=GAUSSIAN_AXES_SIZE_IN,
        fit_x_min_by_case={
            label: DIFFUSIVITY_FIT_X_MIN_MM
            for label, _ in cases_loaded
            if label not in DIFFUSIVITY_FIT_ALL_POINTS
        },
    )
    # Fit sigma^2 = a + b x  ->  D = b U / 2,  sigma0 = sqrt(a),  x0 = -a / b
    # (virtual origin: sigma^2 = 2 D (x - x0) / U, i.e. where a point source would sit).
    print("Effective diffusivity from sigma^2 = sigma0^2 + 2 D x / U = 2 D (x - x0) / U:")
    for label, (d_eff, sigma0, x0) in diffusivity_fits.items():
        print(
            f"  {label:>10s}: D = {d_eff:.4g} mm^2/s ({d_eff / 100.0:.4g} cm^2/s), "
            f"sigma0 = {sigma0:.3g} mm, virtual origin x0 = {x0:.4g} mm"
        )
    print(f"Saved Gaussian mu scatter to {mu_out}")
    print(f"Saved Gaussian sigma scatter to {sigma_out}")


if __name__ == "__main__":
    main()
