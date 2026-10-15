"""
Load integral length scale fields for multiple cases, average across x to get a
profile along y, optionally smooth, and plot profiles for each case.

Expects the outputs of src/sPIV_PLIF_postprocessing/analysis/ILS_mpi_from_npy.py:
ILS_<case>_<velocity component>_<direction>.npy, shape (ny, nx) on the PLIF grid, values in m.
Cases with missing files are skipped.

Edit the settings below, then run:
    python tools/plot_integral_length_scale_profiles.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.lines import Line2D

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASES: list[str] = ["baseline"]
BASE_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Integral_length_scales/baseline")  # directory containing the ILS_<case>_<component>.npy files
VELOCITY = "v"  # velocity component the ILS was computed for ("u", "v", or "w")
COMPONENTS = [f"{VELOCITY}_cross_stream_fulllag", f"{VELOCITY}_streamwise_fulllag"]
FILE_PREFIX = "ILS"
USE_NPZ = False  # allow .npz with the same naming convention in addition to .npy
CASE_ALIASES = {}  # handle filename mismatches, if any
OUT_DIR = BASE_DIR / "Profiles"
QC_DIR = OUT_DIR / "QC"
GRID_DIR = Path("E:/sPIV_PLIF_ProcessedData/PLIF")  # <case>_ygrid.npy for the y axis in mm; None = plot vs index

ROWS_TO_AVG = 5  # number of rows in y to average together for smoothing
PROFILE_SLICE = slice(10, None)  # trim profile by array row index (row 0 = far downstream edge); slice(None) for full
X_AVG_SLICE = slice(100, 847)  # columns (x index) to average across, e.g. slice(300, 650) for the plume core
FLIP_X = True  # plume source is at the top of the array (high row index): plot distance increasing downstream
TRANSPOSE_DATA = False  # True only for the OLD ILS files; new outputs are already (y, x)
MASK_NO_ZERO_CROSSING = False  # set ILS to NaN where f(r) never crossed zero (uses ILS_..._r0.npy)
SCALE = 1000.0  # m -> mm
QC_PLOTS = False  # save QC plots of the fields as loaded
PLOT_COMPONENT_AVG = True  # plot profiles averaged across the two directions
PRINT_MEANS = False  # print domain-mean values for each component and case

XLABEL = "y [mm]" if GRID_DIR is not None else "y index"
Y_LABELS = {
    f"{VELOCITY}_cross_stream_fulllag": f"Integral length scale [mm], {VELOCITY} cross-stream (full lag)",
    f"{VELOCITY}_streamwise_fulllag": f"Integral length scale [mm], {VELOCITY} streamwise (full lag)",
    f"{VELOCITY}_avg": f"Integral length scale [mm], {VELOCITY} avg (cross + stream)/2",
}

DPI = 600
CMAP = cmr.get_sub_cmap("cmr.neutral", 0.0, 0.75)
LEGEND = True
MARKERS_PER_LINE = 11  # target number of markers per line (approx)
# -------------------------------------------------------------------


def _find_ils_file(base_dir: Path, case_name: str, component: str) -> Path:
    """Return the ILS file using the naming convention ILS_<case>_<component>.(npy|npz)."""
    case_token = CASE_ALIASES.get(case_name, case_name)
    base_name = f"{FILE_PREFIX}_{case_token}_{component}"
    for ext in (".npy", ".npz"):
        if ext == ".npz" and not USE_NPZ:
            continue
        candidate = base_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    expected_ext = ".npz" if USE_NPZ else ".npy"
    raise FileNotFoundError(f"Expected {base_name}{expected_ext} in {base_dir}")


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"Ambiguous npz contents in {path}: {data.files}")
        arr = np.array(data[data.files[0]], copy=False)
    else:
        arr = np.load(path)
    arr = arr.astype(float) * SCALE
    if MASK_NO_ZERO_CROSSING:
        r0_path = path.with_name(f"{path.stem}_r0{path.suffix}")
        if not r0_path.exists():
            raise FileNotFoundError(f"MASK_NO_ZERO_CROSSING needs {r0_path}")
        arr[np.isnan(np.load(r0_path))] = np.nan
    return arr.T if TRANSPOSE_DATA else arr


def _smooth_profile(profile: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return profile
    pad = k // 2
    padded = np.pad(profile, pad_width=pad, mode="edge")
    window = np.ones(k, dtype=float) / k
    smoothed = np.convolve(padded, window, mode="valid")
    return smoothed


def _profile_along_y(arr: np.ndarray, rows_to_avg: int) -> np.ndarray:
    """
    Average across x (axis 1) to get a profile along y (axis 0), then smooth along y.
    """
    profile = np.nanmean(arr[:, X_AVG_SLICE], axis=1)
    return _smooth_profile(profile, rows_to_avg)


def _format_x(profile_len: int, case: str) -> np.ndarray:
    """y-axis values for each array row: mm from the PLIF grid if GRID_DIR is set, else row index.
    With FLIP_X, values increase toward row 0 (downstream, since the source is at the top of the array)."""
    if GRID_DIR is not None:
        y_vals = np.load(GRID_DIR / f"{case}_ygrid.npy")[:, 0].astype(float)
        if y_vals.size != profile_len:
            raise ValueError(f"{case} y grid has {y_vals.size} rows; ILS profile has {profile_len}")
        return y_vals.max() - y_vals if FLIP_X else y_vals
    x_vals = np.arange(profile_len)
    return np.flip(x_vals) if FLIP_X else x_vals


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if QC_PLOTS:
        QC_DIR.mkdir(parents=True, exist_ok=True)
    colors = [CMAP(v) for v in np.linspace(0, 1, len(CASES))]
    markers = ["o", "s", "D", "^", "H", "X", "P", "*"]

    arrays: dict[str, dict[str, np.ndarray]] = {}
    means: dict[str, dict[str, float]] = {}
    for case in CASES:
        try:
            paths = {component: _find_ils_file(BASE_DIR, case, component) for component in COMPONENTS}
        except FileNotFoundError as err:
            print(f"Skipping {case}: {err}")
            continue
        arrays[case] = {}
        means[case] = {}
        for component, path in paths.items():
            arr = _load_array(path)
            arrays[case][component] = arr
            means[case][component] = float(np.nanmean(arr))
            if QC_PLOTS:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmr.neutral)
                ax.set_title(f"{FILE_PREFIX} {component} - {case}")
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
                fig.colorbar(im, ax=ax, shrink=0.85)
                fig.tight_layout()
                qc_path = QC_DIR / f"{FILE_PREFIX}_{component}_{case}_qc.png"
                fig.savefig(qc_path, dpi=DPI)
                plt.close(fig)

    if not arrays:
        raise FileNotFoundError(f"No ILS files found for any case in {BASE_DIR}")

    if PRINT_MEANS:
        for case in arrays:
            for component in COMPONENTS:
                mean_val = means[case][component]
                print(f"{case} {component} domain mean: {mean_val:.6f}")

    for component in COMPONENTS:
        plt.figure(figsize=(4.5, 4.5))
        legend_handles: list[Line2D] = []
        legend_labels: list[str] = []
        for idx, case in enumerate(CASES):
            if case not in arrays:
                continue
            arr = arrays[case][component]
            profile = _profile_along_y(arr, ROWS_TO_AVG)
            x_vals = _format_x(profile.shape[0], case)
            x_plot = x_vals[PROFILE_SLICE]
            y_plot = profile[PROFILE_SLICE]
            marker_indices = np.unique(
                np.linspace(0, len(x_plot) - 1, MARKERS_PER_LINE, dtype=int)
            )
            plt.plot(
                x_plot,
                y_plot,
                label=None,
                color=colors[idx],
                linestyle="-",
                linewidth=0.5,
            )
            plt.plot(
                x_plot[marker_indices],
                y_plot[marker_indices],
                label=case,
                color=colors[idx],
                marker=markers[idx % len(markers)],
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=0.5,
                markersize=8,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[idx],
                    marker=markers[idx % len(markers)],
                    linestyle="-",
                    linewidth=0.5,
                    markerfacecolor="none",
                    markeredgewidth=0.5,
                    markersize=8,
                )
            )
            legend_labels.append(case)
        plt.xlabel(XLABEL)
        plt.ylabel(Y_LABELS.get(component, component))
        plt.title(f"{Y_LABELS.get(component, component).replace(' [mm]', '')} across cases", fontsize=9)
        if LEGEND:
            plt.legend(legend_handles, legend_labels)
        plt.tight_layout()
        out_path = OUT_DIR / f"{FILE_PREFIX}_{component}_profiles.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close()
        print(f"Saved {component} profiles to {out_path}")

    if PLOT_COMPONENT_AVG:
        plt.figure(figsize=(4.5, 4.5))
        legend_handles = []
        legend_labels = []
        for idx, case in enumerate(CASES):
            if case not in arrays:
                continue
            avg_arr = 0.5 * (arrays[case][f"{VELOCITY}_cross_stream_fulllag"] + arrays[case][f"{VELOCITY}_streamwise_fulllag"])
            profile = _profile_along_y(avg_arr, ROWS_TO_AVG)
            x_vals = _format_x(profile.shape[0], case)
            x_plot = x_vals[PROFILE_SLICE]
            y_plot = profile[PROFILE_SLICE]
            marker_indices = np.unique(
                np.linspace(0, len(x_plot) - 1, MARKERS_PER_LINE, dtype=int)
            )
            plt.plot(
                x_plot,
                y_plot,
                label=None,
                color=colors[idx],
                linestyle="-",
                linewidth=0.5,
            )
            plt.plot(
                x_plot[marker_indices],
                y_plot[marker_indices],
                label=case,
                color=colors[idx],
                marker=markers[idx % len(markers)],
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=0.5,
                markersize=8,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[idx],
                    marker=markers[idx % len(markers)],
                    linestyle="-",
                    linewidth=0.5,
                    markerfacecolor="none",
                    markeredgewidth=0.5,
                    markersize=8,
                )
            )
            legend_labels.append(case)
        plt.xlabel(XLABEL)
        plt.ylabel(Y_LABELS[f"{VELOCITY}_avg"])
        plt.title(f"{Y_LABELS[f'{VELOCITY}_avg'].replace(' [mm]', '')} across cases", fontsize=9)
        if LEGEND:
            plt.legend(legend_handles, legend_labels)
        plt.tight_layout()
        out_path = OUT_DIR / f"{FILE_PREFIX}_{VELOCITY}_avg_profiles.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close()
        print(f"Saved averaged profiles to {out_path}")


if __name__ == "__main__":
    main()
