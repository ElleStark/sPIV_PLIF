"""
Load integral length scale fields for multiple cases, average across x to get a
profile along y, optionally smooth, and plot profiles for each case.

Edit the settings below, then run:
    python tools/plot_integral_length_scale_profiles.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASES: list[str] = ["smSource", "nearbed", "fractal", "baseline", "buoyant", "diffusive"]
BASE_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Integral_length_scales")
COMPONENTS = ["u_cross_stream", "u_streamwise"]
FILE_PREFIX = "ILS"
USE_NPZ = False  # allow .npz with the same naming convention in addition to .npy
CASE_ALIASES = {}  # handle filename mismatches, if any
OUT_DIR = BASE_DIR / "Profiles"
QC_DIR = OUT_DIR / "QC"

ROWS_TO_AVG = 5  # number of rows in y to average together for smoothing
PROFILE_SLICE = slice(30, None)  # trim profile indices; set to slice(None) for full
FLIP_X = True  # flip the x-axis (y-index) for plotting
TRANSPOSE_DATA = True  # transpose data prior to processing
QC_PLOTS = False  # save QC plots of the transposed fields
PLOT_COMPONENT_AVG = True  # plot profiles averaged across the two directions
PRINT_MEANS = False  # print domain-mean values for each component and case

XLABEL = "y index"
Y_LABELS = {
    "u_cross_stream": "Integral length scale, u cross-stream",
    "u_streamwise": "Integral length scale, u streamwise",
    "u_avg": "Integral length scale, u avg (cross + stream)/2",
}

DPI = 600
CMAP = cmr.get_sub_cmap("cmr.neutral", 0.0, 0.75)
LEGEND = False
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
    profile = np.nanmean(arr, axis=1)
    return _smooth_profile(profile, rows_to_avg)


def _format_x(profile_len: int) -> np.ndarray:
    x_vals = np.arange(profile_len)
    return np.flip(x_vals) if FLIP_X else x_vals


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if QC_PLOTS:
        QC_DIR.mkdir(parents=True, exist_ok=True)
    colors = [CMAP(v) for v in np.linspace(0, 1, len(CASES))]
    markers = ["v", "s", "o", "^", "D", "X", "P", "*"]

    arrays: dict[str, dict[str, np.ndarray]] = {}
    means: dict[str, dict[str, float]] = {}
    for case in CASES:
        arrays[case] = {}
        means[case] = {}
        for component in COMPONENTS:
            path = _find_ils_file(BASE_DIR, case, component)
            arr = _load_array(path)
            arrays[case][component] = arr
            means[case][component] = float(np.nanmean(arr))
            if QC_PLOTS:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmr.neutral)
                ax.set_title(f"Transposed {FILE_PREFIX} {component} - {case}")
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
                fig.colorbar(im, ax=ax, shrink=0.85)
                fig.tight_layout()
                qc_path = QC_DIR / f"{FILE_PREFIX}_{component}_{case}_transposed_qc.png"
                fig.savefig(qc_path, dpi=DPI)
                plt.close(fig)

    if PRINT_MEANS:
        for case in CASES:
            for component in COMPONENTS:
                mean_val = means[case][component]
                print(f"{case} {component} domain mean: {mean_val:.6f}")

    for component in COMPONENTS:
        plt.figure(figsize=(4.5, 4.5))
        for idx, case in enumerate(CASES):
            arr = arrays[case][component]
            profile = _profile_along_y(arr, ROWS_TO_AVG)
            x_vals = _format_x(profile.shape[0])
            x_plot = x_vals[PROFILE_SLICE]
            y_plot = profile[PROFILE_SLICE]
            marker_indices = np.unique(
                np.linspace(0, len(x_plot) - 1, MARKERS_PER_LINE, dtype=int)
            )
            plt.plot(
                x_plot,
                y_plot,
                label=case,
                color=colors[idx],
                linestyle="-",
                linewidth=0.5,
            )
            plt.plot(
                x_plot[marker_indices],
                y_plot[marker_indices],
                label=None,
                color=colors[idx],
                marker=markers[idx % len(markers)],
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=0.5,
                markersize=8,
            )
        plt.xlabel(XLABEL)
        plt.ylabel(Y_LABELS.get(component, component))
        plt.title(f"{Y_LABELS.get(component, component)} across cases")
        if LEGEND:
            plt.legend()
        plt.tight_layout()
        out_path = OUT_DIR / f"{FILE_PREFIX}_{component}_profiles.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close()
        print(f"Saved {component} profiles to {out_path}")

    if PLOT_COMPONENT_AVG:
        plt.figure(figsize=(4.5, 4.5))
        for idx, case in enumerate(CASES):
            avg_arr = 0.5 * (arrays[case]["u_cross_stream"] + arrays[case]["u_streamwise"])
            profile = _profile_along_y(avg_arr, ROWS_TO_AVG)
            x_vals = _format_x(profile.shape[0])
            x_plot = x_vals[PROFILE_SLICE]
            y_plot = profile[PROFILE_SLICE]
            marker_indices = np.unique(
                np.linspace(0, len(x_plot) - 1, MARKERS_PER_LINE, dtype=int)
            )
            plt.plot(
                x_plot,
                y_plot,
                label=case,
                color=colors[idx],
                linestyle="-",
                linewidth=0.5,
            )
            plt.plot(
                x_plot[marker_indices],
                y_plot[marker_indices],
                label=None,
                color=colors[idx],
                marker=markers[idx % len(markers)],
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=0.5,
                markersize=8,
            )
        plt.xlabel(XLABEL)
        plt.ylabel(Y_LABELS.get("u_avg", "u_avg"))
        plt.title(f"{Y_LABELS.get('u_avg', 'u_avg')} across cases")
        if LEGEND:
            plt.legend()
        plt.tight_layout()
        out_path = OUT_DIR / f"{FILE_PREFIX}_u_avg_profiles.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close()
        print(f"Saved averaged profiles to {out_path}")


if __name__ == "__main__":
    main()
