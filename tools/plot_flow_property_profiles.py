"""
Load flow properties for multiple cases, average cross-stream (x-direction),
optionally smooth over neighboring rows, and plot profiles for each property.

Edit the settings below, then run:
    python tools/plot_flow_property_profiles.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASES: list[str] = ["smSource", "nearbed", "fractal", "baseline", "buoyant", "diffusive"]
BASE_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Plots")
PROPERTIES = [
    "turbulence_intensity",
    "Taylor_microscale",
    "Taylor_Re",
    "kolmogorov_length_scale",
    "tke",
    "epsilon",
]
ROWS_TO_AVG = 5  # number of rows in y to average together for smoothing
USE_NPZ = False  # allow .npz with the same naming convention in addition to .npy
OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Plots/profiles")
XLABEL = "x index"
Y_LABELS = {
    "turbulence_intensity": "Turbulence intensity",
    "Taylor_microscale": "Taylor microscale",
    "Taylor_Re": "Taylor Re",
    "kolmogorov_length_scale": "Kolmogorov length scale",
    "tke": "Turbulent Kinetic Energy",
    "epsilon": "Dissipation rate",
}
DPI = 600
CMAP = cmr.get_sub_cmap("cmr.neutral", 0.0, 0.75)
LEGEND = False
MARKERS_PER_LINE = 11  # target number of markers per line (approx)
# -------------------------------------------------------------------


def _find_prop_file(case_dir: Path, prop_key: str, case_name: str) -> Path:
    """Return the property file using the naming convention <prop>_<case>.(npy|npz)."""
    target_dir = case_dir / "FINAL_AllTimeSteps"
    if not target_dir.exists():
        raise FileNotFoundError(f"Missing FINAL_AllTimeSteps directory: {target_dir}")

    base_name = f"{prop_key}_{case_name}"
    for ext in (".npy", ".npz"):
        if ext == ".npz" and not USE_NPZ:
            continue
        candidate = target_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    expected_ext = ".npz" if USE_NPZ else ".npy"
    raise FileNotFoundError(f"Expected {base_name}{expected_ext} in {target_dir}")


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"Ambiguous npz contents in {path}: {data.files}")
        return np.array(data[data.files[0]], copy=False)
    return np.load(path)


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
    Average across y to get a profile along x (axis 0), then smooth along x by averaging
    over neighboring rows (rows_to_avg window).
    """
    profile = np.nanmean(arr, axis=1)  # average across x (axis 1) -> profile vs y (axis 0)
    return _smooth_profile(profile, rows_to_avg)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = [CMAP(v) for v in np.linspace(0, 1, len(CASES))]
    markers = ["o", "s", "D", "^", "H", "X", "P", "*"]

    for prop_key in PROPERTIES:
        plt.figure(figsize=(4.5, 4.5))
        for idx, case in enumerate(CASES):
            case_dir = BASE_DIR / case
            path = _find_prop_file(case_dir, prop_key, case)
            arr = _load_array(path)
            profile = _profile_along_y(arr, ROWS_TO_AVG)
            x_vals = np.arange(profile.shape[0])
            x_plot = np.flip(x_vals)[30:]
            y_plot = profile[30:]
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
        # tke =np.load("E:/sPIV_PLIF_ProcessedData/flow_properties/Plots/baseline/FINAL_AllTimeSteps/tke_baseline.npy")
        # plt.plot(np.flip(np.arange(tke.shape[0])),_profile_along_y(tke,ROWS_TO_AVG),label="baseline",color=colors[0],linewidth=1.0)
        plt.xlabel(XLABEL)
        plt.ylabel(Y_LABELS.get(prop_key, prop_key))
        plt.title(f"{Y_LABELS.get(prop_key, prop_key)} across cases")
        # plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        if LEGEND:
            plt.legend()
        plt.tight_layout()
        out_path = OUT_DIR / f"{prop_key}_profiles.png"
        # out_path = OUT_DIR / f"tke_profiles.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close()
        print(f"Saved {prop_key} profiles to {out_path}")


if __name__ == "__main__":
    main()
