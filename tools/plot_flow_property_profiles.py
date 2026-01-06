"""
Load flow properties for multiple cases, average cross-stream (x-direction),
optionally smooth over neighboring rows, and plot profiles for each property.

Edit the settings below, then run:
    python tools/plot_flow_property_profiles.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

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
PROPERTIES = {
    # "turbulence_intensity": ("turbulence", "intensity"),
    # "Taylor_microscale": ("Taylor", "microscale"),
    # "Taylor_Re": ("Taylor", "Re"),
    # "kolmogorov_time_scale": ("kolmogorov", "time"),
    "tke": ("tke"),
    # "epsilon": ("epsilon"),
}
ROWS_TO_AVG = 5  # number of rows in y to average together for smoothing
USE_NPZ = False  # allow .npz in addition to .npy
OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Plots/profiles")
XLABEL = "x index"
Y_LABELS = {
    # "turbulence_intensity": "Turbulence intensity",
    # "Taylor_microscale": "Taylor microscale",
    # "Taylor_Re": "Taylor Re",
    # "kolmogorov_time_scale": "Kolmogorov time scale",
    "tke": "Turbulent Kinetic Energy",
    # "epsilon": "Dissipation rate",
}
DPI = 600
CMAP = cmr.get_sub_cmap("cmr.rainforest", 0.0, 0.85)
LEGEND = False
# -------------------------------------------------------------------


def _find_prop_file(case_dir: Path, substrings: Iterable[str]) -> Path:
    """Return the first file in FINAL_AllTimeSteps matching all substrings."""
    target_dir = case_dir / "FINAL_AllTimeSteps"
    if not target_dir.exists():
        raise FileNotFoundError(f"Missing FINAL_AllTimeSteps directory: {target_dir}")
    candidates = []
    for ext in (".npy", ".npz"):
        if ext == ".npz" and not USE_NPZ:
            continue
        candidates.extend(target_dir.glob(f"*{ext}"))
    lower_subs = [s.lower() for s in substrings]
    for p in candidates:
        name = p.name.lower()
        if all(sub in name for sub in lower_subs):
            return p
    raise FileNotFoundError(f"No file found in {target_dir} matching substrings {substrings}")


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

    for prop_key, substrings in PROPERTIES.items():
        plt.figure(figsize=(4.5, 4.5))
        for idx, case in enumerate(CASES):
            case_dir = BASE_DIR / case
            path = _find_prop_file(case_dir, substrings)
            arr = _load_array(path)
            profile = _profile_along_y(arr, ROWS_TO_AVG)
            x_vals = np.arange(profile.shape[0])
            plt.plot(
                np.flip(x_vals),
                profile,
                label=case,
                color=colors[idx],
                linewidth=1.0,
            )
        plt.xlabel(XLABEL)
        plt.ylabel(Y_LABELS.get(prop_key, prop_key))
        plt.title(f"{Y_LABELS.get(prop_key, prop_key)} across cases")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        if LEGEND:
            plt.legend()
        plt.tight_layout()
        out_path = OUT_DIR / f"{prop_key}_profiles.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close()
        print(f"Saved {prop_key} profiles to {out_path}")


if __name__ == "__main__":
    main()
