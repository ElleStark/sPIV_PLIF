"""
Plot spatial turbulence intensity maps for all cases.

Reads:
    E:/sPIV_PLIF_ProcessedData/flow_properties/Plots/{case_name}/FINAL_AllTimeSteps
        turbulence_intensity_{case_name}.npy

Saves:
    turbulence_intensity_{case_name}.png (same folder as the .npy)

Run:
    python tools/plot_turbulence_intensity_maps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.interpolate import griddata

# -------------------------------------------------------------------
# Edit these settings as needed
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
PLOTS_ROOT = BASE_PATH / "flow_properties" / "Plots"
DATA_SUBDIR = "FINAL_AllTimeSteps"
CMAP = cmr.ember
VMIN = 0  # set to a float to fix the color scale
VMAX = 0.35  # set to a float to fix the color scale
FIG_DPI = 600
DIFFUSIVE_FILL_NANS = True  # fill NaNs via nearest-neighbor interpolation for diffusive case
# -------------------------------------------------------------------


def _fill_nan_nearest(field: np.ndarray) -> np.ndarray:
    mask = np.isnan(field)
    if not np.any(mask):
        return field
    valid = ~mask
    if not np.any(valid):
        return field
    coords = np.column_stack(np.nonzero(valid))
    values = field[valid]
    missing = np.column_stack(np.nonzero(mask))
    filled = field.copy()
    filled[mask] = griddata(coords, values, missing, method="nearest")
    return filled


def _plot_case(case_name: str, data_dir: Path) -> None:
    npy_path = data_dir / f"turbulence_intensity_{case_name}.npy"
    if not npy_path.exists():
        print(f"Skip {case_name}: missing {npy_path}")
        return

    data = np.load(npy_path, mmap_mode="r")
    if case_name.lower() == "diffusive" and DIFFUSIVE_FILL_NANS:
        data = _fill_nan_nearest(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(data, shading="auto", cmap=CMAP, vmin=VMIN, vmax=VMAX)
    ax.set_title(f"Turbulence intensity: {case_name}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, label="Turbulence intensity")
    fig.tight_layout()

    out_path = data_dir / f"turbulence_intensity_{case_name}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    if not PLOTS_ROOT.exists():
        raise FileNotFoundError(f"Plots root not found: {PLOTS_ROOT}")

    case_dirs = [p for p in PLOTS_ROOT.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"No case folders found in {PLOTS_ROOT}")
        return

    for case_dir in sorted(case_dirs):
        data_dir = case_dir / DATA_SUBDIR
        if not data_dir.exists():
            print(f"Skip {case_dir.name}: missing {data_dir}")
            continue
        _plot_case(case_dir.name, data_dir)


if __name__ == "__main__":
    main()
