"""
Black-and-white version of the K_x vs downstream distance panel of the scalar-flux case comparison
(case_comparison.png, lower left): binned through-origin regression of <u'c'> on dC/dx, one line style per case.

Reads bin_centers_mm and bin_x_K0 from <output_root>/<case>/arrays/diffusivity.npz (run_scalar_flux_diffusivity.py
stage 2). Run:
    python tools/plot_Kx_profiles_bw.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("E:/sPIV_PLIF_ProcessedData/scalar_flux_diffusivity")
CASES = ["baseline", "buoyant", "diffusive", "fractal", "nearbed", "smSource"]
LINESTYLES = {
    "baseline": "-",
    "buoyant": "--",
    "diffusive": ":",
    "fractal": "-.",
    "nearbed": (0, (6, 2, 1, 2, 1, 2)),   # dash-dot-dot
    "smSource": (0, (10, 3)),             # long dashes
}
UNITS_MM2 = True  # True: mm^2/s; False: m^2/s as in case_comparison.png
OUT_DIR = ROOT / "comparison_baseline_buoyant_diffusive_fractal_nearbed_smSource"
OUT_STEM = "Kx_vs_s_bw"
FIGSIZE = (5.0, 3.6)
DPI = 300


def main() -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    scale = 1e6 if UNITS_MM2 else 1.0
    for case in CASES:
        d = np.load(ROOT / case / "arrays" / "diffusivity.npz")
        s, kx = d["bin_centers_mm"], d["bin_x_K0"] * scale
        ax.plot(s, kx, color="k", linestyle=LINESTYLES[case], linewidth=1.3, label=case)
    ax.set_xlabel("downstream distance s (mm)")
    ax.set_ylabel(r"$K_x$ (mm$^2$/s)" if UNITS_MM2 else r"$K_x$ (m$^2$/s)")
    ax.set_xlim(0, 300)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, handlelength=3.0, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT_DIR / OUT_STEM}.png/.pdf")


if __name__ == "__main__":
    main()
