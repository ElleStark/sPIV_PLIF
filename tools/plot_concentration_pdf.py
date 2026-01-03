"""
Compute and plot the probability density function (PDF) of concentration at a specified (y, x) index.

Edit the settings below, then run:
    python tools/plot_concentration_pdf.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

from matplotlib.pyplot import legend
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.analysis import (
    concentration_timeseries_from_file,
    compute_concentration_pdf,
    plot_concentration_pdf,
)

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "diffusive"
CONCENTRATION_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/plif_{CASE_NAME}_smoothed.npy")
Y_IDX = 0  # row index in the concentration array
X_IDX = 340  # column index in the concentration array
T_SLICE = slice(0, 6000)  # time frames to include; set to None for all
BIN_WIDTH = 0.0065  # concentration bin width
PDF_RANGE: tuple[float, float] | None = None  # e.g., (0.0, 0.1) or None to auto; if None, range comes from data
OUT_DIR = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/ConcentrationPDF")
XLIM = (0.0, 0.15)
YLIM = (0.0, 35.0)
GAMMA_POINTS = 500  # number of x-points for smooth gamma overlay
Y_LEGEND = False  # set to True to show legend on plot
# -------------------------------------------------------------------


def main() -> None:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")

    ts = concentration_timeseries_from_file(CONCENTRATION_PATH, y_idx=Y_IDX, x_idx=X_IDX, t_slice=T_SLICE, mmap_mode="r")
    finite = ts[np.isfinite(ts)]
    mean_c = float(np.mean(finite))
    std_c = float(np.std(finite))
    intensity = std_c / mean_c if mean_c != 0 else float("inf")
    print(intensity)
    k_shape = 1.0 / (intensity**2) if intensity > 0 else float("inf")

    if PDF_RANGE is not None:
        c_min, c_max = PDF_RANGE
    else:
        c_min, c_max = float(np.min(finite)), float(np.max(finite))
    edges = np.arange(c_min, c_max + BIN_WIDTH, BIN_WIDTH)
    if edges[-1] < c_max:
        edges = np.append(edges, c_max)
    centers, pdf, edges = compute_concentration_pdf(ts, bins=edges, value_range=None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"concentration_pdf_{CASE_NAME}_y{(600-Y_IDX)/20}_x{(X_IDX-300)/20}.npz"
    np.savez(
        pdf_path,
        bin_centers=centers,
        pdf=pdf,
        bin_edges=edges,
        y_idx=Y_IDX,
        x_idx=X_IDX,
        t_slice=str(T_SLICE),
        mean=mean_c,
        std=std_c,
        intensity=intensity,
        k_shape=k_shape,
        bin_width=BIN_WIDTH,
    )

    fig_path = OUT_DIR / f"concentration_pdf_{CASE_NAME}_y{(600-Y_IDX)/20}_x{(X_IDX-300)/20}.png"
    # Plot as histogram (bars) to show probability density
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(centers, pdf, width=np.diff(edges), align="center", edgecolor="#555555", color="#e0e0e0", label="Histogram (density)")

    # Gamma PDF per provided form: p(c) = k^k / Gamma(k) * c^(k-1) * exp(-k c), k = i^-2
    gamma_x = np.linspace(XLIM[0], XLIM[1], GAMMA_POINTS) if mean_c > 0 else centers
    gamma_pdf = np.zeros_like(gamma_x, dtype=float)
    valid = gamma_x >= 0
    if k_shape not in (float("inf"), 0) and mean_c > 0:
        c_norm = gamma_x / mean_c
        coeff = (k_shape ** k_shape) / math.gamma(k_shape)
        gamma_pdf[valid] = (coeff * (c_norm[valid] ** (k_shape - 1)) * np.exp(-k_shape * c_norm[valid])) / mean_c
        ax.plot(gamma_x, gamma_pdf, color="k", lw=1.5, linestyle="--", label=f"Gamma fit (k={k_shape:.3f}, i={intensity:.3f})")

    ax.set_xlabel("Concentration")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Concentration PDF at (y={(600-Y_IDX)/20}, x={(X_IDX-300)/20})")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    if Y_LEGEND:
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)

    print(f"Loaded {finite.size} samples from {CONCENTRATION_PATH}")
    print(f"Saved PDF data to {pdf_path}")
    print(f"Saved PDF figure to {fig_path}")
    print(
        f"Samples stats: min={float(np.min(finite)):.6f}, max={float(np.max(finite)):.6f}, mean={mean_c:.6f}, median={float(np.median(finite)):.6f}, std={std_c:.6f}, intensity={intensity:.6f}, k={k_shape:.6f}"
    )


if __name__ == "__main__":
    main()
