"""
Compute and plot the probability density function (PDF) of concentration at a specified (y, x) index.

Edit the settings below, then run:
    python tools/plot_concentration_pdf.py
"""

from __future__ import annotations

from pathlib import Path
import sys

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
CASE_NAME = "buoyant"
CONCENTRATION_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/plif_{CASE_NAME}_smoothed.npy")
Y_IDX = 500  # row index in the concentration array
X_IDX = 360  # column index in the concentration array
T_SLICE = slice(0, 6000)  # time frames to include; set to None for all
PDF_BINS = 50
PDF_RANGE: tuple[float, float] | None = None  # e.g., (0.0, 0.1) or None to auto
OUT_DIR = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/ConcentrationPDF")
# -------------------------------------------------------------------


def main() -> None:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")

    ts = concentration_timeseries_from_file(CONCENTRATION_PATH, y_idx=Y_IDX, x_idx=X_IDX, t_slice=T_SLICE, mmap_mode="r")
    centers, pdf, edges = compute_concentration_pdf(ts, bins=PDF_BINS, value_range=PDF_RANGE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"concentration_pdf_{CASE_NAME}_y{(600-Y_IDX)/20}_x{(X_IDX-300)/20}.npz"
    np.savez(pdf_path, bin_centers=centers, pdf=pdf, bin_edges=edges, y_idx=Y_IDX, x_idx=X_IDX, t_slice=str(T_SLICE))

    fig_path = OUT_DIR / f"concentration_pdf_{CASE_NAME}_y{(600-Y_IDX)/20}_x{(X_IDX-300)/20}.png"
    # Plot as histogram (bars) to show probability density
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers, pdf, width=np.diff(edges), align="center", edgecolor="k")
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Concentration PDF at (y={(600-Y_IDX)/20}, x={(X_IDX-300)/20})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    finite = ts[np.isfinite(ts)]
    print(f"Loaded {finite.size} samples from {CONCENTRATION_PATH}")
    print(f"Saved PDF data to {pdf_path}")
    print(f"Saved PDF figure to {fig_path}")
    print(
        f"Samples stats: min={float(np.min(finite)):.6f}, max={float(np.max(finite)):.6f}, mean={float(np.mean(finite)):.6f}, median={float(np.median(finite)):.6f}"
    )


if __name__ == "__main__":
    main()
