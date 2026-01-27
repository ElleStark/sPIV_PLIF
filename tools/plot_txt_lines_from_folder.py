"""
Plot line profiles from multiple .txt files in a folder.

Edit the paths/settings below, then run:
    python tools/plot_txt_lines_from_folder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.visualization.txt_line_plots import plot_lines_from_txt_files

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
DATA_DIR = Path("Data/Energy_testing")
GLOB_PATTERN = "*.txt"
RECURSIVE = False

OUT_PATH = Path("E:/sPIV_PLIF_ProcessedData/Plots/energy_testing.png")
TITLE = "Laser energy testing"
XLABEL = "time (seconds)"
YLABEL = "Energy (mJ)"

DELIMITER: str | None = None  # e.g., "," for CSV-style files
SKIPROWS = 36  # number of header rows to skip
COMMENT: str | None = "#"  # comment marker in the file
XCOL: int | None = 0  # set None to use index for x
YCOL = 1  # y column index if 2D data

Y_SCALE = 1000.0  # J -> mJ
Y_LIMITS = (0.0, 130.0)

ALPHA = 0.6
GRID = False
LEGEND = True
FIGSIZE = (6.0, 4.0)
DPI = 300
LINE_CMAP: str | None = None  # e.g., "cmr.ember"
LINE_COLORS: list[str] | None = ["#cecece", "#082a54", "#e02b35", "#f0c571", "#59a89c", "#a559aa"]  # e.g., ["red", "blue", "green"]
LABELS: list[str] | None = ["8-29-2025", "9-10-2025, pre-sample", "9-10-2025, post-sample", "9-26-2025, pre-sample", "9-26-2025, post-sample", "10-01-2025"]  # e.g., ["run1", "run2", "run3"]

def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    if RECURSIVE:
        txt_paths = sorted(DATA_DIR.rglob(GLOB_PATTERN))
    else:
        txt_paths = sorted(DATA_DIR.glob(GLOB_PATTERN))

    if not txt_paths:
        raise FileNotFoundError(f"No files matched {GLOB_PATTERN} in {DATA_DIR}")

    plot_lines_from_txt_files(
        txt_paths,
        out_path=OUT_PATH,
        delimiter=DELIMITER,
        skiprows=SKIPROWS,
        comment=COMMENT,
        xcol=XCOL,
        ycol=YCOL,
        title=TITLE,
        xlabel=XLABEL,
        ylabel=YLABEL,
        figsize=FIGSIZE,
        dpi=DPI,
        line_cmap=LINE_CMAP,
        line_colors=LINE_COLORS,
        line_alpha=ALPHA,
        labels=LABELS,
        grid=GRID,
        legend=LEGEND,
        ylim=Y_LIMITS,
        y_scale=Y_SCALE,
    )
    print(f"Saved line plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
