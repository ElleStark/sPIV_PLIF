"""Helpers for line plots from multiple .txt files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
from itertools import cycle

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("sPIV_PLIF.txt_line_plots")


def _load_txt_line_data(
    path: Path,
    *,
    delimiter: Optional[str],
    skiprows: int,
    comment: Optional[str],
    xcol: Optional[int],
    ycol: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, comments=comment)
    if data.ndim == 0:
        raise ValueError(f"{path} contains no usable data.")
    if data.ndim == 1:
        y_vals = np.asarray(data, dtype=float)
        x_vals = np.arange(y_vals.size, dtype=float)
        return x_vals, y_vals
    if data.ndim != 2:
        raise ValueError(f"{path} must be 1D or 2D; got shape {data.shape}.")

    ncols = data.shape[1]
    if ycol < 0 or ycol >= ncols:
        raise IndexError(f"{path} ycol {ycol} out of bounds for {ncols} columns.")
    if xcol is None:
        x_vals = np.arange(data.shape[0], dtype=float)
    else:
        if xcol < 0 or xcol >= ncols:
            raise IndexError(f"{path} xcol {xcol} out of bounds for {ncols} columns.")
        x_vals = data[:, xcol].astype(float, copy=False)
    y_vals = data[:, ycol].astype(float, copy=False)
    return x_vals, y_vals


def plot_lines_from_txt_files(
    txt_paths: Sequence[Path],
    *,
    out_path: Path,
    labels: Optional[Sequence[str]] = None,
    delimiter: Optional[str] = None,
    skiprows: int = 0,
    comment: Optional[str] = "#",
    xcol: Optional[int] = 0,
    ycol: int = 1,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 300,
    line_width: float = 1.5,
    line_alpha: float = 0.9,
    line_styles: Optional[Sequence[str]] = None,
    line_colors: Optional[Sequence[object]] = None,
    line_cmap: Optional[str] = None,
    grid: bool = True,
    legend: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    y_scale: float = 1.0,
    print_stats: bool = True,
) -> None:
    """Plot multiple lines from .txt files and save the figure."""
    if len(txt_paths) == 0:
        raise ValueError("No .txt files provided for plotting.")

    labels_to_use: list[str] = []
    if labels is not None:
        if len(labels) != len(txt_paths):
            raise ValueError("labels length must match txt_paths length.")
        labels_to_use = list(labels)
    else:
        labels_to_use = [path.stem for path in txt_paths]

    style_cycle = cycle(line_styles if line_styles is not None else ["solid"])
    color_cycle: Iterable[object]
    if line_colors is not None:
        color_cycle = cycle(line_colors)
    elif line_cmap is not None:
        cmap = cmr.get_sub_cmap(line_cmap, 0.0, 1.0)
        colors = [cmap(v) for v in np.linspace(0, 1, len(txt_paths))] if len(txt_paths) > 1 else [cmap(0.2)]
        color_cycle = iter(colors)
    else:
        cmap = cmr.get_sub_cmap("cmr.rainforest", 0.0, 0.85)
        colors = [cmap(v) for v in np.linspace(0, 1, len(txt_paths))] if len(txt_paths) > 1 else [cmap(0.2)]
        color_cycle = iter(colors)

    plt.figure(figsize=figsize)
    for path, label in zip(txt_paths, labels_to_use):
        x_vals, y_vals = _load_txt_line_data(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            comment=comment,
            xcol=xcol,
            ycol=ycol,
        )
        if y_scale != 1.0:
            y_vals = y_vals * y_scale
        if print_stats:
            mean_val = float(np.nanmean(y_vals))
            std_val = float(np.nanstd(y_vals))
            print(f"{label}: mean={mean_val:.6g}, std={std_val:.6g}")
        plt.plot(
            x_vals,
            y_vals,
            label=label,
            linestyle=next(style_cycle),
            linewidth=line_width,
            alpha=line_alpha,
            color=next(color_cycle),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if grid:
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if legend:
        plt.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    logger.info("Saved line plot to %s", out_path)
