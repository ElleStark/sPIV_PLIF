"""Utilities to compute and plot concentration PDFs at a given (x, y) location."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


def concentration_timeseries_from_file(
    path: Path | str,
    y_idx: int,
    x_idx: int,
    *,
    t_slice: slice | None = None,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    """
    Load a concentration stack and return the time series at (y_idx, x_idx).

    The concentration file is expected to be shaped (y, x, t).
    """
    arr = np.load(path, mmap_mode=mmap_mode)
    if arr.ndim != 3:
        raise ValueError(f"Expected concentration stack with shape (y, x, t); got {arr.shape}")
    return np.asarray(arr[y_idx, x_idx, t_slice])


def compute_concentration_pdf(
    samples: np.ndarray | Iterable[float],
    *,
    bins: int | Iterable[float] = 100,
    value_range: Tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a probability density function from scalar samples.

    Returns (bin_centers, pdf, bin_edges).
    """
    values = np.asarray(samples)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite concentration samples available for PDF computation")
    pdf, edges = np.histogram(finite, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pdf, edges


def plot_concentration_pdf(
    bin_centers: np.ndarray,
    pdf: np.ndarray,
    *,
    out_path: Path,
    title: str = "Concentration PDF",
    xlabel: str = "Concentration",
    ylabel: str = "Probability density",
) -> None:
    """Plot and save a concentration PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, pdf, lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
