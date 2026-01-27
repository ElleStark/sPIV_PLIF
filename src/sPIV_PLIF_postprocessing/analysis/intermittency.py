"""Intermittency calculations for concentration fields."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_intermittency(
    concentration: np.ndarray,
    threshold: float,
    *,
    axis: int = 2,
    percent: bool = True,
) -> np.ndarray:
    """
    Compute intermittency as the fraction of time concentration exceeds a threshold.

    Parameters
    ----------
    concentration
        Concentration stack with time along `axis`. Expected shape (y, x, t) by default.
    threshold
        Threshold value; samples strictly greater than this count as "intermittent".
    axis
        Axis corresponding to time. Defaults to 2 for (y, x, t) arrays.
    percent
        If True, return percentage [0, 100]; otherwise return fraction [0, 1].
    """
    arr = np.asarray(concentration)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D concentration array (y, x, t); got shape {arr.shape}")

    time_axis = axis % arr.ndim
    valid = np.isfinite(arr)
    exceeds = (arr > threshold) & valid

    counts = np.sum(exceeds, axis=time_axis)
    valid_counts = np.sum(valid, axis=time_axis)
    intermittency = np.divide(
        counts,
        valid_counts,
        out=np.zeros_like(counts, dtype=float),
        where=valid_counts > 0,
    )
    if percent:
        intermittency = intermittency * 100.0
    return intermittency


def compute_intermittency_from_file(
    path: Path | str,
    threshold: float,
    *,
    axis: int = 2,
    percent: bool = True,
    allow_pickle: bool | None = False,
    mmap_mode: str | None = None,
) -> np.ndarray:
    """
    Load a concentration stack from disk and compute intermittency.

    Parameters mirror `np.load` for flexibility on pickle/mmap use.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Concentration file not found: {in_path}")
    arr = np.load(in_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
    return compute_intermittency(arr, threshold, axis=axis, percent=percent)
