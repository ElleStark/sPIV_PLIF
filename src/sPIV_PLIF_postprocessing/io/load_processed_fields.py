"""
Helpers to load processed velocity/concentration fields from .npy files.

Separates data loading from plotting so callers can reuse the same arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FieldStacks:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    c: np.ndarray

    def summary(self) -> str:
        return (
            f"u {self.u.shape} dtype={self.u.dtype}, "
            f"v {self.v.shape} dtype={self.v.dtype}, "
            f"w {self.w.shape} dtype={self.w.dtype}, "
            f"c {self.c.shape} dtype={self.c.dtype}"
        )


def load_fields(
    u_path: Path,
    v_path: Path,
    w_path: Path,
    c_path: Path,
    *,
    enforce_float32: bool = True,
    mmap_mode: Optional[str] = None,
) -> FieldStacks:
    """
    Load u/v/w/c stacks from .npy files.

    Parameters
    ----------
    u_path, v_path, w_path, c_path : Path
        Locations of the .npy files.
    enforce_float32 : bool
        If True, cast arrays to float32 on load.
    mmap_mode : str or None
        Optional numpy memmap mode ('r' recommended) to reduce RAM usage.

    Returns
    -------
    FieldStacks
        Container with u, v, w, c arrays.
    """
    def _load(path: Path) -> np.ndarray:
        arr = np.load(path, mmap_mode=mmap_mode)
        if enforce_float32 and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr

    u = _load(u_path)
    v = _load(v_path)
    w = _load(w_path)
    c = _load(c_path)

    return FieldStacks(u=u, v=v, w=w, c=c)


if __name__ == "__main__":
    # Example usage (edit paths as needed)
    base = Path("E:/sPIV_PLIF_ProcessedData/PIV/")
    fields = load_fields(
        base / "example_u.npy",
        base / "example_v.npy",
        base / "example_w.npy",
        base / "example_c.npy",
        mmap_mode=None,
    )
    print(fields.summary())
