"""
Compute mean across time (axis=2) for 3D .npy stacks.

Usage:
    python tools/compute_mean_single.py --inputs PATH1 PATH2 [--axis 2] [--suffix _mean]

Defaults use nanmean to ignore NaNs and save alongside inputs with the suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def compute_mean(in_path: Path, axis: int, suffix: str) -> Path:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    arr = np.load(in_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array; got shape {arr.shape} in {in_path}")
    if axis < 0:
        axis = arr.ndim + axis
    if axis != 2:
        # Allow user override, but remind of default expectation.
        sys.stderr.write(f"Warning: computing mean along axis {axis} (default is 2 for time)\n")

    mean_arr = np.nanmean(arr, axis=axis)
    out_path = in_path.with_name(in_path.name + suffix + ".npy") if in_path.suffix == "" else in_path.with_name(in_path.stem + suffix + in_path.suffix)
    np.save(out_path, mean_arr)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean over axis for 3D .npy stacks.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .npy file(s).")
    parser.add_argument("--axis", type=int, default=2, help="Axis to average over (default: 2).")
    parser.add_argument("--suffix", type=str, default="_mean", help="Suffix to append to output filenames.")
    args = parser.parse_args()

    for in_str in args.inputs:
        in_path = Path(in_str)
        out_path = compute_mean(in_path, axis=args.axis, suffix=args.suffix)
        print(f"Saved mean to {out_path}")


if __name__ == "__main__":
    main()
