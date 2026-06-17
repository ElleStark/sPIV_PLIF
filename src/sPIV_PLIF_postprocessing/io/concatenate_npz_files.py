"""
Concatenate the frame arrays from two .npz files into one .npz file.

The first input is written at the beginning of the output, followed by the
second input. Frame keys are read in numeric arr_0, arr_1, ... order when
available, matching files created by np.savez(..., *frames).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def npz_keys_in_frame_order(npz_file: np.lib.npyio.NpzFile) -> list[str]:
    """Return .npz keys in numeric frame order when keys are arr_0, arr_1, ..."""

    def sort_key(key: str) -> tuple[int, str]:
        if key.startswith("arr_"):
            suffix = key[4:]
            if suffix.isdigit():
                return int(suffix), key
        return 10**12, key

    return sorted(npz_file.files, key=sort_key)


def load_npz_arrays(path: Path) -> list[np.ndarray]:
    """Load arrays from an .npz file in frame order."""
    with np.load(path, allow_pickle=False) as npz_file:
        return [np.asarray(npz_file[key]) for key in npz_keys_in_frame_order(npz_file)]


def concatenate_npz_files(first_npz: Path, second_npz: Path, output_npz: Path) -> Path:
    """Write arrays from first_npz followed by arrays from second_npz."""
    arrays = [*load_npz_arrays(first_npz), *load_npz_arrays(second_npz)]
    if not arrays:
        raise ValueError("Input .npz files do not contain any arrays.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, *arrays)
    return output_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate two .npz frame-list files, with the first file prepended to the second."
    )
    parser.add_argument("first_npz", type=Path, help=".npz file to place at the beginning.")
    parser.add_argument("second_npz", type=Path, help=".npz file to place after the first.")
    parser.add_argument("output_npz", type=Path, help="Output .npz file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = concatenate_npz_files(args.first_npz, args.second_npz, args.output_npz)
    print(f"Saved concatenated .npz to {output_path}")


if __name__ == "__main__":
    main()
