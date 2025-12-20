"""
Collate alternating frames from two LaVision .imx/.set files into a single NumPy stack.

Accepts paired lists of files so multiple datasets (two laser heads each) can
be converted in one run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import lvpyio as lv  # LaVision's package for importing .imx/.vc7 files
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def collate_heads(
    head_a_path: Path,
    head_b_path: Path,
    offset: int = 0,
    limit: Optional[int] = None,
) -> np.ndarray:
    """
    Read two .imx/.set files (head A/B) and interleave frames into one ndarray.

    Frames are assumed to alternate A, B, A, B, ...; an offset can drop the first N frames.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")

    try:
        set_a = lv.read_set(str(head_a_path))
        set_b = lv.read_set(str(head_b_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read sets {head_a_path} or {head_b_path}: {exc}") from exc

    available_pairs = min(len(set_a), len(set_b))
    total_frames = available_pairs * 2
    if offset >= total_frames:
        raise ValueError(f"Offset {offset} exceeds available frame count {total_frames}.")

    total_frames -= offset
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative when provided")
        total_frames = min(total_frames, limit)

    if total_frames == 0:
        raise ValueError("No frames remain after applying offset and limit.")

    # Use whichever head we start with to define the array shape.
    sample_set = set_a if offset % 2 == 0 else set_b
    sample_index = offset // 2
    try:
        sample_array = sample_set[sample_index].as_masked_array().data
    except Exception as exc:
        raise RuntimeError(f"Failed to read sample frame for shape inspection: {exc}") from exc

    combined = np.zeros((*sample_array.shape, total_frames), dtype=np.float32)

    for global_idx in range(offset, offset + total_frames):
        source_set = set_a if global_idx % 2 == 0 else set_b
        source_idx = global_idx // 2
        try:
            buffer = source_set[source_idx]
            frame = buffer.as_masked_array().data
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read frame {global_idx} (source index {source_idx}) "
                f"from {'head A' if global_idx % 2 == 0 else 'head B'}: {exc}"
            ) from exc

        combined[:, :, global_idx - offset] = frame

    return combined


def save_qc_frames(
    data: np.ndarray,
    qc_dir: Path,
    frame_indices: Sequence[int],
    prefix: str,
) -> None:
    """Save quick-look images for specified frames."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_indices:
        if frame < 0 or frame >= data.shape[2]:
            logger.warning("Skipping QC frame %s; index out of range (0-%s).", frame, data.shape[2] - 1)
            continue

        fig, ax = plt.subplots()
        im = ax.imshow(data[:, :, frame], cmap="jet", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        qc_path = qc_dir / f"{prefix}_frame_{frame}.png"
        fig.savefig(qc_path)
        plt.close(fig)
        logger.info("Saved QC frame %s to %s", frame, qc_path)


def build_output_name(
    head_a_path: Path,
    head_b_path: Path,
    provided_name: Optional[str],
) -> str:
    if provided_name:
        return provided_name
    return f"{head_a_path.stem}_and_{head_b_path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collate alternating frames from paired laser-head .imx/.set files into .npy stacks. "
            "Provide matching lists for --head-a and --head-b to process multiple datasets."
        )
    )
    parser.add_argument("--head-a", nargs="+", required=True, help="Paths to head-A .imx/.set files.")
    parser.add_argument("--head-b", nargs="+", required=True, help="Paths to head-B .imx/.set files.")
    parser.add_argument(
        "--names",
        nargs="*",
        help="Optional output file names (without extension) for each dataset; must match number of inputs.",
    )
    parser.add_argument("--save-dir", required=True, type=Path, help="Directory for output .npy files.")
    parser.add_argument(
        "--qc-frames",
        nargs="*",
        type=int,
        default=[],
        help="Frame indices to save as QC PNGs (empty means no QC images).",
    )
    parser.add_argument(
        "--qc-subdir",
        default="QC",
        help="Subdirectory inside save-dir for QC images (used only when --qc-frames is set).",
    )
    parser.add_argument("--offset", type=int, default=0, help="Number of initial frames to skip.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of frames to export after applying offset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_inputs(head_a: Sequence[Path], head_b: Sequence[Path], names: Sequence[str]) -> None:
    if len(head_a) != len(head_b):
        raise ValueError(f"Need the same number of head-A and head-B files (got {len(head_a)} vs {len(head_b)}).")
    if names and len(names) != len(head_a):
        raise ValueError(f"Provided {len(names)} names, but {len(head_a)} datasets.")

    missing = [p for p in list(head_a) + list(head_b) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {', '.join(str(p) for p in missing)}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    head_a_paths = [Path(p) for p in args.head_a]
    head_b_paths = [Path(p) for p in args.head_b]
    names = args.names or []

    try:
        validate_inputs(head_a_paths, head_b_paths, names)
    except Exception as exc:
        logger.error(exc)
        raise SystemExit(1) from exc

    args.save_dir.mkdir(parents=True, exist_ok=True)
    qc_dir: Optional[Path] = None
    if args.qc_frames:
        qc_dir = args.save_dir / args.qc_subdir

    for idx, (head_a, head_b) in enumerate(zip(head_a_paths, head_b_paths)):
        output_name = build_output_name(head_a, head_b, names[idx] if names else None)
        output_path = args.save_dir / f"{output_name}.npy"

        if output_path.exists() and not args.overwrite:
            logger.warning("Output %s exists; skipping (use --overwrite to replace).", output_path)
            continue

        logger.info("Collating %s and %s -> %s", head_a, head_b, output_path)
        try:
            combined_data = collate_heads(head_a, head_b, offset=args.offset, limit=args.limit)
        except Exception as exc:
            logger.error("Failed to collate %s and %s: %s", head_a, head_b, exc)
            continue

        try:
            np.save(output_path, combined_data)
            logger.info("Saved combined stack %s with shape %s", output_path, combined_data.shape)
        except Exception as exc:
            logger.error("Failed to save %s: %s", output_path, exc)
            continue

        if qc_dir and args.qc_frames:
            try:
                save_qc_frames(combined_data, qc_dir, args.qc_frames, output_name)
            except Exception as exc:
                logger.warning("Could not save QC frames for %s: %s", output_name, exc)


if __name__ == "__main__":
    main()
