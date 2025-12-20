"""
Collate alternating frames from two LaVision .imx/.set files into a single NumPy stack.

Accepts paired lists of files so multiple datasets (two laser heads each) can
be converted in one run. Frames are trimmed/interpolated onto a square
physical grid using the calibration/scaling stored in the source files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import lvpyio as lv  # LaVision's package for importing .imx/.vc7 files
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Target grid in millimeters
TARGET_MIN_MM = -149.5
TARGET_MAX_MM = 149.5
TARGET_STEP_MM = 0.5
TARGET_X = np.arange(TARGET_MIN_MM, TARGET_MAX_MM + TARGET_STEP_MM / 2, TARGET_STEP_MM)
TARGET_Y = np.arange(TARGET_MIN_MM, TARGET_MAX_MM + TARGET_STEP_MM / 2, TARGET_STEP_MM)


def ensure_axis_ascending(axis: np.ndarray, frame: np.ndarray, axis_index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure the physical axis is ascending; reverse data if needed.
    """
    if axis.size >= 2 and axis[1] < axis[0]:
        axis = axis[::-1]
        frame = frame[:, ::-1] if axis_index == 1 else frame[::-1, :]
    return axis, frame


def extract_frame_and_axes(buffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get frame data (float32, masked -> NaN) and physical x/y axes from a buffer.
    """
    arr = buffer.as_masked_array()
    frame = np.array(arr.filled(np.nan), dtype=np.float32)
    frameData = buffer[0]
    x_scale = frameData.scales.x
    y_scale = frameData.scales.y
    x_axis = x_scale.offset + x_scale.slope * np.arange(frame.shape[1])
    y_axis = y_scale.offset + y_scale.slope * np.arange(frame.shape[0])
    return frame, x_axis, y_axis


def interpolate_frame_to_target(
    frame: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    x_axis, frame = ensure_axis_ascending(x_axis, frame, axis_index=1)
    y_axis, frame = ensure_axis_ascending(y_axis, frame, axis_index=0)

    interp_fn = RegularGridInterpolator(
        (y_axis, x_axis),
        frame,
        bounds_error=False,
        fill_value=np.nan,
    )
    tgt_y, tgt_x = np.meshgrid(target_y, target_x, indexing="ij")
    points = np.stack([tgt_y.ravel(), tgt_x.ravel()], axis=-1)
    interpolated = interp_fn(points).reshape(len(target_y), len(target_x))
    return interpolated.astype(np.float32)


def collate_heads(
    head_a_path: Path,
    head_b_path: Path,
    offset: int = 0,
    limit: Optional[int] = None,
    target_x: np.ndarray = TARGET_X,
    target_y: np.ndarray = TARGET_Y,
) -> np.ndarray:
    """
    Read two .imx/.set files (head A/B), interleave frames, and interpolate onto a square grid.
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

    combined = np.zeros((len(target_y), len(target_x), total_frames), dtype=np.float32)

    for global_idx in range(offset, offset + total_frames):
        source_set = set_a if global_idx % 2 == 0 else set_b
        source_idx = global_idx // 2
        try:
            buffer = source_set[source_idx]
            frame, x_axis, y_axis = extract_frame_and_axes(buffer)
            interp_frame = interpolate_frame_to_target(frame, x_axis, y_axis, target_x, target_y)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read/interpolate frame {global_idx} (source index {source_idx}) "
                f"from {'head A' if global_idx % 2 == 0 else 'head B'}: {exc}"
            ) from exc

        combined[:, :, global_idx - offset] = interp_frame

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
    parser.add_argument("--head-a", nargs="+", help="Paths to head-A .imx/.set files.")
    parser.add_argument("--head-b", nargs="+", help="Paths to head-B .imx/.set files.")
    parser.add_argument(
        "--pairs-file",
        type=Path,
        help="Optional text file with lines: headA_path | headB_path | optional_name. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        help="Optional output file names (without extension) for each dataset; must match number of inputs.",
    )
    parser.add_argument("--save-dir", type=Path, help="Directory for output .npy files.")
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


def prompt_for_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Lightly interactive prompting for required inputs when running in a TTY.

    If stdin is not a TTY, the function leaves args unchanged so batch runs fail fast.
    """
    if not sys.stdin.isatty():
        return args

    if not args.pairs_file and (not args.head_a or not args.head_b):
        print("No input files supplied. Provide a pairs file or head-A/head-B lists.")
        use_pairs = input("Use pairs file? [y/N]: ").strip().lower().startswith("y")
        if use_pairs:
            pf = input("Path to pairs file: ").strip()
            if pf:
                args.pairs_file = Path(pf)
        else:
            head_a_raw = input("Head-A paths (comma-separated): ").strip()
            head_b_raw = input("Head-B paths (comma-separated): ").strip()
            if head_a_raw:
                args.head_a = [p.strip() for p in head_a_raw.split(",") if p.strip()]
            if head_b_raw:
                args.head_b = [p.strip() for p in head_b_raw.split(",") if p.strip()]
            names_raw = input("Optional names (comma-separated, leave blank for auto): ").strip()
            if names_raw:
                args.names = [n.strip() for n in names_raw.split(",") if n.strip()]

    if args.save_dir is None:
        save_dir_raw = input("Output directory for .npy files: ").strip()
        if save_dir_raw:
            args.save_dir = Path(save_dir_raw)

    return args


def load_pairs_from_file(path: Path) -> Tuple[Sequence[Path], Sequence[Path], Sequence[str]]:
    """
    Read a simple text file where each non-empty, non-comment line is:
    headA_path | headB_path | optional_name
    """
    head_a: list[Path] = []
    head_b: list[Path] = []
    names: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                raise ValueError(f"Line {line_no} in {path} must have at least headA | headB")

            head_a.append(Path(parts[0]))
            head_b.append(Path(parts[1]))
            if len(parts) >= 3 and parts[2]:
                names.append(parts[2])

    # If some lines had names and others did not, ensure alignment
    if names and len(names) != len(head_a):
        raise ValueError(
            f"Found {len(names)} names for {len(head_a)} datasets in {path}; "
            "either provide names on all lines or none."
        )

    if not head_a:
        raise ValueError(f"No usable dataset lines found in {path}")

    return head_a, head_b, names


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
    args = prompt_for_missing_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    head_a_paths: Sequence[Path]
    head_b_paths: Sequence[Path]
    names: Sequence[str]

    if args.pairs_file:
        try:
            head_a_paths, head_b_paths, names_from_file = load_pairs_from_file(args.pairs_file)
        except Exception as exc:
            logger.error("Failed to read pairs file: %s", exc)
            raise SystemExit(1) from exc
        names = names_from_file or (args.names or [])
    else:
        if not args.head_a or not args.head_b:
            logger.error("Provide --head-a and --head-b lists, or --pairs-file.")
            raise SystemExit(1)
        head_a_paths = [Path(p) for p in args.head_a]
        head_b_paths = [Path(p) for p in args.head_b]
        names = args.names or []

    try:
        validate_inputs(head_a_paths, head_b_paths, names)
    except Exception as exc:
        logger.error(exc)
        raise SystemExit(1) from exc

    if args.save_dir is None:
        logger.error("Output directory (--save-dir) is required.")
        raise SystemExit(1)

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
            np.save(output_path, combined_data.astype(np.float32, copy=False))
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
