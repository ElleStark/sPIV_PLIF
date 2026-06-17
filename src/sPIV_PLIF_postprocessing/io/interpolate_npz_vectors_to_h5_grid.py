"""
Interpolate variable-size PIV velocity frames from .npz files onto an HDF5 grid.

The input .npz files are expected to contain one array per frame, as produced by
np.savez(..., *frames). The associated x/y grid files must have the same number
of frames as the selected u/v/w data files. Source and target grids may be 1D
axes or 2D mesh grids.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

logger = logging.getLogger(__name__)


def npz_keys_in_frame_order(npz_file: np.lib.npyio.NpzFile) -> list[str]:
    """Return .npz keys in numeric frame order when keys are arr_0, arr_1, ..."""

    def sort_key(key: str) -> tuple[int, str]:
        if key.startswith("arr_"):
            suffix = key[4:]
            if suffix.isdigit():
                return int(suffix), key
        return 10**12, key

    return sorted(npz_file.files, key=sort_key)


def load_npz_frames(path: Path) -> list[np.ndarray]:
    """Load all arrays from an .npz file in frame order."""
    with np.load(path, allow_pickle=False) as npz_file:
        return [np.asarray(npz_file[key]) for key in npz_keys_in_frame_order(npz_file)]


def load_h5_grid(path: Path, x_dataset: str, y_dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Read x/y target grids from an HDF5 file."""
    with h5py.File(path, "r") as h5_file:
        if x_dataset not in h5_file:
            raise KeyError(f"Dataset {x_dataset!r} was not found in {path}")
        if y_dataset not in h5_file:
            raise KeyError(f"Dataset {y_dataset!r} was not found in {path}")
        target_x = np.asarray(h5_file[x_dataset])
        target_y = np.asarray(h5_file[y_dataset])

    return target_x, target_y


def as_target_mesh(target_x: np.ndarray, target_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return target coordinates as 2D x/y mesh grids."""
    if target_x.ndim == 1 and target_y.ndim == 1:
        return np.meshgrid(target_x, target_y, indexing="xy")
    if target_x.ndim == 2 and target_y.ndim == 2 and target_x.shape == target_y.shape:
        return target_x, target_y
    raise ValueError(
        "Target x/y grids must both be 1D axes or matching 2D mesh grids; "
        f"got x shape {target_x.shape}, y shape {target_y.shape}."
    )


def ensure_axis_ascending(
    axis: np.ndarray,
    field: np.ndarray,
    axis_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure a 1D coordinate axis is ascending and flip field data if needed."""
    if axis.size >= 2 and axis[1] < axis[0]:
        axis = axis[::-1]
        field = field[::-1, :] if axis_index == 0 else field[:, ::-1]
    return axis, field


def looks_like_rectilinear_mesh(x_grid: np.ndarray, y_grid: np.ndarray) -> bool:
    """Check whether 2D mesh grids can be represented by 1D x/y axes."""
    if x_grid.ndim != 2 or y_grid.ndim != 2 or x_grid.shape != y_grid.shape:
        return False
    return bool(
        np.allclose(x_grid, x_grid[0, :][None, :], equal_nan=True)
        and np.allclose(y_grid, y_grid[:, 0][:, None], equal_nan=True)
    )


def source_axes_from_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return 1D x/y axes for rectilinear grids, or None for curvilinear grids.
    """
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        x_axis = x_grid
        y_axis = y_grid
    elif looks_like_rectilinear_mesh(x_grid, y_grid):
        x_axis = x_grid[0, :]
        y_axis = y_grid[:, 0]
    else:
        return None

    if field_shape != (y_axis.size, x_axis.size):
        raise ValueError(
            f"Source grid lengths x={x_axis.size}, y={y_axis.size} do not match "
            f"field shape {field_shape}."
        )

    return np.asarray(x_axis), np.asarray(y_axis)


def interpolate_frame(
    field: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    method: str = "linear",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Interpolate one 2D field onto the target grid."""
    field = np.asarray(field, dtype=np.float32)
    source_x = np.asarray(source_x)
    source_y = np.asarray(source_y)
    target_mesh_x, target_mesh_y = as_target_mesh(target_x, target_y)

    if field.ndim != 2:
        raise ValueError(f"Each field must be 2D; got shape {field.shape}.")

    axes = source_axes_from_grid(source_x, source_y, field.shape)
    if axes is not None:
        x_axis, y_axis = axes
        y_axis, field = ensure_axis_ascending(y_axis, field, axis_index=0)
        x_axis, field = ensure_axis_ascending(x_axis, field, axis_index=1)
        interpolator = RegularGridInterpolator(
            (y_axis, x_axis),
            field,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
        points = np.column_stack((target_mesh_y.ravel(), target_mesh_x.ravel()))
        return interpolator(points).reshape(target_mesh_x.shape).astype(np.float32)

    if source_x.shape != field.shape or source_y.shape != field.shape:
        raise ValueError(
            "Curvilinear source x/y grids must match the field shape; "
            f"got x {source_x.shape}, y {source_y.shape}, field {field.shape}."
        )

    valid = np.isfinite(source_x) & np.isfinite(source_y) & np.isfinite(field)
    if not np.any(valid):
        return np.full(target_mesh_x.shape, fill_value, dtype=np.float32)

    interpolated = griddata(
        (source_x[valid].ravel(), source_y[valid].ravel()),
        field[valid].ravel(),
        (target_mesh_x, target_mesh_y),
        method=method,
        fill_value=fill_value,
    )
    return interpolated.astype(np.float32)


def validate_frame_counts(frame_lists: Iterable[tuple[str, list[np.ndarray]]]) -> int:
    """Ensure all input lists have the same number of frames."""
    counts = {name: len(frames) for name, frames in frame_lists}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        detail = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise ValueError(f"Input frame counts do not match: {detail}.")
    count = unique_counts.pop()
    if count == 0:
        raise ValueError("Input files do not contain any frames.")
    return count


def interpolate_vector_npz_to_grid(
    vector_npz_paths: Mapping[str, Path],
    xgrid_npz_path: Path,
    ygrid_npz_path: Path,
    h5_grid_path: Path,
    output_dir: Path,
    x_dataset: str = "x_grid",
    y_dataset: str = "y_grid",
    output_prefix: str = "",
    method: str = "linear",
    fill_value: float = np.nan,
) -> dict[str, Path]:
    """
    Interpolate velocity component frame lists to the target HDF5 grid and save .npy stacks.

    Saved stacks have shape (ny, nx, n_frames).
    """
    if not vector_npz_paths:
        raise ValueError("At least one velocity component .npz path is required.")

    component_frames = {
        component: load_npz_frames(path)
        for component, path in vector_npz_paths.items()
        if path is not None
    }
    if not component_frames:
        raise ValueError("At least one velocity component .npz path is required.")
    x_frames = load_npz_frames(xgrid_npz_path)
    y_frames = load_npz_frames(ygrid_npz_path)
    frame_count = validate_frame_counts(
        (*component_frames.items(), ("xgrid", x_frames), ("ygrid", y_frames))
    )

    target_x, target_y = load_h5_grid(h5_grid_path, x_dataset, y_dataset)
    target_mesh_x, _ = as_target_mesh(target_x, target_y)
    output_shape = (*target_mesh_x.shape, frame_count)
    outputs = {component: np.empty(output_shape, dtype=np.float32) for component in component_frames}

    for frame_idx, (x_grid, y_grid) in enumerate(zip(x_frames, y_frames, strict=True)):
        shifted_y_grid = y_grid + 150
        logger.info("Interpolating frame %s/%s", frame_idx + 1, frame_count)
        for component, frames in component_frames.items():
            outputs[component][:, :, frame_idx] = interpolate_frame(
                frames[frame_idx],
                x_grid,
                shifted_y_grid,
                target_x,
                target_y,
                method=method,
                fill_value=fill_value,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for component, stack in outputs.items():
        output_path = output_dir / f"{output_prefix}{component}.npy"
        np.save(output_path, stack)
        output_paths[component] = output_path
        logger.info("Saved interpolated %s stack to %s", component, output_path)

    return output_paths


def interpolate_velocity_npz_to_grid(
    u_npz_path: Path,
    v_npz_path: Path,
    xgrid_npz_path: Path,
    ygrid_npz_path: Path,
    h5_grid_path: Path,
    output_dir: Path,
    x_dataset: str = "x_grid",
    y_dataset: str = "y_grid",
    output_prefix: str = "",
    method: str = "linear",
    fill_value: float = np.nan,
    w_npz_path: Path | None = None,
) -> tuple[Path, ...]:
    """
    Backward-compatible wrapper for interpolating u/v and optional w stacks.
    """
    vector_npz_paths = {"u": u_npz_path, "v": v_npz_path}
    if w_npz_path is not None:
        vector_npz_paths["w"] = w_npz_path
    output_paths = interpolate_vector_npz_to_grid(
        vector_npz_paths=vector_npz_paths,
        xgrid_npz_path=xgrid_npz_path,
        ygrid_npz_path=ygrid_npz_path,
        h5_grid_path=h5_grid_path,
        output_dir=output_dir,
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        output_prefix=output_prefix,
        method=method,
        fill_value=fill_value,
    )
    return tuple(output_paths[component] for component in vector_npz_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate variable-size u/v/w .npz frame lists onto x/y grids from an HDF5 file."
    )
    parser.add_argument("--h5-grid", type=Path, required=True, help="HDF5 file containing target x/y grids.")
    parser.add_argument("--h5-x-dataset", default="x_grid", help="Target x-grid dataset name.")
    parser.add_argument("--h5-y-dataset", default="y_grid", help="Target y-grid dataset name.")
    parser.add_argument("--u-npz", type=Path, help=".npz file containing u frames.")
    parser.add_argument("--v-npz", type=Path, help=".npz file containing v frames.")
    parser.add_argument("--w-npz", type=Path, help=".npz file containing w frames.")
    parser.add_argument("--xgrid-npz", type=Path, required=True, help=".npz file containing per-frame x grids.")
    parser.add_argument("--ygrid-npz", type=Path, required=True, help=".npz file containing per-frame y grids.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for saved component .npy files.")
    parser.add_argument("--output-prefix", default="", help="Optional prefix for output filenames.")
    parser.add_argument(
        "--method",
        choices=("linear", "nearest"),
        default="linear",
        help="Interpolation method. Use nearest to avoid NaNs inside sparse/irregular grids.",
    )
    parser.add_argument("--fill-value", type=float, default=np.nan, help="Value outside interpolation bounds.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    vector_npz_paths = {
        component: path
        for component, path in (("u", args.u_npz), ("v", args.v_npz), ("w", args.w_npz))
        if path is not None
    }
    if not vector_npz_paths:
        raise SystemExit("At least one of --u-npz, --v-npz, or --w-npz is required.")

    interpolate_vector_npz_to_grid(
        vector_npz_paths=vector_npz_paths,
        xgrid_npz_path=args.xgrid_npz,
        ygrid_npz_path=args.ygrid_npz,
        h5_grid_path=args.h5_grid,
        output_dir=args.output_dir,
        x_dataset=args.h5_x_dataset,
        y_dataset=args.h5_y_dataset,
        output_prefix=args.output_prefix,
        method=args.method,
        fill_value=args.fill_value,
    )


if __name__ == "__main__":
    main()
