"""
Read paired .vc7 vector sets from two heads, interleave frames, and interpolate u/v/w
onto a shared physical grid (-149.5 to 149.5 mm in x and y at 0.5 mm spacing).

Outputs three float32 .npy stacks (u, v, w) with shape (ny, nx, n_frames).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import lvpyio as lv
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Target grid in millimeters
TARGET_MIN_MM = -149.5
TARGET_MAX_MM = 149.5
TARGET_STEP_MM = 0.5
TARGET_X = np.arange(TARGET_MIN_MM, TARGET_MAX_MM + TARGET_STEP_MM / 2, TARGET_STEP_MM)
TARGET_Y = np.arange(TARGET_MIN_MM, TARGET_MAX_MM + TARGET_STEP_MM / 2, TARGET_STEP_MM)

logger = logging.getLogger(__name__)

# -------------------------
# Helpers
# -------------------------


def ensure_axis_ascending(axis: np.ndarray, field: np.ndarray, axis_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Ensure a coordinate axis is ascending; flip data if needed."""
    if axis.size >= 2 and axis[1] < axis[0]:
        axis = axis[::-1]
        if axis_index == 0:
            field = field[::-1, :]
        else:
            field = field[:, ::-1]
    return axis, field


def extract_axes_from_vecbuffer(vecbuffer) -> tuple[np.ndarray, np.ndarray]:
    """
    Try to pull 1D physical axes from a vector buffer.
    Prefer grid.x/grid.y if present; fall back to scales.x/scales.y.
    """
    sample = vecbuffer[0]
    h, w = sample.shape

    scales = getattr(vecbuffer[0], "scales", None) if hasattr(vecbuffer, "__getitem__") else getattr(vecbuffer, "scales", None)
    if scales and getattr(scales, "x", None) is not None and getattr(scales, "y", None) is not None:
        # Build from slope/offset; assume uniform spacing
        x_axis = scales.x.offset + scales.x.slope * np.arange(w)
        y_axis = scales.y.offset + scales.y.slope * np.arange(h)
        return x_axis, y_axis

    raise RuntimeError("Could not find grid or scales on vector buffer; calibration metadata missing.")


def interpolate_vector_field(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate u/v/w onto target_x/target_y using linear interpolation."""
    if x_axis.size != u.shape[1] or y_axis.size != u.shape[0]:
        raise ValueError(
            f"Axis lengths (x={x_axis.size}, y={y_axis.size}) do not match field shape {u.shape}."
        )
    # Ensure axes ascending and fields aligned
    y_axis, u = ensure_axis_ascending(y_axis, u, axis_index=0)
    y_axis, v = ensure_axis_ascending(y_axis, v, axis_index=0)
    y_axis, w = ensure_axis_ascending(y_axis, w, axis_index=0)
    x_axis, u = ensure_axis_ascending(x_axis, u, axis_index=1)
    x_axis, v = ensure_axis_ascending(x_axis, v, axis_index=1)
    x_axis, w = ensure_axis_ascending(x_axis, w, axis_index=1)

    tgt_y, tgt_x = np.meshgrid(target_y, target_x, indexing="ij")
    points = np.stack([tgt_y.ravel(), tgt_x.ravel()], axis=-1)

    def interp_component(comp: np.ndarray) -> np.ndarray:
        fn = RegularGridInterpolator((y_axis, x_axis), comp, bounds_error=False, fill_value=np.nan)
        return fn(points).reshape(len(target_y), len(target_x)).astype(np.float32)

    return interp_component(u), interp_component(v), interp_component(w)


# -------------------------
# Main processing
# -------------------------


def collate_vectors_to_grid(
    head_a_path: Path,
    head_b_path: Path,
    target_x: np.ndarray = TARGET_X,
    target_y: np.ndarray = TARGET_Y,
    offset: int = 0,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read two vector sets, interleave frames A/B, and interpolate onto target grid.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")

    try:
        set_a = lv.read_set(str(head_a_path))
        set_b = lv.read_set(str(head_b_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read vector sets: {exc}") from exc

    available_pairs = min(len(set_a), len(set_b))
    total_frames = available_pairs * 2
    if offset >= total_frames:
        raise ValueError(f"Offset {offset} exceeds available frame count {total_frames}.")
    total_frames -= offset
    if limit is not None:
        total_frames = min(total_frames, limit)
    if total_frames <= 0:
        raise ValueError("No frames to process after offset/limit.")

    ny, nx = len(target_y), len(target_x)
    all_u = np.zeros((ny, nx, total_frames), dtype=np.float32)
    all_v = np.zeros_like(all_u)
    all_w = np.zeros_like(all_u)

    for global_idx in range(offset, offset + total_frames):
        source_set = set_a if global_idx % 2 == 0 else set_b
        source_idx = global_idx // 2
        head_label = "A" if global_idx % 2 == 0 else "B"
        try:
            vecbuffer = source_set[source_idx]
            vec_data = vecbuffer[0]
            depth = len(vec_data)
            planes = [vec_data.as_masked_array(plane=i) for i in range(depth)]
            x_axis, y_axis = extract_axes_from_vecbuffer(vecbuffer)
            u = np.ma.array([plane["u"] for plane in planes], dtype=np.float32)
            u = np.squeeze(u)
            v = np.ma.array([plane["v"] for plane in planes], dtype=np.float32)
            v = np.squeeze(v)
            w = np.ma.array([plane["w"] for plane in planes], dtype=np.float32)
            w = np.squeeze(w)
            u_i, v_i, w_i = interpolate_vector_field(u, v, w, x_axis, y_axis, target_x, target_y)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process frame {global_idx} (head {head_label}, source idx {source_idx}): {exc}"
            ) from exc

        frame_idx = global_idx - offset
        all_u[:, :, frame_idx] = u_i
        all_v[:, :, frame_idx] = v_i
        all_w[:, :, frame_idx] = w_i

    return all_u, all_v, all_w


def save_stacks(save_dir: Path, base_name: str, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{base_name}u.npy", u.astype(np.float32, copy=False))
    np.save(save_dir / f"{base_name}v.npy", v.astype(np.float32, copy=False))
    np.save(save_dir / f"{base_name}w.npy", w.astype(np.float32, copy=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Hard-coded paths: adjust to your dataset
    save_dir = Path("I:/Processed_Data/PIV/")
    save_name = "10.01.2025_30cms_nearbed_smTG15cm_neuHe0.875_air0.952_PIV0.01_iso_"
    piv_dir = Path("I:/10.01.2025_nearbed_L2/")
    piv_path1 = piv_dir / "10.01.2025_30cms_nearbed_smTG15cm_neuHe0.875_air0.952_PIV0.01_iso/Copy_L2/SubOverTimeMin_sl=all_01/StereoPIV_MPd(2x12x12_75%ov)/Resize.set"
    piv_path2 = piv_dir / "I:/10.01.2025_nearbed_L1/10.01.2025_30cms_nearbed_smTG15cm_neuHe0.875_air0.952_PIV0.01_iso/Copy_L1/SubOverTimeMin_sl=all/StereoPIV_MPd(2x12x12_75%ov)/Resize.set"

    logger.info("Processing %s and %s onto target grid %s..%s mm", piv_path1, piv_path2, TARGET_MIN_MM, TARGET_MAX_MM)
    u_stack, v_stack, w_stack = collate_vectors_to_grid(piv_path1, piv_path2)
    save_stacks(save_dir, save_name, u_stack, v_stack, w_stack)
    logger.info("Saved stacks to %s", save_dir)
