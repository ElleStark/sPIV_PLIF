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
import matplotlib.pyplot as plt
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


def extract_axes_from_vecbuffer(vecbuffer, vec_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Try to pull 1D physical axes from a vector buffer using scales.x/scales.y.
    """
    frame = vecbuffer[0]
    arr = frame.as_masked_array()
    h, w = frame.shape

    scales = getattr(frame, "scales", None)
    if scales and getattr(scales, "x", None) is not None and getattr(scales, "y", None) is not None:
        # Build from slope/offset; assume uniform spacing
        x_axis = scales.x.offset + scales.x.slope * vec_grid * np.arange(w)
        y_axis = scales.y.offset + scales.y.slope * vec_grid * np.arange(h)
        return x_axis, y_axis

    raise RuntimeError("Could not find scales on vector buffer; calibration metadata missing.")


def interpolate_vector_field(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    qc_path: Optional[Path] = None,
    qc_stride: int = 10,
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

    if qc_path is not None:
        mask = np.isfinite(u) & np.isfinite(v)
        X_raw, Y_raw = np.meshgrid(x_axis, y_axis, indexing="xy")
        Xs = X_raw[::qc_stride, ::qc_stride]
        Ys = Y_raw[::qc_stride, ::qc_stride]
        us = u[::qc_stride, ::qc_stride]
        vs = v[::qc_stride, ::qc_stride]
        ms = mask[::qc_stride, ::qc_stride]
        Xs = Xs[ms]
        Ys = Ys[ms]
        us = us[ms]
        vs = vs[ms]
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.quiver(
            Xs,
            Ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=50,
            width=0.002,
            pivot="mid",
        )
        plt.xlabel("x (native units)")
        plt.ylabel("y (native units)")
        plt.title("QC: raw u/v before interpolation")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(qc_path, dpi=200)
        plt.close()
        logger.info("Saved QC raw quiver to %s", qc_path)

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
    vec_grid: int,
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
            x_axis, y_axis = extract_axes_from_vecbuffer(vecbuffer, vec_grid)
            arr = vecbuffer[0].as_masked_array()
            u = np.array(np.ma.filled(arr["u"], np.nan), dtype=np.float32)
            v = np.array(np.ma.filled(arr["v"], np.nan), dtype=np.float32)
            w = np.array(np.ma.filled(arr["w"], np.nan), dtype=np.float32)
            if source_idx == 0:  # save a quick raw quiver for the first output frame
                save_buffer_quiver(u, v, x_axis, y_axis, save_dir, save_name, head_label, frame_idx=global_idx, stride=10)
            qc_path = None
            if source_idx == 0:
                qc_dir = save_dir / "QC"
                qc_path = qc_dir / f"{save_name}raw_interp_input_head{head_label}_frame{global_idx}.png"
            u_i, v_i, w_i = interpolate_vector_field(
                u, v, w, x_axis, y_axis, target_x, target_y, qc_path=qc_path, qc_stride=10
            )
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


def save_qc_quiver(
    u: np.ndarray,
    v: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    save_dir: Path,
    base_name: str,
    frame_idx: int = 0,
    stride: int = 10,
) -> None:
    """Save a quick quiver plot for a given frame on the target grid."""
    qc_dir = save_dir / "QC"
    qc_dir.mkdir(parents=True, exist_ok=True)

    u_f = u[:, :, frame_idx]
    v_f = v[:, :, frame_idx]
    mask = np.isfinite(u_f) & np.isfinite(v_f)

    X, Y = np.meshgrid(target_x, target_y, indexing="xy")
    X_s = X[::stride, ::stride]
    Y_s = Y[::stride, ::stride]
    u_s = u_f[::stride, ::stride]
    v_s = v_f[::stride, ::stride]
    mask_s = mask[::stride, ::stride]

    X_s = X_s[mask_s]
    Y_s = Y_s[mask_s]
    u_s = u_s[mask_s]
    v_s = v_s[mask_s]

    plt.figure(figsize=(8, 6))
    plt.quiver(
        X_s,
        Y_s,
        u_s,
        v_s,
        angles="xy",
        scale_units="xy",
        scale=50,
        width=0.002,
        pivot="mid",
    )
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"Vector field QC (frame {frame_idx})")
    plt.axis("equal")
    plt.xlim(target_x.min(), target_x.max())
    plt.ylim(target_y.min(), target_y.max())
    plt.tight_layout()
    qc_path = qc_dir / f"{base_name}qc_frame{frame_idx}.png"
    plt.savefig(qc_path, dpi=200)
    plt.close()
    logger.info("Saved QC quiver to %s", qc_path)


def save_buffer_quiver(
    u: np.ndarray,
    v: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    save_dir: Path,
    base_name: str,
    head_label: str,
    frame_idx: int,
    stride: int = 10,
) -> None:
    """Save a quiver plot directly from raw buffer data on its native grid."""
    mask = np.isfinite(u) & np.isfinite(v)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")

    X_s = X[::stride, ::stride]
    Y_s = Y[::stride, ::stride]
    u_s = u[::stride, ::stride]
    v_s = v[::stride, ::stride]
    mask_s = mask[::stride, ::stride]

    X_s = X_s[mask_s]
    Y_s = Y_s[mask_s]
    u_s = u_s[mask_s]
    v_s = v_s[mask_s]

    qc_dir = save_dir / "QC"
    qc_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.quiver(
        X_s,
        Y_s,
        u_s,
        v_s,
        angles="xy",
        scale_units="xy",
        scale=50,
        width=0.002,
        pivot="mid",
    )
    plt.xlabel("x (native units)")
    plt.ylabel("y (native units)")
    plt.title(f"Raw buffer quiver (head {head_label}, frame {frame_idx})")
    plt.axis("equal")
    plt.tight_layout()
    qc_path = qc_dir / f"{base_name}raw_buffer_head{head_label}_frame{frame_idx}.png"
    plt.savefig(qc_path, dpi=200)
    plt.close()
    logger.info("Saved raw buffer QC quiver to %s", qc_path)


def save_raw_qc_quiver(
    head_path: Path,
    save_dir: Path,
    base_name: str,
    frame_idx: int = 0,
    stride: int = 10,
) -> None:
    """
    Save a raw quiver plot directly from a vecbuffer (before interpolation) for quick debugging.
    """
    try:
        vec_set = lv.read_set(str(head_path))
        vecbuffer = vec_set[frame_idx]
        x_axis, y_axis = extract_axes_from_vecbuffer(vecbuffer)
        arr = vecbuffer[0].as_masked_array()
        u = np.array(np.ma.filled(arr["u"], np.nan), dtype=np.float32)
        v = np.array(np.ma.filled(arr["v"], np.nan), dtype=np.float32)
    except Exception as exc:
        logger.warning("Skipping raw QC for %s: %s", head_path, exc)
        return

    mask = np.isfinite(u) & np.isfinite(v)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")

    X_s = X[::stride, ::stride]
    Y_s = Y[::stride, ::stride]
    u_s = u[::stride, ::stride]
    v_s = v[::stride, ::stride]
    mask_s = mask[::stride, ::stride]

    X_s = X_s[mask_s]
    Y_s = Y_s[mask_s]
    u_s = u_s[mask_s]
    v_s = v_s[mask_s]

    qc_dir = save_dir / "QC"
    qc_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.quiver(
        X_s,
        Y_s,
        u_s,
        v_s,
        angles="xy",
        scale_units="xy",
        scale=50,
        width=0.002,
        pivot="mid",
    )
    plt.xlabel("x (phys units)")
    plt.ylabel("y (phys units)")
    plt.title(f"Raw vector field (head {head_path.name}, frame {frame_idx})")
    plt.axis("equal")
    plt.tight_layout()
    qc_path = qc_dir / f"{base_name}raw_head{head_path.stem}_frame{frame_idx}.png"
    plt.savefig(qc_path, dpi=200)
    plt.close()
    logger.info("Saved raw QC quiver to %s", qc_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Hard-coded paths: adjust to your dataset
    save_dir = Path("E:/sPIV_PLIF_ProcessedData/PIV/")
    save_name = "9.26.2025_30cms_DiffusiveSource_smTG15cm_neuHe0.876_air0.941_PIV0.02_iso_"
    piv_dir = Path("D:/PIV_20Hz_data/")
    piv_path1 = piv_dir / "9.26.2025_PIV_sourceconfig/9.26.2025_30cms_DiffusiveSource_smTG15cm_neuHe0.876_air0.941_PIV0.02_iso/Copy_L1/AddGeometricMask/StereoPIV_MPd(2x12x12_75%ov)/PostProc/AnisoSmooth_S=5_K21/DeleteMask/Resize/PostProc_interpolate.set"
    piv_path2 = piv_dir / "9.26.2025_PIV_sourceconfig/9.26.2025_30cms_DiffusiveSource_smTG15cm_neuHe0.876_air0.941_PIV0.02_iso/Copy_L2/StereoPIV_MPd(2x12x12_75%ov)/Resize.set"
    vec_grid = 3 # spacing of the vectors in pixels

    logger.info("Processing %s and %s onto target grid %s..%s mm", piv_path1, piv_path2, TARGET_MIN_MM, TARGET_MAX_MM)
    save_raw_qc_quiver(piv_path1, save_dir, save_name + 'A', frame_idx=0, stride=10)
    save_raw_qc_quiver(piv_path2, save_dir, save_name + 'B', frame_idx=0, stride=10)
    u_stack, v_stack, w_stack = collate_vectors_to_grid(piv_path1, piv_path2, vec_grid)
    save_stacks(save_dir, save_name, u_stack, v_stack, w_stack)
    save_qc_quiver(u_stack, v_stack, TARGET_X, TARGET_Y, save_dir, save_name, frame_idx=0, stride=10)
    logger.info("Saved stacks to %s", save_dir)
