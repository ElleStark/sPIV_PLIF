"""Visualization helpers.

Plotting utilities include velocity quiver plots, concentration plots, .
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("sPIV_PLIF.viz")


def save_animation(anim: Any, out_path: str, fps: int = 10, dpi: int = 150) -> None:
    try:
        anim.save(out_path, fps=fps, dpi=dpi)
        logger.info("Saved animation to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save animation: %s", exc)


def quiver_from_npy(
    u_path: Path,
    v_path: Path,
    out_path: Path,
    frame_idx: int = 0,
    stride: int = 10,
    stride_x: Optional[int] = None,
    stride_y: Optional[int] = None,
    scale: float = 50.0,
    headwidth: float = 3.0,
    headlength: float = 5.0,
    tailwidth: float = 0.002,
    x_coords: Optional[Sequence[float]] = None,
    y_coords: Optional[Sequence[float]] = None,
) -> None:
    """
    Load u/v .npy stacks and save a quiver plot for the requested frame.

    If x_coords/y_coords are not provided, pixel indices are used.
    """
    u = np.load(u_path)
    v = np.load(v_path)

    if u.shape != v.shape:
        raise ValueError(f"u and v shapes differ: {u.shape} vs {v.shape}")
    if u.ndim != 3:
        raise ValueError(f"Expected 3D stacks (y, x, frames); got shape {u.shape}")
    if frame_idx < 0 or frame_idx >= u.shape[2]:
        raise IndexError(f"frame_idx {frame_idx} out of range for {u.shape[2]} frames")

    u_f = u[:, :, frame_idx]
    v_f = v[:, :, frame_idx]

    ny, nx = u_f.shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x_coords/y_coords length must match the data grid dimensions.")

    mask = np.isfinite(u_f) & np.isfinite(v_f)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    sx = stride_x if stride_x is not None else stride
    sy = stride_y if stride_y is not None else stride

    X_s = X[::sy, ::sx]
    Y_s = Y[::sy, ::sx]
    u_s = u_f[::sy, ::sx]
    v_s = v_f[::sy, ::sx]
    mask_s = mask[::sy, ::sx]

    X_s = X_s[mask_s]
    Y_s = Y_s[mask_s]
    u_s = u_s[mask_s]
    v_s = v_s[mask_s]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.quiver(
        X_s,
        Y_s,
        u_s,
        v_s,
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=tailwidth,
        pivot="mid",
        headwidth=headwidth,
        headlength=headlength,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Quiver frame {frame_idx}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Saved quiver QC to %s", out_path)
