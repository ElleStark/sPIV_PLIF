"""Visualization helpers.

Plotting utilities include velocity quiver plots, concentration plots, .
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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


def overlay_quiver_figure(
    u: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
    *,
    frame_idx: Optional[int],
    stride_rows: int = 20,
    stride_cols: int = 15,
    scale: float = 0.04,
    headwidth: float = 5.5,
    headlength: float = 6.0,
    headaxislength: float = 4.0,
    tailwidth: float = 0.002,
    cmin: Optional[float] = 0.01,
    cmax: Optional[float] = 1.0,
    x_coords: Optional[Sequence[float]] = None,
    y_coords: Optional[Sequence[float]] = None,
    log_scale: bool = True,
    title: Optional[str] = None,
    cmap_name: str = "cmr.rainforest_r",
    cmap_slice: Tuple[float, float] = (0.0, 0.65),
    figsize: Tuple[float, float] = (8.0, 6.0),
    arrow_color: str = "k",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create an overlay figure of concentration as a colormap with velocity quiver.

    Parameters mirror the ad-hoc tool in tools/draw_figs.py so scripts can reuse
    the plotting logic without duplicating it.
    """

    def _extract_frame(arr: np.ndarray, name: str) -> np.ndarray:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if frame_idx is None:
                raise ValueError(f"{name} is 3D; provide frame_idx to select a slice.")
            if frame_idx < 0 or frame_idx >= arr.shape[2]:
                raise IndexError(f"frame_idx {frame_idx} out of range for {name} with {arr.shape[2]} frames")
            return arr[:, :, frame_idx]
        raise ValueError(f"{name} must be 2D or 3D; got shape {arr.shape}")

    u_f = _extract_frame(u, "u")
    v_f = _extract_frame(v, "v")
    c_f = _extract_frame(c, "c")

    if u_f.shape != v_f.shape:
        raise ValueError(f"u and v shapes differ after slicing: {u_f.shape} vs {v_f.shape}")
    if c_f.shape != u_f.shape:
        raise ValueError(f"c shape {c_f.shape} must match velocity shape {u_f.shape}")

    ny, nx = u_f.shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)

    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x_coords/y_coords length must match the data grid dimensions.")
    if np.ndim(x_coords) != 1 or np.ndim(y_coords) != 1:
        raise ValueError("x_coords and y_coords must be 1D sequences.")

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    # Mask invalid values before plotting.
    mask_vec = np.isfinite(u_f) & np.isfinite(v_f)
    mask_c = np.isfinite(c_f)
    if log_scale:
        mask_c &= c_f > 0  # LogNorm requires positive values

    cmap = cmr.get_sub_cmap(cmap_name, *cmap_slice)

    norm = None
    if log_scale:
        if not np.any(mask_c):
            raise ValueError("No positive concentration values available for log scale.")
        data_pos = c_f[mask_c]
        vmin = cmin if cmin is not None and cmin > 0 else float(np.nanmin(data_pos))
        vmax = cmax if cmax is not None else float(np.nanmax(data_pos))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin_kw = None
        vmax_kw = None
    else:
        vmin_kw = cmin
        vmax_kw = cmax

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.pcolormesh(
        X,
        Y,
        np.ma.array(c_f, mask=~mask_c),
        cmap=cmap,
        shading="auto",
        vmin=vmin_kw,
        vmax=vmax_kw,
        norm=norm,
    )
    fig.colorbar(h, ax=ax, label="c")

    Xs = X[::stride_rows, ::stride_cols]
    Ys = Y[::stride_rows, ::stride_cols]
    us = u_f[::stride_rows, ::stride_cols]
    vs = v_f[::stride_rows, ::stride_cols]
    ms = mask_vec[::stride_rows, ::stride_cols]

    Xs = Xs[ms]
    Ys = Ys[ms]
    us = us[ms]
    vs = vs[ms]

    ax.quiver(
        Xs,
        Ys,
        us,
        vs,
        angles="xy",
        scale_units="xy",
        scale=scale,
        headwidth=headwidth,
        headlength=headlength,
        width=tailwidth,
        headaxislength=headaxislength,
        pivot="mid",
        color=arrow_color,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, ax


def save_overlay_quiver(
    u: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
    out_path: Path,
    **kwargs: Any,
) -> None:
    """Create an overlay figure and save to disk."""
    fig, _ = overlay_quiver_figure(u, v, c, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info("Saved overlay to %s", out_path)
