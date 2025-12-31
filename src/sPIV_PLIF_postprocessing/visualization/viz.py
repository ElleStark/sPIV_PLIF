"""Visualization helpers.

Plotting utilities include velocity quiver plots, concentration plots, .
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
from itertools import cycle

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap

from .gaussian_fit import fit_gaussian_least_squares

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
    cmap_under: Optional[str] = None,
    cmap_under_transition: Optional[float] = None,
    cmap_under_start: Optional[float] = None,
    cmap_under_end: Optional[float] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    pcolormesh_alpha: float = 1.0,
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

    cmap = cmr.get_sub_cmap(cmap_name, *cmap_slice)
    if cmap_under is not None:
        try:
            cmap = cmap.copy()
            cmap.set_under(cmap_under)
        except Exception:
            pass
    if cmap_under_transition is not None and cmap_under_transition > 0:
        transition_fraction = cmap_under_transition
    elif cmap_under_start is not None and cmap_under_end is not None and norm is not None:
        transition_fraction = float(norm(cmap_under_end) - norm(cmap_under_start))
    else:
        transition_fraction = None
    if transition_fraction is not None and transition_fraction > 0:
        try:
            colors = cmap(np.linspace(0, 1, 256))
            n_under = max(2, int(len(colors) * min(transition_fraction, 0.5)))
            white = np.array([1.0, 1.0, 1.0, 1.0])
            first_color = colors[0]
            under_grad = np.stack(
                [
                    white * (1 - t) + first_color * t
                    for t in np.linspace(0, 1, n_under, endpoint=True)
                ],
                axis=0,
            )
            colors = np.vstack([under_grad, colors])
            cmap = ListedColormap(colors)
        except Exception:
            pass

    h = ax.pcolormesh(
        X,
        Y,
        np.ma.array(c_f, mask=~mask_c),
        cmap=cmap,
        shading="auto",
        vmin=vmin_kw,
        vmax=vmax_kw,
        norm=norm,
        alpha=pcolormesh_alpha,
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


def overlay_contour_figure(
    u: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
    *,
    frame_idx: Optional[int],
    contour_levels: Optional[int | Sequence[float]] = 10,
    contour_color: str = "k",
    contour_width: float = 1.0,
    contour_cmap: Optional[str] = None,
    contour_labels: bool = True,
    contour_box: Optional[Tuple[float, float, float, float]] = None,
    contour_levels_in_box: Optional[int | Sequence[float]] = None,
    contour_color_in_box: Optional[str] = None,
    contour_width_in_box: Optional[float] = None,
    contour_cmap_in_box: Optional[str] = None,
    contour_labels_in_box: Optional[bool] = None,
    show_quiver: bool = False,
    stride_rows: int = 20,
    stride_cols: int = 15,
    quiver_scale: float = 0.04,
    quiver_headwidth: float = 5.5,
    quiver_headlength: float = 6.0,
    quiver_headaxislength: float = 4.0,
    quiver_tailwidth: float = 0.002,
    quiver_color: str = "#c0c0c0",
    quiver_cmap: Optional[str] = None,
    quiver_vmin: Optional[float] = None,
    quiver_vmax: Optional[float] = None,
    quiver_colorbar: bool = False,
    quiver_alpha: float = 0.8,
    cmin: Optional[float] = 0.01,
    cmax: Optional[float] = 1.0,
    x_coords: Optional[Sequence[float]] = None,
    y_coords: Optional[Sequence[float]] = None,
    log_scale: bool = True,
    title: Optional[str] = None,
    cmap_name: str = "cmr.rainforest_r",
    cmap_slice: Tuple[float, float] = (0.0, 0.65),
    cmap_under: Optional[str] = None,
    cmap_under_transition: Optional[float] = None,
    cmap_under_start: Optional[float] = None,
    cmap_under_end: Optional[float] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    pcolormesh_alpha: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay concentration colormap with velocity-magnitude contours."""

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

    mask_vec = np.isfinite(u_f) & np.isfinite(v_f)
    mask_c = np.isfinite(c_f)
    if log_scale:
        mask_c &= c_f > 0

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
    cmap = cmr.get_sub_cmap(cmap_name, *cmap_slice)
    if cmap_under is not None:
        try:
            cmap = cmap.copy()
            cmap.set_under(cmap_under)
        except Exception:
            pass
    if cmap_under_transition is not None and cmap_under_transition > 0:
        transition_fraction = cmap_under_transition
    elif cmap_under_start is not None and cmap_under_end is not None and norm is not None:
        transition_fraction = float(norm(cmap_under_end) - norm(cmap_under_start))
    else:
        transition_fraction = None
    if transition_fraction is not None and transition_fraction > 0:
        try:
            colors = cmap(np.linspace(0, 1, 256))
            n_under = max(2, int(len(colors) * min(transition_fraction, 0.5)))
            white = np.array([1.0, 1.0, 1.0, 1.0])
            first_color = colors[0]
            under_grad = np.stack(
                [
                    white * (1 - t) + first_color * t
                    for t in np.linspace(0, 1, n_under, endpoint=True)
                ],
                axis=0,
            )
            colors = np.vstack([under_grad, colors])
            cmap = ListedColormap(colors)
        except Exception:
            pass

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
        alpha=pcolormesh_alpha,
    )
    fig.colorbar(h, ax=ax, label="c")

    speed = np.sqrt(u_f ** 2 + v_f ** 2)
    if contour_levels is not None:
        # Main contours (outside box or whole domain if no box provided).
        mask_out = mask_vec.copy()
        mask_in = None
        if contour_box is not None:
            xmin, xmax, ymin, ymax = contour_box
            in_box = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
            mask_in = mask_vec & in_box
            mask_out &= ~in_box

        contour_kwargs = {
            "levels": contour_levels,
            "linewidths": contour_width,
        }
        if contour_cmap:
            contour_kwargs["cmap"] = contour_cmap
        else:
            contour_kwargs["colors"] = contour_color

        cs = ax.contour(
            X,
            Y,
            np.ma.array(speed, mask=~mask_out),
            **contour_kwargs,
        )
        if contour_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        # Optional lower-density contours inside the box.
        if mask_in is not None and np.any(mask_in):
            contour_kwargs_in = {
                "levels": contour_levels_in_box or contour_levels,
                "linewidths": contour_width_in_box or contour_width,
            }
            cmap_in = contour_cmap_in_box if contour_cmap_in_box is not None else contour_cmap
            color_in = contour_color_in_box if contour_color_in_box is not None else contour_color
            if cmap_in:
                contour_kwargs_in["cmap"] = cmap_in
            else:
                contour_kwargs_in["colors"] = color_in

            cs_in = ax.contour(
                X,
                Y,
                np.ma.array(speed, mask=~mask_in),
                **contour_kwargs_in,
            )
            if contour_labels_in_box if contour_labels_in_box is not None else contour_labels:
                ax.clabel(cs_in, inline=True, fontsize=8, fmt="%.2f")

    if show_quiver:
        Xs = X[::stride_rows, ::stride_cols]
        Ys = Y[::stride_rows, ::stride_cols]
        us = u_f[::stride_rows, ::stride_cols]
        vs = v_f[::stride_rows, ::stride_cols]
        ms = mask_vec[::stride_rows, ::stride_cols]

        Xs = Xs[ms]
        Ys = Ys[ms]
        us = us[ms]
        vs = vs[ms]
        speed_s = np.sqrt(us**2 + vs**2)
        norm_q = None
        if quiver_cmap is not None:
            vmin_q = quiver_vmin if quiver_vmin is not None else float(np.nanmin(speed_s))
            vmax_q = quiver_vmax if quiver_vmax is not None else float(np.nanmax(speed_s))
            norm_q = Normalize(vmin=vmin_q, vmax=vmax_q)
        if quiver_cmap is not None:
            q = ax.quiver(
                Xs,
                Ys,
                us,
                vs,
                speed_s,
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                headwidth=quiver_headwidth,
                headlength=quiver_headlength,
                width=quiver_tailwidth,
                headaxislength=quiver_headaxislength,
                pivot="mid",
                color=None,
                cmap=quiver_cmap,
                norm=norm_q,
                alpha=quiver_alpha,
            )
        else:
            q = ax.quiver(
                Xs,
                Ys,
                us,
                vs,
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                headwidth=quiver_headwidth,
                headlength=quiver_headlength,
                width=quiver_tailwidth,
                headaxislength=quiver_headaxislength,
                pivot="mid",
                color=quiver_color,
                alpha=quiver_alpha,
            )
        if quiver_colorbar and quiver_cmap is not None:
            fig.colorbar(q, ax=ax, label="|u|")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, ax


def save_overlay_contour(
    u: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
    out_path: Path,
    **kwargs: Any,
) -> None:
    """Create a contour overlay figure and save to disk."""
    fig, _ = overlay_contour_figure(u, v, c, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info("Saved contour overlay to %s", out_path)


def _nearest_index(values: Sequence[float], target: float) -> tuple[int, float]:
    """Return index of the closest value to target and the matched value."""
    arr = np.asarray(values)
    idx = int(np.argmin(np.abs(arr - target)))
    return idx, float(arr[idx])


def compute_gaussian_params_at_y(
    cases: Sequence[tuple[str, np.ndarray]],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    target_y: float,
    *,
    normalize_to_max: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    rows_to_average: int = 2,
    fit_x_range: Optional[Tuple[float, float]] = None,
) -> tuple[list[tuple[str, float, float]], float]:
    """
    Compute Gaussian fit parameters (mu, sigma) for each case at a given y-location.

    Returns a list of (label, mu, sigma) tuples and the nearest y value actually used.
    """
    if len(cases) == 0:
        raise ValueError("No cases provided for Gaussian parameter extraction.")

    y_idx, y_match = _nearest_index(y_coords, target_y)
    row_start = max(0, y_idx - rows_to_average)
    row_end = min(len(y_coords), y_idx + rows_to_average + 1)
    if row_end <= row_start:
        raise ValueError(f"Invalid averaging window for y index {y_idx}")

    x_coords_arr = np.asarray(x_coords)
    x_range_mask: Optional[np.ndarray] = None
    if xlim is not None:
        x_range_mask = (x_coords_arr >= xlim[0]) & (x_coords_arr <= xlim[1])
        if not np.any(x_range_mask):
            raise ValueError(f"No x points fall within requested xlim {xlim}.")

    results: list[tuple[str, float, float]] = []
    for label, c_mean in cases:
        if c_mean.ndim != 2:
            raise ValueError(f"{label} mean concentration must be 2D (y, x); got shape {c_mean.shape}")
        if c_mean.shape != (len(y_coords), len(x_coords)):
            raise ValueError(
                f"{label} mean concentration shape {c_mean.shape} does not match coordinate grid "
                f"({len(y_coords)}, {len(x_coords)})."
            )
        window = np.array(c_mean[row_start:row_end, :], copy=False)
        if window.size == 0:
            raise ValueError(f"No data in averaging window for {label}")

        profile = np.nanmean(window, axis=0)
        mask = np.isfinite(profile)
        if x_range_mask is not None:
            mask &= x_range_mask
        if not np.any(mask):
            raise ValueError(f"No finite data for case '{label}' within the requested x-range.")
        y_vals_raw = profile[mask]
        if normalize_to_max:
            case_min = float(np.nanmin(y_vals_raw))
            case_max = float(np.nanmax(y_vals_raw))
            case_range = case_max - case_min
            if not np.isfinite(case_min) or not np.isfinite(case_max):
                raise ValueError(f"Non-finite values encountered for case '{label}' during normalization.")
            if case_range == 0.0:
                raise ValueError(f"Zero range for case '{label}'; cannot normalize to [0, 1].")
            y_vals = np.clip((y_vals_raw - case_min) / case_range, 0.0, 1.0)
        else:
            y_vals = y_vals_raw

        x_subset = x_coords_arr[mask]
        fit_result = fit_gaussian_least_squares(x_subset, y_vals, fit_x_range=fit_x_range)
        if fit_result is None:
            logger.warning("Gaussian fit failed for %s at y=%.3f (nearest %.3f)", label, target_y, y_match)
            results.append((label, float("nan"), float("nan")))
            continue

        _, _, (_, mu_fit, sigma_fit) = fit_result
        logger.info("Gaussian params for %s at y=%.3f: mu=%.4f, sigma=%.4f", label, y_match, mu_fit, sigma_fit)
        results.append((label, mu_fit, sigma_fit))

    return results, y_match


def plot_gaussian_param_scatter(
    gaussian_results: Sequence[tuple[float, float, Sequence[tuple[str, float, float]]]],
    *,
    param: str,
    out_path: Path,
    title: Optional[str] = None,
    xlabel: str = "Case",
    ylabel: Optional[str] = None,
    markers: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    dpi: int = 300,
    marker_size: float = 70.0,
) -> None:
    """
    Scatter plot of Gaussian parameters across cases with marker variations by y-location.

    gaussian_results should be a sequence of (target_y, nearest_y, results) where results is a
    sequence of (label, mu, sigma) tuples.
    """
    if param not in {"mu", "sigma"}:
        raise ValueError("param must be 'mu' or 'sigma'.")
    if len(gaussian_results) == 0:
        raise ValueError("No Gaussian results provided for plotting.")

    case_labels: list[str] = []
    for _, _, entries in gaussian_results:
        for label, _, _ in entries:
            if label not in case_labels:
                case_labels.append(label)
    if len(case_labels) == 0:
        raise ValueError("No case labels found in Gaussian results.")

    num_cases = len(case_labels)
    cmap = cmr.get_sub_cmap("cmr.neutral", 0.0, 0.8)
    colors = [cmap(v) for v in np.linspace(0, 1, num_cases)] if num_cases > 1 else [cmap(0.0)]
    color_map = {label: colors[idx] for idx, label in enumerate(case_labels)}
    marker_cycle = cycle(markers if markers is not None else ["o", "s", "^", "D", "P", "X", "v", "*"])

    case_index = {label: idx for idx, label in enumerate(case_labels)}

    plt.figure(figsize=figsize)
    ax = plt.gca()

    y_handles = []
    for target_y, nearest_y, entries in gaussian_results:
        marker = next(marker_cycle)
        y_label = f"y={target_y:g} mm (nearest {nearest_y:g})" if abs(target_y - nearest_y) > 1e-6 else f"y={target_y:g} mm"
        y_handles.append(
            plt.Line2D([], [], color="k", marker=marker, linestyle="None", markersize=8, label=y_label)
        )
        for label, mu_val, sigma_val in entries:
            val = mu_val if param == "mu" else sigma_val
            if not np.isfinite(val):
                continue
            ax.scatter(
                case_index[label],
                val,
                color=color_map[label],
                marker=marker,
                s=marker_size,
                edgecolors="k",
                linewidths=0.5,
                label=None,
            )

    ax.set_xticks(list(case_index.values()))
    ax.set_xticklabels(case_labels, rotation=20)
    ax.set_xlabel(xlabel)
    param_label = "mu" if param == "mu" else "sigma"
    ax.set_ylabel(ylabel if ylabel is not None else f"Gaussian {param_label}")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    case_handles = [
        plt.Line2D([], [], color=color_map[label], marker="o", linestyle="None", markersize=8, label=label)
        for label in case_labels
    ]
    legend1 = ax.legend(handles=case_handles, title="Cases", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    legend2 = ax.legend(handles=y_handles, title="y locations", loc="upper left", bbox_to_anchor=(1.02, 0.45))
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved Gaussian %s scatter to %s", param, out_path)


def plot_lateral_profiles(
    cases: Sequence[tuple[str, np.ndarray]],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    target_y: float,
    *,
    out_path: Path,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "Mean concentration",
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 300,
    legend: bool = True,
    grid: bool = True,
    normalize_to_max: bool = False,
    line_color: Optional[str] = None,
    linestyles: Optional[Sequence[str]] = None,
    line_width: float = 0.5,
    xlim: Optional[Tuple[float, float]] = None,
    set_ylim_to_data_max: bool = False,
    rows_to_average: int = 2,
    ylim: Optional[Tuple[float, float]] = None,
    fit_x_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot lateral (x-direction) mean concentration profiles for multiple cases at a given y.

    Parameters
    ----------
    cases
        Sequence of (label, c_mean_array) pairs. Arrays must be 2D (y, x).
    x_coords, y_coords
        Coordinate arrays matching the dimensions of the mean fields.
    target_y
        Downstream distance (same units as y_coords) at which to extract the profile.
    rows_to_average
        Number of rows to include on each side of the target index for averaging.
    fit_x_range
        Optional (xmin, xmax) bounds in which to perform Gaussian fitting; defaults to plotted range.
        When a fit succeeds, x is re-parameterized as x/σ for both data and Gaussian overlay.
    """
    if len(cases) == 0:
        raise ValueError("No cases provided for plotting.")

    y_idx, y_match = _nearest_index(y_coords, target_y)
    row_start = max(0, y_idx - rows_to_average)
    row_end = min(len(y_coords), y_idx + rows_to_average + 1)
    if row_end <= row_start:
        raise ValueError(f"Invalid averaging window for y index {y_idx}")
    x_coords_arr = np.asarray(x_coords)
    x_range_mask: Optional[np.ndarray] = None
    if xlim is not None:
        x_range_mask = (x_coords_arr >= xlim[0]) & (x_coords_arr <= xlim[1])
        if not np.any(x_range_mask):
            raise ValueError(f"No x points fall within requested xlim {xlim}.")

    profiles: list[tuple[str, np.ndarray]] = []
    for label, c_mean in cases:
        if c_mean.ndim != 2:
            raise ValueError(f"{label} mean concentration must be 2D (y, x); got shape {c_mean.shape}")
        if c_mean.shape != (len(y_coords), len(x_coords)):
            raise ValueError(
                f"{label} mean concentration shape {c_mean.shape} does not match coordinate grid "
                f"({len(y_coords)}, {len(x_coords)})."
            )
        window = np.array(c_mean[row_start:row_end, :], copy=False)
        if window.size == 0:
            raise ValueError(f"No data in averaging window for {label}")
        profiles.append((label, np.nanmean(window, axis=0)))

    num_profiles = len(profiles)
    style_cycle = cycle(linestyles if linestyles is not None else ["solid"])
    color_iter = None
    if line_color is None:
        cmap = cmr.get_sub_cmap("cmr.neutral", 0.0, 0.8)
        color_values = [cmap(v) for v in np.linspace(0, 1, num_profiles)] if num_profiles > 1 else [cmap(0.0)]
        color_iter = iter(color_values)
    plt.figure(figsize=figsize)
    max_y_val = 0.0
    plot_entries: list[tuple[str, np.ndarray, np.ndarray, Optional[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]]] = []
    for label, profile in profiles:
        mask = np.isfinite(profile)
        effective_mask = mask if x_range_mask is None else (mask & x_range_mask)
        if not np.any(effective_mask):
            raise ValueError(f"No finite data for case '{label}' within the plotting x-range.")
        y_vals_raw = profile[effective_mask]
        y_vals: np.ndarray
        if normalize_to_max:
            case_min = float(np.nanmin(y_vals_raw))
            case_max = float(np.nanmax(y_vals_raw))
            case_range = case_max - case_min
            if not np.isfinite(case_min) or not np.isfinite(case_max):
                raise ValueError(f"Non-finite values encountered for case '{label}' during normalization.")
            if case_range == 0.0:
                raise ValueError(f"Zero range for case '{label}'; cannot normalize to [0, 1].")
            y_vals = np.clip((y_vals_raw - case_min) / case_range, 0.0, 1.0)
        else:
            y_vals = y_vals_raw
        if y_vals.size:
            max_y_val = max(max_y_val, float(np.nanmax(y_vals)))
        x_subset = x_coords_arr[effective_mask]
        fit_result = fit_gaussian_least_squares(x_subset, y_vals, fit_x_range=fit_x_range)
        if fit_result is not None:
            _, _, (_, mu_fit, sigma_fit) = fit_result
            logger.info("Gaussian fit for %s: mu=%.4f, sigma=%.4f", label, mu_fit, sigma_fit)
        plot_entries.append((label, y_vals, x_subset, fit_result))

    all_fits_succeeded = all(entry[3] is not None for entry in plot_entries)
    used_normalized_axis = all_fits_succeeded

    avg_gaussians: list[np.ndarray] = []
    avg_gaussian_x = None

    for label, y_vals, x_subset, fit_result in plot_entries:
        if used_normalized_axis and fit_result is not None:
            x_fit, gaussian_curve, (_, mu_fit, sigma_fit) = fit_result
            x_plot = (x_subset - mu_fit) / sigma_fit
            x_fit_plot = (x_fit - mu_fit) / sigma_fit
        else:
            x_plot = x_subset
            x_fit_plot = None if fit_result is None else fit_result[0]
            gaussian_curve = None if fit_result is None else fit_result[1]

        color_to_use = line_color if line_color is not None else next(color_iter)
        plt.plot(
            x_plot,
            y_vals,
            label=label,
            color=color_to_use,
            linestyle=next(style_cycle),
            linewidth=line_width,
        )

        if fit_result is not None and used_normalized_axis:
            # Accumulate normalized Gaussians for averaging on a common grid.
            x_fit, gaussian_curve, (_, mu_fit, sigma_fit) = fit_result
            z_fit = (x_fit - mu_fit) / sigma_fit
            order = np.argsort(z_fit)
            z_sorted = z_fit[order]
            g_sorted = gaussian_curve[order]
            peak = float(np.nanmax(g_sorted))
            if peak > 0 and np.isfinite(peak):
                z_common = np.linspace(-3.0, 3.0, 200) if avg_gaussian_x is None else avg_gaussian_x
                if avg_gaussian_x is None:
                    avg_gaussian_x = z_common
                g_interp = np.interp(z_common, z_sorted, g_sorted / peak, left=np.nan, right=np.nan)
                avg_gaussians.append(np.array(g_interp, dtype=float))

    if used_normalized_axis:
        plt.xlabel("(x-μ)/σ")
    else:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel + (" (normalized to [0,1] per case)" if normalize_to_max else ""))
    if used_normalized_axis:
        plt.xlim(-3.0, 3.0)
    elif xlim is not None:
        plt.xlim(xlim)
    if used_normalized_axis and avg_gaussians and avg_gaussian_x is not None:
        avg_gaussian = np.nanmean(np.vstack(avg_gaussians), axis=0)
        plt.plot(
            avg_gaussian_x,
            avg_gaussian,
            color="#D21502",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label="avg Gaussian",
        )
    if ylim is not None:
        plt.ylim(ylim)
    elif set_ylim_to_data_max and max_y_val > 0:
        plt.ylim(0, max_y_val)
    plot_title = title if title is not None else f"Mean concentration at y ≈ {y_match:.2f}"
    plt.title(plot_title)
    if grid:
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    if legend:
        plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    logger.info("Saved lateral profile plot at y=%.3f to %s (nearest y=%.3f, idx=%d)", target_y, out_path, y_match, y_idx)
