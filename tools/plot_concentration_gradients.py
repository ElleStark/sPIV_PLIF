"""
Load concentration data, compute spatial/temporal gradients, and plot:
1) time-averaged signed spatial gradients <dc/dx>_t and <dc/dy>_t
2) time-averaged signed products <(dc/dx)*(dc/dt)>_t and <(dc/dy)*(dc/dt)>_t

Run:
    python tools/plot_concentration_gradients.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "nearbed"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
OUT_DIR = BASE_PATH / "Plots" / "concentration_gradients"

X_SLICE = slice(0, 601)
Y_SLICE = slice(100, 500)
T_SLICE = slice(0,6000)  # e.g., slice(0, 2000)
CHUNK_SIZE = 200  # number of frames to process at once when computing means

# Grid/time spacing used in gradient computations
DX = 0.5
DY = 0.5
DT = 0.05
Z_SCORE = True  # Whether to z-score the fields 

XLABEL = "x index"
YLABEL = "y index"
FIG_DPI = 600

# Plot controls
SPATIAL_CMAP = "RdBu_r"
PRODUCT_CMAP = "RdBu_r"
SPATIAL_LOG_SCALE = True  # signed fields can use symmetric log scale
SPATIAL_VMIN = -.01  # None -> symmetric auto range
SPATIAL_VMAX = 0.01  # None -> symmetric auto range
PRODUCT_VMIN = -.01  # None -> symmetric auto range
PRODUCT_VMAX = 0.01  # None -> symmetric auto range
PRODUCT_LOG_SCALE = True  # signed fields can use symmetric log scale
SPATIAL_THRESHOLD = 0.0000001  # Values with |value| < threshold will be masked (not plotted)
PRODUCT_THRESHOLD = 0.0000001  # Values with |value| < threshold will be masked (not plotted)

# QC plot settings
QC_FRAMES = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000]  # Frame index for QC plots
CONC_CMAP = "viridis"  # Colormap for concentration plots
CONC_VMIN = None
CONC_VMAX = None
# -------------------------------------------------------------------


def _open_concentration_stack() -> np.memmap:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")
    conc = np.load(CONCENTRATION_PATH, mmap_mode="r")
    if conc.ndim != 3:
        raise ValueError(f"Expected 3D concentration stack; got shape {conc.shape}")
    return conc


def _compute_gradient_fields(conc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dcdy = np.gradient(conc, DX, axis=0)
    dcdx = np.gradient(conc, DY, axis=1)
    dcdt = np.gradient(conc, DT, axis=2)
    return dcdx, dcdy, dcdt


def _resolve_axis_slice(axis_slice: slice, axis_size: int) -> slice:
    start, stop, step = axis_slice.indices(axis_size)
    if step != 1:
        raise ValueError("Only unit-step slices are supported for concentration gradients.")
    return slice(start, stop)


def _resolve_selected_slices(conc_stack: np.memmap) -> tuple[slice, slice, slice]:
    x_slice = _resolve_axis_slice(X_SLICE, conc_stack.shape[0])
    y_slice = _resolve_axis_slice(Y_SLICE, conc_stack.shape[1])
    t_slice = _resolve_axis_slice(T_SLICE, conc_stack.shape[2])
    if t_slice.stop <= t_slice.start:
        raise ValueError("T_SLICE produced zero frames to process.")
    return x_slice, y_slice, t_slice


def _finalize_mean(sum_accum: np.ndarray, count_accum: np.ndarray) -> np.ndarray:
    mean = sum_accum / np.where(count_accum == 0, np.nan, count_accum)
    return mean.astype(np.float32)


def _compute_HR_correlator(    conc_stack: np.memmap,
    *,
    x_slice: slice,
    y_slice: slice,
    t_slice: slice,
    chunk_size: int,) -> np.ndarray:

    nx = x_slice.stop - x_slice.start
    ny = y_slice.stop - y_slice.start -1  # one less in y due to shift   
    hrCorr_sum = np.zeros((nx, ny), dtype=np.float64)
    hrCorr_count = np.zeros((nx, ny), dtype=np.float64)

    total_frames = t_slice.stop - t_slice.start
    for idx, core_start in enumerate(range(t_slice.start, t_slice.stop, chunk_size), start=1):
        core_stop = min(core_start + chunk_size, t_slice.stop)
        halo_start = max(t_slice.start+1, core_start - 1)
        halo_stop = min(t_slice.stop, core_stop + 1)
        chunk = np.asarray(conc_stack[x_slice, y_slice, halo_start:halo_stop], dtype=np.float32) 
        print(f"Loaded chunk {idx} for HR correlator: t={halo_start}:{halo_stop} (core: {core_start}:{core_stop})")

        # HR correlator computation
        # use second dimension (y, cross-stream direction) for spatial shifts and third dimension (time) for temporal shifts
        shift = 1
        hr_term1 = chunk[:, shift:, shift:] * chunk[:, :-shift, :-shift]  # c(x,y+1,t)*c(x,y,t-1)
        hr_term2 = chunk[:, shift:, :-shift] * chunk[:, :-shift, shift:]  # c(x,y+1,t-1)*c(x,y,t)
        hrCorr_chunk = hr_term1 - hr_term2  # HR correlator: c(x,y+1,t)*c(x,y,t-1) - c(x,y+1,t)*c(x,y+1,t)

        valid_start = core_start 
        valid_stop = valid_start + (core_stop - core_start)
        hrCorr_valid = hrCorr_chunk[:, :, valid_start:valid_stop]
        print(f"Computed HR correlator for chunk {idx}: dims {hrCorr_valid.shape}")
        hrCorr_sum += np.nansum(hrCorr_valid, axis=2)
        hrCorr_count += np.sum(np.isfinite(hrCorr_valid), axis=2)

        processed = core_stop - t_slice.start
        print(
            f"Processed {processed}/{total_frames} frames "
            f"for HR correlator (chunk {idx}, t={core_start}:{core_stop})."
        )   

    return _finalize_mean(hrCorr_sum, hrCorr_count)

def _compute_gradient_means(
    conc_stack: np.memmap,
    *,
    x_slice: slice,
    y_slice: slice,
    t_slice: slice,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = x_slice.stop - x_slice.start
    ny = y_slice.stop - y_slice.start
    dcdx_sum = np.zeros((nx, ny), dtype=np.float64)
    dcdy_sum = np.zeros((nx, ny), dtype=np.float64)
    dcdt_sum = np.zeros((nx, ny), dtype=np.float64)
    dcdx_dcdt_sum = np.zeros((nx, ny), dtype=np.float64)
    dcdy_dcdt_sum = np.zeros((nx, ny), dtype=np.float64)
    dcdx_count = np.zeros((nx, ny), dtype=np.float64)
    dcdy_count = np.zeros((nx, ny), dtype=np.float64)
    dcdt_count = np.zeros((nx, ny), dtype=np.float64)
    dcdx_dcdt_count = np.zeros((nx, ny), dtype=np.float64)
    dcdy_dcdt_count = np.zeros((nx, ny), dtype=np.float64)

    total_frames = t_slice.stop - t_slice.start
    for idx, core_start in enumerate(range(t_slice.start, t_slice.stop, chunk_size), start=1):
        core_stop = min(core_start + chunk_size, t_slice.stop)
        halo_start = max(t_slice.start, core_start - 1)
        halo_stop = min(t_slice.stop, core_stop + 1)
        chunk = np.asarray(conc_stack[x_slice, y_slice, halo_start:halo_stop], dtype=np.float32)
        dcdx_chunk, dcdy_chunk, dcdt_chunk = _compute_gradient_fields(chunk)

        valid_start = core_start - halo_start
        valid_stop = valid_start + (core_stop - core_start)
        dcdx_valid = dcdx_chunk[:, :, valid_start:valid_stop]
        dcdy_valid = dcdy_chunk[:, :, valid_start:valid_stop]
        dcdt_valid = dcdt_chunk[:, :, valid_start:valid_stop]
        dcdx_dcdt_valid = -dcdx_valid * dcdt_valid
        dcdy_dcdt_valid = -dcdy_valid * dcdt_valid

        dcdx_sum += np.nansum(dcdx_valid, axis=2)
        dcdy_sum += np.nansum(dcdy_valid, axis=2)
        dcdt_sum += np.nansum(dcdt_valid, axis=2)
        dcdx_dcdt_sum += np.nansum(dcdx_dcdt_valid, axis=2)
        dcdy_dcdt_sum += np.nansum(dcdy_dcdt_valid, axis=2)

        dcdx_count += np.sum(np.isfinite(dcdx_valid), axis=2)
        dcdy_count += np.sum(np.isfinite(dcdy_valid), axis=2)
        dcdt_count += np.sum(np.isfinite(dcdt_valid), axis=2)
        dcdx_dcdt_count += np.sum(np.isfinite(dcdx_dcdt_valid), axis=2)
        dcdy_dcdt_count += np.sum(np.isfinite(dcdy_dcdt_valid), axis=2)

        processed = core_stop - t_slice.start
        print(
            f"Processed {processed}/{total_frames} frames "
            f"for mean gradients (chunk {idx}, t={core_start}:{core_stop})."
        )

    return (
        _finalize_mean(dcdx_sum, dcdx_count),
        _finalize_mean(dcdy_sum, dcdy_count),
        _finalize_mean(dcdt_sum, dcdt_count),
        _finalize_mean(dcdx_dcdt_sum, dcdx_dcdt_count),
        _finalize_mean(dcdy_dcdt_sum, dcdy_dcdt_count),
    )


def _load_qc_frame_with_gradients(
    conc_stack: np.memmap,
    *,
    frame_idx: int,
    x_slice: slice,
    y_slice: slice,
    t_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abs_frame = t_slice.start + frame_idx
    if abs_frame < t_slice.start or abs_frame >= t_slice.stop:
        raise IndexError(f"QC frame {frame_idx} is out of range for selected frames.")

    halo_start = max(t_slice.start, abs_frame - 1)
    halo_stop = min(t_slice.stop, abs_frame + 2)
    chunk = np.asarray(conc_stack[x_slice, y_slice, halo_start:halo_stop], dtype=np.float32)
    dcdx_chunk, dcdy_chunk, dcdt_chunk = _compute_gradient_fields(chunk)
    local_idx = abs_frame - halo_start
    return (
        chunk[:, :, local_idx],
        dcdx_chunk[:, :, local_idx],
        dcdy_chunk[:, :, local_idx],
        dcdt_chunk[:, :, local_idx],
    )


def _plot_field(
    field_2d: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    fig_path: Path,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
    threshold: float | None = None,
) -> None:
    x_offset = X_SLICE.start or 0
    y_offset = Y_SLICE.start or 0
    x_idx = np.arange(field_2d.shape[0]) + x_offset
    y_idx = np.arange(field_2d.shape[1]) + y_offset

    # Apply threshold masking if specified
    if threshold is not None:
        field_2d = field_2d.copy()
        field_2d[np.abs(field_2d) < threshold] = np.nan

    norm = None
    if log_scale:
        finite_vals = field_2d[np.isfinite(field_2d)]
        if finite_vals.size == 0:
            raise ValueError("Log scale requested, but field has no finite values.")
        abs_max = float(np.nanmax(np.abs(finite_vals)))
        linthresh = max(abs_max * 1e-6, 1e-12)
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(
        y_idx,
        x_idx,
        field_2d,
        shading="auto",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h, ax=ax, label=cbar_label)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)


def _zscore_array(field: np.ndarray) -> np.ndarray:
    mean = np.nanmean(field)
    std = np.nanstd(field)
    if std < 1e-12:
        return np.zeros_like(field)
    return (field - mean) / std


def main() -> None:
    conc_stack = _open_concentration_stack()
    x_slice, y_slice, t_slice = _resolve_selected_slices(conc_stack)
    n_frames = t_slice.stop - t_slice.start

    # QC plots for selected frames without loading the full stack.
    for QC_FRAME in QC_FRAMES:
        if QC_FRAME >= n_frames:
            break
        conc_frame, dcdx_frame, dcdy_frame, dcdt_frame = _load_qc_frame_with_gradients(
            conc_stack,
            frame_idx=QC_FRAME,
            x_slice=x_slice,
            y_slice=y_slice,
            t_slice=t_slice,
        )

        # # Plot concentration frame
        # _plot_field(
        #     conc_frame,
        #     title=f"Concentration frame {QC_FRAME}\ncase={CASE_NAME}",
        #     cbar_label="Concentration",
        #     fig_path=OUT_DIR / f"QC_frames/concentration_frame_{QC_FRAME}_{CASE_NAME}.png",
        #     cmap=CONC_CMAP,
        #     vmin=CONC_VMIN,
        #     vmax=CONC_VMAX,
        #     log_scale=False,
        # )

        # # Plot dc/dx frame
        # _plot_field(
        #     dcdx_frame,
        #     title=f"dc/dx frame {QC_FRAME}\ncase={CASE_NAME}",
        #     cbar_label="dc/dx",
        #     fig_path=OUT_DIR / f"QC_frames/dcdx_frame_{QC_FRAME}_{CASE_NAME}.png",
        #     cmap=SPATIAL_CMAP,
        #     vmin=SPATIAL_VMIN,
        #     vmax=SPATIAL_VMAX,
        #     log_scale=SPATIAL_LOG_SCALE,
        # )

        # # Plot dc/dy frame
        # _plot_field(
        #     dcdy_frame,
        #     title=f"dc/dy frame {QC_FRAME}\ncase={CASE_NAME}",
        #     cbar_label="dc/dy",
        #     fig_path=OUT_DIR / f"QC_frames/dcdy_frame_{QC_FRAME}_{CASE_NAME}.png",
        #     cmap=SPATIAL_CMAP,
        #     vmin=SPATIAL_VMIN,
        #     vmax=SPATIAL_VMAX,
        #     log_scale=SPATIAL_LOG_SCALE,
        # )
        
        # # Plot dc/dt frame
        # _plot_field(
        #     dcdt_frame,
        #     title=f"dc/dt frame {QC_FRAME}\ncase={CASE_NAME}",
        #     cbar_label="dc/dt",
        #     fig_path=OUT_DIR / f"QC_frames/dcdt_frame_{QC_FRAME}_{CASE_NAME}.png",
        #     cmap=SPATIAL_CMAP,
        #     vmin=SPATIAL_VMIN,
        #     vmax=SPATIAL_VMAX,
        #     log_scale=SPATIAL_LOG_SCALE,
        # )

        # # Plot dc/dx*dc/dt frame
        # _plot_field(
        #     -dcdx_frame*dcdt_frame,
        #     title=f"dc/dx*dc/dt frame {QC_FRAME}\ncase={CASE_NAME}",
        #     cbar_label="dc/dx*dc/dt",
        #     fig_path=OUT_DIR / f"QC_frames/dcdx_times_dcdt_frame_{QC_FRAME}_{CASE_NAME}.png",
        #     cmap=SPATIAL_CMAP,
        #     vmin=SPATIAL_VMIN,
        #     vmax=SPATIAL_VMAX,
        #     log_scale=SPATIAL_LOG_SCALE,
        # )

        # Plot Hassentstein Reichardt correlator frame
        # use second dimension (y, cross-stream direction) for spatial shifts and third dimension (time) for temporal shifts
        shift = 1
        chunk = np.asarray(conc_stack[:, :, QC_FRAME:QC_FRAME+2], dtype=np.float32)  # shape (nx, ny, 1)
        hr_term1 = chunk[:, shift:, shift:] * chunk[:, :-shift, :-shift]  # c(x,y+1,t)*c(x,y,t-1)
        hr_term2 = chunk[:, shift:, :-shift] * chunk[:, :-shift, shift:]  # c(x,y+1,t-1)*c(x,y,t)
        hrCorr_frame = np.squeeze(hr_term1 - hr_term2)  # HR correlator: c(x,y+1,t)*c(x,y,t-1) - c(x,y+1,t)*c(x,y+1,t)

        _plot_field(
            hrCorr_frame,
            title=f"HR correlator frame {QC_FRAME}\ncase={CASE_NAME}",
            cbar_label="odor velocity (mm/s)",
            fig_path=OUT_DIR / f"QC_frames/HRcorrelator_frame_{QC_FRAME}_{CASE_NAME}.png",
            cmap=SPATIAL_CMAP,
            vmin=SPATIAL_VMIN,
            vmax=SPATIAL_VMAX,
            log_scale=SPATIAL_LOG_SCALE,
        )

    dcdx_mean, dcdy_mean, dcdt_mean, dcdx_dcdt_mean, dcdy_dcdt_mean = _compute_gradient_means(
        conc_stack,
        x_slice=x_slice,
        y_slice=y_slice,
        t_slice=t_slice,
        chunk_size=CHUNK_SIZE,
    )

    hrCorr_mean = _compute_HR_correlator(
        conc_stack,
        x_slice=x_slice,
        y_slice=y_slice,
        t_slice=t_slice,
        chunk_size=CHUNK_SIZE,
    )

    # if Z_SCORE:
    #     dcdx_mean = _zscore_array(dcdx_mean)
    #     dcdy_mean = _zscore_array(dcdy_mean)
    #     dcdx_dcdt_mean = _zscore_array(dcdx_dcdt_mean)
    #     dcdy_dcdt_mean = _zscore_array(dcdy_dcdt_mean)
        

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"dcdx_mean_{CASE_NAME}.npy", dcdx_mean)
    np.save(OUT_DIR / f"dcdy_mean_{CASE_NAME}.npy", dcdy_mean)
    np.save(OUT_DIR / f"dcdx_times_dcdt_mean_{CASE_NAME}.npy", dcdx_dcdt_mean)
    np.save(OUT_DIR / f"dcdy_times_dcdt_mean_{CASE_NAME}.npy", dcdy_dcdt_mean)
    np.save(OUT_DIR / f"hrCorr_mean_{CASE_NAME}.npy", hrCorr_mean)

    spatial_vmin = SPATIAL_VMIN
    spatial_vmax = SPATIAL_VMAX
    if spatial_vmin is None or spatial_vmax is None:
        merged = np.concatenate([dcdx_mean.ravel(), dcdy_mean.ravel()])
        lim = float(np.nanmax(np.abs(merged))) if np.any(np.isfinite(merged)) else 1.0
        lim = max(lim, 1e-12)
        spatial_vmin = -lim
        spatial_vmax = lim

    # _plot_field(
    #     dcdx_mean,
    #     title=f"Time-averaged signed spatial gradient <dc/dx>_t\ncase={CASE_NAME}",
    #     cbar_label="<dc/dx>_t",
    #     fig_path=OUT_DIR / f"dcdx_mean_{CASE_NAME}.png",
    #     cmap=SPATIAL_CMAP,
    #     vmin=spatial_vmin,
    #     vmax=spatial_vmax,
    #     log_scale=SPATIAL_LOG_SCALE,
    #     threshold=SPATIAL_THRESHOLD,
    # )

    # _plot_field(
    #     dcdy_mean,
    #     title=f"Time-averaged signed spatial gradient <dc/dy>_t\ncase={CASE_NAME}",
    #     cbar_label="<dc/dy>_t",
    #     fig_path=OUT_DIR / f"dcdy_mean_{CASE_NAME}.png",
    #     cmap=SPATIAL_CMAP,
    #     vmin=spatial_vmin,
    #     vmax=spatial_vmax,
    #     log_scale=SPATIAL_LOG_SCALE,
    #     threshold=SPATIAL_THRESHOLD,
    # )

    # _plot_field(
    #     dcdt_mean,
    #     title=f"Time-averaged signed spatial gradient <dc/dt>_t\ncase={CASE_NAME}",
    #     cbar_label="<dc/dt>_t",
    #     fig_path=OUT_DIR / f"dcdt_mean_{CASE_NAME}.png",
    #     cmap=SPATIAL_CMAP,
    #     vmin=spatial_vmin,
    #     vmax=spatial_vmax,
    #     log_scale=SPATIAL_LOG_SCALE,
    #     threshold=SPATIAL_THRESHOLD,
    # )


    # prod_vmin = PRODUCT_VMIN
    # prod_vmax = PRODUCT_VMAX
    # if prod_vmin is None or prod_vmax is None:
    #     merged = np.concatenate([dcdx_dcdt_mean.ravel(), dcdy_dcdt_mean.ravel()])
    #     lim = float(np.nanmax(np.abs(merged))) if np.any(np.isfinite(merged)) else 1.0
    #     lim = max(lim, 1e-12)
    #     prod_vmin = -lim
    #     prod_vmax = lim

    # _plot_field(
    #     dcdx_dcdt_mean,
    #     title=f"Time-averaged signed product <(dc/dx)*(dc/dt)>_t\ncase={CASE_NAME}",
    #     cbar_label="<(dc/dx)*(dc/dt)>_t",
    #     fig_path=OUT_DIR / f"dcdx_times_dcdt_mean_{CASE_NAME}_t{T_SLICE.start}to{T_SLICE.stop}.png",
    #     cmap=PRODUCT_CMAP,
    #     vmin=prod_vmin,
    #     vmax=prod_vmax,
    #     log_scale=PRODUCT_LOG_SCALE,
    # )

    # _plot_field(
    #     dcdy_dcdt_mean,
    #     title=f"Time-averaged signed product <(dc/dy)*(dc/dt)>_t\ncase={CASE_NAME}",
    #     cbar_label="<(dc/dy)*(dc/dt)>_t",
    #     fig_path=OUT_DIR / f"dcdy_times_dcdt_mean_{CASE_NAME}_t{T_SLICE.start}to{T_SLICE.stop}.png",
    #     cmap=PRODUCT_CMAP,
    #     vmin=prod_vmin,
    #     vmax=prod_vmax,
    #     log_scale=PRODUCT_LOG_SCALE,
    # )

    _plot_field(
        hrCorr_mean,
        title=f"Time-averaged HR correlator\ncase={CASE_NAME}",
        cbar_label="HR correlator (mm^2/s)",
        fig_path=OUT_DIR / f"hrCorr_mean_{CASE_NAME}_t{T_SLICE.start}to{T_SLICE.stop}.png",
        cmap=SPATIAL_CMAP,
        vmin=SPATIAL_VMIN,
        vmax=SPATIAL_VMAX,
        log_scale=SPATIAL_LOG_SCALE,
        threshold=SPATIAL_THRESHOLD,
    )

    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
