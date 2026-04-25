"""
Load concentration data, compute the Hassenstein-Reichardt odor-motion correlator,
and plot the time-averaged odor-motion field.

Run:
    python tools/plot_odor_motion.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "smoke_plume"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
# CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
CONCENTRATION_PATH = BASE_PATH / "Emonet_smoke" / "HalfmmGrid_new_smoke_2a.npy"  
OUT_DIR = BASE_PATH / "Plots" / "odor_motion"

X_SLICE = slice(0, 332)
Y_SLICE = slice(0, 528)
T_SLICE = slice(0, 200)
CHUNK_SIZE = 100
PIXELS_PER_CM = 20
C_MIN = 0.001

MM_PER_PX = 0.5
FRAME_PER_SEC = 180 

SHIFT_X = 1
SHIFT_Y = 1
SHIFT_T = 1

XLABEL = "x index"
YLABEL = "y index"
FIG_DPI = 600

FIELD_CMAP = "RdBu_r"
FIELD_LOG_SCALE = True
FIELD_VMIN = -0.01
FIELD_VMAX = 0.01
FIELD_THRESHOLD = 0.00001

VECTOR_CMAP = "Reds"
QUIVER_STRIDE = 1
QUIVER_MIN_LENGTH = 0.5
QUIVER_MAX_LENGTH = 1

QC_FRAMES = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000]
# -------------------------------------------------------------------


def _open_concentration_stack() -> np.memmap:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")
    conc = np.load(CONCENTRATION_PATH, mmap_mode="r")
    print(f"Opened concentration stack, shape: {conc.shape}, dtype: {conc.dtype}, min: {np.nanmin(conc)}, max: {np.nanmax(conc)}")
    if conc.ndim != 3:
        raise ValueError(f"Expected 3D concentration stack; got shape {conc.shape}")
    return conc


# def _resolve_axis_slice(axis_slice: slice, axis_size: int) -> slice:
#     start, stop, step = axis_slice.indices(axis_size)
#     if step != 1:
#         raise ValueError("Only unit-step slices are supported for odor-motion plots.")
#     return slice(start, stop)


# def _resolve_selected_slices(conc_stack: np.memmap) -> tuple[slice, slice, slice]:
#     x_slice = _resolve_axis_slice(X_SLICE, conc_stack.shape[0])
#     y_slice = _resolve_axis_slice(Y_SLICE, conc_stack.shape[1])
#     t_slice = _resolve_axis_slice(T_SLICE, conc_stack.shape[2])
#     if t_slice.stop <= t_slice.start:
#         raise ValueError("T_SLICE produced zero frames to process.")
#     if x_slice.stop - x_slice.start <= SHIFT_X:
#         raise ValueError("X_SLICE is too small for the configured SHIFT_X.")
#     if y_slice.stop - y_slice.start <= SHIFT_Y:
#         raise ValueError("Y_SLICE is too small for the configured SHIFT_Y.")
#     if t_slice.stop - t_slice.start <= SHIFT_T:
#         raise ValueError("T_SLICE is too small for the configured SHIFT_T.")
#     return x_slice, y_slice, t_slice


def _finalize_mean(sum_accum: np.ndarray, count_accum: np.ndarray) -> np.ndarray:
    mean = sum_accum / np.where(count_accum == 0, np.nan, count_accum)
    return mean.astype(np.float32)


def _compute_hr_correlator(
    conc_stack: np.memmap,
    *,
    x_slice: slice,
    y_slice: slice,
    t_slice: slice,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    nx = x_slice.stop - x_slice.start - PIXELS_PER_CM
    ny = y_slice.stop - y_slice.start - PIXELS_PER_CM
    nx = nx // PIXELS_PER_CM
    ny = ny // PIXELS_PER_CM
    # nx = x_slice.stop - x_slice.start - SHIFT_X
    # ny = y_slice.stop - y_slice.start - SHIFT_Y
    hr_x_sum = np.zeros((nx, ny), dtype=np.float32)
    hr_x_count = np.zeros((nx, ny), dtype=np.float32)
    hr_y_sum = np.zeros((nx, ny), dtype=np.float32)
    hr_y_count = np.zeros((nx, ny), dtype=np.float32)

    # loop through time in chunks for memory handling
    total_frames = t_slice.stop - t_slice.start - SHIFT_T
    first_core = t_slice.start + SHIFT_T
    for idx, core_start in enumerate(range(first_core, t_slice.stop, chunk_size), start=1):
        core_stop = min(core_start + chunk_size, t_slice.stop)
        halo_start = core_start 
        halo_stop = core_stop + SHIFT_T
        conc_chunk = np.asarray(conc_stack[x_slice, y_slice, halo_start:halo_stop], dtype=np.float32)
        # print(f"chunk dimensions: {conc_chunk.shape}")

        ######## motion correlator - Kadakia et al 2022 method ################

        # select locations on 1-cm lattice
        # may need to loop since each location may have different idx for argmax
        for i in np.arange(2, nx):
            xidx = int(i * PIXELS_PER_CM)
            # print(f'i: {i} xidx: {xidx}')

            for j in np.arange(2, ny):
                yidx = int(j * PIXELS_PER_CM)
                # print(f'j: {j} yidx: {yidx}')  
                # average y-values +/- 1 cm to get vector of x values +/- 1 cm from point of interest
                x_conc_vals = np.mean(conc_chunk[xidx-int(1.5*PIXELS_PER_CM):xidx+int(1.5*PIXELS_PER_CM), yidx-int(0.5*PIXELS_PER_CM):yidx+int(0.5*PIXELS_PER_CM), :], axis=1)
                # x_conc_vals[x_conc_vals<0.001] = np.nan
                # print(f"x_vector dimensions (expect 60 x tsteps): {x_conc_vals.shape}, min: {np.nanmin(x_conc_vals)}, max: {np.nanmax(x_conc_vals)}")
                # average x-values +/- 1 cm to get vector of y values +/- 1 cm from point of interest
                y_conc_vals = np.mean(conc_chunk[xidx-int(0.5*PIXELS_PER_CM):xidx+int(0.5*PIXELS_PER_CM), yidx-int(1.5*PIXELS_PER_CM):yidx+int(1.5*PIXELS_PER_CM), :], axis=0)
                # y_conc_vals[y_conc_vals<0.001] = np.nan
                # print(f"y_vector dimensions (expect 60 x tsteps): {y_conc_vals.shape}, min: {np.nanmin(y_conc_vals)}, max: {np.nanmax(y_conc_vals)}")
                # loop through time
                for tstep in np.arange(np.shape(conc_chunk)[2] - 1):
                    
                    ### compute x-direction correlator ###
                    # compute covariance C(x, t) * C(x+delta_x, t+delta_t) for delta_x from -20 to 20
                    cov_x_max = 0
                    max_x_idx = -PIXELS_PER_CM
                    for delta_x in np.arange(-PIXELS_PER_CM, PIXELS_PER_CM):
                        cov_temp_sum = 0
                        for idx_shift in np.arange(0, int(PIXELS_PER_CM)):
                            cov_temp_sum += x_conc_vals[idx_shift + PIXELS_PER_CM, tstep] * x_conc_vals[idx_shift + PIXELS_PER_CM + delta_x, tstep + 1]
                        cov_temp = cov_temp_sum / PIXELS_PER_CM
                        if cov_temp > cov_x_max:
                            cov_x_max = cov_temp
                            max_x_idx = delta_x 

                    # if index is +/- 20, skip this value. 
                    if np.abs(max_x_idx)<PIXELS_PER_CM:
                        # else, sum covariance to running total for this location and increment x_count by 1
                        hr_x_count[i, j] += 1
                        hr_x_sum[i, j] += max_x_idx

                    ### compute y-direction correlator ###
                    # compute covariance C(y, t) * C(y+delta_y, t+delta_t) for delta_y from -20 to 20
                    cov_y_max = 0
                    max_y_idx = -PIXELS_PER_CM
                    for delta_y in np.arange(-PIXELS_PER_CM, PIXELS_PER_CM):
                        cov_temp = y_conc_vals[PIXELS_PER_CM, tstep] * y_conc_vals[PIXELS_PER_CM + delta_y, tstep + 1]
                        if cov_temp > cov_y_max:
                            cov_y_max = cov_temp
                            max_y_idx = delta_y 
                    if np.abs(max_y_idx)<PIXELS_PER_CM:
                        hr_y_count[i, j] += 1
                        hr_y_sum[i, j] += max_y_idx


        #################################################################


        ########### HR correlator - Brudner et al method ################

        # hr_x_term1 = conc_chunk[:-SHIFT_X, SHIFT_Y:, SHIFT_T:] * conc_chunk[:-SHIFT_X, :-SHIFT_Y, :-SHIFT_T]
        # hr_x_term2 = conc_chunk[:-SHIFT_X, SHIFT_Y:, :-SHIFT_T] * conc_chunk[:-SHIFT_X, :-SHIFT_Y, SHIFT_T:]
        # hr_x_chunk = hr_x_term1 - hr_x_term2

        # hr_y_term1 = conc_chunk[SHIFT_X:, :-SHIFT_Y, SHIFT_T:] * conc_chunk[:-SHIFT_X, :-SHIFT_Y, :-SHIFT_T]
        # hr_y_term2 = conc_chunk[SHIFT_X:, :-SHIFT_Y, :-SHIFT_T] * conc_chunk[:-SHIFT_X, :-SHIFT_Y, SHIFT_T:]
        # hr_y_chunk = hr_y_term1 - hr_y_term2

        # hr_x_sum += np.nansum(hr_x_chunk, axis=2)
        # hr_x_count += np.sum(np.isfinite(hr_x_chunk), axis=2)
        # hr_y_sum += np.nansum(hr_y_chunk, axis=2)
        # hr_y_count += np.sum(np.isfinite(hr_y_chunk), axis=2)

        ############################################################################


        processed = core_stop - first_core
        print(
            f"Processed {processed}/{total_frames} frames "
            f"for HR correlator (chunk {idx}, t={core_start}:{core_stop})."
        )

    return _finalize_mean(hr_x_sum, hr_x_count), _finalize_mean(hr_y_sum, hr_y_count)


def _load_hr_frame(
    conc_stack: np.memmap,
    *,
    frame_idx: int,
    x_slice: slice,
    y_slice: slice,
    t_slice: slice,
) -> np.ndarray:
    abs_frame = t_slice.start + frame_idx
    if abs_frame < t_slice.start + SHIFT_T or abs_frame >= t_slice.stop:
        raise IndexError(f"QC frame {frame_idx} is out of range for selected frames.")

    chunk = np.asarray(conc_stack[x_slice, y_slice, abs_frame - SHIFT_T:abs_frame + 1], dtype=np.float32)
    hr_term1 = chunk[:-SHIFT_X, SHIFT_Y:, SHIFT_T:] * chunk[:-SHIFT_X, :-SHIFT_Y, :-SHIFT_T]
    hr_term2 = chunk[:-SHIFT_X, SHIFT_Y:, :-SHIFT_T] * chunk[:-SHIFT_X, :-SHIFT_Y, SHIFT_T:]
    return np.squeeze(hr_term1 - hr_term2)


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


def _plot_quiver_field(
    x_component: np.ndarray,
    y_component: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    fig_path: Path,
    cmap,
    threshold: float | None = None,
    stride: int = 1,
) -> None:
    nx = min(x_component.shape[0], y_component.shape[0])
    ny = min(x_component.shape[1], y_component.shape[1])
    u = np.asarray(x_component[:nx, :ny], dtype=np.float32).copy()
    v = np.asarray(y_component[:nx, :ny], dtype=np.float32).copy()
    magnitude = np.sqrt(u**2 + v**2)

    if threshold is not None:
        mask = magnitude < threshold
        u[mask] = np.nan
        v[mask] = np.nan
        magnitude[mask] = np.nan

    if stride > 1:
        u = u[::stride, ::stride]
        v = v[::stride, ::stride]
        magnitude = magnitude[::stride, ::stride]

    finite_mask = np.isfinite(magnitude)
    if not np.any(finite_mask):
        raise ValueError("Quiver plot requested, but vector field has no finite values.")

    log_magnitude = np.full_like(magnitude, np.nan, dtype=np.float32)
    log_magnitude[finite_mask] = np.log10(np.maximum(magnitude[finite_mask], 1e-12))
    log_min = float(np.nanmin(log_magnitude[finite_mask]))
    log_max = float(np.nanmax(log_magnitude[finite_mask]))
    if np.isclose(log_min, log_max):
        normalized_log_mag = np.ones_like(magnitude, dtype=np.float32)
    else:
        normalized_log_mag = np.full_like(magnitude, np.nan, dtype=np.float32)
        normalized_log_mag[finite_mask] = (log_magnitude[finite_mask] - log_min) / (log_max - log_min)

    x_offset = X_SLICE.start or 0
    y_offset = Y_SLICE.start or 0
    x_idx = np.arange(nx) + x_offset
    y_idx = np.arange(ny) + y_offset
    if stride > 1:
        x_idx = x_idx[::stride]
        y_idx = y_idx[::stride]
    yy, xx = np.meshgrid(y_idx, x_idx)

    unit_u = np.zeros_like(u, dtype=np.float32)
    unit_v = np.zeros_like(v, dtype=np.float32)
    unit_u[finite_mask] = u[finite_mask] / magnitude[finite_mask]
    unit_v[finite_mask] = v[finite_mask] / magnitude[finite_mask]
    length_scale = QUIVER_MIN_LENGTH + (QUIVER_MAX_LENGTH - QUIVER_MIN_LENGTH) * normalized_log_mag
    u_scaled = unit_u * length_scale
    v_scaled = unit_v * length_scale

    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver(
        yy,
        xx,
        u_scaled,
        v_scaled,
        log_magnitude,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap=cmap,
        pivot="mid",
        width=0.0025,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(q, ax=ax, label=cbar_label)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    conc_stack = _open_concentration_stack()
    # x_slice, y_slice, t_slice = _resolve_selected_slices(conc_stack)
    # n_frames = t_slice.stop - t_slice.start

    # for qc_frame in QC_FRAMES:
    #     if qc_frame >= n_frames:
    #         break
    #     if qc_frame < SHIFT_T:
    #         continue

    #     hr_frame = _load_hr_frame(
    #         conc_stack,
    #         frame_idx=qc_frame,
    #         x_slice=x_slice,
    #         y_slice=y_slice,
    #         t_slice=t_slice,
    #     )
    #     _plot_field(
    #         hr_frame,
    #         title=f"HR correlator frame {qc_frame}\ncase={CASE_NAME}",
    #         cbar_label="odor velocity (mm/s)",
    #         fig_path=OUT_DIR / f"QC_frames/HRcorrelator_frame_{qc_frame}_{CASE_NAME}.png",
    #         cmap=FIELD_CMAP,
    #         vmin=FIELD_VMIN,
    #         vmax=FIELD_VMAX,
    #         log_scale=FIELD_LOG_SCALE,
    #     )

    hr_x_mean, hr_y_mean = _compute_hr_correlator(
        conc_stack,
        x_slice=X_SLICE,
        y_slice=Y_SLICE,
        t_slice=T_SLICE,
        chunk_size=CHUNK_SIZE,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"KadakiaCorr_mean_{CASE_NAME}.npy", hr_x_mean)
    np.save(OUT_DIR / f"KadakiaCorr_y_mean_{CASE_NAME}.npy", hr_y_mean)

    _plot_quiver_field(
        hr_y_mean,
        hr_x_mean,
        title=f"Time-averaged motion correlator\ncase={CASE_NAME}",
        cbar_label="correlator magnitude",
        fig_path=OUT_DIR / f"hrCorr_quiver_{CASE_NAME}_t{T_SLICE.start}to{T_SLICE.stop}.png",
        cmap=VECTOR_CMAP,
        threshold=FIELD_THRESHOLD,
        stride=QUIVER_STRIDE,
    )

    print(f"Saved odor-motion outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
