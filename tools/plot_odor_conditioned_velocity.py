"""
Compute and plot the time-averaged velocity field conditioned on odor presence.

For each frame, retain velocity only where concentration exceeds ODOR_THRESHOLD and
set velocity to zero elsewhere. Then average the masked velocity components across
time and save the result as arrays plus a quiver plot.

Run:
    python tools/plot_odor_conditioned_velocity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "buoyant"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
U_PATH = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_u.npy"
V_PATH = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_v.npy"
OUT_DIR = BASE_PATH / "Plots" / "odor_conditioned_velocity"

X_SLICE = slice(0, 600)
CONC_Y_SLICE = slice(100, 500)
VEL_Y_SLICE = slice(100, 500)
T_SLICE = slice(0, 6000)
CHUNK_SIZE = 200
PIXELS_PER_CM = 20

ODOR_THRESHOLD = 0.005
COMPUTE_CONDITIONAL_MEAN = True
QUIVER_STEP = 1
QUIVER_MIN_LENGTH = 5
QUIVER_MAX_LENGTH = 20
QUIVER_CMAP = "Reds"
QUIVER_THRESHOLD = 0.001

XLABEL = "x index"
YLABEL = "y index"
FIG_DPI = 600
# -------------------------------------------------------------------

U_MEAN_PATH = OUT_DIR / f"odor_conditioned_u_mean_{CASE_NAME}.npy"
V_MEAN_PATH = OUT_DIR / f"odor_conditioned_v_mean_{CASE_NAME}.npy"


def _resolve_axis_slice(axis_slice: slice, axis_size: int) -> slice:
    start, stop, step = axis_slice.indices(axis_size)
    if step != 1:
        raise ValueError("Only unit-step slices are supported.")
    return slice(start, stop)


def _resolve_common_slices(
    conc_stack: np.memmap,
    u_stack: np.memmap,
    v_stack: np.memmap,
) -> tuple[slice, slice, slice, slice]:
    x_slice = _resolve_axis_slice(X_SLICE, min(conc_stack.shape[0], u_stack.shape[0], v_stack.shape[0]))
    conc_y_slice = _resolve_axis_slice(CONC_Y_SLICE, conc_stack.shape[1])
    vel_y_slice = _resolve_axis_slice(VEL_Y_SLICE, min(u_stack.shape[1], v_stack.shape[1]))
    t_slice = _resolve_axis_slice(T_SLICE, min(conc_stack.shape[2], u_stack.shape[2], v_stack.shape[2]))

    nx = x_slice.stop - x_slice.start
    conc_ny = conc_y_slice.stop - conc_y_slice.start
    vel_ny = vel_y_slice.stop - vel_y_slice.start
    nt = t_slice.stop - t_slice.start

    if nt <= 0:
        raise ValueError("T_SLICE produced zero frames to process.")
    if nx <= 0 or conc_ny <= 0 or vel_ny <= 0:
        raise ValueError("Configured slices produced an empty spatial selection.")
    if conc_ny != vel_ny:
        raise ValueError(
            f"Concentration slice shape {(nx, conc_ny)} does not match velocity slice shape {(nx, vel_ny)}. "
            "Adjust CONC_Y_SLICE or VEL_Y_SLICE so the masked fields align."
        )

    return x_slice, conc_y_slice, vel_y_slice, t_slice


def _open_stack(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    stack = np.load(path, mmap_mode="r")
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack at {path.name}; got shape {stack.shape}")
    return stack


def _coarsen_to_cm_blocks(field_chunk: np.ndarray, pixels_per_cm: int) -> np.ndarray:
    nx, ny, nt = field_chunk.shape
    coarse_nx = nx // pixels_per_cm
    coarse_ny = ny // pixels_per_cm
    if coarse_nx == 0 or coarse_ny == 0:
        raise ValueError(
            f"Selected region {(nx, ny)} is smaller than one {pixels_per_cm}x{pixels_per_cm} block."
        )

    trimmed = field_chunk[: coarse_nx * pixels_per_cm, : coarse_ny * pixels_per_cm, :]
    reshaped = trimmed.reshape(coarse_nx, pixels_per_cm, coarse_ny, pixels_per_cm, nt)
    return np.nanmean(reshaped, axis=(1, 3)).astype(np.float32)


def _compute_masked_velocity_mean(
    conc_stack: np.memmap,
    u_stack: np.memmap,
    v_stack: np.memmap,
    *,
    x_slice: slice,
    conc_y_slice: slice,
    vel_y_slice: slice,
    t_slice: slice,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    nx = (x_slice.stop - x_slice.start) // PIXELS_PER_CM
    conc_ny = (conc_y_slice.stop - conc_y_slice.start) // PIXELS_PER_CM
    vel_ny = (vel_y_slice.stop - vel_y_slice.start) // PIXELS_PER_CM
    if conc_ny != vel_ny:
        raise ValueError("Concentration and velocity selections do not produce the same number of 1 cm blocks.")
    ny = conc_ny
    u_sum = np.zeros((nx, ny), dtype=np.float64)
    v_sum = np.zeros((nx, ny), dtype=np.float64)

    total_frames = t_slice.stop - t_slice.start
    for idx, chunk_start in enumerate(range(t_slice.start, t_slice.stop, chunk_size), start=1):
        chunk_stop = min(chunk_start + chunk_size, t_slice.stop)
        conc_chunk = np.asarray(conc_stack[x_slice, conc_y_slice, chunk_start:chunk_stop], dtype=np.float32)
        u_chunk = np.asarray(u_stack[x_slice, vel_y_slice, chunk_start:chunk_stop], dtype=np.float32)
        v_chunk = np.asarray(v_stack[x_slice, vel_y_slice, chunk_start:chunk_stop], dtype=np.float32)

        conc_chunk = _coarsen_to_cm_blocks(conc_chunk, PIXELS_PER_CM)
        u_chunk = _coarsen_to_cm_blocks(u_chunk, PIXELS_PER_CM)
        v_chunk = _coarsen_to_cm_blocks(v_chunk, PIXELS_PER_CM)

        mask = conc_chunk >= ODOR_THRESHOLD
        u_masked = np.where(mask, u_chunk, 0.0)
        v_masked = np.where(mask, v_chunk, 0.0)

        u_sum += np.nansum(u_masked, axis=2)
        v_sum += np.nansum(v_masked, axis=2)

        processed = chunk_stop - t_slice.start
        print(
            f"Processed {processed}/{total_frames} frames "
            f"for odor-conditioned velocity mean (chunk {idx}, t={chunk_start}:{chunk_stop})."
        )

    divisor = float(total_frames)
    return (u_sum / divisor).astype(np.float32), (v_sum / divisor).astype(np.float32)


def _plot_quiver(
    u_mean: np.ndarray,
    v_mean: np.ndarray,
    *,
    title: str,
    fig_path: Path,
    cbar_label: str,
) -> None:
    magnitude = np.hypot(u_mean, v_mean)
    if QUIVER_THRESHOLD > 0:
        mask = magnitude < QUIVER_THRESHOLD
        u_mean = u_mean.copy()
        v_mean = v_mean.copy()
        magnitude = magnitude.copy()
        u_mean[mask] = np.nan
        v_mean[mask] = np.nan
        magnitude[mask] = np.nan

    u_plot = u_mean[::QUIVER_STEP, ::QUIVER_STEP]
    v_plot = v_mean[::QUIVER_STEP, ::QUIVER_STEP]
    mag_plot = magnitude[::QUIVER_STEP, ::QUIVER_STEP]
    finite_mask = np.isfinite(mag_plot)
    if not np.any(finite_mask):
        raise ValueError("No finite vectors available to plot.")

    x_offset = X_SLICE.start or 0
    y_offset = CONC_Y_SLICE.start or 0
    x_idx = (np.arange(u_mean.shape[0]) + x_offset)[::QUIVER_STEP]
    y_idx = (np.arange(u_mean.shape[1]) + y_offset)[::QUIVER_STEP]
    yy, xx = np.meshgrid(y_idx, x_idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver(
        yy,
        xx,
        u_plot,
        v_plot,
        mag_plot,
        angles="xy",
        scale_units="xy",
        scale=None,
        cmap=QUIVER_CMAP,
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
    if COMPUTE_CONDITIONAL_MEAN:
        conc_stack = _open_stack(CONCENTRATION_PATH)
        u_stack = _open_stack(U_PATH)
        v_stack = _open_stack(V_PATH)
        x_slice, conc_y_slice, vel_y_slice, t_slice = _resolve_common_slices(conc_stack, u_stack, v_stack)

        u_mean, v_mean = _compute_masked_velocity_mean(
            conc_stack,
            u_stack,
            v_stack,
            x_slice=x_slice,
            conc_y_slice=conc_y_slice,
            vel_y_slice=vel_y_slice,
            t_slice=t_slice,
            chunk_size=CHUNK_SIZE,
        )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(U_MEAN_PATH, u_mean)
        np.save(V_MEAN_PATH, v_mean)
    else:
        if not U_MEAN_PATH.exists():
            raise FileNotFoundError(f"Missing precomputed odor-conditioned u mean: {U_MEAN_PATH}")
        if not V_MEAN_PATH.exists():
            raise FileNotFoundError(f"Missing precomputed odor-conditioned v mean: {V_MEAN_PATH}")
        u_mean = np.load(U_MEAN_PATH)
        v_mean = np.load(V_MEAN_PATH)

    fig_path = OUT_DIR / f"odor_conditioned_velocity_quiver_{CASE_NAME}_thr{ODOR_THRESHOLD}_t{T_SLICE.start}to{T_SLICE.stop}.png"
    _plot_quiver(
        u_mean,
        v_mean,
        title=f"Odor-conditioned mean velocity\ncase={CASE_NAME}, c >= {ODOR_THRESHOLD}",
        fig_path=fig_path,
        cbar_label="Odor-conditioned V"
    )

    print(f"Saved odor-conditioned velocity arrays and plot to {OUT_DIR}")


if __name__ == "__main__":
    main()
