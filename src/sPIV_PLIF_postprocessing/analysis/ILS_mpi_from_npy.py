# Source code to run on CU Boulder HPC resources: Blanca cluster
# Computes integral length scales (ILS) of a velocity component in either the streamwise or cross-stream direction
# v3: uses velocity fluctuations (time mean removed), correct axis handling for both directions,
#     correlation coefficient normalization, integration to first zero crossing, and QC autocorrelation plots
# v4: reads the PIV fields interpolated to the PLIF grid, shape (t, y, x), and takes DX from the PLIF grid files.
#     Each process reads a contiguous slab of y rows (efficient for the (t, y, x) layout); for correlations along y,
#     slabs are redistributed with MPI Alltoallv so each process holds full y-columns for its share of x.
# Elle Stark May 2024, updated Oct 2026
#
# Approach generally follows Tootoonian et. al. 2023 (https://github.com/stootoon/fisher-plumes/blob/main/boulder.py):
#   f(r) at each point is the average of the +r and -r spatial autocorrelation coefficients of u' about that point,
#   and the ILS is the integral of f(r) dr. Here the integral is taken to the first zero crossing of f(r);
#   the full-lag integral is also saved for comparison.
#
# Outputs (in OUT_DIR):
#   ILS_<name>_<direction>.npy           ILS integrated to first zero crossing [m]
#   ILS_<name>_<direction>_fulllag.npy   ILS integrated over all available lags [m]
#   ILS_<name>_<direction>_r0.npy        lag of first zero crossing [m]; NaN where f(r) never crosses zero
#   QC/                                  autocorrelation plots + curves (.npz) at QC_LOCATIONS, and summary maps
# All maps are saved with shape (ny, nx) in the same (y, x) layout as the PLIF grid files, with no flip or transpose,
# so streamwise and cross-stream maps line up pixel-for-pixel with each other and with the PLIF data.

import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

trapz = getattr(np, 'trapezoid', None) or np.trapz  # np.trapz is deprecated in NumPy 2.x

# Set up logging for convenient messages
logger = logging.getLogger('ilsmpipy')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warning
DEBUG = logger.debug

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = 'baseline'
COMPONENT = 'v'
DATA_DIR = Path('/scratch/alpine/elst4602/sPIV_PLIF')  # staged copy of the PIV (interpolated to PLIF grid) + PLIF grid files
FILENAME = DATA_DIR / 'PIV' / f'piv_{CASE_NAME}_{COMPONENT}.npy'  # PIV interpolated to PLIF grid, (t, y, x)
X_GRID_PATH = DATA_DIR / 'PLIF' / f'{CASE_NAME}_xgrid.npy'  # (ny, nx) [mm], x varies along axis 1
Y_GRID_PATH = DATA_DIR / 'PLIF' / f'{CASE_NAME}_ygrid.npy'  # (ny, nx) [mm], y varies along axis 0
DATAFILE_NAME = f'{CASE_NAME}_{COMPONENT}'
OUT_DIR = Path('/scratch/alpine/elst4602/sPIV_PLIF/ILS')

# Correlation direction: True = streamwise (along y, rows of the PLIF grid, y = 0-300 mm),
# False = cross-stream (along x, columns of the PLIF grid, x = -150 to 150 mm)
STREAMWISE = True

DX_OVERRIDE = None     # grid spacing [m]; None = compute from the PLIF grid file along the correlation direction
T_SLICE = slice(None)  # frames to use; slice(None) = all frames. Time mean is computed over these frames.
MIN_PAIRS = 100        # minimum number of valid (non-NaN) time samples for a correlation value to be kept

# QC: (y_idx, x_idx) locations on the PLIF grid at which to save autocorrelation plots
QC_LOCATIONS = [(33, 473), (473, 473), (940, 473), (33, 308), (473, 308), (33, 638), (473, 638)]
QC_DPI = 200
# -------------------------------------------------------------------


def split_range(n, num_procs, rank):
    """[start, end) indices of rank's share when n items are split as evenly as possible across num_procs."""
    size, remainder = divmod(n, num_procs)
    start = rank * size + min(rank, remainder)
    return start, start + size + (1 if rank < remainder else 0)


def load_y_slab(filename, y_start, y_end, t_slice):
    """Load rows [y_start, y_end) for all x from a (t, y, x) file; contiguous on disk for each frame.
    Returns float32 array with shape (ny_local, nx, nt)."""
    data = np.load(filename, mmap_mode='r')
    if data.ndim != 3:
        raise ValueError(f'Expected 3D (t, y, x) array; got shape {data.shape}')
    slab = np.array(data[t_slice, y_start:y_end, :], dtype=np.float32)  # copy: mmap view is read-only
    return np.ascontiguousarray(slab.transpose(1, 2, 0))


def build_alltoall_sendbuf(slab, x_ranges):
    """Pack a (ny_local, nx, nt) slab into one buffer of per-destination blocks slab[:, xs:xe, :]."""
    blocks = [np.ascontiguousarray(slab[:, xs:xe, :]).ravel() for xs, xe in x_ranges]
    counts = np.array([b.size for b in blocks], dtype=np.int64)
    return np.concatenate(blocks), counts


def redistribute_to_x_columns(comm, slab, y_ranges, x_ranges, rank, nt):
    """Convert per-rank y-slabs (ny_local, nx, nt) into per-rank x-slabs (ny, nx_local, nt).
    Blocks received from each source rank are (ny_src, nx_local, nt); stacking them in rank order along axis 0
    gives the full y extent, so the receive buffer reshapes directly."""
    sendbuf, scounts = build_alltoall_sendbuf(slab, x_ranges)
    nx_local = x_ranges[rank][1] - x_ranges[rank][0]
    rcounts = np.array([(ye - ys) * nx_local * nt for ys, ye in y_ranges], dtype=np.int64)
    sdispls = np.concatenate([[0], np.cumsum(scounts)[:-1]])
    rdispls = np.concatenate([[0], np.cumsum(rcounts)[:-1]])
    if max(scounts.sum(), rcounts.sum()) >= 2**31:
        raise ValueError('Alltoallv buffer exceeds MPI int32 count limit; use more processes or fewer frames')
    recvbuf = np.empty(rcounts.sum(), dtype=np.float32)
    comm.Alltoallv([sendbuf, (scounts.astype(int), sdispls.astype(int)), MPI.FLOAT],
                   [recvbuf, (rcounts.astype(int), rdispls.astype(int)), MPI.FLOAT])
    ny = y_ranges[-1][1]
    return recvbuf.reshape(ny, nx_local, nt)


def grid_spacing_m(grid, axis):
    """Mean spacing [m] of a (ny, nx) grid in mm along the given axis; warns if non-uniform."""
    d = np.abs(np.diff(grid, axis=axis))
    if np.nanstd(d) > 1e-3 * np.nanmean(d):
        WARN(f'Grid spacing is non-uniform along axis {axis}: mean {np.nanmean(d):.5f}, std {np.nanstd(d):.2e} mm')
    return float(np.nanmean(d)) / 1000


def correlation_coefficients(a, min_pairs):
    """Spatial correlation coefficient matrix rho[j, k] = <a'_j a'_k> / (sigma_j sigma_k) for a (n, nt) array
    of fluctuations, ignoring NaN samples."""
    valid = np.isfinite(a)
    a0 = np.where(valid, a, 0.0).astype(np.float64)
    cov_sum = a0 @ a0.T
    if valid.all():
        counts = np.full(cov_sum.shape, a.shape[1], dtype=np.float64)
    else:
        v = valid.astype(np.float64)
        counts = v @ v.T
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = cov_sum / counts
        cov[counts < min_pairs] = np.nan
        var = np.diag(cov).copy()
        var[var <= 0] = np.nan
        rho = cov / np.sqrt(np.outer(var, var))
    return rho


def autocorrelation_at(rho, j):
    """Return f_plus(r), f_minus(r), and their average f(r) for r = 0..n-1 about point j (NaN where unavailable)."""
    n = rho.shape[0]
    f_plus = np.full(n, np.nan)
    f_minus = np.full(n, np.nan)
    f_plus[:n - j] = rho[j, j:]
    f_minus[:j + 1] = rho[j, j::-1]
    both = np.vstack([f_plus, f_minus])
    n_valid = np.isfinite(both).sum(axis=0)
    f_avg = np.where(n_valid > 0, np.nansum(both, axis=0) / np.maximum(n_valid, 1), np.nan)
    return f_plus, f_minus, f_avg


def integrate_autocorrelation(f, dx):
    """Integrate f(r) (trapezoid rule) to its first zero crossing, linearly interpolating the crossing point.
    Returns (ils_zero, ils_full, r0). If f never crosses zero before running out of finite values,
    ils_zero integrates all finite lags and r0 is NaN."""
    finite = np.isfinite(f)
    if not finite[0]:
        return np.nan, np.nan, np.nan
    # Use only the contiguous run of finite values starting at r=0
    n_fin = np.argmin(finite) if not finite.all() else f.size
    f = f[:n_fin]
    ils_full = trapz(f, dx=dx) if f.size > 1 else 0.0

    below = np.flatnonzero(f <= 0)
    if below.size == 0:
        return ils_full, ils_full, np.nan
    k = below[0]
    frac = f[k - 1] / (f[k - 1] - f[k])  # fraction of the interval [k-1, k] before the crossing
    ils_zero = (trapz(f[:k], dx=dx) if k > 1 else 0.0) + 0.5 * f[k - 1] * frac * dx
    r0 = (k - 1 + frac) * dx
    return ils_zero, ils_full, r0


def plot_qc_autocorrelation(f_plus, f_minus, f_avg, dx, ils_zero, ils_full, r0, title, out_path):
    r_mm = np.arange(f_avg.size) * dx * 1000
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_mm, f_plus, color='tab:blue', lw=0.8, alpha=0.6, label='f(+r)')
    ax.plot(r_mm, f_minus, color='tab:orange', lw=0.8, alpha=0.6, label='f(-r)')
    ax.plot(r_mm, f_avg, color='k', lw=1.5, label='f(r), averaged')
    ax.axhline(0, color='gray', lw=0.8)
    if np.isfinite(r0):
        mask = r_mm <= r0 * 1000
        ax.fill_between(r_mm[mask], 0, f_avg[mask], color='tab:green', alpha=0.25, label='integrated to 1st zero')
        ax.axvline(r0 * 1000, color='tab:green', ls='--', lw=1)
    ax.set_xlabel('r [mm]')
    ax.set_ylabel('autocorrelation coefficient')
    ax.set_title(title, fontsize=9)
    ax.text(0.98, 0.97,
            f'ILS (1st zero) = {ils_zero * 1000:.2f} mm\nILS (all lags) = {ils_full * 1000:.2f} mm\n'
            f'r0 = {r0 * 1000:.2f} mm' if np.isfinite(r0) else
            f'ILS (all lags) = {ils_full * 1000:.2f} mm\nno zero crossing',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.legend(fontsize=7, loc='center right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=QC_DPI)
    plt.close(fig)


def gather_map(comm, rank, local_map, n_other, n_corr):
    """Gather (n_local, n_corr) maps from all ranks into a (n_other, n_corr) array on rank 0."""
    sendbuf = np.ascontiguousarray(local_map, dtype=np.float64).ravel()
    counts = np.array(comm.allgather(sendbuf.size), dtype=int)
    displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    recvbuf = np.empty(n_other * n_corr, dtype=np.float64) if rank == 0 else None
    comm.Gatherv(sendbuf, [recvbuf, counts, displs, MPI.DOUBLE], root=0)
    return recvbuf.reshape(n_other, n_corr) if rank == 0 else None


def main():

    # MPI setup; number of processes is set in the Slurm job script (.sh file)
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    num_procs = comm_world.Get_size()
    if rank == 0:
        INFO(f'RUNNING ON {num_procs} PROCESSES.')

    # Spatial (y, x) axes: streamwise = y = spatial axis 0, cross-stream = x = spatial axis 1
    corr_axis = 0 if STREAMWISE else 1
    other_axis = 1 - corr_axis
    direction = 'streamwise' if STREAMWISE else 'cross_stream'

    data_shape = np.load(FILENAME, mmap_mode='r').shape
    nt = len(range(*T_SLICE.indices(data_shape[0])))
    shape = data_shape[1:]  # (ny, nx)
    n_corr, n_other = shape[corr_axis], shape[other_axis]

    if DX_OVERRIDE is not None:
        dx = DX_OVERRIDE
    else:
        grid = np.load(Y_GRID_PATH if STREAMWISE else X_GRID_PATH)
        if grid.shape != shape:
            raise ValueError(f'Grid shape {grid.shape} does not match data spatial shape {shape}')
        dx = grid_spacing_m(grid, corr_axis)

    if rank == 0:
        INFO(f'Data shape (t, y, x) = {data_shape}, using {nt} frames; correlating along '
             f'{"y" if STREAMWISE else "x"} ({direction}), dx = {dx * 1000:.5f} mm')
        (OUT_DIR / 'QC').mkdir(parents=True, exist_ok=True)
    comm_world.Barrier()

    # Each process reads a contiguous slab of y rows; for streamwise correlations, redistribute into x-columns.
    # chunk has canonical shape (n_corr, n_local, nt), where n_local spans [start_idx, end_idx) of the other axis.
    y_ranges = [split_range(shape[0], num_procs, r) for r in range(num_procs)]
    slab = load_y_slab(FILENAME, *y_ranges[rank], T_SLICE)
    DEBUG(f'Process {rank} loaded y rows {y_ranges[rank]}: slab shape {slab.shape} ({slab.nbytes / 1e6:.1f} MB)')
    if STREAMWISE:
        x_ranges = [split_range(shape[1], num_procs, r) for r in range(num_procs)]
        chunk = redistribute_to_x_columns(comm_world, slab, y_ranges, x_ranges, rank, nt)
        start_idx, end_idx = x_ranges[rank]
    else:
        chunk = np.ascontiguousarray(slab.transpose(1, 0, 2))
        start_idx, end_idx = y_ranges[rank]
    del slab
    n_local = end_idx - start_idx

    # Remove the time mean at each point to get fluctuations
    chunk -= np.nanmean(chunk, axis=2, keepdims=True)
    DEBUG(f'Process {rank}: other-axis indices [{start_idx}, {end_idx}), chunk shape {chunk.shape}')

    # QC locations owned by this process, as (j along corr axis, local column k)
    qc_here = {}
    for loc in QC_LOCATIONS:
        j, other = loc[corr_axis], loc[other_axis]
        if not (0 <= j < n_corr and 0 <= other < n_other):
            if rank == 0:
                WARN(f'QC location {loc} out of bounds for shape {shape}; skipping')
            continue
        if start_idx <= other < end_idx:
            qc_here.setdefault(other - start_idx, []).append((j, loc))

    ils_zero_local = np.full((n_local, n_corr), np.nan)
    ils_full_local = np.full((n_local, n_corr), np.nan)
    r0_local = np.full((n_local, n_corr), np.nan)

    for k in range(n_local):
        rho = correlation_coefficients(chunk[:, k, :], MIN_PAIRS)
        for j in range(n_corr):
            f_plus, f_minus, f_avg = autocorrelation_at(rho, j)
            ils_zero, ils_full, r0 = integrate_autocorrelation(f_avg, dx)
            ils_zero_local[k, j], ils_full_local[k, j], r0_local[k, j] = ils_zero, ils_full, r0

        for j, loc in qc_here.get(k, []):
            f_plus, f_minus, f_avg = autocorrelation_at(rho, j)
            ils_zero, ils_full, r0 = integrate_autocorrelation(f_avg, dx)
            tag = f'{DATAFILE_NAME}_{direction}_{loc[0]}_{loc[1]}'
            plot_qc_autocorrelation(f_plus, f_minus, f_avg, dx, ils_zero, ils_full, r0,
                                    f'{DATAFILE_NAME}, {direction}, (y_idx, x_idx) = {loc}',
                                    OUT_DIR / 'QC' / f'autocorr_{tag}.png')
            np.savez(OUT_DIR / 'QC' / f'autocorr_{tag}.npz', r=np.arange(n_corr) * dx,
                     f_plus=f_plus, f_minus=f_minus, f_avg=f_avg, ils_zero=ils_zero, ils_full=ils_full, r0=r0)
            INFO(f'QC {loc}: ILS(1st zero)={ils_zero * 1000:.2f} mm, ILS(all lags)={ils_full * 1000:.2f} mm, '
                 f'r0={r0 * 1000:.2f} mm')

    DEBUG(f'Process {rank} completed')

    # GATHER ALL RESULTS INTO PROCESS 0, then return to (y, x) layout
    maps = {}
    for name, local in (('', ils_zero_local), ('_fulllag', ils_full_local), ('_r0', r0_local)):
        gathered = gather_map(comm_world, rank, local, n_other, n_corr)
        if rank == 0:
            maps[name] = gathered.T if corr_axis == 0 else gathered

    if rank == 0:
        for name, arr in maps.items():
            np.save(OUT_DIR / f'ILS_{DATAFILE_NAME}_{direction}{name}.npy', arr.astype(np.float32))

        no_zero = np.isnan(maps['_r0']) & np.isfinite(maps[''])
        INFO(f'ILS map shape: {maps[""].shape}')
        INFO(f'Domain-mean ILS (1st zero) = {np.nanmean(maps[""]) * 1000:.2f} mm; '
             f'(all lags) = {np.nanmean(maps["_fulllag"]) * 1000:.2f} mm')
        INFO(f'Points with no zero crossing: {no_zero.sum()} ({100 * no_zero.mean():.1f}%)')

        # QC summary maps
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        panels = [(maps[''], 'ILS to 1st zero [mm]'), (maps['_fulllag'], 'ILS, all lags [mm]'),
                  (maps['_r0'], 'r0, 1st zero crossing [mm]')]
        for ax, (arr, title) in zip(axs, panels):
            vals = arr[np.isfinite(arr)] * 1000
            vmax = np.percentile(vals, 99) if vals.size else None
            im = ax.pcolormesh(arr * 1000, cmap='magma_r', vmin=0 if vmax else None, vmax=vmax, shading='auto')
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('x index')
            ax.set_ylabel('y index')
            ax.set_aspect('equal')
        for loc in QC_LOCATIONS:
            axs[0].plot(loc[1] + 0.5, loc[0] + 0.5, 'c+', ms=8)
        fig.suptitle(f'{DATAFILE_NAME}, {direction} (QC locations marked +)')
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'QC' / f'ILS_maps_{DATAFILE_NAME}_{direction}.png', dpi=QC_DPI)
        plt.close(fig)


if __name__ == "__main__":
    main()
