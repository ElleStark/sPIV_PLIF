"""
Compute flow properties using the helpers in analysis.flow_properties.

Edit the paths/settings below, then run:
    python tools/compute_flow_properties.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmasher as cmr

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.analysis import (
    compute_fluctuating_components,
    compute_fluctuating_strain_rates,
    compute_turbulence_intensity,
    load_mean_velocity_components,
    save_arrays,
)

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "smSource" 
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
INTERMITTENCY_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Intermittency/intermittency_{CASE_NAME}.npy")
X_SLICE = slice(0, 600)
Y_SLICE = slice(100, 500)
T_SLICE = slice(0, 6000)
DX = 0.0005  # m
DT = 0.05  # sec
NU = 1.5e-5  # kinematic viscosity, m2/s
INTERMITTENCY_THRESHOLD = 0.02  # concentration threshold used to build intermittency
PLUME_ENVELOPE_CUTOFF = 0.10  # intermittency >= this value is considered inside the plume envelope
SUMMARY_TXT_PATH = BASE_PATH / "flow_properties" / f"flow_property_stats_{CASE_NAME}.txt"
CMAP = cmr.get_sub_cmap("cmr.rainforest", 0.1, 1.0)
BINSIZE = 10  # bin size along x for mean profile plots
CHUNK_SIZE = 100  # number of time frames to process at once
# -------------------------------------------------------------------


def _plot_field(
    arr: np.ndarray,
    title: str,
    out_path: Path,
    *,
    vmin=None,
    vmax=None,
    cmap: str = "viridis",
    log_scale: bool = False,
) -> None:
    """Save a simple pcolormesh plot for a 2D field."""
    plt.figure(figsize=(6, 5))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if log_scale:
        data = np.asarray(arr)
        positive = data[data > 0]
        if positive.size == 0:
            raise ValueError(f"Cannot plot {title} on log scale: no positive values")
        if vmin is None or vmin <= 0:
            vmin = np.percentile(positive, 1)
        if vmax is None:
            vmax = np.percentile(positive, 99)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    plt.pcolormesh(arr, cmap=cmap, shading="auto", norm=norm)
    plt.colorbar(label=title)
    plt.xlabel("y index")
    plt.ylabel("x index")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_line_along_x(arr: np.ndarray, title: str, out_path: Path, *, bin_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Plot the mean profile along x (averaged over y), with optional binning over x rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = np.nanmean(arr, axis=1)
    if bin_size > 1:
        n_bins = profile.shape[0] // bin_size
        trimmed = profile[: n_bins * bin_size]
        profile = np.nanmean(trimmed.reshape(n_bins, bin_size), axis=1)
        x_vals = np.arange(n_bins) * bin_size + (bin_size - 1) / 2.0
    else:
        x_vals = np.arange(profile.shape[0])
    plt.figure(figsize=(7, 4))
    plt.plot(np.flip(x_vals), profile, lw=1.6)
    plt.xlabel("x index")
    plt.ylabel(title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return x_vals, profile


def _report_stats(name: str, arr: np.ndarray, *, log: list[str] | None = None) -> None:
    """Print mean, median, and selected quantiles for a field and capture them for logging."""
    finite = np.asarray(arr)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        line = f"{name}: no finite values"
        print(line)
        if log is not None:
            log.append(line)
        return

    quantiles = [0, 10, 25, 50, 75, 90, 100]
    values = np.percentile(finite, quantiles)
    formatted = ", ".join(f"{q}%={v:.5f}" for q, v in zip(quantiles, values))
    line = f"{name} mean={np.mean(finite):.5f}, median={np.median(finite):.5f}; quantiles: {formatted}"
    print(line)
    if log is not None:
        log.append(line)


def _report_region_stats(name: str, arr: np.ndarray, mask: np.ndarray, *, log: list[str]) -> None:
    """Report stats inside and outside the plume envelope mask."""
    _report_stats(f"{name} (inside envelope)", arr[mask], log=log)
    _report_stats(f"{name} (outside envelope)", arr[~mask], log=log)


def main() -> None:
    log_lines: list[str] = [f"Flow property stats for case '{CASE_NAME}'"]
    log_lines.append(
        f"Intermittency threshold={INTERMITTENCY_THRESHOLD} (concentration), envelope cutoff={PLUME_ENVELOPE_CUTOFF} (intermittency >= cutoff)"
    )

    # Load mean fields (small) and set up mmap readers for velocity
    u_mean_full, v_mean_full, w_mean_full = load_mean_velocity_components(CASE_NAME, base_path=BASE_PATH)
    u_mean = u_mean_full[X_SLICE, Y_SLICE]
    v_mean = v_mean_full[X_SLICE, Y_SLICE]
    w_mean = w_mean_full[X_SLICE, Y_SLICE]

    u_path = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_u.npy"
    v_path = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_v.npy"
    w_path = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_w.npy"
    u_mmap = np.load(u_path, mmap_mode="r")
    v_mmap = np.load(v_path, mmap_mode="r")
    w_mmap = np.load(w_path, mmap_mode="r")

    x_indices = range(*X_SLICE.indices(u_mmap.shape[0]))
    y_indices = range(*Y_SLICE.indices(u_mmap.shape[1]))
    t_indices = range(*T_SLICE.indices(u_mmap.shape[2]))
    if t_indices.step != 1:
        raise ValueError("Chunked processing currently assumes contiguous time slices (step=1)")
    if len(t_indices) == 0:
        raise ValueError("T_SLICE produced zero frames to process")
    nx, ny, nt = len(x_indices), len(y_indices), len(t_indices)

    if not INTERMITTENCY_PATH.exists():
        raise FileNotFoundError(f"Intermittency file not found: {INTERMITTENCY_PATH}")
    intermittency_full = np.load(INTERMITTENCY_PATH)
    intermittency = intermittency_full[X_SLICE, Y_SLICE]
    plume_mask = intermittency >= PLUME_ENVELOPE_CUTOFF
    inside_px = int(np.count_nonzero(plume_mask))
    total_px = int(plume_mask.size)
    log_lines.append(f"Plume mask pixels: inside={inside_px}, outside={total_px - inside_px}")
    # if plume_mask.shape != u.shape[:2]:
    #     raise ValueError(f"Intermittency mask shape {plume_mask.shape} does not match velocity fields {u.shape[:2]}")

    # Pre-allocate outputs and accumulators for chunked processing
    flx_dir = BASE_PATH / "flow_properties" / "flx_u_v_w"
    flx_dir.mkdir(parents=True, exist_ok=True)
    strain_dir = BASE_PATH / "flow_properties" / "flx_StrainRates"
    strain_dir.mkdir(parents=True, exist_ok=True)

    u_flx_mm = np.lib.format.open_memmap(flx_dir / f"u_flx_{CASE_NAME}.npy", mode="w+", dtype=u_mmap.dtype, shape=(nx, ny, nt))
    v_flx_mm = np.lib.format.open_memmap(flx_dir / f"v_flx_{CASE_NAME}.npy", mode="w+", dtype=v_mmap.dtype, shape=(nx, ny, nt))
    w_flx_mm = np.lib.format.open_memmap(flx_dir / f"w_flx_{CASE_NAME}.npy", mode="w+", dtype=w_mmap.dtype, shape=(nx, ny, nt))

    du_dx_mm = np.lib.format.open_memmap(strain_dir / f"duflx_dx_{CASE_NAME}.npy", mode="w+", dtype=np.float32, shape=(nx, ny, nt))
    dv_dy_mm = np.lib.format.open_memmap(strain_dir / f"dvflx_dy_{CASE_NAME}.npy", mode="w+", dtype=np.float32, shape=(nx, ny, nt))
    dw_dz_mm = np.lib.format.open_memmap(strain_dir / f"dwflx_dz_{CASE_NAME}.npy", mode="w+", dtype=np.float32, shape=(nx, ny, nt))

    sum_u2 = np.zeros((nx, ny), dtype=np.float64)
    sum_v2 = np.zeros((nx, ny), dtype=np.float64)
    sum_w2 = np.zeros((nx, ny), dtype=np.float64)
    sum_du2 = np.zeros((nx, ny), dtype=np.float64)
    sum_dv2 = np.zeros((nx, ny), dtype=np.float64)
    sum_dw2 = np.zeros((nx, ny), dtype=np.float64)
    frame_count = 0

    for t_start in range(0, nt, CHUNK_SIZE):
        t_stop = min(t_start + CHUNK_SIZE, nt)
        raw_start = t_indices[t_start]
        raw_stop = t_indices[t_stop - 1] + 1  # exclusive
        chunk_slice = slice(raw_start, raw_stop)
        u_chunk = np.asarray(u_mmap[X_SLICE, Y_SLICE, chunk_slice])
        v_chunk = np.asarray(v_mmap[X_SLICE, Y_SLICE, chunk_slice])
        w_chunk = np.asarray(w_mmap[X_SLICE, Y_SLICE, chunk_slice])

        u_flx_chunk, v_flx_chunk, w_flx_chunk = compute_fluctuating_components(u_chunk, v_chunk, w_chunk, u_mean, v_mean, w_mean)
        chunk_len = u_flx_chunk.shape[2]
        u_flx_mm[:, :, t_start:t_stop] = u_flx_chunk
        v_flx_mm[:, :, t_start:t_stop] = v_flx_chunk
        w_flx_mm[:, :, t_start:t_stop] = w_flx_chunk

        duflx_dx_chunk, dvflx_dy_chunk, dwflx_dz_chunk = compute_fluctuating_strain_rates(
            u_flx_chunk, v_flx_chunk, dx=DX, dt=DT
        )
        du_dx_mm[:, :, t_start:t_stop] = duflx_dx_chunk
        dv_dy_mm[:, :, t_start:t_stop] = dvflx_dy_chunk
        dw_dz_mm[:, :, t_start:t_stop] = dwflx_dz_chunk

        sum_u2 += np.sum(u_flx_chunk**2, axis=2)
        sum_v2 += np.sum(v_flx_chunk**2, axis=2)
        sum_w2 += np.sum(w_flx_chunk**2, axis=2)
        sum_du2 += np.sum(duflx_dx_chunk**2, axis=2)
        sum_dv2 += np.sum(dvflx_dy_chunk**2, axis=2)
        sum_dw2 += np.sum(dwflx_dz_chunk**2, axis=2)
        frame_count += chunk_len

    u_flx_mm.flush()
    v_flx_mm.flush()
    w_flx_mm.flush()
    du_dx_mm.flush()
    dv_dy_mm.flush()
    dw_dz_mm.flush()
    print(f"Saved fluctuating components to {flx_dir}")
    print(f"Saved fluctuating strain rates to {strain_dir}")

    mean_u2 = sum_u2 / frame_count
    mean_v2 = sum_v2 / frame_count
    mean_w2 = sum_w2 / frame_count

    mean_du2 = sum_du2 / frame_count
    mean_dv2 = sum_dv2 / frame_count
    mean_dw2 = sum_dw2 / frame_count

    # Viscous dissipation
    epsilon = 5 * NU * (mean_du2 + mean_dv2 + mean_dw2)
    eps_path = BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" /f"epsilon_{CASE_NAME}.npy"
    save_arrays([(eps_path, epsilon)])
    _plot_field(epsilon, "epsilon", eps_path.with_suffix(".png"), vmin=np.percentile(epsilon, 1), vmax=np.percentile(epsilon, 90), cmap=CMAP)
    print(f"Saved viscous dissipation to {eps_path}")
    _report_stats("epsilon", epsilon, log=log_lines)
    _report_region_stats("epsilon", epsilon, plume_mask, log=log_lines)

    # Taylor scales and Reynolds number
    avg_rms = np.sqrt((1/3) * (mean_u2 + mean_v2 + mean_w2))
    safe_epsilon = np.where(epsilon <= 0, np.nan, epsilon)
    Taylor_microscale = np.sqrt(15 * NU / safe_epsilon) * avg_rms
    kolmogorov_length_scale = (NU**3 / safe_epsilon) ** 0.25
    kolmogorov_time_scale = (NU / safe_epsilon) ** 0.5
    Taylor_Re = avg_rms * Taylor_microscale / NU

    save_arrays(
        [
            (BASE_PATH / "flow_properties" /"Plots"/ f"{CASE_NAME}" / f"Taylor_microscale_{CASE_NAME}.npy", Taylor_microscale),
            (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"Taylor_Re_{CASE_NAME}.npy", Taylor_Re),
            (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"kolmogorov_length_scale_{CASE_NAME}.npy", kolmogorov_length_scale),
            (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"kolmogorov_time_scale_{CASE_NAME}.npy", kolmogorov_time_scale),
        ]
    )
    _plot_field(Taylor_microscale, "Taylor microscale", (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"Taylor_microscale_{CASE_NAME}.png"), vmin=0, vmax=np.percentile(Taylor_microscale, 99))
    _plot_field(Taylor_Re, "Taylor Re", (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"Taylor_Re_{CASE_NAME}.png"), vmin=0, vmax=np.percentile(Taylor_Re, 99))
    _plot_field(kolmogorov_length_scale, "Kolmogorov length scale", (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"kolmogorov_length_scale_{CASE_NAME}.png"), vmin=0, vmax=np.percentile(kolmogorov_length_scale, 99))
    _plot_field(kolmogorov_time_scale, "Kolmogorov time scale", (BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"kolmogorov_time_scale_{CASE_NAME}.png"), vmin=0, vmax=np.percentile(kolmogorov_time_scale, 99))
    _report_stats("Taylor microscale", Taylor_microscale, log=log_lines)
    _report_region_stats("Taylor microscale", Taylor_microscale, plume_mask, log=log_lines)
    _report_stats("Taylor Re", Taylor_Re, log=log_lines)
    _report_region_stats("Taylor Re", Taylor_Re, plume_mask, log=log_lines)
    _report_stats("Kolmogorov length scale", kolmogorov_length_scale, log=log_lines)
    _report_region_stats("Kolmogorov length scale", kolmogorov_length_scale, plume_mask, log=log_lines)
    _report_stats("Kolmogorov time scale", kolmogorov_time_scale, log=log_lines)
    _report_region_stats("Kolmogorov time scale", kolmogorov_time_scale, plume_mask, log=log_lines)

    # Turbulent kinetic energy
    u_mnsq, v_mnsq, w_mnsq = mean_u2, mean_v2, mean_w2
    tke = 0.5 * (u_mnsq + v_mnsq + w_mnsq)
    tke_path = BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" /f"tke_{CASE_NAME}.npy"
    save_arrays([(tke_path, tke)])
    _plot_field(tke, "TKE", tke_path.with_suffix(".png"), vmin=0, vmax=np.percentile(tke, 99), cmap=CMAP)
    x_tke, tke_profile = _plot_line_along_x(
        tke,
        f"Mean TKE along x (avg over y, binned x={BINSIZE})",
        tke_path.with_suffix(".profile.png"),
        bin_size=BINSIZE,
    )
    np.savez(tke_path.with_suffix(".profile.npz"), x_index_center=x_tke, tke_profile=tke_profile)
    print(f"Saved TKE to {tke_path}")
    _report_stats("TKE", tke, log=log_lines)
    _report_region_stats("TKE", tke, plume_mask, log=log_lines)
    anisotropy_line = (
        f"anisotropy ratios: <u'^2>/TKE={np.nanmean(u_mnsq/tke)}: <v'^2>/TKE={np.nanmean(v_mnsq/tke)}, <w'^2>/TKE={np.nanmean(w_mnsq/tke)})"
    )
    print(anisotropy_line)
    log_lines.append(anisotropy_line)

    t_intensity_avg = compute_turbulence_intensity(u_flx_mm, v_flx_mm, w_flx_mm, u_mean=0.30)
    t_intensity_path = BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" / f"turbulence_intensity_{CASE_NAME}.npy"
    save_arrays([(t_intensity_path, t_intensity_avg)])
    _plot_field(
        t_intensity_avg,
        "Turbulence intensity",
        t_intensity_path.with_suffix(".png"),
        log_scale=False,
        vmin = 0.0,
        vmax = 0.4,
        cmap=CMAP,
    )
    x_profile, ti_profile = _plot_line_along_x(
        t_intensity_avg,
        f"Mean turbulence intensity along x (avg over y, binned x={BINSIZE})",
        BASE_PATH / "flow_properties" / "Plots" / f"{CASE_NAME}" / f"turbulence_intensity_profile_{CASE_NAME}.png",
        bin_size=BINSIZE,
    )
    profile_vec_path = BASE_PATH / "flow_properties" / "Plots" / f"{CASE_NAME}" / f"turbulence_intensity_profile_{CASE_NAME}.npz"
    np.savez(profile_vec_path, x_index_center=x_profile, ti_profile=ti_profile)
    print(f"Saved turbulence intensity to {t_intensity_path}")
    _report_stats("turbulence intensity", t_intensity_avg, log=log_lines)
    _report_region_stats("turbulence intensity", t_intensity_avg, plume_mask, log=log_lines)

    SUMMARY_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Wrote summary stats to {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()
