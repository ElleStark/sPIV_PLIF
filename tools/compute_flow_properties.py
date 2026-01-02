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
    compute_taylor_scales,
    compute_turbulence_intensity,
    compute_turbulent_kinetic_energy,
    compute_viscous_dissipation,
    load_mean_velocity_components,
    load_velocity_components,
    save_arrays,
)

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "diffusive" 
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
INTERMITTENCY_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Intermittency/intermittency_{CASE_NAME}.npy")
X_SLICE = slice(0, 570)
Y_SLICE = slice(100, 500)
T_SLICE = slice(0, 50)
DX = 0.0005  # m
DT = 0.05  # sec
NU = 1.5e-5  # kinematic viscosity, m2/s
INTERMITTENCY_THRESHOLD = 0.02  # concentration threshold used to build intermittency
PLUME_ENVELOPE_CUTOFF = 0.10  # intermittency >= this value is considered inside the plume envelope
SUMMARY_TXT_PATH = BASE_PATH / "flow_properties" / f"flow_property_stats_{CASE_NAME}.txt"
CMAP = cmr.get_sub_cmap("cmr.rainforest", 0.1, 1.0)
BINSIZE = 16  # bin size along x for mean profile plots
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

    # Load inputs
    u, v, w = load_velocity_components(CASE_NAME, base_path=BASE_PATH, x_slice=X_SLICE, y_slice=Y_SLICE, t_slice=T_SLICE)
    u_mean_full, v_mean_full, w_mean_full = load_mean_velocity_components(CASE_NAME, base_path=BASE_PATH)
    # Align mean fields to the spatial slice used for instantaneous data
    u_mean = u_mean_full[X_SLICE, Y_SLICE]
    v_mean = v_mean_full[X_SLICE, Y_SLICE]
    w_mean = w_mean_full[X_SLICE, Y_SLICE]

    if not INTERMITTENCY_PATH.exists():
        raise FileNotFoundError(f"Intermittency file not found: {INTERMITTENCY_PATH}")
    intermittency_full = np.load(INTERMITTENCY_PATH)
    intermittency = intermittency_full[X_SLICE, Y_SLICE]
    plume_mask = intermittency >= PLUME_ENVELOPE_CUTOFF
    inside_px = int(np.count_nonzero(plume_mask))
    total_px = int(plume_mask.size)
    log_lines.append(f"Plume mask pixels: inside={inside_px}, outside={total_px - inside_px}")
    if plume_mask.shape != u.shape[:2]:
        raise ValueError(f"Intermittency mask shape {plume_mask.shape} does not match velocity fields {u.shape[:2]}")

    # Fluctuating components
    u_flx, v_flx, w_flx = compute_fluctuating_components(u, v, w, u_mean, v_mean, w_mean)
    flx_dir = BASE_PATH / "flow_properties" / "flx_u_v_w"
    save_arrays(
        [
            (flx_dir / "u_flx.npy", u_flx),
            (flx_dir / "v_flx.npy", v_flx),
            (flx_dir / "w_flx.npy", w_flx),
        ]
    )
    print(f"Saved fluctuating components to {flx_dir}")

    # Strain rates from fluctuations
    duflx_dx, dvflx_dy, dwflx_dz = compute_fluctuating_strain_rates(
        u_flx, v_flx, dx=DX, dt=DT
    )
    strain_dir = BASE_PATH / "flow_properties" / "flx_StrainRates"
    save_arrays(
        [
            (strain_dir / f"duflx_dx_{CASE_NAME}.npy", duflx_dx),
            (strain_dir / f"dvflx_dy_{CASE_NAME}.npy", dvflx_dy),
            (strain_dir / f"dwflx_dz_{CASE_NAME}.npy", dwflx_dz),
        ]
    )
    print(f"Saved fluctuating strain rates to {strain_dir}")

    # Viscous dissipation
    epsilon = compute_viscous_dissipation(duflx_dx, dvflx_dy, dwflx_dz, nu=NU)
    eps_path = BASE_PATH / "flow_properties" / "Plots"/ f"{CASE_NAME}" /f"epsilon_{CASE_NAME}.npy"
    save_arrays([(eps_path, epsilon)])
    _plot_field(epsilon, "epsilon", eps_path.with_suffix(".png"), vmin=np.percentile(epsilon, 1), vmax=np.percentile(epsilon, 90), cmap=CMAP)
    print(f"Saved viscous dissipation to {eps_path}")
    _report_stats("epsilon", epsilon, log=log_lines)
    _report_region_stats("epsilon", epsilon, plume_mask, log=log_lines)

    # Taylor scales and Reynolds number
    Taylor_microscale, kolmogorov_length_scale, kolmogorov_time_scale, Taylor_Re = compute_taylor_scales(
        u_flx, v_flx, w_flx, epsilon, nu=NU
    )

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
    tke, u_mnsq, v_mnsq, w_mnsq = compute_turbulent_kinetic_energy(u_flx, v_flx, w_flx)
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
        f"anisotropy ratios: <u'^2>/TKE={np.mean(u_mnsq/tke)}: <v'^2>/TKE={np.mean(v_mnsq/tke)}, <w'^2>/TKE={np.mean(w_mnsq/tke)})"
    )
    print(anisotropy_line)
    log_lines.append(anisotropy_line)

    t_intensity_avg = compute_turbulence_intensity(u_flx, v_flx, w_flx, u_mean=0.30)
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
