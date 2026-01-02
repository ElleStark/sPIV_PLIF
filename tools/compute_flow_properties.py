"""
Compute flow properties using the helpers in analysis.flow_properties.

Edit the paths/settings below, then run:
    python tools/compute_flow_properties.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
CASE_NAME = "baseline" 
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
X_SLICE = slice(100, 200)
Y_SLICE = slice(0, 500)
T_SLICE = slice(0, 200)
DX = 0.0005  # m
DT = 0.05  # sec
NU = 1.5e-5  # kinematic viscosity, m2/s


def main() -> None:
    # Load inputs
    u, v, w = load_velocity_components(CASE_NAME, base_path=BASE_PATH, x_slice=X_SLICE, y_slice=Y_SLICE, t_slice=T_SLICE)
    u_mean_full, v_mean_full, w_mean_full = load_mean_velocity_components(CASE_NAME, base_path=BASE_PATH)
    # Align mean fields to the spatial slice used for instantaneous data
    u_mean = u_mean_full[X_SLICE, Y_SLICE]
    v_mean = v_mean_full[X_SLICE, Y_SLICE]
    w_mean = w_mean_full[X_SLICE, Y_SLICE]

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
    eps_path = BASE_PATH / "flow_properties" / f"epsilon_{CASE_NAME}.npy"
    save_arrays([(eps_path, epsilon)])
    print(f"Saved viscous dissipation to {eps_path}")
    print(f"Average epsilon: {float(np.mean(epsilon))}")

    # Taylor scales and Reynolds number
    Taylor_microscale, kolmogorov_length_scale, kolmogorov_time_scale, Taylor_Re = compute_taylor_scales(
        u_flx, v_flx, w_flx, epsilon, nu=NU
    )
    save_arrays(
        [
            (BASE_PATH / "flow_properties" / f"Taylor_microscale_{CASE_NAME}.npy", Taylor_microscale),
            (BASE_PATH / "flow_properties" / f"Taylor_Re_{CASE_NAME}.npy", Taylor_Re),
            (BASE_PATH / "flow_properties" / f"kolmogorov_length_scale_{CASE_NAME}.npy", kolmogorov_length_scale),
            (BASE_PATH / "flow_properties" / f"kolmogorov_time_scale_{CASE_NAME}.npy", kolmogorov_time_scale),
        ]
    )
    print(f"Average Taylor microscale: {float(np.mean(Taylor_microscale))}")
    print(f"Average Taylor Re: {float(np.mean(Taylor_Re))}")
    print(f"Average Kolmogorov length scale: {float(np.mean(kolmogorov_length_scale))}")

    # Turbulent kinetic energy
    tke, u_mnsq, v_mnsq, w_mnsq = compute_turbulent_kinetic_energy(u_flx, v_flx, w_flx)
    tke_path = BASE_PATH / "flow_properties" / f"tke_{CASE_NAME}.npy"
    save_arrays([(tke_path, tke)])
    print(f"Saved TKE to {tke_path}")
    print(f"anisotropy ratios: <u'^2>/TKE={np.mean(u_mnsq/tke)}: <v'^2>/TKE={np.mean(v_mnsq/tke)}, <w'^2>/TKE={np.mean(w_mnsq/tke)})")

    # Turbulence intensity (requires RMS fields on disk)
    u_rms_full = np.load(BASE_PATH / "rms_fields" / f"{CASE_NAME}_u_rms.npy")
    v_rms_full = np.load(BASE_PATH / "rms_fields" / f"{CASE_NAME}_v_rms.npy")
    w_rms_full = np.load(BASE_PATH / "rms_fields" / f"{CASE_NAME}_w_rms.npy")
    # Align RMS and mean fields to the same spatial subset
    u_rms = u_rms_full[X_SLICE, Y_SLICE]
    v_rms = v_rms_full[X_SLICE, Y_SLICE]
    w_rms = w_rms_full[X_SLICE, Y_SLICE]
    t_intensity_avg = compute_turbulence_intensity(u_rms, v_rms, w_rms, u_mean)
    t_intensity_path = BASE_PATH / "flow_properties" / f"turbulence_intensity_{CASE_NAME}.npy"
    save_arrays([(t_intensity_path, t_intensity_avg)])
    print(f"Saved turbulence intensity to {t_intensity_path}")


if __name__ == "__main__":
    main()
