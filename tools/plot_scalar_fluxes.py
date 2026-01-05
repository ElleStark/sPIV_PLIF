"""
Compute and plot turbulent scalar fluxes c'u', c'v', and c'w' for a given case.

Fluxes are computed as time averages of the products between fluctuating concentration
and fluctuating velocity components. Velocity fluctuations are loaded from
E:/sPIV_PLIF_ProcessedData/flow_properties/flx_u_v_w/*_flx_{CASE}_FINAL_AllTimeSteps.npy.
Concentration fluctuations come from the PLIF stack.

The plot shows a quiver of <c'u'> and <c'v'> with shading by <c'w'>.

Run:
    python tools/plot_scalar_fluxes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "diffusive"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
FLX_DIR = BASE_PATH / "flow_properties" / "flx_u_v_w"
X_SLICE = slice(0, 600)
CONC_Y_SLICE = slice(100, 500)  # apply to concentration to align with velocity trim
VEL_Y_SLICE = slice(None)  # velocity fluctuations already trimmed in y; keep all
T_SLICE = slice(0, 6000)
QUIVER_STEP = 8  # subsample for quiver display
OUT_DIR = BASE_PATH / "flow_properties" / "scalar_fluxes"
XLABEL = "y index"
YLABEL = "x index"
CMAP = cmr.prinsenvlag
QUIVER_MAG_THRESHOLD = 0.00015  # skip arrows below this magnitude
WC_VMIN = -0.00015  # set None to auto-scale
WC_VMAX = 0.00015   # set None to auto-scale
MAG_VMIN = 0     # magnitude color lower bound; None auto-scales
MAG_VMAX = 0.005     # magnitude color upper bound; None auto-scales
FIG_DPI = 600
COMPUTE_FLUXES = False  # set to False to skip computation and only plot existing data
# -------------------------------------------------------------------


def main() -> None:
    # Load fluctuating velocity components if need to compute  (already trimmed in y)
    if COMPUTE_FLUXES:
        u_flx = np.load(FLX_DIR / f"u_flx_{CASE_NAME}_FINAL_AllTimeSteps.npy", mmap_mode="r")
        v_flx = np.load(FLX_DIR / f"v_flx_{CASE_NAME}_FINAL_AllTimeSteps.npy", mmap_mode="r")
        w_flx = np.load(FLX_DIR / f"w_flx_{CASE_NAME}_FINAL_AllTimeSteps.npy", mmap_mode="r")
        u_flx = u_flx[X_SLICE, VEL_Y_SLICE, T_SLICE]
        v_flx = v_flx[X_SLICE, VEL_Y_SLICE, T_SLICE]
        w_flx = w_flx[X_SLICE, VEL_Y_SLICE, T_SLICE]

        # Load concentration and compute fluctuations on the same region/time span
        if not CONCENTRATION_PATH.exists():
            raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")
        conc = np.load(CONCENTRATION_PATH, mmap_mode="r")
        conc_mean = np.mean(conc, axis=2, keepdims=True)
        conc_mean = conc_mean[X_SLICE, CONC_Y_SLICE, :]
        conc = conc[X_SLICE, CONC_Y_SLICE, T_SLICE]
        conc_flx = conc - conc_mean

        # Compute time-averaged scalar fluxes
        uc_flux = np.mean(u_flx * conc_flx, axis=2)
        vc_flux = np.mean(v_flx * conc_flx, axis=2)
        wc_flux = np.mean(w_flx * conc_flx, axis=2)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(OUT_DIR / f"uc_flux_{CASE_NAME}.npy", uc_flux)
        np.save(OUT_DIR / f"vc_flux_{CASE_NAME}.npy", vc_flux)
        np.save(OUT_DIR / f"wc_flux_{CASE_NAME}.npy", wc_flux)

    else:
        # Load precomputed fluxes
        uc_flux = np.load(OUT_DIR / f"uc_flux_{CASE_NAME}.npy", mmap_mode="r")
        vc_flux = np.load(OUT_DIR / f"vc_flux_{CASE_NAME}.npy", mmap_mode="r")
        wc_flux = np.load(OUT_DIR / f"wc_flux_{CASE_NAME}.npy", mmap_mode="r")

    # Prepare quiver grid (subsample for readability)
    x_offset = X_SLICE.start or 0
    y_offset = CONC_Y_SLICE.start or 0
    X_idx = np.arange(uc_flux.shape[0]) + x_offset
    Y_idx = np.arange(uc_flux.shape[1]) + y_offset
    Xg, Yg = np.meshgrid(Y_idx, X_idx)  # note: pcolormesh uses (y, x)

    step = QUIVER_STEP
    Xg_q = Xg[::step, ::step]
    Yg_q = Yg[::step, ::step]
    uc_q = uc_flux[::step, ::step]
    vc_q = vc_flux[::step, ::step]
    mag_q = np.hypot(uc_q, vc_q)
    # Normalize vectors so arrow length stays short; color encodes magnitude
    mag_safe = np.maximum(mag_q, 1e-12)
    uc_dir = uc_q / mag_safe
    vc_dir = vc_q / mag_safe
    mask = mag_q >= QUIVER_MAG_THRESHOLD
    uc_dir = np.where(mask, uc_dir, np.nan)
    vc_dir = np.where(mask, vc_dir, np.nan)
    mag_plot = np.where(mask, mag_q, np.nan)

    # Log-scale sizing for arrows (applied to direction vectors)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mag = np.log10(mag_plot)
    log_min = np.nanmin(log_mag) if np.isfinite(np.nanmin(log_mag)) else 0.0
    log_max = np.nanmax(log_mag) if np.isfinite(np.nanmax(log_mag)) else 1.0
    log_span = log_max - log_min if log_max > log_min else 1.0
    size_factor = (log_mag - log_min) / log_span
    size_factor = 0.2 + 0.8 * np.clip(size_factor, 0.0, 1.0)  # keep a floor so arrows remain visible
    uc_draw = uc_dir * size_factor
    vc_draw = vc_dir * size_factor

    wc_vmin = WC_VMIN if WC_VMIN is not None else float(np.nanmin(wc_flux))
    wc_vmax = WC_VMAX if WC_VMAX is not None else float(np.nanmax(wc_flux))
    try:
        mag_auto_min = float(np.nanmin(mag_plot))
        mag_auto_max = float(np.nanmax(mag_plot))
    except ValueError:
        mag_auto_min = 0.0
        mag_auto_max = 1.0
    mag_vmin = MAG_VMIN if MAG_VMIN is not None else mag_auto_min
    mag_vmax = MAG_VMAX if MAG_VMAX is not None else mag_auto_max
    mag_norm = Normalize(vmin=mag_vmin, vmax=mag_vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(Y_idx, X_idx, wc_flux, cmap=CMAP, shading="auto", vmin=wc_vmin, vmax=wc_vmax)
    quiv = ax.quiver(
        Xg_q,
        Yg_q,
        vc_draw,
        uc_draw,
        mag_plot,
        angles="xy",
        scale_units="xy",
        scale=0.02,
        cmap=cmr.get_sub_cmap(cmr.neutral_r, 0.15, 1.0),
        norm=mag_norm,
        alpha=0.6,
        linewidth=0.5,
        width=0.005,
        headwidth=3,
        headlength=4,
        headaxislength=3,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Scalar fluxes: quiver (<c'v'>, <c'u'>) shaded by <c'w'>\ncase={CASE_NAME}")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h, ax=ax, label="<c'w'>")
    fig.colorbar(quiv, ax=ax, label="|<c'v'>,<c'u'>|")
    fig.tight_layout()

    fig_path = OUT_DIR / f"scalar_flux_quiver_{CASE_NAME}.png"
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Saved scalar flux arrays to {OUT_DIR}")
    print(f"Saved scalar flux plot to {fig_path}")


if __name__ == "__main__":
    main()
