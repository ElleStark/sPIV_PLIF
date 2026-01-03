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

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "fractal"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
FLX_DIR = BASE_PATH / "flow_properties" / "flx_u_v_w"
X_SLICE = slice(0, 600)
CONC_Y_SLICE = slice(100, 500)  # apply to concentration to align with velocity trim
VEL_Y_SLICE = slice(None)  # velocity fluctuations already trimmed in y; keep all
T_SLICE = slice(0, 6000)
QUIVER_STEP = 4  # subsample for quiver display
OUT_DIR = BASE_PATH / "flow_properties" / "scalar_fluxes"
XLABEL = "y index"
YLABEL = "x index"
CMAP = cmr.prinsenvlag
FIG_DPI = 600
# -------------------------------------------------------------------


def main() -> None:
    # Load fluctuating velocity components (already trimmed in y)
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

    vmin = -0.00015
    vmax = 0.00015

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(Y_idx, X_idx, wc_flux, cmap=CMAP, shading="auto", vmin=vmin, vmax=vmax)
    ax.quiver(
        Xg_q,
        Yg_q,
        vc_q,
        uc_q,
        angles="xy",
        scale_units="xy",
        scale=None,
        color="k",
        linewidth=0.6,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Scalar fluxes: quiver (<c'v'>, <c'u'>) shaded by <c'w'>\ncase={CASE_NAME}")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h, ax=ax, label="⟨c'w'⟩")
    fig.tight_layout()

    fig_path = OUT_DIR / f"scalar_flux_quiver_{CASE_NAME}.png"
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Saved scalar flux arrays to {OUT_DIR}")
    print(f"Saved scalar flux plot to {fig_path}")


if __name__ == "__main__":
    main()
