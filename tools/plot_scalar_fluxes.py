"""
Compute and plot turbulent scalar fluxes c'u', c'v', and c'w' for a given case.

Fluxes are computed as time averages of the products between fluctuating concentration
and fluctuating velocity components. Velocity fluctuations are loaded from
E:/sPIV_PLIF_ProcessedData/flow_properties/flx_u_v_w/*_flx_{CASE}_FINAL_AllTimeSteps.npy.
Concentration fluctuations come from the PLIF stack.

The plot shows a quiver of <c'u'> and <c'v'> with shading by |<c'v'>,<c'u'>|.

Run:
    python tools/plot_scalar_fluxes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.analysis.flow_properties import load_mean_velocity_components

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASE_NAME = "baseline" 
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CONCENTRATION_PATH = BASE_PATH / "PLIF" / f"plif_{CASE_NAME}_smoothed.npy"
U_PATH = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_u.npy"  # used only for total fluxes, not turbulent fluxes
V_PATH = BASE_PATH / "PIV" / f"piv_{CASE_NAME}_v.npy"  # used only for total fluxes, not turbulent fluxes
FLX_DIR = BASE_PATH / "flow_properties" / "flx_u_v_w"
X_SLICE = slice(0, 600)
CONC_Y_SLICE = slice(100, 500)  # apply to concentration to align with velocity trim
VEL_Y_SLICE = slice(None)  # velocity fluctuations already trimmed in y; keep all
T_SLICE = slice(0, 6000)
QUIVER_STEP = 15  # subsample for quiver display
OUT_DIR = BASE_PATH / "flow_properties" / "scalar_fluxes"
MEAN_X_SLICE = X_SLICE
MEAN_Y_SLICE = CONC_Y_SLICE
XLABEL = "x index"
YLABEL = "y index"
MAG_CMAP = cmr.get_sub_cmap(cmr.rainforest_r, 0, 0.6)
TURB_QUIVER_MAG_THRESHOLD = 0.00005  # turbulent flux quiver threshold
ADV_QUIVER_MAG_THRESHOLD = 0.001  # advective flux quiver threshold
TOTAL_QUIVER_MAG_THRESHOLD = 0.0012 # total flux quiver threshold
QUIVER_LENGTH = 16.0  # constant arrow length in axis units
SHOW_TURB_MAG_OVERLAY = True  # show |<c'v'>,<c'u'>| background for turbulent fluxes
#MAG_VMIN = 0.0001     # magnitude color lower bound; None auto-scales
MAG_VMIN = -0.005
MAG_VMAX = 0.005     # magnitude color upper bound; None auto-scales
SHOW_ADV_MAG_OVERLAY = True  # show |<v><c>, <u><c>| background for advective fluxes
# ADV_MAG_VMIN = 0.001  # set None to auto-scale for advective magnitude
ADV_MAG_VMIN = -0.1
ADV_MAG_VMAX = 0.1  # set None to auto-scale for advective magnitude
DIFFUSIVE_FILL_NANS = True  # fill NaNs via nearest-neighbor interpolation for diffusive case
FIG_DPI = 600
COMPUTE_FLUXES = False  # set to False to skip computation and only plot existing data
PLOT_TURBULENT = False  # plot turbulent scalar fluxes c'u', c'v', c'w'
PLOT_ADVECTIVE = False  # plot mean-velocity * mean-concentration fluxes
SAVE_ADVECTIVE = True  # save advective flux arrays to disk
PLOT_TOTAL = True  # plot total fluxes <u c>, <v c>, <w c>
SAVE_TOTAL = False  # save total flux arrays to disk
SHOW_TOTAL_MAG_OVERLAY = True  # show |<v c>, <u c>| background for total fluxes
TOTAL_MAG_VMIN = 0.001  # set None to auto-scale for total magnitude
TOTAL_MAG_VMAX = 0.01  # set None to auto-scale for total magnitude
PLOT_TOTAL_COMPONENTS = True  # plot total u-flux and v-flux separately
TOTAL_COMPONENT_CMAP = "RdBu_r"
TOTAL_COMPONENT_VMIN = None  # fallback only when per-component limits are not passed
TOTAL_COMPONENT_VMAX = None  # fallback only when per-component limits are not passed
TOTAL_COMPONENT_QUIVER_THRESHOLD = 0.0012  # fallback only when per-component threshold is not passed
TOTAL_COMPONENT_LOG_SCALE = True  # use signed-log color scaling for component overlays
TOTAL_COMPONENT_LOG_LINTHRESH = None  # None -> auto from quiver threshold
# -------------------------------------------------------------------


def _plot_flux_quiver(
    uc_flux: np.ndarray,
    vc_flux: np.ndarray,
    wc_flux: np.ndarray | None = None,
    *,
    title: str,
    fig_path: Path,
    mag_vmin: float | None,
    mag_vmax: float | None,
    show_mag_overlay: bool,
    mag_label: str = "|<c'v'>,<c'u'>|",
    quiver_mag_threshold: float | None = None,
) -> None:
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
    # Normalize vectors so arrows show direction only
    mag_safe = np.maximum(mag_q, 1e-12)
    uc_dir = uc_q / mag_safe
    vc_dir = vc_q / mag_safe
    threshold = TOTAL_QUIVER_MAG_THRESHOLD if quiver_mag_threshold is None else quiver_mag_threshold
    mask = mag_q >= threshold
    uc_dir = np.where(mask, uc_dir, np.nan)
    vc_dir = np.where(mask, vc_dir, np.nan)

    uc_draw = uc_dir * QUIVER_LENGTH
    vc_draw = vc_dir * QUIVER_LENGTH

    mag_full = np.hypot(uc_flux, vc_flux)
    try:
        mag_auto_min = float(np.nanmin(mag_full))
        mag_auto_max = float(np.nanmax(mag_full))
    except ValueError:
        mag_auto_min = 0.0
        mag_auto_max = 1.0
    mag_vmin = mag_vmin if mag_vmin is not None else mag_auto_min
    mag_vmax = mag_vmax if mag_vmax is not None else mag_auto_max
    if mag_vmin is None or mag_vmin <= 0:
        mag_vmin = max(1e-12, mag_auto_min if mag_auto_min > 0 else 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = None
    if show_mag_overlay:
        h = ax.pcolormesh(
            Y_idx,
            X_idx,
            mag_full,
            cmap=MAG_CMAP,
            shading="auto",
            norm=LogNorm(vmin=mag_vmin, vmax=mag_vmax),
        )
    quiv = ax.quiver(
        Xg_q,
        Yg_q,
        uc_draw,
        vc_draw,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        alpha=0.8,
        linewidth=0.3,
        width=0.006,
        headwidth=3,
        headlength=4,
        headaxislength=3,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if h is not None:
        fig.colorbar(h, ax=ax, label=mag_label)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)


def _plot_component_flux_vector_overlay(
    component_flux: np.ndarray,
    *,
    component: str,
    title: str,
    fig_path: Path,
    value_vmin: float | None = None,
    value_vmax: float | None = None,
    quiver_abs_threshold: float | None = None,
) -> None:
    """Plot a single flux component with 1D-oriented vectors overlaid on signed values."""
    x_offset = X_SLICE.start or 0
    y_offset = CONC_Y_SLICE.start or 0
    X_idx = np.arange(component_flux.shape[0]) + x_offset
    Y_idx = np.arange(component_flux.shape[1]) + y_offset
    Xg, Yg = np.meshgrid(Y_idx, X_idx)

    comp_vals = np.asarray(component_flux)
    finite_vals = comp_vals[np.isfinite(comp_vals)]
    if finite_vals.size:
        auto_lim = float(np.nanmax(np.abs(finite_vals)))
    else:
        auto_lim = 1.0
    if auto_lim <= 0:
        auto_lim = 1.0
    if value_vmin is None or value_vmax is None:
        value_vmin = -auto_lim
        value_vmax = auto_lim

    step = QUIVER_STEP
    comp_q = comp_vals[::step, ::step]
    sign_q = np.sign(comp_q)
    threshold = TOTAL_COMPONENT_QUIVER_THRESHOLD if quiver_abs_threshold is None else quiver_abs_threshold
    mask = np.abs(comp_q) >= threshold
    sign_q = np.where(mask, sign_q, np.nan)

    if component.lower() == "u":
        uq = sign_q * QUIVER_LENGTH
        vq = np.zeros_like(uq)
        value_label = "<u c>"
    elif component.lower() == "v":
        uq = np.zeros_like(sign_q)
        vq = sign_q * QUIVER_LENGTH
        value_label = "<v c>"
    else:
        raise ValueError(f"Unsupported component '{component}'. Use 'u' or 'v'.")

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.pcolormesh(
        Y_idx,
        X_idx,
        comp_vals,
        cmap=TOTAL_COMPONENT_CMAP,
        shading="auto",
        norm=(
            SymLogNorm(
                linthresh=(
                    TOTAL_COMPONENT_LOG_LINTHRESH
                    if TOTAL_COMPONENT_LOG_LINTHRESH is not None
                    else max(1e-12, threshold)
                ),
                linscale=1.0,
                base=10,
                vmin=value_vmin,
                vmax=value_vmax,
            )
            if TOTAL_COMPONENT_LOG_SCALE
            else TwoSlopeNorm(vmin=value_vmin, vcenter=0.0, vmax=value_vmax)
        ),
    )
    ax.quiver(
        Xg[::step, ::step],
        Yg[::step, ::step],
        uq,
        vq,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        alpha=0.8,
        linewidth=0.3,
        width=0.006,
        headwidth=3,
        headlength=4,
        headaxislength=3,
    )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h, ax=ax, label=value_label)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)


def _fill_nan_nearest(field: np.ndarray) -> np.ndarray:
    mask = np.isnan(field)
    if not np.any(mask):
        return field
    valid = ~mask
    if not np.any(valid):
        return field
    coords = np.column_stack(np.nonzero(valid))
    values = field[valid]
    missing = np.column_stack(np.nonzero(mask))
    filled = field.copy()
    filled[mask] = griddata(coords, values, missing, method="nearest")
    return filled


def _load_concentration_mean() -> np.ndarray:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(f"Concentration stack not found: {CONCENTRATION_PATH}")
    conc = np.load(CONCENTRATION_PATH, mmap_mode="r")
    conc = conc[X_SLICE, CONC_Y_SLICE, T_SLICE]
    return np.mean(conc, axis=2)


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
        # conc_mean = _load_concentration_mean()
        conc = np.load(CONCENTRATION_PATH, mmap_mode="r")[X_SLICE, CONC_Y_SLICE, :]
        conc_mean = np.nanmean(conc, axis=2)
        conc = conc[:, :, T_SLICE]

        conc_flx = conc - conc_mean[:, :, None]

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

    uc_plot = uc_flux
    vc_plot = vc_flux
    wc_plot = wc_flux
    if CASE_NAME.lower() == "diffusive" and DIFFUSIVE_FILL_NANS:
        uc_plot = _fill_nan_nearest(uc_plot)
        vc_plot = _fill_nan_nearest(vc_plot)
        wc_plot = _fill_nan_nearest(wc_plot)

    if PLOT_TURBULENT:
        _plot_flux_quiver(
            uc_plot,
            vc_plot,
            wc_plot,
            title=f"Scalar fluxes: quiver (<c'v'>, <c'u'>) shaded by |<c'v'>,<c'u'>|\ncase={CASE_NAME}",
            fig_path=OUT_DIR / f"scalar_flux_quiver_{CASE_NAME}_t_{T_SLICE}.png",
            mag_vmin=MAG_VMIN,
            mag_vmax=MAG_VMAX,
            show_mag_overlay=SHOW_TURB_MAG_OVERLAY,
            quiver_mag_threshold=TURB_QUIVER_MAG_THRESHOLD,
        )

    if PLOT_ADVECTIVE:
        conc_mean = _load_concentration_mean()
        u_mean_full, v_mean_full, w_mean_full = load_mean_velocity_components(CASE_NAME, base_path=BASE_PATH)
        u_mean = u_mean_full[MEAN_X_SLICE, MEAN_Y_SLICE]
        v_mean = v_mean_full[MEAN_X_SLICE, MEAN_Y_SLICE]
        w_mean = w_mean_full[MEAN_X_SLICE, MEAN_Y_SLICE]
        if u_mean.shape != conc_mean.shape:
            raise ValueError(
                f"Mean velocity shape {u_mean.shape} does not match concentration mean {conc_mean.shape}. "
                f"Check MEAN_X_SLICE/MEAN_Y_SLICE vs CONC_Y_SLICE."
            )

        uc_adv = u_mean * conc_mean
        vc_adv = v_mean * conc_mean
        wc_adv = w_mean * conc_mean

        if SAVE_ADVECTIVE:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.save(OUT_DIR / f"uc_advective_{CASE_NAME}.npy", uc_adv)
            np.save(OUT_DIR / f"vc_advective_{CASE_NAME}.npy", vc_adv)
            np.save(OUT_DIR / f"wc_advective_{CASE_NAME}.npy", wc_adv)

        uc_adv_plot = uc_adv
        vc_adv_plot = vc_adv
        wc_adv_plot = wc_adv
        if CASE_NAME.lower() == "diffusive" and DIFFUSIVE_FILL_NANS:
            uc_adv_plot = _fill_nan_nearest(uc_adv_plot)
            vc_adv_plot = _fill_nan_nearest(vc_adv_plot)
            wc_adv_plot = _fill_nan_nearest(wc_adv_plot)

        _plot_flux_quiver(
            uc_adv_plot,
            vc_adv_plot,
            wc_adv_plot,
            title=f"Advective fluxes: quiver (<v><c>, <u><c>) shaded by |<v><c>, <u><c>|\ncase={CASE_NAME}",
            fig_path=OUT_DIR / f"advective_flux_quiver_{CASE_NAME}.png",
            mag_vmin=ADV_MAG_VMIN,
            mag_vmax=ADV_MAG_VMAX,
            show_mag_overlay=SHOW_ADV_MAG_OVERLAY,
            mag_label="|<v><c>, <u><c>|",
            quiver_mag_threshold=ADV_QUIVER_MAG_THRESHOLD,
        )

    if PLOT_TOTAL:
        if not COMPUTE_FLUXES:
            uc_total = np.load(OUT_DIR / f"uc_total_{CASE_NAME}.npy", mmap_mode="r")
            vc_total = np.load(OUT_DIR / f"vc_total_{CASE_NAME}.npy", mmap_mode="r")
        else:
            conc = np.load(CONCENTRATION_PATH, mmap_mode="r")
            conc = conc[X_SLICE, CONC_Y_SLICE, T_SLICE]
            u = np.load(U_PATH, mmap_mode="r")
            v = np.load(V_PATH, mmap_mode="r")

            u_total = u[X_SLICE, CONC_Y_SLICE, T_SLICE]
            v_total = v[X_SLICE, CONC_Y_SLICE, T_SLICE]

            uc_total = np.nanmean(u_total * conc, axis=2)
            vc_total = np.nanmean(v_total * conc, axis=2)

            if SAVE_TOTAL:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                np.save(OUT_DIR / f"uc_total_{CASE_NAME}.npy", uc_total)
                np.save(OUT_DIR / f"vc_total_{CASE_NAME}.npy", vc_total)

        uc_total_plot = uc_total
        vc_total_plot = vc_total

        if CASE_NAME.lower() == "diffusive" and DIFFUSIVE_FILL_NANS:
            uc_total_plot = _fill_nan_nearest(uc_total_plot)
            vc_total_plot = _fill_nan_nearest(vc_total_plot)

        _plot_flux_quiver(
            uc_total_plot,
            vc_total_plot,
            title=f"Total fluxes: quiver (<v c>, <u c>) shaded by |<v c>, <u c>|\ncase={CASE_NAME}",
            fig_path=OUT_DIR / f"total_flux_quiver_{CASE_NAME}_t_{T_SLICE}.png",
            mag_vmin=TOTAL_MAG_VMIN,
            mag_vmax=TOTAL_MAG_VMAX,
            show_mag_overlay=SHOW_TOTAL_MAG_OVERLAY,
            mag_label="|<v c>, <u c>|",
            quiver_mag_threshold=TOTAL_QUIVER_MAG_THRESHOLD,
        )

        if PLOT_TOTAL_COMPONENTS:
            _plot_component_flux_vector_overlay(
                uc_total_plot,
                component="u",
                title=f"Total u flux: vector direction overlaid on <u c>\ncase={CASE_NAME}",
                fig_path=OUT_DIR / f"total_u_flux_vectors_{CASE_NAME}_t_{T_SLICE}.png",
                value_vmin=MAG_VMIN,
                value_vmax=MAG_VMAX,
                quiver_abs_threshold=TURB_QUIVER_MAG_THRESHOLD,
            )
            _plot_component_flux_vector_overlay(
                vc_total_plot,
                component="v",
                title=f"Total v flux: vector direction overlaid on <v c>\ncase={CASE_NAME}",
                fig_path=OUT_DIR / f"total_v_flux_vectors_{CASE_NAME}_t_{T_SLICE}.png",
                value_vmin=ADV_MAG_VMIN,
                value_vmax=ADV_MAG_VMAX,
                quiver_abs_threshold=ADV_QUIVER_MAG_THRESHOLD,
            )


    print(f"Saved scalar flux arrays to {OUT_DIR}")


if __name__ == "__main__":
    main()
