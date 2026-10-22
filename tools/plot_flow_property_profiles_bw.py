"""
Black-and-white L(s)-style profiles of TKE, dissipation rate, Kolmogorov (~ Batchelor) scale and Taylor microscale,
one figure per quantity and region, matching tools/plot_ils_profiles_bw.py.

Regions:
    background  grid turbulence: the flow-property background mask (|x| >= 70 mm, outside the plume envelope);
                the flow-property maps only span |x| <= 100 mm, so this is the same pixel set used for the
                epsilon cross-check and the development-region checks
    core        combined source wake + plume: centerline +/- 100 columns (373-573), as for L_core_bw.png
Profiles are row medians within the region, smoothed along s.

Inputs: flow_properties/Plots/<case>/{tke, epsilon, kolmogorov_length_scale, Taylor_microscale}_<case>.npy (SI units)
        and epsilon_estimates_<case>.npz (background mask)
Outputs (OUT_DIR): {tke, epsilon, kolmogorov, taylor_microscale}_{background, core}_bw.png

Run from the repo root:
    python -m tools.plot_flow_property_profiles_bw
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.plot_ils_profiles_bw import CASES, COLOR, FIGSIZE, LEGEND_FONTSIZE, LINESTYLES, LINEWIDTH, OUT_DIR, XLABEL, \
    _savefig

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
FP_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Plots")
OLD_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/OLD_STATS/pre_epsilon_fix_Oct2026")  # pre-fix arrays
FP_COL0 = int(50 / (1000 * 0.000317))  # full-grid column of the flow-property maps' first column (Y_SLICE)
CORE_COLS_FULL = (373, 574)            # core columns on the full 947-column grid (same as ils_characteristic_scale.py)
ROWS_TO_AVG = 9                        # running mean along s (~2.9 mm)
S_LIMITS_MM = (0.0, 285.0)             # the last ~15 rows carry image-edge artifacts in the gradient-based fields
SCHMIDT = 1.0                          # Batchelor scale eta_B = eta Sc^-1/2
# also plot the plain mean over the entire cross-stream width of the map (|x| <= 100 mm) -> <name>_fullwidth_mean_bw.png
FULL_WIDTH_MEAN_QUANTITIES = ["epsilon_WRONG_transverse"]
# mean-based (not median), un-smoothed profiles for background, core and the full width
# -> <name>_{background,core,fullwidth}_mean_unsmoothed_bw.png
MEAN_UNSMOOTHED_QUANTITIES = ["tke", "epsilon", "epsilon_step4", "epsilon_WRONG_transverse"]
# mean-unsmoothed plots of the quantities in each group share one y-range per region (covering all of them)
SHARED_YLIM_GROUPS = [("tke", "epsilon_step4")]
YLIM_LOG_MARGIN = 1.15  # factor beyond the data min/max

# Reference power law k = C (s + x0)^-n drawn as a red dashed line on an extra copy of the background mean
# (un-smoothed) TKE plot -> tke_background_mean_unsmoothed_powerlaw_bw.png. C is fitted (log space, n fixed) to the
# POWERLAW_FIT_CASES over POWERLAW_FIT_S_MM, then multiplied by POWERLAW_OFFSET so the line sits just above the data.
POWERLAW_N = 2.2                    # k decay exponent (development checks: n = 2.1-2.3 for s <= 150 mm)
POWERLAW_X0_MM = 150.0              # grid-to-domain distance; x_g = s + x0
POWERLAW_FIT_CASES = ["baseline", "buoyant", "diffusive", "smSource"]
POWERLAW_FIT_S_MM = (3.0, 150.0)
POWERLAW_OFFSET = 1.5
POWERLAW_DRAW_S_MM = (0.0, 150.0)   # s range over which the line is drawn (= the fit range)

# Outer-band TKE over the full image width (+/-150 mm), from the scalar-flux streaming moments (u, v, w rms;
# samples with w' < -1 m/s rejected). The flow-property maps only reach |x| = 100 mm, which for the fractal grid lies
# inside the wakes of its largest bars (L0 = 160 mm square -> bars at |x| = 80 +/- 8.75 mm if centred on the source).
#   outerband   mean over 70 <= |x| <= 145 mm, outside the plume envelope (intermittency >= 0.10)
#   betweenbar  outer-band columns whose near-grid k (mean over s <= 15 mm) is in the lowest quartile
#   behindbar   the remaining outer-band columns
# -> tke_{outerband,betweenbar,behindbar}_mean_unsmoothed_bw.png (shared y-range)
MOMENTS_TEMPLATE = "E:/sPIV_PLIF_ProcessedData/scalar_flux_diffusivity/{case}/arrays/moments.npz"
INTERMITTENCY_TEMPLATE = "E:/sPIV_PLIF_ProcessedData/Plots/Intermittency/intermittency_{case}.npy"
OUTER_BAND_ABS_X_MM = (70.0, 145.0)  # last ~5 mm at the image edge excluded
NEAR_GRID_S_MM = 15.0
BETWEEN_BAR_QUANTILE = 25.0

# Between-bar profiles of the length scales and dissipation rate (same columns as the between-bar TKE):
#   L from the ILS maps (full width); epsilon, eta, lambda from tools/compute_outer_band_flow_properties.py
# -> <name>_betweenbar_mean_unsmoothed_bw.png
ILS_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Integral_length_scales/ILS")
OUTER_FP_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Outer_bands")
OUTER_FP_KEYS = {  # plot name: (key in the outer-band npz, scale to plot units)
    "epsilon": ("eps_hybrid_step1", 1.0),
    "epsilon_step4": ("eps_hybrid_step4", 1.0),
    "kolmogorov": ("kolmogorov_length", 1000.0),
    "taylor_microscale": ("taylor_microscale", 1000.0),
}
# plot-only quantities (labels; not loaded by the main loop)
EXTRA_QUANTITIES = {
    "ils_L": (None, 1.0, r"$L = \frac{1}{2}(L_{u,x} + L_{v,y})$ (mm)", False),
}

# name: (source, scale to plot units, y label, log y-axis); source is a .npy file stem, or "npz:<key>" for an array in
# epsilon_estimates_<case>.npz (eps_hybrid / eps_iso_long at central-difference steps 1, 2, 4)
QUANTITIES = {
    "tke": ("tke", 1.0, r"$k$ (m$^2$ s$^{-2}$)", True),
    "epsilon": ("epsilon", 1.0, r"$\varepsilon$ (m$^2$ s$^{-3}$)", True),
    "epsilon_step4": ("npz:eps_hybrid_step4", 1.0, r"$\varepsilon$, 2.54 mm gradient span (m$^2$ s$^{-3}$)", True),
    "epsilon_iso_long": ("npz:eps_iso_long_step1", 1.0, r"$\varepsilon_{iso} = 5\nu\,\sum_i \langle (\partial u_i/\partial x_i)^2 \rangle$ (m$^2$ s$^{-3}$)", True),
    # check only: the original (incorrect) estimate, 5 nu [<(du/dy)^2> + <(dv/dx)^2> + <(du/dy + dv/dx)^2>], i.e.
    # transverse gradients in place of longitudinal ones (pre-fix arrays backed up in OLD_DIR)
    "epsilon_WRONG_transverse": ("old:epsilon", 1.0, r"$\varepsilon$, transverse gradients (WRONG) (m$^2$ s$^{-3}$)", True),
    "kolmogorov": ("kolmogorov_length_scale", 1000.0, r"$\eta \approx \eta_B$ (mm)", False),
    "taylor_microscale": ("Taylor_microscale", 1000.0, r"$\lambda$ (mm)", False),
}


def _smooth(p: np.ndarray, k: int) -> np.ndarray:
    fin = np.isfinite(p)
    num = np.convolve(np.where(fin, p, 0.0), np.ones(k), mode="same")
    den = np.convolve(fin.astype(float), np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _row_median(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(mask & np.isfinite(arr), arr, np.nan), axis=1)


def region_masks(case: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    est = np.load(FP_DIR / case / f"epsilon_estimates_{case}.npz")
    bg = est["background_mask"]
    ny, nx = bg.shape
    core = np.zeros_like(bg)
    core[:, CORE_COLS_FULL[0] - FP_COL0:CORE_COLS_FULL[1] - FP_COL0] = True
    return est["s_mm"], {"background": bg, "core": core}


def plot_profile(profiles: dict, quantity: str, region: str, ylim: tuple[float, float] | None = None,
                 reference=None, suffix: str = "") -> tuple:
    _, _, ylabel, logy = QUANTITIES[quantity] if quantity in QUANTITIES else EXTRA_QUANTITIES[quantity]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for case in CASES:
        if (quantity, region) not in profiles[case]:  # case not available for this quantity
            continue
        s, p = profiles[case]["s_mm"], profiles[case][(quantity, region)]
        keep = (s >= S_LIMITS_MM[0]) & (s <= S_LIMITS_MM[1])
        ax.plot(s[keep], p[keep], color=COLOR, ls=LINESTYLES[case], lw=LINEWIDTH, label=case)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(*S_LIMITS_MM)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    if reference is not None:
        reference(ax)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=3, frameon=False, handlelength=4.0, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    used_ylim = ax.get_ylim()
    _savefig(fig, OUT_DIR / f"{quantity}_{region}{suffix}_bw.png")
    return used_ylim


def powerlaw_reference(profiles: dict, key: str):
    """Fit C in k = C (s + x0)^-n (n fixed) to the fit cases and return a function that draws the offset line."""
    n, x0 = POWERLAW_N, POWERLAW_X0_MM
    logc = []
    for case in POWERLAW_FIT_CASES:
        s, p = profiles[case]["s_mm"], profiles[case][("tke", key)]
        ok = (s >= POWERLAW_FIT_S_MM[0]) & (s <= POWERLAW_FIT_S_MM[1]) & np.isfinite(p) & (p > 0)
        logc.append(np.log(p[ok]) + n * np.log(s[ok] + x0))
    C = float(np.exp(np.mean(np.concatenate(logc))))
    print(f"TKE reference: k = {C:.4g} (s + {x0:g})^-{n:g} (s in mm), drawn x{POWERLAW_OFFSET:g}")

    def draw(ax):
        s_ref = np.linspace(*POWERLAW_DRAW_S_MM, 300)
        ax.plot(s_ref, POWERLAW_OFFSET * C * (s_ref + x0) ** -n, color="red", ls="--", lw=LINEWIDTH, zorder=0,
                label=rf"$\propto (s + {x0:g}\,\mathrm{{mm}})^{{-{n:g}}}$")
    return draw


def outer_band_tke_profiles() -> dict:
    """Mean-unsmoothed k(s) over the outer band and its between-bar / behind-bar column subsets, per case."""
    out = {}
    for case in CASES:
        m = np.load(MOMENTS_TEMPLATE.format(case=case))
        x = m["x_mm"]
        s = m["y_mm"].max() - m["y_mm"]
        k = 0.5 * (m["u_rms"] ** 2 + m["v_rms"] ** 2 + m["w_rms"] ** 2)
        k = np.where(np.load(INTERMITTENCY_TEMPLATE.format(case=case)) >= 0.10, np.nan, k)  # plume envelope out
        band = (np.abs(x) >= OUTER_BAND_ABS_X_MM[0]) & (np.abs(x) <= OUTER_BAND_ABS_X_MM[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            k_near = np.nanmean(k[s <= NEAR_GRID_S_MM], axis=0)
            low = band & (k_near <= np.nanpercentile(k_near[band], BETWEEN_BAR_QUANTILE))
            high = band & ~low
            out[case] = {"s_mm": s, "cols": {"outerband": band, "betweenbar": low, "behindbar": high}}
            for name, cols in out[case]["cols"].items():
                out[case][("tke", f"{name}_mean_unsmoothed")] = np.nanmean(k[:, cols], axis=1)
        sel = np.abs(x[low])
        print(f"{case}: between-bar columns {low.sum()} of {band.sum()}, |x| in [{sel.min():.0f}, {sel.max():.0f}] mm")
    return out


def _outer_fp_complete(case: str) -> bool:
    """Outer-band flow properties exist for this case and used every frame (partial/test files are ignored)."""
    path = OUTER_FP_DIR / f"outer_band_flow_properties_{case}.npz"
    if not path.exists():
        return False
    with np.load(path) as d:
        return "n_frames_used" in d and int(d["n_frames_used"]) == int(d["n_frames_total"])


def between_bar_scale_profiles(outer: dict) -> tuple[dict, list[str]]:
    """Mean-unsmoothed between-bar profiles of L (ILS maps, full width) and, where the outer-band flow properties
    have been computed, epsilon (spans 0.63 and 2.54 mm), eta ~ eta_B and lambda. Columns as in outer_band_tke_profiles."""
    out, available = {}, ["ils_L"]
    fp_cases = [c for c in CASES if _outer_fp_complete(c)]
    if fp_cases:
        available += [q for q in OUTER_FP_KEYS]
    for case in CASES:
        have_fp = case in fp_cases
        cols = outer[case]["cols"]["betweenbar"]
        out[case] = {"s_mm": outer[case]["s_mm"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lux = np.load(ILS_DIR / f"ILS_{case}_u_cross_stream.npy").astype(float) * 1000
            lvy = np.load(ILS_DIR / f"ILS_{case}_v_streamwise.npy").astype(float) * 1000
            out[case][("ils_L", "betweenbar_mean_unsmoothed")] = np.nanmean(0.5 * (lux + lvy)[:, cols], axis=1)
            if have_fp:
                d = np.load(OUTER_FP_DIR / f"outer_band_flow_properties_{case}.npz")
                full = {}
                for q, (key, scale) in OUTER_FP_KEYS.items():
                    arr = np.full(lux.shape, np.nan)
                    for side in ("left", "right"):
                        arr[:, d[f"{side}|cols"]] = d[f"{side}|{key}"] * scale
                    full[q] = arr
                for q, arr in full.items():
                    out[case][(q, "betweenbar_mean_unsmoothed")] = np.nanmean(arr[:, cols], axis=1)
    return out, available


def main() -> None:
    profiles = {}
    for case in CASES:
        s, masks = region_masks(case)
        profiles[case] = {"s_mm": s}
        for q, (stem, scale, _, _) in QUANTITIES.items():
            if stem.startswith("old:"):
                arr = np.load(OLD_DIR / case / f"{stem[4:]}_{case}.npy").astype(float) * scale
            elif stem.startswith("npz:"):
                arr = np.load(FP_DIR / case / f"epsilon_estimates_{case}.npz")[stem[4:]].astype(float) * scale
            else:
                arr = np.load(FP_DIR / case / f"{stem}_{case}.npy").astype(float) * scale
            if q == "kolmogorov":
                arr = arr / np.sqrt(SCHMIDT)
            for region, mask in masks.items():
                profiles[case][(q, region)] = _smooth(_row_median(arr, mask), ROWS_TO_AVG)
            if q in FULL_WIDTH_MEAN_QUANTITIES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    profiles[case][(q, "fullwidth_mean")] = _smooth(np.nanmean(arr, axis=1), ROWS_TO_AVG)
            if q in MEAN_UNSMOOTHED_QUANTITIES:
                full = dict(masks, fullwidth=np.ones_like(masks["background"]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for region, mask in full.items():
                        profiles[case][(q, f"{region}_mean_unsmoothed")] = np.nanmean(
                            np.where(mask & np.isfinite(arr), arr, np.nan), axis=1)
    shared = {}
    for group in SHARED_YLIM_GROUPS:
        for region in ("background", "core", "fullwidth"):
            key = f"{region}_mean_unsmoothed"
            vals = []
            for q in group:
                for case in CASES:
                    s, p = profiles[case]["s_mm"], profiles[case][(q, key)]
                    keep = (s >= S_LIMITS_MM[0]) & (s <= S_LIMITS_MM[1]) & np.isfinite(p) & (p > 0)
                    vals.append(p[keep])
            v = np.concatenate(vals)
            for q in group:
                shared[(q, key)] = (v.min() / YLIM_LOG_MARGIN, v.max() * YLIM_LOG_MARGIN)
    for q in QUANTITIES:
        for region in ("background", "core"):
            plot_profile(profiles, q, region)
        if q in FULL_WIDTH_MEAN_QUANTITIES:
            plot_profile(profiles, q, "fullwidth_mean")
        if q in MEAN_UNSMOOTHED_QUANTITIES:
            for region in ("background", "core", "fullwidth"):
                key = f"{region}_mean_unsmoothed"
                plot_profile(profiles, q, key, ylim=shared.get((q, key)))
    key = "background_mean_unsmoothed"
    plot_profile(profiles, "tke", key, ylim=shared.get(("tke", key)), reference=powerlaw_reference(profiles, key),
                 suffix="_powerlaw")

    outer = outer_band_tke_profiles()
    keys = [f"{n}_mean_unsmoothed" for n in ("outerband", "betweenbar", "behindbar")]
    vals = []
    for case in CASES:
        s = outer[case]["s_mm"]
        for key in keys:
            p = outer[case][("tke", key)]
            ok = (s >= S_LIMITS_MM[0]) & (s <= S_LIMITS_MM[1]) & np.isfinite(p) & (p > 0)
            vals.append(p[ok])
    v = np.concatenate(vals)
    for key in keys:
        plot_profile(outer, "tke", key, ylim=(v.min() / YLIM_LOG_MARGIN, v.max() * YLIM_LOG_MARGIN))

    scales, available = between_bar_scale_profiles(outer)
    fp_cases = [c for c in CASES if _outer_fp_complete(c)]
    for q in available:
        # epsilon/eta/lambda plots show only the cases with complete outer-band results; the file name lists them
        # until all cases are available
        partial = q in OUTER_FP_KEYS and len(fp_cases) < len(CASES)
        plot_profile(scales, q, "betweenbar_mean_unsmoothed", suffix="_" + "_".join(fp_cases) if partial else "")
    if not fp_cases:
        print("No complete outer-band flow properties: only the between-bar L plot was made "
              "(run tools/compute_outer_band_flow_properties.py for epsilon, eta, lambda)")


if __name__ == "__main__":
    main()
