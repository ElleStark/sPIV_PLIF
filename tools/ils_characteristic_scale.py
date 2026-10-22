"""
Characteristic integral length scale for each experiment, and the dimensionless groups built from it.

Definitions (all from the ILS_mpi_from_npy.py maps, first-zero-crossing integration, shape (y, x), values in m):
    L_u,x  longitudinal scale of the cross-stream velocity u along x      (ILS_<case>_u_cross_stream)
    L_v,y  longitudinal scale of the streamwise velocity v along y (L_11) (ILS_<case>_v_streamwise)
    L(s)   = 1/2 [L_u,x(s) + L_v,y(s)]   region-averaged across x, per downstream station s
    A(s)   = L_u,x / L_v,y               anisotropy ratio (isotropic turbulence: 1)
The longitudinal scales are used because they are the conventional grid-turbulence definition (L_11), L_u,x is the
dispersing scale for cross-stream spreading, and both are insensitive to first-zero vs full-lag integration
(transverse scales differ by up to 2x between the two).

Regions:
    background  ambient grid turbulence: columns 60-199 and 747-886 (the outer 60 columns at each image edge are
                dropped: there the cross-stream autocorrelation has one-sided support and L_u,x is biased)
    core        centerline +/- 100 columns (373-573): source wake + plume turbulence, fixed width (+/- 32 mm)
    plume       |x - x_c(s)| <= sigma_x(s) (plume-relative core; only for cases with plume-width estimates)

Outputs (OUT_DIR):
    L_profiles_background.png / L_profiles_core.png   all cases on one figure: L(s), A(s), fraction never crossing zero
    edge_trim_check.png                               L_u,x background with vs without the edge-column trim
    sigma_over_L.png                                  sigma_x / L_bg and sigma_x / L_u,x,bg for the plume-width cases,
                                                      with the transition sigma_x = L and earlier proxies
    characteristic_scale.npz, characteristic_scale_summary.json

Run from the repo root:
    python tools/ils_characteristic_scale.py
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASES: list[str] = ["baseline", "buoyant", "diffusive", "fractal", "nearbed", "smSource"]
ILS_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Integral_length_scales/ILS")
GRID_DIR = Path("E:/sPIV_PLIF_ProcessedData/PLIF")
SF_DIR = Path("E:/sPIV_PLIF_ProcessedData/scalar_flux_diffusivity")  # plume widths, earlier transition proxies
# comparison_summary.json from `run_scalar_flux_diffusivity --compare ...` (flank-rho and plume-region sigma/L proxies);
# the first existing folder is used
COMPARISON_DIRS = [SF_DIR / "comparison_baseline_buoyant_diffusive_fractal_nearbed_smSource",
                   SF_DIR / "comparison_buoyant_fractal"]
OUT_DIR = ILS_DIR / "Characteristic_scale"

REGIONS: dict[str, list[tuple[int, int]]] = {
    "background": [(60, 200), (747, 887)],
    "core": [(373, 574)],
}
UNTRIMMED_BACKGROUND = [(0, 200), (747, 947)]  # for the edge-trim check only
ROWS_TO_AVG = 9  # running mean along s (~2.9 mm)
NEVER_CROSS_FLAG = 0.5  # shade where the mean never-crossing fraction of L_u,x and L_v,y exceeds this (lower bounds)

# Source diameter for d/L. The source is a half-inch tube; None = unknown (d/L skipped and reported).
SOURCE_D_MM: dict[str, float | None] = {
    "baseline": 12.7, "buoyant": 12.7, "diffusive": 12.7, "fractal": 12.7, "nearbed": 12.7, "smSource": 3.2,
}
NEAR_SOURCE_S_MM = (5.0, 30.0)  # stations averaged for the near-source L in d/L
SIGMA_MAX_VALID_S_MM = 240.0  # plume widths unreliable closer to the image edge (scalar-flux step 6)
DPI = 300
COLORS = dict(zip(CASES, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]))
# -------------------------------------------------------------------


def _savefig(fig, out_path: Path, retries: int = 5) -> None:
    """Save via a temporary file and atomic replace (overwrites on E: intermittently fail with Errno 22)."""
    tmp = out_path.with_name(out_path.stem + "_tmp" + out_path.suffix)
    fig.savefig(tmp, dpi=DPI)
    plt.close(fig)
    for attempt in range(retries):
        try:
            os.replace(tmp, out_path)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    raise OSError(f"Could not replace {out_path}; new figure left at {tmp}")


def _cols(ranges, nx):
    return np.unique(np.concatenate([np.arange(max(0, a), min(nx, b)) for a, b in ranges]))


def _smooth(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return p
    fin = np.isfinite(p)
    num = np.convolve(np.where(fin, p, 0.0), np.ones(k), mode="same")
    den = np.convolve(fin.astype(float), np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _row_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.where(mask, arr, np.nan), axis=1)


def first_crossing_up(s: np.ndarray, p: np.ndarray, level: float) -> float:
    """First s (scanning from the source) where p rises through level, linearly interpolated; NaN if never."""
    o = np.argsort(s)
    s, p = s[o], p[o]
    ok = np.isfinite(p)
    s, p = s[ok], p[ok]
    d = p - level
    hit = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    if hit.size == 0:
        return float("nan")
    i = hit[0]
    return float(s[i] + (s[i + 1] - s[i]) * (-d[i]) / (d[i + 1] - d[i]))


def region_scales(Lux, Lvy, nzu, nzv, mask) -> dict[str, np.ndarray]:
    """Row-averaged L_u,x, L_v,y, their mean, anisotropy and never-crossing fractions within ``mask`` [mm]."""
    lu = _smooth(_row_mean(Lux, mask), ROWS_TO_AVG)
    lv = _smooth(_row_mean(Lvy, mask), ROWS_TO_AVG)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = lu / lv
    fu = _smooth(_row_mean(nzu.astype(float), mask & np.isfinite(Lux)), ROWS_TO_AVG)
    fv = _smooth(_row_mean(nzv.astype(float), mask & np.isfinite(Lvy)), ROWS_TO_AVG)
    return {"L_ux": lu, "L_vy": lv, "L": 0.5 * (lu + lv), "A": A, "nz_ux": fu, "nz_vy": fv}


def plot_all_cases(s_by: dict, R: dict, region: str, title: str, path: Path) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    for case, r in R.items():
        s, col = s_by[case], COLORS[case]
        axs[0].plot(s, r["L"], color=col, lw=1.6, label=case)
        flag = 0.5 * (r["nz_ux"] + r["nz_vy"]) > NEVER_CROSS_FLAG
        axs[0].plot(s, np.where(flag, r["L"], np.nan), color=col, lw=4, alpha=0.3)
        axs[1].plot(s, r["A"], color=col, lw=1.2)
        axs[2].plot(s, r["nz_ux"], color=col, lw=1.0, ls="-")
        axs[2].plot(s, r["nz_vy"], color=col, lw=1.0, ls="--")
    axs[0].set_ylabel(r"$L = \frac{1}{2}(L_{u,x} + L_{v,y})$ [mm]")
    axs[0].set_ylim(bottom=0)
    axs[0].legend(fontsize=8, ncol=3)
    axs[0].set_title(f"{title}\n(thick translucent: on average > {NEVER_CROSS_FLAG:.0%} of pixels never crossed zero; "
                     "values there are field-of-view-limited lower bounds)", fontsize=9)
    axs[1].axhline(1, color="0.5", lw=0.7)
    axs[1].set_ylabel(r"$A = L_{u,x} / L_{v,y}$")
    axs[2].set_ylabel("fraction never\ncrossing zero")
    axs[2].set_ylim(0, 1)
    axs[2].text(0.99, 0.95, r"solid: $L_{u,x}$, dashed: $L_{v,y}$", transform=axs[2].transAxes, ha="right", va="top",
                fontsize=7)
    axs[2].set_xlabel("downstream distance s [mm] (source at s = 0)")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s_by, R_bg, R_core, R_plume, R_untrim, saved = {}, {}, {}, {}, {}, {}
    summary = {"definitions": {"L": "0.5*(L_u,x + L_v,y), first zero crossing", "regions": {k: v for k, v in REGIONS.items()},
                               "rows_to_avg": ROWS_TO_AVG}, "cases": {}}
    sig_by = {}
    for case in CASES:
        ld = lambda name: np.load(ILS_DIR / f"ILS_{case}_{name}.npy").astype(float)  # noqa: E731
        Lux, Lvy = ld("u_cross_stream") * 1000, ld("v_streamwise") * 1000
        nzu = np.isnan(ld("u_cross_stream_r0")) & np.isfinite(Lux)
        nzv = np.isnan(ld("v_streamwise_r0")) & np.isfinite(Lvy)
        ny, nx = Lux.shape
        y = np.load(GRID_DIR / f"{case}_ygrid.npy")[:, 0].astype(float)
        s = y.max() - y
        s_by[case] = s
        col_mask = lambda ranges: np.broadcast_to(np.isin(np.arange(nx), _cols(ranges, nx)), (ny, nx))  # noqa: E731
        R_bg[case] = region_scales(Lux, Lvy, nzu, nzv, col_mask(REGIONS["background"]))
        R_core[case] = region_scales(Lux, Lvy, nzu, nzv, col_mask(REGIONS["core"]))
        R_untrim[case] = region_scales(Lux, Lvy, nzu, nzv, col_mask(UNTRIMMED_BACKGROUND))

        # plume-relative core, and plume widths, where the scalar-flux analysis exists
        dz = SF_DIR / case / "arrays" / "diffusivity.npz"
        st3 = SF_DIR / case / "arrays" / "stage3.npz"
        if dz.exists() and st3.exists():
            D, S3 = np.load(dz), np.load(st3)
            x = np.load(GRID_DIR / f"{case}_xgrid.npy")[0].astype(float)
            mask = np.abs(x[None, :] - D["xc_mm"][:, None]) <= D["sigma_x_mm"][:, None]
            R_plume[case] = region_scales(Lux, Lvy, nzu, nzv, mask)
            sig_by[case] = {"sigma_g": S3["sigma_g_mm"], "sigma_mom": S3["sigma_mom_mm"]}

        # near-source L and d/L
        near = (s >= NEAR_SOURCE_S_MM[0]) & (s <= NEAR_SOURCE_S_MM[1])
        L0 = float(np.nanmedian(R_bg[case]["L"][near]))
        Lu0 = float(np.nanmedian(R_bg[case]["L_ux"][near]))
        d = SOURCE_D_MM.get(case)
        trusted = s <= SIGMA_MAX_VALID_S_MM
        summary["cases"][case] = {
            "L_bg_near_source_mm": L0, "L_ux_bg_near_source_mm": Lu0,
            "L_bg_median_mm": float(np.nanmedian(R_bg[case]["L"][trusted])),
            "A_bg_median": float(np.nanmedian(R_bg[case]["A"][trusted])),
            "L_core_median_mm": float(np.nanmedian(R_core[case]["L"][trusted])),
            "A_core_median": float(np.nanmedian(R_core[case]["A"][trusted])),
            "source_d_mm": d,
            "d_over_L_bg": (d / L0) if d else None, "d_over_L_ux_bg": (d / Lu0) if d else None,
            "edge_trim_effect_on_L_ux_bg_median": float(np.nanmedian(R_untrim[case]["L_ux"][trusted] / R_bg[case]["L_ux"][trusted])),
        }
        if d is None:
            summary["cases"][case]["note"] = "source diameter unknown: set SOURCE_D_MM to compute d/L"
        for nm, R in (("bg", R_bg), ("core", R_core), ("plume", R_plume)):
            if case in R:
                for k, v in R[case].items():
                    saved[f"{case}|{nm}|{k}"] = v
        saved[f"{case}|s_mm"] = s

    # ---- figures: all cases, background and core
    plot_all_cases(s_by, R_bg, "background", "Background (ambient grid turbulence), columns 60-199 and 747-886",
                   OUT_DIR / "L_profiles_background.png")
    plot_all_cases(s_by, R_core, "core", "Core (source wake + plume), centerline +/- 100 columns (373-573, +/- 32 mm)",
                   OUT_DIR / "L_profiles_core.png")
    if R_plume:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for case in R_plume:
            col = COLORS[case]
            axs[0].plot(s_by[case], R_core[case]["L"], color=col, lw=1.4, label=f"{case}: core +/- 100 cols")
            axs[0].plot(s_by[case], R_plume[case]["L"], color=col, lw=1.4, ls="--", label=rf"{case}: |x - x_c| $\leq \sigma_x$")
            axs[1].plot(s_by[case], R_core[case]["L_ux"], color=col, lw=1.4, label=f"{case}: core")
            axs[1].plot(s_by[case], R_plume[case]["L_ux"], color=col, lw=1.4, ls="--", label=f"{case}: plume-relative")
        axs[0].set_title("L (mean of longitudinal scales)")
        axs[1].set_title(r"$L_{u,x}$")
        axs[0].set_ylabel("[mm]")
        for ax in axs:
            ax.set_xlabel("downstream distance s [mm]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("Fixed-width core vs plume-relative core region")
        fig.tight_layout()
        _savefig(fig, OUT_DIR / "core_fixed_vs_plume_relative.png")

    # ---- edge-trim check
    fig, ax = plt.subplots(figsize=(8, 4))
    for case in CASES:
        ax.plot(s_by[case], R_bg[case]["L_ux"], color=COLORS[case], lw=1.4, label=f"{case} trimmed")
        ax.plot(s_by[case], R_untrim[case]["L_ux"], color=COLORS[case], lw=0.9, ls=":", label=f"{case} untrimmed")
    ax.set_xlabel("downstream distance s [mm]")
    ax.set_ylabel(r"background $L_{u,x}$ [mm]")
    ax.set_title("Edge-column trim check: background L_u,x with (solid) and without (dotted) dropping the outer 60 columns",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=3)
    fig.tight_layout()
    _savefig(fig, OUT_DIR / "edge_trim_check.png")

    # ---- sigma_x / L and the meander -> relative-dispersion transition
    if sig_by:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        trans = {}
        for case, sg in sig_by.items():
            s, col = s_by[case], COLORS[case]
            valid = s <= SIGMA_MAX_VALID_S_MM
            ratios = {}
            for wname, ls, lw in (("sigma_g", "-", 1.5), ("sigma_mom", "--", 1.0)):
                sigma = np.where(valid, sg[wname], np.nan)
                for ax, (name, L) in zip(axs, (("L_bg", R_bg[case]["L"]), ("L_ux_bg", R_bg[case]["L_ux"]))):
                    rat = sigma / L
                    ratios[f"{wname}|{name}"] = rat
                    lab = f"{case} ({'Gaussian' if wname == 'sigma_g' else 'second-moment'} width)"
                    ax.plot(s, rat, color=col, lw=lw, ls=ls, label=lab)
                    sx = first_crossing_up(s, rat, 1.0)
                    trans.setdefault(case, {})[f"s_{wname}_eq_{name}_mm"] = sx
                    if np.isfinite(sx):
                        ax.axvline(sx, color=col, lw=lw, ls=ls)
            # earlier proxies (plume-region L_u,x and flank-rho decay) for reference
            comp = next((d / "comparison_summary.json" for d in COMPARISON_DIRS if (d / "comparison_summary.json").exists()),
                        None)
            if comp is not None:
                prev = json.loads(comp.read_text()).get(case, {})
                for k, ls in (("s_sigma_L", ":"), ("s_rho_abs_0.40", "-.")):
                    if prev.get(k) is not None:
                        trans[case][f"previous_{k}_mm"] = prev[k]
                        for ax in axs:
                            ax.axvline(prev[k], color=col, lw=0.8, ls=ls, alpha=0.7)
            for k, v in ratios.items():
                saved[f"{case}|{k}"] = v
        for ax, lab in zip(axs, [r"$\sigma_x / L_{bg}$, $L_{bg} = \frac{1}{2}(L_{u,x}+L_{v,y})$", r"$\sigma_x / L_{u,x,bg}$"]):
            ax.axhline(1, color="0.5", lw=0.7)
            ax.set_title(lab)
            ax.set_xlabel("downstream distance s [mm]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axs[0].set_ylabel(r"plume width / background integral scale")
        fig.suptitle("Meander -> relative dispersion. Curves: solid = Gaussian-fit width, dashed = second-moment width; "
                     "verticals: crossing sigma_x = L (same line style)\ndotted = previous estimate (plume-region L_u,x), "
                     "dash-dot = flank rho_uc falls to 0.40; widths beyond s = 240 mm excluded", fontsize=9)
        fig.tight_layout()
        _savefig(fig, OUT_DIR / "sigma_over_L.png")
        summary["transition"] = trans

    np.savez_compressed(OUT_DIR / "characteristic_scale.npz", **saved)
    (OUT_DIR / "characteristic_scale_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["cases"], indent=1))
    print(json.dumps(summary.get("transition", {}), indent=1))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
