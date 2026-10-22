"""
Checks for whether the field of view lies in the developing (near-grid) region of the grid turbulence.

Distance from the grid: x = s + GRID_TO_DOMAIN_MM (the grid is 150 mm upstream of s = 0), mesh M = 25.4 mm,
so the field of view spans x/M ~ 5.9 - 17.5.

Check 1  transverse homogeneity: amplitude of the cross-stream modulation at the mesh period M in mean V, u_rms and
         v_rms (bar wakes not yet merged), from a least-squares sinusoid fit over a range of periods on each side of
         the plume (|x| in SIDE_X_RANGE_MM), per downstream band. Reported relative to the local mean (V: to |V|).
Check 2  TKE decay: background k against x/M on log-log axes, local slope n_loc = -d ln k / d ln x, power-law fits
         over the full window and s <= FIT_SPLIT_MM, and the implied L exponent 1 - n/2 compared with the measured one.
Check 3  normalized dissipation C_eps = eps L / u'^3 (u' = sqrt(2k/3)) against x/M and Re_lambda. Roughly constant
         in equilibrium decay; varies (~ Re^-1) in the non-equilibrium near-grid region.

Inputs (all existing outputs; nothing is recomputed from the raw stacks):
    scalar_flux_diffusivity/<case>/arrays/moments.npz           full-width mean V, u_rms, v_rms (check 1)
    flow_properties/Plots/<case>/tke, epsilon, Taylor_Re maps    + epsilon_estimates (background mask) (checks 2-3)
    flow_properties/Integral_length_scales/ILS/ILS_<case>_*.npy  L_u,x, L_v,y (checks 2-3)
Background (checks 2-3) = the flow-property background mask (|x| >= 70 mm, outside the plume envelope), so k, eps
and L come from the same pixels.

Outputs: OUT_DIR (see below): PNGs, development_checks.npz, development_checks_summary.json

Run from the repo root:
    python -m tools.grid_development_checks
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
BASE = Path("E:/sPIV_PLIF_ProcessedData")
SF_DIR = BASE / "scalar_flux_diffusivity"
FP_DIR = BASE / "flow_properties" / "Plots"
ILS_DIR = BASE / "flow_properties" / "Integral_length_scales" / "ILS"
OUT_DIR = BASE / "flow_properties" / "Development_region_checks"

CASES = ["baseline", "buoyant", "diffusive", "fractal", "nearbed", "smSource"]
REGULAR_GRID_CASES = ["baseline", "buoyant", "diffusive", "smSource"]  # same regular grid, ambient background
MESH_MM = 25.4
GRID_TO_DOMAIN_MM = 150.0
NU = 1.5e-5
FP_COL0 = int(50 / (1000 * 0.000317))  # flow-property maps start at this full-grid column (Y_SLICE in compute_flow_properties)

# check 1
SIDE_X_RANGE_MM = (45.0, 140.0)       # |x| range on each side of the plume used for the transverse spectrum
PERIODS_MM = np.arange(8.0, 70.5, 0.5)
MESH_BAND = (0.85, 1.18)              # periods within this factor of M count as the mesh peak
BAND_ROWS = 16                        # rows per downstream band (~5 mm)
# check 2/3
ROWS_TO_AVG = 15                      # running mean along s for the background profiles (~4.8 mm)
FIT_SPLIT_MM = 150.0                  # also fit over s <= this (L less affected by the field of view)
S_MAX_MM = 285.0  # the last ~15 rows carry image-edge artifacts in k, eps and L
L_EXPONENT_MEASURED = {"full": 1.03, "near": 0.73}  # background L fits with x0 = 150 mm (plot_ils_profiles_bw)

LINESTYLES = {"baseline": "-", "buoyant": (0, (3, 2)), "diffusive": (0, (1, 1.5)), "fractal": (0, (9, 2.5, 1.5, 2.5)),
              "nearbed": (0, (4, 1.5, 1, 1.5, 1, 1.5, 1, 1.5)), "smSource": (0, (12, 4))}
FONT_FAMILY = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
DPI = 300
plt.rcParams.update({"font.family": "serif", "font.serif": FONT_FAMILY, "mathtext.fontset": "stix", "font.size": 11})


def _savefig(fig, out_path: Path, retries: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    for attempt in range(retries):
        try:
            os.replace(tmp, out_path)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)
    print(f"Saved {out_path}")


def _smooth(p: np.ndarray, k: int) -> np.ndarray:
    fin = np.isfinite(p)
    num = np.convolve(np.where(fin, p, 0.0), np.ones(k), mode="same")
    den = np.convolve(fin.astype(float), np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < k / 2] = np.nan
    return out


def _row_median(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(mask, arr, np.nan), axis=1)


def _style(case):
    return dict(color="black", ls=LINESTYLES[case], lw=1.1, label=case)


# ------------------------------------------------------------------- check 1
def transverse_spectrum(profile: np.ndarray, x: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Sinusoid amplitude vs period, each fitted jointly with a quadratic trend (least squares)."""
    ok = np.isfinite(profile)
    x, p = x[ok], profile[ok]
    xc = (x - x.mean()) / np.ptp(x)
    base = np.column_stack([np.ones_like(xc), xc, xc ** 2])
    amp = np.full(periods.size, np.nan)
    for i, P in enumerate(periods):
        k = 2 * np.pi / P
        A = np.column_stack([base, np.cos(k * x), np.sin(k * x)])
        coef, *_ = np.linalg.lstsq(A, p, rcond=None)
        amp[i] = np.hypot(coef[3], coef[4])
    return amp


def check1(case: str) -> dict:
    m = np.load(SF_DIR / case / "arrays" / "moments.npz")
    x, y = m["x_mm"], m["y_mm"]
    s = y.max() - y
    sides = [(x >= SIDE_X_RANGE_MM[0]) & (x <= SIDE_X_RANGE_MM[1]), (x <= -SIDE_X_RANGE_MM[0]) & (x >= -SIDE_X_RANGE_MM[1])]
    starts = np.arange(0, len(s) - BAND_ROWS + 1, BAND_ROWS)
    out = {"s_mm": np.array([s[r0:r0 + BAND_ROWS].mean() for r0 in starts]), "periods_mm": PERIODS_MM}
    mesh = (PERIODS_MM >= MESH_BAND[0] * MESH_MM) & (PERIODS_MM <= MESH_BAND[1] * MESH_MM)
    for var in ("v_mean", "u_rms", "v_rms"):
        spec = np.full((starts.size, PERIODS_MM.size), np.nan)
        for b, r0 in enumerate(starts):
            band = np.nanmean(m[var][r0:r0 + BAND_ROWS], axis=0)
            amps = []
            for sd in sides:
                prof = band[sd]
                ref = np.nanmean(np.abs(prof))
                amps.append(transverse_spectrum(prof, x[sd], PERIODS_MM) / ref * 100.0)
            spec[b] = np.sqrt(np.nanmean(np.square(amps), axis=0))  # rms over the two sides, % of local mean
        out[f"{var}_spec_pct"] = spec
        out[f"{var}_mesh_pct"] = np.nanmax(spec[:, mesh], axis=1)
        out[f"{var}_offmesh_pct"] = np.nanmedian(spec[:, ~mesh & (PERIODS_MM < 50)], axis=1)
        out[f"{var}_peak_period_mm"] = PERIODS_MM[np.nanargmax(spec, axis=1)]
    return out


# ------------------------------------------------------------------- checks 2-3
def background_profiles(case: str) -> dict:
    d = FP_DIR / case
    est = np.load(d / f"epsilon_estimates_{case}.npz")
    bg = est["background_mask"]
    s = est["s_mm"]
    ny, nx = bg.shape
    k = np.load(d / f"tke_{case}.npy")
    eps = np.load(d / f"epsilon_{case}.npy")
    rel = np.load(d / f"Taylor_Re_{case}.npy")
    cols = slice(FP_COL0, FP_COL0 + nx)
    Lux = np.load(ILS_DIR / f"ILS_{case}_u_cross_stream.npy")[:ny, cols].astype(float) * 1000
    Lvy = np.load(ILS_DIR / f"ILS_{case}_v_streamwise.npy")[:ny, cols].astype(float) * 1000
    prof = {"s_mm": s}
    for name, arr in (("k", k), ("eps", eps), ("Re_lambda", rel), ("L_ux", Lux), ("L_vy", Lvy)):
        prof[name] = _smooth(_row_median(arr, bg & np.isfinite(arr)), ROWS_TO_AVG)
    prof["L"] = 0.5 * (prof["L_ux"] + prof["L_vy"])
    up = np.sqrt(2.0 * prof["k"] / 3.0)
    prof["u_prime"] = up
    with np.errstate(invalid="ignore", divide="ignore"):
        prof["C_eps"] = prof["eps"] * prof["L"] / 1000 / up ** 3
        prof["C_eps_Lux"] = prof["eps"] * prof["L_ux"] / 1000 / up ** 3
        prof["Re_L"] = up * prof["L"] / 1000 / NU
    xg = s + GRID_TO_DOMAIN_MM
    with np.errstate(invalid="ignore", divide="ignore"):
        lk, lx = np.log(prof["k"]), np.log(xg)
    o = np.argsort(xg)
    n_loc = np.full_like(s, np.nan)
    fin = np.isfinite(lk[o])
    if fin.sum() > 61:
        dl = np.gradient(savgol_filter(lk[o][fin], 61, 2), lx[o][fin])
        tmp = np.full(o.size, np.nan)
        tmp[np.flatnonzero(fin)] = -dl
        n_loc[o] = tmp
    prof["n_local"] = n_loc
    return prof


def power_fit(s: np.ndarray, p: np.ndarray, s_max: float) -> float:
    ok = np.isfinite(p) & (p > 0) & (s <= s_max) & (s >= 3.0)
    return float(-np.polyfit(np.log(s[ok] + GRID_TO_DOMAIN_MM), np.log(p[ok]), 1)[0])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, summary = {}, {"settings": {"mesh_mm": MESH_MM, "grid_to_domain_mm": GRID_TO_DOMAIN_MM,
                                       "side_x_range_mm": SIDE_X_RANGE_MM, "fit_split_mm": FIT_SPLIT_MM}, "cases": {}}
    C1, P = {}, {}
    for case in CASES:
        C1[case] = check1(case)
        P[case] = background_profiles(case)
        for k_, v in C1[case].items():
            saved[f"{case}|check1|{k_}"] = v
        for k_, v in P[case].items():
            saved[f"{case}|bg|{k_}"] = v

    # ---- check 1 figures
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, var, lab in zip(axs, ("v_mean", "u_rms", "v_rms"), (r"mean $V$", r"$u_{rms}$", r"$v_{rms}$")):
        for case in CASES:
            r = C1[case]
            ax.plot(r["s_mm"] + GRID_TO_DOMAIN_MM, r[f"{var}_mesh_pct"], **_style(case))
        r = C1["baseline"]
        ax.plot(r["s_mm"] + GRID_TO_DOMAIN_MM, r[f"{var}_offmesh_pct"], color="0.6", lw=0.9, label="baseline off-mesh floor")
        ax.set_title(lab)
        ax.set_xlabel("distance from grid $x$ (mm)")
        sec = ax.secondary_xaxis("top", functions=(lambda v: v / MESH_MM, lambda v: v * MESH_MM))
        sec.set_xlabel("$x/M$")
    axs[0].set_ylabel("modulation amplitude at period $\\approx M$ (% of local mean)")
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _savefig(fig, OUT_DIR / "check1_mesh_modulation_vs_x.png")

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, case in enumerate(("baseline", "fractal")):
        r = C1[case]
        for col, (var, lab) in enumerate(zip(("v_mean", "u_rms", "v_rms"), (r"mean $V$", r"$u_{rms}$", r"$v_{rms}$"))):
            ax = axs[row, col]
            im = ax.pcolormesh(r["periods_mm"] / MESH_MM, (r["s_mm"] + GRID_TO_DOMAIN_MM) / MESH_MM, r[f"{var}_spec_pct"],
                               shading="nearest", cmap="Greys", vmin=0, vmax=np.nanpercentile(r[f"{var}_spec_pct"], 99))
            ax.axvline(1.0, color="red", lw=0.7, ls="--")
            ax.set_title(f"{case}: {lab}", fontsize=10)
            fig.colorbar(im, ax=ax, label="% of local mean")
            if row == 1:
                ax.set_xlabel("transverse period / $M$")
            if col == 0:
                ax.set_ylabel("$x/M$")
    fig.tight_layout()
    _savefig(fig, OUT_DIR / "check1_transverse_spectra.png")

    # ---- check 2 figures
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for case in CASES:
        p = P[case]
        keep = p["s_mm"] <= S_MAX_MM
        xm = (p["s_mm"][keep] + GRID_TO_DOMAIN_MM) / MESH_MM
        axs[0].loglog(xm, p["k"][keep], **_style(case))
        axs[1].plot(xm, p["n_local"][keep], **_style(case))
    xr = np.array([6.0, 17.5])
    for n_ref, txt in ((1.2, "$n = 1.2$"),):
        axs[0].loglog(xr, 9e-4 * (xr / 6.0) ** (-n_ref), color="red", ls="--", lw=0.9, label=txt)
    axs[0].set_xlabel("$x/M$")
    axs[0].set_ylabel("background $k$ (m$^2$ s$^{-2}$)")
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].axhspan(1.1, 1.4, color="0.85", zorder=0, label="classical decay range")
    axs[1].set_xlabel("$x/M$")
    axs[1].set_ylabel(r"local decay exponent $n = -d\ln k / d\ln x$")
    axs[1].set_ylim(-1.5, 3.5)
    axs[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _savefig(fig, OUT_DIR / "check2_tke_decay.png")

    # ---- check 3 figures
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for case in CASES:
        p = P[case]
        keep = p["s_mm"] <= S_MAX_MM
        xm = (p["s_mm"][keep] + GRID_TO_DOMAIN_MM) / MESH_MM
        axs[0].plot(xm, p["C_eps"][keep], **_style(case))
        axs[1].loglog(p["Re_lambda"][keep], p["C_eps"][keep], **_style(case))
    axs[0].axhspan(0.5, 1.0, color="0.85", zorder=0, label="equilibrium range (~0.5-1)")
    axs[0].set_xlabel("$x/M$")
    axs[0].set_ylabel(r"$C_\varepsilon = \varepsilon L / u'^3$")
    axs[0].legend(fontsize=8, frameon=False)
    re = np.array([8.0, 40.0])
    axs[1].loglog(re, 1.0 * (re / 10.0) ** -1, color="red", ls="--", lw=0.9, label=r"$\propto Re_\lambda^{-1}$")
    axs[1].set_xlabel(r"background $Re_\lambda$")
    axs[1].set_ylabel(r"$C_\varepsilon$")
    axs[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _savefig(fig, OUT_DIR / "check3_normalized_dissipation.png")

    # ---- summary
    for case in CASES:
        p, r = P[case], C1[case]
        s = p["s_mm"]
        n_full, n_near = power_fit(s, p["k"], S_MAX_MM), power_fit(s, p["k"], FIT_SPLIT_MM)
        near, far = r["s_mm"] < 60, (r["s_mm"] > 230) & (r["s_mm"] < S_MAX_MM)
        ce = p["C_eps"][(s <= S_MAX_MM)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary["cases"][case] = {
                "check1_mesh_pct_near_far": {v: [float(np.nanmean(r[f"{v}_mesh_pct"][near])), float(np.nanmean(r[f"{v}_mesh_pct"][far]))]
                                             for v in ("v_mean", "u_rms", "v_rms")},
                "check1_offmesh_floor_pct": {v: float(np.nanmedian(r[f"{v}_offmesh_pct"])) for v in ("v_mean", "u_rms", "v_rms")},
                "check2_k_decay_exponent_full": n_full, "check2_k_decay_exponent_s_le_split": n_near,
                "check2_implied_L_exponent_full": 1 - n_full / 2, "check2_implied_L_exponent_s_le_split": 1 - n_near / 2,
                "check2_local_n_range": [float(np.nanpercentile(p["n_local"][s <= S_MAX_MM], 10)),
                                         float(np.nanpercentile(p["n_local"][s <= S_MAX_MM], 90))],
                "check3_C_eps_near_far": [float(np.nanmedian(ce[s[s <= S_MAX_MM] < 60])), float(np.nanmedian(ce[s[s <= S_MAX_MM] > 230]))],
                "check3_Re_lambda_near_far": [float(np.nanmedian(p["Re_lambda"][s < 60])), float(np.nanmedian(p["Re_lambda"][(s > 230) & (s <= S_MAX_MM)]))],
            }
    summary["measured_L_exponents_x0_150mm"] = L_EXPONENT_MEASURED
    np.savez_compressed(OUT_DIR / "development_checks.npz", **saved)
    (OUT_DIR / "development_checks_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["cases"], indent=1))


if __name__ == "__main__":
    main()
