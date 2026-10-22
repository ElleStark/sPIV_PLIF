"""
Summary statistics of the flux / down-gradient angle maps saved by run_scalar_flux_diffusivity.py (stage 3).

The angle at each pixel is between the time-averaged in-plane flux (<u'c'>, <v'c'>) and -grad(mean C)
(0 deg = down-gradient, 180 deg = counter-gradient). Stage 3 stores it already masked to the plume region,
significant flux (either component >= significance_factor * SE) and |grad C| >= gradient_mask_frac * row max,
i.e. exactly the pixels shown in figures/flux_gradient_overlay.png.

Per case, over those pixels (optionally restricted to a downstream range):
    mean, median          of theta
    rms                   sqrt(mean(theta^2)): rms deviation from perfectly down-gradient
    pct_gt_<T>            % of pixels with theta > T  (area-based misalignment)
    flux_wtd_pct_gt_15    % of total |flux| carried by pixels with theta > 15 deg
    flux_wtd_cos          sum(|F| cos theta) / sum(|F|): fraction of flux magnitude that is down-gradient
    coverage_pct          % of plume-region pixels that pass the mask (the rest are not in the statistics)

Run:
    python tools/flux_gradient_angle_stats.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("E:/sPIV_PLIF_ProcessedData/scalar_flux_diffusivity")
CASES = ["smSource", "nearbed", "fractal", "baseline", "buoyant", "diffusive"]
THRESHOLDS_DEG = [15.0, 45.0, 90.0]
S_RANGE_MM: tuple[float, float] | None = None  # e.g. (50.0, 240.0) to restrict downstream distance
# True: stage-3 mask (plume region & significant flux & strong gradient), as in flux_gradient_overlay.png.
# False: plume region only; the angle is recomputed from the saved fluxes and gradients.
USE_EXTRA_FILTERS = True
_SUFFIX = "" if USE_EXTRA_FILTERS else "_region_only"
OUT_CSV = ROOT / f"flux_gradient_angle_stats{_SUFFIX}.csv"
OUT_COMPONENT_CSV = ROOT / f"flux_component_alignment_stats{_SUFFIX}.csv"
OUT_COMPONENT_FIG = ROOT / f"flux_component_alignment_vs_s{_SUFFIX}.png"
PRIMARY_SIGMA_MM = 2.0  # gradient smoothing used for the angle map ([gradients] primary_sigma_mm)
BIN_EDGES_MM = (10.0, 291.0, 10.0)  # downstream bins for the profile figure (start, stop, step)
MIN_PX_PER_BIN = 50


def _angle_map(case: str) -> np.ndarray:
    """Angle map [deg], NaN outside the selected mask (see USE_EXTRA_FILTERS)."""
    arrays = ROOT / case / "arrays"
    if USE_EXTRA_FILTERS:
        return np.load(arrays / "stage3.npz")["angle"]
    mom = np.load(arrays / "moments.npz")
    grads = np.load(arrays / "gradients.npz")
    region = np.load(arrays / "diffusivity.npz")["region"].astype(bool)
    fx, fy = mom["uc_flux"], mom["vc_flux"]
    gx = grads[f"dCdx_sigma{PRIMARY_SIGMA_MM:g}mm"]
    gy = grads[f"dCdy_sigma{PRIMARY_SIGMA_MM:g}mm"]
    # same definition as analysis.scalar_flux.diffusivity.flux_gradient_angle
    norm = np.hypot(fx, fy) * np.hypot(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.degrees(np.arccos(np.clip(-(fx * gx + fy * gy) / norm, -1.0, 1.0)))
    return np.where(region & (norm > 0), theta, np.nan)


def case_stats(case: str) -> dict[str, float]:
    arrays = ROOT / case / "arrays"
    st3 = np.load(arrays / "stage3.npz")
    region = np.load(arrays / "diffusivity.npz")["region"].astype(bool)
    mom = np.load(arrays / "moments.npz")
    theta = _angle_map(case)
    s_mm = st3["s_mm"]
    fmag = np.hypot(mom["uc_flux"], mom["vc_flux"])

    rows = np.ones_like(s_mm, dtype=bool)
    if S_RANGE_MM is not None:
        rows = (s_mm >= S_RANGE_MM[0]) & (s_mm <= S_RANGE_MM[1])
    valid = np.isfinite(theta) & rows[:, None]
    th = theta[valid]
    w = fmag[valid]

    out: dict[str, float] = {
        "case": case,
        "n_px": int(th.size),
        "coverage_pct": 100.0 * th.size / max(int(np.count_nonzero(region & rows[:, None])), 1),
        "mean_deg": float(np.mean(th)),
        "median_deg": float(np.median(th)),
        "rms_deg": float(np.sqrt(np.mean(th**2))),
    }
    for t in THRESHOLDS_DEG:
        out[f"pct_gt_{t:g}"] = 100.0 * float(np.mean(th > t))
    out["flux_wtd_pct_gt_15"] = 100.0 * float(np.sum(w[th > 15.0]) / np.sum(w))
    out["flux_wtd_cos"] = float(np.sum(w * np.cos(np.radians(th))) / np.sum(w))
    return out


def _load_components(case: str):
    """Masked pixels (same as the angle map), downstream distance per pixel, and per-component flux/gradient."""
    arrays = ROOT / case / "arrays"
    st3 = np.load(arrays / "stage3.npz")
    mom = np.load(arrays / "moments.npz")
    grads = np.load(arrays / "gradients.npz")
    s_mm = st3["s_mm"]
    valid = np.isfinite(_angle_map(case))
    if S_RANGE_MM is not None:
        valid &= ((s_mm >= S_RANGE_MM[0]) & (s_mm <= S_RANGE_MM[1]))[:, None]
    s_px = np.broadcast_to(s_mm[:, None], valid.shape)[valid]
    comps = {
        "x": (mom["uc_flux"][valid], grads[f"dCdx_sigma{PRIMARY_SIGMA_MM:g}mm"][valid]),  # cross-stream
        "y": (mom["vc_flux"][valid], grads[f"dCdy_sigma{PRIMARY_SIGMA_MM:g}mm"][valid]),  # streamwise
    }
    return s_px, comps


def _alignment(f: np.ndarray, g: np.ndarray) -> tuple[float, float, float]:
    """(% pixels down-gradient, flux-weighted % down-gradient, net alignment index).
    Down-gradient: -f * g > 0. Net index = sum(-f sign(g)) / sum(|f|) in [-1, 1]:
    +1 all flux down-gradient, -1 all counter-gradient, 0 balanced."""
    down = -f * g > 0
    af = np.abs(f)
    return (100.0 * float(np.mean(down)), 100.0 * float(np.sum(af[down]) / np.sum(af)),
            float(np.sum(-f * np.sign(g)) / np.sum(af)))


def component_stats(case: str) -> dict[str, float]:
    _, comps = _load_components(case)
    (fx, gx), (fy, gy) = comps["x"], comps["y"]
    out: dict[str, float] = {"case": case}
    for comp, (f, g) in comps.items():
        pct, fw_pct, net = _alignment(f, g)
        out[f"{comp}_pct_down"] = pct
        out[f"{comp}_flux_wtd_pct_down"] = fw_pct
        out[f"{comp}_net_alignment"] = net
    # share of in-plane flux "energy" carried by the streamwise component
    out["y_share_of_flux_sq_pct"] = 100.0 * float(np.sum(fy**2) / np.sum(fx**2 + fy**2))
    out["median_flux_ratio_y_over_x"] = float(np.median(np.abs(fy) / np.abs(fx)))
    out["median_grad_ratio_y_over_x"] = float(np.median(np.abs(gy) / np.abs(gx)))
    # implied diffusivity anisotropy if each component were down-gradient: |f/g| ratio
    out["median_Kratio_y_over_x"] = float(np.median(np.abs(fy / gy) / np.abs(fx / gx)))
    return out


def plot_component_profiles(cases: list[str], out_path: Path) -> None:
    """Net alignment index of each flux component and |<v'c'>|/|<u'c'>| vs downstream distance."""
    import cmasher as cmr
    import matplotlib.pyplot as plt

    edges = np.arange(*BIN_EDGES_MM)
    centers = 0.5 * (edges[1:] + edges[:-1])
    cmap = cmr.get_sub_cmap("cmr.rainforest", 0.0, 0.85)
    colors = {c: cmap(v) for c, v in zip(cases, np.linspace(0, 1, len(cases)))}

    fig, axs = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True)
    for case in cases:
        s_px, comps = _load_components(case)
        idx = np.digitize(s_px, edges) - 1
        prof = {k: np.full(centers.size, np.nan) for k in ("x", "y", "ratio")}
        for b in range(centers.size):
            m = idx == b
            if np.count_nonzero(m) < MIN_PX_PER_BIN:
                continue
            for comp in ("x", "y"):
                prof[comp][b] = _alignment(comps[comp][0][m], comps[comp][1][m])[2]
            prof["ratio"][b] = np.median(np.abs(comps["y"][0][m]) / np.abs(comps["x"][0][m]))
        axs[0].plot(centers, prof["x"], color=colors[case], label=case)
        axs[1].plot(centers, prof["y"], color=colors[case])
        axs[2].plot(centers, prof["ratio"], color=colors[case])

    axs[0].set_ylabel("net alignment\ncross-stream $\\langle u'c'\\rangle$")
    axs[1].set_ylabel("net alignment\nstreamwise $\\langle v'c'\\rangle$")
    for ax in axs[:2]:
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
    axs[2].set_ylabel("median $|\\langle v'c'\\rangle| / |\\langle u'c'\\rangle|$")
    axs[2].axhline(1, color="k", linewidth=0.8, linestyle=":")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("downstream distance s (mm)")
    for ax in axs:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axs[0].set_title("Flux component alignment with $-\\nabla C$  (+1 down-gradient, -1 counter-gradient)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Cases")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comp_results = [component_stats(c) for c in CASES]
    with OUT_COMPONENT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_results[0]))
        writer.writeheader()
        writer.writerows(comp_results)
    plot_component_profiles(CASES, OUT_COMPONENT_FIG)
    print("Per-component alignment (x = cross-stream <u'c'>, y = streamwise <v'c'>)")
    print(f"{'case':>10s} {'x%dn':>6s} {'xFw%':>6s} {'xNet':>6s} {'y%dn':>6s} {'yFw%':>6s} {'yNet':>6s} "
          f"{'ySh%':>6s} {'|f|y/x':>7s} {'|g|y/x':>7s} {'Ky/Kx':>6s}")
    for r in comp_results:
        print(f"{r['case']:>10s} {r['x_pct_down']:6.1f} {r['x_flux_wtd_pct_down']:6.1f} {r['x_net_alignment']:6.2f} "
              f"{r['y_pct_down']:6.1f} {r['y_flux_wtd_pct_down']:6.1f} {r['y_net_alignment']:6.2f} "
              f"{r['y_share_of_flux_sq_pct']:6.1f} {r['median_flux_ratio_y_over_x']:7.2f} "
              f"{r['median_grad_ratio_y_over_x']:7.2f} {r['median_Kratio_y_over_x']:6.1f}")
    print(f"Saved {OUT_COMPONENT_CSV} and {OUT_COMPONENT_FIG}\n")

    results = [case_stats(c) for c in CASES]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    hdr = f"{'case':>10s} {'cov%':>5s} {'mean':>6s} {'median':>6s} {'rms':>6s} " + " ".join(
        f"{'>' + format(t, 'g'):>5s}%" for t in THRESHOLDS_DEG) + f" {'Fw>15%':>7s} {'Fw cos':>6s}"
    print(f"Angle between flux and -grad C [deg], S_RANGE_MM = {S_RANGE_MM}")
    print(hdr)
    for r in results:
        print(f"{r['case']:>10s} {r['coverage_pct']:5.0f} {r['mean_deg']:6.1f} {r['median_deg']:6.1f} {r['rms_deg']:6.1f} "
              + " ".join(f"{r[f'pct_gt_{t:g}']:6.1f}" for t in THRESHOLDS_DEG)
              + f" {r['flux_wtd_pct_gt_15']:7.1f} {r['flux_wtd_cos']:6.3f}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
