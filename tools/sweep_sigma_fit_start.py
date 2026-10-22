"""
Sweep the downstream start location of the sigma^2-vs-x diffusivity fit for each case.

For each case and each candidate start x_s, fit sigma^2 = a + b x over x >= x_s
(see viz.fit_sigma_diffusivity) and record D, x0, sigma0^2 and goodness of fit.
A start location is then selected per case:

    best x_s = the smallest x_s (longest fit range) with nRMSE <= NRMSE_TOL,
               falling back to the x_s with minimum nRMSE if none meet the tolerance.

nRMSE = RMSE(sigma - sigma_fit) / mean(sigma), evaluated over the fitted points.
Use MANUAL_FIT_START_MM to override the automatic choice for specific cases.

Uses the mean-field paths and Gaussian settings from draw_mean_lateral_profiles.py. Run:
    python tools/sweep_sigma_fit_start.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.draw_mean_lateral_profiles import (  # noqa: E402
    GAUSSIAN_MARKER_EVERY,
    GAUSSIAN_MARKERS,
    MEAN_VELOCITY_MM_S,
    compute_dense_gaussian_results,
    load_inputs,
)
from src.sPIV_PLIF_postprocessing.visualization.viz import (  # noqa: E402
    fit_sigma_diffusivity,
    plot_gaussian_param_scatter,
)

# -------------------------------------------------------------------
# Sweep settings
# -------------------------------------------------------------------
OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/Plots/Mean/Profiles/fit_start_sweep")
MIN_FIT_SPAN_MM = 100.0  # every candidate fit covers at least this much downstream distance
NRMSE_TOL = 0.01  # accept a start if RMSE(sigma) <= 2% of mean sigma over the fit range
MANUAL_FIT_START_MM: dict[str, float] = {}  # e.g. {"smSource": 120.0} overrides the selection

SWEEP_FIELDS = ["case", "x_start_mm", "n_points", "D_mm2_s", "D_cm2_s", "x0_mm",
                "sigma0_sq_mm2", "rmse_mm", "nrmse", "r2"]


def _case_sigma_points(gaussian_results) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-case (x downstream, sigma) arrays, sorted by x, matching the sigma plot's x-axis."""
    points: dict[str, list[tuple[float, float]]] = {}
    for target_y, nearest_y, entries in gaussian_results:
        x = 300 - (nearest_y if np.isfinite(nearest_y) else target_y)
        for label, _, sigma in entries:
            if np.isfinite(sigma):
                points.setdefault(label, []).append((x, sigma))
    out = {}
    for label, pts in points.items():
        pts.sort()
        x, s = zip(*pts)
        out[label] = (np.asarray(x), np.asarray(s))
    return out


def _sweep_case(x: np.ndarray, sigma: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for x_start in x[x <= x.max() - MIN_FIT_SPAN_MM]:
        mask = x >= x_start
        fit = fit_sigma_diffusivity(x[mask], sigma[mask], MEAN_VELOCITY_MM_S)
        rows.append({"x_start_mm": float(x_start), **fit})
    return rows


def _select_start(rows: list[dict[str, float]], tol: float = NRMSE_TOL) -> dict[str, float]:
    ok = [r for r in rows if r["nrmse"] <= tol]
    if ok:
        return min(ok, key=lambda r: r["x_start_mm"])
    return min(rows, key=lambda r: r["nrmse"])


def _to_csv_row(label: str, r: dict[str, float]) -> dict[str, object]:
    return {
        "case": label,
        "x_start_mm": r["x_start_mm"],
        "n_points": r["n_points"],
        "D_mm2_s": r["D"],
        "D_cm2_s": r["D"] / 100.0,
        "x0_mm": r["x0"],
        "sigma0_sq_mm2": r["intercept"],
        "rmse_mm": r["rmse"],
        "nrmse": r["nrmse"],
        "r2": r["r2"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_sweep_summary(
    sweeps: dict[str, list[dict[str, float]]],
    selected: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    labels = list(sweeps)
    cmap = cmr.get_sub_cmap("cmr.rainforest", 0.0, 0.85)
    colors = {lab: cmap(v) for lab, v in zip(labels, np.linspace(0, 1, len(labels)))}
    markers = {lab: GAUSSIAN_MARKERS[i % len(GAUSSIAN_MARKERS)] for i, lab in enumerate(labels)}

    panels = [("D", "D (mm$^2$/s)"), ("x0", "$x_0$ (mm)"), ("nrmse", "nRMSE of $\\sigma$")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 9.0), sharex=True)
    for ax, (key, ylabel) in zip(axes, panels):
        for lab in labels:
            xs = [r["x_start_mm"] for r in sweeps[lab]]
            ys = [r[key] for r in sweeps[lab]]
            ax.plot(xs, ys, color=colors[lab], linewidth=1.2, label=lab)
            sel = selected[lab]
            ax.scatter(sel["x_start_mm"], sel[key], color=colors[lab], marker=markers[lab],
                       s=70, edgecolors="k", linewidths=0.6, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[-1].axhline(NRMSE_TOL, color="k", linestyle=":", linewidth=1.0, label=f"tolerance ({NRMSE_TOL:g})")
    axes[-1].set_yscale("log")
    axes[-1].set_xlabel("fit start, downstream distance (mm)")
    axes[0].set_title("Diffusivity fit vs. fit start location (markers = selected start)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Cases")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handles=axes[-1].get_lines()[-1:])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_loaded, x_coords_loaded, y_coords_loaded = load_inputs()
    gaussian_results = compute_dense_gaussian_results(cases_loaded, x_coords_loaded, y_coords_loaded)
    case_points = _case_sigma_points(gaussian_results)

    sweeps: dict[str, list[dict[str, float]]] = {}
    selected: dict[str, dict[str, float]] = {}
    for label, (x, sigma) in case_points.items():
        sweeps[label] = _sweep_case(x, sigma)
        if not sweeps[label]:
            raise ValueError(f"No candidate starts for {label}; reduce MIN_FIT_SPAN_MM.")
        if label in MANUAL_FIT_START_MM:
            target = MANUAL_FIT_START_MM[label]
            selected[label] = min(sweeps[label], key=lambda r: abs(r["x_start_mm"] - target))
        else:
            selected[label] = _select_start(sweeps[label])

    _write_csv(OUT_DIR / "sweep_results.csv",
               [_to_csv_row(lab, r) for lab, rows in sweeps.items() for r in rows])
    _write_csv(OUT_DIR / "selected_fits.csv",
               [_to_csv_row(lab, r) for lab, r in selected.items()])
    _plot_sweep_summary(sweeps, selected, OUT_DIR / "fit_start_sweep_summary.png")
    plot_gaussian_param_scatter(
        gaussian_results,
        param="sigma",
        out_path=OUT_DIR / "sigma_selected_fits.png",
        title="Gaussian sigma with selected fit ranges",
        ylabel="sigma (mm)",
        markers=GAUSSIAN_MARKERS,
        marker_every=GAUSSIAN_MARKER_EVERY,
        fit_diffusivity=True,
        mean_velocity_mm_s=MEAN_VELOCITY_MM_S,
        hollow_markers=True,
        fit_x_min_by_case={lab: r["x_start_mm"] for lab, r in selected.items()},
    )

    print(f"Selection: earliest start with nRMSE <= {NRMSE_TOL:g} (min span {MIN_FIT_SPAN_MM:g} mm)")
    for label, r in selected.items():
        how = "manual" if label in MANUAL_FIT_START_MM else ("tol" if r["nrmse"] <= NRMSE_TOL else "min nRMSE")
        print(
            f"  {label:>10s}: start = {r['x_start_mm']:5.1f} mm [{how}], D = {r['D']:.4g} mm^2/s, "
            f"x0 = {r['x0']:.4g} mm, nRMSE = {r['nrmse']:.4f}, R^2 = {r['r2']:.4f}"
        )
    print(f"Saved sweep outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
