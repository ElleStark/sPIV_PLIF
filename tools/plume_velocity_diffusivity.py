"""
Effective diffusivity from the Gaussian plume width using the measured advection velocity inside the plume
envelope instead of an assumed constant U.

Plume envelope (from run_scalar_flux_diffusivity.py, diffusivity.npz "region"): mean C - background > 0.02 and
|x - x_c(s)| <= 2.5 sigma_x(s). U_env(s) = mean over the envelope pixels in row s of |time-averaged v|.

Fit ranges are the per-case starts selected by sweep_sigma_fit_start.py (same selection rule and settings).

Two estimates per case:
  (1) constant U:  sigma^2 = a + b s on s >= s_start;  D_U = b * Ubar / 2,  Ubar = mean U_env over the fit range.
      The virtual origin x0 = -a / b does not depend on U.
  (2) travel time: t(s) = int_{s_ref}^{s} ds' / U_env(s');  sigma^2 = a' + 2 D_t t on s >= s_start.
      Virtual origin in time t0 = -a' / (2 D_t), mapped back to distance s0 with t(s) (linear extrapolation
      with U_env at the nearest end if t0 falls outside the sampled range).

Run:
    python tools/plume_velocity_diffusivity.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.draw_mean_lateral_profiles import MEAN_VELOCITY_MM_S, compute_dense_gaussian_results, load_inputs  # noqa: E402
from tools.sweep_sigma_fit_start import (  # noqa: E402
    MANUAL_FIT_START_MM,
    OUT_DIR,
    _case_sigma_points,
    _select_start,
    _sweep_case,
)

FLUX_ROOT = Path("E:/sPIV_PLIF_ProcessedData/scalar_flux_diffusivity")
NRMSE_TOL = 0.02  # fit-start tolerance (sweep_sigma_fit_start._select_start); overrides the sweep's setting
OUT_CSV = OUT_DIR / f"velocity_corrected_fits_tol{NRMSE_TOL:g}.csv"
OUT_FIG = OUT_DIR / f"plume_envelope_velocity_tol{NRMSE_TOL:g}.png"


def envelope_velocity(case: str) -> tuple[np.ndarray, np.ndarray]:
    """(s_mm, U_env [mm/s]) per row, sorted by increasing s."""
    arrays = FLUX_ROOT / case / "arrays"
    v = np.load(arrays / "moments.npz")["v_mean"]
    region = np.load(arrays / "diffusivity.npz")["region"].astype(bool)
    s_mm = np.load(arrays / "stage3.npz")["s_mm"]
    m = region & np.isfinite(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = 1000.0 * np.sum(np.where(m, np.abs(v), 0.0), axis=1) / np.sum(m, axis=1)
    order = np.argsort(s_mm)
    return s_mm[order], u[order]


def travel_time(s_eval: np.ndarray, s_row: np.ndarray, u_row: np.ndarray, s_ref: float) -> np.ndarray:
    """t(s) = int_{s_ref}^{s} ds / U [s], trapezoid on the row grid, interpolated to s_eval."""
    ok = np.isfinite(u_row) & (u_row > 0)
    s_ok, inv_u = s_row[ok], 1.0 / u_row[ok]
    t_row = np.concatenate([[0.0], np.cumsum(0.5 * (inv_u[1:] + inv_u[:-1]) * np.diff(s_ok))])
    t_row -= np.interp(s_ref, s_ok, t_row)
    return np.interp(s_eval, s_ok, t_row)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    cases_loaded, x_coords_loaded, y_coords_loaded = load_inputs()
    points = _case_sigma_points(compute_dense_gaussian_results(cases_loaded, x_coords_loaded, y_coords_loaded))

    rows, u_profiles = [], {}
    for case, (s, sigma) in points.items():
        sweep = _sweep_case(s, sigma)
        if case in MANUAL_FIT_START_MM:
            sel = min(sweep, key=lambda r: abs(r["x_start_mm"] - MANUAL_FIT_START_MM[case]))
        else:
            sel = _select_start(sweep, NRMSE_TOL)
        s_start = sel["x_start_mm"]
        fit = s >= s_start

        s_row, u_row = envelope_velocity(case)
        u_profiles[case] = (s_row, u_row, s_start)
        u_at = np.interp(s, s_row, u_row)
        u_bar = float(np.mean(u_at[fit]))

        # (1) constant U over the fit range
        b, a = np.polyfit(s[fit], sigma[fit] ** 2, 1)
        d_const = 0.5 * b * u_bar
        x0 = -a / b

        # (2) travel time
        t = travel_time(s, s_row, u_row, s_ref=float(s.min()))
        bt, at = np.polyfit(t[fit], sigma[fit] ** 2, 1)
        d_t = 0.5 * bt
        t0 = -at / bt
        t_fine = travel_time(s_row, s_row, u_row, s_ref=float(s.min()))
        okf = np.isfinite(t_fine)
        if t_fine[okf].min() <= t0 <= t_fine[okf].max():
            s0 = float(np.interp(t0, t_fine[okf], s_row[okf]))
        else:  # extrapolate with the velocity at the nearer end
            i = 0 if t0 < t_fine[okf].min() else -1
            s0 = float(s_row[okf][i] + (t0 - t_fine[okf][i]) * u_row[okf][i])
        resid = sigma[fit] - np.sqrt(np.clip(at + bt * t[fit], 0, None))

        rows.append({
            "case": case,
            "s_start_mm": s_start,
            "U_env_mean_mm_s": u_bar,
            "U_env_min_mm_s": float(np.min(u_at[fit])),
            "U_env_max_mm_s": float(np.max(u_at[fit])),
            "D_assumedU_mm2_s": 0.5 * b * MEAN_VELOCITY_MM_S,
            "D_constU_mm2_s": d_const,
            "D_traveltime_mm2_s": d_t,
            "x0_mm": x0,
            "s0_traveltime_mm": s0,
            "t0_s": t0,
            "nrmse_traveltime": float(np.sqrt(np.mean(resid**2)) / np.mean(sigma[fit])),
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'case':>10s} {'start':>6s} {'Ubar':>6s} {'Umin':>5s} {'Umax':>5s} {'D@300':>7s} {'D_Ubar':>7s} "
          f"{'D_t':>7s} {'x0':>6s} {'s0_t':>6s} {'nRMSE_t':>8s}")
    for r in rows:
        print(f"{r['case']:>10s} {r['s_start_mm']:6.0f} {r['U_env_mean_mm_s']:6.0f} {r['U_env_min_mm_s']:5.0f} "
              f"{r['U_env_max_mm_s']:5.0f} {r['D_assumedU_mm2_s']:7.0f} {r['D_constU_mm2_s']:7.0f} "
              f"{r['D_traveltime_mm2_s']:7.0f} {r['x0_mm']:6.0f} {r['s0_traveltime_mm']:6.0f} "
              f"{r['nrmse_traveltime']:8.4f}")
    print(f"(velocities mm/s, D mm^2/s, distances mm; fit-start tolerance nRMSE <= {NRMSE_TOL:g})")

    import matplotlib.pyplot as plt
    import cmasher as cmr

    cmap = cmr.get_sub_cmap("cmr.rainforest", 0.0, 0.85)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for (case, (s_row, u_row, s_start)), col in zip(u_profiles.items(), np.linspace(0, 1, len(u_profiles))):
        ax.plot(s_row, u_row, color=cmap(col), linewidth=0.8, alpha=0.5)
        m = s_row >= s_start
        ax.plot(s_row[m], u_row[m], color=cmap(col), linewidth=1.8, label=case)
    ax.axhline(MEAN_VELOCITY_MM_S, color="k", linestyle=":", linewidth=1.0, label="assumed 300 mm/s")
    ax.set_xlabel("downstream distance s (mm)")
    ax.set_ylabel("plume-envelope mean |v| (mm/s)")
    ax.set_title("Advection velocity within the plume envelope (bold = fit range)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_CSV} and {OUT_FIG}")


if __name__ == "__main__":
    main()
