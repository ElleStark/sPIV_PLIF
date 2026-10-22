"""
Compute flow properties using the helpers in analysis.flow_properties.

Array layout: velocity stacks are (t, y, x) on the PLIF grid. y (rows, axis 1) is streamwise with velocity v,
x (columns, axis 2) is cross-stream with velocity u, w is out of plane. (The slice names below are historical:
X_SLICE selects rows = y, Y_SLICE selects columns = x.)

Dissipation rate (Oct 2026 revision; the previous version used transverse derivatives du/dy, dv/dx in the
isotropic longitudinal formula and overestimated epsilon by ~2-3x):
    * gradients of the fluctuating velocity: all x- and y-derivatives of u, v, w (stereo PIV) by central
      differences spanning 2*step grid points, dw/dz from continuity;
    * eps_hybrid (primary) = nu <du_i/dx_j du_i/dx_j> with only <(du/dz)^2>, <(dv/dz)^2> from isotropy;
      eps_iso_long = 5 nu [<(du/dx)^2> + <(dv/dy)^2> + <(dw/dz)^2>] reported for comparison;
    * resolution: the field is PIV data (0.76 mm vectors, ~3 mm interrogation windows) interpolated onto the
      0.317 mm PLIF grid, so gradients are reported for several stencil spans; the primary span (2 x 0.317 mm =
      0.63 mm) is close to the native PIV vector spacing. Direct PIV gradients underestimate epsilon when the
      interrogation window exceeds ~ the Kolmogorov scale;
    * cross-checks: background TKE decay eps = U |dk/ds| (advection-dissipation balance in decaying grid
      turbulence), and inertial-range second-order longitudinal structure functions eps(r) = (D_LL/C2)^(3/2)/r.
TKE and rms are NaN-aware (w' < -1 m/s samples rejected, per-pixel counts); turbulence intensity uses each case's
measured free-stream speed.

Edit the settings below, then run from the repo root:
    python tools/compute_flow_properties.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmasher as cmr
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.analysis import (  # noqa: E402
    compute_fluctuating_components,
    load_mean_velocity_components,
    save_arrays,
)
from src.sPIV_PLIF_postprocessing.analysis import flow_properties as FP  # noqa: E402

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASES: list[str] = ["baseline", "buoyant", "diffusive", "fractal", "nearbed", "smSource"]
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
OUTPUT_ROOT = BASE_PATH / "flow_properties"  # outputs: <OUTPUT_ROOT>/Plots/<case>/, <OUTPUT_ROOT>/flow_property_stats_<case>.txt
INTERMITTENCY_TEMPLATE = "Plots/Intermittency/intermittency_{case}.npy"
X_TEMPLATE = "PLIF/{case}_xgrid.npy"
Y_TEMPLATE = "PLIF/{case}_ygrid.npy"
DX = 0.000317  # m, nominal; the actual spacing is read from each case's grid
DT = 0.05  # sec
X_SLICE = slice(0, int(300 / (1000 * DX)))  # rows (y, streamwise)
Y_SLICE = slice(int(50 / (1000 * DX)), int(250 / (1000 * DX)))  # columns (x, cross-stream)
T_SLICE = slice(0, 6000)

NU = 1.5e-5  # kinematic viscosity, m2/s
INTERMITTENCY_THRESHOLD = 0.02  # concentration threshold used to build intermittency
PLUME_ENVELOPE_CUTOFF = 0.10  # intermittency >= this value is considered inside the plume envelope
CMAP = cmr.get_sub_cmap("cmr.rainforest", 0.1, 1.0)
BINSIZE = 10  # bin size along y (rows) for mean profile plots
CHUNK_SIZE = 25  # frames per chunk (gradients for several stencils are held in memory)

GRADIENT_STEPS = [1, 2, 4]  # central-difference half-spans in grid points (spans 0.63, 1.27, 2.54 mm)
PRIMARY_STEP = 1
PRIMARY_ESTIMATOR = "eps_hybrid"  # or "eps_iso_long"

BACKGROUND_MIN_ABS_X_MM = 70.0  # background (for decay check and free-stream speed): |x| >= this and outside envelope
DECAY_WINDOW_MM = 30.0  # Savitzky-Golay window for dk/ds
SF_FRAME_STRIDE = 10  # frames sampled for structure functions
SF_R_PX = [2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 32, 40]
SF_C2 = 2.0  # Kolmogorov constant for the second-order longitudinal structure function
SF_FIT_R_MM = (3.0, 10.0)  # separations used for the (approximate) inertial-range estimate
BANDS_S_MM = [(20, 70), (70, 120), (120, 170), (170, 220), (220, 270)]  # downstream bands for region statistics

VELOCITY_FLUCTUATION_LIMIT = 1.0  # m/s: |q'| above this -> NaN for u, v, w (~3x the free stream). Extends the old
# one-sided w' < -1 rule: two pixels (rows 119/120, cols 161/238) hold a raw w of ~-1.7e18 that also contaminates
# the stored w mean, giving w' ~ +3e14 in every other frame there.
WRITE_FLUCTUATION_STACKS = False  # rewrite flow_properties/flx_u_v_w/*_flx_<case>.npy (~14 GB each)
# The old flow_properties/flx_StrainRates/* stacks hold the pre-fix TRANSVERSE derivatives and are no longer written.
# -------------------------------------------------------------------


def _plot_field(
    arr: np.ndarray,
    title: str,
    out_path: Path,
    *,
    vmin=None,
    vmax=None,
    cmap: str = "viridis",
    log_scale: bool = False,
) -> None:
    """Save a simple pcolormesh plot for a 2D field."""
    plt.figure(figsize=(6, 5))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if log_scale:
        data = np.asarray(arr)
        positive = data[data > 0]
        if positive.size == 0:
            raise ValueError(f"Cannot plot {title} on log scale: no positive values")
        if vmin is None or vmin <= 0:
            vmin = np.percentile(positive, 1)
        if vmax is None:
            vmax = np.percentile(positive, 99)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    plt.pcolormesh(arr, cmap=cmap, shading="auto", norm=norm)
    plt.colorbar(label=title)
    plt.xlabel("x index (cross-stream)")
    plt.ylabel("y index (streamwise)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_line_along_x(arr: np.ndarray, title: str, out_path: Path, *, bin_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Plot the mean profile along the rows (streamwise), averaged across columns, with optional row binning."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = np.nanmean(arr, axis=1)
    if bin_size > 1:
        n_bins = profile.shape[0] // bin_size
        trimmed = profile[: n_bins * bin_size]
        profile = np.nanmean(trimmed.reshape(n_bins, bin_size), axis=1)
        x_vals = np.arange(n_bins) * bin_size + (bin_size - 1) / 2.0
    else:
        x_vals = np.arange(profile.shape[0])
    plt.figure(figsize=(7, 4))
    plt.plot(np.flip(x_vals), profile, lw=1.6)
    plt.xlabel("row index")
    plt.ylabel(title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return x_vals, profile


def _report_stats(name: str, arr: np.ndarray, *, log: list[str] | None = None) -> None:
    """Print mean, median, and selected quantiles for a field and capture them for logging."""
    finite = np.asarray(arr)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        line = f"{name}: no finite values"
        print(line)
        if log is not None:
            log.append(line)
        return
    quantiles = [0, 10, 25, 50, 75, 90, 100]
    values = np.percentile(finite, quantiles)
    formatted = ", ".join(f"{q}%={v:.5g}" for q, v in zip(quantiles, values))
    line = f"{name} mean={np.mean(finite):.5g}, median={np.median(finite):.5g}; quantiles: {formatted}"
    print(line)
    if log is not None:
        log.append(line)


def _report_region_stats(name: str, arr: np.ndarray, mask: np.ndarray, *, log: list[str]) -> None:
    """Report stats inside and outside the plume envelope mask."""
    _report_stats(f"{name} (inside envelope)", arr[mask], log=log)
    _report_stats(f"{name} (outside envelope)", arr[~mask], log=log)


def _check_orientation(intermittency: np.ndarray, case: str) -> None:
    """The plume envelope must narrow toward the source at the high-row end (velocity-stack orientation)."""
    w = (intermittency >= PLUME_ENVELOPE_CUTOFF).sum(axis=1)
    n = w.size
    if not np.nanmean(w[-n // 6:]) < np.nanmean(w[: n // 6]):
        raise ValueError(f"{case}: intermittency envelope is not narrowest at the high-row (source) end; "
                         "is the intermittency map flipped relative to the velocity stacks?")


def _ds_derivative(f: np.ndarray, s_mm: np.ndarray, window_mm: float) -> np.ndarray:
    ds = float(np.mean(np.diff(s_mm))) / 1000
    win = max(5, int(round(window_mm / abs(ds * 1000))) // 2 * 2 + 1)
    fin = np.isfinite(f)
    idx = np.arange(f.size)
    g = np.interp(idx, idx[fin], f[fin])
    d = savgol_filter(g, win, 2, deriv=1, delta=ds, mode="interp")
    d[~fin] = np.nan
    return d


def process_case(case: str) -> None:
    t0 = time.time()
    out_dir = OUTPUT_ROOT / "Plots" / case
    summary_path = OUTPUT_ROOT / f"flow_property_stats_{case}.txt"
    log_lines: list[str] = [f"Flow property stats for case '{case}' (Oct 2026 revision: corrected dissipation estimator)"]
    log_lines.append(f"Intermittency threshold={INTERMITTENCY_THRESHOLD} (concentration), envelope cutoff={PLUME_ENVELOPE_CUTOFF}")

    u_mean_full, v_mean_full, w_mean_full = load_mean_velocity_components(case, base_path=BASE_PATH)
    u_mean, v_mean, w_mean = (m[X_SLICE, Y_SLICE] for m in (u_mean_full, v_mean_full, w_mean_full))
    paths = {c: BASE_PATH / "PIV" / "Interpolated_to_PLIF" / f"piv_{case}_{c}.npy" for c in "uvw"}
    mm = {c: np.load(p, mmap_mode="r") for c, p in paths.items()}
    rows = range(*X_SLICE.indices(mm["u"].shape[1]))
    cols = range(*Y_SLICE.indices(mm["u"].shape[2]))
    t_idx = range(*T_SLICE.indices(mm["u"].shape[0]))
    if t_idx.step != 1 or len(t_idx) == 0:
        raise ValueError("T_SLICE must be a non-empty contiguous slice")
    ny, nx, nt = len(rows), len(cols), len(t_idx)

    xg = np.load(BASE_PATH / X_TEMPLATE.format(case=case)).astype(float)
    yg = np.load(BASE_PATH / Y_TEMPLATE.format(case=case)).astype(float)
    x_mm = xg[0, Y_SLICE]
    y_mm = yg[X_SLICE, 0]
    dx = float(np.mean(np.diff(x_mm))) / 1000
    dy = float(np.mean(np.diff(y_mm))) / 1000
    s_mm = yg[:, 0].max() - y_mm  # downstream distance (source at the high-row end)

    intermittency = np.load(BASE_PATH / INTERMITTENCY_TEMPLATE.format(case=case))
    _check_orientation(intermittency, case)
    plume_mask = intermittency[X_SLICE, Y_SLICE] >= PLUME_ENVELOPE_CUTOFF
    bg_mask = ~plume_mask & (np.abs(x_mm)[None, :] >= BACKGROUND_MIN_ABS_X_MM)
    log_lines.append(f"Grid spacing dx={dx * 1000:.4f} mm, dy={dy * 1000:.4f} mm; plume pixels={int(plume_mask.sum())}, "
                     f"background pixels (|x| >= {BACKGROUND_MIN_ABS_X_MM:g} mm, outside envelope)={int(bg_mask.sum())}")

    sf_masks = {}
    for name, region in (("background", bg_mask), ("plume", plume_mask)):
        for a, b in BANDS_S_MM:
            sf_masks[f"{name}|{a}-{b}mm"] = region & ((s_mm >= a) & (s_mm < b))[:, None]
    sf = FP.StructureFunctionAccumulator(sf_masks, SF_R_PX)
    gms = {st: FP.GradientMoments((ny, nx)) for st in GRADIENT_STEPS}
    sums = {c: np.zeros((ny, nx)) for c in "uvw"}
    counts = {c: np.zeros((ny, nx)) for c in "uvw"}

    if WRITE_FLUCTUATION_STACKS:
        flx_dir = BASE_PATH / "flow_properties" / "flx_u_v_w"
        flx_dir.mkdir(parents=True, exist_ok=True)
        flx_mm = {c: np.lib.format.open_memmap(flx_dir / f"{c}_flx_{case}.npy", mode="w+", dtype=np.float32, shape=(nt, ny, nx))
                  for c in "uvw"}

    for t_start in range(0, nt, CHUNK_SIZE):
        t_stop = min(t_start + CHUNK_SIZE, nt)
        sl = slice(t_idx[t_start], t_idx[t_stop - 1] + 1)
        chunk = {c: np.asarray(mm[c][sl, X_SLICE, Y_SLICE], dtype=np.float64) for c in "uvw"}
        uf, vf, wf = compute_fluctuating_components(chunk["u"], chunk["v"], chunk["w"], u_mean, v_mean, w_mean)
        for a in (uf, vf, wf):
            with np.errstate(invalid="ignore"):
                a[np.abs(a) > VELOCITY_FLUCTUATION_LIMIT] = np.nan
        if WRITE_FLUCTUATION_STACKS:
            for c, a in zip("uvw", (uf, vf, wf)):
                flx_mm[c][t_start:t_stop] = a
        for c, a in zip("uvw", (uf, vf, wf)):
            fin = np.isfinite(a)
            sums[c] += np.where(fin, a * a, 0.0).sum(axis=0)
            counts[c] += fin.sum(axis=0)
        for st in GRADIENT_STEPS:
            gms[st].add(FP.velocity_gradients(uf, vf, wf, dx=dx, dy=dy, step=st))
        sampled = [i for i in range(t_stop - t_start) if (t_start + i) % SF_FRAME_STRIDE == 0]
        if sampled:
            sf.add(uf[sampled], vf[sampled])
        if (t_start // CHUNK_SIZE) % 40 == 0:
            print(f"  [{case}] frames {t_start}-{t_stop} of {nt}", flush=True)
    if WRITE_FLUCTUATION_STACKS:
        for a in flx_mm.values():
            a.flush()

    with np.errstate(invalid="ignore", divide="ignore"):
        ms = {c: np.where(counts[c] > 0, sums[c] / counts[c], np.nan) for c in "uvw"}
    tke = 0.5 * (ms["u"] + ms["v"] + ms["w"])
    log_lines.append(f"Samples rejected (|q'| > {VELOCITY_FLUCTUATION_LIMIT:g} m/s or NaN), relative to u: "
                     f"v {1 - counts['v'].sum() / max(counts['u'].sum(), 1):.2e}, w {1 - counts['w'].sum() / max(counts['u'].sum(), 1):.2e}")

    # ---- dissipation estimates
    estimates, ratios = {}, None
    for st in GRADIENT_STEPS:
        gm = gms[st].means(min_count=max(10, nt // 20))
        for k, v in FP.dissipation_estimates(gm, nu=NU).items():
            estimates[f"{k}_step{st}"] = v
        if st == PRIMARY_STEP:
            ratios = FP.isotropy_ratios(gm)
    epsilon = estimates[f"{PRIMARY_ESTIMATOR}_step{PRIMARY_STEP}"]
    span_mm = 2 * PRIMARY_STEP * abs(dx) * 1000
    eps_path = out_dir / f"epsilon_{case}.npy"
    save_arrays([(eps_path, epsilon)])
    np.savez_compressed(out_dir / f"epsilon_estimates_{case}.npz", **estimates, **{f"isotropy_{k}": v for k, v in ratios.items()},
                        s_mm=s_mm, x_mm=x_mm, plume_mask=plume_mask, background_mask=bg_mask)
    finite_eps = epsilon[np.isfinite(epsilon)]
    _plot_field(epsilon, f"epsilon [m2/s3] ({PRIMARY_ESTIMATOR}, span {span_mm:.2f} mm)", eps_path.with_suffix(".png"),
                vmin=np.percentile(finite_eps, 1), vmax=np.percentile(finite_eps, 99), cmap=CMAP)
    log_lines.append(f"Primary epsilon: {PRIMARY_ESTIMATOR}, central-difference span {span_mm:.2f} mm")
    _report_stats("epsilon", epsilon, log=log_lines)
    _report_region_stats("epsilon", epsilon, plume_mask, log=log_lines)
    for key, v in estimates.items():
        log_lines.append(f"  {key}: median inside envelope {np.nanmedian(v[plume_mask]):.4g}, background {np.nanmedian(v[bg_mask]):.4g}")
    for key, v in ratios.items():
        log_lines.append(f"  isotropy {key}: median inside {np.nanmedian(v[plume_mask]):.3f}, background {np.nanmedian(v[bg_mask]):.3f} "
                         "(isotropic: trans/long 2, dwdz/long 1, cross/long -0.5)")

    # ---- cross-check 1: background TKE decay
    k_bg = np.nanmean(np.where(bg_mask, tke, np.nan), axis=1)
    U_bg = np.abs(np.nanmean(np.where(bg_mask, v_mean, np.nan), axis=1))
    eps_decay = -U_bg * _ds_derivative(k_bg, s_mm, DECAY_WINDOW_MM)
    eps_bg_profile = np.nanmedian(np.where(bg_mask, epsilon, np.nan), axis=1)
    eps_pl_profile = np.nanmedian(np.where(plume_mask, epsilon, np.nan), axis=1)
    # ---- cross-check 2: structure functions
    sf_res = sf.results(abs(dx), c2=SF_C2)
    sf_eps, sf_ratio = {}, {}
    log_lines.append("Cross-checks by downstream band (medians). Structure functions: at Re_lambda ~ 20-60 there is no inertial "
                     f"range (lambda ~ 10 mm), so the inertial estimate (D_LL/C2)^1.5/r over r={SF_FIT_R_MM} mm is NOT a valid "
                     "epsilon; reported instead: D_LL/(eps_grad r^2/(15 nu)) at r = 0.95 mm (dissipation-range consistency, ~1 "
                     "if gradients and structure functions agree; < 1 where PIV filtering attenuates small separations)")
    for a, b in BANDS_S_MM:
        band = (s_mm >= a) & (s_mm < b)
        for name, prof in (("background", eps_bg_profile), ("plume", eps_pl_profile)):
            e_g = float(np.nanmedian(prof[band]))
            parts = []
            for comp in "uv":
                r = sf_res[(comp, f"{name}|{a}-{b}mm")]
                sel = (r["r_m"] * 1000 >= SF_FIT_R_MM[0]) & (r["r_m"] * 1000 <= SF_FIT_R_MM[1])
                sf_eps[(comp, f"{name}|{a}-{b}mm")] = float(np.nanmedian(r["eps_r"][sel]))
                i = int(np.argmin(np.abs(r["r_m"] * 1000 - 0.95)))
                ratio = r["D_LL"][i] / (e_g * r["r_m"][i] ** 2 / (15 * NU))
                sf_ratio[(comp, f"{name}|{a}-{b}mm")] = float(ratio)
                parts.append(f"{comp} {ratio:.2f}")
            line = f"  s={a}-{b} mm {name:10s}: eps gradient={e_g:.3g}"
            if name == "background":
                line += f", TKE decay U|dk/ds|={np.nanmedian(eps_decay[band]):.3g}"
            line += ", structure-fn dissipation-range ratio " + ", ".join(parts)
            log_lines.append(line)
    np.savez_compressed(out_dir / f"epsilon_crosscheck_{case}.npz", s_mm=s_mm, eps_decay=eps_decay, k_bg=k_bg, U_bg=U_bg,
                        eps_bg_profile=eps_bg_profile, eps_plume_profile=eps_pl_profile,
                        **{f"sf_{c}_{n}_{k}".replace("|", "_"): v for (c, n), r in sf_res.items() for k, v in r.items()})

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    for key, ls in ((f"eps_hybrid_step{PRIMARY_STEP}", "-"), (f"eps_iso_long_step{PRIMARY_STEP}", "--")):
        for name, mask, col in (("background", bg_mask, "tab:blue"), ("plume", plume_mask, "tab:red")):
            ax.semilogy(s_mm, np.nanmedian(np.where(mask, estimates[key], np.nan), axis=1), color=col, ls=ls, lw=1.1,
                        label=f"{key.split('_step')[0]}, {name}")
    for st in GRADIENT_STEPS:
        if st != PRIMARY_STEP:
            ax.semilogy(s_mm, np.nanmedian(np.where(bg_mask, estimates[f"eps_hybrid_step{st}"], np.nan), axis=1), color="tab:blue",
                        ls=":", lw=0.8, label=f"eps_hybrid background, span {2 * st * abs(dx) * 1000:.2f} mm")
    ax.semilogy(s_mm, np.where(eps_decay > 0, eps_decay, np.nan), color="k", lw=1.4, label="TKE decay U|dk/ds| (background)")
    ax.set_xlabel("downstream distance s [mm]")
    ax.set_ylabel(r"$\varepsilon$ [m$^2$/s$^3$]")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")
    ax = axs[1]
    a_b = BANDS_S_MM[2]
    band = (s_mm >= a_b[0]) & (s_mm < a_b[1])
    for (comp, name), r in sf_res.items():
        if name.endswith(f"{a_b[0]}-{a_b[1]}mm"):
            col = "tab:blue" if name.startswith("background") else "tab:red"
            ax.loglog(r["r_m"] * 1000, r["D_LL"], marker="o" if comp == "u" else "s", ms=3, color=col, label=f"D_LL {comp}, {name}")
    rr = np.array(SF_R_PX) * abs(dx)
    for name, prof, col in (("background", eps_bg_profile, "tab:blue"), ("plume", eps_pl_profile, "tab:red")):
        e_g = float(np.nanmedian(prof[band]))
        ax.loglog(rr * 1000, e_g * rr**2 / (15 * NU), color=col, ls="--", lw=0.8, label=f"eps_grad r^2/(15 nu), {name}")
        ax.loglog(rr * 1000, SF_C2 * (e_g * rr) ** (2 / 3), color=col, ls=":", lw=0.8, label=f"C2 (eps_grad r)^(2/3), {name}")
    ax.set_xlabel("separation r [mm]")
    ax.set_ylabel(r"$D_{LL}(r)$ [m$^2$/s$^2$]")
    ax.set_title(f"structure functions at s = {a_b[0]}-{a_b[1]} mm vs dissipation-range (dashed) and inertial (dotted) forms",
                 fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"{case}: dissipation-rate estimates and cross-checks")
    fig.tight_layout()
    fig.savefig(out_dir / f"epsilon_crosscheck_{case}.png", dpi=300)
    plt.close(fig)

    # ---- scales from the primary epsilon
    sc = FP.turbulence_scales(epsilon, tke, nu=NU)
    names = {"taylor_microscale": "Taylor_microscale", "taylor_Re": "Taylor_Re", "kolmogorov_length": "kolmogorov_length_scale",
             "kolmogorov_time": "kolmogorov_time_scale"}
    labels = {"taylor_microscale": "Taylor microscale", "taylor_Re": "Taylor Re", "kolmogorov_length": "Kolmogorov length scale",
              "kolmogorov_time": "Kolmogorov time scale"}
    for key, fname in names.items():
        arr = sc[key]
        path = out_dir / f"{fname}_{case}.npy"
        save_arrays([(path, arr)])
        _plot_field(arr, labels[key], path.with_suffix(".png"), vmin=0, vmax=np.percentile(arr[np.isfinite(arr)], 99))
        _report_stats(labels[key], arr, log=log_lines)
        _report_region_stats(labels[key], arr, plume_mask, log=log_lines)

    # ---- TKE and turbulence intensity
    tke_path = out_dir / f"tke_{case}.npy"
    save_arrays([(tke_path, tke)])
    _plot_field(tke, "TKE", tke_path.with_suffix(".png"), vmin=0, vmax=np.percentile(tke[np.isfinite(tke)], 99), cmap=CMAP)
    x_tke, tke_profile = _plot_line_along_x(tke, f"Mean TKE along y (avg over x, binned {BINSIZE} rows)",
                                            tke_path.with_suffix(".profile.png"), bin_size=BINSIZE)
    np.savez(tke_path.with_suffix(".profile.npz"), x_index_center=x_tke, tke_profile=tke_profile)
    _report_stats("TKE", tke, log=log_lines)
    _report_region_stats("TKE", tke, plume_mask, log=log_lines)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_lines.append(f"anisotropy ratios: <u'^2>/TKE={np.nanmean(ms['u'] / tke):.4f}, <v'^2>/TKE={np.nanmean(ms['v'] / tke):.4f}, "
                         f"<w'^2>/TKE={np.nanmean(ms['w'] / tke):.4f}")
    U_inf = float(np.nanmedian(np.abs(v_mean[bg_mask])))
    t_intensity = sc["u_prime"] / U_inf
    ti_path = out_dir / f"turbulence_intensity_{case}.npy"
    save_arrays([(ti_path, t_intensity)])
    _plot_field(t_intensity, f"Turbulence intensity (u'/U_inf, U_inf={U_inf:.3f} m/s)", ti_path.with_suffix(".png"),
                vmin=0.0, vmax=0.4, cmap=CMAP)
    x_profile, ti_profile = _plot_line_along_x(t_intensity, f"Mean turbulence intensity along y (binned {BINSIZE} rows)",
                                               out_dir / f"turbulence_intensity_profile_{case}.png", bin_size=BINSIZE)
    np.savez(out_dir / f"turbulence_intensity_profile_{case}.npz", x_index_center=x_profile, ti_profile=ti_profile)
    log_lines.append(f"Free-stream speed U_inf (median |V| in background) = {U_inf:.4f} m/s")
    _report_stats("turbulence intensity", t_intensity, log=log_lines)
    _report_region_stats("turbulence intensity", t_intensity, plume_mask, log=log_lines)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[{case}] done in {(time.time() - t0) / 60:.1f} min; summary -> {summary_path}")


def main() -> None:
    for case in CASES:
        process_case(case)


if __name__ == "__main__":
    main()
