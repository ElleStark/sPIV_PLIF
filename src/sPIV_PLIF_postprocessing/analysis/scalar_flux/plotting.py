"""Figures for the scalar-flux analysis. Axes are physical: x [mm] cross-stream (horizontal), y [mm] streamwise
grid coordinate (vertical, source at the top), matching the orientation of the existing repo figures."""
from __future__ import annotations

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm, Normalize, SymLogNorm

from .geometry import Grid, Point

SIGNED_CMAP = "RdBu_r"  # as TOTAL_COMPONENT_CMAP in tools/plot_scalar_fluxes.py
MAG_CMAP = cmr.get_sub_cmap(cmr.rainforest, 0.0, 0.85)
MASK_CMAP = ListedColormap([(0.6, 0.6, 0.6, 0.75)])
FLUX_LABELS = {"u": r"$\langle u'c'\rangle$", "v": r"$\langle v'c'\rangle$", "w": r"$\langle w'c'\rangle$"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "savefig.bbox": "tight"})


def _map(ax, grid: Grid, field: np.ndarray, *, cmap, norm, title: str, mask: np.ndarray | None = None):
    h = ax.pcolormesh(grid.x_mm, grid.y_mm, field, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    if mask is not None and mask.any():
        ax.pcolormesh(grid.x_mm, grid.y_mm, np.ma.masked_where(~mask, np.ones_like(field)), cmap=MASK_CMAP,
                      shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    return h


def _save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _symlog_limits(fields: list[np.ndarray], lin_frac: float = 0.02) -> tuple[float, float]:
    vals = np.concatenate([np.abs(f[np.isfinite(f)]) for f in fields])
    vmax = float(np.percentile(vals, 99.5)) if vals.size else 1.0
    return vmax, vmax * lin_frac


def plot_flux_maps(grid, mom, signif, case, path, dpi, points: list[Point] = ()):
    """<u'c'>, <v'c'>, <w'c'> with non-significant pixels (|flux| < k SE) greyed out."""
    fields = [mom[f"{q}c_flux"] for q in "uvw"]
    vmax, lin = _symlog_limits(fields)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, q, f in zip(axs, "uvw", fields):
        h = _map(ax, grid, f, cmap=SIGNED_CMAP, norm=SymLogNorm(lin, vmin=-vmax, vmax=vmax, base=10),
                 title=f"{FLUX_LABELS[q]} [m/s]", mask=~signif[q] & np.isfinite(f))
        for p in points:
            ax.plot(p.x_mm, p.y_mm, "k+", ms=6, mew=1)
    axs[0].set_ylabel("y [mm] (source at top)")
    fig.colorbar(h, ax=axs, shrink=0.8, label="flux [m/s]  (grey: |flux| < SE, not significant)")
    fig.suptitle(f"{case}: fixed-frame turbulent scalar fluxes")
    _save(fig, path, dpi)


def plot_rho_maps(grid, mom, case, path, dpi):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, q in zip(axs, "uvw"):
        h = _map(ax, grid, mom[f"{q}c_rho"], cmap=SIGNED_CMAP, norm=Normalize(-0.6, 0.6),
                 title=rf"$\rho_{{{q}c}}$")
    axs[0].set_ylabel("y [mm] (source at top)")
    fig.colorbar(h, ax=axs, shrink=0.8, label="correlation coefficient")
    fig.suptitle(f"{case}: velocity-scalar correlation coefficients")
    _save(fig, path, dpi)


def plot_rms_maps(grid, mom, c_rms_corr, case, path, dpi):
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, (key, f, label) in zip(axs, [("u", mom["u_rms"], "u_rms [m/s]"), ("v", mom["v_rms"], "v_rms [m/s]"),
                                         ("w", mom["w_rms"], "w_rms [m/s]"), ("c", c_rms_corr, "c_rms (noise removed)")]):
        vals = f[np.isfinite(f)]
        h = _map(ax, grid, f, cmap=MAG_CMAP, norm=Normalize(0, np.percentile(vals, 99.5)), title=label)
        fig.colorbar(h, ax=ax, shrink=0.7)
    axs[0].set_ylabel("y [mm] (source at top)")
    fig.suptitle(f"{case}: rms fields")
    _save(fig, path, dpi)


def plot_uncertainty_maps(grid, mom, unc, case, path, dpi):
    """Bootstrap SE and N_eff = var(p)/SE^2 for each flux."""
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.4), sharex=True, sharey=True)
    for j, q in enumerate("uvw"):
        se = unc[f"{q}c_se"]
        vals = se[np.isfinite(se) & (se > 0)]
        h = _map(axs[0, j], grid, se, cmap=MAG_CMAP, norm=LogNorm(*np.percentile(vals, [1, 99.5])),
                 title=f"bootstrap SE of {FLUX_LABELS[q]} [m/s]")
        fig.colorbar(h, ax=axs[0, j], shrink=0.7)
        ne = unc[f"{q}c_neff"]
        h = _map(axs[1, j], grid, ne, cmap=MAG_CMAP, norm=LogNorm(10, max(100, np.nanpercentile(ne, 99))),
                 title=r"$N_{eff}$ = var(p)/SE$^2$")
        fig.colorbar(h, ax=axs[1, j], shrink=0.7)
    for ax in axs[:, 0]:
        ax.set_ylabel("y [mm]")
    fig.suptitle(f"{case}: flux uncertainty (moving-block bootstrap)")
    _save(fig, path, dpi)


def plot_time_scales(grid, tint, coarse_rows, coarse_cols, case, path, dpi, block_s: float, region=None):
    """Per-panel colour limits from the 99th percentile inside ``region`` (plume); values above are clipped
    (e.g. slow background drift of c outside the plume)."""
    keys = [k for k in ("c", "u", "v", "uc", "vc", "wc") if f"T_{k}" in tint]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 4.6), sharey=True)
    xs, ys = grid.x_mm[coarse_cols], grid.y_mm[coarse_rows]
    region = np.ones_like(tint[f"T_{keys[0]}"], dtype=bool) if region is None else region
    for ax, k in zip(axs, keys):
        T = tint[f"T_{k}"]
        vmax = float(np.nanpercentile(T[region], 99))
        h = ax.pcolormesh(xs, ys, T, cmap=MAG_CMAP, norm=Normalize(0, vmax), shading="auto")
        fig.colorbar(h, ax=ax, orientation="horizontal", pad=0.15, shrink=0.9, label="[s]")
        nc = ~tint[f"crossed_{k}"]
        if nc.any():
            ax.pcolormesh(xs, ys, np.ma.masked_where(~nc, np.ones(nc.shape)), cmap=MASK_CMAP, shading="auto")
        label = k + "'" if len(k) == 1 else f"{k[0]}'{k[1]}'"
        ax.set_title(f"T_int of {label}\n(plume median {np.nanmedian(T[region]):.2f} s)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")
    fig.suptitle(f"{case}: integral time scales (coarse grid; colour clipped at plume p99; grey: rho never "
                 f"crossed zero); bootstrap block = {block_s:.1f} s")
    _save(fig, path, dpi)


def plot_convergence(points: list[Point], running: dict, final: dict, se: dict, dt: float, case, path, dpi):
    """Running flux estimate vs. record length at each point, with final value +/- SE and the SE*sqrt(N/n)
    envelope expected for a converging mean."""
    fig, axs = plt.subplots(len(points), 3, figsize=(11, 2.1 * len(points)), sharex=True, squeeze=False)
    for i, p in enumerate(points):
        for j, q in enumerate("uvw"):
            ax = axs[i, j]
            r = running[q][:, i]
            n = np.arange(1, r.size + 1)
            t = n * dt
            f, s = final[q][i], se[q][i]
            ax.plot(t, r, color="k", lw=0.8)
            ax.axhline(f, color="tab:blue", lw=0.8)
            ax.fill_between(t, f - s, f + s, color="tab:blue", alpha=0.25, lw=0)
            env = s * np.sqrt(r.size / n)
            ax.plot(t, f + env, color="tab:blue", lw=0.5, ls=":")
            ax.plot(t, f - env, color="tab:blue", lw=0.5, ls=":")
            lim = 4 * max(abs(f), s) + 1e-12
            ax.set_ylim(f - lim, f + lim)
            if i == 0:
                ax.set_title(FLUX_LABELS[q] + " [m/s]")
            if j == 0:
                ax.set_ylabel(f"{p.name}\n(x={p.x_mm:.0f}, s={p.downstream_mm:.0f} mm)", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("record length [s]")
    fig.suptitle(f"{case}: running flux estimates (band: final +/- SE; dotted: SE*sqrt(N/n))")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_gradient_sensitivity(grid, sens: dict, case, path, dpi, stations_mm: list[float]):
    """Maps of dC/dx, dC/dy (symmetric-log colour, separate scales) for each smoothing scale, and cross-stream
    profiles at each station (one panel per station, so far-field profiles are not hidden by the near field)."""
    sig = sorted(sens)
    ns = len(stations_mm)
    fig = plt.figure(figsize=(max(4 * len(sig), 3.6 * ns), 15))
    gs = fig.add_gridspec(4, max(len(sig), ns), height_ratios=[1, 1, 0.55, 0.55])
    s_down = grid.downstream_mm
    for i, key in enumerate(("dCdx", "dCdy")):
        gmax, lin = _symlog_limits([sens[s][key] for s in sig], lin_frac=0.01)
        axs_row = []
        for j, s in enumerate(sig):
            ax = fig.add_subplot(gs[i, j])
            h = _map(ax, grid, sens[s][key], cmap=SIGNED_CMAP, norm=SymLogNorm(lin, vmin=-gmax, vmax=gmax, base=10),
                     title=f"{'dC/dx' if key == 'dCdx' else 'dC/dy'} [1/m], smoothing {s:g} mm")
            for st in stations_mm:
                ax.axhline(grid.y_mm[int(np.argmin(np.abs(s_down - st)))], color="k", lw=0.4, ls="--")
            if j == 0:
                ax.set_ylabel("y [mm]")
            axs_row.append(ax)
        fig.colorbar(h, ax=axs_row, shrink=0.8, label="[C/m]")
    styles = ["-", "--", ":", "-."]
    for i, key in enumerate(("dCdx", "dCdy")):
        for j, st in enumerate(stations_mm):
            ax = fig.add_subplot(gs[2 + i, j])
            row = int(np.argmin(np.abs(s_down - st)))
            for k, s in enumerate(sig):
                ax.plot(grid.x_mm, sens[s][key][row], color="k", ls=styles[k % 4], lw=0.9, label=f"{s:g} mm")
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_title(f"{'dC/dx' if key == 'dCdx' else 'dC/dy'} at s = {st:g} mm", fontsize=8)
            ax.set_xlabel("x [mm]")
            if j == 0:
                ax.set_ylabel("[1/m]")
                ax.legend(title="smoothing", fontsize=7)
    fig.suptitle(f"{case}: mean-gradient sensitivity to smoothing scale (dashed lines: profile stations)")
    _save(fig, path, dpi)


# ============================================================================ stage 2: gradient diffusion / K
COMP = {"x": ("u", "dC/dx", r"$K_x$"), "y": ("v", "dC/dy", r"$K_y$")}


def plot_gradient_diffusion_scatter(grid, flux, grad, valid, rows_per_station, centers_mm, xc, sigma, test,
                                    comp: str, case, path, dpi):
    """Flux vs. mean gradient at each station, coloured by cross-stream position (x - x_c)/sigma_x, with the
    through-origin (solid) and with-intercept (dashed) fits."""
    q, glabel, _ = COMP[comp]
    n = len(centers_mm)
    ncol = min(3, n)
    nrow = -(-n // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    xnorm = (grid.x_mm[None, :] - xc[:, None]) / sigma[:, None]
    for i, (rows, c) in enumerate(zip(rows_per_station, centers_mm)):
        ax = axs.flat[i]
        m = valid[rows] & np.isfinite(flux[rows]) & np.isfinite(grad[rows])
        g, f, xn = grad[rows][m], flux[rows][m], xnorm[rows][m]
        h = ax.scatter(g, f, c=xn, s=2, cmap=SIGNED_CMAP, vmin=-2.5, vmax=2.5, rasterized=True)
        if g.size:
            gl = np.array([np.nanmin(g), np.nanmax(g)])
            ax.plot(gl, -test["K0"][i] * gl, "k-", lw=1)
            ax.plot(gl, test["intercept1"][i] - test["K1"][i] * gl, "k--", lw=0.8)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_title(f"s = {c:g} mm: K0 = {test['K0'][i]:.2e} m$^2$/s, R$^2$ = {test['r2_0'][i]:.2f}\n"
                     f"down-gradient {100 * test['frac_down'][i]:.0f}%  (n = {int(test['n'][i])})", fontsize=8)
        ax.set_xlabel(f"{glabel} [1/m]")
        ax.set_ylabel(f"{FLUX_LABELS[q]} [m/s]")
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.colorbar(h, ax=axs, shrink=0.6, label=r"$(x - x_c)/\sigma_x$")
    fig.suptitle(f"{case}: gradient-diffusion test, {FLUX_LABELS[q]} vs {glabel} "
                 "(solid: fit through origin, dashed: with intercept; significant fluxes, plume pixels)")
    _save(fig, path, dpi)


def plot_gradient_diffusion_summary(centers_mm, tests: dict, sens_K: dict, case, path, dpi):
    """Down-gradient fraction, binned K (+/- formal SE) and R^2 vs. downstream distance, plus smoothing sensitivity."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for comp, color in (("x", "tab:blue"), ("y", "tab:red")):
        t = tests[comp]
        lab = COMP[comp][2]
        axs[0, 0].plot(centers_mm, t["frac_down"], "o-", color=color, ms=3, label=f"{lab} ({COMP[comp][0]}'c' vs {COMP[comp][1]})")
        axs[0, 1].errorbar(centers_mm, t["K0"], yerr=t["K0_se"], fmt="o-", color=color, ms=3, lw=1, capsize=2,
                           label=f"{lab}, through origin")
        axs[0, 1].plot(centers_mm, t["K1"], "s--", color=color, ms=3, lw=0.8, mfc="none", label=f"{lab}, with intercept")
        axs[1, 0].plot(centers_mm, t["r2_0"], "o-", color=color, ms=3, label=f"{lab}, through origin (uncentered)")
        axs[1, 0].plot(centers_mm, t["r2_1"], "s--", color=color, ms=3, mfc="none", label=f"{lab}, with intercept")
        for k, (s, K) in enumerate(sorted(sens_K[comp].items())):
            axs[1, 1].plot(centers_mm, K, ["-", "--", ":"][k % 3], color=color, lw=1, label=f"{lab}, smoothing {s:g} mm")
    axs[0, 0].axhline(0.5, color="0.6", lw=0.6, ls=":")
    axs[0, 0].set_ylim(0, 1.02)
    axs[0, 0].set_ylabel("down-gradient fraction")
    axs[0, 1].set_ylabel("K [m$^2$/s]")
    axs[1, 0].set_ylabel("R$^2$")
    axs[1, 1].set_ylabel("K (through origin) [m$^2$/s]")
    for ax in axs.flat:
        ax.axhline(0, color="0.8", lw=0.5)
        ax.legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("downstream distance s [mm]")
    fig.suptitle(f"{case}: gradient-diffusion summary by streamwise bin (error bars: formal SE, independent-pixel "
                 "assumption, understates uncertainty)")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_K_maps(grid, maps: dict, region: np.ndarray, case, path, dpi, title_suffix=""):
    """K maps (symmetric-log diverging; negative = counter-gradient); grey = outside plume or masked."""
    names = list(maps)
    vals = np.concatenate([np.abs(maps[k][np.isfinite(maps[k])]) for k in names])
    vmax = float(np.percentile(vals, 99)) if vals.size else 1e-3
    lin = vmax * 0.01
    ncol = 2
    nrow = -(-len(names) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, k in zip(axs.flat, names):
        K = maps[k]
        h = _map(ax, grid, K, cmap=SIGNED_CMAP, norm=SymLogNorm(lin, vmin=-vmax, vmax=vmax, base=10), title=k,
                 mask=region & ~np.isfinite(K))
        neg = np.mean(K[region & np.isfinite(K)] < 0) if np.any(region & np.isfinite(K)) else np.nan
        cov = np.mean(np.isfinite(K[region]))
        ax.text(0.02, 0.02, f"K<0: {100 * neg:.0f}% of valid\nvalid: {100 * cov:.0f}% of plume", transform=ax.transAxes,
                fontsize=7, va="bottom", bbox=dict(fc="w", alpha=0.7, lw=0))
    for ax in axs[:, 0]:
        ax.set_ylabel("y [mm]")
    for ax in axs.flat[len(names):]:
        ax.axis("off")
    fig.colorbar(h, ax=axs, shrink=0.7, label="K [m$^2$/s] (blue: counter-gradient; grey: masked inside plume)")
    fig.suptitle(f"{case}: effective eddy diffusivity{title_suffix}")
    _save(fig, path, dpi)


def plot_K_profiles(s_mm, profiles: dict, centers_mm, tests: dict, case, path, dpi, ylabel="K [m$^2$/s]",
                    title="downstream profiles of cross-stream-median K"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, comp in zip(axs, ("x", "y")):
        for name, prof in profiles[comp].items():
            ax.plot(s_mm, prof, lw=1, label=name)
        if tests is not None:
            ax.plot(centers_mm, tests[comp]["K0"], "ko", ms=3, label="binned regression (through origin)")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("downstream distance s [mm]")
        ax.set_ylabel(ylabel)
        ax.set_title(COMP[comp][2])
        ax.legend(fontsize=7)
    fig.suptitle(f"{case}: {title}")
    fig.tight_layout()
    _save(fig, path, dpi)


def similarity_profiles(field: np.ndarray, grid, rows_per_station, xc, sigma, *, xn_edges=None):
    """Bin-average ``field`` against eta = (x - x_c(y)) / sigma_x(y) over the rows of each station.
    Returns eta centres and a (n_station, n_eta) array of means (NaN where no data)."""
    xn_edges = np.arange(-3.0, 3.0001, 0.1) if xn_edges is None else xn_edges
    eta_c = 0.5 * (xn_edges[1:] + xn_edges[:-1])
    out = np.full((len(rows_per_station), eta_c.size), np.nan)
    for i, rows in enumerate(rows_per_station):
        eta = (grid.x_mm[None, :] - xc[rows, None]) / sigma[rows, None]
        f = field[rows]
        m = np.isfinite(f) & np.isfinite(eta)
        idx = np.digitize(eta[m], xn_edges) - 1
        ok = (idx >= 0) & (idx < eta_c.size)
        s = np.bincount(idx[ok], weights=f[m][ok], minlength=eta_c.size)
        n = np.bincount(idx[ok], minlength=eta_c.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(n > 0, s / n, np.nan)
    return eta_c, out


def plot_rho_profiles(eta, rho: dict, rho_err: dict, centers_mm, sigma_at: list[float], case, path, dpi):
    """rho_qc vs (x - x_c)/sigma_x at each station; band = +/- 2 SE of the flux divided by q_rms c_rms."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(centers_mm)))
    for ax, q in zip(axs, "uvw"):
        for i, (c, col) in enumerate(zip(centers_mm, colors)):
            r, e = rho[q][i], rho_err[q][i]
            ax.plot(eta, r, color=col, lw=1.2, label=rf"s = {c:g} mm ($\sigma_x$ = {sigma_at[i]:.0f} mm)")
            ax.fill_between(eta, r - 2 * e, r + 2 * e, color=col, alpha=0.2, lw=0)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_title(rf"$\rho_{{{q}c}}$")
        ax.set_xlabel(r"$(x - x_c)/\sigma_x$")
    axs[0].set_ylabel("correlation coefficient")
    axs[0].set_ylim(-1, 1)
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{case}: velocity-scalar correlation coefficients vs. normalized cross-stream position "
                 "(bands: +/- 2 SE from the flux bootstrap)")
    fig.tight_layout()
    _save(fig, path, dpi)


# ============================================================================ cross-case comparison
def plot_case_comparison(s_by_case: dict, prof_by_case: dict, met_by_case: dict, bins_by_case: dict, path, dpi):
    """Six panels vs downstream distance: sigma_x and L_u,x; sigma/L; flank |rho_uc|; binned K_x and K_y;
    K_x/(u_rms L). Vertical lines: transition estimates (solid s_sigma_L, dashed s_rho_50 with shaded 25-75% zone,
    dotted s_K_90)."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = dict(zip(s_by_case, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]))
    for case, s in s_by_case.items():
        p, m, b, col = prof_by_case[case], met_by_case[case], bins_by_case[case], colors[case]
        axs[0, 0].plot(s, p["sigma_x_mm"], color=col, lw=1.2, label=rf"{case} $\sigma_x$")
        axs[0, 0].plot(s, p["L_ux_mm"], color=col, lw=1.2, ls="--", label=rf"{case} $L_{{u,x}}$")
        axs[0, 1].plot(s, p["sigma_over_L"], color=col, lw=1.2, label=case)
        axs[0, 2].plot(s, p["rho_flank"], color=col, lw=1.2, label=case)
        axs[1, 0].plot(b["centers"], b["Kx"], "o-", color=col, ms=3, lw=1, label=rf"{case} $K_x$")
        axs[1, 1].plot(b["centers"], b["Ky"], "s--", color=col, ms=3, lw=0.8, mfc="none", label=rf"{case} $K_y$")
        axs[1, 2].plot(s, p["Kx_nd"], color=col, lw=1.2, label=case)
        for ax in axs.flat:
            if np.isfinite(m["s_sigma_L"]):
                ax.axvline(m["s_sigma_L"], color=col, lw=0.9, ls="-", alpha=0.7)
            if np.isfinite(m["s_rho_50"]):
                ax.axvline(m["s_rho_50"], color=col, lw=0.9, ls="--", alpha=0.7)
            if np.isfinite(m["s_rho_25"]) and np.isfinite(m["s_rho_75"]):
                ax.axvspan(m["s_rho_25"], m["s_rho_75"], color=col, alpha=0.07, lw=0)
            if np.isfinite(m["s_K_90"]):
                ax.axvline(m["s_K_90"], color=col, lw=0.9, ls=":", alpha=0.7)
        axs[0, 2].axhline(0.5 * (m["rho_flank_near"] + m["rho_flank_far"]), color=col, lw=0.5, ls="--")
    axs[0, 1].axhline(1.0, color="0.5", lw=0.6)
    axs[0, 0].set_ylabel("[mm]")
    axs[0, 1].set_ylabel(r"$\sigma_x / L_{u,x}$")
    axs[0, 2].set_ylabel(r"flank $|\rho_{uc}|$  (0.5 $\leq |\eta| \leq$ 1.5)")
    axs[1, 0].set_ylabel(r"$K_x$ (binned regression) [m$^2$/s]")
    axs[1, 1].set_ylabel(r"$K_y$ (binned, through origin) [m$^2$/s]")
    axs[1, 2].set_ylabel(r"$K_x / (u_{rms} L_{u,x})$")
    for ax in axs.flat:
        ax.legend(fontsize=7)
        ax.axhline(0, color="0.85", lw=0.5)
    for ax in axs[1]:
        ax.set_xlabel("downstream distance s [mm]")
    fig.suptitle(r"Case comparison. Transition estimates: solid = $\sigma_x = L_{u,x}$, dashed = flank $\rho_{uc}$ "
                 r"half-decay (shaded 25-75%), dotted = 90% of $K_x/(u_{rms}L)$ plateau")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_rho_profiles_comparison(rho_by_case: dict, path, dpi):
    """rho_uc vs eta at each station, one panel per case, shared axes."""
    cases = list(rho_by_case)
    fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2), sharey=True, squeeze=False)
    for ax, case in zip(axs[0], cases):
        r = rho_by_case[case]
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(r["station_centers_mm"])))
        for i, (c, col) in enumerate(zip(r["station_centers_mm"], colors)):
            ax.plot(r["eta"], r["rho_uc"][i], color=col, lw=1.2, label=rf"s = {c:g} mm ($\sigma_x$ = {r['sigma_x_mm'][i]:.0f} mm)")
            ax.fill_between(r["eta"], r["rho_uc"][i] - 2 * r["rho_uc_err"][i], r["rho_uc"][i] + 2 * r["rho_uc_err"][i],
                            color=col, alpha=0.2, lw=0)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_title(rf"{case}: $\rho_{{uc}}$")
        ax.set_xlabel(r"$(x - x_c)/\sigma_x$")
        ax.legend(fontsize=7, loc="upper left")
    axs[0, 0].set_ylabel("correlation coefficient")
    axs[0, 0].set_ylim(-0.8, 0.8)
    fig.tight_layout()
    _save(fig, path, dpi)


# ============================================================================ stage 3: spreading, spectra, overlay
def plot_spreading(s_mm, d: dict, bins_s, K_FG, case, path, dpi, flag_s_mm=None, max_valid_s_mm=None):
    """sigma_x (moment, Gaussian fit), U, scalar mass flux, and the three K_x estimates vs downstream distance."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax = axs[0, 0]
    ax.plot(s_mm, d["sigma_mom_mm"], lw=1.2, label=r"$\sigma_x$ second moment")
    ax.plot(s_mm, d["sigma_g_mm"], lw=1.2, label=r"$\sigma_x$ Gaussian fit (+ offset)")
    ax.set_ylabel(r"$\sigma_x$ [mm]")
    ax.set_ylim(0, 1.3 * np.nanmax(d["sigma_mom_mm"]))
    ax2 = ax.twinx()
    ax2.plot(s_mm, d["b0"], color="0.6", lw=0.8, ls=":")
    ax2.axhline(d["background"], color="0.6", lw=0.6, ls="--")
    ax2.set_ylabel("Gaussian-fit offset (dotted) / strip background (dashed)", color="0.5", fontsize=7)
    ax = axs[0, 1]
    ax.plot(s_mm, d["Q_rel"], lw=1.2, label=r"$\int |V| (C-b)\,dx$ / value at s = 30 mm")
    ax.plot(s_mm, d["intC_rel"], lw=1.0, ls="--", label=r"$\int (C-b)\,dx$ / value at s = 30 mm")
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_ylabel("relative in-plane scalar flux")
    ax3 = ax.twinx()
    ax3.plot(s_mm, d["U"], color="tab:green", lw=0.8)
    ax3.set_ylabel("U = C-weighted |V| [m/s]", color="tab:green")
    ax = axs[1, 0]
    ax.plot(s_mm, d["K_T_mom"], lw=1.2, label=r"Taylor $\frac{1}{2}U\,d\sigma^2/ds$ (moment $\sigma$)")
    ax.plot(s_mm, d["K_T_g"], lw=1.2, label=r"Taylor (Gaussian $\sigma$)")
    ax.plot(s_mm, d["K_FM"], lw=1.2, label=r"flux moment $\int x\langle u'c'\rangle dx / \int C dx$")
    ax.plot(bins_s, K_FG, "ko", ms=3, label="flux-gradient (binned regression)")
    ax.set_ylabel(r"$K_x$ [m$^2$/s]")
    kref = np.nanmax(np.abs(np.asarray(K_FG, dtype=float)))
    ax.set_ylim(-0.5 * kref, 3.0 * kref)
    ax = axs[1, 1]
    order = np.argsort(bins_s)
    Kfg_interp = np.interp(s_mm, np.asarray(bins_s)[order], np.asarray(K_FG)[order], left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ax.plot(s_mm, d["K_T_g"] / d["K_FM"], lw=1.0, label=r"$K_T$(Gauss) / $K_{FM}$")
        ax.plot(s_mm, d["K_FM"] / Kfg_interp, lw=1.0, label=r"$K_{FM}$ / $K_{FG}$")
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_ylim(0, 3)
    ax.set_ylabel("ratio")
    for a in axs.flat:
        a.axhline(0, color="0.85", lw=0.5)
        a.legend(fontsize=7, loc="upper left")
        if flag_s_mm is not None and np.isfinite(flag_s_mm):
            a.axvspan(flag_s_mm, np.nanmax(s_mm), color="tab:red", alpha=0.06, lw=0)
        if max_valid_s_mm is not None:
            a.axvspan(max_valid_s_mm, np.nanmax(s_mm), color="0.5", alpha=0.25, lw=0)
    for a in axs[1]:
        a.set_xlabel("downstream distance s [mm]")
    title = f"{case}: plume spreading and the Taylor-dispersion comparison"
    if flag_s_mm is not None and np.isfinite(flag_s_mm):
        title += (f"\n(shaded: s > {flag_s_mm:.0f} mm, in-plane scalar flux has decayed beyond tolerance: expected for a "
                  "3D plume spreading out of the sheet; Taylor/flux-moment relations still hold for a separable plume)")
    if max_valid_s_mm is not None:
        title += f"\n(dark grey: s > {max_valid_s_mm:g} mm, widths unreliable near the image edge; excluded from the comparison)"
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_spectra(points, spec: dict, cumK: dict, rms_prod: dict, case, path, dpi):
    """Per point: premultiplied cospectra f Co(f) / |flux| (with CI), coherence with significance level, and
    cumulative K(<f)/K where the point has a usable gradient."""
    n = len(points)
    fig, axs = plt.subplots(n, 3, figsize=(13, 2.4 * n), sharex=True, squeeze=False)
    cols = {"u": "tab:blue", "v": "tab:red", "w": "tab:green"}
    for i, p in enumerate(points):
        sig_level = np.nan
        for q in "uvw":
            sp = spec[(p.name, q)]
            f = sp["f"][1:]
            scale = rms_prod[(p.name, q)] or 1.0
            y = f * sp["co"][1:] / scale
            e = f * sp["co_ci"][1:] / scale
            axs[i, 0].semilogx(f, y, color=cols[q], lw=1, label=FLUX_LABELS[q])
            axs[i, 0].fill_between(f, y - e, y + e, color=cols[q], alpha=0.2, lw=0)
            axs[i, 1].semilogx(f, sp["coh"][1:], color=cols[q], lw=1, label=FLUX_LABELS[q])
            sig_level = sp["coh_sig"]
        axs[i, 1].axhline(sig_level, color="0.4", lw=0.7, ls="--")
        for q, ls in (("u", "-"), ("v", "--")):
            if (p.name, q) in cumK:
                fK, cK = cumK[(p.name, q)]
                axs[i, 2].semilogx(fK[1:], cK[1:], color=cols[q], ls=ls, lw=1, label=f"K(<f)/K, {q}'c'")
        axs[i, 2].axhline(1, color="0.6", lw=0.6)
        axs[i, 2].axhline(0, color="0.85", lw=0.5)
        axs[i, 0].axhline(0, color="0.7", lw=0.5)
        axs[i, 0].set_ylabel(f"{p.name}\n(x={p.x_mm:.0f}, s={p.downstream_mm:.0f} mm)", fontsize=8)
        if len(axs[i, 2].lines) > 2:
            axs[i, 2].legend(fontsize=7)
        else:
            axs[i, 2].text(0.5, 0.5, "no usable gradient here\n(|grad C| below mask or flux not significant)",
                           transform=axs[i, 2].transAxes, ha="center", va="center", fontsize=7, color="0.4")
    axs[0, 0].set_title(r"$f\,Co_{qc}(f) / (q_{rms} c_{rms})$; area vs ln f = $\rho_{qc}$ (95% CI)")
    axs[0, 1].set_title(r"coherence $\gamma^2_{qc}$ (dashed: 95% zero-coherence level)")
    axs[0, 2].set_title("cumulative K(<f) / K (points with a usable gradient)")
    axs[0, 0].legend(fontsize=7)
    for a in axs[-1]:
        a.set_xlabel("f [Hz]")
    fig.suptitle(f"{case}: single-point cospectra and coherence (Welch, Hann, 50% overlap, linear detrend)")
    fig.tight_layout()
    _save(fig, path, dpi)


def _log_quiver_arrays(fx, fy, vmin, vmax, lmin, lmax):
    mag = np.hypot(fx, fy)
    show = np.isfinite(mag) & (mag >= vmin)
    frac = np.full_like(mag, np.nan)
    frac[show] = np.clip((np.log10(mag[show]) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin)), 0, 1)
    length = lmin + (lmax - lmin) * frac
    safe = np.where(show, mag, 1.0)
    return (np.ma.masked_where(~show, fx / safe * length), np.ma.masked_where(~show, fy / safe * length),
            np.ma.masked_where(~show, np.clip(mag, vmin, vmax)))


def plot_flux_gradient_overlay(grid, C, fx, fy, gx, gy, angle, region, case, path, dpi, *, stride_px: int,
                               flux_vmin: float, flux_vmax: float, background: float):
    """Left: mean-C contours, flux vectors (log-scaled length and colour, as in tools/plot_scalar_fluxes.py) and the
    down-gradient direction -grad C (thin black, direction only). Right: angle between flux and -grad C."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    ax = axs[0]
    Cb = np.clip(C - background, 1e-4, None)
    cs = ax.contour(grid.x_mm, grid.y_mm, Cb, levels=np.logspace(-2, 0, 7), colors="0.55", linewidths=0.6)
    ax.clabel(cs, fmt="%.2g", fontsize=6)
    sl = (slice(stride_px // 2, None, stride_px), slice(stride_px // 2, None, stride_px))
    X, Y = np.meshgrid(grid.x_mm, grid.y_mm)
    step_mm = stride_px * abs(grid.dx_m) * 1000
    U_, V_, col = _log_quiver_arrays(np.where(region, fx, np.nan)[sl], np.where(region, fy, np.nan)[sl],
                                     flux_vmin, flux_vmax, 0.2 * step_mm, 0.95 * step_mm)
    q = ax.quiver(X[sl], Y[sl], U_, V_, col, cmap=MAG_CMAP, norm=LogNorm(flux_vmin, flux_vmax), angles="xy",
                  scale_units="xy", scale=1.0, pivot="mid", width=0.0035, headwidth=3.5, headlength=4,
                  headaxislength=3.5)
    gmag = np.hypot(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = -gx / gmag, -gy / gmag
    ok = region & np.isfinite(ux)
    ax.quiver(X[sl], Y[sl], np.where(ok, ux, np.nan)[sl] * 0.5 * step_mm, np.where(ok, uy, np.nan)[sl] * 0.5 * step_mm,
              angles="xy", scale_units="xy", scale=1.0, pivot="tail", color="k", width=0.0015, headwidth=3,
              headlength=3, headaxislength=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm] (source at top)")
    ax.set_title(r"flux (coloured, log length) and $-\nabla C$ (black, direction); grey: mean C contours")
    fig.colorbar(q, ax=ax, shrink=0.7, label=r"$|(\langle u'c'\rangle, \langle v'c'\rangle)|$ [m/s]")
    h = _map(axs[1], grid, np.where(region, angle, np.nan), cmap="coolwarm", norm=Normalize(0, 180),
             title=r"angle between flux and $-\nabla C$ [deg] (0 = down-gradient, 180 = counter-gradient)")
    fig.colorbar(h, ax=axs[1], shrink=0.7, label="deg")
    fig.suptitle(f"{case}: in-plane flux vs. down-gradient direction")
    _save(fig, path, dpi)



# ============================================================================ step 9: robustness
def plot_robustness(centers, groups: dict, case, path, dpi, crossover: dict):
    """Columns = groups of variants (smoothing, masks, record halves); rows = K_x, K_y, down-gradient fractions.
    ``groups`` = {group: {variant: {"Kx", "Ky", "fdx", "fdy"}}}; the first variant in each group is the reference."""
    ng = len(groups)
    fig, axs = plt.subplots(3, ng, figsize=(5.2 * ng, 10), sharex=True, squeeze=False)
    for j, (g, variants) in enumerate(groups.items()):
        colors = plt.cm.tab10(np.arange(len(variants)))
        for k, ((name, v), col) in enumerate(zip(variants.items(), colors)):
            lw, ls = (1.8, "-") if k == 0 else (1.0, "--")
            axs[0, j].plot(centers, v["Kx"], color=col, lw=lw, ls=ls, label=name)
            axs[1, j].plot(centers, v["Ky"], color=col, lw=lw, ls=ls, label=f"{name} (fd<0.5 from {crossover[name]:.0f} mm)")
            axs[2, j].plot(centers, v["fdx"], color=col, lw=lw, ls=ls)
            axs[2, j].plot(centers, v["fdy"], color=col, lw=lw, ls=":")
        axs[0, j].set_title(g)
        for i in range(3):
            axs[i, j].axhline(0, color="0.8", lw=0.5)
            axs[i, j].legend(fontsize=6.5)
        axs[2, j].axhline(0.5, color="0.6", lw=0.6, ls=":")
        axs[2, j].set_ylim(0, 1.02)
        axs[2, j].set_xlabel("downstream distance s [mm]")
    axs[0, 0].set_ylabel(r"$K_x$ binned [m$^2$/s]")
    axs[1, 0].set_ylabel(r"$K_y$ binned [m$^2$/s]")
    axs[2, 0].set_ylabel("down-gradient fraction\n(solid/dashed: x, dotted: y)")
    fig.suptitle(f"{case}: robustness of K_x, K_y and down-gradient fractions (thick line = reference analysis)")
    fig.tight_layout()
    _save(fig, path, dpi)
