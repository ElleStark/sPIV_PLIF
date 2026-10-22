"""
Driver for the turbulent scalar-flux / gradient-diffusion / effective-diffusivity analysis.

Fixed-frame (Eulerian) decomposition at each pixel: plume meander is folded into the turbulent fluxes.
Single plane: only in-plane gradients; no dC/dz, no K_z. See
src/sPIV_PLIF_postprocessing/analysis/scalar_flux/METHODS.md.

Run from the repo root:
    python -m tools.run_scalar_flux_diffusivity --config configs/scalar_flux_diffusivity.toml --stage 1
Options:
    --cases baseline buoyant   override [cases].run
    --recompute                redo the streaming pass even if stream_raw.npz exists
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np

from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import gradients, moments, plotting, uncertainty
from src.sPIV_PLIF_postprocessing.analysis.scalar_flux.geometry import Point, check_orientation, load_grid, select_points

VEL = ("u", "v", "w")


def case_paths(cfg: dict, case: str) -> dict[str, Path]:
    base = Path(cfg["paths"]["base"])
    keys = ("plif", "u", "v", "w", "xgrid", "ygrid", "ils")
    return {k: base / cfg["paths"][k].format(case=case) for k in keys}


def output_dirs(cfg: dict, case: str) -> dict[str, Path]:
    root = Path(cfg["paths"]["output_root"]) / case
    dirs = {"root": root, "arrays": root / "arrays", "figures": root / "figures", "blocks": root / "block_sums"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_stream(res: moments.StreamResult, d: dict[str, Path], points: list[Point]) -> None:
    np.savez(d["arrays"] / "stream_raw.npz", **{f"sum_{k}": v for k, v in res.sums.items()},
             **{f"shift_{k}": v for k, v in res.shift.items()}, n_frames=res.n_frames,
             block_frames=res.block_frames, coarse_rows=res.coarse_rows, coarse_cols=res.coarse_cols,
             **{f"points_{q}": v for q, v in res.point_series.items()})
    for q, v in res.coarse_series.items():
        np.save(d["arrays"] / f"coarse_series_{q}.npy", v)
    (d["arrays"] / "points.json").write_text(json.dumps([p.__dict__ for p in points], indent=1))


def _load_stream(d: dict[str, Path]) -> tuple[moments.StreamResult, list[Point]]:
    z = np.load(d["arrays"] / "stream_raw.npz")
    res = moments.StreamResult(
        sums={k[4:]: z[k] for k in z.files if k.startswith("sum_")},
        shift={k[6:]: z[k] for k in z.files if k.startswith("shift_")},
        n_frames=int(z["n_frames"]),
        block_paths={f"{q}_{f}": d["blocks"] / f"block_{q}_{f}.npy" for q in VEL for f in moments.BLOCK_FIELDS},
        block_frames=z["block_frames"], coarse_rows=z["coarse_rows"], coarse_cols=z["coarse_cols"],
        coarse_series={q: np.load(d["arrays"] / f"coarse_series_{q}.npy", mmap_mode="r") for q in moments.ALL},
        point_series={k[7:]: z[k] for k in z.files if k.startswith("points_")},
    )
    points = [Point(**p) for p in json.loads((d["arrays"] / "points.json").read_text())]
    return res, points


def run_stage1(cfg: dict, case: str, *, recompute: bool = False) -> dict:
    """Steps 1-3: decomposition and fluxes, convergence and bootstrap uncertainty, mean gradients."""
    t0 = time.time()
    p, d = case_paths(cfg, case), output_dirs(cfg, case)
    sc, uc, gc, fc = cfg["stream"], cfg["uncertainty"], cfg["gradients"], cfg["figures"]
    case_cfg = cfg["cases"].get(case, {}) if isinstance(cfg["cases"].get(case), dict) else {}
    flip_plif = bool(case_cfg.get("flip_plif_rows", False))
    grid = load_grid(p["xgrid"], p["ygrid"], source_at_high_row=sc["source_at_high_row"])
    dt = sc["dt_s"]

    # ---- step 1: one streaming pass (cached)
    if recompute or not (d["arrays"] / "stream_raw.npz").exists():
        c_mm = np.load(p["plif"], mmap_mode="r")
        C_pre = np.asarray(c_mm[:: sc["prepass_frame_stride"]], dtype=np.float64).mean(axis=0)
        if flip_plif:
            C_pre = C_pre[::-1]
        v_pre = np.asarray(np.load(p["v"], mmap_mode="r")[:: 10 * sc["prepass_frame_stride"]], dtype=np.float64)
        orient = check_orientation(C_pre, v_pre, source_at_high_row=sc["source_at_high_row"])
        print(f"[{case}] orientation OK (flip_plif_rows={flip_plif}): {orient}")
        (d["arrays"] / "orientation_check.json").write_text(json.dumps({**orient, "flip_plif_rows": flip_plif}, indent=1))
        # plume widths need the PLIF background removed (an offset of ~0.005 over 300 mm inflates sigma ~3x)
        xm = np.zeros(grid.x_mm.size, dtype=bool)
        for lo, hi in cfg["noise"]["background_strips_mm"]:
            xm |= (grid.x_mm >= lo) & (grid.x_mm <= hi)
        bg_pre = float(np.median(C_pre[:, xm]))
        points = select_points(cfg["points"], grid, C_pre, background=bg_pre)
        print(f"[{case}] points: " + ", ".join(f"{q.name}(row {q.row}, col {q.col})" for q in points))
        settings = moments.StreamSettings(
            chunk_frames=sc["chunk_frames"], base_block_frames=sc["base_block_frames"],
            coarse_stride=sc["coarse_stride"], w_outlier_threshold=sc["w_outlier_threshold"], flip_plif_rows=flip_plif,
        )
        res = moments.stream_moments({"u": p["u"], "v": p["v"], "w": p["w"], "c": p["plif"]}, d["blocks"],
                                     settings=settings, points=[(q.row, q.col) for q in points])
        _save_stream(res, d, points)
        print(f"[{case}] streaming pass done in {(time.time() - t0) / 60:.1f} min")
    else:
        res, points = _load_stream(d)
        print(f"[{case}] loaded cached streaming pass")

    mom = moments.finalize_moments(res.sums, res.shift, min_valid_frames=sc["min_valid_frames"])
    noise = moments.noise_floor_from_background(mom["c_rms"], mom["c_mean"], grid.x_mm, grid.y_mm,
                                                cfg["noise"]["background_strips_mm"])
    rho_min_c = uc["rho_min_c_rms"] or 3 * noise["noise_std"]
    mom = moments.finalize_moments(res.sums, res.shift, min_valid_frames=sc["min_valid_frames"],
                                   rho_min_rms={"c": rho_min_c, **{q: uc["rho_min_vel_rms"] for q in VEL}})
    c_rms_corr = moments.noise_corrected_rms(mom["c_rms"], noise["noise_std"])

    # ---- step 2a: integral time scales on the coarse grid -> block length
    cs = {q: np.asarray(res.coarse_series[q], dtype=np.float64) for q in moments.ALL}
    tint = {}
    for k in ("c", "u", "v", "w"):
        tint[f"T_{k}"], tint[f"crossed_{k}"] = uncertainty.integral_time_scales(cs[k], dt)
    for q in VEL:
        tint[f"T_{q}c"], tint[f"crossed_{q}c"] = uncertainty.integral_time_scales(
            uncertainty.product_series(cs[q], cs["c"]), dt)
    del cs
    rows, cols = res.coarse_rows, res.coarse_cols
    region = mom["c_mean"][np.ix_(rows, cols)] > uc["plume_c_threshold"]
    T_basis = np.nanmax(np.stack([tint[f"T_{k}"] for k in uc["block_basis"]]), axis=0)
    n_base = res.block_frames.size
    k_blocks, T_used = uncertainty.choose_block_length(
        T_basis, region, dt=dt, base_block_frames=sc["base_block_frames"], n_base_blocks=n_base,
        factor=uc["block_factor"], percentile=uc["block_percentile"])
    block_s = k_blocks * sc["base_block_frames"] * dt
    print(f"[{case}] T_int (p{uc['block_percentile']:.0f}, plume) = {T_used:.2f} s -> block {block_s:.1f} s "
          f"({k_blocks} base blocks, {n_base // k_blocks} blocks per record)")

    # ---- step 2b: moving-block bootstrap
    M = uncertainty.moving_block_multiplicities(n_base, k_blocks, uc["n_boot"], np.random.default_rng(uc["seed"]))
    unc, signif = {}, {}
    for q in VEL:
        b = uncertainty.bootstrap_flux_se(res.block_paths, mom[f"_{q}c_Qs"], mom[f"_{q}c_Cs"], q, M)
        se = b["se"]
        unc[f"{q}c_se"] = se
        unc[f"{q}c_ci_halfwidth"] = b["ci_halfwidth"]
        with np.errstate(invalid="ignore", divide="ignore"):
            unc[f"{q}c_neff"] = mom[f"{q}c_pvar"] / se**2
            unc[f"{q}c_neff_T"] = res.n_frames * dt / (2 * tint[f"T_{q}c"])
        signif[q] = np.abs(mom[f"{q}c_flux"]) >= uc["significance_factor"] * se
        unc[f"{q}c_significant"] = signif[q]

    # ---- step 2c: running-mean convergence at the representative points
    running = {q: uncertainty.running_covariance(res.point_series[q].astype(float), res.point_series["c"].astype(float))
               for q in VEL}
    final = {q: np.array([mom[f"{q}c_flux"][pt.row, pt.col] for pt in points]) for q in VEL}
    se_pts = {q: np.array([unc[f"{q}c_se"][pt.row, pt.col] for pt in points]) for q in VEL}

    # ---- step 3: mean gradients at several smoothing scales
    dx_mm, dy_mm = grid.dx_m * 1000, grid.dy_m * 1000
    sens = gradients.gradient_sensitivity(mom["c_mean"], gc["smoothing_sigmas_mm"], dx_mm, dy_mm, method=gc["method"])

    # ---- save
    np.savez_compressed(d["arrays"] / "moments.npz", **{k: v for k, v in mom.items() if not k.startswith("_")},
                        c_rms_noise_corrected=c_rms_corr, x_mm=grid.x_mm, y_mm=grid.y_mm)
    np.savez_compressed(d["arrays"] / "uncertainty.npz", **unc, block_length_s=block_s, T_used_s=T_used,
                        **{f"running_{q}": running[q] for q in VEL})
    np.savez_compressed(d["arrays"] / "timescales.npz", **tint, coarse_rows=rows, coarse_cols=cols)
    np.savez_compressed(d["arrays"] / "gradients.npz",
                        **{f"{key}_sigma{s:g}mm": g[key] for s, g in sens.items() for key in g})

    plume = mom["c_mean"] > uc["plume_c_threshold"]
    summary = {
        "case": case, "flip_plif_rows": flip_plif, "n_frames": res.n_frames, "dt_s": dt, "grid_spacing_mm": [dx_mm, dy_mm],
        "noise_floor": noise, "rho_min_c_rms": rho_min_c,
        "w_outlier_rejected_fraction": float(1 - np.nansum(mom["w_n"]) / np.nansum(mom["u_n"])),
        "block_length_s": block_s, "T_int_used_s": T_used,
        "T_int_median_plume_s": {k: float(np.nanmedian(tint[f"T_{k}"][region])) for k in ("c", "u", "v", "uc", "vc", "wc")},
        "T_int_not_crossed_fraction_plume": {k: float(np.mean(~tint[f"crossed_{k}"][region])) for k in ("c", "uc", "vc")},
        "significant_fraction_plume": {q: float(np.mean(signif[q][plume])) for q in VEL},
        "neff_median_plume": {q: float(np.nanmedian(unc[f"{q}c_neff"][plume])) for q in VEL},
        "points": [{**pt.__dict__, **{f"{q}c_{k}": float(v[q][i]) for q in VEL for k, v in (("flux", final), ("se", se_pts))}}
                   for i, pt in enumerate(points)],
    }
    (d["root"] / "summary_stage1.json").write_text(json.dumps(summary, indent=1))

    # ---- figures
    dpi, f = fc["dpi"], d["figures"]
    plotting.plot_flux_maps(grid, mom, signif, case, f / "fluxes.png", dpi, points)
    plotting.plot_rho_maps(grid, mom, case, f / "correlation_coefficients.png", dpi)
    plotting.plot_rms_maps(grid, mom, c_rms_corr, case, f / "rms.png", dpi)
    plotting.plot_uncertainty_maps(grid, mom, unc, case, f / "flux_uncertainty.png", dpi)
    plotting.plot_time_scales(grid, tint, rows, cols, case, f / "integral_time_scales.png", dpi, block_s, region=region)
    plotting.plot_convergence(points, running, final, se_pts, dt, case, f / "convergence.png", dpi)
    plotting.plot_gradient_sensitivity(grid, sens, case, f / "gradient_sensitivity.png", dpi,
                                       gc["sensitivity_stations_mm"])
    print(f"[{case}] stage 1 done in {(time.time() - t0) / 60:.1f} min -> {d['root']}")
    return summary


def _load_ils(cfg: dict, case: str, member_key: str) -> np.ndarray | None:
    """Integral length scale map [m] from the ILS zip (ILS_mpi_from_npy.py outputs), or None if unavailable."""
    import io
    import zipfile

    zpath = Path(cfg["paths"]["base"]) / cfg["paths"]["ils_zip"]
    member = cfg["paths"][member_key].format(case=case)
    if not zpath.exists():
        return None
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.split("/")[-1] == member]
        return np.load(io.BytesIO(z.read(names[0]))).astype(np.float64) if names else None


def run_stage2(cfg: dict, case: str) -> dict:
    """Steps 4-5: gradient-diffusion test and effective diffusivity (in-plane only; no K_z)."""
    t0 = time.time()
    p, d = case_paths(cfg, case), output_dirs(cfg, case)
    sc, uc, gc, dc, fc = cfg["stream"], cfg["uncertainty"], cfg["gradients"], cfg["diffusion"], cfg["figures"]
    grid = load_grid(p["xgrid"], p["ygrid"], source_at_high_row=sc["source_at_high_row"])
    a = d["arrays"]
    mom = dict(np.load(a / "moments.npz"))
    unc = dict(np.load(a / "uncertainty.npz"))
    grads = np.load(a / "gradients.npz")
    s1 = json.loads((d["root"] / "summary_stage1.json").read_text())
    bg = s1["noise_floor"]["background_mean"]
    dx_mm = abs(grid.dx_m) * 1000

    # plume region and significance
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import diffusivity as D
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux.geometry import plume_centroid_and_width
    C = mom["c_mean"]
    xc_mom, sig_mom = plume_centroid_and_width(C, grid.x_mm, background=bg)
    if dc.get("width_method", "gaussian") == "gaussian":
        # Gaussian + offset fit: the second moment is inflated by faint wide skirts (smSource: 17 mm vs ~2 mm near
        # the source); fall back to the moments where the fit fails
        from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import spreading as SPR
        gfit = SPR.gaussian_fit_widths(C, grid.x_mm, xc_mom, sig_mom, background=bg)
        ok = np.isfinite(gfit["sigma_g"]) & np.isfinite(gfit["x0"])
        xc = np.where(ok, gfit["x0"], xc_mom)
        sig = np.where(ok, gfit["sigma_g"], sig_mom)
    else:
        xc, sig = xc_mom, sig_mom
    region = D.plume_region(C, grid.x_mm, background=bg, c_threshold=uc["plume_c_threshold"], xc=xc, sigma=sig,
                            n_sigma=dc["plume_n_sigma"])
    k_sig = uc["significance_factor"]
    signif = {q: np.abs(mom[f"{q}c_flux"]) >= k_sig * unc[f"{q}c_se"] for q in VEL}
    flux = {"x": mom["uc_flux"], "y": mom["vc_flux"]}
    sigmas = [float(s) for s in gc["smoothing_sigmas_mm"]]
    G = {s: (grads[f"dCdx_sigma{s:g}mm"], grads[f"dCdy_sigma{s:g}mm"]) for s in sigmas}
    gx, gy = G[float(gc["primary_sigma_mm"])]
    grad = {"x": gx, "y": gy}
    sig_comp = {"x": signif["u"], "y": signif["v"]}

    # ---- step 4: gradient-diffusion test by station and by streamwise bin
    s_mm = grid.downstream_mm
    st_centers = dc["station_centers_mm"]
    bin_centers = list(np.arange(dc["profile_bin_start_mm"], dc["profile_bin_stop_mm"] + 1e-9, dc["profile_bin_width_mm"]))
    st_rows = D.station_rows(s_mm, st_centers, dc["station_width_mm"])
    bin_rows = D.station_rows(s_mm, bin_centers, dc["profile_bin_width_mm"])

    def valid_for(comp, gxs, gys):
        g = gxs if comp == "x" else gys
        return region & sig_comp[comp] & D.gradient_mask(gxs, gys, g, dc["gradient_mask_frac"], dc["gradient_mask_mode"])

    tests_st, tests_bin, sens_K = {}, {}, {"x": {}, "y": {}}
    for comp in ("x", "y"):
        v = valid_for(comp, gx, gy)
        tests_st[comp] = D.gradient_diffusion_test(flux[comp], grad[comp], v, st_rows)
        tests_bin[comp] = D.gradient_diffusion_test(flux[comp], grad[comp], v, bin_rows)
        for s in sigmas:
            gxs, gys = G[s]
            sens_K[comp][s] = D.gradient_diffusion_test(flux[comp], gxs if comp == "x" else gys,
                                                        valid_for(comp, gxs, gys), bin_rows)["K0"]

    # ---- step 5: pointwise, windowed, tensor
    win = max(3, int(round(dc["window_mm"] / dx_mm)) | 1)
    twin = max(3, int(round(dc["tensor_window_mm"] / dx_mm)) | 1)
    K = {}
    for comp in ("x", "y"):
        gm = D.gradient_mask(gx, gy, grad[comp], dc["gradient_mask_frac"], dc["gradient_mask_mode"])
        K[f"K{comp}_pointwise"] = D.pointwise_K(flux[comp], grad[comp], gm & sig_comp[comp] & region)
        w = D.windowed_K(flux[comp], grad[comp], region, win)
        w["K"][~gm] = np.nan  # same weak-gradient mask as pointwise, for comparability
        K[f"K{comp}_windowed"], K[f"r{comp}_windowed"] = w["K"], w["r"]
    T = D.windowed_tensor(flux["x"], flux["y"], gx, gy, region, twin, cond_max=dc["tensor_cond_max"])

    # nondimensionalization
    ils = {"x": _load_ils(cfg, case, "ils_x_member"), "y": _load_ils(cfg, case, "ils_y_member")}
    rms = {"x": mom["u_rms"], "y": mom["v_rms"]}
    nondim, notes = {}, []
    for comp in ("x", "y"):
        if ils[comp] is None or ils[comp].shape != C.shape:
            notes.append(f"K_{comp}/(rms L) skipped: ILS map unavailable or wrong shape")
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            nondim[f"K{comp}_windowed_over_rmsL"] = K[f"K{comp}_windowed"] / (rms[comp] * ils[comp])
    D_mol = (cfg["cases"].get(case) or {}).get("molecular_diffusivity_m2s") if isinstance(cfg["cases"].get(case), dict) else None
    if D_mol:
        for comp in ("x", "y"):
            nondim[f"K{comp}_windowed_over_Dmol"] = K[f"K{comp}_windowed"] / D_mol
    else:
        notes.append("K/D_molecular skipped: no molecular_diffusivity_m2s in config for this case")

    # downstream profiles (cross-stream median over the plume region)
    prof = {comp: {f"windowed ({dc['window_mm']:g} mm)": D.cross_stream_profile(K[f"K{comp}_windowed"], region),
                   "pointwise": D.cross_stream_profile(K[f"K{comp}_pointwise"], region)} for comp in ("x", "y")}
    prof_nd = {comp: {k: D.cross_stream_profile(v, region) for k, v in nondim.items() if k.startswith(f"K{comp}")}
               for comp in ("x", "y")}

    # ---- flags
    def frac_neg(Km):
        m = region & np.isfinite(Km)
        return float(np.mean(Km[m] < 0)) if m.any() else np.nan
    flags = {f"{k}_negative_fraction": frac_neg(v) for k, v in K.items() if k.startswith("K")}
    flags.update({f"{k}_valid_fraction_of_plume": float(np.mean(np.isfinite(v[region]))) for k, v in K.items() if k.startswith("K")})
    counter_bins = {comp: [float(c) for c, f in zip(bin_centers, tests_bin[comp]["frac_down"]) if f < 0.5] for comp in ("x", "y")}

    # ---- save
    np.savez_compressed(a / "diffusivity.npz", **K, **{f"T_{k}": v for k, v in T.items()}, **nondim, region=region,
                        xc_mm=xc, sigma_x_mm=sig, bin_centers_mm=bin_centers, station_centers_mm=st_centers,
                        **{f"bin_{comp}_{k}": v for comp in ("x", "y") for k, v in tests_bin[comp].items()},
                        **{f"station_{comp}_{k}": v for comp in ("x", "y") for k, v in tests_st[comp].items()},
                        **{f"sensK_{comp}_sigma{s:g}mm": v for comp in ("x", "y") for s, v in sens_K[comp].items()})
    summary = {
        "case": case, "significance_factor": k_sig, "primary_sigma_mm": gc["primary_sigma_mm"], "window_px": win,
        "tensor_window_px": twin, "notes": notes, "flags": flags, "counter_gradient_bins_mm": counter_bins,
        "stations": {comp: {k: [float(x) for x in v] for k, v in tests_st[comp].items()} for comp in ("x", "y")},
        "station_centers_mm": st_centers,
        "plume_median": {k: float(np.nanmedian(v[region])) for k, v in {**K, **nondim}.items()},
        "tensor_median": {k: float(np.nanmedian(T[k][region])) for k in ("Kxx", "Kxy", "Kyx", "Kyy")},
        "tensor_valid_fraction": float(np.mean(np.isfinite(T["Kxx"][region]))),
    }
    (d["root"] / "summary_stage2.json").write_text(json.dumps(summary, indent=1))

    # ---- figures
    dpi, f = fc["dpi"], d["figures"]
    for comp in ("x", "y"):
        plotting.plot_gradient_diffusion_scatter(grid, flux[comp], grad[comp], valid_for(comp, gx, gy), st_rows, st_centers,
                                                 xc, sig, tests_st[comp], comp, case, f / f"gradient_diffusion_scatter_{comp}.png", dpi)
    plotting.plot_gradient_diffusion_summary(bin_centers, tests_bin, sens_K, case, f / "gradient_diffusion_summary.png", dpi)
    plotting.plot_K_maps(grid, {k: v for k, v in K.items() if k.startswith("K")}, region, case, f / "K_maps.png", dpi,
                         f" (smoothing {gc['primary_sigma_mm']:g} mm, window {dc['window_mm']:g} mm, |flux| >= {k_sig:g} SE for pointwise)")
    plotting.plot_K_maps(grid, {"Kxx": T["Kxx"], "Kxy": T["Kxy"], "Kyx": T["Kyx"], "Kyy": T["Kyy"]}, region, case,
                         f / "K_tensor_maps.png", dpi, f" tensor (window {dc['tensor_window_mm']:g} mm, cond <= {dc['tensor_cond_max']:g})")
    plotting.plot_K_profiles(s_mm, prof, bin_centers, tests_bin, case, f / "K_profiles.png", dpi)
    if nondim:
        plotting.plot_K_profiles(s_mm, prof_nd, bin_centers, None, case, f / "K_profiles_nondim.png", dpi,
                                 ylabel="K / (rms L)", title="cross-stream-median K / (q_rms L_longitudinal)")
    plot_rho_similarity(cfg, case, grid, mom, unc, xc, sig, st_rows, st_centers)
    print(f"[{case}] stage 2 done in {(time.time() - t0) / 60:.1f} min")
    return summary


def plot_rho_similarity(cfg, case, grid, mom, unc, xc, sig, st_rows, st_centers) -> None:
    """rho_qc profiles vs (x - x_c)/sigma_x at the stations; error = flux SE / (q_rms c_rms)."""
    rho, err = {}, {}
    for q in VEL:
        with np.errstate(invalid="ignore", divide="ignore"):
            e = unc[f"{q}c_se"] / (mom[f"{q}_rms"] * mom["c_rms"])
        e[~np.isfinite(mom[f"{q}c_rho"])] = np.nan
        eta, rho[q] = plotting.similarity_profiles(mom[f"{q}c_rho"], grid, st_rows, xc, sig)
        _, err[q] = plotting.similarity_profiles(e, grid, st_rows, xc, sig)
    sigma_at = [float(np.nanmedian(sig[r])) for r in st_rows]
    d = output_dirs(cfg, case)
    np.savez_compressed(d["arrays"] / "rho_profiles.npz", eta=eta, station_centers_mm=st_centers, sigma_x_mm=sigma_at,
                        **{f"rho_{q}c": rho[q] for q in VEL}, **{f"rho_{q}c_err": err[q] for q in VEL})
    plotting.plot_rho_profiles(eta, rho, err, st_centers, sigma_at, case, d["figures"] / "rho_profiles.png", cfg["figures"]["dpi"])


def run_compare(cfg: dict, cases: list[str]) -> dict:
    """Compare effective diffusivities and meander -> relative-dispersion transition estimates across cases
    (requires stages 1-2 for each case). Writes to <output_root>/comparison_<case1>_<case2>...>."""
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import comparison as CMP

    out = Path(cfg["paths"]["output_root"]) / ("comparison_" + "_".join(cases))
    out.mkdir(parents=True, exist_ok=True)
    s_by, prof_by, met_by, bins_by, rho_by = {}, {}, {}, {}, {}
    for case in cases:
        p, d = case_paths(cfg, case), output_dirs(cfg, case)
        grid = load_grid(p["xgrid"], p["ygrid"], source_at_high_row=cfg["stream"]["source_at_high_row"])
        mom = np.load(d["arrays"] / "moments.npz")
        D = np.load(d["arrays"] / "diffusivity.npz")
        L = _load_ils(cfg, case, "ils_x_member")
        if L is None:
            raise FileNotFoundError(f"ILS map for {case} not found; needed for sigma/L and K/(u_rms L)")
        r = CMP.transition_metrics(grid.downstream_mm, grid.x_mm, D["sigma_x_mm"], D["xc_mm"], mom["uc_rho"], L,
                                   D["Kx_windowed_over_rmsL"], D["region"])
        s_by[case], prof_by[case], met_by[case] = grid.downstream_mm, r["profiles"], r["metrics"]
        bins_by[case] = {"centers": D["bin_centers_mm"], "Kx": D["bin_x_K0"], "Ky": D["bin_y_K0"]}
        rho_by[case] = dict(np.load(d["arrays"] / "rho_profiles.npz"))
    dpi = cfg["figures"]["dpi"]
    plotting.plot_case_comparison(s_by, prof_by, met_by, bins_by, out / "case_comparison.png", dpi)
    plotting.plot_rho_profiles_comparison(rho_by, out / "rho_uc_profiles_comparison.png", dpi)
    summary = {}
    for case in cases:
        b, m = bins_by[case], met_by[case]
        far = np.asarray(b["centers"]) >= 150
        summary[case] = {**{k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in m.items()},
                         "Kx_binned_far_median_m2s": float(np.nanmedian(np.asarray(b["Kx"])[far])),
                         "Kx_binned_by_s": dict(zip([float(c) for c in b["centers"]], [float(k) for k in b["Kx"]]))}
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / "comparison_profiles.npz",
                        **{f"{c}_{k}": v for c in cases for k, v in prof_by[c].items()},
                        **{f"{c}_s_mm": s_by[c] for c in cases})
    print(f"comparison written to {out}")
    return summary


def run_stage3(cfg: dict, case: str) -> dict:
    """Steps 6-8: plume spreading and Taylor comparison, single-point cospectra/coherence, flux vs -grad C overlay."""
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import diffusivity as D
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import spectra as SP
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import spreading as SPR

    t0 = time.time()
    p, d = case_paths(cfg, case), output_dirs(cfg, case)
    sc, gc, dc, fc = cfg["stream"], cfg["gradients"], cfg["diffusion"], cfg["figures"]
    pc, spc, oc = cfg["spreading"], cfg["spectra"], cfg["overlay"]
    grid = load_grid(p["xgrid"], p["ygrid"], source_at_high_row=sc["source_at_high_row"])
    a = d["arrays"]
    mom = dict(np.load(a / "moments.npz"))
    unc = dict(np.load(a / "uncertainty.npz"))
    Dz = dict(np.load(a / "diffusivity.npz"))
    grads = np.load(a / "gradients.npz")
    gx, gy = grads[f"dCdx_sigma{gc['primary_sigma_mm']:g}mm"], grads[f"dCdy_sigma{gc['primary_sigma_mm']:g}mm"]
    s1 = json.loads((d["root"] / "summary_stage1.json").read_text())
    bg = s1["noise_floor"]["background_mean"]
    s_mm, x_mm = grid.downstream_mm, grid.x_mm
    C, V = mom["c_mean"], mom["v_mean"]
    xc, sig, region = Dz["xc_mm"], Dz["sigma_x_mm"], Dz["region"]

    # ---- step 6: spreading
    gfit = SPR.gaussian_fit_widths(C, x_mm, xc, sig, background=bg)
    U = SPR.advection_speed(V, C, bg)
    Q = SPR.scalar_mass_flux(V, C, x_mm, bg)
    intC = np.sum(C - bg, axis=1) * abs(grid.dx_m)
    r_ref = int(np.argmin(np.abs(s_mm - pc["reference_s_mm"])))
    band = np.abs(x_mm[None, :] - xc[:, None]) <= pc["flux_moment_band_sigma"] * sig[:, None]
    sp6 = {
        "sigma_mom_mm": sig, "sigma_g_mm": gfit["sigma_g"], "b0": gfit["b0"], "background": bg, "U": U,
        "Q_rel": Q / Q[r_ref], "intC_rel": intC / intC[r_ref],
        "K_T_mom": SPR.taylor_K(sig, U, s_mm, pc["taylor_window_mm"]),
        "K_T_g": SPR.taylor_K(gfit["sigma_g"], U, s_mm, pc["taylor_window_mm"]),
        "K_FM": SPR.flux_moment_K(mom["uc_flux"], C, x_mm, xc, bg, band),
    }
    beyond = (s_mm > pc["conservation_check_from_mm"]) & (np.abs(sp6["Q_rel"] - 1) > pc["mass_flux_tolerance"])
    flag_s = float(np.min(s_mm[beyond])) if beyond.any() else float("nan")

    # ---- step 7: spectra at the representative points
    z = np.load(a / "stream_raw.npz")
    points = [Point(**q) for q in json.loads((a / "points.json").read_text())]
    series = {q: z[f"points_{q}"].astype(float) for q in ("u", "v", "w", "c")}
    fs = 1.0 / sc["dt_s"]
    spec, cumK, flux_at, rms_prod, spec_summary = {}, {}, {}, {}, {}
    for i, pt in enumerate(points):
        for q in VEL:
            sp = SP.cospectrum(series[q][:, i], series["c"][:, i], fs, nperseg=spc["nperseg"], ci=spc["ci"])
            spec[(pt.name, q)] = sp
            flux_at[(pt.name, q)] = float(mom[f"{q}c_flux"][pt.row, pt.col])
            rms_prod[(pt.name, q)] = float(np.nanstd(series[q][:, i]) * np.nanstd(series["c"][:, i]))
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (sp["co"][1:] + sp["co"][:-1]) * np.diff(sp["f"]))])
            tot = cum[-1]
            f50 = float(np.interp(0.5, cum / tot, sp["f"])) if tot != 0 and np.all(np.diff(cum / tot) > -1) else float("nan")
            spec_summary[f"{pt.name}_{q}"] = {
                "flux": flux_at[(pt.name, q)], "cospectrum_integral": float(tot),
                "frac_below_1Hz": float(np.interp(1.0, sp["f"], cum) / tot) if tot != 0 else float("nan"),
                "f50_Hz": f50, "max_coherence": float(np.nanmax(sp["coh"][1:])), "coh_sig": float(sp["coh_sig"]),
                "gap_fraction": sp["gap_fraction"], "n_segments": int(sp["n_segments"]),
            }
        for q, g in (("u", gx), ("v", gy)):
            gval = g[pt.row, pt.col]
            gmask = D.gradient_mask(gx, gy, g, dc["gradient_mask_frac"], dc["gradient_mask_mode"])[pt.row, pt.col]
            signif = abs(flux_at[(pt.name, q)]) >= cfg["uncertainty"]["significance_factor"] * unc[f"{q}c_se"][pt.row, pt.col]
            if np.isfinite(gval) and gmask and signif:
                sp = spec[(pt.name, q)]
                ck = SP.cumulative_K(sp["f"], sp["co"], gval)
                cumK[(pt.name, q)] = (sp["f"], ck / ck[-1])
                spec_summary[f"{pt.name}_{q}"]["K_point"] = float(-flux_at[(pt.name, q)] / gval)

    # ---- step 8: flux vs -grad C
    angle = D.flux_gradient_angle(mom["uc_flux"], mom["vc_flux"], gx, gy)
    k_sig = cfg["uncertainty"]["significance_factor"]
    sig_any = (np.abs(mom["uc_flux"]) >= k_sig * unc["uc_se"]) | (np.abs(mom["vc_flux"]) >= k_sig * unc["vc_se"])
    gmag = np.hypot(gx, gy)
    with np.errstate(invalid="ignore"):
        strong = gmag >= dc["gradient_mask_frac"] * np.nanmax(np.where(np.isfinite(gmag), gmag, -np.inf), axis=1, keepdims=True)
    angle_mask = region & sig_any & strong
    angle_m = np.where(angle_mask, angle, np.nan)
    st_rows = D.station_rows(s_mm, dc["station_centers_mm"], dc["station_width_mm"])
    angle_stats = {}
    for c_mm, rows in zip(dc["station_centers_mm"], st_rows):
        v = angle_m[rows]
        v = v[np.isfinite(v)]
        angle_stats[str(c_mm)] = {"median_deg": float(np.median(v)) if v.size else None,
                                  "frac_within_45deg": float(np.mean(v < 45)) if v.size else None,
                                  "frac_counter_gt_90deg": float(np.mean(v > 90)) if v.size else None}

    # ---- save
    np.savez_compressed(a / "stage3.npz", **{k: v for k, v in sp6.items() if isinstance(v, np.ndarray)},
                        background=bg, flag_s_mm=flag_s, angle=angle_m, s_mm=s_mm,
                        **{f"spec_{n}_{q}_{k}": np.asarray(v) for (n, q), sp in spec.items() for k, v in sp.items()})
    st_c = np.asarray(dc["station_centers_mm"], dtype=float)
    def at_st(arr):
        return [float(np.nanmedian(arr[r])) if c <= pc["max_valid_s_mm"] else float("nan")
                for c, r in zip(dc["station_centers_mm"], st_rows)]
    summary = {
        "case": case, "in_plane_flux_decay_beyond_tolerance_from_s_mm": flag_s,
        "note_in_plane_flux": "int U (C-b) dx in a single plane decays as the plume spreads out of the sheet; this is "
                              "expected for a 3D plume and is not a conservation failure. Not used to infer anything about z.",
        "stations_mm": st_c.tolist(),
        "sigma_mom_mm": at_st(sig), "sigma_gauss_mm": at_st(gfit["sigma_g"]), "U_m_s": at_st(U),
        "Q_rel": at_st(sp6["Q_rel"]), "K_T_gauss": at_st(sp6["K_T_g"]), "K_T_mom": at_st(sp6["K_T_mom"]),
        "K_FM": at_st(sp6["K_FM"]),
        "K_FG": [float(x) for x in np.interp(st_c, Dz["bin_centers_mm"], Dz["bin_x_K0"])],
        "angle": angle_stats, "spectra": spec_summary,
    }
    (d["root"] / "summary_stage3.json").write_text(json.dumps(summary, indent=1))

    # ---- figures
    dpi, f = fc["dpi"], d["figures"]
    plotting.plot_spreading(s_mm, sp6, Dz["bin_centers_mm"], Dz["bin_x_K0"], case, f / "spreading_taylor.png", dpi, flag_s,
                            max_valid_s_mm=pc["max_valid_s_mm"])
    plotting.plot_spectra(points, spec, cumK, rms_prod, case, f / "cospectra_coherence.png", dpi)
    Fmag = np.hypot(mom["uc_flux"], mom["vc_flux"])[region]
    Fmag = Fmag[np.isfinite(Fmag) & (Fmag > 0)]
    stride = max(2, int(round(oc["quiver_stride_mm"] / (abs(grid.dx_m) * 1000))))
    plotting.plot_flux_gradient_overlay(grid, C, mom["uc_flux"], mom["vc_flux"], gx, gy, angle_m, region, case,
                                        f / "flux_gradient_overlay.png", dpi, stride_px=stride,
                                        flux_vmin=float(np.percentile(Fmag, 5)), flux_vmax=float(np.percentile(Fmag, 99)),
                                        background=bg)
    print(f"[{case}] stage 3 done in {(time.time() - t0) / 60:.1f} min")
    return summary


def run_robustness(cfg: dict, case: str) -> dict:
    """Step 9: sensitivity of K_x, K_y and down-gradient fractions to smoothing (isotropic and anisotropic-y),
    mask thresholds, and the first vs second half of the record; PLIF noise-floor effect on c_rms and rho."""
    from src.sPIV_PLIF_postprocessing.analysis.scalar_flux import diffusivity as D

    t0 = time.time()
    p, d = case_paths(cfg, case), output_dirs(cfg, case)
    sc, uc, gc, dc, rc = cfg["stream"], cfg["uncertainty"], cfg["gradients"], cfg["diffusion"], cfg["robustness"]
    grid = load_grid(p["xgrid"], p["ygrid"], source_at_high_row=sc["source_at_high_row"])
    a = d["arrays"]
    mom = dict(np.load(a / "moments.npz"))
    unc = dict(np.load(a / "uncertainty.npz"))
    Dz = dict(np.load(a / "diffusivity.npz"))
    grads = np.load(a / "gradients.npz")
    region = Dz["region"]
    s_mm = grid.downstream_mm
    dx_mm, dy_mm = grid.dx_m * 1000, grid.dy_m * 1000
    centers = [float(c) for c in Dz["bin_centers_mm"]]
    rows = D.station_rows(s_mm, centers, dc["profile_bin_width_mm"])
    k_ref, f_ref, sig_ref = uc["significance_factor"], dc["gradient_mask_frac"], float(gc["primary_sigma_mm"])

    def tests(fx, fy, sex, sey, gx, gy, k_sig=k_ref, frac=f_ref):
        out = {}
        for comp, f, se, g in (("x", fx, sex, gx), ("y", fy, sey, gy)):
            valid = region & (np.abs(f) >= k_sig * se) & D.gradient_mask(gx, gy, g, frac, dc["gradient_mask_mode"])
            t = D.gradient_diffusion_test(f, g, valid, rows)
            out[f"K{comp}"], out[f"fd{comp}"] = t["K0"], t["frac_down"]
        return out

    fx, fy, sex, sey = mom["uc_flux"], mom["vc_flux"], unc["uc_se"], unc["vc_se"]
    G = lambda s: (grads[f"dCdx_sigma{s:g}mm"], grads[f"dCdy_sigma{s:g}mm"])  # noqa: E731
    ref_name = f"iso {sig_ref:g} mm (reference)"
    groups = {"smoothing": {ref_name: tests(fx, fy, sex, sey, *G(sig_ref))}, "masks": {}, "record halves": {}}
    ripple = {}
    xm = np.zeros(grid.x_mm.size, dtype=bool)
    for lo, hi in cfg["noise"]["background_strips_mm"]:
        xm |= (grid.x_mm >= lo) & (grid.x_mm <= hi)

    def bg_rms(gy):
        v = gy[:, xm]
        return float(np.sqrt(np.nanmean(v**2)))

    ripple[ref_name] = bg_rms(G(sig_ref)[1])
    for s in gc["smoothing_sigmas_mm"]:
        if float(s) != sig_ref:
            name = f"iso {s:g} mm"
            groups["smoothing"][name] = tests(fx, fy, sex, sey, *G(float(s)))
            ripple[name] = bg_rms(G(float(s))[1])
    for sy in rc["anisotropic_sigma_y_mm"]:
        g = gradients.anisotropic_gradients(mom["c_mean"], sig_ref, sy, dx_mm, dy_mm)
        name = f"aniso x {sig_ref:g} / y {sy:g} mm"
        groups["smoothing"][name] = tests(fx, fy, sex, sey, g["dCdx"], g["dCdy"])
        ripple[name] = bg_rms(g["dCdy"])
    groups["masks"][ref_name] = groups["smoothing"][ref_name]
    for frac in rc["gradient_mask_fracs"]:
        if frac != f_ref:
            groups["masks"][f"grad mask {frac:g}"] = tests(fx, fy, sex, sey, *G(sig_ref), frac=frac)
    for k in rc["significance_factors"]:
        if k != k_ref:
            groups["masks"][f"signif {k:g} SE"] = tests(fx, fy, sex, sey, *G(sig_ref), k_sig=k)

    # first vs second half of the record (exact sub-record decompositions from the block sums)
    z = np.load(a / "stream_raw.npz")
    shift_c = z["shift_c"]
    block_paths = {f"{q}_{f}": d["blocks"] / f"block_{q}_{f}.npy" for q in VEL for f in moments.BLOCK_FIELDS}
    nb = int(z["block_frames"].size)
    groups["record halves"][ref_name] = groups["smoothing"][ref_name]
    half_flux = {}
    for name, sl in (("first half", slice(0, nb // 2)), ("second half", slice(nb // 2, None))):
        hu = uncertainty.subset_moments_from_blocks(block_paths, "u", sl)
        hv = uncertainty.subset_moments_from_blocks(block_paths, "v", sl)
        C_half = hu["c_mean_shifted"] + shift_c
        g = gradients.gradient_sensitivity(C_half, [sig_ref], dx_mm, dy_mm)[sig_ref]
        groups["record halves"][name] = tests(hu["flux"], hv["flux"], sex * np.sqrt(2), sey * np.sqrt(2), g["dCdx"], g["dCdy"])
        half_flux[name] = (hu["flux"], hv["flux"])
    # half-record difference as an empirical check of the bootstrap SE (should be ~ sqrt(2) * sqrt(2) SE = 2 SE)
    plume = region
    emp = {}
    for q, i, se in (("u", 0, sex), ("v", 1, sey)):
        dif = half_flux["first half"][i] - half_flux["second half"][i]
        m = plume & np.isfinite(dif) & np.isfinite(se)
        emp[q] = float(np.median(np.abs(dif[m]) / (2 * se[m])) / 0.6745)  # ~1 if SE is right (MAD-based)

    # crossover of the streamwise down-gradient fraction below 0.5 (onset of counter-gradient <v'c'>)
    def crossover(fd):
        fd = np.asarray(fd, dtype=float)
        idx = np.flatnonzero((fd[:-1] >= 0.5) & (fd[1:] < 0.5))
        if idx.size == 0:
            return float("nan")
        i = idx[0]
        return float(centers[i] + (centers[i + 1] - centers[i]) * (fd[i] - 0.5) / (fd[i] - fd[i + 1]))

    xover = {name: crossover(v["fdy"]) for grp in groups.values() for name, v in grp.items()}

    # PLIF noise-floor effect on c_rms and rho
    s1 = json.loads((d["root"] / "summary_stage1.json").read_text())
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = mom["c_rms"] / mom["c_rms_noise_corrected"]
    noise = {"noise_std": s1["noise_floor"]["noise_std"],
             "c_rms_over_corrected_median_plume": float(np.nanmedian(ratio[plume])),
             "c_rms_over_corrected_p95_plume": float(np.nanpercentile(ratio[plume], 95)),
             "note": "rho with noise-corrected c_rms = rho * (c_rms / c_rms_corrected); fluxes unaffected to first order"}

    # spread summary over the trusted range
    trusted = (np.asarray(centers) >= rc["summary_s_min_mm"]) & (np.asarray(centers) <= rc["summary_s_max_mm"])
    ref = groups["smoothing"][ref_name]
    spread = {}
    for grp, variants in groups.items():
        for name, v in variants.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(np.asarray(v["Kx"]) / np.asarray(ref["Kx"]) - 1)[trusted]
            spread[name] = {"Kx_max_rel_dev": float(np.nanmax(rel)), "Kx_median_rel_dev": float(np.nanmedian(rel)),
                            "fdx_min": float(np.nanmin(np.asarray(v["fdx"])[trusted])),
                            "fdy_crossover_mm": xover[name],
                            "Ky_far_median": float(np.nanmedian(np.asarray(v["Ky"])[np.asarray(centers) >= 180]))}
    summary = {"case": case, "trusted_s_range_mm": [rc["summary_s_min_mm"], rc["summary_s_max_mm"]],
               "variants": spread, "background_dCdy_rms_per_m": ripple,
               "bootstrap_SE_check_half_difference": emp, "noise_floor": noise}
    (d["root"] / "summary_robustness.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(a / "robustness.npz", centers=centers,
                        **{f"{g}|{n}|{k}": np.asarray(v[k]) for g, vs in groups.items() for n, v in vs.items() for k in v})
    plotting.plot_robustness(centers, groups, case, d["figures"] / "robustness.png", cfg["figures"]["dpi"], xover)
    print(f"[{case}] robustness (step 9) done in {(time.time() - t0) / 60:.1f} min")
    return summary


def replot_stage1(cfg: dict, case: str) -> None:
    """Regenerate stage-1 figures from the saved arrays (no streaming pass, no bootstrap)."""
    p, d = case_paths(cfg, case), output_dirs(cfg, case)
    sc, uc, gc, fc = cfg["stream"], cfg["uncertainty"], cfg["gradients"], cfg["figures"]
    grid = load_grid(p["xgrid"], p["ygrid"], source_at_high_row=sc["source_at_high_row"])
    a = d["arrays"]
    mom = dict(np.load(a / "moments.npz"))
    unc = dict(np.load(a / "uncertainty.npz"))
    tint = dict(np.load(a / "timescales.npz"))
    grads = np.load(a / "gradients.npz")
    sens = {s: {k: grads[f"{k}_sigma{s:g}mm"] for k in ("C_s", "dCdx", "dCdy")}
            for s in map(float, gc["smoothing_sigmas_mm"])}
    points = [Point(**q) for q in json.loads((a / "points.json").read_text())]
    rows, cols = tint["coarse_rows"], tint["coarse_cols"]
    region = mom["c_mean"][np.ix_(rows, cols)] > uc["plume_c_threshold"]
    signif = {q: unc[f"{q}c_significant"] for q in VEL}
    running = {q: unc[f"running_{q}"] for q in VEL}
    final = {q: np.array([mom[f"{q}c_flux"][pt.row, pt.col] for pt in points]) for q in VEL}
    se_pts = {q: np.array([unc[f"{q}c_se"][pt.row, pt.col] for pt in points]) for q in VEL}
    dpi, f, block_s = fc["dpi"], d["figures"], float(unc["block_length_s"])
    plotting.plot_flux_maps(grid, mom, signif, case, f / "fluxes.png", dpi, points)
    plotting.plot_rho_maps(grid, mom, case, f / "correlation_coefficients.png", dpi)
    plotting.plot_rms_maps(grid, mom, mom["c_rms_noise_corrected"], case, f / "rms.png", dpi)
    plotting.plot_uncertainty_maps(grid, mom, unc, case, f / "flux_uncertainty.png", dpi)
    plotting.plot_time_scales(grid, tint, rows, cols, case, f / "integral_time_scales.png", dpi, block_s, region=region)
    plotting.plot_convergence(points, running, final, se_pts, sc["dt_s"], case, f / "convergence.png", dpi)
    plotting.plot_gradient_sensitivity(grid, sens, case, f / "gradient_sensitivity.png", dpi,
                                       gc["sensitivity_stations_mm"])
    print(f"[{case}] stage-1 figures regenerated in {f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scalar_flux_diffusivity.toml")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                    help="1: steps 1-3, 2: steps 4-5, 3: steps 6-8, 4: step 9 robustness")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--plots-only", action="store_true", help="regenerate figures from saved arrays")
    ap.add_argument("--compare", nargs="+", metavar="CASE", help="compare cases (after stages 1-2) and exit")
    args = ap.parse_args(argv)
    cfg = tomllib.loads(Path(args.config).read_text())
    if args.compare:
        run_compare(cfg, args.compare)
        return
    for case in args.cases or cfg["cases"]["run"]:
        if args.stage == 1 and args.plots_only:
            replot_stage1(cfg, case)
        elif args.stage == 1:
            run_stage1(cfg, case, recompute=args.recompute)
        elif args.stage == 2:
            run_stage2(cfg, case)
        elif args.stage == 3:
            run_stage3(cfg, case)
        elif args.stage == 4:
            run_robustness(cfg, case)


if __name__ == "__main__":
    main()
