"""Tests for the scalar-flux / effective-diffusivity analysis on synthetic data with known answers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sPIV_PLIF_postprocessing.analysis.scalar_flux import gradients, moments, uncertainty  # noqa: E402
from sPIV_PLIF_postprocessing.analysis.scalar_flux.geometry import load_grid  # noqa: E402
from sPIV_PLIF_postprocessing.analysis.scalar_flux.synthetic import ar1, gaussian_plume_mean, write_synthetic_case  # noqa: E402

DT = 0.05


def _paths(d: Path, name: str = "synth") -> dict:
    return {"u": d / f"piv_{name}_u.npy", "v": d / f"piv_{name}_v.npy", "w": d / f"piv_{name}_w.npy", "c": d / f"{name}_PLIF.npy"}


def _run(d: Path, points=((10, 30), (40, 36)), **kw):
    settings = moments.StreamSettings(**kw)
    res = moments.stream_moments(_paths(d), d / "blocks", settings=settings, points=points, progress=False)
    return res, moments.finalize_moments(res.sums, res.shift, min_valid_frames=10)


@pytest.fixture(scope="module")
def plume_case(tmp_path_factory):
    d = tmp_path_factory.mktemp("plume")
    truth = write_synthetic_case(d, kind="plume", nt=2400, seed=1)
    res, mom = _run(d)
    return d, truth, res, mom


@pytest.fixture(scope="module")
def noise_case(tmp_path_factory):
    d = tmp_path_factory.mktemp("noise")
    truth = write_synthetic_case(d, kind="noise", nt=2400, seed=2)
    res, mom = _run(d)
    return d, truth, res, mom


# ---------------------------------------------------------------- step 1: decomposition and fluxes
def test_streaming_matches_direct_numpy(tmp_path):
    """Chunked shifted sums reproduce direct NaN-aware statistics, including a partial last chunk and NaNs."""
    write_synthetic_case(tmp_path, kind="plume", nt=410, ny=20, nx=24, seed=3)
    res, mom = _run(tmp_path, points=((5, 10),), w_outlier_threshold=None)
    p = _paths(tmp_path)
    a = {q: np.load(p[q]).astype(np.float64) for q in p}
    for q in ("u", "v", "w"):
        m = np.isfinite(a[q]) & np.isfinite(a["c"])
        qn, cn = np.where(m, a[q], np.nan), np.where(m, a["c"], np.nan)
        flux = np.nanmean((qn - np.nanmean(qn, 0)) * (cn - np.nanmean(cn, 0)), axis=0)
        ok = np.isfinite(flux)
        np.testing.assert_allclose(mom[f"{q}c_flux"][ok], flux[ok], rtol=1e-6, atol=1e-12)
        pv = np.nanvar((qn - np.nanmean(qn, 0)) * (cn - np.nanmean(cn, 0)), axis=0)
        np.testing.assert_allclose(mom[f"{q}c_pvar"][ok], pv[ok], rtol=1e-5, atol=1e-14)
        np.testing.assert_allclose(mom[f"{q}_rms"][ok], np.nanstd(a[q], 0)[ok], rtol=1e-6)
        assert np.all(np.isnan(mom[f"{q}c_flux"][:, :2]))  # NaN edge columns stay masked
    np.testing.assert_allclose(mom["c_mean"], a["c"].mean(0), rtol=1e-6)
    assert res.block_frames[-1] == 10 and res.block_frames.sum() == 410


def test_block_sums_reproduce_full_flux(plume_case):
    d, truth, res, mom = plume_case
    full = uncertainty.subset_flux_from_blocks(res.block_paths, "u", slice(None))
    ok = np.isfinite(mom["uc_flux"])
    np.testing.assert_allclose(full[ok], mom["uc_flux"][ok], rtol=1e-4, atol=1e-10)


def test_w_outlier_rule():
    w = np.array([[0.1], [-1.5], [0.0]])
    out = moments.apply_w_outlier_rule(w, -1.0)
    assert np.isnan(out[1, 0]) and np.isfinite(out[[0, 2], 0]).all()


def test_plume_fluxes_follow_known_gradient_diffusion(plume_case):
    """<u'c'> = -K_x dC/dx and <v'c'> = -K_y dC/dy for the mixing-length synthetic plume."""
    _, truth, _, mom = plume_case
    for flux, grad, K in (("uc_flux", "dCdx", truth["K_x"]), ("vc_flux", "dCdy", truth["K_y"])):
        f, g = mom[flux], truth[grad]
        ok = np.isfinite(f) & (np.abs(g) > 0.2 * np.nanmax(np.abs(g)))
        slope = -np.sum(f[ok] * g[ok]) / np.sum(g[ok] ** 2)
        assert slope == pytest.approx(K, rel=0.1)
    ok = np.isfinite(mom["wc_flux"])
    assert abs(np.nanmean(mom["wc_flux"][ok])) < 3 * np.nanstd(mom["wc_flux"][ok]) / np.sqrt(ok.sum())


def test_running_covariance_ends_at_full_flux(plume_case):
    _, _, res, mom = plume_case
    run = uncertainty.running_covariance(res.point_series["u"].astype(float), res.point_series["c"].astype(float))
    np.testing.assert_allclose(run[-1], [mom["uc_flux"][10, 30], mom["uc_flux"][40, 36]], rtol=1e-4)


def test_noise_floor_from_background_strips(plume_case):
    _, truth, _, mom = plume_case
    nf = moments.noise_floor_from_background(mom["c_rms"], mom["c_mean"], truth["x_mm"], truth["y_mm"],
                                             [(-36, -30), (30, 36)], y_range_mm=(40, 63))
    assert nf["noise_std"] == pytest.approx(0.01, rel=0.15)


# ---------------------------------------------------------------- step 2: convergence and uncertainty
def test_integral_time_scale_ar1():
    phi = 0.8
    x = ar1((6000, 200), phi, np.random.default_rng(4))
    T, crossed = uncertainty.integral_time_scales(x, DT)
    expected = (1 + phi) / (2 * (1 - phi)) * DT
    assert np.median(T) == pytest.approx(expected, rel=0.15)
    assert crossed.mean() > 0.9


def test_autocorrelation_handles_nans():
    x = ar1((4000, 3), 0.7, np.random.default_rng(5))
    xn = x.copy()
    xn[::7, 0] = np.nan
    r_full = uncertainty.autocorrelation(x, max_lag=20)
    r_nan = uncertainty.autocorrelation(xn, max_lag=20)
    assert np.allclose(r_full[:, 0], r_nan[:, 0], atol=0.05)
    assert r_nan[0, 0] == pytest.approx(1.0)


def _bootstrap(res, mom, q, k, n_boot=400):
    M = uncertainty.moving_block_multiplicities(res.block_frames.size, k, n_boot, np.random.default_rng(6))
    return uncertainty.bootstrap_flux_se(res.block_paths, mom[f"_{q}c_Qs"], mom[f"_{q}c_Cs"], q, M)


def test_noise_fluxes_indistinguishable_from_zero(noise_case):
    """Pure noise: |flux| < 2 SE for ~95% of pixels, and SE matches the actual scatter of the flux estimates."""
    _, truth, res, mom = noise_case
    # phi=0.8 -> T_int = 4.5 frames; base block 20 frames >> 2 T_int, so k=1 base block is adequate
    for q in ("u", "v", "w"):
        se = _bootstrap(res, mom, q, k=1)["se"]
        f = mom[f"{q}c_flux"]
        ok = np.isfinite(f) & np.isfinite(se)
        frac_within = np.mean(np.abs(f[ok]) < 2 * se[ok])
        assert 0.90 < frac_within < 0.99, (q, frac_within)
        assert np.std(f[ok]) / np.median(se[ok]) == pytest.approx(1.0, abs=0.2)


def test_iid_bootstrap_underestimates_correlated_noise(tmp_path):
    """Sanity check of the block choice: with 1-frame blocks the SE of time-correlated data is too small."""
    write_synthetic_case(tmp_path, kind="noise", nt=2400, ny=16, nx=16, nan_edge_cols=0, phi=0.9, seed=7)
    res, mom = _run(tmp_path, points=(), base_block_frames=1, chunk_frames=20)
    se_iid = _bootstrap(res, mom, "u", k=1)["se"]
    se_blk = _bootstrap(res, mom, "u", k=40)["se"]
    assert np.median(se_blk / se_iid) > 2.0  # sqrt(2 T_int) ~ sqrt(19) ~ 4.4 for products of two AR(1)


# ---------------------------------------------------------------- step 3: gradients
def test_gradients_match_analytic_gaussian():
    x_mm = (np.arange(121) - 60) * 0.5
    s_mm = np.arange(161) * 0.5
    y_mm = s_mm.max() - s_mm
    C, dCdx, dCds, _ = gaussian_plume_mean(x_mm, s_mm, amp=1.0, sigma0_mm=6.0, K=2e-4, U=0.25)
    # rows are ordered by s, so the grid coordinate y = s_max - s decreases with row: signed spacing -0.5 mm
    sens = gradients.gradient_sensitivity(C, [0.0, 0.5, 1.0], 0.5, -0.5)
    scale = np.nanmax(np.abs(dCdx))
    for s, g in sens.items():
        err_x = np.nanmax(np.abs(g["dCdx"] - dCdx)) / scale
        assert err_x < 0.03, (s, err_x)  # everywhere outside the masked edge band
        err_y = np.nanmax(np.abs(g["dCdy"] - (-dCds))) / scale
        assert err_y < 0.03, (s, err_y)


def test_normalized_convolution_has_no_edge_artifact():
    C = np.ones((50, 60))
    valid = np.ones_like(C, dtype=bool)
    valid[:, :5] = False
    Cs = gradients.smooth_mean_field(C, 2.0, 0.5, 0.5, valid=valid)
    assert np.nanmax(np.abs(Cs - 1.0)) < 1e-12  # zero padding would pull edges toward 0
    gx, gy = gradients.mean_gradient(Cs, 5e-4, 5e-4, edge_band=gradients.edge_band_px(2.0, 0.5, 0.5))
    assert np.nanmax(np.abs(gx)) < 1e-9 and np.isnan(gx[:, :5 + 8]).all() and np.isfinite(gx[10:-10, 15:-10]).all()


def test_grid_loader(tmp_path):
    write_synthetic_case(tmp_path, kind="noise", nt=20, ny=10, nx=12, seed=8)
    g = load_grid(tmp_path / "synth_xgrid.npy", tmp_path / "synth_ygrid.npy")
    assert g.shape == (10, 12) and g.dx_m == pytest.approx(1e-3) and g.downstream_mm[-1] == 0


def test_flip_plif_rows_on_read(tmp_path):
    """A PLIF stack stored upside down, read with flip_plif_rows=True, gives the same fluxes as the correct one."""
    write_synthetic_case(tmp_path, kind="plume", nt=200, ny=20, nx=24, seed=9)
    ref_res, ref = _run(tmp_path, points=(), w_outlier_threshold=None)
    c = np.load(tmp_path / "synth_PLIF.npy")
    np.save(tmp_path / "synth_PLIF.npy", c[:, ::-1, :])
    _, flipped = _run(tmp_path, points=(), w_outlier_threshold=None, flip_plif_rows=True)
    ok = np.isfinite(ref["uc_flux"])
    np.testing.assert_allclose(flipped["uc_flux"][ok], ref["uc_flux"][ok], rtol=1e-6)
    np.testing.assert_allclose(flipped["c_mean"], ref["c_mean"], rtol=1e-6)


def test_orientation_check():
    from sPIV_PLIF_postprocessing.analysis.scalar_flux.geometry import check_orientation
    C = np.zeros((200, 30))
    C[-60:, 15] = 1.0  # source at high rows
    v_down = np.full((5, 200, 30), -0.25)  # flow toward low rows: consistent
    assert check_orientation(C, v_down, source_at_high_row=True)["source_at_high_row_detected"]
    with pytest.raises(ValueError, match="toward the scalar source"):
        check_orientation(C, -v_down, source_at_high_row=True)
    with pytest.raises(ValueError, match="flipped"):
        check_orientation(C[::-1], v_down, source_at_high_row=True)


# ---------------------------------------------------------------- steps 4-5: gradient diffusion and K
from sPIV_PLIF_postprocessing.analysis.scalar_flux import diffusivity  # noqa: E402


def _estimated_gradients(mom, truth, sigma_mm=1.5):
    sens = gradients.gradient_sensitivity(mom["c_mean"], [sigma_mm], 1.0, 1.0)  # synthetic grid: +1 mm spacing
    g = sens[sigma_mm]
    return g["dCdx"], g["dCdy"]


def test_fit_helpers_exact_line():
    g = np.linspace(-1, 1, 50)
    o = diffusivity.fit_through_origin(g, -3.0 * g)
    w = diffusivity.fit_with_intercept(g, 0.5 - 3.0 * g)
    assert o["K"] == pytest.approx(3.0) and o["r2"] == pytest.approx(1.0)
    assert w["K"] == pytest.approx(3.0) and w["intercept"] == pytest.approx(0.5) and w["r2"] == pytest.approx(1.0)
    assert diffusivity.downgradient_fractions(g, -g) == (1.0, 0.0)


def test_gradient_diffusion_recovers_known_K(plume_case):
    """Binned regressions and windowed K recover the synthetic K_x (estimated gradients from the noisy mean)."""
    _, truth, _, mom = plume_case
    gx, gy = _estimated_gradients(mom, truth)
    s = truth["y_mm"].max() - truth["y_mm"]
    rows = diffusivity.station_rows(s, [15, 30, 45], 8.0)
    strong = diffusivity.gradient_mask(gx, gy, gx, 0.2)
    test = diffusivity.gradient_diffusion_test(mom["uc_flux"], gx, strong, rows)
    np.testing.assert_allclose(test["K0"], truth["K_x"], rtol=0.15)
    assert np.all(test["frac_down"] > 0.95) and np.all(test["r2_0"] > 0.8)
    w = diffusivity.windowed_K(mom["uc_flux"], gx, np.isfinite(gx), 9)
    assert np.nanmedian(w["K"][strong]) == pytest.approx(truth["K_x"], rel=0.15)
    # streamwise: weaker gradients, looser tolerance
    sy = diffusivity.gradient_mask(gx, gy, gy, 0.2)
    ky = diffusivity.fit_through_origin(gy[sy & np.isfinite(mom["vc_flux"])], mom["vc_flux"][sy & np.isfinite(mom["vc_flux"])])
    assert ky["K"] == pytest.approx(truth["K_y"], rel=0.35)


def test_pointwise_K_masks_weak_gradients(plume_case):
    _, truth, _, mom = plume_case
    gx, gy = _estimated_gradients(mom, truth)
    mask = diffusivity.gradient_mask(gx, gy, gx, 0.2)
    K = diffusivity.pointwise_K(mom["uc_flux"], gx, mask)
    assert np.isnan(K[~mask]).all()
    assert np.nanmedian(K) == pytest.approx(truth["K_x"], rel=0.15)


def test_windowed_tensor_diagonal(plume_case):
    """Mixing-length synthetic has K = diag(K_x, K_y): off-diagonals small where the tensor is identifiable."""
    _, truth, _, mom = plume_case
    gx, gy = _estimated_gradients(mom, truth)
    t = diffusivity.windowed_tensor(mom["uc_flux"], mom["vc_flux"], gx, gy, np.isfinite(gx), 15, cond_max=30)
    ok = np.isfinite(t["Kxx"])
    assert ok.sum() > 50
    assert np.nanmedian(t["Kxx"][ok]) == pytest.approx(truth["K_x"], rel=0.3)
    assert abs(np.nanmedian(t["Kyx"][ok])) < 0.3 * truth["K_x"]


def test_noise_case_gives_K_indistinguishable_from_zero(noise_case):
    _, truth, _, mom = noise_case
    gx, gy = _estimated_gradients(mom, truth, sigma_mm=3.0)
    s = truth["y_mm"].max() - truth["y_mm"]
    rows = diffusivity.station_rows(s, [15, 30, 45], 8.0)
    strong = diffusivity.gradient_mask(gx, gy, gx, 0.2)
    test = diffusivity.gradient_diffusion_test(mom["uc_flux"], gx, strong, rows)
    assert np.all(np.abs(test["K0"]) < 3.5 * test["K0_se"])
    assert np.all(np.abs(test["frac_down"] - 0.5) < 0.15)


# ---------------------------------------------------------------- cross-case comparison / transition metrics
from sPIV_PLIF_postprocessing.analysis.scalar_flux import comparison  # noqa: E402


def test_first_crossing_interpolates():
    s = np.arange(0, 100, 10.0)
    p = np.linspace(1.0, 0.0, s.size)
    assert comparison.first_crossing(s, p, 0.5, direction="down") == pytest.approx(45.0)
    assert np.isnan(comparison.first_crossing(s, p, 2.0, direction="down"))


def test_transition_metrics_on_constructed_fields():
    """sigma grows linearly, L constant -> s_sigma_L at sigma = L; flank rho decays as a tanh step at 120 mm."""
    s = np.linspace(0, 300, 301)
    x = np.linspace(-150, 150, 301)
    sigma = 5 + 0.15 * s  # sigma = L = 20 mm at s = 100
    xc = np.zeros_like(s)
    eta = x[None, :] / sigma[:, None]
    amp = 0.4 + 0.2 * np.tanh(-(s - 120) / 20)  # 0.6 -> 0.2, midpoint 120 mm
    rho = amp[:, None] * np.tanh(eta)
    L = np.full((301, 301), 0.020)
    Knd = np.broadcast_to(np.clip(s / 80, 0, 1)[:, None] * 0.4, (301, 301)).copy()  # plateau 0.4 from 80 mm
    region = np.abs(eta) <= 2.5
    r = comparison.transition_metrics(s, x, sigma, xc, rho, L, Knd, region, smooth_mm=2.0)["metrics"]
    assert r["s_sigma_L"] == pytest.approx(100.0, abs=2.0)
    assert r["s_rho_50"] == pytest.approx(120.0, abs=6.0)
    assert r["s_K_90"] == pytest.approx(72.0, abs=3.0)


# ---------------------------------------------------------------- steps 6-8: spreading, spectra, angles
from sPIV_PLIF_postprocessing.analysis.scalar_flux import spectra, spreading  # noqa: E402
from sPIV_PLIF_postprocessing.analysis.scalar_flux.geometry import plume_centroid_and_width  # noqa: E402


def test_gaussian_fit_recovers_width_with_offset():
    x = np.linspace(-100, 100, 401)
    C = (0.3 * np.exp(-((x - 4) ** 2) / (2 * 12.0**2)) + 0.006)[None, :].repeat(3, axis=0)
    xc, sig = plume_centroid_and_width(C, x, background=0.0)  # biased by the un-removed offset
    g = spreading.gaussian_fit_widths(C, x, xc, sig)
    assert g["sigma_g"] == pytest.approx(np.full(3, 12.0), rel=1e-3)
    assert g["b0"] == pytest.approx(np.full(3, 0.006), rel=1e-2)
    assert np.all(sig > 14)  # second moment inflated by the offset: why the background matters


def test_taylor_and_flux_moment_K_match_synthetic(plume_case):
    """sigma^2 = sigma0^2 + 2 K s / U exactly in the synthetic mean field, and <u'c'> = -K dC/dx, so both the
    Taylor estimate (from widths) and the flux-moment estimate (from fluxes) should return K_x."""
    _, truth, _, mom = plume_case
    x, y = truth["x_mm"], truth["y_mm"]
    s = y.max() - y
    K_T = spreading.taylor_K(truth["sigma_mm"], np.full(s.size, truth["U"]), s, 15.0)
    assert np.nanmedian(K_T[10:-10]) == pytest.approx(truth["K_x"], rel=0.02)
    xc, sig = plume_centroid_and_width(truth["C"], x)
    band = np.abs(x[None, :] - xc[:, None]) <= 3 * sig[:, None]
    K_FM = spreading.flux_moment_K(mom["uc_flux"], mom["c_mean"], x, xc, 0.0, band)
    # pixels lost to NaN edge columns and the finite band bias this slightly; tolerance accordingly
    assert np.nanmedian(K_FM[5:-5]) == pytest.approx(truth["K_x"], rel=0.15)


def test_cospectrum_integrates_to_covariance():
    rng = np.random.default_rng(10)
    u = ar1((12000,), 0.7, rng)
    c = 0.6 * u + 0.8 * ar1((12000,), 0.5, rng)
    sp = spectra.cospectrum(u, c, 20.0, nperseg=256)
    integral = np.trapz(sp["co"], sp["f"])
    assert integral == pytest.approx(np.cov(u, c)[0, 1], rel=0.05)
    assert np.all(sp["co_ci"] > 0)


def test_coherence_of_independent_noise_below_significance():
    rng = np.random.default_rng(11)
    sp = spectra.cospectrum(rng.standard_normal(6000), rng.standard_normal(6000), 20.0, nperseg=256)
    assert np.mean(sp["coh"] > sp["coh_sig"]) < 0.1


def test_flux_gradient_angle():
    g = np.array([1.0, 0.0])
    assert diffusivity.flux_gradient_angle(-g[0], -g[1], g[0], g[1]) == pytest.approx(0.0)
    assert diffusivity.flux_gradient_angle(g[0], g[1], g[0], g[1]) == pytest.approx(180.0)
    assert diffusivity.flux_gradient_angle(0.0, 1.0, 1.0, 0.0) == pytest.approx(90.0)


# ---------------------------------------------------------------- step 9: robustness helpers
def test_anisotropic_smoothing_removes_striation_ripple():
    """A stationary 0.5% multiplicative ripple along y (12 mm wavelength) corrupts dC/dy; sigma_y = 5 mm removes it
    while dC/dx is essentially unchanged."""
    d = 0.3172
    x_mm = (np.arange(301) - 150) * d
    s_mm = np.arange(601) * d
    C, dCdx, dCds, _ = gaussian_plume_mean(x_mm, s_mm, amp=1.0, sigma0_mm=10.0, K=4e-4, U=0.28)
    ripple = 1 + 0.005 * np.sin(2 * np.pi * s_mm / 12.0)[:, None]
    Cr = C * ripple
    true_gy = -dCds  # rows ordered by s, y = s_max - s -> signed spacing -d
    iso = gradients.gradient_sensitivity(Cr, [2.0], d, -d)[2.0]
    ani = gradients.anisotropic_gradients(Cr, 2.0, 5.0, d, -d)
    core = (slice(100, -100), slice(120, -120))
    err_iso = np.nanmax(np.abs(iso["dCdy"] - true_gy)[core]) / np.nanmax(np.abs(true_gy[core]))
    err_ani = np.nanmax(np.abs(ani["dCdy"] - true_gy)[core]) / np.nanmax(np.abs(true_gy[core]))
    assert err_iso > 0.2 and err_ani < 0.1, (err_iso, err_ani)
    ex = np.nanmax(np.abs(ani["dCdx"] - dCdx)[core]) / np.nanmax(np.abs(dCdx[core]))
    assert ex < 0.05


def test_half_record_moments_match_direct(tmp_path):
    """Sub-record fluxes and means from block sums equal a direct computation on that half of the record."""
    write_synthetic_case(tmp_path, kind="plume", nt=400, ny=16, nx=20, seed=12)
    res, mom = _run(tmp_path, points=(), w_outlier_threshold=None)
    p = _paths(tmp_path)
    u, c = np.load(p["u"]).astype(float), np.load(p["c"]).astype(float)
    half = res.block_frames.size // 2
    h = uncertainty.subset_moments_from_blocks(res.block_paths, "u", slice(half, None))
    uu, cc = u[half * 20:], c[half * 20:]
    m = np.isfinite(uu)
    with np.errstate(invalid="ignore"):
        un, cn = np.where(m, uu, np.nan), np.where(m, cc, np.nan)
        flux = np.nanmean((un - np.nanmean(un, 0)) * (cn - np.nanmean(cn, 0)), axis=0)
        cmean = np.nanmean(cn, axis=0)
    ok = np.isfinite(flux)
    np.testing.assert_allclose(h["flux"][ok], flux[ok], rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose((h["c_mean_shifted"] + res.shift["c"])[ok], cmean[ok], rtol=1e-5)
