"""Tests for the dissipation-rate and turbulence-scale helpers in analysis.flow_properties."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sPIV_PLIF_postprocessing.analysis import flow_properties as fp  # noqa: E402


def _grid(nt=3, ny=40, nx=50, d=1e-3):
    y = np.arange(ny) * d
    x = np.arange(nx) * d
    Y, X = np.meshgrid(y, x, indexing="ij")
    return np.broadcast_to(X, (nt, ny, nx)), np.broadcast_to(Y, (nt, ny, nx)), d


def test_velocity_gradients_linear_fields_and_continuity():
    X, Y, d = _grid()
    u, v, w = 2.0 * X + 3.0 * Y, -5.0 * X + 7.0 * Y, 11.0 * X - 13.0 * Y
    for step in (1, 2, 4):
        g = fp.velocity_gradients(u, v, w, dx=d, dy=d, step=step)
        core = (slice(None), slice(step, -step), slice(step, -step))
        for k, val in (("du_dx", 2), ("du_dy", 3), ("dv_dx", -5), ("dv_dy", 7), ("dw_dx", 11), ("dw_dy", -13), ("dw_dz", -9)):
            np.testing.assert_allclose(g[k][core], val, rtol=1e-9)
        assert np.isnan(g["du_dx"][:, :, :step]).all()


def test_legacy_strain_rates_now_longitudinal():
    X, Y, d = _grid()
    u, v = 2.0 * X + 3.0 * Y, -5.0 * X + 7.0 * Y
    dudx, dvdy, dwdz = fp.compute_fluctuating_strain_rates(u, v, dx=d)
    np.testing.assert_allclose(dudx[:, 5:-5, 5:-5], 2.0)
    np.testing.assert_allclose(dvdy[:, 5:-5, 5:-5], 7.0)
    np.testing.assert_allclose(dwdz[:, 5:-5, 5:-5], -9.0)


def test_estimators_agree_for_isotropic_moments():
    L, nu = 2.0, 1.5e-5
    gm = {"du_dx": L, "dv_dy": L, "dw_dz": L, "du_dy": 2 * L, "dv_dx": 2 * L, "dw_dx": 2 * L, "dw_dy": 2 * L,
          "du_dx*dv_dy": -0.5 * L, "du_dy*dv_dx": -0.5 * L}
    e = fp.dissipation_estimates({k: np.array(v) for k, v in gm.items()}, nu=nu)
    assert e["eps_iso_long"] == pytest.approx(15 * nu * L)
    assert e["eps_hybrid"] == pytest.approx(15 * nu * L)
    r = fp.isotropy_ratios({k: np.array(v) for k, v in gm.items()})
    assert r["u_trans_over_long"] == pytest.approx(2) and r["dwdz_over_long"] == pytest.approx(1)


def test_gradient_moments_nan_aware():
    gm = fp.GradientMoments((2, 2))
    g = {k: np.ones((3, 2, 2)) for k in fp.GRADIENT_TERMS}
    g["du_dx"][0, 0, 0] = np.nan
    gm.add(g)
    m = gm.means()
    assert m["du_dx"][0, 0] == pytest.approx(1.0) and gm.counts["du_dx"][0, 0] == 2


def test_structure_function_sine():
    nx, d, A, lam = 400, 1e-3, 0.3, 0.05
    x = np.arange(nx) * d
    u = np.broadcast_to(A * np.sin(2 * np.pi * x / lam), (2, 30, nx)).copy()
    v = np.zeros_like(u)
    masks = {"all": np.ones((30, nx), dtype=bool)}
    sf = fp.StructureFunctionAccumulator(masks, [5, 10, 25])
    sf.add(u, v)
    res = sf.results(d)[("u", "all")]
    expected = A**2 * (1 - np.cos(2 * np.pi * res["r_m"] / lam))
    np.testing.assert_allclose(res["D_LL"], expected, rtol=0.03)


def test_turbulence_scales_formulas():
    nu, eps, k = 1.5e-5, 1e-3, 3e-4
    s = fp.turbulence_scales(np.array(eps), np.array(k), nu=nu)
    up = np.sqrt(2 * k / 3)
    assert s["kolmogorov_length"] == pytest.approx((nu**3 / eps) ** 0.25)
    assert s["taylor_microscale"] == pytest.approx(np.sqrt(15 * nu * up**2 / eps))
    assert s["taylor_Re"] == pytest.approx(up * s["taylor_microscale"] / nu)
    assert np.isnan(fp.turbulence_scales(np.array(-1.0), np.array(k), nu=nu)["kolmogorov_length"])
