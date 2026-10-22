"""
Video of the FTLE / instantaneous concentration overlay (same style as tools/overlay_ftle_concentration.py:
FTLE as filled grey contours underneath, concentration on a log scale semi-transparent on top).

Stage 1 computes a time series of backward-time FTLE fields from the PIV velocity interpolated to the PLIF grid
and caches it to FTLE_CACHE (skipped if the file exists). Stage 2 renders frames straight into ffmpeg.

Backward FTLE at time t0 (attracting structures, which scalar filaments align with):
    particles on a regular grid at t0 are advected with RK4 from t0 to t0 - T (step = -DT_S, for both position and
    time), using the velocity linearly interpolated in time between acquired frames and with INTERP_ORDER splines in
    space; FTLE = ln(sqrt(lambda_max(C)))/T with C the Cauchy-Green tensor of the flow map gradient (central
    differences on the particle grid). NaN velocity pixels are filled with the nearest valid value.

Velocity outside the field of view (EXTENSION = "linear", as in tools/FTLE_mpi_from_npy.py, after Tang, Chan &
Haller 2010): per frame, an incompressible linear field fitted to the measured velocity about the domain centre,
    u_l = A11 x' + A12 y' + <u>,  v_l = A21 x' - A11 y' + <v>   (x', y' relative to the centre)
    A11 = <x'u - y'v> / <x'^2 + y'^2>,  A12 = <y'u> / <y'^2>,  A21 = <x'v> / <x'^2>,
blended into the measured field over a band of width EXT_DELTA_MM inside each edge with the cubic weights
    d(r) = Delta^3 inside, 2r^3 + 3(Delta - 2R) r^2 + 6R(R - Delta) r + R^2 (3 Delta - 2R) in the band (0 at r = R),
    u = u_l + (u_data - u_l) d(|x'|) d(|y'|) / Delta^6.
EXTENSION = "nearest" instead continues with the velocity at the nearest edge point.

Run from the repo root:
    python -m tools.ftle_conc_overlay_video
"""
from __future__ import annotations

import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cmasher as cmr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
CASE_NAME = "baseline"
BASE = Path("E:/sPIV_PLIF_ProcessedData")
U_PATH = BASE / f"PIV/Interpolated_to_PLIF/piv_{CASE_NAME}_u.npy"   # (t, y, x), m/s
V_PATH = BASE / f"PIV/Interpolated_to_PLIF/piv_{CASE_NAME}_v.npy"
C_PATH = BASE / f"PLIF/{CASE_NAME}_PLIF.npy"                         # (t, y, x)
X_PATH = BASE / f"PLIF/{CASE_NAME}_xgrid.npy"                        # mm
Y_PATH = BASE / f"PLIF/{CASE_NAME}_ygrid.npy"
FLIP_PLIF_ROWS = {"baseline": True}  # baseline_PLIF.npy is stored upside down relative to its PIV stacks

DT_S = 0.05             # acquisition interval (20 Hz)
START_FRAME = 1000      # t = 50 s, as for the existing baseline FTLE snapshot
DURATION_S = 30.0
FPS = 20                # 20 = real time

# FTLE
INTEGRATION_T_S = 0.4   # as in analysis/FTLE_mpi_from_npy.py
PARTICLE_SPACING_MM = 0.5   # as in the HPC script
INTERP_ORDER = 3            # spatial spline order (HPC script: cubic)
EXTENSION = "linear"        # "linear" (Tang et al. 2010, as in the HPC script) or "nearest"
EXT_DELTA_MM = 15.0         # blending band width (HPC script: 0.0005 m * 30)
MASK_EXITED = False         # True: NaN (hatched) where the backward trajectory left the measured field
MASK_HATCH_COLOR = "0.6"
FTLE_CACHE = BASE / (f"FTLE_data/FTLE_{CASE_NAME}_f{START_FRAME}_{DURATION_S:g}s_T{INTEGRATION_T_S:g}s"
                     f"_backward_{PARTICLE_SPACING_MM:g}mm_{EXTENSION}ext{'_masked' if MASK_EXITED else ''}.npy")

# Figure (as tools/overlay_ftle_concentration.py)
CONC_CMAP = cmr.ocean_r
FTLE_CMAP = cmr.get_sub_cmap("cmr.neutral_r", 0, 0.85)
FTLE_LEVELS = 40
FTLE_LIMITS: tuple[float, float] | None = (1.0, 8.0)  # common scale for all cases; None -> FTLE_PERCENTILES
FTLE_PERCENTILES = (50.0, 99.5)  # lower limit at the median keeps the background white so ridges stand out
CONC_THRESHOLD = 0.015          # c/c0; lower end of the concentration colour scale
CONC_BELOW_COLOR: str | None = "white"  # c < threshold drawn in this colour (with CONC_ALPHA); None = transparent
CONC_VMIN, CONC_VMAX = CONC_THRESHOLD, 1.0
# Overlay field: "conc" (c/c0) or "grad" (|grad c| in (c/c0)/mm, log scale, from the frame smoothed with a NaN-aware
# Gaussian of GRAD_SMOOTH_SIGMA_MM; as tools/overlay_ftle_concentration_gradient.py, which used pre-smoothed PLIF)
OVERLAY_FIELD = "conc"
GRAD_SMOOTH_SIGMA_MM = 1.0
GRAD_LIMITS: tuple[float, float] | None = None  # None -> (vmax / GRAD_DYNAMIC_RANGE, 99.5th pct in view) per frame
GRAD_DYNAMIC_RANGE = 100.0
GRAD_CMAP = cmr.ocean_r
GRAD_ALPHA = 0.7
CONC_ALPHA = 0.6
XLIM: tuple[float, float] | None = (-100.0, 100.0)
YLIM: tuple[float, float] | None = (0.0, 250.0)  # source at y = 300 mm: shows s >= 50 mm downstream
# Axis labelling: with DOWNSTREAM_ORIGIN_MM set, the vertical axis is labelled "x" = downstream distance
# ORIGIN - y (0 at the source, increasing downward) and the horizontal axis "y"; None keeps the raw grid coordinates.
DOWNSTREAM_ORIGIN_MM: float | None = 300.0
FIGSIZE = (6.4, 6.0)
DPI = 200
CRF = 18
OUT_PATH = BASE / f"Plots/FTLE/videos/ftle_conc_overlay_{CASE_NAME}_f{START_FRAME}_{DURATION_S:g}s_{FPS}fps.mp4"
MAX_FRAMES_TEST: int | None = None   # e.g. 5 for a quick test (FTLE cache is then not written)
N_WORKERS = 10                       # parallel FTLE processes (each caches ~25 velocity frames, ~0.3 GB)
TEST_PNG: Path | None = None         # write the first rendered frame here as well
# Frames chosen for the per-case snapshots (20 Hz: t = frame * 0.05 s). Run e.g.
#     for c, f in SNAPSHOT_FRAMES.items(): START_FRAME = f; set_case(c); snapshot(f)
SNAPSHOT_FRAMES = {"baseline": 1000, "buoyant": 175, "diffusive": 700, "fractal": 175, "nearbed": 500, "smSource": 39}
SNAPSHOT_DPI = 300
SNAPSHOT_PATH = BASE / f"Plots/FTLE/videos/ftle_conc_overlay_{CASE_NAME}_f{START_FRAME}_snapshot.png"


def set_case(case: str) -> None:
    """Point the module-level paths at another case (e.g. for a loop over cases)."""
    global CASE_NAME, U_PATH, V_PATH, C_PATH, X_PATH, Y_PATH, FTLE_CACHE, OUT_PATH, SNAPSHOT_PATH
    CASE_NAME = case
    U_PATH = BASE / f"PIV/Interpolated_to_PLIF/piv_{case}_u.npy"
    V_PATH = BASE / f"PIV/Interpolated_to_PLIF/piv_{case}_v.npy"
    C_PATH = BASE / f"PLIF/{case}_PLIF.npy"
    X_PATH = BASE / f"PLIF/{case}_xgrid.npy"
    Y_PATH = BASE / f"PLIF/{case}_ygrid.npy"
    FTLE_CACHE = BASE / (f"FTLE_data/FTLE_{case}_f{START_FRAME}_{DURATION_S:g}s_T{INTEGRATION_T_S:g}s"
                         f"_backward_{PARTICLE_SPACING_MM:g}mm_{EXTENSION}ext{'_masked' if MASK_EXITED else ''}.npy")
    OUT_PATH = BASE / f"Plots/FTLE/videos/ftle_conc_overlay_{case}_f{START_FRAME}_{DURATION_S:g}s_{FPS}fps.mp4"
    SNAPSHOT_PATH = BASE / f"Plots/FTLE/videos/ftle_conc_overlay_{case}_f{START_FRAME}_snapshot.png"


def _fill_nan(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _blend_weight(r: np.ndarray, R: float, delta: float) -> np.ndarray:
    """Cubic transition weight d(r)/Delta^3: 1 for r <= R - Delta, 0 for r >= R (HPC script / Tang et al. 2010)."""
    d = 2 * r**3 + 3 * (delta - 2 * R) * r**2 + 6 * R * (R - delta) * r + R**2 * (3 * delta - 2 * R)
    w = np.where(r <= R - delta, 1.0, d / delta**3)
    return np.clip(np.where(r >= R, 0.0, w), 0.0, 1.0)


class VelocityField:
    """Velocity in mm/s at arbitrary (t, x, y) in mm: linear in time between frames, splines in space, and the
    EXTENSION model outside the field of view."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.u = np.load(U_PATH, mmap_mode="r")
        self.v = np.load(V_PATH, mmap_mode="r")
        self.x0, self.dx = x[0], x[1] - x[0]
        self.y0, self.dy = y[0], y[1] - y[0]
        self.xc, self.yc = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        self.Rx, self.Ry = 0.5 * (x.max() - x.min()), 0.5 * (y.max() - y.min())
        self.Xr, self.Yr = np.meshgrid(x - self.xc, y - self.yc)
        self.bounds = (x.min(), x.max(), y.min(), y.max())
        self.cache: dict[int, dict] = {}

    def _linear_fit(self, u: np.ndarray, v: np.ndarray) -> tuple[float, ...]:
        ok = np.isfinite(u) & np.isfinite(v)
        X, Y, uu, vv = self.Xr[ok], self.Yr[ok], u[ok], v[ok]
        a11 = np.mean(X * uu - Y * vv) / np.mean(X**2 + Y**2)
        a12 = np.mean(Y * uu) / np.mean(Y**2)
        a21 = np.mean(X * vv) / np.mean(X**2)
        return a11, a12, a21, float(np.mean(uu)), float(np.mean(vv))

    def frame(self, k: int) -> dict:
        if k not in self.cache:
            u = 1000.0 * np.array(self.u[k], dtype=np.float64)
            v = 1000.0 * np.array(self.v[k], dtype=np.float64)
            fill = (lambda a: ndimage.spline_filter(_fill_nan(a), order=INTERP_ORDER, mode="nearest")) \
                if INTERP_ORDER > 1 else _fill_nan
            self.cache[k] = {"u": fill(u), "v": fill(v), "lin": self._linear_fit(u, v)}
            for old in [j for j in self.cache if j > k + 12 or j < k - 12]:  # one FTLE needs frames f-8..f
                del self.cache[old]
        return self.cache[k]

    def _frame_velocity(self, fr: dict, px: np.ndarray, py: np.ndarray):
        coords = np.array([(py - self.y0) / self.dy, (px - self.x0) / self.dx])
        kw = {"order": INTERP_ORDER, "mode": "nearest", "prefilter": False}
        ui = ndimage.map_coordinates(fr["u"], coords, **kw)
        vi = ndimage.map_coordinates(fr["v"], coords, **kw)
        if EXTENSION != "linear":
            return ui, vi
        a11, a12, a21, um, vm = fr["lin"]
        xr, yr = px - self.xc, py - self.yc
        ul, vl = a11 * xr + a12 * yr + um, a21 * xr - a11 * yr + vm
        w = _blend_weight(np.abs(xr), self.Rx, EXT_DELTA_MM) * _blend_weight(np.abs(yr), self.Ry, EXT_DELTA_MM)
        return ul + (ui - ul) * w, vl + (vi - vl) * w

    def __call__(self, t: float, px: np.ndarray, py: np.ndarray):
        f = t / DT_S
        k0 = int(np.floor(f + 1e-6))  # tolerance: t = 49.95 s must give frame 999, not 998
        w = f - k0
        uo, vo = 0.0, 0.0
        for k, wk in ((k0, 1.0 - w), (k0 + 1, w)):
            if wk < 1e-6:
                continue
            uk, vk = self._frame_velocity(self.frame(k), px, py)
            uo, vo = uo + wk * uk, vo + wk * vk
        return uo, vo


def backward_ftle(vel: VelocityField, frame: int, gx: np.ndarray, gy: np.ndarray, spacing: float) -> np.ndarray:
    """Backward FTLE on the particle grid; with MASK_EXITED, NaN where the particle or one of its central-difference
    neighbours left the measured field at any step of the integration (its FTLE then depends on the extension)."""
    px, py = gx.ravel().copy(), gy.ravel().copy()
    xlo, xhi, ylo, yhi = vel.bounds
    exited = np.zeros(px.shape, dtype=bool)
    h = -DT_S
    t = frame * DT_S
    for _ in range(int(round(INTEGRATION_T_S / DT_S))):
        k1 = vel(t, px, py)
        k2 = vel(t + h / 2, px + h / 2 * k1[0], py + h / 2 * k1[1])
        k3 = vel(t + h / 2, px + h / 2 * k2[0], py + h / 2 * k2[1])
        k4 = vel(t + h, px + h * k3[0], py + h * k3[1])
        px = px + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        py = py + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        exited |= (px < xlo) | (px > xhi) | (py < ylo) | (py > yhi)
        t += h
    fx, fy = px.reshape(gx.shape), py.reshape(gy.shape)
    dfx_dy, dfx_dx = np.gradient(fx, spacing)
    dfy_dy, dfy_dx = np.gradient(fy, spacing)
    c11 = dfx_dx**2 + dfy_dx**2
    c12 = dfx_dx * dfx_dy + dfy_dx * dfy_dy
    c22 = dfx_dy**2 + dfy_dy**2
    lam = 0.5 * (c11 + c22) + np.sqrt(np.maximum(0.25 * (c11 - c22) ** 2 + c12**2, 0.0))
    ftle = (np.log(np.sqrt(lam)) / INTEGRATION_T_S).astype(np.float32)
    if MASK_EXITED:
        cross = ndimage.generate_binary_structure(2, 1)  # the point and its 4 central-difference neighbours
        ftle[ndimage.binary_dilation(exited.reshape(gx.shape), structure=cross)] = np.nan
    return ftle


def _conc_cmap(base=None):
    cmap = (CONC_CMAP if base is None else base).copy()
    if CONC_BELOW_COLOR is not None:
        cmap.set_under(CONC_BELOW_COLOR)
    return cmap


def conc_gradient(c: np.ndarray, x: np.ndarray, y: np.ndarray, sigma_mm: float = GRAD_SMOOTH_SIGMA_MM) -> np.ndarray:
    """|grad c| [(c/c0)/mm] after a NaN-aware Gaussian smoothing of sigma_mm (0 = none)."""
    c = np.asarray(c, dtype=float)
    if sigma_mm > 0:
        sig_px = (sigma_mm / abs(y[1] - y[0]), sigma_mm / abs(x[1] - x[0]))
        valid = np.isfinite(c)
        num = ndimage.gaussian_filter(np.where(valid, c, 0.0), sig_px, mode="nearest")
        den = ndimage.gaussian_filter(valid.astype(float), sig_px, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(valid, num / den, np.nan)
    dcdy, dcdx = np.gradient(c, y, x, edge_order=2)
    return np.hypot(dcdx, dcdy)


def _overlay_norm_label(grad_lims):
    if OVERLAY_FIELD == "grad":
        return LogNorm(*grad_lims), _conc_cmap(GRAD_CMAP), r"$|\nabla c|$ ($c_0$/mm)"
    return LogNorm(CONC_VMIN, CONC_VMAX), _conc_cmap(), r"Concentration $c/c_0$"


def _label_axes(ax) -> None:
    """Axis labels; with DOWNSTREAM_ORIGIN_MM, tick labels show downstream distance ORIGIN - y (data unchanged)."""
    if DOWNSTREAM_ORIGIN_MM is None:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        return
    origin = DOWNSTREAM_ORIGIN_MM
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(origin - np.arange(0.0, origin + 1e-9, 50.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{origin - v:g}"))
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("x (mm)")


def _setup_figure(x, y, lo, hi, dpi, grad_lims=None):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=dpi)
    norm, cmap, label = _overlay_norm_label(grad_lims)
    sm_c = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_f = plt.cm.ScalarMappable(cmap=FTLE_CMAP)
    sm_f.set_clim(lo, hi)
    fig.colorbar(sm_c, ax=ax, label=label, shrink=0.8, pad=0.02,
                 extend="min" if CONC_BELOW_COLOR is not None else "neither")
    fig.colorbar(sm_f, ax=ax, label="backward FTLE (1/s)", shrink=0.8, pad=0.02)
    _label_axes(ax)
    ax.set_xlim(XLIM if XLIM else (x.min(), x.max()))
    ax.set_ylim(YLIM if YLIM else (y.min(), y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("placeholder title for layout")  # reserve room so the per-frame title is not clipped
    fig.tight_layout()
    return fig, ax


def _draw_masked_hatch(ax, gx, gy, ftle_k):
    """Hatch the FTLE pixels masked because their trajectories left the field (empty list if none)."""
    m = ~np.isfinite(ftle_k)
    if not m.any():
        return []
    with plt.rc_context({"hatch.color": MASK_HATCH_COLOR, "hatch.linewidth": 0.5}):
        h = ax.contourf(gx, gy, m.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    return [h]


def _draw_frame(ax, gx, gy, ftle_k, x, y, c, lo, hi, t_s, grad_lims=None):
    """c is the concentration frame; with OVERLAY_FIELD = "grad" its gradient magnitude is drawn instead."""
    levels = np.linspace(lo, hi, FTLE_LEVELS)
    norm, cmap, _ = _overlay_norm_label(grad_lims)
    if OVERLAY_FIELD == "grad":
        field, vmin, vmax, alpha = conc_gradient(c, x, y), grad_lims[0], grad_lims[1], GRAD_ALPHA
    else:
        field, vmin, vmax, alpha = c, CONC_VMIN, CONC_VMAX, CONC_ALPHA
    # below vmin: CONC_BELOW_COLOR (or transparent); above vmax: clipped (avoids speckled edges).
    # The floor keeps values <= 0 out of LogNorm, which would otherwise mask those pixels.
    if CONC_BELOW_COLOR is not None:
        field = np.ma.masked_invalid(np.clip(field, vmin * 1e-6, vmax))
    else:
        field = np.ma.masked_where(~np.isfinite(field) | (field < vmin), np.clip(field, None, vmax))
    # extend="both": values at/above the top level are filled (clipping alone left them white)
    cf = ax.contourf(gx, gy, np.ma.masked_invalid(ftle_k), levels=levels, cmap=FTLE_CMAP, extend="both")
    hatch = _draw_masked_hatch(ax, gx, gy, ftle_k)
    pm = ax.pcolormesh(x, y, field, cmap=cmap, norm=norm, shading="auto", alpha=alpha, rasterized=True)
    ax.set_title(f"{CASE_NAME}, t = {t_s:5.2f} s  (FTLE T = {INTEGRATION_T_S:g} s, backward)")
    return [cf, pm, *hatch]


def _load_conc(c_mm, f):
    c = np.array(c_mm[f], dtype=np.float32)
    return c[::-1] if FLIP_PLIF_ROWS.get(CASE_NAME, False) else c


def _particle_grid(x: np.ndarray, y: np.ndarray):
    xs = np.arange(x.min(), x.max() + 1e-9, PARTICLE_SPACING_MM)
    ys = np.arange(y.min(), y.max() + 1e-9, PARTICLE_SPACING_MM)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, PARTICLE_SPACING_MM


def snapshot(frame: int = START_FRAME, *, reuse: bool = True, out_dir: Path | str | None = None,
             grad_lims: tuple[float, float] | None = None, ftle_only: bool = True) -> None:
    """Render one frame of the planned video format (overlay of OVERLAY_FIELD), plus the FTLE field alone (figure
    and .npy) next to SNAPSHOT_PATH when ftle_only. The FTLE is computed for that frame only, or loaded from the
    saved .npy when reuse and it exists."""
    x = np.load(X_PATH)[0, :].astype(float)
    y = np.load(Y_PATH)[:, 0].astype(float)
    gx, gy, spacing = _particle_grid(x, y)
    tag = f"T{INTEGRATION_T_S:g}s_{EXTENSION}ext" + ("_masked" if MASK_EXITED else "")
    ftle_file = SNAPSHOT_PATH.with_name(f"ftle_{CASE_NAME}_f{frame}_{tag}.npy")
    if reuse and ftle_file.exists():
        ftle_k = np.load(ftle_file)
        print(f"Loaded FTLE {ftle_file.name}")
    else:
        t0 = time.time()
        ftle_k = backward_ftle(VelocityField(x, y), frame, gx, gy, spacing)
        print(f"FTLE computed in {time.time() - t0:.1f} s on a {gx.shape[1]} x {gx.shape[0]} particle grid")
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(ftle_file, ftle_k)
    lo, hi = FTLE_LIMITS if FTLE_LIMITS else np.nanpercentile(ftle_k, FTLE_PERCENTILES)
    if MASK_EXITED:
        print(f"Masked (trajectory left the field): {100 * np.mean(~np.isfinite(ftle_k)):.1f}% of particles")

    c = _load_conc(np.load(C_PATH, mmap_mode="r"), frame)
    if OVERLAY_FIELD == "grad" and grad_lims is None:
        grad_lims = GRAD_LIMITS
        if grad_lims is None:
            g = conc_gradient(c, x, y)
            vmax = float(np.nanpercentile(g[g > 0], 99.5))
            grad_lims = (vmax / GRAD_DYNAMIC_RANGE, vmax)
    out_dir = Path(out_dir) if out_dir else SNAPSHOT_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = SNAPSHOT_PATH.stem.replace("ftle_conc_", "ftle_concgrad_") if OVERLAY_FIELD == "grad" else SNAPSHOT_PATH.stem
    fig, ax = _setup_figure(x, y, lo, hi, SNAPSHOT_DPI, grad_lims)
    _draw_frame(ax, gx, gy, ftle_k, x, y, c, lo, hi, frame * DT_S, grad_lims)  # absolute time in the record
    fig.savefig(out_dir / f"{stem}_{tag}.png", dpi=SNAPSHOT_DPI)
    plt.close(fig)
    if not ftle_only:
        print(f"Saved {out_dir / f'{stem}_{tag}.png'}")
        return

    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    cf = ax.contourf(gx, gy, np.ma.masked_invalid(ftle_k), levels=np.linspace(lo, hi, FTLE_LEVELS), cmap=FTLE_CMAP,
                     extend="both")
    _draw_masked_hatch(ax, gx, gy, ftle_k)
    fig.colorbar(cf, ax=ax, label="backward FTLE (1/s)", shrink=0.8, pad=0.02,
                 ticks=np.arange(np.ceil(lo), np.floor(hi) + 1))
    ax.set_xlim(XLIM if XLIM else (x.min(), x.max()))
    ax.set_ylim(YLIM if YLIM else (y.min(), y.max()))
    ax.set_aspect("equal")
    _label_axes(ax)
    ax.set_title(f"{CASE_NAME}, t = {frame * DT_S:.2f} s: backward FTLE (T = {INTEGRATION_T_S:g} s, {EXTENSION} ext.)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(SNAPSHOT_PATH.with_name(f"ftle_only_{CASE_NAME}_f{frame}_{tag}.png"), dpi=SNAPSHOT_DPI)
    plt.close(fig)
    print(f"FTLE colour limits: {lo:.2f} to {hi:.2f} 1/s")
    print(f"Saved snapshot, FTLE-only figure and FTLE array ({tag}) in {SNAPSHOT_PATH.parent}")


def _ftle_chunk(args, verbose: bool = False) -> np.ndarray:
    """FTLE for a list of frames of one case (top-level so it can run in a worker process; module defaults are used
    for every other setting)."""
    case, frames = args
    set_case(case)
    x = np.load(X_PATH)[0, :].astype(float)
    y = np.load(Y_PATH)[:, 0].astype(float)
    gx, gy, spacing = _particle_grid(x, y)
    vel = VelocityField(x, y)
    out = np.empty((len(frames), *gx.shape), dtype=np.float32)
    t0 = time.time()
    for k, f in enumerate(frames):
        out[k] = backward_ftle(vel, int(f), gx, gy, spacing)
        if verbose and k % 50 == 0:
            print(f"  FTLE {k + 1}/{len(frames)}, {time.time() - t0:.0f} s", flush=True)
    return out


def run_case(case: str) -> None:
    """30 s video for one case, starting at its SNAPSHOT_FRAMES frame."""
    global START_FRAME
    START_FRAME = SNAPSHOT_FRAMES[case]
    set_case(case)
    main()


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    x = np.load(X_PATH)[0, :].astype(float)
    y = np.load(Y_PATH)[:, 0].astype(float)
    frames = list(range(START_FRAME, START_FRAME + int(round(DURATION_S / DT_S))))
    if MAX_FRAMES_TEST:
        frames = frames[:MAX_FRAMES_TEST]

    gx, gy, spacing = _particle_grid(x, y)

    # ---- stage 1: FTLE time series
    if FTLE_CACHE.exists() and not MAX_FRAMES_TEST:
        ftle = np.load(FTLE_CACHE, mmap_mode="r")
        print(f"Loaded FTLE cache {FTLE_CACHE} {ftle.shape}")
    else:
        t0 = time.time()
        if N_WORKERS > 1 and len(frames) > N_WORKERS:
            # contiguous chunks, so each worker reuses its cached velocity frames
            chunks = [list(c) for c in np.array_split(frames, N_WORKERS)]
            with mp.get_context("spawn").Pool(N_WORKERS) as pool:
                parts = pool.map(_ftle_chunk, [(CASE_NAME, c) for c in chunks])
            ftle = np.concatenate(parts, axis=0)
        else:
            ftle = _ftle_chunk((CASE_NAME, frames), verbose=True)
        print(f"FTLE for {len(frames)} frames computed in {time.time() - t0:.0f} s ({N_WORKERS} workers)", flush=True)
        if not MAX_FRAMES_TEST:
            FTLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.save(FTLE_CACHE, ftle)
            print(f"Saved FTLE cache {FTLE_CACHE}")

    lo, hi = FTLE_LIMITS if FTLE_LIMITS else np.nanpercentile(np.asarray(ftle[:20]), FTLE_PERCENTILES)
    print(f"FTLE colour limits: {lo:.2f} to {hi:.2f} 1/s")

    # ---- stage 2: render
    c_mm = np.load(C_PATH, mmap_mode="r")
    fig, ax = _setup_figure(x, y, lo, hi, DPI)
    width, height = fig.canvas.get_width_height()
    width, height = width - width % 2, height - height % 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF), str(OUT_PATH)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    artists: list = []
    t0 = time.time()
    for k, f in enumerate(frames):
        for a in artists:
            a.remove()
        artists = _draw_frame(ax, gx, gy, ftle[k], x, y, _load_conc(c_mm, f), lo, hi, f * DT_S)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3]
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
        if k == 0 and TEST_PNG:
            plt.imsave(TEST_PNG, img)
        if k % 50 == 0:
            print(f"  frame {k + 1}/{len(frames)}, {time.time() - t0:.0f} s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    plt.close(fig)
    print(f"Saved {OUT_PATH} ({len(frames)} frames, {width}x{height}, {len(frames) / FPS:.1f} s at {FPS} fps)")


if __name__ == "__main__":
    # python -m tools.ftle_conc_overlay_video <case>   (default: CASE_NAME from the settings)
    run_case(sys.argv[1] if len(sys.argv) > 1 else CASE_NAME)
