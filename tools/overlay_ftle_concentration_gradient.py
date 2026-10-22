"""
Overlay an instantaneous concentration-gradient magnitude on a grey FTLE field (companion to
overlay_ftle_concentration.py, which overlays the concentration itself).

    |grad C| = sqrt((dC/dx)^2 + (dC/dy)^2)

computed from the same (smoothed) PLIF frame with second-order central differences (np.gradient) on the physical
grid, so the units are concentration per mm. Values below the colour-scale minimum are left transparent so the FTLE
field shows through. One figure per FTLE file; the concentration frame matches the FTLE start time,
frame = t0 / DT (e.g. FTLE_t100.0to100.05s... -> frame 2000 at 20 Hz).

Edit the settings below, then run from the repo root:
    python tools/overlay_ftle_concentration_gradient.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.colors import LogNorm, Normalize
from scipy import ndimage

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
FTLE_DIR = Path("E:/sPIV_PLIF_ProcessedData/FTLE_data")
# case -> FTLE file name; the concentration frame is taken from the FTLE start time in the name
FTLE_FILES: dict[str, str] = {
    "baseline": "FTLE_t50.0to50.05s_rk4_baselineCase.npy",
    "buoyant": "FTLE_t8.75to8.8s_rk4_buoyantCase.npy",
    "diffusive": "FTLE_t35.0to35.05s_rk4_diffusiveCase.npy",
    "fractal": "FTLE_t50.0to50.05s_fractalCasev2.npy",
    "nearbed": "FTLE_t100.0to100.05s_rk4_nearbedCase.npy",
    "smSource": "FTLE_t1.95to2.0s_rk4_smSourceCase.npy",
}
CONC_DIR = Path("E:/sPIV_PLIF_ProcessedData/PLIF/Old_smoothed")  # plif_<case>_smoothed.npy, (y, x, t) on the 599 grid
CONC_TEMPLATE = "plif_{case}_smoothed.npy"
DT = 0.05  # s per frame
FRAME_OVERRIDE: dict[str, int] = {}  # e.g. {"nearbed": 2000} to force a frame

# Coordinates [mm]. If the files are missing, the documented 599 grid is used: x = -149.5..149.5 mm, y = x + 150 mm
# (0.5 mm spacing; see io/readwrite_vc7_to_npy.py), matching the y = 0-250 mm limits of overlay_ftle_concentration.py.
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
DEFAULT_X_MM = np.arange(-149.5, 149.5 + 0.25, 0.5)
DEFAULT_Y_MM = DEFAULT_X_MM + 150.0

OUT_DIR = Path("E:/sPIV_PLIF_ProcessedData/Plots/FTLE")
OUT_TEMPLATE = "ftle_concgrad_overlay_{case}.png"

GRAD_SMOOTH_SIGMA_MM = 0.0  # optional extra Gaussian smoothing of C before differentiating (0 = none)
GRAD_CMAP = cmr.ocean_r  # same colormap family as the concentration overlay
GRAD_LOG_SCALE = True
GRAD_VMIN: float | None = None  # [C/mm]; None -> GRAD_VMAX / GRAD_DYNAMIC_RANGE
GRAD_VMAX: float | None = None  # [C/mm]; None -> 99.5th percentile of |grad C| in the plotted window
GRAD_DYNAMIC_RANGE = 100.0  # decades shown when GRAD_VMIN is None (log scale): vmax / vmin
GRAD_ALPHA = 0.7

FTLE_CMAP = cmr.get_sub_cmap("cmr.neutral_r", 0, 0.6)
FTLE_LEVELS: int | list[float] | None = 40
FTLE_VMIN: float | None = -5
FTLE_VMAX: float | None = 1

XLIM: tuple[float, float] | None = (-100.0, 100.0)
YLIM: tuple[float, float] | None = (0, 250)
DPI = 600
# -------------------------------------------------------------------


def _load_ftle(path: Path) -> np.ndarray:
    ftle = np.load(path)
    while ftle.ndim > 2 and ftle.shape[0] == 1:
        ftle = np.squeeze(ftle, axis=0)
    if ftle.ndim == 3:
        ftle = ftle[0, :, :]
    if ftle.ndim != 2:
        raise ValueError(f"FTLE array must be 2D after squeezing; got shape {ftle.shape}")
    return ftle.astype(float)


def _frame_from_name(name: str) -> int:
    m = re.search(r"_t([0-9.]+)to", name)
    if not m:
        raise ValueError(f"Cannot parse the FTLE start time from '{name}'; set FRAME_OVERRIDE")
    return int(round(float(m.group(1)) / DT))


def _coords(n_x: int, n_y: int) -> tuple[np.ndarray, np.ndarray]:
    if X_PATH is not None and Y_PATH is not None and X_PATH.exists() and Y_PATH.exists():
        x, y = np.load(X_PATH).astype(float), np.load(Y_PATH).astype(float)
    else:
        print("Coordinate files not found; using the documented 599 grid (x = -149.5..149.5 mm, y = 0.5..299.5 mm)")
        x, y = DEFAULT_X_MM, DEFAULT_Y_MM
    if x.size != n_x or y.size != n_y:
        raise ValueError(f"Coordinate lengths ({x.size}, {y.size}) do not match the grid ({n_x}, {n_y})")
    return x, y


def gradient_magnitude(conc: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, smooth_sigma_mm: float = 0.0) -> np.ndarray:
    """|grad C| [C/mm] with second-order central differences on the (possibly non-uniform) physical grid."""
    c = np.asarray(conc, dtype=float)
    if smooth_sigma_mm > 0:
        sig_px = (smooth_sigma_mm / abs(np.mean(np.diff(y_mm))), smooth_sigma_mm / abs(np.mean(np.diff(x_mm))))
        valid = np.isfinite(c)
        num = ndimage.gaussian_filter(np.where(valid, c, 0.0), sig_px, mode="nearest")
        den = ndimage.gaussian_filter(valid.astype(float), sig_px, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(valid, num / den, np.nan)
    dcdy, dcdx = np.gradient(c, y_mm, x_mm, edge_order=2)
    return np.hypot(dcdx, dcdy)


def plot_case(case: str, ftle_name: str) -> None:
    ftle_path = FTLE_DIR / ftle_name
    conc_path = CONC_DIR / CONC_TEMPLATE.format(case=case)
    if not ftle_path.exists() or not conc_path.exists():
        print(f"Skipping {case}: missing {ftle_path if not ftle_path.exists() else conc_path}")
        return
    frame = FRAME_OVERRIDE.get(case, _frame_from_name(ftle_name))
    ftle = _load_ftle(ftle_path)
    stack = np.load(conc_path, mmap_mode="r")
    if stack.ndim != 3 or not (0 <= frame < stack.shape[2]):
        raise ValueError(f"{case}: frame {frame} not available in stack of shape {stack.shape}")
    conc = np.array(stack[:, :, frame], dtype=float)
    if ftle.shape != conc.shape:
        raise ValueError(f"{case}: FTLE shape {ftle.shape} does not match concentration slice {conc.shape}")

    x, y = _coords(ftle.shape[1], ftle.shape[0])
    X, Y = np.meshgrid(x, y, indexing="xy")
    conc_plot = np.fliplr(conc)  # same orientation as overlay_ftle_concentration.py
    grad = gradient_magnitude(conc_plot, x, y, GRAD_SMOOTH_SIGMA_MM)

    in_view = np.ones_like(grad, dtype=bool)
    if XLIM is not None:
        in_view &= (X >= min(XLIM)) & (X <= max(XLIM))
    if YLIM is not None:
        in_view &= (Y >= min(YLIM)) & (Y <= max(YLIM))
    vals = grad[in_view & np.isfinite(grad) & (grad > 0)]
    vmax = GRAD_VMAX if GRAD_VMAX is not None else float(np.percentile(vals, 99.5))
    vmin = GRAD_VMIN if GRAD_VMIN is not None else vmax / GRAD_DYNAMIC_RANGE
    norm = LogNorm(vmin=vmin, vmax=vmax) if GRAD_LOG_SCALE else Normalize(vmin=vmin, vmax=vmax)

    ftle_vmin = float(np.nanmin(ftle)) if FTLE_VMIN is None else FTLE_VMIN
    ftle_vmax = float(np.nanmax(ftle)) if FTLE_VMAX is None else FTLE_VMAX
    ftle = np.clip(ftle, ftle_vmin, None)

    fig, ax = plt.subplots(figsize=(8, 6))
    if FTLE_LEVELS is not None:
        ax.contourf(X, Y, ftle, levels=FTLE_LEVELS, cmap=FTLE_CMAP, vmin=ftle_vmin, vmax=ftle_vmax, alpha=1.0)
    h = ax.pcolormesh(X, Y, np.ma.masked_where(~np.isfinite(grad) | (grad < vmin), grad), cmap=GRAD_CMAP, norm=norm,
                      shading="auto", alpha=GRAD_ALPHA)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if XLIM is not None:
        ax.set_xlim(XLIM)
    if YLIM is not None:
        ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{case}: FTLE ({ftle_name.split('_rk4')[0].split('_fractal')[0].replace('FTLE_', '')}) with "
                 f"|grad C| at frame {frame} (t = {frame * DT:.2f} s)", fontsize=9)
    fig.colorbar(h, ax=ax, label=r"$|\nabla C| = \sqrt{(\partial C/\partial x)^2 + (\partial C/\partial y)^2}$ [C/mm]")
    if FTLE_LEVELS is not None:
        sm = plt.cm.ScalarMappable(cmap=FTLE_CMAP)
        sm.set_clim(ftle_vmin, ftle_vmax)
        fig.colorbar(sm, ax=ax, label="FTLE")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_TEMPLATE.format(case=case)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  (frame {frame}, |grad C| colour range {vmin:.3g}-{vmax:.3g} C/mm)")


def main() -> None:
    for case, name in FTLE_FILES.items():
        plot_case(case, name)


if __name__ == "__main__":
    main()
