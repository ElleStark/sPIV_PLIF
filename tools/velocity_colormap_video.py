"""
Videos of instantaneous fields shown as colour maps (no vectors), on a colour scale fixed for the whole video and
shared by all cases:
    FIELD = "speed"  in-plane speed sqrt(u^2 + v^2) [m/s] (or "u", "v", "w"), PIV interpolated to the PLIF grid,
                     viridis
    FIELD = "conc"   PLIF concentration c/c0 on a log scale, jet with a white-to-jet blend at the low end (as the
                     instantaneous snapshots of tools/draw_inst_overlay.py, e.g. plif_fractal_175LogJet.png)
Frames are piped straight into ffmpeg; FPS = 20 (the acquisition rate) plays in real time.

Axes as the FTLE/concentration videos: vertical axis labelled x = downstream distance DOWNSTREAM_ORIGIN_MM - y
(0 at the source, increasing downward), horizontal axis labelled y.

Run from the repo root:
    python -m tools.velocity_colormap_video [case] [field]
"""
from __future__ import annotations

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
from matplotlib.colors import ListedColormap, LogNorm, Normalize  # noqa: E402

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
CASE_NAME = "baseline"
BASE = Path("E:/sPIV_PLIF_ProcessedData")
DT_S = 0.05
DURATION_S = 30.0
FPS = 20
# start frames match the FTLE/concentration videos (tools/ftle_conc_overlay_video.SNAPSHOT_FRAMES)
START_FRAMES = {"baseline": 1000, "buoyant": 175, "diffusive": 700, "fractal": 175, "nearbed": 500, "smSource": 39}
START_FRAME = 1000          # used for cases not in START_FRAMES

FIELD = "speed"             # "speed", "u", "v", "w" or "conc"

# velocity
VEL_LIMITS = (0.05, 0.45)   # m/s, shared by all cases (0.5-99.5th pct: baseline 0.03-0.33, fractal 0.14-0.41)
VEL_CMAP = "viridis"
NAN_COLOR = "0.85"          # PIV pixels without data

# concentration (draw_inst_overlay.py: jet, white-to-jet blend over the lowest 10%, white below the minimum)
CONC_LIMITS = (0.02, 1.0)   # c/c0, log scale; floor 0.02 hides background PLIF noise (0.005 showed it)
CONC_CMAP = "jet"
CONC_WHITE_BLEND = 0.1
CONC_ALPHA = 0.85
FLIP_PLIF_ROWS = {"baseline": True}  # baseline_PLIF.npy is stored upside down relative to its PIV stacks
# FIELD = "log1pconc": ln(1 + c/c0) on a linear scale with the same colormap; limits = ln(1.02)..ln(2), i.e. the
# same c = 0.02..1 range as the concentration videos (white below)
LOG1P_LIMITS = (float(np.log1p(0.02)), float(np.log1p(1.0)))
SNAPSHOT_DIR = BASE / "Plots/Instantaneous/log1p_concentration"
SNAPSHOT_DPI = 300

XLIM = (-100.0, 100.0)
YLIM = (0.0, 250.0)         # source at y = 300 mm: shows x >= 50 mm downstream, as the FTLE videos
DOWNSTREAM_ORIGIN_MM: float | None = 300.0  # None: raw grid labels
FIGSIZE = (5.6, 6.0)
DPI = 200
CRF = 18
OUT_DIR = BASE / "Plots/Instantaneous/videos"
LABELS = {"speed": r"speed $\sqrt{u^2 + v^2}$ (m/s)", "u": "u, cross-stream (m/s)", "v": "v, streamwise (m/s)",
          "w": "w, out of plane (m/s)", "conc": r"Concentration $c/c_0$", "log1pconc": r"$\ln(1 + c/c_0)$"}
TITLES = {"conc": "instantaneous concentration", "log1pconc": r"$\ln(1 + c/c_0)$"}
CONC_FIELDS = ("conc", "log1pconc")


def white_to_jet_cmap() -> ListedColormap:
    """jet with a white -> jet blend prepended over CONC_WHITE_BLEND of the scale and white below the minimum
    (same construction as viz.overlay_contour_figure with cmap_under='white', cmap_under_transition=0.1)."""
    colors = cmr.get_sub_cmap(CONC_CMAP, 0.0, 1.0)(np.linspace(0, 1, 256))
    n_under = max(2, int(len(colors) * min(CONC_WHITE_BLEND, 0.5)))
    white = np.array([1.0, 1.0, 1.0, 1.0])
    blend = np.stack([white * (1 - t) + colors[0] * t for t in np.linspace(0, 1, n_under)], axis=0)
    cmap = ListedColormap(np.vstack([blend, colors]))
    cmap.set_under("white")
    cmap.set_bad("white")
    return cmap


def _field(arrs: dict, f: int, flip: bool) -> np.ndarray:
    if FIELD == "speed":
        return np.hypot(np.asarray(arrs["u"][f], dtype=float), np.asarray(arrs["v"][f], dtype=float))
    a = np.asarray(arrs["conc" if FIELD in CONC_FIELDS else FIELD][f], dtype=float)
    if FIELD in CONC_FIELDS:
        a = a[::-1] if flip else a
    if FIELD == "conc":
        a = np.clip(a, CONC_LIMITS[0] * 1e-3, None)  # c <= 0 -> drawn as "under" (white), not masked
    elif FIELD == "log1pconc":
        a = np.log1p(np.clip(a, 0.0, None))      # c < 0 (noise) -> 0, drawn white
    return np.ma.masked_invalid(a)


def _label_axes(ax) -> None:
    if DOWNSTREAM_ORIGIN_MM is None:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        return
    origin = DOWNSTREAM_ORIGIN_MM
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(origin - np.arange(0.0, origin + 1e-9, 50.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{origin - v:g}"))
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("x (mm)")


def _load(case: str):
    if FIELD in CONC_FIELDS:
        arrs = {"conc": np.load(BASE / f"PLIF/{case}_PLIF.npy", mmap_mode="r")}
    else:
        comps = ("u", "v") if FIELD == "speed" else (FIELD,)
        arrs = {q: np.load(BASE / f"PIV/Interpolated_to_PLIF/piv_{case}_{q}.npy", mmap_mode="r") for q in comps}
    x = np.load(BASE / f"PLIF/{case}_xgrid.npy")[0, :].astype(float)
    y = np.load(BASE / f"PLIF/{case}_ygrid.npy")[:, 0].astype(float)
    return arrs, x, y, FLIP_PLIF_ROWS.get(case, False)


def _build_figure(field0: np.ndarray, x: np.ndarray, y: np.ndarray, dpi: int):
    """Figure, colour mesh and title artist with the FIELD's colormap, fixed scale and axis labelling."""
    if FIELD == "conc":
        cmap, norm, alpha, extend = white_to_jet_cmap(), LogNorm(*CONC_LIMITS), CONC_ALPHA, "min"
    elif FIELD == "log1pconc":
        cmap, norm, alpha, extend = white_to_jet_cmap(), Normalize(*LOG1P_LIMITS), CONC_ALPHA, "min"
    else:
        cmap = plt.get_cmap(VEL_CMAP).copy()
        cmap.set_bad(NAN_COLOR)
        norm, alpha, extend = Normalize(*VEL_LIMITS), 1.0, "both"
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=dpi)
    pm = ax.pcolormesh(x, y, field0, cmap=cmap, norm=norm, alpha=alpha, shading="auto", rasterized=True)
    fig.colorbar(pm, ax=ax, label=LABELS[FIELD], shrink=0.8, pad=0.02, extend=extend)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    _label_axes(ax)
    title = ax.set_title("placeholder title for layout")
    fig.tight_layout()
    return fig, pm, title


def _title(case: str, f: int) -> str:
    return f"{case}, t = {f * DT_S:5.2f} s: {TITLES.get(FIELD, 'instantaneous velocity')}"


def _stem() -> str:
    return {"conc": "concentration", "log1pconc": "log1p_concentration"}.get(FIELD, f"velocity_{FIELD}")


def snapshot(case: str = CASE_NAME, frame: int | None = None) -> Path:
    """Single-frame PNG of FIELD in the video format, saved to SNAPSHOT_DIR."""
    frame = START_FRAMES.get(case, START_FRAME) if frame is None else frame
    arrs, x, y, flip = _load(case)
    fig, _, title = _build_figure(_field(arrs, frame, flip), x, y, SNAPSHOT_DPI)
    title.set_text(_title(case, frame))
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT_DIR / f"{_stem()}_{case}_f{frame}.png"
    fig.savefig(out, dpi=SNAPSHOT_DPI)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main(case: str = CASE_NAME) -> None:
    start = START_FRAMES.get(case, START_FRAME)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    arrs, x, y, flip = _load(case)
    frames = range(start, start + int(round(DURATION_S / DT_S)))
    fig, pm, title = _build_figure(_field(arrs, frames[0], flip), x, y, DPI)
    width, height = fig.canvas.get_width_height()
    width, height = width - width % 2, height - height % 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_stem()}_{case}_f{start}_{DURATION_S:g}s_{FPS}fps.mp4"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF), str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k, f in enumerate(frames):
        pm.set_array(_field(arrs, f, flip).ravel())
        title.set_text(_title(case, f))
        fig.canvas.draw()
        proc.stdin.write(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3]).tobytes())
        if k % 100 == 0:
            print(f"  frame {k + 1}/{len(frames)}, {time.time() - t0:.0f} s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    plt.close(fig)
    print(f"Saved {out} ({len(frames)} frames, {width}x{height}, {len(frames) / FPS:.1f} s at {FPS} fps)")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        FIELD = sys.argv[2]
    main(sys.argv[1] if len(sys.argv) > 1 else CASE_NAME)
