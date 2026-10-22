"""
Video of the instantaneous concentration + velocity overlay, using the same figure function and settings as the
static snapshots from tools/draw_inst_overlay.py (Plots/Instantaneous/<Case>/).

Frames are rendered with viz.overlay_contour_figure and piped straight into ffmpeg (no intermediate PNGs). With
FPS = 20 (the acquisition rate) the video plays in real time: DURATION_S seconds of video = DURATION_S seconds of flow.

Edit the settings below, then run from the repo root:
    python -m tools.draw_inst_overlay_video
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
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.visualization.viz import overlay_contour_figure  # noqa: E402

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
CASE_NAME = "baseline"
BASE = Path("E:/sPIV_PLIF_ProcessedData")
U_PATH = BASE / f"PIV/Interpolated_to_PLIF/piv_{CASE_NAME}_u.npy"
V_PATH = BASE / f"PIV/Interpolated_to_PLIF/piv_{CASE_NAME}_v.npy"
C_PATH = BASE / f"PLIF/{CASE_NAME}_PLIF.npy"
X_PATH = BASE / f"PLIF/{CASE_NAME}_xgrid.npy"
Y_PATH = BASE / f"PLIF/{CASE_NAME}_ygrid.npy"
# baseline_PLIF.npy is stored upside down relative to its PIV stacks (source at row 0); flip it on read
FLIP_PLIF_ROWS = {"baseline": True}

DT_S = 0.05            # acquisition interval (20 Hz)
START_FRAME = 1000
DURATION_S = 30.0      # seconds of flow
FPS = 20               # 20 = real time; 10 = half speed (same frames, video twice as long)
FRAME_STEP = 1         # use every n-th acquired frame

# Display-only temporal smoothing of the velocity arrows: centred Gaussian in time with this standard deviation in
# frames (radius 3 sigma, NaN-aware). None/0 = instantaneous. Concentration is always instantaneous.
VEL_TIME_SIGMA_FRAMES: float | None = None

# Fixed-length arrows: every arrow has the same length (direction only); speed is shown by arrow color instead.
# False = the static-snapshot quiver (length proportional to speed, settings below).
FIXED_LENGTH_ARROWS = True
ARROW_LENGTH_MM = 4.5                       # drawn length of every arrow [mm]; arrow spacing = stride x 0.317 mm
ARROW_CMAP = "gray"                         # speed colormap ...
ARROW_CMAP_SLICE = (0.0, 0.7)               # ... truncated so the fastest arrows stay visible on white
ARROW_SPEED_LIMITS = (0.05, 0.40)           # m/s mapped to the colormap ends
ARROW_WIDTH = 0.003                         # shaft width (fraction of axes width)
ARROW_SPEED_COLORBAR = True                 # horizontal speed colorbar under the plot

OUT_TAG = "_dense18_fixedlen"   # appended to the file name to keep variants side by side
OUT_PATH = BASE / f"Plots/Instantaneous/{CASE_NAME.capitalize()}/videos/" \
                  f"overlay_{CASE_NAME}_f{START_FRAME}_{DURATION_S:g}s_{FPS}fps{OUT_TAG}.mp4"
DPI = 200
CRF = 18               # x264 quality (lower = better, larger file)
CROP_PAD_IN = 0.1      # padding around the tight bounding box of the first frame [in]
TITLE_FMT = "{case}, t = {t:6.2f} s"

# Figure settings: same values as tools/draw_inst_overlay.py
CMIN = 0.005
CMAX = 0.6
LOG_SCALE = False
CMAP_NAME = "jet"
CMAP_SLICE = (0.0, 1.0)
C_UNDER: str | None = "white"
C_UNDER_TRANSITION: float | None = 0.1
C_UNDER_START: float | None = 0.01
C_UNDER_END: float | None = 0.02
PCOLORMESH_ALPHA = 0.85
X_LIMITS: tuple[float, float] | None = (-100, 100)
SHOW_QUIVER = True
QUIVER_CMAP: str | None = "grey"
QUIVER_COLOR = None
QUIVER_COLORBAR = False
QUIVER_ALPHA = 1.0
QUIVER_VMIN: float | None = 0.1
QUIVER_VMAX: float | None = 0.5
STRIDE_ROWS = 18      # static snapshots use 30
STRIDE_COLS = 18
QUIVER_SCALE = 0.05   # static: 0.035 (larger = shorter arrows)
QUIVER_HEADWIDTH = 4.0
QUIVER_HEADLENGTH = 4
QUIVER_HEADAXISLENGTH = 3
QUIVER_TAILWIDTH = 0.003  # static: 0.004
MAX_FRAMES_TEST: int | None = None  # e.g. 5 for a quick timing test


def _fixed_length_quiver(fig, ax, u, v, x, y):
    """Unit-direction arrows of constant drawn length, colored by speed |(u, v)|."""
    X, Y = np.meshgrid(x, y)
    sl = (slice(None, None, STRIDE_ROWS), slice(None, None, STRIDE_COLS))
    Xs, Ys, us, vs = X[sl], Y[sl], u[sl], v[sl]
    speed = np.hypot(us, vs)
    ok = np.isfinite(speed) & (speed > 0)
    if X_LIMITS is not None:
        ok &= (Xs >= min(X_LIMITS)) & (Xs <= max(X_LIMITS))
    Xs, Ys, us, vs, speed = Xs[ok], Ys[ok], us[ok], vs[ok], speed[ok]
    cmap = cmr.get_sub_cmap(ARROW_CMAP, *ARROW_CMAP_SLICE)
    q = ax.quiver(Xs, Ys, us / speed * ARROW_LENGTH_MM, vs / speed * ARROW_LENGTH_MM, speed,
                  angles="xy", scale_units="xy", scale=1.0, pivot="mid", cmap=cmap,
                  norm=Normalize(*ARROW_SPEED_LIMITS), width=ARROW_WIDTH, headwidth=QUIVER_HEADWIDTH,
                  headlength=QUIVER_HEADLENGTH, headaxislength=QUIVER_HEADAXISLENGTH)
    if ARROW_SPEED_COLORBAR:
        # vertical bar in the empty space left of the plot (the figure canvas does not extend below the axes)
        cax = ax.inset_axes([-0.36, 0.0, 0.05, 1.0])
        cb = fig.colorbar(q, cax=cax, orientation="vertical")
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
        cb.set_label(r"speed $\sqrt{u^2 + v^2}$ (m/s)")


def _figure(u, v, c, x, y, title):
    fig, ax = overlay_contour_figure(
        u, v, None, c, frame_idx=None, include_w=False, xlim=X_LIMITS,
        cmin=CMIN, cmax=CMAX, x_coords=x, y_coords=y, log_scale=LOG_SCALE, title=title,
        cmap_name=CMAP_NAME, cmap_slice=CMAP_SLICE, cmap_under=C_UNDER, cmap_under_transition=C_UNDER_TRANSITION,
        cmap_under_start=C_UNDER_START, cmap_under_end=C_UNDER_END, pcolormesh_alpha=PCOLORMESH_ALPHA,
        contour_levels=None, show_quiver=SHOW_QUIVER and not FIXED_LENGTH_ARROWS, quiver_cmap=QUIVER_CMAP,
        quiver_color=QUIVER_COLOR,
        quiver_colorbar=QUIVER_COLORBAR, quiver_alpha=QUIVER_ALPHA, quiver_vmin=QUIVER_VMIN, quiver_vmax=QUIVER_VMAX,
        stride_rows=STRIDE_ROWS, stride_cols=STRIDE_COLS, quiver_scale=QUIVER_SCALE, quiver_headwidth=QUIVER_HEADWIDTH,
        quiver_headlength=QUIVER_HEADLENGTH, quiver_headaxislength=QUIVER_HEADAXISLENGTH,
        quiver_tailwidth=QUIVER_TAILWIDTH,
    )
    if SHOW_QUIVER and FIXED_LENGTH_ARROWS:
        _fixed_length_quiver(fig, ax, u, v, x, y)
    fig.set_dpi(DPI)
    return fig


def _rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]


def main() -> None:
    plt.style.use("default")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    mm = {k: np.load(p, mmap_mode="r") for k, p in (("u", U_PATH), ("v", V_PATH), ("c", C_PATH))}
    x = np.load(X_PATH)[0, :].astype(float)
    y = np.load(Y_PATH)[:, 0].astype(float)
    flip = FLIP_PLIF_ROWS.get(CASE_NAME, False)
    n = int(round(DURATION_S / DT_S))
    frames = list(range(START_FRAME, min(START_FRAME + n, mm["u"].shape[0]), FRAME_STEP))
    if MAX_FRAMES_TEST:
        frames = frames[:MAX_FRAMES_TEST]

    # velocity for the arrows: instantaneous, or a centred Gaussian time average (NaN-aware) for display
    nt = mm["u"].shape[0]
    radius = int(np.ceil(3 * VEL_TIME_SIGMA_FRAMES)) if VEL_TIME_SIGMA_FRAMES else 0
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / VEL_TIME_SIGMA_FRAMES) ** 2) if radius else np.ones(1)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def raw_uv(j):
        if j not in cache:
            cache[j] = (np.array(mm["u"][j], dtype=np.float32), np.array(mm["v"][j], dtype=np.float32))
            for old in [k for k in cache if k < j - 2 * radius - 1]:
                del cache[old]
        return cache[j]

    def velocity(i):
        if radius == 0:
            return raw_uv(i)
        num_u = np.zeros((len(y), len(x)))
        num_v = np.zeros_like(num_u)
        den = np.zeros_like(num_u)
        for off, wt in zip(offsets, weights):
            j = i + off
            if 0 <= j < nt:
                uj, vj = raw_uv(j)
                ok = np.isfinite(uj) & np.isfinite(vj)
                num_u += np.where(ok, uj, 0.0) * wt
                num_v += np.where(ok, vj, 0.0) * wt
                den += ok * wt
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num_u / den).astype(np.float32), (num_v / den).astype(np.float32)

    def frame_fig(i):
        c = np.array(mm["c"][i], dtype=np.float32)
        if flip:
            c = c[::-1]
        u, v = velocity(i)
        return _figure(u, v, c, x, y, TITLE_FMT.format(case=CASE_NAME, t=(i - START_FRAME) * DT_S))

    # fixed crop from the first frame's tight bounding box (same pixel box for every frame)
    fig = frame_fig(frames[0])
    img = _rgb(fig)
    renderer = fig.canvas.get_renderer()
    all_axes = fig.axes + [ch for a in fig.axes for ch in a.child_axes]  # include inset axes (speed colorbar)
    boxes = [fig.get_tightbbox(renderer)] + \
            [a.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()) for a in all_axes]
    bb = Bbox.union(boxes).padded(CROP_PAD_IN)
    H = img.shape[0]
    r0, r1 = max(0, int(H - bb.y1 * DPI)), min(H, int(H - bb.y0 * DPI))
    c0, c1 = max(0, int(bb.x0 * DPI)), min(img.shape[1], int(bb.x1 * DPI))
    r1 -= (r1 - r0) % 2
    c1 -= (c1 - c0) % 2
    width, height = c1 - c0, r1 - r0
    plt.close(fig)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF), str(OUT_PATH)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k, i in enumerate(frames):
        fig = frame_fig(i)
        proc.stdin.write(np.ascontiguousarray(_rgb(fig)[r0:r1, c0:c1]).tobytes())
        plt.close(fig)
        if k % 50 == 0:
            print(f"  frame {k + 1}/{len(frames)} (acquired frame {i}), {time.time() - t0:.0f} s elapsed", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"Saved {OUT_PATH} ({len(frames)} frames, {width}x{height}, {len(frames) / FPS:.1f} s at {FPS} fps)")


if __name__ == "__main__":
    main()
