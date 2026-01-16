"""
Draw RMS overlay: concentration RMS pcolormesh + RMS velocity contours (no quiver).

Edit the paths/settings below, then run:
    python tools/draw_rms_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cmasher as cmr

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.sPIV_PLIF_postprocessing.io.load_processed_fields import load_fields
from src.sPIV_PLIF_postprocessing.visualization.viz import save_overlay_contour

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
# -------------------------------------------------------------------
CASE_NAME = "smSource"
U_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PIV/piv_{CASE_NAME}_u.npy")
V_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PIV/piv_{CASE_NAME}_v.npy")
W_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PIV/piv_{CASE_NAME}_w.npy")
C_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/plif_{CASE_NAME}_NOTsmoothed.npy")
OUT_PATH = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/RMS/rms_overlay_{CASE_NAME}.png")
CMIN = 0.01
CMAX = 0.35
RMS_OUT_DIR = Path(f"E:/sPIV_PLIF_ProcessedData/rms_fields/")
C_RMS_FILES = []
# C_RMS_FILES: list[Path] = [f"E:/sPIV_PLIF_ProcessedData/rms_fields/{CASE_NAME}_c_rms.npy"]
# C_RMS_FILES: list[Path] = [ "E:/sPIV_PLIF_ProcessedData/rms_fields/baseline_c_rms.npy", 
# "E:/sPIV_PLIF_ProcessedData/rms_fields/buoyant_c_rms.npy",
# "E:/sPIV_PLIF_ProcessedData/rms_fields/diffusive_c_rms.npy",
# "E:/sPIV_PLIF_ProcessedData/rms_fields/fractal_c_rms.npy",
# "E:/sPIV_PLIF_ProcessedData/rms_fields/smSource_c_rms.npy",
# "E:/sPIV_PLIF_ProcessedData/rms_fields/nearbed_c_rms.npy"
# ]  # Optional list of precomputed c_rms files to plot additionally
X_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/x_coords.npy")
Y_PATH: Path | None = Path("E:/sPIV_PLIF_ProcessedData/y_coords.npy")
LOG_SCALE = True
CMAP_NAME = "jet"
CMAP_SLICE = (0.0, 1.0)
C_UNDER: str | None = "white"
C_UNDER_TRANSITION: float | None = 0.05
C_UNDER_START: float | None = 1e-4
C_UNDER_END: float | None = 5e-4
PCOLORMESH_ALPHA = 0.85
CONTOUR_LEVELS: int | list[float] | None = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10]
CONTOUR_COLOR = "#000000"
CONTOUR_WIDTH = 0.75
CONTOUR_BOX_FRACTION: tuple[float, float, float, float] | None = None
CONTOUR_LEVELS_IN_BOX: int | list[float] | None = None
CONTOUR_WIDTH_IN_BOX: float | None = 0.8
CONTOUR_COLOR_IN_BOX: str | None = None
CONTOUR_CMAP: str | None = cmr.get_sub_cmap("cmr.neutral_r", 0.2, 1.0)
CONTOUR_CMAP_IN_BOX: str | None = "cmr.ember"
CONTOUR_LABELS = True
CONTOUR_LABELS_IN_BOX: bool | None = False
SHOW_VELOCITY_CONTOURS = False

# Quiver settings (disable to use velocity contours instead)
SHOW_QUIVER = False
QUIVER_CMAP: str | None = "cmr.neutral"
QUIVER_COLOR = "#333333"
QUIVER_COLORBAR = True
QUIVER_ALPHA = 1.0
QUIVER_VMIN: float | None = None
QUIVER_VMAX: float | None = None
STRIDE_ROWS = 30
STRIDE_COLS = 20
QUIVER_SCALE = 0.01
QUIVER_HEADWIDTH = 4.5
QUIVER_HEADLENGTH = 5.0
QUIVER_HEADAXISLENGTH = 3.5
QUIVER_TAILWIDTH = 0.003
SPEED_INCLUDE_W = False  # if True, speed contours use sqrt(u^2+v^2+w^2); otherwise u/v only

USE_MEMMAP = False
APPLY_MEDIAN_SMOOTH = False
MEDIAN_WINDOW = 3  # pixels
SKIP_C_RMS_CALC_WHEN_LISTED = True  # If True and C_RMS_FILES is non-empty, skip computing/saving c_rms


def _median_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply a kxk median filter (per frame if 3D)."""
    if k % 2 == 0 or k < 1:
        raise ValueError("Median window size must be an odd positive integer.")

    def _smooth2d(a: np.ndarray) -> np.ndarray:
        pad = k // 2
        padded = np.pad(a, pad_width=pad, mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(padded, (k, k))
        return np.nanmedian(windows, axis=(-2, -1))

    if arr.ndim == 2:
        return _smooth2d(arr)
    if arr.ndim == 3:
        return np.stack([_smooth2d(arr[:, :, i]) for i in range(arr.shape[2])], axis=2)
    raise ValueError(f"Median smooth expects 2D or 3D input; got shape {arr.shape}")


def _rms(arr: np.ndarray) -> np.ndarray:
    """Compute RMS of fluctuations along the last axis (time)."""
    if arr.ndim != 3:
        raise ValueError(f"RMS expects 3D input; got shape {arr.shape}")
    mean = np.nanmean(arr, axis=2, keepdims=True)
    fluctuations = arr - mean
    return np.sqrt(np.nanmean(np.square(fluctuations), axis=2))


def _overlay_out_path(base_path: Path, label: str | None) -> Path:
    """Return an output path with an optional suffix before the extension."""
    if label is None:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{label}{base_path.suffix}")


def _render_overlay(
    u_rms: np.ndarray,
    v_rms: np.ndarray,
    w_rms: np.ndarray | None,
    c_rms: np.ndarray,
    out_path: Path,
    title_suffix: str | None,
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
    contour_box: tuple[float, float, float, float] | None,
) -> None:
    title = "RMS overlay" if title_suffix is None else f"RMS overlay ({title_suffix})"
    save_overlay_contour(
        u_rms,
        v_rms,
        c_rms,
        out_path=out_path,
        w=w_rms,
        include_w=SPEED_INCLUDE_W,
        frame_idx=None,
        cmin=CMIN,
        cmax=CMAX,
        x_coords=x_coords,
        y_coords=y_coords,
        log_scale=LOG_SCALE,
        title=title,
        cmap_name=CMAP_NAME,
        cmap_slice=CMAP_SLICE,
        cmap_under=C_UNDER,
        cmap_under_transition=C_UNDER_TRANSITION,
        cmap_under_start=C_UNDER_START,
        cmap_under_end=C_UNDER_END,
        pcolormesh_alpha=PCOLORMESH_ALPHA,
        contour_levels=CONTOUR_LEVELS if SHOW_VELOCITY_CONTOURS else None,
        contour_color=CONTOUR_COLOR,
        contour_width=CONTOUR_WIDTH,
        contour_box=contour_box,
        contour_levels_in_box=CONTOUR_LEVELS_IN_BOX if SHOW_VELOCITY_CONTOURS else None,
        contour_color_in_box=CONTOUR_COLOR_IN_BOX,
        contour_width_in_box=CONTOUR_WIDTH_IN_BOX,
        contour_cmap=CONTOUR_CMAP,
        contour_cmap_in_box=CONTOUR_CMAP_IN_BOX,
        contour_labels=CONTOUR_LABELS,
        contour_labels_in_box=CONTOUR_LABELS_IN_BOX,
        show_quiver=SHOW_QUIVER,
        quiver_cmap=QUIVER_CMAP,
        quiver_color=QUIVER_COLOR,
        quiver_colorbar=QUIVER_COLORBAR,
        quiver_alpha=QUIVER_ALPHA,
        quiver_vmin=QUIVER_VMIN,
        quiver_vmax=QUIVER_VMAX,
        stride_rows=STRIDE_ROWS,
        stride_cols=STRIDE_COLS,
        quiver_scale=QUIVER_SCALE,
        quiver_headwidth=QUIVER_HEADWIDTH,
        quiver_headlength=QUIVER_HEADLENGTH,
        quiver_headaxislength=QUIVER_HEADAXISLENGTH,
        quiver_tailwidth=QUIVER_TAILWIDTH,
        xlim=(-100, 100),
    )


def main() -> None:
    x_coords = np.load(X_PATH) if X_PATH else None
    y_coords = np.load(Y_PATH) if Y_PATH else None

    skip_c_rms = SKIP_C_RMS_CALC_WHEN_LISTED and len(C_RMS_FILES) > 0
    c_rms: np.ndarray | None = None
    w_rms: np.ndarray | None = None
    if skip_c_rms:
        # Avoid heavy stack loads; rely on existing RMS arrays.
        u_rms_path = RMS_OUT_DIR / f"{CASE_NAME}_u_rms.npy"
        v_rms_path = RMS_OUT_DIR / f"{CASE_NAME}_v_rms.npy"
        if not u_rms_path.exists() or not v_rms_path.exists():
            raise FileNotFoundError("SKIP_C_RMS_CALC_WHEN_LISTED is True but u_rms/v_rms files are missing.")
        u_rms = np.load(u_rms_path)
        v_rms = np.load(v_rms_path)
        if SPEED_INCLUDE_W:
            w_rms_path = RMS_OUT_DIR / f"{CASE_NAME}_w_rms.npy"
            if not w_rms_path.exists():
                raise FileNotFoundError("SPEED_INCLUDE_W is True but w_rms file is missing.")
            w_rms = np.load(w_rms_path)
        grid_shape = u_rms.shape
    else:
        stacks = load_fields(
            U_PATH,
            V_PATH,
            W_PATH,
            C_PATH,
            enforce_float32=True,
            mmap_mode="r" if USE_MEMMAP else None,
            frame_idx=None,  # load full stacks for RMS
        )

        u_rms = _rms(stacks.u)
        v_rms = _rms(stacks.v)
        w_rms = _rms(stacks.w) if SPEED_INCLUDE_W else None
        c_rms = _rms(stacks.c)
        c_rms = _median_smooth(c_rms, MEDIAN_WINDOW) if APPLY_MEDIAN_SMOOTH else c_rms
        RMS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(RMS_OUT_DIR / "u_rms.npy", u_rms)
        np.save(RMS_OUT_DIR / "v_rms.npy", v_rms)
        np.save(RMS_OUT_DIR / "c_rms.npy", c_rms)
        if w_rms is not None:
            np.save(RMS_OUT_DIR / "w_rms.npy", w_rms)
        grid_shape = stacks.u.shape

    contour_box = None
    if CONTOUR_BOX_FRACTION is not None:
        xmin_f, xmax_f, ymin_f, ymax_f = CONTOUR_BOX_FRACTION
        x_min, x_max = (
            (float(np.min(x_coords)), float(np.max(x_coords))) if x_coords is not None else (0.0, float(grid_shape[1] - 1))
        )
        y_min, y_max = (
            (float(np.min(y_coords)), float(np.max(y_coords))) if y_coords is not None else (0.0, float(grid_shape[0] - 1))
        )
        contour_box = (
            x_min + xmin_f * (x_max - x_min),
            x_min + xmax_f * (x_max - x_min),
            y_min + ymin_f * (y_max - y_min),
            y_min + ymax_f * (y_max - y_min),
        )

    overlays: list[tuple[str | None, np.ndarray]] = []
    if c_rms is not None:
        overlays.append((None, c_rms))
    expected_shape = c_rms.shape if c_rms is not None else u_rms.shape
    for path in C_RMS_FILES:
        p = Path(path)
        loaded = np.load(p)
        if loaded.shape != expected_shape:
            raise ValueError(f"{p} has shape {loaded.shape}, expected {expected_shape}")
        overlays.append((p.stem, loaded))
    if not overlays:
        raise ValueError("No c_rms data available. Provide C_RMS_FILES or disable SKIP_C_RMS_CALC_WHEN_LISTED.")

    for label, c_field in overlays:
        out_path = _overlay_out_path(OUT_PATH, label)
        _render_overlay(u_rms, v_rms, w_rms, c_field, out_path, label, x_coords, y_coords, contour_box)
        print(f"Saved RMS overlay to {out_path}")

    print(f"Saved RMS fields to {RMS_OUT_DIR}")


if __name__ == "__main__":
    main()
