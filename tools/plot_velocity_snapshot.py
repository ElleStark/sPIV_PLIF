"""
Plot an instantaneous velocity snapshot for each component (u, v, w)
and the total speed magnitude in a 2x2 panel.

Edit the paths/settings below, then run:
    python tools/plot_velocity_snapshot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
PIV_DIR = BASE_PATH / "PIV"
OUT_DIR = BASE_PATH / "Plots" / "Instantaneous"
CASE_NAME = "baseline"
FRAME_IDX = 0
U_PATH = PIV_DIR / f"piv_{CASE_NAME}_u.npy"
V_PATH = PIV_DIR / f"piv_{CASE_NAME}_v.npy"
W_PATH = PIV_DIR / f"piv_{CASE_NAME}_w.npy"
X_COORDS_PATH: Path | None = BASE_PATH / "x_coords.npy"
Y_COORDS_PATH: Path | None = BASE_PATH / "y_coords.npy"
USE_MEMMAP = True
CMAP = "RdBu_r"
COMPONENT_VMIN: float | None = -0.1
COMPONENT_VMAX: float | None = 0.1
SPEED_VMIN: float | None = 0.0
SPEED_VMAX: float | None = None
X_LIMITS: tuple[float, float] | None = (-100.0, 100.0)
FIGSIZE = (12, 10)
FIG_DPI = 600
XLABEL = "y"
YLABEL = "x"
# -------------------------------------------------------------------


def _load_coords(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path)


def _load_frame(path: Path, frame_idx: int, *, use_memmap: bool) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing velocity file: {path}")
    stack = np.load(path, mmap_mode="r" if use_memmap else None)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D velocity stack at {path.name}; got shape {stack.shape}")
    if frame_idx < 0 or frame_idx >= stack.shape[2]:
        raise IndexError(f"FRAME_IDX {frame_idx} out of range for {path.name} with {stack.shape[2]} frames")
    return np.array(stack[:, :, frame_idx], copy=False)


def _resolve_coords(
    shape: tuple[int, int],
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    if x_coords is None:
        x_coords = np.arange(nx)
    if y_coords is None:
        y_coords = np.arange(ny)
    if x_coords.ndim != 1 or y_coords.ndim != 1:
        raise ValueError("x_coords and y_coords must be 1D arrays.")
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("x_coords/y_coords length must match the field grid.")
    return np.meshgrid(x_coords, y_coords, indexing="xy")


def _subset_x(
    fields: list[np.ndarray],
    x_coords: np.ndarray | None,
    x_limits: tuple[float, float] | None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if x_limits is None:
        return fields, x_coords

    nx = fields[0].shape[1]
    x_arr = np.asarray(x_coords) if x_coords is not None else np.arange(nx)
    if x_arr.ndim != 1 or len(x_arr) != nx:
        raise ValueError("x_coords must be a 1D array matching the field width.")

    x_min, x_max = sorted(x_limits)
    mask = (x_arr >= x_min) & (x_arr <= x_max)
    if not np.any(mask):
        raise ValueError(f"No x points fall within requested X_LIMITS {x_limits}.")

    x_idx = np.where(mask)[0]
    return [field[:, x_idx] for field in fields], x_arr[x_idx]


def _component_limits(fields: list[np.ndarray]) -> tuple[float, float]:
    if COMPONENT_VMIN is not None and COMPONENT_VMAX is not None:
        return COMPONENT_VMIN, COMPONENT_VMAX

    finite = np.concatenate([field[np.isfinite(field)] for field in fields])
    if finite.size == 0:
        raise ValueError("No finite component velocities available to determine color limits.")
    abs_max = float(np.nanmax(np.abs(finite)))
    return (
        COMPONENT_VMIN if COMPONENT_VMIN is not None else -abs_max,
        COMPONENT_VMAX if COMPONENT_VMAX is not None else abs_max,
    )


def _speed_limits(speed: np.ndarray) -> tuple[float, float]:
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        raise ValueError("No finite speed values available to determine color limits.")
    vmin = SPEED_VMIN if SPEED_VMIN is not None else float(np.nanmin(finite))
    vmax = SPEED_VMAX if SPEED_VMAX is not None else float(np.nanmax(finite))
    return vmin, vmax


def main() -> None:
    u_frame = _load_frame(U_PATH, FRAME_IDX, use_memmap=USE_MEMMAP)
    v_frame = _load_frame(V_PATH, FRAME_IDX, use_memmap=USE_MEMMAP)
    w_frame = _load_frame(W_PATH, FRAME_IDX, use_memmap=USE_MEMMAP)

    if u_frame.shape != v_frame.shape or u_frame.shape != w_frame.shape:
        raise ValueError(
            f"Velocity component shapes must match; got u {u_frame.shape}, v {v_frame.shape}, w {w_frame.shape}"
        )

    speed = np.sqrt(u_frame**2 + v_frame**2 + w_frame**2)
    x_coords = _load_coords(X_COORDS_PATH)
    y_coords = _load_coords(Y_COORDS_PATH)
    (u_frame, v_frame, w_frame, speed), x_coords = _subset_x(
        [u_frame, v_frame, w_frame, speed],
        x_coords,
        X_LIMITS,
    )
    X, Y = _resolve_coords(u_frame.shape, x_coords, y_coords)

    component_vmin, component_vmax = _component_limits([u_frame, v_frame, w_frame])
    speed_vmin, speed_vmax = _speed_limits(speed)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    plot_specs = [
        (u_frame, "Instantaneous u", "u", Normalize(vmin=component_vmin, vmax=component_vmax)),
        (v_frame, "Instantaneous v", "v", Normalize(vmin=component_vmin, vmax=component_vmax)),
        (w_frame, "Instantaneous w", "w", Normalize(vmin=component_vmin, vmax=component_vmax)),
        (speed, "Instantaneous |V|", "|V|", Normalize(vmin=speed_vmin, vmax=speed_vmax)),
    ]

    for ax, (field, title, cbar_label, norm) in zip(axes.flat, plot_specs):
        mesh = ax.pcolormesh(X, Y, field, shading="auto", cmap=CMAP, norm=norm)
        fig.colorbar(mesh, ax=ax, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(YLABEL)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Instantaneous velocity snapshot: {CASE_NAME}, frame {FRAME_IDX}")

    out_path = OUT_DIR / f"velocity_snapshot_{CASE_NAME}_frame{FRAME_IDX}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved velocity snapshot to {out_path}")


if __name__ == "__main__":
    main()
