"""Grid handling, plume centroid/width, and representative-point selection.

Grid convention: x (columns) is cross-stream, y (rows) is streamwise. The PLIF grid files store
x varying along axis 1 and y along axis 0, both in mm. The source is at the high-row end, so the
downstream distance is s = y_max - y (configurable via ``source_at_high_row``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Physical coordinates of the (ny, nx) analysis grid."""

    x_mm: np.ndarray  # (nx,) cross-stream coordinate of each column [mm]
    y_mm: np.ndarray  # (ny,) streamwise grid coordinate of each row [mm]
    source_at_high_row: bool = True

    @property
    def shape(self) -> tuple[int, int]:
        return self.y_mm.size, self.x_mm.size

    @property
    def dx_m(self) -> float:
        """Signed column spacing [m] (d x / d column)."""
        return float(np.mean(np.diff(self.x_mm))) / 1000

    @property
    def dy_m(self) -> float:
        """Signed row spacing [m] (d y / d row)."""
        return float(np.mean(np.diff(self.y_mm))) / 1000

    @property
    def downstream_mm(self) -> np.ndarray:
        """Downstream distance s [mm] of each row from the source end of the grid."""
        return self.y_mm.max() - self.y_mm if self.source_at_high_row else self.y_mm - self.y_mm.min()


def load_grid(x_path: Path | str, y_path: Path | str, *, source_at_high_row: bool = True, rtol: float = 1e-3) -> Grid:
    """Load PLIF grid files ((ny, nx) arrays in mm) and check that the spacing is uniform."""
    xg = np.load(x_path)
    yg = np.load(y_path)
    if xg.shape != yg.shape or xg.ndim != 2:
        raise ValueError(f"Grid shapes differ or are not 2D: {xg.shape}, {yg.shape}")
    x_mm = np.asarray(xg[0, :], dtype=float)
    y_mm = np.asarray(yg[:, 0], dtype=float)
    for name, v in (("x", x_mm), ("y", y_mm)):
        d = np.diff(v)
        if np.std(d) > rtol * abs(np.mean(d)):
            raise ValueError(f"Non-uniform {name} grid spacing: mean {np.mean(d):.5f} mm, std {np.std(d):.2e} mm")
    return Grid(x_mm=x_mm, y_mm=y_mm, source_at_high_row=source_at_high_row)


def plume_centroid_and_width(
    C: np.ndarray, x_mm: np.ndarray, *, background: float = 0.0, min_peak: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-stream centroid x_c(y) and second-moment width sigma_x(y) of the mean concentration, per row.

    sigma_x^2 = sum (x - x_c)^2 (C - b)_+ / sum (C - b)_+, with b the background level and (.)_+ = max(., 0).
    Rows whose peak (C - b) is below ``min_peak`` return NaN.
    """
    Cp = np.clip(np.nan_to_num(C - background, nan=0.0), 0.0, None)
    w = Cp.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xc = (Cp * x_mm[None, :]).sum(axis=1) / w
        var = (Cp * (x_mm[None, :] - xc[:, None]) ** 2).sum(axis=1) / w
    bad = (w <= 0) | (Cp.max(axis=1) < min_peak)
    xc[bad] = np.nan
    var[bad] = np.nan
    return xc, np.sqrt(var)


def check_orientation(C: np.ndarray, v: np.ndarray, *, source_at_high_row: bool, band: int = 50) -> dict:
    """Consistency check between the scalar and velocity stacks before any fluxes are computed.

    Source end = the end (first/last ``band`` rows) with the larger row-peak mean concentration.
    Flow direction = sign of the median streamwise velocity v (v < 0: toward decreasing row).
    Raises ValueError unless the source end matches ``source_at_high_row`` AND the mean flow points away from it.
    """
    pk = np.nanmax(C, axis=1)
    lo, hi = float(np.nanmean(pk[:band])), float(np.nanmean(pk[-band:]))
    v_med = float(np.nanmedian(v))
    source_high = hi > lo
    flow_to_low_rows = v_med < 0
    info = {"peak_C_low_rows": lo, "peak_C_high_rows": hi, "median_v": v_med,
            "source_at_high_row_detected": bool(source_high), "flow_toward_low_rows": bool(flow_to_low_rows)}
    if source_high != source_at_high_row:
        raise ValueError(f"Scalar source is at the {'high' if source_high else 'low'}-row end but config says "
                         f"source_at_high_row={source_at_high_row}; is the PLIF stack flipped? {info}")
    if flow_to_low_rows != source_high:
        raise ValueError(f"Mean flow (median v = {v_med:+.3f} m/s) points toward the scalar source: the PLIF and "
                         f"PIV stacks are inconsistent (flip_plif_rows?). {info}")
    return info


@dataclass(frozen=True)
class Point:
    name: str
    row: int
    col: int
    x_mm: float
    y_mm: float
    downstream_mm: float


def select_points(
    specs: list[dict], grid: Grid, C: np.ndarray, *, background: float = 0.0
) -> list[Point]:
    """Resolve representative points from specs like
    {"name": ..., "downstream_mm": s, "x_offset_sigma": k} (x = x_c + k sigma_x at that row) or
    {"name": ..., "downstream_mm": s, "x_mm": x} (absolute cross-stream position)."""
    xc, sig = plume_centroid_and_width(C, grid.x_mm, background=background)
    s = grid.downstream_mm
    points = []
    for spec in specs:
        row = int(np.argmin(np.abs(s - spec["downstream_mm"])))
        if "x_mm" in spec:
            x = float(spec["x_mm"])
        else:
            x = float(xc[row] + spec.get("x_offset_sigma", 0.0) * sig[row])
            if not np.isfinite(x):
                raise ValueError(f"Point {spec['name']}: no plume centroid at downstream {spec['downstream_mm']} mm")
        col = int(np.argmin(np.abs(grid.x_mm - x)))
        points.append(Point(spec["name"], row, col, float(grid.x_mm[col]), float(grid.y_mm[row]), float(s[row])))
    return points
