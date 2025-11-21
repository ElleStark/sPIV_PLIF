"""Processing helpers: grid creation and interpolation orchestration."""
from __future__ import annotations

from typing import Any, Dict, Tuple
import logging
import numpy as np

logger = logging.getLogger("sPIV_PLIF.process")


def make_shared_grid(x_vec: np.ndarray, y_vec: np.ndarray, nx: int = 540, ny: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    x_min = float(min(x_vec.min(), 0))
    x_max = float(x_vec.max())
    y_min = float(min(y_vec.min(), 0))
    y_max = float(y_vec.max())
    xg, yg = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    return xg, yg


def interp_frame(im7_data: Any, vec_df: Any, xg: np.ndarray, yg: np.ndarray):
    """Call the project's interpolation helper to map data onto shared grid."""
    # Try package-local helper first
    try:
        from sPIV_PLIF_postprocessing.utils import interp_shared_grid as isc
    except Exception:
        try:
            import importlib

            isc = importlib.import_module("utils.interp_shared_grid")
        except Exception:
            isc = None

    if isc is None:
        logger.error("Interpolation helper not found; cannot interpolate frame")
        return None, None, None

    return isc.interp_to_shared_grid(im7_data, vec_df, xg, yg)
