"""Pipeline helpers for sPIV/PLIF postprocessing.

This module concentrates dataset-agnostic logic so tools can remain thin.
Implementations here call into the existing utility modules when available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging

from . import io_helpers
from . import process
from .visualization import viz


logger = logging.getLogger("sPIV_PLIF.pipeline")


def run(cfg: Dict[str, Any]) -> None:
    """Orchestrate a small demo of the pipeline using decomposed helpers.

    This implementation performs minimal steps: discover files, read the
    first frames (if IO backends are installed), build a shared grid, and
    run a single-frame interpolation to demonstrate connectivity.
    """
    logger.info("Running pipeline for dataset: %s", cfg.get("dataset", {}).get("name"))

    piv_dir = cfg.get("dataset", {}).get("piv_dir")
    plif_dir = cfg.get("dataset", {}).get("plif_dir")
    if not piv_dir or not plif_dir:
        logger.error("`piv_dir` and `plif_dir` must be set in dataset config")
        return

    im7_files, vc7_files = io_helpers.find_files(piv_dir, plif_dir)
    if not im7_files or not vc7_files:
        logger.error("No input files found in configured directories")
        return

    scalar, vec = io_helpers.read_first_frames(im7_files, vc7_files)
    if scalar is None or vec is None:
        logger.warning("IO backends missing or no files available; aborting interpolation demo")
        return

    # extract vector arrays used to create shared grid
    x_vec = vec["x"].values
    y_vec = vec["y"].values
    xg, yg = process.make_shared_grid(x_vec, y_vec, nx=cfg.get("processing", {}).get("nx", 540), ny=cfg.get("processing", {}).get("ny", 640))

    h_interp, u_interp, v_interp = process.interp_frame(scalar, vec, xg, yg)
    if h_interp is None:
        logger.error("Interpolation failed; see earlier errors")
        return

    logger.info("Interpolation successful — shapes: h=%s, u=%s, v=%s", getattr(h_interp, "shape", None), getattr(u_interp, "shape", None), getattr(v_interp, "shape", None))

