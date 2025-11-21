"""I/O helpers for the sPIV_PLIF pipeline.

Provides small wrappers around `lvpyio` and `pivpy` with graceful
degradation when those libraries are unavailable (so tests and CI can run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple
import logging

logger = logging.getLogger("sPIV_PLIF.io_helpers")


def find_files(piv_dir: str | Path, plif_dir: str | Path) -> Tuple[list[str], list[str]]:
    from glob import glob
    import os

    piv_dir = str(piv_dir)
    plif_dir = str(plif_dir)
    im7_files = sorted(glob(os.path.join(plif_dir, "*.im7")))
    vc7_files = sorted(glob(os.path.join(piv_dir, "*.vc7")))
    return im7_files, vc7_files


def read_first_frames(im7_files: Iterable[str], vc7_files: Iterable[str]) -> Tuple[Any, Any]:
    """Return (scalar_frame, vec_frame) or (None, None) if IO backends missing."""
    try:
        import lvpyio as lv
        from pivpy import io as piv_io
    except Exception:
        logger.warning("lvpyio or pivpy not available; IO helpers will return None")
        return None, None

    if not im7_files or not vc7_files:
        return None, None

    scalar = lv.read_buffer(im7_files[0])[0]
    vec = piv_io.load_vc7(vc7_files[0])
    return scalar, vec
