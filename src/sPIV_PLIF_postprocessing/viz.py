"""Visualization helpers (placeholders).

Keep plotting utilities here so `pipeline` stays focused on orchestration.
"""
from __future__ import annotations

from typing import Any
import logging

logger = logging.getLogger("sPIV_PLIF.viz")


def save_animation(anim: Any, out_path: str, fps: int = 10, dpi: int = 150) -> None:
    try:
        anim.save(out_path, fps=fps, dpi=dpi)
        logger.info("Saved animation to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save animation: %s", exc)
