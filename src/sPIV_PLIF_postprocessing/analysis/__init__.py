"""sPIV_PLIF_postprocessing.analysis package
Analysis utilities for processing PIV/PLIF measurement data.
"""

from .intermittency import compute_intermittency, compute_intermittency_from_file

__all__ = [
    "compute_intermittency",
    "compute_intermittency_from_file",
]
