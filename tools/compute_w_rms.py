"""
Compute RMS of the w-component for multiple cases and save to rms_fields.

Edit the case list and base path below, then run:
    python tools/compute_w_rms.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
CASE_NAMES = ["smSource", "nearbed", "fractal", "diffusive", "buoyant", "baseline"]
TIME_AXIS = 2  # axis along which RMS is computed


def compute_rms(arr: np.ndarray, axis: int) -> np.ndarray:
    """Compute RMS along the specified axis using nanmean to ignore NaNs."""
    return np.sqrt(np.nanmean(arr**2, axis=axis))


def main() -> None:
    piv_dir = BASE_PATH / "PIV"
    out_dir = BASE_PATH / "rms_fields"
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASE_NAMES:
        w_path = piv_dir / f"piv_{case}_w.npy"
        if not w_path.exists():
            print(f"[skip] missing w file for case '{case}': {w_path}")
            continue
        w = np.load(w_path)
        if w.ndim != 3:
            raise ValueError(f"Expected 3D w array for case '{case}', got shape {w.shape}")
        w_rms = compute_rms(w, axis=TIME_AXIS)
        out_path = out_dir / f"{case}_w_rms.npy"
        np.save(out_path, w_rms)
        print(f"[ok] saved w_rms for case '{case}' -> {out_path}")


if __name__ == "__main__":
    main()
