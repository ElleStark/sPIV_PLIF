"""
Load selected flow properties from FINAL_AllTimeSteps outputs and print basic stats.

Edit the settings below, then run:
    python tools/read_flow_properties.py
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
CASE_NAME = "fractal"
BASE_DIR = Path(f"E:/sPIV_PLIF_ProcessedData/flow_properties/Plots/{CASE_NAME}/FINAL_AllTimeSteps")
# Substrings to search for (case-insensitive) within filenames
PROP_KEYS = {
    "turbulence_intensity": ("turbulence", "intensity"),
    "taylor_re": ("taylor", "re"),
    "kolmogorov_time_scale": ("kolmogorov", "time"),
}
ALLOW_NPZ = True  # allow loading .npz as well as .npy
# -------------------------------------------------------------------


def _find_prop_file(base_dir: Path, substrings: tuple[str, ...]) -> Path:
    """Return the first file in base_dir matching all substrings (case-insensitive)."""
    candidates = []
    for ext in (".npy", ".npz"):
        if ext == ".npz" and not ALLOW_NPZ:
            continue
        candidates.extend(base_dir.glob(f"*{ext}"))
    lowered = [(p, p.name.lower()) for p in candidates]
    for path, name in lowered:
        if all(sub in name for sub in (s.lower() for s in substrings)):
            return path
    raise FileNotFoundError(f"No file in {base_dir} matching substrings {substrings}")


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"Ambiguous npz contents in {path}: {data.files}")
        return np.array(data[data.files[0]], copy=False)
    return np.load(path)


def _report(name: str, arr: np.ndarray) -> None:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"{name}: no finite values (shape {arr.shape})")
        return
    print(
        f"{name}: shape={arr.shape}, min={finite.min():.4g}, max={finite.max():.4g}, "
        f"mean={finite.mean():.4g}, median={np.median(finite):.4g}"
    )


def main() -> None:
    base_dir = BASE_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    loaded = {}
    for label, keys in PROP_KEYS.items():
        path = _find_prop_file(base_dir, keys)
        arr = _load_array(path)
        loaded[label] = (path, arr)

    for label, (path, arr) in loaded.items():
        print(f"\nLoaded {label} from {path}")
        _report(label, arr)


if __name__ == "__main__":
    main()
