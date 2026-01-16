"""
Update mean_fields w means and recompute w fluctuating stacks.

Edit the paths/settings below, then run:
    python tools/update_w_mean_and_flx.py
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

# -------------------------------------------------------------------
# Edit these paths/settings for your dataset
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData")
MEAN_FIELDS_DIR = BASE_PATH / "mean_fields"
PIV_DIR = BASE_PATH / "PIV"
FLX_DIR = BASE_PATH / "flow_properties" / "flx_u_v_w"
CHUNK_SIZE = 100
# -------------------------------------------------------------------


def _case_from_mean_fields(path: Path) -> str | None:
    match = re.match(r"^mean_fields_(.+)\.npz$", path.name)
    if match:
        return match.group(1)
    return None


def _case_from_w_flx(path: Path) -> str | None:
    match = re.match(r"^w_flx_(.+?)_FINAL", path.name)
    if match:
        return match.group(1)
    return None


def _compute_nanmean_time(w_path: Path, chunk_size: int) -> np.ndarray:
    w_stack = np.load(w_path, mmap_mode="r")
    if w_stack.ndim != 3:
        raise ValueError(f"Expected 3D w stack; got shape {w_stack.shape} in {w_path}")
    nx, ny, nt = w_stack.shape
    sum_accum = np.zeros((nx, ny), dtype=np.float64)
    count_accum = np.zeros((nx, ny), dtype=np.float64)
    for t_start in range(0, nt, chunk_size):
        t_stop = min(t_start + chunk_size, nt)
        chunk = np.asarray(w_stack[:, :, t_start:t_stop])
        finite = np.isfinite(chunk)
        sum_accum += np.nansum(chunk, axis=2)
        count_accum += np.sum(finite, axis=2)
    mean = sum_accum / np.where(count_accum == 0, np.nan, count_accum)
    return mean.astype(np.float32)


def _update_mean_fields(npz_path: Path, w_mean: np.ndarray) -> None:
    with np.load(npz_path) as data:
        payload = {key: data[key] for key in data.files}
    payload["w"] = w_mean
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **payload)
    tmp_path.replace(npz_path)


def _update_w_flx(w_flx_path: Path, w_path: Path, w_mean: np.ndarray, chunk_size: int) -> None:
    w_stack = np.load(w_path, mmap_mode="r")
    if w_stack.ndim != 3:
        raise ValueError(f"Expected 3D w stack; got shape {w_stack.shape} in {w_path}")
    if w_mean.shape != w_stack.shape[:2]:
        raise ValueError(
            f"w_mean shape {w_mean.shape} does not match w stack {w_stack.shape[:2]} for {w_flx_path}"
        )
    out = np.load(w_flx_path, mmap_mode="r+")
    # if out.shape != w_stack.shape:
    #     raise ValueError(f"w_flx shape {out.shape} does not match w stack {w_stack.shape} in {w_flx_path}")

    nx, ny, nt = w_stack.shape
    w_mean_3d = w_mean[:, 100:500, None]
    for t_start in range(0, nt, chunk_size):
        t_stop = min(t_start + chunk_size, nt)
        w_chunk = np.asarray(w_stack[:, 100:500, t_start:t_stop])
        out[:, :, t_start:t_stop] = w_chunk - w_mean_3d
    out.flush()


def main() -> None:
    mean_fields = sorted(MEAN_FIELDS_DIR.glob("mean_fields_*.npz"))
    # if not mean_fields:
    #     raise FileNotFoundError(f"No mean_fields_*.npz found in {MEAN_FIELDS_DIR}")

    # w_means: dict[str, np.ndarray] = {}
    # for npz_path in mean_fields:
    #     case = _case_from_mean_fields(npz_path)
    #     if case is None:
    #         print(f"Skipping unexpected mean_fields file: {npz_path}")
    #         continue
    #     w_path = PIV_DIR / f"piv_{case}_w.npy"
    #     if not w_path.exists():
    #         raise FileNotFoundError(f"Missing w stack for case '{case}': {w_path}")
    #     w_mean = _compute_nanmean_time(w_path, CHUNK_SIZE)
    #     _update_mean_fields(npz_path, w_mean)
    #     w_means[case] = w_mean
    #     print(f"Updated mean_fields w for case '{case}' -> {npz_path}")

    w_means: dict[str, np.ndarray] = {}
    for npz_path in mean_fields:
        case = _case_from_mean_fields(npz_path)
        w_means[case] = np.load(npz_path)["w"]

    w_flx_files = sorted(FLX_DIR.glob("w_*.npy"))
    if not w_flx_files:
        raise FileNotFoundError(f"No w_*.npy found in {FLX_DIR}")

    for w_flx_path in w_flx_files:
        case = _case_from_w_flx(w_flx_path)
        if case is None:
            print(f"Skipping unexpected w file: {w_flx_path}")
            continue
        w_path = PIV_DIR / f"piv_{case}_w.npy"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing w stack for case '{case}': {w_path}")
        if case not in w_means:
            w_means[case] = _compute_nanmean_time(w_path, CHUNK_SIZE)
        _update_w_flx(w_flx_path, w_path, w_means[case], CHUNK_SIZE)
        print(f"Updated w fluctuations for case '{case}' -> {w_flx_path}")


if __name__ == "__main__":
    main()
