"""
TKE, dissipation rate and derived scales in the two outer cross-stream bands (|x| ~ 62-150 mm), which lie outside the
|x| <= 100 mm sub-domain of tools/compute_flow_properties.py. Needed for the fractal grid, whose between-bar (gap)
columns are at |x| ~ 120-145 mm.

Same methods and settings as tools/compute_flow_properties.py: fluctuations about the stored time-mean fields,
|q'| > 1 m/s rejected per component, NaN-aware moments, central-difference gradients with half-spans 1, 2, 4 grid
points, dw/dz from continuity, eps_hybrid (primary) and eps_iso_long, gradient moments kept where >= max(10, N/20)
valid samples; lambda, Re_lambda, eta from analysis.flow_properties.turbulence_scales. Each band is a contiguous block
of columns, differentiated on its own (the outer 4 columns of each block are undefined for the widest stencil).

Crash-tolerant, segmented execution (the machine has rebooted under sustained heavy I/O):
  * running sums are checkpointed every CHECKPOINT_EVERY_S seconds to Outer_bands/checkpoint_<case>.npz and a rerun
    resumes from the last checkpoint;
  * each invocation stops after MAX_SECONDS_PER_RUN (checkpoint written) - run it repeatedly until it reports that all
    cases are complete; SLEEP_BETWEEN_CHUNKS_S throttles the read rate.

Output: flow_properties/Outer_bands/outer_band_flow_properties_<case>.npz with full-grid column indices, x_mm, s_mm and
(947, ncols) maps per band ("left", "right"), plus n_frames_used / n_frames_total.

Run from the repo root (repeat until "ALL CASES COMPLETE"):
    python -m tools.compute_outer_band_flow_properties
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from src.sPIV_PLIF_postprocessing.analysis import compute_fluctuating_components, load_mean_velocity_components  # noqa: E402
from src.sPIV_PLIF_postprocessing.analysis import flow_properties as FP  # noqa: E402

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
CASES = ["fractal", "baseline", "buoyant", "diffusive", "nearbed", "smSource"]
BASE = Path("E:/sPIV_PLIF_ProcessedData")
OUT_DIR = BASE / "flow_properties" / "Outer_bands"
BAND_ABS_X_MM = (62.0, 150.1)       # |x| range of each block (70-145 mm analysis band + stencil margin)
NU = 1.5e-5
VELOCITY_FLUCTUATION_LIMIT = 1.0    # m/s, as in compute_flow_properties
GRADIENT_STEPS = [1, 2, 4]
CHUNK_SIZE = 10                     # frames per read (smaller = less memory per step)
SLEEP_BETWEEN_CHUNKS_S = 0.1        # throttle
CHECKPOINT_EVERY_S = 120.0
MAX_SECONDS_PER_RUN = 480.0         # stop (with checkpoint) after this long; rerun to continue
MAX_CHUNKS_TEST: int | None = None  # e.g. 4 for a timing test (no checkpoint / no completion mark)


def _out_path(case: str) -> Path:
    return OUT_DIR / f"outer_band_flow_properties_{case}.npz"


def _ckpt_path(case: str) -> Path:
    return OUT_DIR / f"checkpoint_{case}.npz"


def is_complete(case: str) -> bool:
    """True if this case's output exists and used every frame (test runs and partial files are redone)."""
    path = _out_path(case)
    if not path.exists():
        return False
    with np.load(path) as d:
        return "n_frames_used" in d and int(d["n_frames_used"]) == int(d["n_frames_total"])


def _save_atomic(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _save_checkpoint(case: str, acc: dict, next_start: int) -> None:
    arrays = {"next_start": next_start}
    for name, a in acc.items():
        for c in "uvw":
            arrays[f"{name}|sums|{c}"] = a["sums"][c]
            arrays[f"{name}|counts|{c}"] = a["counts"][c]
        for st, gm in a["gms"].items():
            for key in gm.sums:
                arrays[f"{name}|gm{st}|sum|{key}"] = gm.sums[key]
                arrays[f"{name}|gm{st}|cnt|{key}"] = gm.counts[key]
    _save_atomic(_ckpt_path(case), arrays)


def _load_checkpoint(case: str, acc: dict) -> int:
    path = _ckpt_path(case)
    if not path.exists():
        return 0
    with np.load(path) as d:
        for name, a in acc.items():
            for c in "uvw":
                a["sums"][c] = d[f"{name}|sums|{c}"].copy()
                a["counts"][c] = d[f"{name}|counts|{c}"].copy()
            for st, gm in a["gms"].items():
                for key in gm.sums:
                    gm.sums[key] = d[f"{name}|gm{st}|sum|{key}"].copy()
                    gm.counts[key] = d[f"{name}|gm{st}|cnt|{key}"].copy()
        return int(d["next_start"])


def process_case(case: str, deadline: float) -> bool:
    """Accumulate until done or the deadline; returns True when the case output is written."""
    t0 = time.time()
    xg = np.load(BASE / f"PLIF/{case}_xgrid.npy").astype(float)
    yg = np.load(BASE / f"PLIF/{case}_ygrid.npy").astype(float)
    x, y = xg[0], yg[:, 0]
    dx = float(np.mean(np.diff(x))) / 1000
    dy = float(np.mean(np.diff(y))) / 1000
    s = y.max() - y
    blocks = {}
    for name, sign in (("left", -1), ("right", 1)):
        idx = np.flatnonzero((sign * x >= BAND_ABS_X_MM[0]) & (sign * x <= BAND_ABS_X_MM[1]))
        blocks[name] = slice(int(idx.min()), int(idx.max()) + 1)
    means = load_mean_velocity_components(case, base_path=BASE)
    mm = {c: np.load(BASE / f"PIV/Interpolated_to_PLIF/piv_{case}_{c}.npy", mmap_mode="r") for c in "uvw"}
    nt, ny = mm["u"].shape[0], mm["u"].shape[1]

    acc = {}
    for name, cs in blocks.items():
        shape = (ny, cs.stop - cs.start)
        acc[name] = {"sums": {c: np.zeros(shape) for c in "uvw"}, "counts": {c: np.zeros(shape) for c in "uvw"},
                     "gms": {st: FP.GradientMoments(shape) for st in GRADIENT_STEPS},
                     "mean": [m[:, cs].astype(np.float64) for m in means]}
    start = 0 if MAX_CHUNKS_TEST else _load_checkpoint(case, acc)
    if start:
        print(f"  [{case}] resuming at frame {start} of {nt}", flush=True)
    starts = list(range(start, nt, CHUNK_SIZE))
    if MAX_CHUNKS_TEST:
        starts = starts[:MAX_CHUNKS_TEST]
    last_ckpt = time.time()
    next_start = start
    for n, t_start in enumerate(starts):
        sl = slice(t_start, min(t_start + CHUNK_SIZE, nt))
        for name, cs in blocks.items():
            a = acc[name]
            chunk = [np.asarray(mm[c][sl, :, cs], dtype=np.float64) for c in "uvw"]
            fl = compute_fluctuating_components(*chunk, *a["mean"])
            for arr in fl:
                with np.errstate(invalid="ignore"):
                    arr[np.abs(arr) > VELOCITY_FLUCTUATION_LIMIT] = np.nan
            for c, arr in zip("uvw", fl):
                fin = np.isfinite(arr)
                a["sums"][c] += np.where(fin, arr * arr, 0.0).sum(axis=0)
                a["counts"][c] += fin.sum(axis=0)
            for st in GRADIENT_STEPS:
                a["gms"][st].add(FP.velocity_gradients(*fl, dx=dx, dy=dy, step=st))
        next_start = sl.stop
        if n % 100 == 0:
            print(f"  [{case}] frames {t_start}-{sl.stop} of {nt}, {time.time() - t0:.0f} s", flush=True)
        if not MAX_CHUNKS_TEST and next_start < nt:
            if time.time() - last_ckpt > CHECKPOINT_EVERY_S:
                _save_checkpoint(case, acc, next_start)
                last_ckpt = time.time()
            if time.time() > deadline:
                _save_checkpoint(case, acc, next_start)
                print(f"  [{case}] time budget reached: checkpoint at frame {next_start} of {nt}", flush=True)
                return False
        time.sleep(SLEEP_BETWEEN_CHUNKS_S)

    n_used = next_start
    out = {"s_mm": s, "nu": NU, "n_frames_used": n_used, "n_frames_total": nt}
    for name, cs in blocks.items():
        a = acc[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            ms = {c: np.where(a["counts"][c] > 0, a["sums"][c] / a["counts"][c], np.nan) for c in "uvw"}
        k = 0.5 * (ms["u"] + ms["v"] + ms["w"])
        out[f"{name}|cols"] = np.arange(cs.start, cs.stop)
        out[f"{name}|x_mm"] = x[cs]
        out[f"{name}|tke"] = k
        for st in GRADIENT_STEPS:
            gm = a["gms"][st].means(min_count=max(10, n_used // 20))
            for key, val in FP.dissipation_estimates(gm, nu=NU).items():
                out[f"{name}|{key}_step{st}"] = val
        sc = FP.turbulence_scales(out[f"{name}|eps_hybrid_step1"], k, nu=NU)
        for key in ("taylor_microscale", "taylor_Re", "kolmogorov_length"):
            out[f"{name}|{key}"] = sc[key]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_atomic(_out_path(case), out)
    if not MAX_CHUNKS_TEST and _ckpt_path(case).exists():
        _ckpt_path(case).unlink()
    print(f"[{case}] saved {_out_path(case)} ({n_used} frames)", flush=True)
    return True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + MAX_SECONDS_PER_RUN
    for case in CASES:
        if not MAX_CHUNKS_TEST and is_complete(case):
            continue
        if not process_case(case, deadline) or time.time() > deadline:
            done = [c for c in CASES if is_complete(c)]
            print(f"Stopped for this run. Complete: {done or 'none'}. Rerun to continue.", flush=True)
            return
    print("ALL CASES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
