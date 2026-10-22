"""
Streamwise profiles of integral length scales (ILS) averaged across the cross-stream coordinate within two
flow regions, one figure per case and region:

    background : grid-turbulence background, columns 0-199 and 747-946 (200 columns from each edge)
    core       : combined turbulence/wake region, centerline +/- 100 columns (373-573)

Each figure has four panels: u cross-stream, u streamwise, v cross-stream, v streamwise ILS.
Inputs are the ILS_mpi_from_npy.py outputs ILS_<case>_<u|v>_<cross_stream|streamwise>.npy, shape (y, x) on the
PLIF grid, values in m (integrated to the first zero crossing of the autocorrelation).

Profiles are plotted against downstream distance s = y_max - y [mm] (source at the high-row end, s = 0), taken
from each case's PLIF y-grid. Row means are NaN-aware (edge columns without PIV data are skipped).

Note: near the left/right edges the cross-stream autocorrelation has one-sided support only, which biases the
cross-stream ILS there; the background region includes those columns.

Edit the settings below, then run from the repo root:
    python tools/plot_ils_region_profiles.py
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Edit these settings for your dataset
CASES: list[str] = ["baseline", "buoyant", "diffusive", "fractal", "nearbed", "smSource"]
ILS_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Integral_length_scales/ILS")
GRID_DIR = Path("E:/sPIV_PLIF_ProcessedData/PLIF")  # <case>_ygrid.npy for the downstream coordinate; None = row index
OUT_DIR = ILS_DIR / "Profiles_regions"
FILE_PREFIX = "ILS"
COMPONENTS = ["u_cross_stream", "u_streamwise", "v_cross_stream", "v_streamwise"]

# Column ranges (start, stop) with stop exclusive, i.e. Python slices
REGIONS: dict[str, list[tuple[int, int]]] = {
    "background": [(0, 200), (747, 947)],
    "core": [(373, 574)],
}
REGION_TITLES = {
    "background": "grid-turbulence background (columns 0-199 and 747-946)",
    "core": "turbulence/wake region (centerline +/- 100 columns, 373-573)",
}

COMPARE_FULLLAG = True  # also save figures overlaying first-zero (solid) and full-lag (dashed) ILS
ROWS_TO_AVG = 5  # running mean along the streamwise direction (rows); 1 = no smoothing
SCALE = 1000.0  # m -> mm
SHARE_Y = True  # same y-axis limits for all four panels of a figure
DPI = 300
TITLES = {
    "u_cross_stream": "u, cross-stream",
    "u_streamwise": "u, streamwise",
    "v_cross_stream": "v, cross-stream",
    "v_streamwise": "v, streamwise",
}
# -------------------------------------------------------------------


def _savefig(fig, out_path: Path, dpi: int, retries: int = 5) -> None:
    """Save via a temporary file and atomic replace, retrying: overwriting a PNG on E: intermittently fails with
    Errno 22 while another process (viewer, indexer, antivirus) briefly holds the file."""
    tmp = out_path.with_name(out_path.stem + "_tmp" + out_path.suffix)
    fig.savefig(tmp, dpi=dpi)
    for attempt in range(retries):
        try:
            os.replace(tmp, out_path)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    raise OSError(f"Could not replace {out_path} (file locked?); new figure left at {tmp}")


def _region_columns(ranges: list[tuple[int, int]], nx: int) -> np.ndarray:
    cols = np.concatenate([np.arange(max(0, a), min(nx, b)) for a, b in ranges])
    return np.unique(cols)


def _smooth(profile: np.ndarray, k: int) -> np.ndarray:
    """Centered running mean over k rows, NaN-aware (edges use the available samples)."""
    if k <= 1:
        return profile
    fin = np.isfinite(profile)
    kernel = np.ones(k)
    num = np.convolve(np.where(fin, profile, 0.0), kernel, mode="same")
    den = np.convolve(fin.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _region_profile(arr: np.ndarray, cols: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        return np.nanmean(arr[:, cols], axis=1)


def _downstream_coordinate(case: str, ny: int) -> tuple[np.ndarray, str]:
    if GRID_DIR is not None:
        path = GRID_DIR / f"{case}_ygrid.npy"
        if path.exists():
            y = np.load(path)[:, 0].astype(float)
            if y.size == ny:
                return y.max() - y, "downstream distance s [mm] (source at s = 0)"
    rows = np.arange(ny)
    return rows.max() - rows, "rows from the source end"


def _plot_zero_vs_fulllag(case: str, maps_zero: dict, s: np.ndarray, xlabel: str) -> dict:
    """Per region: first-zero-crossing ILS (solid) vs full-lag ILS (dashed), with the fraction of pixels whose
    autocorrelation never crossed zero (r0 = NaN; both estimates are then identical) on a secondary axis."""
    saved = {}
    full, nozero = {}, {}
    for comp in maps_zero:
        f_path = ILS_DIR / f"{FILE_PREFIX}_{case}_{comp}_fulllag.npy"
        r_path = ILS_DIR / f"{FILE_PREFIX}_{case}_{comp}_r0.npy"
        if f_path.exists():
            full[comp] = np.load(f_path).astype(float) * SCALE
        if r_path.exists():
            r0 = np.load(r_path)
            nozero[comp] = np.isnan(r0) & np.isfinite(maps_zero[comp])
    nx = next(iter(maps_zero.values())).shape[1]
    for region, ranges in REGIONS.items():
        cols = _region_columns(ranges, nx)
        fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=SHARE_Y)
        vmax = 0.0
        for ax, comp in zip(axs.flat, COMPONENTS):
            if comp not in maps_zero:
                continue
            pz = _smooth(_region_profile(maps_zero[comp], cols), ROWS_TO_AVG)
            ax.plot(s, pz, color="k", lw=1.2, label="first zero crossing")
            vmax = max(vmax, np.nanmax(pz))
            if comp in full:
                pf = _smooth(_region_profile(full[comp], cols), ROWS_TO_AVG)
                ax.plot(s, pf, color="tab:red", lw=1.2, ls="--", label="full lag")
                vmax = max(vmax, np.nanmax(pf))
                saved[f"{case}|{region}|{comp}|fulllag"] = pf
            if comp in nozero:
                frac = _smooth(nozero[comp][:, cols].mean(axis=1), ROWS_TO_AVG)
                ax2 = ax.twinx()
                ax2.plot(s, frac, color="0.6", lw=0.8, ls=":")
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("fraction never crossing zero", color="0.5", fontsize=7)
                ax2.tick_params(axis="y", labelsize=7, colors="0.5")
                saved[f"{case}|{region}|{comp}|frac_no_zero"] = frac
            ax.set_title(TITLES[comp])
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7, loc="upper left")
        if SHARE_Y and vmax > 0:
            axs[0, 0].set_ylim(0, 1.1 * vmax)
        for ax in axs[1]:
            ax.set_xlabel(xlabel)
        for ax in axs[:, 0]:
            ax.set_ylabel("integral length scale [mm]")
        fig.suptitle(f"{case}: {REGION_TITLES.get(region, region)}\nfirst-zero-crossing (solid) vs full-lag (dashed) "
                     "ILS; dotted: fraction of pixels whose autocorrelation never crossed zero")
        fig.tight_layout()
        out_path = OUT_DIR / f"{FILE_PREFIX}_profiles_{region}_{case}_zero_vs_fulllag.png"
        _savefig(fig, out_path, DPI)
        plt.close(fig)
        print(f"Saved {out_path}")
    return saved


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    for case in CASES:
        maps = {}
        for comp in COMPONENTS:
            path = ILS_DIR / f"{FILE_PREFIX}_{case}_{comp}.npy"
            if not path.exists():
                print(f"Skipping {case} {comp}: {path.name} not found")
                continue
            maps[comp] = np.load(path).astype(float) * SCALE
        if not maps:
            continue
        ny, nx = next(iter(maps.values())).shape
        s, xlabel = _downstream_coordinate(case, ny)

        for region, ranges in REGIONS.items():
            cols = _region_columns(ranges, nx)
            profiles = {comp: _smooth(_region_profile(a, cols), ROWS_TO_AVG) for comp, a in maps.items()}
            for comp, prof in profiles.items():
                saved[f"{case}|{region}|{comp}"] = prof
            saved[f"{case}|s_mm"] = s

            fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=SHARE_Y)
            for ax, comp in zip(axs.flat, COMPONENTS):
                if comp in profiles:
                    ax.plot(s, profiles[comp], color="k", lw=1.2)
                else:
                    ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center")
                ax.set_title(TITLES[comp])
                ax.grid(alpha=0.3)
            for ax in axs[1]:
                ax.set_xlabel(xlabel)
            for ax in axs[:, 0]:
                ax.set_ylabel("integral length scale [mm]")
            if SHARE_Y:
                vals = np.concatenate([p[np.isfinite(p)] for p in profiles.values()])
                if vals.size:
                    axs[0, 0].set_ylim(0, 1.1 * np.nanmax(vals))
            fig.suptitle(f"{case}: {REGION_TITLES.get(region, region)}\n"
                         f"cross-stream average of {len(cols)} columns, {ROWS_TO_AVG}-row running mean")
            fig.tight_layout()
            out_path = OUT_DIR / f"{FILE_PREFIX}_profiles_{region}_{case}.png"
            _savefig(fig, out_path, DPI)
            plt.close(fig)
            print(f"Saved {out_path}")

        if COMPARE_FULLLAG:
            saved.update(_plot_zero_vs_fulllag(case, maps, s, xlabel))

    np.savez_compressed(OUT_DIR / f"{FILE_PREFIX}_region_profiles.npz", **saved,
                        regions=np.array([f"{r}: {REGIONS[r]}" for r in REGIONS]))
    print(f"Saved profile arrays to {OUT_DIR / f'{FILE_PREFIX}_region_profiles.npz'}")


if __name__ == "__main__":
    main()
