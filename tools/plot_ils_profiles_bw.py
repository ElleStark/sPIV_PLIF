"""
Black-and-white single-panel versions of the L(s) and A(s) profiles from tools/ils_characteristic_scale.py.

Reads characteristic_scale.npz written by that script (run it first) and saves four figures, one line style per case:
    L_background_bw.png   L = 1/2 (L_u,x + L_v,y), background columns
    A_background_bw.png   A = L_u,x / L_v,y, background columns
    L_core_bw.png         L, core (centerline +/- 100 columns)
    A_core_bw.png         A, core

Run from the repo root:
    python -m tools.plot_ils_profiles_bw
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
SCALE_DIR = Path("E:/sPIV_PLIF_ProcessedData/flow_properties/Integral_length_scales/ILS/Characteristic_scale")
NPZ_PATH = SCALE_DIR / "characteristic_scale.npz"
OUT_DIR = SCALE_DIR / "bw"

CASES: list[str] = ["baseline", "buoyant", "diffusive", "fractal", "nearbed", "smSource"]
LINESTYLES = {
    "baseline": "-",
    "buoyant": (0, (3, 2)),                # short dash
    "diffusive": (0, (1, 1.5)),            # dotted
    "fractal": (0, (9, 2.5, 1.5, 2.5)),    # dash-dot
    "nearbed": (0, (4, 1.5, 1, 1.5, 1, 1.5, 1, 1.5)),  # dash-dot-dot-dot
    "smSource": (0, (12, 4)),              # long dash
}
LINEWIDTH = 1.1
COLOR = "black"

# Shade (thick light-grey underlay) where on average > NEVER_CROSS_FLAG of pixels never crossed zero:
# L there is a field-of-view-limited lower bound. Same rule as ils_characteristic_scale.py.
SHOW_LOWER_BOUND = False
NEVER_CROSS_FLAG = 0.5
S_LIMITS_MM: tuple[float, float] | None = (0.0, 295.0)  # data outside are dropped (last rows are an edge artifact)

# Reference power law L = C (s + x0)^n on the background L plot only, drawn beneath the data. C and x0 are least-squares
# fits to the background L of SCALING_FIT_CASES over S_LIMITS_MM (s is measured from the source, so x0 is a virtual
# origin standing in for the unknown grid-to-source distance). Set SCALING_EXPONENT = None to omit.
SCALING_EXPONENT: float | str | None = 2 / 5  # a number (e.g. 2/5), "fit" to fit n as well, or None
SCALING_X0_MM: float | None = 150.0  # grid-to-domain distance (grid is 15 cm upstream of s = 0); None = fit x0 too
SCALING_FIT_S_MM: tuple[float, float] | None = None  # s range used for the fit; None = S_LIMITS_MM
SCALING_FIT_CASES = ["baseline", "buoyant", "diffusive", "smSource"]  # fractal and nearbed have different backgrounds
SCALING_COLOR = "red"
SCALING_LABEL = r"$\propto (s + {x0})^{{{n}}}$"  # {n}, {x0} replaced by the exponent and origin [mm]

FIGSIZE = (5.5, 4.5)
DPI = 300
FONT_FAMILY = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
BASE_FONTSIZE = 12
LEGEND_FONTSIZE = 10
XLABEL = "downstream distance $s$ (mm)"
L_YLABEL = r"$L = \frac{1}{2}(L_{u,x} + L_{v,y})$ (mm)"
A_YLABEL = r"$A = L_{u,x} / L_{v,y}$"

plt.rcParams.update({"font.family": "serif", "font.serif": FONT_FAMILY, "mathtext.fontset": "stix",
                     "font.size": BASE_FONTSIZE, "axes.labelsize": BASE_FONTSIZE,
                     "xtick.labelsize": BASE_FONTSIZE, "ytick.labelsize": BASE_FONTSIZE})


def fit_power_law(data, region: str, n: float | str) -> tuple[float, float, float]:
    """Least-squares C, x0 (and n if n == "fit") for L = C (s + x0)^n over the fit cases."""
    S, L = [], []
    for case in SCALING_FIT_CASES:
        s, p = data[f"{case}|s_mm"], data[f"{case}|{region}|L"]
        rng = SCALING_FIT_S_MM or S_LIMITS_MM
        keep = np.isfinite(p) if rng is None else np.isfinite(p) & (s >= rng[0]) & (s <= rng[1])
        S.append(s[keep])
        L.append(p[keep])
    S, L = np.concatenate(S), np.concatenate(L)
    if SCALING_X0_MM is not None:  # fixed origin: fit C (and n)
        x0 = SCALING_X0_MM
        if n == "fit":
            r = least_squares(lambda q: q[0] * (S + x0) ** q[1] - L, [0.1, 1.0])
            C, nf = r.x
        else:
            b = (S + x0) ** n
            C, nf = float(np.dot(L, b) / np.dot(b, b)), n
        rms = float(np.sqrt(np.mean((C * (S + x0) ** nf - L) ** 2)))
        print(f"{region} L fit: L = {C:.4g} (s + {x0:.1f})^{nf:.3f} mm, rms residual {rms:.2f} mm")
        return C, x0, nf
    if n == "fit":
        best = None
        for n0 in (0.4, 0.8, 1.0, 1.5, 2.0, 3.0):  # multi-start: C, x0 and n trade off strongly
            for x00 in (10.0, 50.0, 200.0, 1000.0):
                C0 = np.mean(L) / np.mean((S + x00) ** n0)
                r = least_squares(lambda q: q[0] * (S + q[1]) ** q[2] - L, [C0, x00, n0],
                                  bounds=([0.0, 0.0, 0.01], [np.inf, 1e4, 10.0]))
                rms = float(np.sqrt(np.mean(r.fun ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, *r.x)
        rms, C, x0, nf = best
        print(f"{region} L fit: L = {C:.3g} (s + {x0:.1f})^{nf:.3f} mm, rms residual {rms:.2f} mm")
        return C, x0, nf
    best = (np.inf, 0.0, 0.0)
    for x0 in np.arange(0.0, 2000.0, 0.5):
        b = (S + x0) ** n
        C = float(np.dot(L, b) / np.dot(b, b))
        rms = float(np.sqrt(np.mean((C * b - L) ** 2)))
        if rms < best[0]:
            best = (rms, C, x0)
    print(f"{region} L fit: L = {best[1]:.3f} (s + {best[2]:.1f})^{n:.3f} mm, rms residual {best[0]:.2f} mm")
    return best[1], best[2], n


def plot_profile(data, region: str, quantity: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if quantity == "L" and region == "bg" and SCALING_EXPONENT is not None:
        C, x0, n = fit_power_law(data, region, SCALING_EXPONENT)
        s_ref = np.linspace(*(S_LIMITS_MM or (0.0, 300.0)), 300)
        n_txt = "2/5" if abs(n - 0.4) < 1e-9 else f"{n:.2f}"
        ax.plot(s_ref, C * (s_ref + x0) ** n, color=SCALING_COLOR, ls="--", lw=LINEWIDTH,
                label=SCALING_LABEL.format(n=n_txt, x0=f"{x0:.0f}"), zorder=0)
    for case in CASES:
        s = data[f"{case}|s_mm"]
        p = data[f"{case}|{region}|{quantity}"]
        keep = np.ones(s.shape, bool) if S_LIMITS_MM is None else (s >= S_LIMITS_MM[0]) & (s <= S_LIMITS_MM[1])
        s, p = s[keep], p[keep]
        if SHOW_LOWER_BOUND and quantity == "L":
            flag = (0.5 * (data[f"{case}|{region}|nz_ux"] + data[f"{case}|{region}|nz_vy"]) > NEVER_CROSS_FLAG)[keep]
            ax.plot(s, np.where(flag, p, np.nan), color="0.8", lw=5, solid_capstyle="butt", zorder=1)
        ax.plot(s, p, color=COLOR, ls=LINESTYLES[case], lw=LINEWIDTH, label=case, zorder=2)

    if quantity == "L":
        ax.set_ylabel(L_YLABEL)
        ax.set_ylim(bottom=0)
    else:
        ax.axhline(1.0, color="0.6", lw=0.7, zorder=0)
        ax.set_ylabel(A_YLABEL)
    ax.set_xlabel(XLABEL)
    if S_LIMITS_MM is not None:
        ax.set_xlim(*S_LIMITS_MM)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=3, frameon=False, handlelength=4.0, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))  # above the axes, clear of the curves
    fig.tight_layout()
    _savefig(fig, out_path)


def _savefig(fig, out_path: Path, retries: int = 5) -> None:
    """Write to a temp file, then replace (overwriting a PNG on E: intermittently fails with Errno 22)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=DPI)
    plt.close(fig)
    for attempt in range(retries):
        try:
            os.replace(tmp, out_path)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)
    print(f"Saved {out_path}")


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"{NPZ_PATH} not found: run tools/ils_characteristic_scale.py first")
    data = np.load(NPZ_PATH)
    for region, tag in (("bg", "background"), ("core", "core")):
        for quantity in ("L", "A"):
            filename = f"{quantity}_{tag}_bw_2over5.png" if quantity == "L" and region == "bg" else f"{quantity}_{tag}_bw.png"
            plot_profile(data, region, quantity, OUT_DIR / filename)


if __name__ == "__main__":
    main()
