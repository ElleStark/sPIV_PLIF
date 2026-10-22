"""
Plot laser pulse energy over time from StarLab energy-meter logs, split by laser head.

The two laser heads fire alternately, so each log is split into two series that take every other pulse
(pulses 0, 2, 4, ... and 1, 3, 5, ...). Both heads from the same day share one color; within each log the
head with the higher mean energy is drawn solid and the lower one dotted.

Edit the settings below, then run from the repo root:
    python -m tools.plot_laser_energy
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\elles\OneDrive - UCB-O365\Fluids_Research\sPIV_PLIF_project\Energy_testing")
GLOB_PATTERN = "*.txt"  # files in not_used/ are ignored (not recursive)
EXCLUDE_DATES = ["2025-09-29"]  # days to leave out (YYYY-MM-DD)

OUT_PATH = Path("E:/sPIV_PLIF_ProcessedData/Plots/laser_energy_alternating_heads.png")
PANEL_OUT_PATH = Path("E:/sPIV_PLIF_ProcessedData/Plots/laser_energy_alternating_heads_panels.png")
MAKE_OVERLAY = True  # all logs on one axis
MAKE_PANELS = True  # one panel per day
TITLE = "Laser energy by head (alternating pulses)"
XLABEL = "time (s)"
YLABEL = "Energy (mJ)"

Y_SCALE = 1000.0  # J -> mJ
Y_LIMITS: tuple[float, float] | None = None  # e.g. (80.0, 130.0)

LATER_LOG_LIGHTEN = 0.5  # 2nd+ log of a day is blended this far toward white
DAY_COLORS = ["#082a54", "#e02b35", "#f0c571", "#59a89c", "#a559aa", "#cecece"]
LINEWIDTH = 0.8
ALPHA = 0.85
FIGSIZE = (10.0, 4.5)
PANEL_WIDTH = 10.0
PANEL_HEIGHT = 2.6  # per day
DPI = 300
FONT_FAMILY = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]  # first available is used
BASE_FONTSIZE = 13  # axis labels and tick labels
LEGEND_FONTSIZE = 13  # panel figure; the overlay uses LEGEND_FONTSIZE - 2
SAME_DAY_TAGS = ["pre-sample", "post-sample"]  # appended to the labels when a day has several logs, in time order

plt.rcParams.update({"font.family": "serif", "font.serif": FONT_FAMILY, "mathtext.fontset": "stix",
                     "font.size": BASE_FONTSIZE, "axes.labelsize": BASE_FONTSIZE, "xtick.labelsize": BASE_FONTSIZE,
                     "ytick.labelsize": BASE_FONTSIZE, "figure.titlesize": BASE_FONTSIZE + 2,
                     "axes.titlesize": BASE_FONTSIZE + 1})

# Filenames end in e.g. "_1030AM_9.29.2025.txt" or "_1730PM_9.10.2025.txt"
NAME_RE = re.compile(r"_(\d{1,4})(AM|PM)_(\d{1,2})\.(\d{1,2})\.(\d{4})$")


def parse_start(path: Path) -> datetime:
    """Log start time from the filename (time of day and m.d.yyyy date)."""
    m = NAME_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Cannot parse date/time from filename: {path.name}")
    hhmm, ampm, month, day, year = m.groups()
    hour, minute = (int(hhmm[:-2]), int(hhmm[-2:])) if len(hhmm) > 2 else (int(hhmm), 0)
    if ampm == "PM" and hour < 12:
        hour += 12
    return datetime(int(year), int(month), int(day), hour, minute)


def read_starlab_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_s, energy_J) from a StarLab log; data follow the 'Timestamp' header line."""
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Timestamp")) + 1
    data = np.loadtxt(lines[start:], usecols=(0, 1))
    return data[:, 0], data[:, 1]


def load_heads(paths: list[Path]) -> list[dict]:
    """One record per (log, head): time, energy [mJ], stats, and whether it is the higher-mean head of its log."""
    records = []
    for path in paths:
        start = parse_start(path)
        t, e = read_starlab_log(path)
        e = e * Y_SCALE
        heads = [(t[0::2], e[0::2]), (t[1::2], e[1::2])]
        means = [np.mean(h[1]) for h in heads]
        for rank, idx in enumerate(np.argsort(means)[::-1]):  # higher mean first -> solid
            th, eh = heads[idx]
            records.append(dict(path=path, start=start, t=th, e=eh, mean=float(np.mean(eh)),
                                std=float(np.std(eh, ddof=1)), higher=rank == 0))
    return records


def _lighten(color: str, frac: float) -> tuple[float, float, float]:
    return tuple(c + (1.0 - c) * frac for c in to_rgb(color))


def assign_colors(records: list[dict], all_days: list) -> None:
    """Same base color per day (indexed over all days, so excluding a day keeps the others' colors);
    each later log on the same day is progressively lighter."""
    days = sorted({r["start"].date() for r in records})
    base = {d: DAY_COLORS[i % len(DAY_COLORS)] for i, d in enumerate(all_days)}
    for day in days:
        starts = sorted({r["start"] for r in records if r["start"].date() == day})
        for r in records:
            if r["start"].date() == day:
                n = starts.index(r["start"])
                r["color"] = _lighten(base[day], 1.0 - (1.0 - LATER_LOG_LIGHTEN) ** n)
                r["tag"] = (SAME_DAY_TAGS[n] if n < len(SAME_DAY_TAGS) else f"log {n + 1}") if len(starts) > 1 else ""


def _style(rec: dict) -> str:
    return "-" if rec["higher"] else ":"


def _label(rec: dict, with_date: bool = True) -> str:
    which = "head 1" if rec["higher"] else "head 2"  # head 1 = higher-mean (solid) series of each log
    tag = f" ({rec['tag']})" if rec["tag"] else ""
    when = f"{rec['start']:%m-%d-%Y}, " if with_date else ""
    return f"{when}{which}{tag}: {rec['mean']:.1f} ± {rec['std']:.1f} mJ"


def plot_overlay(records: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for rec in records:
        ax.plot(rec["t"], rec["e"], _style(rec), color=rec["color"], lw=LINEWIDTH, alpha=ALPHA,
                label=_label(rec))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE)
    if Y_LIMITS is not None:
        ax.set_ylim(*Y_LIMITS)
    ax.legend(title="mean ± std", fontsize=LEGEND_FONTSIZE - 2, title_fontsize=LEGEND_FONTSIZE - 2, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    _save(fig, OUT_PATH)


def plot_panels(records: list[dict]) -> None:
    days = sorted({r["start"].date() for r in records})
    fig, axes = plt.subplots(len(days), 1, figsize=(PANEL_WIDTH, PANEL_HEIGHT * len(days)), sharex=True, sharey=True,
                             squeeze=False)
    for ax, day in zip(axes[:, 0], days):
        day_recs = [r for r in records if r["start"].date() == day]
        for rec in day_recs:
            ax.plot(rec["t"], rec["e"], _style(rec), color=rec["color"], lw=LINEWIDTH, alpha=ALPHA,
                    label=_label(rec, with_date=False))
        ax.set_title(f"{day:%m-%d-%Y}", fontsize=LEGEND_FONTSIZE + 1, loc="left")
        ax.set_ylabel(YLABEL)
        ax.legend(title="mean ± std", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False)
    if Y_LIMITS is not None:
        axes[0, 0].set_ylim(*Y_LIMITS)
    axes[-1, 0].set_xlabel(XLABEL)
    fig.suptitle(TITLE)
    _save(fig, PANEL_OUT_PATH)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    all_paths = sorted(DATA_DIR.glob(GLOB_PATTERN), key=parse_start)
    all_days = sorted({parse_start(p).date() for p in all_paths})
    paths = [p for p in all_paths if f"{parse_start(p):%Y-%m-%d}" not in EXCLUDE_DATES]
    if not paths:
        raise FileNotFoundError(f"No files matched {GLOB_PATTERN} in {DATA_DIR}")

    records = load_heads(paths)
    assign_colors(records, all_days)
    for r in records:
        print(f"{r['path'].name}: {'higher' if r['higher'] else 'lower '} head {r['mean']:.2f} +/- {r['std']:.2f} mJ "
              f"(n={len(r['e'])})")

    if MAKE_OVERLAY:
        plot_overlay(records)
    if MAKE_PANELS:
        plot_panels(records)


if __name__ == "__main__":
    main()
