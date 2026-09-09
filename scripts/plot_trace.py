"""Plot scripts/trace_shape.py logs: commanded vs actual path in the x-y
plane, plus tracking error vs time with phase boundaries marked.

    python -m scripts.plot_trace logs/trace_shape-20260908-212513
    python -m scripts.plot_trace logs/trace_shape-... --out /tmp/panto_plots/box.png
    python -m scripts.plot_trace --compare logs/run_a logs/run_b logs/run_c

``--compare`` overlays several log dirs, one colour per run, on the x-y panel
only (the error-vs-time panel doesn't make sense across runs with different
durations/phases, so single-run mode is where that lives).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- palette (validated categorical set from the dataviz skill, light mode) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PURPLE = "#8756d1"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e2dc"
CAP_COLOR = "#e34948"

COMPARE_COLORS = [BLUE, ORANGE, AQUA, PURPLE, "#eda100"]


def load_run(log_dir: Path) -> dict:
    with open(log_dir / "meta.json") as f:
        meta = json.load(f)
    t, cmd_x, cmd_y, x, y, err_mm, phase = [], [], [], [], [], [], []
    with open(log_dir / "samples.jsonl") as f:
        for line in f:
            d = json.loads(line)
            t.append(d["t"])
            cmd_x.append(d["anchor"][0])
            cmd_y.append(d["anchor"][1])
            x.append(d["pose"][0])
            y.append(d["pose"][1])
            err_mm.append(d.get("tracking_error_mm", float("nan")))
            phase.append(d.get("phase", ""))
    arr = lambda v: np.asarray(v, dtype=float)
    return dict(t=arr(t), cmd_x=arr(cmd_x) * 1e3, cmd_y=arr(cmd_y) * 1e3,
               x=arr(x) * 1e3, y=arr(y) * 1e3, err_mm=arr(err_mm), phase=phase, meta=meta)


def summary_str(meta: dict) -> str:
    return (f"shape={meta.get('shape')}  size={meta.get('size_mm')}mm  "
            f"speed={meta.get('speed_mm_s')}mm/s  K={meta.get('stiffness')}N/m  "
            f"vel_gain={meta.get('vel_gain')}")


def _add_direction_arrows(ax, xs, ys, n_arrows=8, color=TEXT_SECONDARY):
    if len(xs) < 2:
        return
    idxs = np.linspace(0, len(xs) - 2, n_arrows, dtype=int)
    for i in idxs:
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        if dx == 0 and dy == 0:
            continue
        ax.annotate("", xy=(xs[i] + dx, ys[i] + dy), xytext=(xs[i], ys[i]),
                   arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, alpha=0.7,
                                    shrinkA=0, shrinkB=0))


def plot_single(log_dir: Path, out_path: Path) -> Path:
    d = load_run(log_dir)
    t, err_mm, phase = d["t"], d["err_mm"], d["phase"]

    fig, (ax_xy, ax_err) = plt.subplots(
        2, 1, figsize=(1400 / 150, 900 / 150), dpi=150,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.suptitle(f"{log_dir.name} — {summary_str(d['meta'])}", fontsize=11, color=TEXT_PRIMARY)

    ax_xy.plot(d["cmd_x"], d["cmd_y"], "--", color=TEXT_SECONDARY, linewidth=1.5, label="commanded")
    ax_xy.plot(d["x"], d["y"], "-", color=BLUE, linewidth=1.8, label="actual")
    _add_direction_arrows(ax_xy, d["cmd_x"], d["cmd_y"])
    ax_xy.plot(d["cmd_x"][0], d["cmd_y"][0], "o", color=ORANGE, markersize=8, zorder=5, label="start")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x (mm)", color=TEXT_SECONDARY)
    ax_xy.set_ylabel("y (mm)", color=TEXT_SECONDARY)
    ax_xy.grid(True, color=GRID, linewidth=0.7)
    ax_xy.tick_params(colors=TEXT_SECONDARY)
    ax_xy.legend(fontsize=8, frameon=False)

    ax_err.plot(t, err_mm, "-", color=AQUA, linewidth=1.3)
    # mark phase boundaries
    last = None
    for i, p in enumerate(phase):
        if p != last:
            ax_err.axvline(t[i], color=GRID, linewidth=0.8, linestyle=":")
            last = p
    ax_err.axhline(5.0, color=CAP_COLOR, linewidth=1, linestyle="--", label="5mm verdict threshold")
    ax_err.set_xlabel("time (s)", color=TEXT_SECONDARY)
    ax_err.set_ylabel("tracking error (mm)", color=TEXT_PRIMARY)
    ax_err.grid(True, color=GRID, linewidth=0.7)
    ax_err.tick_params(colors=TEXT_SECONDARY)
    ax_err.legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_compare(log_dirs: list[Path], out_path: Path) -> Path:
    fig, ax_xy = plt.subplots(figsize=(1400 / 150, 900 / 150), dpi=150)

    for i, log_dir in enumerate(log_dirs):
        d = load_run(log_dir)
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        label = f"{log_dir.name} ({summary_str(d['meta'])})"
        ax_xy.plot(d["cmd_x"], d["cmd_y"], "--", color=color, linewidth=1.0, alpha=0.5)
        ax_xy.plot(d["x"], d["y"], "-", color=color, linewidth=1.8, label=label)

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x (mm)", color=TEXT_SECONDARY)
    ax_xy.set_ylabel("y (mm)", color=TEXT_SECONDARY)
    ax_xy.grid(True, color=GRID, linewidth=0.7)
    ax_xy.tick_params(colors=TEXT_SECONDARY)
    ax_xy.legend(fontsize=8, frameon=False)
    fig.suptitle("trace_shape comparison — dashed = commanded, solid = actual",
                fontsize=11, color=TEXT_PRIMARY)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logdir", nargs="?", help="run log directory (single-run mode)")
    p.add_argument("--compare", nargs="+", metavar="LOGDIR",
                   help="overlay several log dirs' x-y paths, one colour per run")
    p.add_argument("--out", type=str, default=None, help="output PNG path")
    args = p.parse_args()

    if args.compare:
        log_dirs = [Path(d) for d in args.compare]
        out_path = Path(args.out) if args.out else Path("/tmp/panto_plots/compare_trace.png")
        out = plot_compare(log_dirs, out_path)
    else:
        if not args.logdir:
            raise SystemExit("give a logdir, or --compare LOGDIR [LOGDIR ...]")
        log_dir = Path(args.logdir)
        out_path = Path(args.out) if args.out else Path("/tmp/panto_plots") / f"{log_dir.name}.png"
        out = plot_single(log_dir, out_path)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
