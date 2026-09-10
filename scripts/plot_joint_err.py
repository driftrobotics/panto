"""Plot joint-space error (theta0, theta1 minus the commanded q_target, deg)
plus per-joint current for one or more step_response/trace_shape sample logs.

    python -m scripts.plot_joint_err logs/step_response-<a> logs/step_response-<b> --out x.png

Each run is a column: theta0 error, theta1 error, Iq0/Iq1. Titles are pulled
from the run's summary/meta (K, vel_gain, vel_limit, dir) when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(run: Path):
    f = run / "samples.jsonl" if run.is_dir() else run
    rows = [json.loads(l) for l in open(f)]
    t = np.array([r["t"] for r in rows])
    q = np.degrees(np.array([r["q"] for r in rows]))
    qt = np.degrees(np.array([r["sent"]["q_target"] for r in rows]))
    I = np.array([r["currents"] for r in rows])
    title = run.name
    for meta in ("summary.json", "meta.json"):
        p = (run if run.is_dir() else run.parent) / meta
        if p.exists():
            m = json.load(open(p))
            c = m.get("config", m)
            vg = c.get("vel_gain")
            vg = vg[0] if isinstance(vg, list) else vg
            title = (f"K{c.get('stiffness')} vg{vg} vl{c.get('vel_limit')} {c.get('dir', c.get('direction', ''))}"
                     f"  [{m.get('verdict', '')}]")
            break
    return t, q - qt, I, title


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--titles", type=str, default=None, help="'|'-separated titles overriding auto")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(args.runs)
    titles = args.titles.split("|") if args.titles else [None] * n
    fig, axs = plt.subplots(3, n, figsize=(5.5 * n, 9), sharex="col", squeeze=False)
    for col, run in enumerate(args.runs):
        t, e, I, title = load(Path(run))
        title = titles[col] or title
        a0, a1, a2 = axs[0, col], axs[1, col], axs[2, col]
        a0.plot(t, e[:, 0], lw=0.8); a0.set_title(title, fontsize=9); a0.set_ylabel("theta0 err deg"); a0.grid(alpha=.3)
        a1.plot(t, e[:, 1], lw=0.8, color="C1"); a1.set_ylabel("theta1 err deg"); a1.grid(alpha=.3)
        a2.plot(t, I[:, 0], lw=0.6, label="Iq0"); a2.plot(t, I[:, 1], lw=0.6, label="Iq1", alpha=.7)
        a2.set_ylabel("A"); a2.set_xlabel("t s"); a2.legend(fontsize=8); a2.grid(alpha=.3)
        rms0 = np.sqrt((e[:, 0] ** 2).mean()); rms1 = np.sqrt((e[:, 1] ** 2).mean())
        a0.text(0.02, 0.9, f"rms {rms0:.3f} deg", transform=a0.transAxes, fontsize=8)
        a1.text(0.02, 0.9, f"rms {rms1:.3f} deg", transform=a1.transAxes, fontsize=8)
    plt.tight_layout(); plt.savefig(args.out, dpi=100)
    print(args.out)


if __name__ == "__main__":
    main()
