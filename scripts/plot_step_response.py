#!/usr/bin/env python3
"""Plot step-response logs for the panto 2-DOF arm.

Reads samples.jsonl + meta.json for each run (scp'd to /tmp/panto_plots/<run>/)
and produces per-run joint figures plus a cross-run shoulder comparison figure.

To add a run once its log has been copied to /tmp/panto_plots/<run>/, add one
entry to the RUNS list below.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = "/tmp/panto_plots"

# label -> run directory name (must exist under PLOTS_DIR with samples.jsonl + meta.json)
RUNS = [
    ("A", "step_response-20260904-215823"),
    ("B", "step_response-20260904-220221"),
    ("D", "step_response-20260904-220514"),
    ("E", "step_response-20260904-235029"),
    # ("F", "step_response-YYYYMMDD-HHMMSS"),  # add run F here when available
]

# --- palette (validated categorical set from the dataviz skill, light mode) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e2dc"
CAP_COLOR = "#e34948"  # red, status/limit line

RUN_COLORS = {"A": BLUE, "B": ORANGE, "D": AQUA, "E": "#eda100"}

CURRENT_CAP_MAX = 2.0
CURRENT_CAP_FLOOR_SHOULDER = 0.9
CURRENT_CAP_FLOOR_ELBOW = 0.5

WINDOW_PRE_S = 0.3
WINDOW_POST_S = 3.0


def load_run(run_dir):
    path = os.path.join(PLOTS_DIR, run_dir)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    t, q0, q1, qt0, qt1, i0, i1, phase = [], [], [], [], [], [], [], []
    with open(os.path.join(path, "samples.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            t.append(d["t"])
            q0.append(d["q"][0])
            q1.append(d["q"][1])
            qt0.append(d["sent"]["q_target"][0])
            qt1.append(d["sent"]["q_target"][1])
            i0.append(d["currents"][0])
            i1.append(d["currents"][1])
            phase.append(d["phase"])
    arr = lambda x: np.asarray(x, dtype=float)
    data = dict(
        t=arr(t), q0=arr(q0), q1=arr(q1), qt0=arr(qt0), qt1=arr(qt1),
        i0=arr(i0), i1=arr(i1), phase=phase, meta=meta,
    )
    # step instant = first transition away from the initial phase
    step_t = None
    for i in range(1, len(phase)):
        if phase[i] != phase[0]:
            step_t = t[i]
            break
    data["step_t"] = step_t if step_t is not None else 0.0
    return data


def config_str(meta):
    return (f"K={meta['stiffness']:.0f}  vel_gain={meta['vel_gain']}  "
            f"ff_scale={meta['ff_scale']}  cap={meta['current']}A "
            f"(floor {meta['cap_min']}A)")


def plot_run(label, run_dir):
    d = load_run(run_dir)
    t, step_t = d["t"], d["step_t"]
    x0, x1 = step_t - WINDOW_PRE_S, step_t + WINDOW_POST_S
    mask = (t >= x0) & (t <= x1)
    tt = t[mask] - step_t

    fig, axes = plt.subplots(2, 2, figsize=(1400 / 150, 900 / 150), dpi=150)
    fig.suptitle(f"Run {label} ({run_dir}) — {config_str(d['meta'])}", fontsize=11, color=TEXT_PRIMARY)

    joint_specs = [
        (0, "Shoulder", axes[0, 0], axes[0, 1], CURRENT_CAP_FLOOR_SHOULDER),
        (1, "Elbow", axes[1, 0], axes[1, 1], CURRENT_CAP_FLOOR_ELBOW),
    ]

    for idx, name, ax_pos, ax_cur, cap_floor in joint_specs:
        q = np.degrees(d[f"q{idx}"][mask])
        qt = np.degrees(d[f"qt{idx}"][mask])
        err = q - qt
        cur = d[f"i{idx}"][mask]

        ax_pos.plot(tt, qt, "--", color=TEXT_SECONDARY, linewidth=1.5, label="commanded")
        ax_pos.plot(tt, q, "-", color=BLUE, linewidth=2, label="measured")
        ax_pos.axvline(0, color=CAP_COLOR, linewidth=1, linestyle=":")
        ax_pos.set_ylabel(f"{name} angle (deg)", color=TEXT_PRIMARY)
        ax_pos.set_xlabel("time since step (s)", color=TEXT_SECONDARY)
        ax_pos.grid(True, color=GRID, linewidth=0.7)
        ax_pos.tick_params(colors=TEXT_SECONDARY)
        ax_pos.legend(fontsize=8, frameon=False)

        err_ax = ax_pos.twinx()
        err_ax.plot(tt, err, "-", color=ORANGE, linewidth=1, alpha=0.6, label="error")
        err_ax.set_ylabel("error (deg)", color=ORANGE)
        err_ax.tick_params(axis="y", colors=ORANGE)

        ax_cur.plot(tt, cur, "-", color=AQUA, linewidth=1.5)
        ax_cur.axhline(CURRENT_CAP_MAX, color=CAP_COLOR, linewidth=1, linestyle="--")
        ax_cur.axhline(-CURRENT_CAP_MAX, color=CAP_COLOR, linewidth=1, linestyle="--")
        ax_cur.axhline(cap_floor, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
        ax_cur.axhline(-cap_floor, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
        ax_cur.axvline(0, color=CAP_COLOR, linewidth=1, linestyle=":")
        ax_cur.set_ylabel(f"{name} current (A)", color=TEXT_PRIMARY)
        ax_cur.set_xlabel("time since step (s)", color=TEXT_SECONDARY)
        ax_cur.grid(True, color=GRID, linewidth=0.7)
        ax_cur.tick_params(colors=TEXT_SECONDARY)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, f"{run_dir}_joints.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path, d


def run_label(label, d):
    return f"{label} (vg={d['meta']['vel_gain']}, ff={d['meta']['ff_scale']})"


def plot_comparison(run_data, joint_idx, joint_name):
    fig, (ax_err, ax_cur) = plt.subplots(2, 1, figsize=(1400 / 150, 900 / 150), dpi=150, sharex=True)
    fig.suptitle(f"{joint_name.capitalize()} step response comparison", fontsize=12, color=TEXT_PRIMARY)

    for label, d in run_data:
        t, step_t = d["t"], d["step_t"]
        x0, x1 = step_t - WINDOW_PRE_S, step_t + WINDOW_POST_S
        mask = (t >= x0) & (t <= x1)
        tt = t[mask] - step_t
        q = np.degrees(d[f"q{joint_idx}"][mask])
        qt = np.degrees(d[f"qt{joint_idx}"][mask])
        err = q - qt
        cur = d[f"i{joint_idx}"][mask]
        color = RUN_COLORS.get(label, TEXT_PRIMARY)
        lbl = run_label(label, d)
        ax_err.plot(tt, err, "-", color=color, linewidth=1.8, label=lbl)
        ax_cur.plot(tt, cur, "-", color=color, linewidth=1.5, label=lbl)

    ax_err.axvline(0, color=CAP_COLOR, linewidth=1, linestyle=":")
    ax_err.axhline(0, color=GRID, linewidth=1)
    ax_err.set_ylabel(f"{joint_name} error (deg)", color=TEXT_PRIMARY)
    ax_err.grid(True, color=GRID, linewidth=0.7)
    ax_err.tick_params(colors=TEXT_SECONDARY)
    ax_err.legend(fontsize=9, frameon=False)

    ax_cur.axvline(0, color=CAP_COLOR, linewidth=1, linestyle=":")
    ax_cur.axhline(CURRENT_CAP_MAX, color=CAP_COLOR, linewidth=1, linestyle="--")
    ax_cur.axhline(-CURRENT_CAP_MAX, color=CAP_COLOR, linewidth=1, linestyle="--")
    ax_cur.set_ylabel(f"{joint_name} current (A)", color=TEXT_PRIMARY)
    ax_cur.set_xlabel("time since step (s)", color=TEXT_SECONDARY)
    ax_cur.grid(True, color=GRID, linewidth=0.7)
    ax_cur.tick_params(colors=TEXT_SECONDARY)
    ax_cur.legend(fontsize=9, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, f"compare_{joint_name}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def compute_metrics(label, d):
    rows = []
    for idx, name in [(0, "shoulder"), (1, "elbow")]:
        t, step_t = d["t"], d["step_t"]
        post_mask = (t >= step_t) & (t <= step_t + WINDOW_POST_S)
        q = np.degrees(d[f"q{idx}"][post_mask])
        qt = np.degrees(d[f"qt{idx}"][post_mask])
        tt = t[post_mask] - step_t
        cur = d[f"i{idx}"][post_mask]
        target_final = qt[-1] if len(qt) else float("nan")
        target_initial = np.degrees(d[f"q{idx}"][t < step_t][-1]) if np.any(t < step_t) else q[0]
        step_size = target_final - target_initial
        err = q - target_final

        if step_size != 0 and len(q):
            if step_size > 0:
                overshoot = max(0.0, (q.max() - target_final))
            else:
                overshoot = max(0.0, (target_final - q.min()))
        else:
            overshoot = 0.0

        settle_idx = None
        abs_err = np.abs(err)
        for i in range(len(abs_err)):
            if np.all(abs_err[i:] <= 0.5):
                settle_idx = i
                break
        t_settle = tt[settle_idx] if settle_idx is not None else float("nan")

        rms_current = float(np.sqrt(np.mean(cur ** 2))) if len(cur) else float("nan")

        rows.append((label, name, overshoot, t_settle, rms_current))
    return rows


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    run_data = []
    png_paths = []
    for label, run_dir in RUNS:
        path = os.path.join(PLOTS_DIR, run_dir)
        if not os.path.exists(os.path.join(path, "samples.jsonl")):
            print(f"skipping {label} ({run_dir}): no samples.jsonl found")
            continue
        out_path, d = plot_run(label, run_dir)
        png_paths.append(out_path)
        run_data.append((label, d))

    png_paths.append(plot_comparison(run_data, 0, "shoulder"))
    png_paths.append(plot_comparison(run_data, 1, "elbow"))

    print("\nGenerated plots:")
    for p in png_paths:
        print(" ", p)

    print("\nMetrics (post-step window = 3.0 s):")
    header = f"{'run':4s} {'joint':9s} {'overshoot(deg)':>15s} {'t_to_0.5deg(s)':>15s} {'rms_current(A)':>15s}"
    print(header)
    print("-" * len(header))
    for label, d in run_data:
        for row in compute_metrics(label, d):
            _, name, overshoot, t_settle, rms_cur = row
            print(f"{label:4s} {name:9s} {overshoot:15.3f} {t_settle:15.3f} {rms_cur:15.3f}")


if __name__ == "__main__":
    main()
