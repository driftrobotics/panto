"""Fit the harness holding-torque model used by PositionBackend._hold_ff from a
run's samples.jsonl (quasi-static samples only, |qd| < --qd-max on both joints):

    I_hold_j = const_a + per_deg[0]*q0_deg + per_deg[1]*q1_deg    (joint-frame A)

    python -m scripts.fit_hold_ff logs/trace_shape-<stamp>/samples.jsonl [--write calibration.json]

Prints the coefficients + residual RMS per joint; --write patches them into the
given calibration.json's motors[i] (hold_ff_const_a, hold_ff_per_deg) leaving
hold_ff_scale untouched (0 = off until you opt in via --hold-ff on a script).
Logged currents are motor-frame Iq; flips from the calibration are applied so the
fit is joint-frame like Torque_FF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit(samples: Path, flips: list[bool], qd_max: float) -> list[dict]:
    rows = [json.loads(l) for l in open(samples)]
    q = np.degrees(np.array([r["q"] for r in rows]))
    I = np.array([r["currents"] for r in rows], float)
    qd = np.array([r["qd"] for r in rows])
    for j, f in enumerate(flips):
        if f:
            I[:, j] *= -1.0
    m = (np.abs(qd[:, 0]) < qd_max) & (np.abs(qd[:, 1]) < qd_max)
    A = np.column_stack([q[m, 0], q[m, 1], np.ones(int(m.sum()))])
    out = []
    for j in range(I.shape[1]):
        coef, *_ = np.linalg.lstsq(A, I[m, j], rcond=None)
        r = I[m, j] - A @ coef
        out.append({"hold_ff_const_a": round(float(coef[2]), 4),
                    "hold_ff_per_deg": [round(float(coef[0]), 5), round(float(coef[1]), 5)],
                    "resid_rms_a": round(float(r.std()), 3), "n": int(m.sum())})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples")
    p.add_argument("--calibration", default="calibration.json", help="for motor flips")
    p.add_argument("--qd-max", type=float, default=0.3, help="rad/s, quasi-static threshold")
    p.add_argument("--write", type=str, default=None, help="patch coefficients into this calibration.json")
    args = p.parse_args()
    cal = json.load(open(args.calibration))
    flips = [bool(m.get("flip", False)) for m in cal["motors"]]
    res = fit(Path(args.samples), flips, args.qd_max)
    for j, r in enumerate(res):
        print(f"motor{j}: {r}")
    if args.write:
        cal_w = json.load(open(args.write))
        for j, r in enumerate(res):
            cal_w["motors"][j]["hold_ff_const_a"] = r["hold_ff_const_a"]
            cal_w["motors"][j]["hold_ff_per_deg"] = r["hold_ff_per_deg"]
        Path(args.write).write_text(json.dumps(cal_w, indent=2) + "\n")
        print(f"wrote hold_ff_* into {args.write}")


if __name__ == "__main__":
    main()
