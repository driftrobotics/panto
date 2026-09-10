"""Generate a chronological experiment log (markdown table) from the run logs.

    python -m scripts.experiment_log --logs logs --since 2026-09-08 --out experiments/runs.md

Walks logs/<script>-<stamp>/, reads meta.json + summary.json, and prints one
row per run: stamp, script, preset/K/vel_gain/vel_limit/cap/dir/step, verdict,
the key metrics, I2t and the notes field if meta.json has one. The narrative
("what we tried / learned") lives in experiments/log.md; this table is the
run-name index it refers to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _g(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, list):
        return "/".join(_fmt(x, nd) for x in v)
    return str(v)


def row(run: Path) -> str | None:
    meta = json.load(open(run / "meta.json")) if (run / "meta.json").exists() else {}
    summ = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else {}
    cfg = summ.get("config", {})
    name = run.name
    script, _, stamp = name.partition("-")
    K = _g(cfg, "stiffness", default=_g(meta, "stiffness"))
    vg = _g(cfg, "vel_gain", default=_g(meta, "vel_gain"))
    vl = _g(cfg, "vel_limit", default=_g(meta, "vel_limit"))
    cap = _g(cfg, "current", default=_g(meta, "current"))
    d = _g(cfg, "dir", "direction", "shape", default=_g(meta, "direction", "shape", "mode", default=""))
    step = _g(cfg, "step_mm", "size_mm", default=_g(meta, "step_mm", "size_mm", default=""))
    extra = []
    if _g(cfg, "anchor_ramp_s"):
        extra.append(f"ramp{cfg['anchor_ramp_s']}s")
    if _g(cfg, "speed_mm_s"):
        extra.append(f"{cfg['speed_mm_s']}mm/s")
    if _g(cfg, "hold_ff", default=_g(meta, "hold_ff")) is not None:
        extra.append(f"holdff{_g(cfg, 'hold_ff', default=_g(meta, 'hold_ff'))}")
    if meta.get("label"):
        extra.append(str(meta["label"]))
    verdict = summ.get("verdict", "")
    metrics = []
    if "overshoot_mm" in summ:
        metrics.append(f"os {_fmt(summ['overshoot_mm'])} ss {_fmt(summ.get('steady_state_error_mm'))} "
                       f"settle {_fmt(summ.get('settling_time_s'))} osc {_fmt(summ.get('osc_freq_hz'), 1)}Hz")
    if "rms_error_mm" in summ:
        metrics.append(f"rms {_fmt(summ['rms_error_mm'])} max {_fmt(summ.get('max_error_mm'))}")
    i2t = summ.get("i2t_a2s")
    return (f"| {stamp} | {script} | {_g(meta, 'preset', default=_g(summ, 'preset', default=''))} | "
            f"{_fmt(K, 0)} | {_fmt(vg, 3)} | {_fmt(vl, 1)} | {_fmt(cap, 1)} | {d} {step} {' '.join(extra)} | "
            f"{verdict} | {'; '.join(metrics)} | {_fmt(i2t, 2)} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default="logs")
    p.add_argument("--since", default="", help="YYYY-MM-DD; only runs on/after this date")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    since = args.since.replace("-", "")
    runs = sorted(d for d in Path(args.logs).iterdir() if d.is_dir() and "-" in d.name)
    lines = ["| stamp (UTC) | script | preset | K | vel_gain | vel_limit | cap A | what | verdict | metrics | I2t A2s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        stamp = r.name.split("-", 1)[1][:8]
        if since and stamp < since:
            continue
        line = row(r)
        if line:
            lines.append(line)
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"{len(lines) - 2} runs -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
