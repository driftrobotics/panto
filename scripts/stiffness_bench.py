"""Stiffness ladder benchmark: run scripts.step_response over a K grid at a
fixed current cap, with the mandatory per-run cooldown, and write one report.

    python -m scripts.stiffness_bench --current 2.0 \\
        --stiffness 50,100,200,400,800 --dirs +x,+y --step-mm 5 --cooldown-s 60

Each grid point is a separate `step_response` subprocess (fresh CAN handle,
gains restored in its own `finally`), so a crash in one point cannot leave
gains applied for the next. Points run current-major, K ascending; once a K
trips (`limit_cycle`/`stall`/abort) every higher K at that cap+direction is
skipped -- never repeat or escalate past a tripped config. Between points
`reset_pose` recentres the arm if the previous run drifted (or always with
`--reset-every`). Everything else (preset, vel_gain, vel_limit, cap_slope,
ff_scale) comes from `--preset` (default pos-bw300-K100) so only K and the cap
vary. Output: logs/stiffness_bench-<stamp>/{report.md,results.json}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from panto.bench_logic import (
    BenchResult, best_converged, build_grid, parse_summary_line, report_table, should_skip,
)

DEFAULT_PRESET = "pos-bw300-K100"


def _run(cmd: list[str]) -> tuple[int, str]:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write("    " + line)
        out.append(line)
    proc.wait()
    return proc.returncode, "".join(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="stiffness_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=DEFAULT_PRESET)
    p.add_argument("--current", type=str, default="2.0",
                   help="comma list of current caps, A (each gets its own K ladder)")
    p.add_argument("--stiffness", type=str, default="50,100,200,400,800",
                   help="comma list of EE stiffness K, N/m")
    p.add_argument("--dirs", type=str, default="+x,+y")
    p.add_argument("--step-mm", type=float, default=5.0)
    p.add_argument("--cooldown-s", type=float, default=60.0,
                   help="idle after every run, enforced inside step_response")
    p.add_argument("--reset-every", action="store_true",
                   help="recentre with reset_pose before every point, not only after a trip")
    p.add_argument("--reset-preset", default="move-2A-sched")
    p.add_argument("--pose", type=str, default=None,
                   help="recentre target 'x,y' mm for every point (default: config.test_pose). "
                        "Shoulder friction is pose-dependent, so a bench is only comparable at one pose")
    p.add_argument("--reset-tol-mm", type=float, default=10.0,
                   help="reset_pose convergence tolerance; step_response accepts starts within 60 mm")
    p.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    p.add_argument("--logs", default="logs")
    args = p.parse_args()

    currents = [float(x) for x in args.current.split(",")]
    stiffs = [float(x) for x in args.stiffness.split(",")]
    dirs = [d.strip() for d in args.dirs.split(",")]
    grid = build_grid(stiffs, currents, dirs, args.step_mm)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logs) / f"stiffness_bench-{stamp}"
    print(f"grid: {len(grid)} points, preset={args.preset}, cooldown={args.cooldown_s:.0f}s")
    for g in grid:
        print(f"  cap {g.current:.1f}A  K={g.stiffness:g}  {g.direction}  {g.step_mm:g}mm")
    if args.dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    results: list[BenchResult] = []
    skipped: list[tuple] = []
    need_reset = args.reset_every
    t0 = time.time()

    def write_report() -> None:
        best = best_converged(results)
        md = [f"# stiffness bench {stamp}", "",
              f"preset `{args.preset}`, pose {args.pose or 'test_pose'}, step {args.step_mm:g} mm, "
              f"cooldown {args.cooldown_s:.0f} s, "
              f"elapsed {time.time() - t0:.0f} s", "",
              report_table(results, skipped), ""]
        if best:
            md.append(f"**Highest K converged in all directions:** K={best.point.stiffness:g} N/m "
                      f"at {best.point.current:.1f} A")
        else:
            md.append("**No K converged in all directions.**")
        (out_dir / "report.md").write_text("\n".join(md) + "\n")
        (out_dir / "results.json").write_text(json.dumps(
            [{"point": r.point.__dict__, "summary": r.summary, "log_dir": r.log_dir,
              "returncode": r.returncode} for r in results]
            + [{"point": g.__dict__, "skipped": why} for g, why in skipped], indent=2))

    for g in grid:
        why = should_skip(g, results)
        if why:
            print(f"\nSKIP cap {g.current:.1f}A K={g.stiffness:g} {g.direction}: {why}")
            skipped.append((g, why))
            continue
        if need_reset:
            cmd = [py, "-m", "scripts.reset_pose", "--preset", args.reset_preset,
                   "--tol-mm", f"{args.reset_tol_mm:g}"]
            if args.pose:
                cmd += ["--target", args.pose]
            rc, _ = _run(cmd)
            if rc != 0:
                print("reset_pose failed; stopping bench")
                break
            need_reset = args.reset_every
        print(f"\n=== cap {g.current:.1f}A  K={g.stiffness:g}  {g.direction} ===")
        rc, out = _run([py, "-m", "scripts.step_response", "--preset", args.preset,
                        "--stiffness", f"{g.stiffness:g}", "--current", f"{g.current:g}",
                        "--step-mm", f"{g.step_mm:g}", "--dir", g.direction,
                        "--cooldown-s", f"{args.cooldown_s:g}"])
        summary = parse_summary_line(out) or {}
        log_dir = None
        for line in out.splitlines():
            if line.startswith("log: "):
                log_dir = line[5:].strip()
        r = BenchResult(g, summary, log_dir, rc)
        results.append(r)
        print(f"--> {r.verdict}  ss={summary.get('steady_state_error_mm')}  "
              f"overshoot={summary.get('overshoot_mm')}  peakI={summary.get('peak_current_a')}")
        if r.verdict != "converged":
            need_reset = True   # a tripped/aborted run may have left the arm off-centre
        write_report()

    write_report()
    print(f"\nreport: {out_dir / 'report.md'}")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
