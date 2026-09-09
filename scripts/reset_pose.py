"""One-command reset-to-test-pose, so gain tuning always has a known-good
fallback to return to between experiments.

    python -m scripts.reset_pose                       # preset move-slow-0p3, default target
    python -m scripts.reset_pose --target 120,80        # explicit xy, mm
    python -m scripts.reset_pose --preset move-1p5-sched --passes 3

Wraps ``scripts.goto_pose.goto_pose`` (this module owns the ramp/guard loop;
this script just drives it, once per pass, with a fixed idle pause between
passes). Refuses to run at all if calibration/test_pose is missing or joint
limits are unset -- there is no safe "reset" without both. Never raises the
preset's current above its own configured value across passes (i.e. every
pass uses exactly the preset's ``current`` -- there is no escalation beyond
what the preset says is safe; see the 2026-09-04 recentring note where 1.5A
stalled the shoulder twice on a 103mm move and 2.0A finished it, which is
why ``move-2A-sched`` is the default rather than ``move-1p5-sched``).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from panto.can_link import AXIS_STATE_IDLE, CanLink
from panto.config import Config
from panto.kinematics import forward
from panto.limits import format_limits_deg, has_limits
from panto.presets import resolve as resolve_preset
from panto.step_logic import parse_per_joint, to_per_joint

from scripts.goto_pose import goto_pose

DEFAULT_PRESET = "move-slow-0p3"
CONVERGED_TOL_MM = 3.0
INTER_PASS_PAUSE_S = 20.0
S_PER_100MM = 8.0
MIN_DURATION_S = 4.0


def compute_duration_s(dist_mm: float, *, s_per_100mm: float = S_PER_100MM,
                       min_s: float = MIN_DURATION_S) -> float:
    """8s per 100mm of distance (the pace found not to stall the shoulder on
    a slow ramped move), floored at ``min_s`` so a tiny residual offset still
    gets a sane ramp instead of an near-instant step."""
    return max(min_s, s_per_100mm * dist_mm / 100.0)


def parse_target(spec: str | None, config: Config) -> np.ndarray:
    """``None`` -> ``config.test_pose_xy_m`` (raises if unset). ``"x,y"`` ->
    that xy in mm, converted to metres. Pure/no I/O beyond ``config``."""
    if spec is None:
        target = config.test_pose_xy_m
        if target is None:
            raise ValueError("no test_pose configured in calibration.json -- "
                             "refusing to reset without a known target")
        return target
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--target expects 'x,y' (mm), got {spec!r}")
    return np.array([float(parts[0]), float(parts[1])]) * 1e-3


def joint_limits_known(config: Config) -> bool:
    return all(has_limits(m) for m in config.motors)


def main() -> None:
    p = argparse.ArgumentParser(prog="reset_pose", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--preset", type=str, default=DEFAULT_PRESET,
                   help=f"named preset from presets.json (default {DEFAULT_PRESET!r})")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds for the ramp (default: 8s per 100mm of distance, min 4s)")
    p.add_argument("--passes", type=int, default=2,
                   help="repeat the goto while remaining distance exceeds "
                        f"{CONVERGED_TOL_MM}mm, up to this many passes, with a "
                        f"{INTER_PASS_PAUSE_S:.0f}s idle pause between them")
    p.add_argument("--target", type=str, default=None,
                   help="'x,y' mm; default config.test_pose")
    p.add_argument("--rate", type=float, default=200.0, help="control loop Hz")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    if config.test_pose is None:
        raise SystemExit("no test_pose configured in calibration.json -- refusing to reset "
                         "without a known-good reference pose")
    if not joint_limits_known(config):
        raise SystemExit("joint limits are not configured on one or both motors -- refusing "
                         "to reset without known limits (see calibration.json q_min_rad/q_max_rad)")

    try:
        target = parse_target(args.target, config)
    except ValueError as exc:
        raise SystemExit(str(exc))

    resolved = resolve_preset(
        {"stiffness": None, "vel_gain": None, "vel_limit": None, "current": None,
         "cap_slope": None, "cap_min": None, "ff_scale": None},
        args.preset,
    )
    if resolved["stiffness"] is None or resolved["vel_gain"] is None or resolved["vel_limit"] is None \
            or resolved["current"] is None:
        raise SystemExit(f"preset {args.preset!r} does not fully specify stiffness/vel_gain/"
                         f"vel_limit/current -- reset_pose needs all four")
    cap_min = tuple(resolved["cap_min"]) if resolved["cap_min"] is not None else (0.5, 0.5)
    cap_slope = to_per_joint(resolved["cap_slope"]) or (0.0, 0.0)

    link = CanLink(config, sim=False)
    link.start()
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        dist_mm = float(np.linalg.norm(target - pose0) * 1e3)
        print(f"reset_pose: preset={args.preset}  start=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
              f"target=({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm  dist={dist_mm:.1f}mm  "
              f"limits={format_limits_deg(config.motors)}")
    finally:
        link.close()

    duration = args.duration if args.duration is not None else compute_duration_s(dist_mm)

    remaining_mm = dist_mm
    for pass_i in range(1, args.passes + 1):
        print(f"\n-- pass {pass_i}/{args.passes}: duration={duration:.1f}s current={resolved['current']}A --")
        link = CanLink(config, sim=False)
        result = goto_pose(
            link, config, target,
            stiffness=resolved["stiffness"], current=resolved["current"],
            vel_gain=resolved["vel_gain"], vel_limit=resolved["vel_limit"],
            duration=duration, hold=1.0, rate=args.rate,
            cap_slope=cap_slope, cap_min=cap_min, ff_scale=resolved["ff_scale"],
        )
        remaining_mm = result.remaining_dist_mm
        print(f"  pass {pass_i} result: aborted={result.aborted} "
              f"({result.abort_reason if result.aborted else 'ok'})  "
              f"end=({result.end_xy_m[0]*1e3:.1f},{result.end_xy_m[1]*1e3:.1f})mm  "
              f"remaining={remaining_mm:.1f}mm")
        if remaining_mm <= CONVERGED_TOL_MM:
            break
        if pass_i < args.passes:
            print(f"  {remaining_mm:.1f}mm > {CONVERGED_TOL_MM}mm -- "
                  f"pausing {INTER_PASS_PAUSE_S:.0f}s before next pass")
            time.sleep(INTER_PASS_PAUSE_S)
        # re-approach from the current (post-pass) pose next time round;
        # duration is fixed at the original distance's pace, not recomputed
        # from the (now much smaller) remaining distance -- a slow final
        # crawl is fine and keeps the schedule simple/predictable.

    link = CanLink(config, sim=False)
    link.start()
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        status = link.node_status()
        idle_ok = all(s.axis_state == AXIS_STATE_IDLE for s in status)
        errors_ok = all(not s.active_errors for s in status)
        print(f"\nfinal check: idle={idle_ok} errors_clear={errors_ok} "
              f"remaining={remaining_mm:.1f}mm (tol {CONVERGED_TOL_MM}mm)")
        if not idle_ok or not errors_ok:
            for s in status:
                print(f"  node {s.node_id}: axis_state={s.axis_state} active_errors=0x{s.active_errors:x}")
    finally:
        link.close()

    if remaining_mm > CONVERGED_TOL_MM or not idle_ok or not errors_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
