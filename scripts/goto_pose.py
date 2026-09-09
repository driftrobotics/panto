"""One-off: gently ramp the anchor from the current pose to `config.test_pose`,
so a subsequent scripts.step_response run starts within its 60mm guard.

    python -m scripts.goto_pose --duration 4 --stiffness 20 --current 1.0
    python -m scripts.goto_pose --preset move-1p5-sched --duration 8

Ramps linearly over --duration seconds (no steps, no oscillation risk at this
K), then holds 1s, then relax/IDLE. Aborts on joint-limit approach, drive
disarm, or feedback age > 30ms -- same guard family as step_response.py.

2026-09-04: the plain constant-cap loop at 1.0A wobbled visibly on the way in.
--cap-slope/--cap-min apply the same velocity-scheduled current cap as
step_response.py (see panto/backends/position.py's _vel_scheduled_cap) so this
recentring move gets the same anti-oscillation benefit.

The move itself lives in ``goto_pose()`` below (not just inside ``main()``) so
scripts/reset_pose.py can call it directly instead of duplicating the ramp/
guard logic.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from panto.backends import PositionBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.constraints import Point
from panto.kinematics import forward
from panto.limits import JointLimitViolation, format_limits_deg, q_deg
from panto.presets import apply_to_config, drive_defaults, resolve as resolve_preset
from panto.step_logic import parse_per_joint, to_per_joint


@dataclass
class GotoResult:
    aborted: bool
    abort_reason: str | None
    start_xy_m: np.ndarray
    end_xy_m: np.ndarray
    target_xy_m: np.ndarray
    start_dist_mm: float
    remaining_dist_mm: float


def goto_pose(link: CanLink, config: Config, target: np.ndarray, *,
             stiffness: float, current: float,
             vel_gain: float | tuple[float, float], vel_limit: float,
             duration: float, hold: float = 1.0, rate: float = 200.0,
             cap_slope: float | tuple[float, float] = 0.0,
             cap_min: tuple[float, float] = (0.5, 0.5),
             ff_scale: float | None = None, print_progress: bool = True) -> GotoResult:
    """Ramp the anchor from the current pose to ``target`` (metres, xy) over
    ``duration`` seconds, then hold ``hold`` seconds, then relax/IDLE.

    ``link`` must already be constructed (not started) -- this function calls
    ``link.start()``/``link.close()`` itself, so it owns the whole CAN session
    for this move. Returns a ``GotoResult`` describing what happened; never
    raises for an in-run abort (reports it in the result instead), but does
    raise for a bad target/setup and for KeyboardInterrupt.
    """
    vel_gain_per_joint = to_per_joint(vel_gain)
    for m in config.motors:
        m.current_soft_max = current
    default_vel_gain = drive_defaults(config)
    apply_to_config(config, {
        "vel_gain": list(vel_gain_per_joint), "max_pos_gain": None,
        "cap_slope": cap_slope, "cap_min": list(cap_min), "ff_scale": ff_scale,
    })
    nodes = [m.node_id for m in config.motors]
    K = stiffness * np.eye(2)

    backend = PositionBackend(link, config)
    backend.vel_limit_rad_s = vel_limit
    link.start()
    armed = False
    aborted = False
    abort_reason = None
    pose0 = None
    pose_end = None
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        dist_mm = float(np.linalg.norm(target - pose0) * 1e3)
        if print_progress:
            pos_gain0 = PositionBackend.pos_gains_for(K, q0, config)
            print(f"start pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
                  f"target=({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm  dist={dist_mm:.1f}mm  "
                  f"vel_gain={vel_gain}  pos_gain@start={[round(g, 1) for g in pos_gain0]}  "
                  f"limits={format_limits_deg(config.motors)}")

        for i, nid in enumerate(nodes):
            link.set_vel_gains(nid, vel_gain_per_joint[i], 0.0)
        backend.enter()
        for i, nid in enumerate(nodes):
            link.set_input_pos(nid, float(q0[i]))
        try:
            link.enter_closed_loop(timeout=5.0)
        except CanLinkError as exc:
            raise SystemExit(f"refusing to arm: {exc}")
        armed = True
        for i, nid in enumerate(nodes):
            link.clear_errors(nid)
            link.set_vel_gains(nid, vel_gain_per_joint[i], 0.0)

        period = 1.0 / rate
        t0 = time.monotonic()
        total = duration + hold
        while True:
            t = time.monotonic() - t0
            if t >= total:
                break
            age_s = link.feedback_age_s()
            if age_s > 0.03:
                raise RuntimeError(f"feedback age {age_s*1e3:.1f}ms exceeds 30ms")
            status = link.node_status()
            for s in status:
                if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    reason = decode_error_flags(s.disarm_reason or 0)
                    raise RuntimeError(f"drive disarmed: node {s.node_id} reason={reason}")
            frac = min(1.0, t / duration) if duration > 0 else 1.0
            anchor = pose0 + frac * (target - pose0)
            q, qd = link.joint_state()
            pose = forward(q, config.geo)
            proj = Point(at=anchor).project(pose)
            cmd = ImpedanceCommand(pose=pose, q=q, qd=qd, anchor=proj.anchor, stiffness=K, force_limit=50.0)
            try:
                backend.apply(cmd)
            except JointLimitViolation as exc:
                raise RuntimeError(f"joint limit violation ({exc})")
            if print_progress and t % 1.0 < period:
                print(f"  t={t:4.1f}s frac={frac:.2f} pose=({pose[0]*1e3:.1f},{pose[1]*1e3:.1f})mm "
                      f"q_deg={q_deg(q)}")
            time.sleep(period)
        q, _ = link.joint_state()
        pose_end = forward(q, config.geo)
        if print_progress:
            print(f"done: pose=({pose_end[0]*1e3:.1f},{pose_end[1]*1e3:.1f})mm  "
                  f"remaining dist={float(np.linalg.norm(target-pose_end)*1e3):.1f}mm")
    except Exception as exc:  # noqa: BLE001
        aborted = True
        abort_reason = f"{type(exc).__name__}: {exc}"
        if print_progress:
            print(f"! {type(exc).__name__}: {exc}")
    finally:
        if armed:
            try:
                backend.relax()
            except Exception:  # noqa: BLE001
                pass
            for nid in nodes:
                try:
                    link.set_vel_gains(nid, default_vel_gain[nid], 0.0)
                except Exception:  # noqa: BLE001
                    pass
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.2)
        link.close()

    if pose0 is None:
        pose0 = np.array([float("nan"), float("nan")])
    if pose_end is None:
        pose_end = pose0
    return GotoResult(
        aborted=aborted, abort_reason=abort_reason,
        start_xy_m=pose0, end_xy_m=pose_end, target_xy_m=target,
        start_dist_mm=float(np.linalg.norm(target - pose0) * 1e3),
        remaining_dist_mm=float(np.linalg.norm(target - pose_end) * 1e3),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--preset", type=str, default=None,
                   help="named preset from presets.json; supplies stiffness/current/vel-gain/"
                        "vel-limit/cap-slope/cap-min/ff-scale not given explicitly below")
    p.add_argument("--stiffness", type=float, default=None)
    p.add_argument("--current", type=float, default=None)
    p.add_argument("--vel-limit", type=float, default=None)
    p.add_argument("--vel-gain", type=str, default=None,
                   help="'V' (both joints) or 'V0,V1' (shoulder,elbow)")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--hold", type=float, default=1.0)
    p.add_argument("--rate", type=float, default=200.0)
    p.add_argument("--cap-slope", type=str, default=None,
                   help="A per rad/s, velocity-scheduled current cap (0 = off, constant --current). "
                        "'V' or 'V0,V1' (shoulder,elbow)")
    p.add_argument("--cap-min", type=str, default=None,
                   help="A, floor for the velocity-scheduled cap. 'V' or 'V0,V1' (shoulder,elbow)")
    p.add_argument("--ff-scale", type=float, default=None,
                   help="override motor.ff_scale (0 disables Coulomb feedforward)")
    args = p.parse_args()

    config = Config.load(args.config)
    resolved = resolve_preset(
        {
            "stiffness": args.stiffness,
            "vel_gain": parse_per_joint(args.vel_gain) if args.vel_gain is not None else None,
            "vel_limit": args.vel_limit,
            "current": args.current,
            "cap_slope": parse_per_joint(args.cap_slope) if args.cap_slope is not None else None,
            "cap_min": parse_per_joint(args.cap_min) if args.cap_min is not None else None,
            "ff_scale": args.ff_scale,
        },
        args.preset,
    )
    resolved["stiffness"] = 20.0 if resolved["stiffness"] is None else resolved["stiffness"]
    resolved["current"] = 1.0 if resolved["current"] is None else resolved["current"]
    resolved["vel_limit"] = 20.0 if resolved["vel_limit"] is None else resolved["vel_limit"]
    resolved["vel_gain"] = 0.09 if resolved["vel_gain"] is None else resolved["vel_gain"]
    vel_gain = to_per_joint(resolved["vel_gain"])
    cap_slope = to_per_joint(resolved["cap_slope"]) or (0.0, 0.0)
    cap_min = tuple(resolved["cap_min"]) if resolved["cap_min"] is not None else (0.5, 0.5)

    target = config.test_pose_xy_m
    if target is None:
        raise SystemExit("no test_pose configured")

    link = CanLink(config, sim=False)
    result = goto_pose(
        link, config, target,
        stiffness=resolved["stiffness"], current=resolved["current"],
        vel_gain=vel_gain, vel_limit=resolved["vel_limit"],
        duration=args.duration, hold=args.hold, rate=args.rate,
        cap_slope=cap_slope, cap_min=cap_min, ff_scale=resolved["ff_scale"],
    )
    if result.aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
