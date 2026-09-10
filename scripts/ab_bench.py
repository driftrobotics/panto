"""A/B bench: compare ``PositionBackend`` and ``TorqueBackend`` on the two
tasks where they are expected to differ -- tangential drag along a diagonal
snap-to-line, and a one-sided wall -- behind the identical ``ImpedanceCommand``
interface (see CONTRACTS.md's "ImpedanceBackend" and milestone 6).

    python -m scripts.ab_bench --sim
    python -m scripts.ab_bench --sim --backends torque --speed-mm-s 20
    python -m scripts.ab_bench --sim --preset hover-K25-pj --pose 100,80

For each requested backend, in one continuous armed session:

  1. Move to ``--pose`` (default ``config.test_pose`` if set, else the sim
     home pose ``forward(SIM_HOME_JOINT_RAD, geo)``) with a ramped Point
     anchor, using that backend.
  2. Line task: a ``Line`` at 45 degrees through the pose. The commanded
     anchor slides ``--line-mm`` (default 30) along the line at
     ``--speed-mm-s`` (default 10) and back (a ramp/hold/ramp/hold schedule
     from ``panto.step_logic.anchor_offset_m`` applied to the line's
     tangential coordinate). Metrics: tangential-drag RMS current, lateral
     RMS deviation from the line, overshoot/settle time/steady-state error of
     the tangential tracking (``panto.step_logic.analyze_step`` on the
     tangential displacement), peak/RMS/I2t current.
  3. Wall task: a ``Wall`` through the pose with normal +y (free side +y).
     A ``Point`` "push" target steps ``--wall-mm`` (default 5) from the wall
     surface into the blocked (-y) side and back, combined with the Wall
     constraint via ``panto.ab_logic.combine_constraints`` (same sum-bilateral
     / gate-unilateral policy as ``Runtime._combine``) so the wall actually
     resists the push instead of being commanded straight through. Metrics:
     normal compliance (mm penetration per amp of restoring current), the
     penetration channel's overshoot/settle/steady-state-error (same
     ``analyze_step`` reuse -- see the note in panto/ab_logic.py about what
     "stall" means here), pull-away time after the push retreats, peak/RMS/I2t
     current.

Gains are read-only from ``--preset`` (default ``hover-K25-pj``, falling back
to ``pos-bw300-K100`` with a printed/logged reason if the named preset isn't
in this worktree's presets.json) -- no CLI override for stiffness/vel_gain/
etc, so an A/B run can't quietly drift off the registry. ``--current``/
``--cooldown-s`` mirror scripts/stiffness_bench.py's hardware flags; the same
disarm/heartbeat/feedback-age/excursion aborts and
``panto.guard.OscillationGuard`` as scripts/impedance_step.py guard every
tick. Position-backend ``vel_gain`` is clamped to <= 0.05 regardless of what
the preset says (41 Hz shoulder chatter above that on the real rig).

Only ``--sim`` has been exercised for this bench (per the work-stream's
scope); it must complete end-to-end and produce the report, but nothing here
should be run against hardware.

Output: ``logs/ab_bench-<utc>/report.json`` (raw metrics) and ``report.md``
(the side-by-side table), written incrementally after each backend so a
crash on the second backend still leaves the first backend's results.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import argparse
import numpy as np

from panto.ab_logic import (
    analyze_line, analyze_wall, build_report_md, combine_constraints,
    line_direction, project_current,
)
from panto.backends import PositionBackend, TorqueBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import (
    AXIS_STATE_CLOSED_LOOP_CONTROL, SIM_HOME_JOINT_RAD, CanLink, CanLinkError,
    decode_error_flags,
)
from panto.config import Config
from panto.constraints import Line, Point, Wall
from panto.guard import OscillationGuard
from panto.kinematics import forward, jacobian
from panto.limits import JointLimitViolation, format_limits_deg, q_deg
from panto.presets import PresetError, apply_to_config, drive_defaults, get_preset, restore_drive_defaults
from panto.step_logic import anchor_offset_m, analyze_step, phase_name, to_per_joint
from panto.telemetry import RunLogger

DEFAULT_PRESET = "hover-K25-pj"
FALLBACK_PRESET = "pos-bw300-K100"
MAX_VEL_GAIN = 0.05   # never send the position cascade a higher vel_gain -- 41 Hz shoulder chatter

PRE_S = 0.5           # settle at the task's start pose before ramping
HOLD_S = 0.6           # hold at the extremum before ramping back

MAX_FEEDBACK_AGE_S = 0.03
MAX_HEARTBEAT_AGE_S = 2.0
FORCE_LIMIT_N = 50.0   # generous Cartesian force ceiling; the current cap is the real limit

LINE_ANGLE_DEG = 45.0


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    pass


class DriveLost(Aborted):
    pass


# --------------------------------------------------------------------- setup

def _resolve_bench_preset(name: str) -> tuple[str, dict, str | None]:
    """(preset_name_used, tuning_dict, fallback_reason_or_None)."""
    try:
        preset = get_preset(name)
        return name, preset.tuning_dict(), None
    except PresetError:
        reason = f"preset {name!r} not found in presets.json; falling back to {FALLBACK_PRESET!r}"
        preset = get_preset(FALLBACK_PRESET)  # let this raise if even the fallback is missing
        return FALLBACK_PRESET, preset.tuning_dict(), reason


def _clamp_vel_gain(vel_gain) -> tuple[float, float]:
    per_joint = to_per_joint(vel_gain) if vel_gain is not None else (0.02, 0.02)
    return tuple(min(v, MAX_VEL_GAIN) for v in per_joint)


def _resolve_target_pose(args, config: Config) -> np.ndarray:
    if args.pose:
        parts = [p.strip() for p in args.pose.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"--pose must be 'x,y' mm, got {args.pose!r}")
        return np.array([float(parts[0]), float(parts[1])]) * 1e-3
    if config.test_pose_xy_m is not None:
        return config.test_pose_xy_m
    if not args.sim:
        raise SystemExit("no test_pose configured and --pose not given -- refusing to pick an "
                          "arbitrary target on hardware")
    return forward(np.array(SIM_HOME_JOINT_RAD, dtype=float), config.geo)


def _check_health(link: CanLink, t: float) -> None:
    for s in link.node_status():
        if s.age_s > MAX_HEARTBEAT_AGE_S:
            raise DriveLost(f"drive lost at t={t:.3f}s: node {s.node_id} heartbeat age "
                             f"{s.age_s:.2f}s exceeds {MAX_HEARTBEAT_AGE_S:.1f}s cap")
        if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
            reason = decode_error_flags(s.disarm_reason or 0)
            raise DriveDisarmed(f"drive disarmed at t={t:.3f}s: node {s.node_id} axis_state "
                                 f"{s.axis_state} (left CLOSED_LOOP) reason={reason}")
    age_s = link.feedback_age_s()
    if age_s > MAX_FEEDBACK_AGE_S:
        raise Aborted(f"feedback age {age_s * 1e3:.1f}ms exceeds {MAX_FEEDBACK_AGE_S * 1e3:.0f}ms cap")


# ------------------------------------------------------------------ one backend run

def run_one_backend(name: str, args, target_xy: np.ndarray,
                     resolved: dict, wall_stiffness: float, log: RunLogger) -> dict:
    """Run the move + line task + wall task for one backend. Never raises for
    an in-run abort (records it in the result); does raise for setup errors
    the caller can't sensibly continue past."""
    config = Config.load(args.config)
    for m in config.motors:
        m.current_soft_max = resolved["current"]
    default_vel_gain = drive_defaults(config)

    K_push = resolved["stiffness"] * np.eye(2)
    vel_gain_per_joint = _clamp_vel_gain(resolved["vel_gain"])

    if name == "position":
        apply_to_config(config, {**resolved, "vel_gain": list(vel_gain_per_joint)})

    link = CanLink(config, sim=args.sim)
    if name == "torque":
        backend = TorqueBackend(link, config)
        # damping/vel_lpf_hz aren't preset fields (TUNING_FIELDS is
        # position-cascade-shaped) -- TorqueBackend's own module docstring
        # says an undamped spring at CAN-delay + host-loop latency rings, so
        # a bare `damping=0.0` default here would make the line/wall tasks
        # measure that ringing instead of the backend's steady-state
        # behaviour. --damping/--vel-lpf-hz are plain CLI knobs, same as
        # scripts/impedance_step.py and scripts/trace_shape.py already do.
        backend.damping = args.damping
        backend.vel_lpf_hz = args.vel_lpf_hz
    else:
        backend = PositionBackend(link, config)
        backend.vel_limit_rad_s = resolved["vel_limit"] or backend.vel_limit_rad_s

    nodes = [m.node_id for m in config.motors]
    result: dict = {"backend": name, "aborted": False, "abort_reason": None,
                     "line": {}, "wall": {}}
    armed = False
    bus_lost = False
    rows_line: list[dict] = []
    rows_wall: list[dict] = []

    def idle_all() -> None:
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                log.event(f"[{name}] idle node {nid}: {exc}", level="ERROR")

    try:
        link.start()
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"[{name}] node {s.node_id} has an active error at rest "
                                  f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        log.event(f"[{name}] start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm "
                  f"target=({target_xy[0]*1e3:.1f},{target_xy[1]*1e3:.1f})mm  "
                  f"limits={format_limits_deg(config.motors)}")

        if name == "position":
            for i, nid in enumerate(nodes):
                link.set_vel_gains(nid, vel_gain_per_joint[i], 0.0)

        backend.enter()
        for i, nid in enumerate(nodes):
            link.set_input_pos(nid, float(q0[i]))
        try:
            link.enter_closed_loop(timeout=5.0)
        except CanLinkError as exc:
            raise SystemExit(f"[{name}] refusing to arm: {exc}")
        armed = True
        for nid in nodes:
            link.clear_errors(nid)
        if name == "position":
            for i, nid in enumerate(nodes):
                link.set_vel_gains(nid, vel_gain_per_joint[i], 0.0)

        rate = args.rate or (500.0 if name == "torque" else 250.0)
        period = 1.0 / rate

        # --------------------------------------------------------- move to pose
        move_guard = OscillationGuard(osc_mm=args.osc_mm)
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= args.move_s + args.move_hold_s:
                break
            _check_health(link, t)
            frac = min(1.0, t / args.move_s) if args.move_s > 0 else 1.0
            anchor = pose0 + frac * (target_xy - pose0)
            q, qd = link.joint_state()
            pose = forward(q, config.geo)
            exc_mm = float(np.linalg.norm(pose - pose0) * 1e3)
            if exc_mm > args.max_move_mm + float(np.linalg.norm(target_xy - pose0) * 1e3):
                raise Aborted(f"move excursion {exc_mm:.1f}mm exceeds cap")
            cmd = ImpedanceCommand(pose=pose, q=q, qd=qd, anchor=anchor, stiffness=K_push,
                                    force_limit=FORCE_LIMIT_N)
            backend.apply(cmd)
            cur = link.motor_currents()
            # Guard on deviation-from-anchor, not raw pose: the anchor itself
            # is intentionally moving (150mm+ moves are normal here), so a
            # raw-pose std check would trip on the commanded motion itself.
            # See OscillationGuard's docstring -- it was written for a static
            # target (point_hold/offset_sweep); feeding it (pose - anchor)
            # instead of pose keeps its buzz/stall detectors meaningful for a
            # moving target too.
            move_guard.push(t, (pose - anchor) * 1e3, cur, resolved["current"],
                             float(np.linalg.norm(pose - anchor) * 1e3))
            trip = move_guard.check()
            if trip:
                raise Aborted(f"oscillation during move-to-pose: {trip}")
            time.sleep(period)

        q, _ = link.joint_state()
        pose_task = forward(q, config.geo)
        log.event(f"[{name}] at task pose ({pose_task[0]*1e3:.1f},{pose_task[1]*1e3:.1f})mm")

        # --------------------------------------------------------------- line task
        d_hat = line_direction(LINE_ANGLE_DEG)
        line = Line(a=pose_task, d=d_hat)
        distance_m = args.line_mm * 1e-3
        speed_m_s = args.speed_mm_s * 1e-3
        ramp_s = distance_m / speed_m_s if speed_m_s > 0 else 0.0
        # anchor_offset_m's schedule: pre-hold, ramp out, hold, ramp back, hold.
        line_total_s = PRE_S + ramp_s + HOLD_S + ramp_s + HOLD_S

        guard = OscillationGuard(osc_mm=args.osc_mm)
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= line_total_s:
                break
            _check_health(link, t)
            s_m = anchor_offset_m(t, distance_m, ramp_s=ramp_s, pre_s=PRE_S,
                                   step_hold_s=HOLD_S, back_hold_s=HOLD_S)
            anchor = pose_task + s_m * d_hat
            q, qd = link.joint_state()
            pose = forward(q, config.geo)
            exc_mm = float(np.linalg.norm(pose - pose_task) * 1e3)
            if exc_mm > args.max_excursion_mm:
                raise Aborted(f"line task excursion {exc_mm:.1f}mm exceeds cap")
            cmd = ImpedanceCommand(pose=pose, q=q, qd=qd, anchor=anchor, stiffness=K_push,
                                    force_limit=FORCE_LIMIT_N)
            try:
                backend.apply(cmd)
            except JointLimitViolation as exc:
                raise Aborted(f"joint limit violation ({exc})")
            cur = link.motor_currents()
            guard.push(t, (pose - anchor) * 1e3, cur, resolved["current"],
                       float(np.linalg.norm(pose - anchor) * 1e3))
            trip = guard.check()
            if trip:
                raise Aborted(f"oscillation during line task: {trip}")
            J = jacobian(q, config.geo)
            proj = line.project(pose)
            rows_line.append({
                "t": t, "phase": phase_name(t, ramp_s=ramp_s, pre_s=PRE_S,
                                             step_hold_s=HOLD_S, back_hold_s=HOLD_S),
                "s_cmd_mm": s_m * 1e3,
                "s_actual_mm": float((pose - pose_task) @ d_hat) * 1e3,
                "lateral_mm": proj.penetration * 1e3,
                "tangential_current_a": project_current(cur, J, d_hat),
                "currents": cur.tolist(),
            })
            log.sample(t=t, task="line", backend=name, q=q, qd=qd, pose=pose, anchor=anchor,
                       currents=cur, node_status=link.node_status())
            time.sleep(period)

        # --------------------------------------------------------------- wall task
        normal = np.array([0.0, 1.0])   # free side +y
        wall = Wall(a=pose_task, normal=normal)
        depth_m = args.wall_mm * 1e-3
        ramp_s_w = depth_m / speed_m_s if speed_m_s > 0 else 0.0
        wall_step_end_s = PRE_S + ramp_s_w + HOLD_S
        wall_total_s = wall_step_end_s + ramp_s_w + HOLD_S

        guard = OscillationGuard(osc_mm=args.osc_mm)
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= wall_total_s:
                break
            _check_health(link, t)
            push_depth_m = anchor_offset_m(t, depth_m, ramp_s=ramp_s_w, pre_s=PRE_S,
                                            step_hold_s=HOLD_S, back_hold_s=HOLD_S)
            push_target = pose_task - push_depth_m * normal
            q, qd = link.joint_state()
            pose = forward(q, config.geo)
            exc_mm = float(np.linalg.norm(pose - pose_task) * 1e3)
            if exc_mm > args.max_excursion_mm:
                raise Aborted(f"wall task excursion {exc_mm:.1f}mm exceeds cap")
            push_proj = Point(at=push_target).project(pose)
            wall_proj = wall.project(pose)
            K, pull, active = combine_constraints(
                ((push_proj, resolved["stiffness"]), (wall_proj, wall_stiffness))
            )
            anchor = np.linalg.solve(K, pull) if active else pose.copy()
            cmd = ImpedanceCommand(pose=pose, q=q, qd=qd, anchor=anchor, stiffness=K,
                                    force_limit=FORCE_LIMIT_N)
            try:
                backend.apply(cmd)
            except JointLimitViolation as exc:
                raise Aborted(f"joint limit violation ({exc})")
            cur = link.motor_currents()
            guard.push(t, (pose - anchor) * 1e3, cur, resolved["current"],
                       float(np.linalg.norm(pose - anchor) * 1e3))
            trip = guard.check()
            if trip:
                raise Aborted(f"oscillation during wall task: {trip}")
            J = jacobian(q, config.geo)
            rows_wall.append({
                "t": t, "phase": phase_name(t, ramp_s=ramp_s_w, pre_s=PRE_S,
                                             step_hold_s=HOLD_S, back_hold_s=HOLD_S),
                "push_depth_mm": push_depth_m * 1e3,
                "penetration_mm": wall_proj.penetration * 1e3,
                "restoring_current_a": project_current(cur, J, normal),
                "currents": cur.tolist(),
            })
            log.sample(t=t, task="wall", backend=name, q=q, qd=qd, pose=pose, anchor=anchor,
                       currents=cur, node_status=link.node_status())
            time.sleep(period)

    except DriveLost as exc:
        result["aborted"] = True
        bus_lost = True
        result["abort_reason"] = str(exc)
        log.event(f"[{name}] aborted (bus lost): {exc}", level="ERROR")
    except (Aborted, DriveDisarmed, JointLimitViolation) as exc:
        result["aborted"] = True
        result["abort_reason"] = str(exc)
        log.event(f"[{name}] aborted: {exc}", level="ERROR")
    except SystemExit as exc:
        result["aborted"] = True
        result["abort_reason"] = str(exc)
        log.event(f"[{name}] {exc}", level="ERROR")
    finally:
        if bus_lost:
            log.event(f"[{name}] bus lost -- skipping relax/idle/restore writes", level="WARN")
        else:
            if armed:
                try:
                    backend.relax()
                except Exception:  # noqa: BLE001
                    pass
                restore_drive_defaults(link, default_vel_gain, log)
            idle_all()
            time.sleep(0.1)
        link.close()

    period_line = 1.0 / (args.rate or (500.0 if name == "torque" else 250.0))

    if rows_line:
        t_all = np.array([r["t"] for r in rows_line])
        s_actual = np.array([r["s_actual_mm"] for r in rows_line])
        lateral = np.array([r["lateral_mm"] for r in rows_line])
        tang_cur = np.array([r["tangential_current_a"] for r in rows_line])
        currents = np.array([r["currents"] for r in rows_line])
        step_mask = np.array([r["phase"] in ("ramp_up", "step_hold") for r in rows_line])
        line_metrics = analyze_line(tang_cur, lateral, currents, period_line)
        result["line"] = {**line_metrics.__dict__, "verdict": "no_data"}
        if step_mask.any():
            t_step = t_all[step_mask] - t_all[step_mask][0]
            step_res = analyze_step(t_step, s_actual[step_mask], args.line_mm)
            result["line"].update({
                "overshoot_mm": step_res.overshoot_mm,
                "settling_time_s": step_res.settling_time_s,
                "steady_state_error_mm": step_res.steady_state_error_mm,
                "verdict": step_res.verdict,
            })

    if rows_wall:
        t_all = np.array([r["t"] for r in rows_wall])
        penetration = np.array([r["penetration_mm"] for r in rows_wall])
        restoring = np.array([r["restoring_current_a"] for r in rows_wall])
        currents = np.array([r["currents"] for r in rows_wall])
        step_mask = np.array([r["phase"] in ("ramp_up", "step_hold") for r in rows_wall])
        hold_mask = np.array([r["phase"] == "step_hold" for r in rows_wall])
        wall_metrics = analyze_wall(
            penetration[hold_mask] if hold_mask.any() else penetration,
            restoring[hold_mask] if hold_mask.any() else restoring,
            t_all, penetration, wall_step_end_s, currents, period_line,
        )
        result["wall"] = {**wall_metrics.__dict__, "verdict": "no_data"}
        if step_mask.any():
            t_step = t_all[step_mask] - t_all[step_mask][0]
            step_res = analyze_step(t_step, penetration[step_mask], args.wall_mm)
            result["wall"].update({
                "overshoot_mm": step_res.overshoot_mm,
                "settling_time_s": step_res.settling_time_s,
                "steady_state_error_mm": step_res.steady_state_error_mm,
                "verdict": step_res.verdict,
            })

    return result


# ------------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(prog="ab_bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--sim", action="store_true", help="use the in-process sim bus instead of hardware")
    p.add_argument("--backends", type=str, default="position,torque",
                   help="comma list from {position, torque}")
    p.add_argument("--preset", default=DEFAULT_PRESET,
                   help=f"named preset from presets.json (default {DEFAULT_PRESET!r}, falls back to "
                        f"{FALLBACK_PRESET!r} if absent); gains are read-only from the registry")
    p.add_argument("--current", type=float, default=None, help="override the preset's current cap, A")
    p.add_argument("--pose", type=str, default=None,
                   help="'x,y' mm task pose; default config.test_pose, else the sim home pose")
    p.add_argument("--speed-mm-s", type=float, default=10.0, help="tangential/push anchor speed, mm/s")
    p.add_argument("--line-mm", type=float, default=30.0, help="line task slide distance, mm")
    p.add_argument("--wall-mm", type=float, default=5.0, help="wall task push depth, mm")
    p.add_argument("--wall-stiffness", type=float, default=None,
                   help="wall normal stiffness, N/m (default config.control.wall_stiffness_n_per_m)")
    p.add_argument("--damping", type=float, default=1.0,
                   help="torque backend only: isotropic tip damping B, N.s/m -- not a preset field "
                        "(see panto.backends.torque's module docstring on why an undamped host-side "
                        "spring rings; 0 reproduces that ringing)")
    p.add_argument("--vel-lpf-hz", type=float, default=20.0,
                   help="torque backend only: low-pass cutoff (Hz) on qd before the damping term")
    p.add_argument("--rate", type=float, default=None,
                   help="control loop Hz; default 250 (position) / 500 (torque)")
    p.add_argument("--move-s", type=float, default=3.0, help="ramp duration to reach the task pose")
    p.add_argument("--move-hold-s", type=float, default=0.5)
    p.add_argument("--max-move-mm", type=float, default=150.0,
                   help="abort if the move-to-pose excursion exceeds (move distance + this), mm")
    p.add_argument("--max-excursion-mm", type=float, default=80.0,
                   help="abort a task if the pose strays this far from the task pose, mm")
    p.add_argument("--osc-mm", type=float, default=10.0,
                   help="panto.guard.OscillationGuard pose-deviation threshold, mm. Guarded signal "
                        "is (pose - anchor), not raw pose (the anchor moves 30mm+ by design here), so "
                        "this is loosened from OscillationGuard's point-hold default (3mm) to tolerate "
                        "ordinary following error/lag during a ramp; the current-pinned+not-converging/"
                        "alternating check still catches genuine stalls/buzz at this threshold")
    p.add_argument("--cooldown-s", type=float, default=5.0,
                   help="idle pause after each backend's run (mirrors stiffness_bench.py; sim doesn't "
                        "need thermal recovery, kept small by default -- raise for hardware)")
    p.add_argument("--logs", default="logs")
    args = p.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in ("position", "torque"):
            raise SystemExit(f"unknown backend {b!r}; expected position or torque")

    meta_config = Config.load(args.config)
    preset_used, tuning, fallback_reason = _resolve_bench_preset(args.preset)
    if fallback_reason:
        print(f"! {fallback_reason}")
    resolved = dict(tuning)
    if args.current is not None:
        resolved["current"] = args.current
    if resolved.get("current") is None:
        resolved["current"] = 0.8
    if resolved.get("stiffness") is None:
        raise SystemExit(f"preset {preset_used!r} has no stiffness")

    wall_stiffness = (args.wall_stiffness if args.wall_stiffness is not None
                       else meta_config.control.wall_stiffness_n_per_m)
    target_xy = _resolve_target_pose(args, meta_config)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logs) / f"ab_bench-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log = RunLogger("ab_bench", sim=args.sim, backends=backends, preset=preset_used,
                     preset_requested=args.preset, current=resolved["current"],
                     pose_mm=(target_xy * 1e3).tolist(), speed_mm_s=args.speed_mm_s,
                     line_mm=args.line_mm, wall_mm=args.wall_mm, wall_stiffness=wall_stiffness)

    meta = {
        "utc": stamp, "preset": preset_used, "preset_requested": args.preset,
        "preset_fallback_reason": fallback_reason, "sim": args.sim,
        "pose_mm": (target_xy * 1e3).round(2).tolist(),
        "speed_mm_s": args.speed_mm_s, "line_distance_mm": args.line_mm,
        "wall_depth_mm": args.wall_mm, "wall_stiffness_n_per_m": wall_stiffness,
        "current_a": resolved["current"],
    }
    results: dict = {}

    def write_report() -> None:
        (out_dir / "report.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
        (out_dir / "report.md").write_text(build_report_md(meta, results) + "\n")

    for name in backends:
        print(f"\n=== backend={name} preset={preset_used} pose={meta['pose_mm']}mm ===")
        r = run_one_backend(name, args, target_xy, resolved, wall_stiffness, log)
        results[name] = r
        print(f"  aborted={r['aborted']} reason={r.get('abort_reason')}")
        if r["line"]:
            print(f"  line: tangential_I_rms={r['line'].get('tangential_current_rms_a'):.4f}A "
                  f"lateral_rms={r['line'].get('lateral_rms_mm'):.3f}mm "
                  f"verdict={r['line'].get('verdict')}")
        if r["wall"]:
            print(f"  wall: compliance={r['wall'].get('normal_compliance_mm_per_a')} mm/A "
                  f"pull_away={r['wall'].get('pull_away_time_s')} s "
                  f"verdict={r['wall'].get('verdict')}")
        write_report()
        if args.cooldown_s > 0:
            time.sleep(args.cooldown_s)

    write_report()
    log.close()
    print(f"\nreport: {out_dir / 'report.md'}")
    print((out_dir / "report.md").read_text())

    if any(r["aborted"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
