"""Symmetric +/-x, +/-y offset-hold sweep — averaged authority measurement.

    python -m scripts.offset_sweep                         # 0.4 A, 200 N/m, +-15mm
    python -m scripts.offset_sweep --current 0.5 --stiffness 250 --step 20

For each of +x, -x, +y, -y (in that order): step the anchor `--step` mm off the
start pose, hold `--settle` s to reach steady state, record the settled error
and current, then return to the start pose and re-settle before the next
direction. Reports per-direction results plus the average and a +/- symmetry
check, rather than a single anecdotal direction.

Every tick is logged via panto.telemetry.RunLogger (pose, anchor, error,
currents, node_status, and PositionBackend.last_command — what was actually
sent, not just measured). Gentle defaults; current-capped, no I2t protection in
this script (bypasses Runtime), so keep --current modest for multi-direction
runs. relax + IDLE on any exit path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

from panto.backends import PositionBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.constraints import Point
from panto.guard import OscillationGuard
from panto.kinematics import Unreachable, forward, inverse, min_singular_value
from panto.limits import JointLimitViolation, check_armable, format_limits_deg, q_deg
from panto.presets import apply_to_config, drive_defaults, resolve as resolve_preset
from panto.step_logic import parse_per_joint, to_per_joint
from panto.telemetry import RunLogger

MAX_FEEDBACK_AGE_S = 0.1

DIRECTIONS = {
    "+x": np.array([1.0, 0.0]),
    "-x": np.array([-1.0, 0.0]),
    "+y": np.array([0.0, 1.0]),
    "-y": np.array([0.0, -1.0]),
}


@dataclass
class DirResult:
    name: str
    settle_err_mm: float
    settle_cur: tuple
    peak_cur: tuple
    pose_end_mm: tuple
    pos_gain: tuple
    k_joint: tuple


class OscillationTripped(RuntimeError):
    pass


class ExcursionExceeded(OscillationTripped):
    """Measured tip moved farther than --max-excursion-mm from the startup
    pose. Subclasses OscillationTripped so every existing `except
    OscillationTripped` call site already aborts + relaxes on this too."""


class DriveDisarmed(OscillationTripped):
    """A node left CLOSED_LOOP_CONTROL mid-run (e.g. a protective trip like
    VELOCITY_LIMIT_VIOLATION). Subclasses OscillationTripped so every existing
    `except OscillationTripped` call site aborts on this too, but carries
    structured node/t/reason so main() can record it distinctly in the
    summary JSON. See run offset_sweep-20260904-202640: node 1 disarmed with
    VELOCITY_LIMIT_VIOLATION 2s into the approach ramp and the script kept
    ramping/settling for 6 more seconds, commanding position to a node that
    was no longer listening."""

    def __init__(self, msg: str, node_id: int, t: float, reason: str):
        super().__init__(msg)
        self.node_id = node_id
        self.t = t
        self.reason = reason


def _check_disarm(link: CanLink, t: float) -> None:
    """Raise DriveDisarmed the instant any node has left CLOSED_LOOP_CONTROL.
    Call this every tick, before commanding anything else -- a disarmed node
    ignores position/torque commands anyway, so there is nothing to gain by
    continuing to ramp or settle toward a target."""
    for s in link.node_status():
        if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
            reason = decode_error_flags(s.disarm_reason or 0)
            msg = (f"drive disarmed at t={t:.3f}s: node {s.node_id} axis_state "
                  f"{s.axis_state} (left CLOSED_LOOP) reason={reason}")
            raise DriveDisarmed(msg, node_id=s.node_id, t=t, reason=reason)


GUARD_GRACE_S = 0.5  # ignore the step-transient at the start of each _run_to call


def _run_to(link, backend, config, log, K, force_limit, target, seconds, rate_hz, tag,
           guard: OscillationGuard | None = None, collect: bool = False, guard_off: bool = False,
           excursion_ref=None, max_excursion_mm: float | None = None):
    """Drive toward `target` for `seconds`; return (final_err_m, final_cur, peak_cur, final_pose_m, last_sent).

    If `collect`, also returns (poses_mm, currents) sampled every tick, for
    the rest-dither metric. If `guard` is given (and not `guard_off`), checks
    it every tick (pose std / current-stall / current-buzz trip) and aborts
    with OscillationTripped; also aborts on stale feedback (>100ms). If
    `excursion_ref`/`max_excursion_mm` are given, aborts with
    ExcursionExceeded the instant the *measured* tip strays farther than that
    from `excursion_ref` (the fixed test_pose reference, NOT wherever this
    particular run started -- see the drift incident below) -- independent of
    what target was commanded, so a runaway can't walk past the excursion cap
    just because nothing checked the measured pose.

    2026-09-04: `--centre-here` re-centred each run on the *previous* run's
    end pose, and every failed -x direction left the elbow a little more
    folded; because each run's own excursion check was measured from its own
    (already-drifted) start, no single run ever saw a large excursion, and
    consecutive runs walked the arm into the fold limit / cable harness. The
    fix is structural, not a bigger margin: `excursion_ref` must be a fixed
    external reference (`config.test_pose`) shared across every invocation.
    """
    period = 1.0 / rate_hz
    constraint = Point(at=target)
    if guard is not None:
        # a fresh target means a step transient -- reset so the window doesn't
        # mix pre/post-step samples, which would look like oscillation.
        guard.reset()
    t0 = time.monotonic()
    t_end = t0 + seconds
    peak = np.zeros(2)
    err = 0.0
    cur = np.zeros(2)
    pose = np.zeros(2)
    sent: dict = {}
    poses_mm: list = []
    currents: list = []
    while time.monotonic() < t_end:
        age_s = link.feedback_age_s()
        if age_s > MAX_FEEDBACK_AGE_S:
            msg = f"feedback age {age_s*1e3:.1f}ms exceeds {MAX_FEEDBACK_AGE_S*1e3:.0f}ms"
            log.event(msg, level="ERROR")
            raise OscillationTripped(msg)

        try:
            _check_disarm(link, time.monotonic() - t0)
        except DriveDisarmed as exc:
            log.event(str(exc), level="ERROR")
            raise

        q, qd = link.joint_state()
        pose = forward(q, config.geo)

        if excursion_ref is not None and max_excursion_mm is not None:
            excursion_mm = float(np.linalg.norm(pose - excursion_ref) * 1e3)
            if excursion_mm > max_excursion_mm:
                msg = (f"measured tip {excursion_mm:.1f}mm from test_pose exceeds "
                      f"--max-excursion-mm={max_excursion_mm}")
                log.event(msg, level="ERROR")
                raise ExcursionExceeded(msg)

        proj = constraint.project(pose)
        cmd = ImpedanceCommand(pose=pose, q=q, anchor=proj.anchor, stiffness=K,
                               force_limit=force_limit)
        try:
            backend.apply(cmd)
        except JointLimitViolation as exc:
            log.event(f"joint limit violation ({exc})", level="ERROR")
            raise OscillationTripped(f"joint limit violation ({exc})")
        err = float(np.linalg.norm(proj.anchor - pose))
        cur = link.motor_currents()
        peak = np.maximum(peak, np.abs(cur))
        status = link.node_status()
        sent = backend.last_command or {}
        log.sample(tag=tag, q=q, pose=pose, anchor=proj.anchor, err_m=err,
                  currents=cur, node_status=status, sent=sent)

        if collect:
            poses_mm.append((pose * 1e3).tolist())
            currents.append(cur.tolist())

        if guard is not None and not guard_off:
            t = time.monotonic() - t0
            if t >= GUARD_GRACE_S:
                cap = sent.get("current_cap_a", [force_limit, force_limit])
                guard.push(t, pose * 1e3, cur, cap, err * 1e3)
                reason = guard.check()
                if reason is not None:
                    log.event(f"oscillation guard tripped ({reason})", level="ERROR")
                    raise OscillationTripped(reason)

        time.sleep(period)
    if collect:
        return err, cur, peak, pose, sent, poses_mm, currents
    return err, cur, peak, pose, sent


def _ramp_to(link, backend, config, log, K, force_limit, target, ramp_s, rate_hz, tag,
            tol_mm=3.0, hold_s=0.3, excursion_ref=None, max_excursion_mm=None):
    """Slowly move the anchor from the current measured pose to `target` over
    `ramp_s` seconds (linear interpolation, not a step), hold briefly, then
    verify the measured tip is within `tol_mm`. Returns (ok, final_pose_m,
    final_err_mm) -- never raises for a failed verify, only for a hard fault
    (stale feedback, joint limit); callers decide what a failed verify means.

    Used to approach/return to `config.test_pose` gently rather than snapping
    an anchor there in one tick -- a large instantaneous step is exactly the
    kind of thing the oscillation guard's grace period has to paper over, and
    for the one motion every run makes regardless of what's being swept
    (recentre and the final return-to-test_pose), gentle is worth the 2s.
    """
    q0, _ = link.joint_state()
    pose_start = forward(q0, config.geo)
    period = 1.0 / rate_hz
    t0 = time.monotonic()
    t_end = t0 + ramp_s + hold_s
    pose = pose_start.copy()
    while time.monotonic() < t_end:
        age_s = link.feedback_age_s()
        if age_s > MAX_FEEDBACK_AGE_S:
            msg = f"feedback age {age_s*1e3:.1f}ms exceeds {MAX_FEEDBACK_AGE_S*1e3:.0f}ms"
            log.event(msg, level="ERROR")
            raise OscillationTripped(msg)

        elapsed = time.monotonic() - t0
        try:
            _check_disarm(link, elapsed)
        except DriveDisarmed as exc:
            log.event(str(exc), level="ERROR")
            raise

        frac = min(1.0, elapsed / ramp_s) if ramp_s > 0 else 1.0
        anchor = pose_start + frac * (target - pose_start)

        q, qd = link.joint_state()
        pose = forward(q, config.geo)

        if excursion_ref is not None and max_excursion_mm is not None:
            excursion_mm = float(np.linalg.norm(pose - excursion_ref) * 1e3)
            if excursion_mm > max_excursion_mm:
                msg = (f"measured tip {excursion_mm:.1f}mm from test_pose exceeds "
                      f"--max-excursion-mm={max_excursion_mm} during ramp to {tag}")
                log.event(msg, level="ERROR")
                raise ExcursionExceeded(msg)

        cmd = ImpedanceCommand(pose=pose, q=q, anchor=anchor, stiffness=K, force_limit=force_limit)
        try:
            backend.apply(cmd)
        except JointLimitViolation as exc:
            log.event(f"joint limit violation ({exc})", level="ERROR")
            raise OscillationTripped(f"joint limit violation ({exc})")
        cur = link.motor_currents()
        status = link.node_status()
        sent = backend.last_command or {}
        err = float(np.linalg.norm(anchor - pose))
        log.sample(tag=tag, q=q, pose=pose, anchor=anchor, err_m=err,
                  currents=cur, node_status=status, sent=sent)
        time.sleep(period)

    final_err_mm = float(np.linalg.norm(target - pose) * 1e3)
    ok = final_err_mm <= tol_mm
    return ok, pose, final_err_mm


def main() -> None:
    p = argparse.ArgumentParser(prog="offset_sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--preset", type=str, default=None,
                   help="named preset from presets.json; supplies stiffness/current/vel-gain/"
                        "vel-limit/max-pos-gain not given explicitly below")
    p.add_argument("--current", type=float, default=None, help="per-axis current cap, A (default 0.4 or preset)")
    p.add_argument("--stiffness", type=float, default=None,
                   help="isotropic EE stiffness, N/m (default 200.0 or preset)")
    p.add_argument("--step", type=float, default=15.0, help="offset magnitude, mm")
    p.add_argument("--settle", type=float, default=3.0, help="seconds to settle at each offset")
    p.add_argument("--return-settle", type=float, default=2.0,
                   help="seconds to re-settle at the start pose between directions")
    p.add_argument("--rate", type=float, default=100.0, help="control loop Hz")
    p.add_argument("--vel-gain", type=str, default=None,
                   help="override motor.vel_gain: 'V' (both joints) or 'V0,V1' (shoulder,elbow)")
    p.add_argument("--max-pos-gain", type=float, default=None,
                   help="override motor.max_pos_gain (config clamp) for this run")
    p.add_argument("--vel-limit", type=float, default=None,
                   help="ODrive vel_limit, joint rad/s (default: config vel_limit)")
    p.add_argument("--clear-errors", dest="clear_errors", action="store_true", default=True,
                   help="send Clear_Errors to both nodes right after arming (default on)")
    p.add_argument("--no-clear-errors", dest="clear_errors", action="store_false")
    p.add_argument("--osc-mm", type=float, default=3.0,
                   help="oscillation guard: trip if pose std over 0.5s exceeds this, mm")
    p.add_argument("--guard-off", action="store_true", help="disable the oscillation guard (diagnostics only)")
    p.add_argument("--centre-here", dest="centre_here", action="store_true", default=False,
                   help="explicit opt-in: use the measured pose at startup as the sweep centre, "
                        "no move. Default is OFF -- the default centre is config.test_pose "
                        "(calibration.json), approached with a slow ramp + verify. "
                        "2026-09-04: --centre-here as the default let repeated runs drift "
                        "toward a mechanical/harness limit because each run's own excursion "
                        "check was measured from its own already-drifted start.")
    p.add_argument("--centre-r", type=float, default=175.0,
                   help="explicit sweep centre (only used with --no-centre-here, no test_pose): "
                        "radius from origin, mm")
    p.add_argument("--centre-angle", type=float, default=None,
                   help="explicit sweep centre (only used with --no-centre-here, no test_pose): "
                        "polar angle, deg (default: current pose's angle)")
    p.add_argument("--max-excursion-mm", type=float, default=50.0,
                   help="refuse any centre/offset target, and abort at runtime, farther than this "
                        "from config.test_pose (never from wherever this run happened to start), mm")
    p.add_argument("--ramp-s", type=float, default=2.0,
                   help="seconds to ramp the anchor to/from test_pose at start/end of the run")
    p.add_argument("--offset-ramp-s", type=float, default=0.5,
                   help="seconds to ramp the anchor for each +/-x/y offset move and each "
                        "return-to-centre (was an instantaneous step -- a 15mm anchor jump "
                        "throws the loop outside its linear band and triggers a saturated "
                        "limit cycle). The settle timer (--settle/--return-settle) starts "
                        "after this ramp completes, not from the step.")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    resolved = resolve_preset(
        {"stiffness": args.stiffness,
         "vel_gain": parse_per_joint(args.vel_gain) if args.vel_gain is not None else None,
         "vel_limit": args.vel_limit,
         "current": args.current, "max_pos_gain": args.max_pos_gain},
        args.preset,
    )
    resolved["current"] = 0.4 if resolved["current"] is None else resolved["current"]
    resolved["stiffness"] = 200.0 if resolved["stiffness"] is None else resolved["stiffness"]
    if resolved["vel_gain"] is not None:
        resolved["vel_gain"] = list(to_per_joint(resolved["vel_gain"]))

    default_vel_gain = drive_defaults(config)
    vel_limit = resolved["vel_limit"] if resolved["vel_limit"] is not None else config.motors[0].vel_limit
    for m in config.motors:
        m.current_soft_max = resolved["current"]
    apply_to_config(config, resolved)

    K = resolved["stiffness"] * np.eye(2)
    force_limit = 5.0  # current_soft_max is the real limit; see point_hold.py

    log = RunLogger("offset_sweep", interface=config.can.interface, channel=config.can.channel,
                    current=resolved["current"], stiffness=resolved["stiffness"], step_mm=args.step,
                    settle=args.settle, return_settle=args.return_settle, rate=args.rate,
                    vel_gain=(resolved["vel_gain"] if resolved["vel_gain"] is not None
                             else [default_vel_gain[m.node_id] for m in config.motors]),
                    max_pos_gain=resolved["max_pos_gain"], vel_limit=vel_limit,
                    clear_errors=args.clear_errors, osc_mm=args.osc_mm, guard_off=args.guard_off,
                    centre_here=args.centre_here, centre_r_mm=args.centre_r,
                    centre_angle_deg=args.centre_angle, max_excursion_mm=args.max_excursion_mm,
                    ramp_s=args.ramp_s, offset_ramp_s=args.offset_ramp_s,
                    test_pose_configured=config.test_pose_xy_m is not None,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=False)
    backend = PositionBackend(link, config)
    backend.vel_limit_rad_s = vel_limit
    guard = OscillationGuard(window_s=0.5, osc_mm=args.osc_mm, current_frac=0.9)
    CENTRE_TOL_MM = 3.0
    print(f"opening {config.can.interface}/{config.can.channel}  "
          f"(cap {resolved['current']} A, K {resolved['stiffness']} N/m, "
          f"vel_gain {[m.vel_gain for m in config.motors]}, "
          f"vel_limit {vel_limit}, step {args.step} mm)")
    link.start()

    def idle_all() -> None:
        for m in config.motors:
            try:
                link.set_idle(m.node_id)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {m.node_id}: {exc}", level="ERROR")

    def restore_vel_gain() -> None:
        if resolved["vel_gain"] is None:
            return
        for m in config.motors:
            try:
                link.set_vel_gains(m.node_id, default_vel_gain[m.node_id], 0.0)
                log.event(f"restored node {m.node_id} vel_gain -> {default_vel_gain[m.node_id]}")
            except Exception as exc:  # noqa: BLE001
                log.event(f"restore vel_gain node {m.node_id}: {exc}", level="ERROR")

    results: list[DirResult] = []
    armed = False
    run_start_pose = None    # measured pose at script start, before any move
    end_pose = None          # measured pose right before teardown
    excursion_ref = None     # fixed reference for --max-excursion-mm (test_pose, once known)
    test_pose_xy = config.test_pose_xy_m   # None unless calibration.json sets it
    disarm_info = None       # set on DriveDisarmed: {"node": id, "t": s, "reason": str}
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        run_start_pose = pose0.copy()
        sig0 = min_singular_value(q0, config.geo)
        pos_gain0 = PositionBackend.pos_gains_for(K, q0, config)
        print(f"  start pose=({pose0[0]*1e3:.1f}, {pose0[1]*1e3:.1f})mm  sigma_min={sig0:.4f}  "
              f"pos_gain@start={[round(g, 1) for g in pos_gain0]}  "
              f"limits={format_limits_deg(config.motors)}")
        log.event(f"start q_deg={q_deg(q0)} limits={format_limits_deg(config.motors)}")
        if test_pose_xy is None and not args.centre_here:
            msg = ("no test_pose configured in calibration.json -- refusing to guess a centre. "
                  "Either add calibration.json's test_pose, or pass --centre-here to "
                  "explicitly opt into centring on wherever the arm happens to be right now.")
            print(f"\n! {msg}")
            log.event(msg, level="ERROR")
            raise SystemExit(1)

        backend.enter()
        for i, m in enumerate(config.motors):
            link.set_input_pos(m.node_id, float(q0[i]))
        print(">>> entering CLOSED_LOOP_CONTROL <<<")
        log.event(">>> entering CLOSED_LOOP_CONTROL <<<")
        try:
            link.enter_closed_loop(timeout=5.0)
        except CanLinkError as exc:
            print(f"\n! refusing to arm: {exc}")
            log.event(f"refusing to arm: {exc}", level="ERROR")
            raise SystemExit(1)
        armed = True
        if args.clear_errors:
            for m in config.motors:
                link.clear_errors(m.node_id)
            log.event("cleared errors on both nodes")
        if resolved["vel_gain"] is not None:
            for i, m in enumerate(config.motors):
                link.set_vel_gains(m.node_id, resolved["vel_gain"][i], 0.0)
            log.event(f"set vel_gain -> {resolved['vel_gain']} (per node)")

        # excursion_ref is the ONE fixed reference the excursion cap is always
        # measured against -- config.test_pose when available, regardless of
        # how many prior invocations happened or where each of them ended up.
        # Falling back to this run's own start pose only happens in the
        # explicit --centre-here diagnostic path (no test_pose requirement),
        # and even then is logged loudly since it reintroduces the drift risk
        # described in _run_to's docstring.
        if test_pose_xy is not None:
            excursion_ref = test_pose_xy.copy()
        else:
            excursion_ref = pose0.copy()
            log.event("no test_pose configured -- excursion cap measured from THIS run's own "
                     "start pose, not a fixed reference (drift risk across repeated runs)",
                     level="WARN")

        if args.centre_here:
            # No move: sweep around wherever the arm is actually sitting right
            # now. Explicit opt-in only -- see the --centre-here help text.
            centre = pose0.copy()
            msg = f"centre-here: using startup pose as centre, no move -- ({centre[0]*1e3:.1f},{centre[1]*1e3:.1f})mm"
            print(f"  {msg}")
            log.event(msg)
        elif test_pose_xy is not None:
            centre = test_pose_xy.copy()
            msg = f"centre = config.test_pose -- ({centre[0]*1e3:.1f},{centre[1]*1e3:.1f})mm"
            print(f"  {msg}")
            log.event(msg)
        else:
            # Legacy explicit r/angle path -- only reachable if test_pose_xy is
            # None, which the pre-flight check above already refused unless
            # --centre-here was also passed (handled above). Kept for the
            # (rare) case a caller wants a specific r/angle even with a
            # test_pose configured -- not currently reachable via argparse
            # without also editing this branch's guard, intentionally, so it
            # doesn't silently override test_pose.
            centre_angle_deg = args.centre_angle if args.centre_angle is not None else \
                float(np.degrees(np.arctan2(pose0[1], pose0[0])))
            centre_angle_rad = np.radians(centre_angle_deg)
            centre = (args.centre_r * 1e-3) * np.array(
                [np.cos(centre_angle_rad), np.sin(centre_angle_rad)])
            msg = (f"moving to explicit centre r={args.centre_r:.1f}mm angle={centre_angle_deg:.1f}deg "
                  f"-> ({centre[0]*1e3:.1f},{centre[1]*1e3:.1f})mm")
            print(f"  {msg}")
            log.event(msg)

        # Pre-flight: IK the centre and every +/-step offset *before* moving
        # anywhere, and check each against --max-excursion-mm from
        # excursion_ref (test_pose, not wherever this run started).
        preflight_targets = {"centre": centre}
        for name, unit in DIRECTIONS.items():
            preflight_targets[name] = centre + unit * (args.step * 1e-3)
        preflight_problems = []
        for name, target in preflight_targets.items():
            excursion_mm = float(np.linalg.norm(target - excursion_ref) * 1e3)
            if excursion_mm > args.max_excursion_mm:
                preflight_problems.append(
                    f"{name} ({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm is {excursion_mm:.1f}mm "
                    f"from test_pose, exceeds --max-excursion-mm={args.max_excursion_mm}"
                )
                continue
            try:
                q_t = inverse(target, config.geo, elbow=backend.elbow or config.elbow)
            except Unreachable as exc:
                preflight_problems.append(f"{name} ({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm "
                                          f"unreachable: {exc}")
                continue
            problems = check_armable(q_t, config.motors)
            if problems:
                preflight_problems.append(
                    f"{name} ({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm out of limits: "
                    + "; ".join(problems)
                )
        if preflight_problems:
            err_msg = ("refusing to start sweep -- centre/offset targets out of range:\n  "
                      + "\n  ".join(preflight_problems))
            print(f"\n! {err_msg}")
            log.event(err_msg, level="ERROR")
            raise SystemExit(1)

        # Approach the centre gently (ramp, not a step) and verify within
        # CENTRE_TOL_MM before touching dither/offsets -- unconditionally
        # unless --centre-here (where centre == current pose by definition).
        if not args.centre_here:
            dist_to_centre_mm = float(np.linalg.norm(pose0 - centre) * 1e3)
            if dist_to_centre_mm > CENTRE_TOL_MM:
                msg = f"{dist_to_centre_mm:.1f}mm from centre -- ramping over {args.ramp_s}s"
                print(f"  {msg}")
                log.event(msg)
                ok, pose0, final_err_mm = _ramp_to(
                    link, backend, config, log, K, force_limit, centre, args.ramp_s, args.rate,
                    "approach_centre", tol_mm=CENTRE_TOL_MM,
                    excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
                if not ok:
                    msg = (f"failed to reach centre: {final_err_mm:.2f}mm from target "
                          f"(tolerance {CENTRE_TOL_MM}mm) after {args.ramp_s}s ramp -- "
                          f"refusing to sweep from an unconverged/wrong centre")
                    log.event(msg, level="ERROR")
                    raise SystemExit(msg)

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        sig0 = min_singular_value(q0, config.geo)
        centre_off_mm = float(np.linalg.norm(pose0 - centre) * 1e3)
        print(f"  centre pose=({pose0[0]*1e3:.1f}, {pose0[1]*1e3:.1f})mm  "
              f"target=({centre[0]*1e3:.1f},{centre[1]*1e3:.1f})mm  "
              f"off_by={centre_off_mm:.2f}mm  sigma_min={sig0:.4f}")
        log.event(f"centre: pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm "
                 f"off_by={centre_off_mm:.2f}mm sigma_min={sig0:.4f}")
        if centre_off_mm > CENTRE_TOL_MM:
            msg = (f"centring failed: {centre_off_mm:.2f}mm from target centre "
                  f"(tolerance {CENTRE_TOL_MM}mm) -- refusing to sweep from an "
                  f"unconverged/wrong centre")
            log.event(msg, level="ERROR")
            raise SystemExit(msg)

        print(f"    closed loop. Sweeping {list(DIRECTIONS)}\n")

        # settle at the centre
        _run_to(link, backend, config, log, K, force_limit, pose0, 1.0, args.rate, "center0",
               guard=guard, guard_off=args.guard_off,
               excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)

        # rest-dither metric: 2s at centre, RMS current + pose std
        _, _, _, _, _, dither_poses, dither_currents = _run_to(
            link, backend, config, log, K, force_limit, pose0, 2.0, args.rate, "dither",
            guard=guard, guard_off=args.guard_off, collect=True,
            excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
        dither_poses_a = np.array(dither_poses)
        dither_cur_a = np.array(dither_currents)
        i_rms = np.sqrt(np.mean(dither_cur_a ** 2, axis=0)) if len(dither_cur_a) else np.zeros(2)
        pose_std = dither_poses_a.std(axis=0) if len(dither_poses_a) else np.zeros(2)
        print(f"  dither: i_rms=[{i_rms[0]:.4f},{i_rms[1]:.4f}] A  "
              f"pose_std=[{pose_std[0]:.3f},{pose_std[1]:.3f}] mm")
        log.event(f"dither: i_rms=[{i_rms[0]:.4f},{i_rms[1]:.4f}] A "
                 f"pose_std=[{pose_std[0]:.3f},{pose_std[1]:.3f}] mm")

        guard_tripped = False
        trip_reason = None
        tracked_fraction = {}
        for name, unit in DIRECTIONS.items():
            target = pose0 + unit * (args.step * 1e-3)
            log.event(f"-> {name}: target=({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm "
                     f"(ramp {args.offset_ramp_s}s)")
            try:
                # Ramp the anchor to the offset over --offset-ramp-s (not an
                # instantaneous step -- a 15mm jump throws the loop outside
                # its linear band and triggers a saturated limit cycle). The
                # settle timer starts only after this ramp completes: no
                # hold_s here (0.0), the following _run_to's --settle is the
                # real dwell.
                _ramp_to(link, backend, config, log, K, force_limit, target,
                        args.offset_ramp_s, args.rate, f"ramp_offset_{name}", hold_s=0.0,
                        excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
                err, cur, peak, pose_end, sent = _run_to(
                    link, backend, config, log, K, force_limit, target,
                    args.settle, args.rate, f"offset_{name}", guard=guard, guard_off=args.guard_off,
                    excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
            except Unreachable as exc:
                print(f"  {name:>3}  UNREACHABLE from this pose ({exc}) -- skipped")
                log.event(f"{name} unreachable: {exc}", level="WARN")
                continue
            except DriveDisarmed as exc:
                guard_tripped = True
                trip_reason = str(exc)
                disarm_info = {"node": exc.node_id, "t": exc.t, "reason": exc.reason}
                print(f"  {name:>3}  DRIVE DISARMED ({exc}) -- aborting sweep, no further "
                      f"ramping/settling")
                break
            except ExcursionExceeded as exc:
                guard_tripped = True
                trip_reason = str(exc)
                print(f"  {name:>3}  EXCURSION CAP EXCEEDED ({exc}) -- aborting sweep")
                break
            except OscillationTripped as exc:
                guard_tripped = True
                trip_reason = str(exc)
                print(f"  {name:>3}  OSCILLATION GUARD TRIPPED ({exc}) -- aborting sweep")
                break
            pos_gain = tuple(sent.get("pos_gain", [None, None]))
            k_joint = tuple(m.vel_gain * pg if pg is not None else None
                            for m, pg in zip(config.motors, pos_gain))
            moved_mm = float(np.dot((pose_end - pose0) * 1e3, unit))
            tracked_fraction[name] = moved_mm / args.step if args.step else None
            results.append(DirResult(name, err * 1e3, tuple(cur), tuple(peak),
                                     tuple((pose_end * 1e3).tolist()), pos_gain, k_joint))
            print(f"  {name:>3}  settle_err={err*1e3:6.2f}mm  moved={moved_mm:6.2f}mm "
                  f"(tracked={tracked_fraction[name]:.2f})  "
                  f"cur=({cur[0]:+.3f},{cur[1]:+.3f})A  peak=({peak[0]:.3f},{peak[1]:.3f})A  "
                  f"pos_gain=({pos_gain[0]:.0f},{pos_gain[1]:.0f})")
            # return to centre before the next direction -- same ramp-then-
            # settle shape as the offset move above, not an instantaneous
            # step back.
            try:
                _ramp_to(link, backend, config, log, K, force_limit, pose0,
                        args.offset_ramp_s, args.rate, f"ramp_return_{name}", hold_s=0.0,
                        excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
                _run_to(link, backend, config, log, K, force_limit, pose0,
                        args.return_settle, args.rate, f"return_{name}", guard=guard,
                        guard_off=args.guard_off,
                        excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
            except DriveDisarmed as exc:
                guard_tripped = True
                trip_reason = str(exc)
                disarm_info = {"node": exc.node_id, "t": exc.t, "reason": exc.reason}
                print(f"  ! DRIVE DISARMED on return ({exc}) -- aborting sweep, no further "
                      f"ramping/settling")
                break
            except ExcursionExceeded as exc:
                guard_tripped = True
                trip_reason = str(exc)
                print(f"  ! EXCURSION CAP EXCEEDED on return ({exc}) -- aborting sweep")
                break
            except OscillationTripped as exc:
                guard_tripped = True
                trip_reason = str(exc)
                print(f"  ! OSCILLATION GUARD TRIPPED on return ({exc}) -- aborting sweep")
                break

        if not results and not guard_tripped:
            raise SystemExit("every direction was unreachable from this pose")

        q_end, _ = link.joint_state()
        end_pose = forward(q_end, config.geo)

        by_name = {r.name: r for r in results}
        avg_err = np.mean([r.settle_err_mm for r in results]) if results else float("nan")
        avg_cur_mag = np.mean([np.linalg.norm(r.settle_cur) for r in results]) if results else float("nan")
        peak_currents = (np.max([r.peak_cur for r in results], axis=0).tolist()
                        if results else [None, None])
        if results:
            print(f"\n  avg settle_err = {avg_err:.2f} mm   avg |current| = {avg_cur_mag:.3f} A  "
                  f"(n={len(results)}/4 directions)")
        symmetry = {}
        for a, b in (("+x", "-x"), ("+y", "-y")):
            if a in by_name and b in by_name:
                asym = by_name[a].settle_err_mm - by_name[b].settle_err_mm
                symmetry[f"{a}-{b}"] = asym
                print(f"  {a} - {b} = {asym:+.2f} mm  (symmetry check)")
                log.event(f"symmetry {a}-{b} = {asym:.3f}mm")
        log.event(f"summary: avg_err_mm={avg_err:.3f} avg_cur_a={avg_cur_mag:.3f} "
                 f"n={len(results)}/4")

        summary = {
            "vel_gain": (resolved["vel_gain"] if resolved["vel_gain"] is not None
                        else [default_vel_gain[m.node_id] for m in config.motors]),
            "max_pos_gain": resolved["max_pos_gain"] if resolved["max_pos_gain"] is not None
                else config.motors[0].max_pos_gain,
            "stiffness": resolved["stiffness"],
            "current_cap_a": resolved["current"],
            "vel_limit": vel_limit,
            "offset_ramp_s": args.offset_ramp_s,
            "centre_here": args.centre_here,
            "max_excursion_mm": args.max_excursion_mm,
            "test_pose_mm": None if test_pose_xy is None else (test_pose_xy * 1e3).tolist(),
            "start_pose_mm": (run_start_pose * 1e3).tolist(),
            "end_pose_mm": (end_pose * 1e3).tolist(),
            "centre_mm": (pose0 * 1e3).tolist(),
            "centre_off_mm": centre_off_mm,
            "sigma_min_at_centre": sig0,
            "residual_mm": {r.name: r.settle_err_mm for r in results},
            "mean_residual_mm": None if np.isnan(avg_err) else float(avg_err),
            "symmetry_mm": symmetry,
            "tracked_fraction": tracked_fraction,
            "pos_gain_sent": {r.name: r.pos_gain for r in results},
            "k_joint_nm_rad": {r.name: r.k_joint for r in results},
            "dither_i_rms_a": i_rms.tolist(),
            "dither_pose_std_mm": pose_std.tolist(),
            "peak_current_a": peak_currents,
            "guard_tripped": guard_tripped,
            "guard_trip_reason": trip_reason,
            "disarmed": disarm_info,
        }
        print("\nSUMMARY_JSON " + json.dumps(summary))
        log.event("summary_json " + json.dumps(summary))
        if guard_tripped:
            raise SystemExit(f"oscillation guard tripped ({trip_reason})")
    except DriveDisarmed as exc:
        # Raised from a call site with no per-direction try/except of its own
        # (centre approach, center0 settle, dither) -- the per-direction loop
        # catches DriveDisarmed itself and falls through to the full summary
        # above instead. Either way: stop immediately, no further
        # ramping/settling, record disarmed in a summary JSON, exit non-zero.
        disarm_info = {"node": exc.node_id, "t": exc.t, "reason": exc.reason}
        print(f"\n! {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
        summary = {"disarmed": disarm_info, "aborted_before_sweep": True}
        print("\nSUMMARY_JSON " + json.dumps(summary))
        log.event("summary_json " + json.dumps(summary))
    except KeyboardInterrupt:
        print("\n! interrupted")
        log.event("interrupted", level="WARN")
    except Exception as exc:  # noqa: BLE001
        print(f"\n! {type(exc).__name__}: {exc}")
        log.event(f"{type(exc).__name__}: {exc}", level="ERROR")
    finally:
        # Return to test_pose (same slow ramp + verify) BEFORE relax/IDLE, so
        # the arm never drifts across invocations -- this is the other half
        # of the fix for the 2026-09-04 drift-into-the-fold-limit incident.
        # Only attempted if we actually armed (nothing to ramp from
        # otherwise) and test_pose is configured. A failed return is logged
        # as ERROR but never blocks relax/IDLE -- getting the arm off-torque
        # always wins over getting it back to a tidy pose.
        if disarm_info is not None:
            msg = (f"skipping return-to-test_pose -- node {disarm_info['node']} is disarmed "
                  f"(reason={disarm_info['reason']}); no further ramping/settling after a disarm")
            print(f"\n! {msg}")
            log.event(msg, level="ERROR")
        elif armed and test_pose_xy is not None:
            print(f"\nreturning to test_pose before IDLE (ramp {args.ramp_s}s)...")
            try:
                ok, final_pose, final_err_mm = _ramp_to(
                    link, backend, config, log, K, force_limit, test_pose_xy, args.ramp_s,
                    args.rate, "return_test_pose", tol_mm=CENTRE_TOL_MM,
                    excursion_ref=excursion_ref, max_excursion_mm=args.max_excursion_mm)
                if ok:
                    msg = f"returned to test_pose, {final_err_mm:.2f}mm off"
                    print(f"  {msg}")
                    log.event(msg)
                else:
                    msg = (f"FAILED to return to test_pose: {final_err_mm:.2f}mm off "
                          f"(tolerance {CENTRE_TOL_MM}mm) after {args.ramp_s}s ramp -- "
                          f"arm is NOT at test_pose, leaving IDLE anyway")
                    print(f"  ! {msg}")
                    log.event(msg, level="ERROR")
            except Exception as exc:  # noqa: BLE001
                msg = f"return-to-test_pose failed: {type(exc).__name__}: {exc} -- leaving IDLE anyway"
                print(f"  ! {msg}")
                log.event(msg, level="ERROR")
        elif armed and test_pose_xy is None:
            log.event("no test_pose configured -- cannot return to it before IDLE "
                     "(arm will be left wherever this run's last move put it)",
                     level="WARN")

        print("\nrelax + IDLE")
        try:
            backend.relax()
        except Exception:  # noqa: BLE001
            pass
        restore_vel_gain()
        idle_all()
        time.sleep(0.2)
        link.close()
        log.close()
        print(f"closed. log: {log.dir}")

    if disarm_info is not None:
        raise SystemExit(f"drive disarmed: node {disarm_info['node']} "
                         f"reason={disarm_info['reason']} at t={disarm_info['t']:.3f}s")


if __name__ == "__main__":
    main()
