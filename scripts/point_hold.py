"""Closed-loop Cartesian point-hold via the real PositionBackend — step 3a.

    python -m scripts.point_hold                        # hold current pose, 8 s
    python -m scripts.point_hold --step 15              # then jog the anchor +15 mm in x
    python -m scripts.point_hold --current 0.3 --stiffness 400

Exercises the actual `PositionBackend` + `Point` constraint against hardware, in
a bare ~100 Hz loop (no Runtime, no web). Gentle by default: 0.25 A per-axis cap.

  1. read current joint state -> FK pose -> anchor a `Point` there
  2. backend.enter() (position mode + current cap), enter CLOSED_LOOP_CONTROL
  3. loop: project(pose) -> ImpedanceCommand -> backend.apply(); print periodically
  4. optional: after half the hold, jog the anchor by --step mm in +x
  5. backend.relax(), both axes -> IDLE

Any exception / Ctrl-C -> relax + IDLE + close. Push the arm by hand during the
hold to feel the spring; it should pull back toward the anchor.

NOTE: pos_gain is derived as k_rad*2pi / motor.vel_gain, using the *config*
vel_gain (default 2.5e-3, the sane not-buzzy start on EM3215). If the drives'
stored vel_gain differs, the rendered stiffness scales by that ratio — pass
--vel-gain to match, or nail it in the milestone-3 vel_gain sweep. The 0.25 A
cap bounds it either way. Also watch max_pos_gain (500): with vel_gain 2.5e-3 a
modest k_rad already saturates it, so effective stiffness may clamp — raise it
in a config.local.json if the hold is still soft at the current cap.

Every control tick is logged (not just the printed ones) via `panto.telemetry
.RunLogger`, including per-node `axis_state`/`active_errors`/`disarm_reason` —
so a fault or oscillation that happens between printouts is still on disk. This
is the first slice of the "architect a telemetry/logging stack" TODO; see the
Notion progress page.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from panto.backends import PositionBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.constraints import Point
from panto.guard import OscillationGuard
from panto.kinematics import forward, min_singular_value
from panto.limits import JointLimitViolation, format_limits_deg, q_deg
from panto.presets import apply_to_config, drive_defaults, resolve as resolve_preset
from panto.step_logic import parse_per_joint, to_per_joint
from panto.telemetry import RunLogger

MAX_FEEDBACK_AGE_S = 0.1
GUARD_GRACE_S = 0.5  # ignore the step-transient at the start of a hold/after a step


class DriveDisarmed(RuntimeError):
    """A node left CLOSED_LOOP_CONTROL mid-hold (e.g. a protective trip). See
    torque_step.py/breakaway.py's DriveDisarmed and the 2026-09-04 incident
    (offset_sweep-20260904-202640: a disarmed node kept being commanded for
    several more seconds because nothing checked axis_state) -- same fix
    here: check every tick, stop immediately, don't keep commanding a node
    that isn't listening."""

    def __init__(self, msg: str, node_id: int, t: float, reason: str):
        super().__init__(msg)
        self.node_id = node_id
        self.t = t
        self.reason = reason


def main() -> None:
    p = argparse.ArgumentParser(prog="point_hold", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--preset", type=str, default="hover-K25-pj",
                   help="named preset from presets.json; supplies stiffness/current/vel-gain/"
                        "vel-limit/max-pos-gain not given explicitly below")
    p.add_argument("--current", type=float, default=None, help="per-axis current cap, A (default 0.25 or preset)")
    p.add_argument("--stiffness", type=float, default=None,
                   help="isotropic EE stiffness, N/m (default 300.0 or preset)")
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
    p.add_argument("--hold", type=float, default=8.0, help="seconds")
    p.add_argument("--step", type=float, default=0.0, help="jog anchor +x by this many mm at hold/2")
    p.add_argument("--rate", type=float, default=100.0, help="control loop Hz")
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
    resolved["current"] = 0.25 if resolved["current"] is None else resolved["current"]
    resolved["stiffness"] = 300.0 if resolved["stiffness"] is None else resolved["stiffness"]
    vel_gain_per_joint = to_per_joint(resolved["vel_gain"]) if resolved["vel_gain"] is not None else None
    if vel_gain_per_joint is not None:
        resolved["vel_gain"] = list(vel_gain_per_joint)

    default_vel_gain = drive_defaults(config)
    vel_limit = resolved["vel_limit"] if resolved["vel_limit"] is not None else config.motors[0].vel_limit
    for m in config.motors:
        m.current_soft_max = resolved["current"]
    apply_to_config(config, resolved)

    nodes = [m.node_id for m in config.motors]
    K = resolved["stiffness"] * np.eye(2)

    log = RunLogger("point_hold", interface=config.can.interface, channel=config.can.channel,
                    preset=args.preset,
                    current=resolved["current"], stiffness=resolved["stiffness"],
                    vel_gain=resolved["vel_gain"], max_pos_gain=resolved["max_pos_gain"],
                    vel_limit=vel_limit,
                    clear_errors=args.clear_errors, osc_mm=args.osc_mm, guard_off=args.guard_off,
                    hold=args.hold, step=args.step, rate=args.rate,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=False)
    backend = PositionBackend(link, config)
    backend.vel_limit_rad_s = vel_limit
    guard = OscillationGuard(window_s=0.5, osc_mm=args.osc_mm, current_frac=0.9)
    print(f"opening {config.can.interface}/{config.can.channel}  preset={args.preset}  "
          f"(cap {resolved['current']} A, K {resolved['stiffness']} N/m, "
          f"vel_gain {[m.vel_gain for m in config.motors]}, "
          f"vel_limit {vel_limit})")
    link.start()

    def idle_all() -> None:
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {nid}: {exc}", level="ERROR")

    def restore_vel_gain() -> None:
        if resolved["vel_gain"] is None:
            return
        for nid in nodes:
            try:
                link.set_vel_gains(nid, default_vel_gain[nid], 0.0)
                log.event(f"restored node {nid} vel_gain -> {default_vel_gain[nid]}")
            except Exception as exc:  # noqa: BLE001
                log.event(f"restore vel_gain node {nid}: {exc}", level="ERROR")

    disarm_info = None  # set on DriveDisarmed: {"node": id, "t": s, "reason": str}
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        pos_gain0 = PositionBackend.pos_gains_for(K, q0, config)
        print(f"  start q=({np.degrees(q0[0]):.2f}, {np.degrees(q0[1]):.2f})deg  "
              f"pose=({pose0[0]*1e3:.1f}, {pose0[1]*1e3:.1f})mm  "
              f"pos_gain@start={[round(g, 1) for g in pos_gain0]}  "
              f"limits={format_limits_deg(config.motors)}")
        log.event(f"start q_deg={q_deg(q0)} limits={format_limits_deg(config.motors)}")
        constraint = Point(at=pose0.copy())

        backend.enter()
        for i, nid in enumerate(nodes):
            link.set_input_pos(nid, float(q0[i]))
        print(">>> entering CLOSED_LOOP_CONTROL <<<")
        log.event(">>> entering CLOSED_LOOP_CONTROL <<<")
        try:
            link.enter_closed_loop(timeout=5.0)
        except CanLinkError as exc:
            print(f"\n! refusing to arm: {exc}")
            log.event(f"refusing to arm: {exc}", level="ERROR")
            raise SystemExit(1)
        if args.clear_errors:
            for nid in nodes:
                link.clear_errors(nid)
            log.event("cleared errors on both nodes")
        if resolved["vel_gain"] is not None:
            for i, nid in enumerate(nodes):
                link.set_vel_gains(nid, resolved["vel_gain"][i], 0.0)
            log.event(f"set vel_gain -> {resolved['vel_gain']} (per node)")
        print("    closed loop; holding\n")

        period = 1.0 / args.rate
        t0 = time.monotonic()
        stepped = False
        next_print = 0.0
        while True:
            t = time.monotonic() - t0
            if t >= args.hold:
                break

            age_s = link.feedback_age_s()
            if age_s > MAX_FEEDBACK_AGE_S:
                msg = f"feedback age {age_s*1e3:.1f}ms exceeds {MAX_FEEDBACK_AGE_S*1e3:.0f}ms"
                log.event(msg, level="ERROR")
                raise SystemExit(msg)

            for s in link.node_status():
                if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    reason = decode_error_flags(s.disarm_reason or 0)
                    msg = (f"drive disarmed at t={t:.3f}s: node {s.node_id} axis_state "
                          f"{s.axis_state} (left CLOSED_LOOP) reason={reason}")
                    log.event(msg, level="ERROR")
                    raise DriveDisarmed(msg, node_id=s.node_id, t=t, reason=reason)

            if args.step and not stepped and t >= args.hold / 2:
                tgt = pose0 + np.array([args.step * 1e-3, 0.0])
                constraint = Point(at=tgt)
                stepped = True
                step_t = t  # start of step transient -- give the guard a grace period
                guard.reset()
                msg = f"anchor -> ({tgt[0]*1e3:.1f}, {tgt[1]*1e3:.1f})mm"
                print(f"  -- {msg} --")
                log.event(msg)

            q, qd = link.joint_state()
            pose = forward(q, config.geo)
            proj = constraint.project(pose)
            cmd = ImpedanceCommand(pose=pose, q=q, anchor=proj.anchor,
                                   stiffness=K, force_limit=5.0)
            try:
                backend.apply(cmd)
            except JointLimitViolation as exc:
                log.event(f"joint limit violation ({exc})", level="ERROR")
                raise SystemExit(f"joint limit violation ({exc})")

            err = float(np.linalg.norm(proj.anchor - pose))
            cur = link.motor_currents()
            sig = min_singular_value(q, config.geo)
            status = link.node_status()
            sent = backend.last_command or {}

            cap = sent.get("current_cap_a", [resolved["current"], resolved["current"]])
            if not args.guard_off and t - (step_t if stepped else 0.0) >= GUARD_GRACE_S:
                guard.push(t, pose * 1e3, cur, cap, err * 1e3)
                reason = guard.check()
                if reason is not None:
                    log.event(f"oscillation guard tripped ({reason})", level="ERROR")
                    raise SystemExit(f"oscillation guard tripped ({reason})")
            # every tick, not just printed ones -- a fault between prints is
            # still on disk, not just visible on the LEDs for a few ms. Also log
            # what we *commanded* (pos_gain, current cap actually sent), not just
            # what we measured -- "it didn't move" is undiagnosable from the
            # measured side alone.
            log.sample(t=t, q=q, qd=qd, pose=pose, anchor=proj.anchor, err_m=err,
                      currents=cur, sigma_min=sig,
                      feedback_age_ms=link.feedback_age_s() * 1e3, node_status=status,
                      sent=sent)

            if t >= next_print:
                estr = "ok" if all(s.active_errors == 0 for s in status) else \
                    " ".join(f"ax{s.node_id}=0x{s.active_errors:x}" for s in status)
                pg = sent.get("pos_gain", ["?", "?"])
                cc = sent.get("current_cap_a", ["?", "?"])
                print(f"  t={t:4.1f}s  pose=({pose[0]*1e3:6.1f},{pose[1]*1e3:6.1f})mm  "
                      f"err={err*1e3:5.2f}mm  i=({cur[0]:+.3f},{cur[1]:+.3f})A  "
                      f"pos_gain=({pg[0]:.0f},{pg[1]:.0f})  cap=({cc[0]:.2f},{cc[1]:.2f})A  "
                      f"sig={sig:.3f}  age={link.feedback_age_s()*1e3:.1f}ms  err={estr}")
                next_print = t + 0.5
            time.sleep(period)
    except DriveDisarmed as exc:
        # No further ramping/settling -- the loop already stopped commanding
        # the instant the disarm was seen (checked every tick, above).
        disarm_info = {"node": exc.node_id, "t": exc.t, "reason": exc.reason}
        print(f"\n! {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
        summary = {"disarmed": disarm_info}
        print("\nSUMMARY_JSON " + json.dumps(summary))
        log.event("summary_json " + json.dumps(summary))
    except KeyboardInterrupt:
        print("\n! interrupted")
        log.event("interrupted", level="WARN")
    except Exception as exc:  # noqa: BLE001
        print(f"\n! {type(exc).__name__}: {exc}")
        log.event(f"{type(exc).__name__}: {exc}", level="ERROR")
    finally:
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
