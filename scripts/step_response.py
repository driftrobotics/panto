"""Closed-loop Cartesian step response, for the K/vel_gain/vel_limit tuning
sweep (2026-09-04, position-cascade limit-cycle investigation).

    python -m scripts.step_response --stiffness 100 --vel-gain 0.09 \\
        --vel-limit 20 --current 2.0 --step-mm 5 --dir +x

Arms at the CURRENT pose (must be within 60mm of `config.test_pose`, joint
limits active, 50mm excursion guard from test_pose -- same family of guards as
scripts/point_hold.py / torque_step.py). Pushes `set_vel_gains(vel_gain)` and
`set_limits(vel_limit, current)` on both nodes, then runs the real
`PositionBackend` at isotropic EE stiffness `--stiffness` (K) with the anchor
held on the start pose for 1s, then steps the anchor by `--step-mm` along
`--dir` (one of +x/-x/+y/-y in the calibrated frame), holds 2s, steps back,
holds 2s, relaxes, IDLE. Returns the arm to its start pose before IDLE (the
anchor is already back on it by the time the loop ends, so this is mostly
"don't leave gains applied", not a separate move).

`--anchor-ramp-s` (default 0 = instantaneous) ramps the anchor step/step-back
over that many seconds instead of stepping it, since haptic rendering may
prefer a ramp -- see panto/step_logic.py's schedule.

Logs every tick to samples.jsonl via panto.telemetry.RunLogger (t, pose,
anchor, q, qd, currents, sent pos_gain/current_cap, axis_state) and aborts
immediately on: any node leaving CLOSED_LOOP_CONTROL (decoded disarm reason),
a joint-limit approach (panto.limits.check_runtime, raised inside
PositionBackend.apply), a >50mm excursion from test_pose, or feedback age
>30ms. Post-processes in-script (panto.step_logic.analyze_step) over the
step-hold window: rise time, overshoot (mm), settling time to +/-1mm, steady-
state error, dominant oscillation frequency (FFT of pose error along the step
direction), oscillation decay ratio (pose-error std, last 0.5s / first 0.5s of
the step-hold window), peak/RMS current per joint, and a verdict --
`converged` / `damped_oscillation` / `limit_cycle` / `stall`. Prints one JSON
summary line (also written to `summary.json` in the log dir).
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
from panto.kinematics import forward
from panto.limits import JointLimitViolation, format_limits_deg, q_deg
from panto.presets import apply_to_config, drive_defaults, resolve as resolve_preset
from panto.step_logic import (
    DIRS, anchor_offset_m, phase_name, total_duration_s, analyze_step, parse_per_joint,
    to_per_joint,
)
from panto.telemetry import RunLogger

MAX_FEEDBACK_AGE_S = 0.03           # per the brief: 30ms
MAX_EXCURSION_MM = 50.0             # from test_pose, hard bound
MAX_START_OFFSET_MM = 60.0          # start pose must be within this of test_pose
MAX_HEARTBEAT_AGE_S = 2.0           # no heartbeat this long -> drive is gone from the bus


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    pass


class DriveLost(Aborted):
    """A node's Heartbeat hasn't been seen in > MAX_HEARTBEAT_AGE_S -- it has
    dropped off the bus entirely (power loss, CAN fault, etc), as opposed to
    DriveDisarmed (still on the bus, but left CLOSED_LOOP_CONTROL). Abort
    cleanly and do not retry writes to a bus that may not have a listener --
    see the 2026-09-04 cooldown/heartbeat incident."""


def main() -> None:
    p = argparse.ArgumentParser(prog="step_response", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--preset", type=str, default=None,
                   help="named preset from presets.json (panto.presets); supplies any of "
                        "stiffness/vel-gain/vel-limit/current/cap-slope/cap-min/ff-scale/"
                        "max-pos-gain not given explicitly below")
    p.add_argument("--stiffness", type=float, default=None, help="isotropic EE stiffness K, N/m")
    p.add_argument("--vel-gain", type=str, default=None,
                   help="ODrive vel_gain: 'V' (both joints) or 'V0,V1' (shoulder,elbow)")
    p.add_argument("--vel-limit", type=float, default=None, help="ODrive vel_limit, joint rad/s")
    p.add_argument("--current", type=float, default=None, help="per-axis current cap, A (cap_max)")
    p.add_argument("--cap-slope", type=str, default=None,
                   help="A per rad/s, velocity-scheduled current cap: "
                        "cap = clamp(cap_max - cap_slope*|qd|, cap_min, cap_max). "
                        "'V' applies to both joints, 'V0,V1' is shoulder,elbow. "
                        "0 = off, constant cap_max (old behaviour). Default: 0.0 or preset value")
    p.add_argument("--cap-min", type=str, default=None,
                   help="A, floor for the velocity-scheduled cap (only matters if --cap-slope > 0). "
                        "'V' applies to both joints, 'V0,V1' is shoulder,elbow (shoulder/elbow "
                        "breakaway differs a lot -- see calibration.json's coulomb_*_nm). "
                        "Default: 0.5 or preset value")
    p.add_argument("--ff-scale", type=float, default=None,
                   help="override motor.ff_scale (Coulomb friction feedforward strength); "
                        "0 disables feedforward outright regardless of calibration.json's "
                        "coulomb_pos_nm/coulomb_neg_nm. Default: preset value, else each motor's "
                        "configured value")
    p.add_argument("--max-pos-gain", type=float, default=None, help="override motor.max_pos_gain")
    p.add_argument("--step-mm", type=float, required=True, help="anchor step size, mm (magnitude)")
    p.add_argument("--dir", choices=sorted(DIRS), required=True,
                   help="step direction in the calibrated frame")
    p.add_argument("--anchor-ramp-s", type=float, default=0.0,
                   help="ramp the anchor step/step-back over this many seconds "
                        "(0 = instantaneous step)")
    p.add_argument("--rate", type=float, default=250.0, help="control loop Hz")
    p.add_argument("--cooldown-s", type=float, default=60.0,
                   help="mandatory idle wait after every run (success OR abort), before exit -- "
                        "thermal cooldown between grid points. Not skippable via 0 by accident: "
                        "pass --cooldown-s 0 explicitly if you really want none (e.g. quick "
                        "interactive iteration)")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    resolved = resolve_preset(
        {
            "stiffness": args.stiffness,
            "vel_gain": parse_per_joint(args.vel_gain) if args.vel_gain is not None else None,
            "vel_limit": args.vel_limit,
            "current": args.current,
            "cap_slope": parse_per_joint(args.cap_slope) if args.cap_slope is not None else None,
            "cap_min": parse_per_joint(args.cap_min) if args.cap_min is not None else None,
            "ff_scale": args.ff_scale, "max_pos_gain": args.max_pos_gain,
        },
        args.preset,
    )
    if resolved["stiffness"] is None or resolved["vel_gain"] is None or resolved["vel_limit"] is None:
        raise SystemExit("--stiffness/--vel-gain/--vel-limit are required (pass explicitly or via --preset)")
    resolved["current"] = 2.0 if resolved["current"] is None else resolved["current"]
    cap_slope_per_joint = to_per_joint(resolved["cap_slope"]) or (0.0, 0.0)
    resolved["cap_slope"] = list(cap_slope_per_joint)
    cap_min_per_joint = tuple(resolved["cap_min"]) if resolved["cap_min"] is not None else (0.5, 0.5)
    vel_gain_per_joint = to_per_joint(resolved["vel_gain"])
    resolved["vel_gain"] = list(vel_gain_per_joint)

    for m in config.motors:
        m.current_soft_max = resolved["current"]
    default_vel_gain = drive_defaults(config)
    apply_to_config(config, resolved)

    nodes = [m.node_id for m in config.motors]
    K = resolved["stiffness"] * np.eye(2)
    step_m = args.step_mm * 1e-3
    dir_vec = DIRS[args.dir]
    total_s = total_duration_s()

    test_pose_xy = config.test_pose_xy_m

    log = RunLogger("step_response", interface=config.can.interface, channel=config.can.channel,
                    preset=args.preset,
                    stiffness=resolved["stiffness"], vel_gain=resolved["vel_gain"],
                    vel_limit=resolved["vel_limit"], current=resolved["current"],
                    cap_slope=resolved["cap_slope"], cap_min=list(cap_min_per_joint),
                    ff_scale=resolved["ff_scale"], max_pos_gain=resolved["max_pos_gain"],
                    step_mm=args.step_mm, direction=args.dir,
                    anchor_ramp_s=args.anchor_ramp_s, rate=args.rate,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=False)
    backend = PositionBackend(link, config)
    backend.vel_limit_rad_s = resolved["vel_limit"]
    print(f"opening {config.can.interface}/{config.can.channel}  preset={args.preset}  "
          f"K={resolved['stiffness']}N/m  vel_gain={resolved['vel_gain']}  "
          f"vel_limit={resolved['vel_limit']}rad/s  cap={resolved['current']}A  "
          f"step={args.step_mm}mm {args.dir}  ramp={args.anchor_ramp_s}s")
    link.start()

    def idle_all() -> None:
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {nid}: {exc}", level="ERROR")

    def restore_vel_gain() -> None:
        for nid in nodes:
            try:
                link.set_vel_gains(nid, default_vel_gain[nid], 0.0)
                log.event(f"restored node {nid} vel_gain -> {default_vel_gain[nid]}")
            except Exception as exc:  # noqa: BLE001
                log.event(f"restore vel_gain node {nid}: {exc}", level="ERROR")

    armed = False
    aborted = False
    bus_lost = False
    abort_reason = None
    rows: list[dict] = []  # t, disp_along_dir_mm, currents -- for post-processing
    i2t_a2s = {0: 0.0, 1: 0.0}  # integral Iq^2 dt, both nodes, whole armed run
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        pos_gain0 = PositionBackend.pos_gains_for(
            resolved["stiffness"] * np.eye(2), q0, config
        )
        print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
              f"pos_gain@start={[round(g, 1) for g in pos_gain0]}  "
              f"limits={format_limits_deg(config.motors)}")
        log.event(f"start q_deg={q_deg(q0)}")

        if test_pose_xy is None:
            raise SystemExit("no test_pose configured in calibration.json -- refusing to run "
                             "without a known excursion/start-pose reference")
        start_offset_mm = float(np.linalg.norm(pose0 - test_pose_xy) * 1e3)
        print(f"  {start_offset_mm:.1f}mm from test_pose (cap {MAX_START_OFFSET_MM}mm)")
        if start_offset_mm > MAX_START_OFFSET_MM:
            raise SystemExit(f"start pose is {start_offset_mm:.1f}mm from test_pose, exceeds "
                             f"the {MAX_START_OFFSET_MM}mm cap -- move the arm back first")

        # set gains BEFORE arming so the very first closed-loop tick already
        # uses the sweep's vel_gain/vel_limit, not the config default.
        for i, nid in enumerate(nodes):
            link.set_vel_gains(nid, resolved["vel_gain"][i], 0.0)
        log.event(f"set vel_gain -> {resolved['vel_gain']} (per node)")

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
        armed = True
        for nid in nodes:
            link.clear_errors(nid)
        log.event("cleared errors on both nodes")
        # re-assert vel_gain post-arm/clear_errors in case either touched it
        for i, nid in enumerate(nodes):
            link.set_vel_gains(nid, resolved["vel_gain"][i], 0.0)

        anchor0 = pose0.copy()
        constraint = Point(at=anchor0.copy())
        period = 1.0 / args.rate
        t0 = time.monotonic()
        last_phase = None

        while True:
            t = time.monotonic() - t0
            if t >= total_s:
                break

            status = link.node_status()
            for s in status:
                if s.age_s > MAX_HEARTBEAT_AGE_S:
                    msg = (f"drive lost at t={t:.3f}s: node {s.node_id} heartbeat age "
                          f"{s.age_s:.2f}s exceeds {MAX_HEARTBEAT_AGE_S:.1f}s cap -- "
                          f"off the bus, not just disarmed")
                    log.event(msg, level="ERROR")
                    raise DriveLost(msg)

            age_s = link.feedback_age_s()
            if age_s > MAX_FEEDBACK_AGE_S:
                raise Aborted(f"feedback age {age_s*1e3:.1f}ms exceeds "
                              f"{MAX_FEEDBACK_AGE_S*1e3:.0f}ms cap")

            for s in status:
                if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    reason = decode_error_flags(s.disarm_reason or 0)
                    msg = (f"drive disarmed at t={t:.3f}s: node {s.node_id} axis_state "
                          f"{s.axis_state} (left CLOSED_LOOP) reason={reason}")
                    log.event(msg, level="ERROR")
                    raise DriveDisarmed(msg)

            phase = phase_name(t, ramp_s=args.anchor_ramp_s)
            if phase != last_phase:
                print(f"  -- t={t:5.2f}s phase={phase} --")
                log.event(f"phase -> {phase} at t={t:.3f}s")
                last_phase = phase

            offset = anchor_offset_m(t, step_m, ramp_s=args.anchor_ramp_s)
            anchor = anchor0 + offset * dir_vec
            constraint = Point(at=anchor)

            q, qd = link.joint_state()
            pose = forward(q, config.geo)

            exc_mm = float(np.linalg.norm(pose - test_pose_xy) * 1e3)
            if exc_mm > MAX_EXCURSION_MM:
                raise Aborted(f"excursion {exc_mm:.1f}mm exceeds cap {MAX_EXCURSION_MM}mm")

            # force_limit is deliberately generous (not the real constraint) --
            # PositionBackend.apply()'s Set_Limits current cap is clamped to
            # motor.current_soft_max regardless (see _current_cap), which is
            # already set to --current above. This just avoids a smaller,
            # accidental clamp from a tight force_limit at low sigma_min.
            proj = constraint.project(pose)
            cmd = ImpedanceCommand(pose=pose, q=q, qd=qd, anchor=proj.anchor, stiffness=K,
                                   force_limit=50.0)
            try:
                backend.apply(cmd)
            except JointLimitViolation as exc:
                log.event(f"joint limit violation ({exc})", level="ERROR")
                raise Aborted(f"joint limit violation ({exc})")

            cur = link.motor_currents()
            for node_i in (0, 1):
                i2t_a2s[node_i] += float(cur[node_i]) ** 2 * period
            sent = backend.last_command or {}
            disp_along_dir_mm = float(np.dot(pose - anchor0, dir_vec) * 1e3)

            rows.append({"t": t, "phase": phase, "disp_mm": disp_along_dir_mm,
                        "currents": cur.tolist()})

            log.sample(t=t, phase=phase, q=q, qd=qd, pose=pose, anchor=proj.anchor,
                      currents=cur, sent=sent, node_status=status,
                      feedback_age_ms=age_s * 1e3)

            time.sleep(period)

        print(f"\ndone: {total_s:.1f}s schedule complete")
        log.event("schedule complete")

    except DriveLost as exc:
        # Off the bus entirely -- don't retry writes to it below (relax/idle/
        # restore_vel_gain all transmit; a dead bus just makes them hang or
        # fail loudly for no benefit). Abort cleanly, once.
        aborted = True
        bus_lost = True
        abort_reason = str(exc)
        print(f"\n! {exc}")
        log.event(f"aborted (bus lost): {exc}", level="ERROR")
    except DriveDisarmed as exc:
        aborted = True
        abort_reason = str(exc)
        print(f"\n! {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
    except (Aborted, JointLimitViolation) as exc:
        aborted = True
        abort_reason = str(exc)
        print(f"\n! aborted: {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
    except KeyboardInterrupt:
        aborted = True
        abort_reason = "interrupted"
        print("\n! interrupted")
        log.event("interrupted", level="WARN")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        aborted = True
        abort_reason = f"{type(exc).__name__}: {exc}"
        print(f"\n! {type(exc).__name__}: {exc}")
        log.event(f"{type(exc).__name__}: {exc}", level="ERROR")
    finally:
        if bus_lost:
            print("\n! bus lost -- skipping relax/idle/vel-gain-restore writes "
                  "(nothing there to hear them); do not retry")
            log.event("bus lost: skipped relax/idle/restore writes", level="WARN")
        else:
            print("\nrelax + IDLE")
            if armed:
                try:
                    backend.relax()
                except Exception:  # noqa: BLE001
                    pass
                restore_vel_gain()
            idle_all()
            time.sleep(0.2)

        summary: dict = {"preset": args.preset,
                        "config": {"stiffness": resolved["stiffness"], "vel_gain": resolved["vel_gain"],
                                    "vel_limit": resolved["vel_limit"], "current": resolved["current"],
                                    "cap_slope": resolved["cap_slope"], "cap_min": list(cap_min_per_joint),
                                    "ff_scale": resolved["ff_scale"], "max_pos_gain": resolved["max_pos_gain"],
                                    "step_mm": args.step_mm, "dir": args.dir,
                                    "anchor_ramp_s": args.anchor_ramp_s},
                        "aborted": aborted, "abort_reason": abort_reason, "bus_lost": bus_lost,
                        "i2t_a2s": [i2t_a2s[0], i2t_a2s[1]]}
        print(f"  I2t (whole run, both nodes): node0={i2t_a2s[0]:.3f}A2.s "
              f"node1={i2t_a2s[1]:.3f}A2.s")

        step_rows = [r for r in rows if r["phase"] in ("ramp_up", "step_hold")]
        if step_rows and not aborted:
            # re-baseline t to the step-hold window's start (t=0 at first
            # ramp_up/step_hold sample) so analyze_step sees onset-relative time
            t_arr = np.array([r["t"] for r in step_rows])
            t_arr = t_arr - t_arr[0]
            disp_arr = np.array([r["disp_mm"] for r in step_rows])
            cur_arr = np.array([r["currents"] for r in step_rows])
            metrics = analyze_step(t_arr, disp_arr, args.step_mm, currents=cur_arr)
            peak_i = np.abs(cur_arr).max(axis=0).tolist() if len(cur_arr) else [float("nan")] * 2
            rms_i = np.sqrt((cur_arr ** 2).mean(axis=0)).tolist() if len(cur_arr) else [float("nan")] * 2
            summary.update({
                "rise_time_s": metrics.rise_time_s,
                "overshoot_mm": metrics.overshoot_mm,
                "settling_time_s": metrics.settling_time_s,
                "steady_state_error_mm": metrics.steady_state_error_mm,
                "osc_freq_hz": metrics.osc_freq_hz,
                "osc_amplitude_mm": metrics.osc_amplitude_mm,
                "decay_ratio": metrics.decay_ratio,
                "peak_current_a": peak_i,
                "rms_current_a": rms_i,
                "verdict": metrics.verdict,
            })
        else:
            summary["verdict"] = "aborted" if aborted else "no_data"

        (log.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\nSUMMARY_JSON " + json.dumps(summary))
        log.event("summary_json " + json.dumps(summary))

        link.close()
        log.close()
        print(f"log: {log.dir}")

        # Mandatory thermal cooldown -- runs even on abort/bus_lost (this is
        # just a local sleep, no further CAN traffic, so it's safe either
        # way). Deliberately after link.close()/log.close() so a long
        # cooldown doesn't hold the CAN handle or log files open.
        if args.cooldown_s > 0:
            print(f"\ncooldown: {args.cooldown_s:.1f}s idle before exit...")
            time.sleep(args.cooldown_s)

    if aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
