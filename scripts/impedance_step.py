"""Closed-loop Cartesian step response using the host-side torque
(impedance) backend -- the TorqueBackend analogue of scripts/step_response.py
(PositionBackend), for evaluating whether host-side impedance control can hit
a higher free-air stiffness than the ODrive position cascade's ~10 N/m
elbow-mode ceiling.

    python -m scripts.impedance_step --stiffness 25 --damping 1 \\
        --current 0.8 --step-mm 5 --dir +x

Same run shape and guard rails as step_response.py (copy, not a shared
import, so the two scripts can diverge -- do not edit step_response.py from
here): arm at the current pose (must be within 60mm of `config.test_pose`),
hold the anchor 1s, step it `--step-mm` along `--dir`, hold 2s, step back,
hold 2s, relax, IDLE. Aborts on: a node leaving CLOSED_LOOP_CONTROL, a
joint-limit approach (panto.limits, raised inside TorqueBackend.apply), a
>50mm excursion from test_pose, feedback age >30ms, or the control loop
itself falling behind (see --rate; the drives broadcast Get_Encoder_Estimates
every 2ms/500Hz, so a host loop slower than that is working from stale
feedback more often than not -- warned, not aborted, since a single slow tick
isn't fatal the way an actual disarm is).

Tuning knobs specific to the torque backend (see panto/backends/torque.py):
``--damping`` (N.s/m tip damping), ``--vel-lpf-hz`` (low-pass on qd before
the damping term, default 20 Hz), ``--notch-hz``/``--notch-q`` (optional
notch on that same filtered velocity, off by default), ``--slew`` (N.m/s
per-joint torque slew limit, 0 = off), ``--torque-vel-gain`` (vel_gain pushed
during torque mode to raise the vel_limit plateau above the current cap).

Logs every tick (t, pose, anchor, q, qd, currents, F command, tau command,
axis_state) and writes the same summary.json shape as step_response.py
(rise/overshoot/settling/steady-state error/oscillation freq+decay/peak+rms
current/I2t/verdict), via the same panto.step_logic.analyze_step.

`--sim` runs against the in-process CAN sim instead of hardware.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from panto.backends import TorqueBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.constraints import Point
from panto.kinematics import forward
from panto.limits import JointLimitViolation, format_limits_deg, q_deg
from panto.presets import drive_defaults, restore_drive_defaults
from panto.step_logic import DIRS, anchor_offset_m, phase_name, total_duration_s, analyze_step
from panto.telemetry import RunLogger

MAX_FEEDBACK_AGE_S = 0.03
MAX_EXCURSION_MM = 50.0
MAX_START_OFFSET_MM = 60.0
MAX_HEARTBEAT_AGE_S = 2.0
#: drives broadcast Get_Encoder_Estimates every 2ms -- a host tick that takes
#: much longer than that is working off stale feedback more often than not.
LOOP_OVERRUN_WARN_S = 0.01


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    pass


class DriveLost(Aborted):
    """See step_response.py's DriveLost -- same rationale (heartbeat gone,
    don't retry writes to a bus with no listener)."""


def main() -> None:
    p = argparse.ArgumentParser(prog="impedance_step", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--sim", action="store_true", help="use the in-process sim bus instead of hardware")
    p.add_argument("--stiffness", type=float, required=True, help="isotropic EE stiffness K, N/m")
    p.add_argument("--damping", type=float, default=0.0, help="isotropic tip damping B, N.s/m")
    p.add_argument("--vel-lpf-hz", type=float, default=20.0,
                   help="low-pass cutoff (Hz) on qd before the damping term; <=0 disables")
    p.add_argument("--notch-hz", type=float, default=0.0, help="optional notch centre (Hz) on filtered qd; 0=off")
    p.add_argument("--notch-q", type=float, default=4.0, help="notch quality factor")
    p.add_argument("--slew", type=float, default=0.0, help="per-joint torque slew limit, N.m/s; 0=off")
    p.add_argument("--torque-vel-gain", type=float, default=0.01,
                   help="vel_gain pushed on both nodes in torque mode, to raise the "
                        "vel_limit*vel_gain plateau above the current cap (see "
                        "panto.breakaway_logic.plateau_vel_limit_rad_s)")
    p.add_argument("--current", type=float, default=0.8, help="per-axis current cap, A")
    p.add_argument("--step-mm", type=float, required=True, help="anchor step size, mm (magnitude)")
    p.add_argument("--dir", choices=sorted(DIRS), required=True, help="step direction in the calibrated frame")
    p.add_argument("--anchor-ramp-s", type=float, default=0.0,
                   help="ramp the anchor step/step-back over this many seconds (0 = instant)")
    p.add_argument("--rate", type=float, default=500.0, help="control loop Hz")
    p.add_argument("--cooldown-s", type=float, default=60.0,
                   help="mandatory idle wait after every run; pass 0 explicitly to skip")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    for m in config.motors:
        m.current_soft_max = args.current

    default_vel_gain = drive_defaults(config)

    nodes = [m.node_id for m in config.motors]
    K = args.stiffness * np.eye(2)
    step_m = args.step_mm * 1e-3
    dir_vec = DIRS[args.dir]
    total_s = total_duration_s()

    test_pose_xy = config.test_pose_xy_m

    log = RunLogger("impedance_step", interface=config.can.interface, channel=config.can.channel,
                    sim=args.sim, stiffness=args.stiffness, damping=args.damping,
                    vel_lpf_hz=args.vel_lpf_hz, notch_hz=args.notch_hz, notch_q=args.notch_q,
                    slew_nm_s=args.slew, torque_vel_gain=args.torque_vel_gain,
                    current=args.current, step_mm=args.step_mm, direction=args.dir,
                    anchor_ramp_s=args.anchor_ramp_s, rate=args.rate,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=args.sim)
    backend = TorqueBackend(link, config)
    backend.torque_vel_gain = args.torque_vel_gain
    backend.damping = args.damping
    backend.vel_lpf_hz = args.vel_lpf_hz
    backend.notch_hz = args.notch_hz
    backend.notch_q = args.notch_q
    backend.slew_nm_s = args.slew
    print(f"opening {config.can.interface}/{config.can.channel} (sim={args.sim})  "
          f"K={args.stiffness}N/m  B={args.damping}N.s/m  cap={args.current}A  "
          f"step={args.step_mm}mm {args.dir}  ramp={args.anchor_ramp_s}s")
    link.start()

    def idle_all() -> None:
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {nid}: {exc}", level="ERROR")

    armed = False
    aborted = False
    bus_lost = False
    abort_reason = None
    rows: list[dict] = []
    i2t_a2s = {0: 0.0, 1: 0.0}
    overruns = 0
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
              f"limits={format_limits_deg(config.motors)}")
        log.event(f"start q_deg={q_deg(q0)}")

        if test_pose_xy is None:
            if not args.sim:
                raise SystemExit("no test_pose configured in calibration.json -- refusing to "
                                 "run on hardware without a known excursion/start-pose reference")
            print("  ! no test_pose configured -- --sim run, using this run's own start pose "
                  "as the excursion reference")
            log.event("no test_pose configured -- sim run, excursion measured from start pose",
                     level="WARN")
            test_pose_xy = pose0.copy()
        start_offset_mm = float(np.linalg.norm(pose0 - test_pose_xy) * 1e3)
        print(f"  {start_offset_mm:.1f}mm from test_pose (cap {MAX_START_OFFSET_MM}mm)")
        if start_offset_mm > MAX_START_OFFSET_MM:
            raise SystemExit(f"start pose is {start_offset_mm:.1f}mm from test_pose, exceeds "
                             f"the {MAX_START_OFFSET_MM}mm cap -- move the arm back first")

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

        backend.enter()
        log.event(f"torque mode entered, torque_vel_gain={args.torque_vel_gain}")

        anchor0 = pose0.copy()
        constraint = Point(at=anchor0.copy())
        period = 1.0 / args.rate
        t0 = time.monotonic()
        last_phase = None

        while True:
            tick_t0 = time.monotonic()
            t = tick_t0 - t0
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
                      currents=cur, F_n=sent.get("F_n"), tau_nm=sent.get("tau_nm"),
                      node_status=status, feedback_age_ms=age_s * 1e3)

            elapsed_tick = time.monotonic() - tick_t0
            if elapsed_tick > max(period, LOOP_OVERRUN_WARN_S):
                overruns += 1
                if overruns <= 5 or overruns % 50 == 0:
                    log.event(f"loop overrun: tick took {elapsed_tick*1e3:.1f}ms "
                             f"(period {period*1e3:.1f}ms) -- feedback may be stale "
                             f"more often than not", level="WARN")
            sleep_s = period - elapsed_tick
            if sleep_s > 0:
                time.sleep(sleep_s)

        print(f"\ndone: {total_s:.1f}s schedule complete "
              f"({overruns} loop overruns of {len(rows)} ticks)")
        log.event(f"schedule complete, overruns={overruns}")

    except DriveLost as exc:
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
                restore_drive_defaults(link, default_vel_gain, log)
            idle_all()
            time.sleep(0.2)

        summary: dict = {"config": {"stiffness": args.stiffness, "damping": args.damping,
                                    "vel_lpf_hz": args.vel_lpf_hz, "notch_hz": args.notch_hz,
                                    "notch_q": args.notch_q, "slew_nm_s": args.slew,
                                    "torque_vel_gain": args.torque_vel_gain,
                                    "current": args.current, "step_mm": args.step_mm,
                                    "dir": args.dir, "anchor_ramp_s": args.anchor_ramp_s},
                        "aborted": aborted, "abort_reason": abort_reason, "bus_lost": bus_lost,
                        "loop_overruns": overruns,
                        "i2t_a2s": [i2t_a2s[0], i2t_a2s[1]]}
        print(f"  I2t (whole run, both nodes): node0={i2t_a2s[0]:.3f}A2.s "
              f"node1={i2t_a2s[1]:.3f}A2.s")

        step_rows = [r for r in rows if r["phase"] in ("ramp_up", "step_hold")]
        if step_rows and not aborted:
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

        if args.cooldown_s > 0:
            print(f"\ncooldown: {args.cooldown_s:.1f}s idle before exit...")
            time.sleep(args.cooldown_s)

    if aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
