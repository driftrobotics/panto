"""Static-friction breakaway measurement, one joint at a time.

    python -m scripts.breakaway

For each joint (shoulder=node0, elbow=node1) and each direction (+/-): hold the
*other* joint in position mode (modest stiffness) at its current angle, put the
test joint in torque mode, and ramp its commanded torque linearly from 0 at
--torque-rate (N.m/s) until it has moved --break-deg from its start -- that is
static-friction breakaway. Record (torque, current) at that instant, then
command zero torque, switch the test joint back to position mode, ramp it back
to its start angle, and move to the next direction/joint.

Hard aborts (see panto/breakaway_logic.py for the pure decision logic, unit
tested in tests/test_breakaway.py): current hits --current-cap before
--break-deg (torque-stalled, not a breakaway), motion exceeds --abort-deg, or
--abort-s elapses. Also: panto.limits joint-limit guard (measured q within
margin/2 of a configured limit) and an excursion cap (--max-excursion-mm from
config.test_pose) -- both abort immediately, command zero torque, and try to
return to start/test_pose in position mode.

Every tick logged via panto.telemetry.RunLogger, same shape as the other
bring-up scripts. relax (zero torque, hold pos) + IDLE on every exit path.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from panto.breakaway_logic import check_breakaway, plateau_vel_limit_rad_s
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.kinematics import forward
from panto.limits import JointLimitViolation, check_armable, check_runtime, format_limits_deg, q_deg
from panto.telemetry import RunLogger

MAX_FEEDBACK_AGE_S = 0.1
JOINT_NAMES = {0: "shoulder", 1: "elbow"}


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    """A node left CLOSED_LOOP_CONTROL mid-trial (e.g. a protective trip like
    DC_BUS_OVER_CURRENT). Get_Iq freezes at its last value once a node is
    IDLE, which would otherwise silently corrupt the breakaway current
    reading and any mean/peak-current statistic -- see torque_step.py's
    2026-09-04 incident note."""


def _hold_other(link, config, other_idx, other_q, hold_k_nm_rad):
    """One tick of position-mode holding for the non-test joint. Scalar
    joint-space stiffness (this is a 1-DOF hold, not an EE impedance) --
    pos_gain = k_rad / vel_gain, same mapping PositionBackend uses."""
    motor = config.motors[other_idx]
    vel_gain = motor.vel_gain
    pos_gain = 0.0 if vel_gain <= 0 else min(hold_k_nm_rad / vel_gain, motor.max_pos_gain)
    link.set_input_pos(motor.node_id, float(other_q))
    link.set_pos_gain(motor.node_id, pos_gain)


def _ramp_back(link, config, log, node_id, joint_idx, target_q, rate_hz, ramp_s, tol_deg,
              excursion_ref_m, max_excursion_mm, geo):
    """Position-mode ramp of one joint back to target_q, verified within
    tol_deg. Mirrors offset_sweep._ramp_to's shape but in joint space (this
    script is single-joint torque/position, not EE impedance)."""
    period = 1.0 / rate_hz
    q_all, _ = link.joint_state()
    q0 = float(q_all[joint_idx])
    t0 = time.monotonic()
    t_end = t0 + ramp_s + 0.3
    motor = config.motors[joint_idx]
    pos_gain = 0.0 if motor.vel_gain <= 0 else min(40.0 / motor.vel_gain, motor.max_pos_gain)
    q_final = q0
    while time.monotonic() < t_end:
        elapsed = time.monotonic() - t0
        frac = min(1.0, elapsed / ramp_s) if ramp_s > 0 else 1.0
        q_cmd = q0 + frac * (target_q - q0)
        link.set_input_pos(node_id, float(q_cmd))
        link.set_pos_gain(node_id, pos_gain)
        q_all, _ = link.joint_state()
        q_final = float(q_all[joint_idx])
        pose = forward(q_all, geo)
        if excursion_ref_m is not None:
            exc_mm = float(np.linalg.norm(pose - excursion_ref_m) * 1e3)
            if exc_mm > max_excursion_mm:
                log.event(f"excursion {exc_mm:.1f}mm exceeds cap during ramp-back", level="ERROR")
                raise Aborted("excursion cap exceeded during ramp-back")
        log.sample(tag="ramp_back", joint=joint_idx, q=q_all.tolist(), pose=pose.tolist())
        time.sleep(period)
    ok = abs(np.degrees(q_final - target_q)) <= tol_deg
    return ok, q_final


def main() -> None:
    p = argparse.ArgumentParser(prog="breakaway", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--current-cap", type=float, default=0.8, help="A, hard abort + Set_Limits cap")
    p.add_argument("--torque-rate", type=float, default=0.002, help="N.m/s ramp rate (~0.09 A/s at Kt=0.02235)")
    p.add_argument("--break-deg", type=float, default=2.0, help="deg of motion that counts as breakaway")
    p.add_argument("--abort-deg", type=float, default=5.0, help="deg of motion: hard abort (runaway)")
    p.add_argument("--abort-s", type=float, default=15.0, help="seconds: hard abort (timeout)")
    p.add_argument("--hold-stiffness", type=float, default=15.0, help="N.m/rad, the non-test joint's hold gain")
    p.add_argument("--vel-limit", type=float, default=10.0, help="ODrive vel_limit, joint rad/s, held joint / position moves")
    p.add_argument("--torque-vel-gain", type=float, default=0.01,
                   help="vel_gain during the torque-mode ramp (both nodes; restored to config "
                        "default in finally). ODrive's enable_torque_mode_vel_limit clamps "
                        "effective torque to vel_gain*(vel_limit-|vel|) even in TORQUE_CONTROL. "
                        "vel_limit for the joint under test is now sized automatically from "
                        "--current-cap (see panto.breakaway_logic.plateau_vel_limit_rad_s) so the "
                        "plateau always sits 1.5x above the requested cap, not a fixed value -- "
                        "2026-09-04: a fixed vel_limit sized for one cap silently re-capped Iq in "
                        "higher-cap runs.")
    p.add_argument("--max-excursion-mm", type=float, default=50.0,
                   help="abort if measured tip exceeds this from config.test_pose")
    p.add_argument("--ramp-back-s", type=float, default=4.0, help="seconds to ramp back to start after each trial")
    p.add_argument("--rate", type=float, default=100.0, help="control loop Hz")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    for m in config.motors:
        m.current_soft_max = args.current_cap

    test_pose_xy = config.test_pose_xy_m
    log = RunLogger("breakaway", interface=config.can.interface, channel=config.can.channel,
                    current_cap=args.current_cap, torque_rate=args.torque_rate,
                    break_deg=args.break_deg, abort_deg=args.abort_deg, abort_s=args.abort_s,
                    hold_stiffness=args.hold_stiffness, vel_limit=args.vel_limit,
                    max_excursion_mm=args.max_excursion_mm,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=False)
    print(f"opening {config.can.interface}/{config.can.channel} (cap {args.current_cap}A, "
          f"rate {args.torque_rate} N.m/s, break {args.break_deg}deg)")
    link.start()

    def idle_all() -> None:
        for m in config.motors:
            try:
                link.set_idle(m.node_id)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {m.node_id}: {exc}", level="ERROR")

    results = []
    i2t: dict = {}          # node_id -> A^2.s, integrated over the whole process
    ibus_samples: dict = {}  # node_id -> list of (ibus, vbus) while that node is armed
    armed = False
    aborted = False
    disarm_summary = None
    default_vel_gain = None
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm "
              f"limits={format_limits_deg(config.motors)}")
        log.event(f"start q_deg={q_deg(q0)} limits={format_limits_deg(config.motors)}")

        problems = check_armable(q0, config.motors)
        if problems:
            raise SystemExit("refusing to arm -- " + "; ".join(problems))

        excursion_ref_m = test_pose_xy if test_pose_xy is not None else pose0.copy()
        if test_pose_xy is None:
            log.event("no test_pose configured -- excursion measured from this run's own start",
                     level="WARN")

        for i, m in enumerate(config.motors):
            link.set_controller_mode(m.node_id, "position")
            link.set_limits(m.node_id, args.vel_limit, args.current_cap)
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
        for m in config.motors:
            link.clear_errors(m.node_id)
        default_vel_gain = config.motors[0].vel_gain
        for m in config.motors:
            link.set_vel_gains(m.node_id, args.torque_vel_gain, 0.0)
        log.event(f"set vel_gain -> {args.torque_vel_gain} on both nodes "
                 f"(raises the torque-mode-vel-limit plateau; restored to {default_vel_gain} at exit)")
        log.event("cleared errors on both nodes")

        period = 1.0 / args.rate
        for joint_idx, motor in enumerate(config.motors):
            other_idx = 1 - joint_idx
            other_motor = config.motors[other_idx]
            for sign, dirname in ((+1, "+"), (-1, "-")):
                q_all, _ = link.joint_state()
                start_q = float(q_all[joint_idx])
                other_q = float(q_all[other_idx])
                tag = f"breakaway_{JOINT_NAMES[joint_idx]}_{dirname}"
                print(f"\n-- {JOINT_NAMES[joint_idx]} {dirname}: start={np.degrees(start_q):.2f}deg --")
                log.event(f"{tag}: start q={np.degrees(start_q):.2f}deg")

                vel_limit_rad_s, plateau_a = plateau_vel_limit_rad_s(
                    args.current_cap, motor.torque_constant, args.torque_vel_gain)
                print(f"   plateau = {plateau_a:.2f}A, cap = {args.current_cap:.2f}A "
                      f"(vel_limit={vel_limit_rad_s:.2f}rad/s)")
                log.event(f"{tag}: plateau={plateau_a:.3f}A cap={args.current_cap:.3f}A "
                         f"vel_limit={vel_limit_rad_s:.3f}rad/s")
                link.set_controller_mode(motor.node_id, "torque")
                link.set_limits(motor.node_id, vel_limit_rad_s, args.current_cap)
                link.set_limits(other_motor.node_id, args.vel_limit, other_motor.current_soft_max)

                t0 = time.monotonic()
                result = None
                while result is None:
                    elapsed = time.monotonic() - t0
                    age_s = link.feedback_age_s()
                    if age_s > MAX_FEEDBACK_AGE_S:
                        raise Aborted(f"feedback age {age_s*1e3:.1f}ms exceeds cap")

                    status = link.node_status()
                    for s in status:
                        if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                            reason = decode_error_flags(s.disarm_reason or 0)
                            msg = (f"drive disarmed at t={elapsed:.3f}s: node {s.node_id} "
                                  f"axis_state {s.axis_state} (left CLOSED_LOOP) reason={reason}")
                            log.event(msg, level="ERROR")
                            raise DriveDisarmed(msg)

                    q_all, _ = link.joint_state()
                    check_runtime(q_all, config.motors)
                    pose = forward(q_all, config.geo)
                    exc_mm = float(np.linalg.norm(pose - excursion_ref_m) * 1e3)
                    if exc_mm > args.max_excursion_mm:
                        raise Aborted(f"excursion {exc_mm:.1f}mm exceeds cap {args.max_excursion_mm}mm")

                    tau = args.torque_rate * elapsed * sign
                    link.set_input_torque(motor.node_id, tau)
                    _hold_other(link, config, other_idx, other_q, args.hold_stiffness)

                    cur = link.motor_currents()
                    vbus, ibus = link.bus_voltage_current()
                    for idx, m in enumerate(config.motors):
                        i2t.setdefault(m.node_id, 0.0)
                        i2t[m.node_id] += float(cur[idx]) ** 2 * period
                        ibus_samples.setdefault(m.node_id, []).append(
                            (float(ibus[idx]), float(vbus[idx])))
                    moved_deg = float(np.degrees(q_all[joint_idx] - start_q))
                    log.sample(tag=tag, q=q_all.tolist(), pose=pose.tolist(), currents=cur.tolist(),
                              tau_cmd_nm=tau, moved_deg=moved_deg, node_status=status)

                    result = check_breakaway(
                        elapsed, moved_deg, float(cur[joint_idx]),
                        rate_nm_s=args.torque_rate, sign=sign, current_cap_a=args.current_cap,
                        break_deg=args.break_deg, abort_deg=args.abort_deg, abort_s=args.abort_s)
                    time.sleep(period)

                print(f"   {result.status}: tau={result.torque_nm*1e3:.2f}mN.m "
                      f"I={result.current_a:.3f}A moved={result.moved_deg:.2f}deg "
                      f"t={result.elapsed_s:.2f}s")
                log.event(f"{tag}: {result.status} tau_mnm={result.torque_nm*1e3:.3f} "
                         f"i_a={result.current_a:.4f} moved_deg={result.moved_deg:.3f} "
                         f"elapsed_s={result.elapsed_s:.2f}")
                results.append({
                    "joint": JOINT_NAMES[joint_idx], "node_id": motor.node_id, "direction": dirname,
                    "status": result.status, "torque_nm": result.torque_nm,
                    "current_a": result.current_a, "moved_deg": result.moved_deg,
                    "elapsed_s": result.elapsed_s,
                })

                # zero torque, then ramp back to start in position mode
                link.set_input_torque(motor.node_id, 0.0)
                link.set_controller_mode(motor.node_id, "position")
                ok, q_final = _ramp_back(link, config, log, motor.node_id, joint_idx, start_q,
                                         args.rate, args.ramp_back_s, tol_deg=1.0,
                                         excursion_ref_m=excursion_ref_m,
                                         max_excursion_mm=args.max_excursion_mm, geo=config.geo)
                if not ok:
                    msg = (f"{tag}: failed to return to start "
                          f"({np.degrees(q_final - start_q):.2f}deg off)")
                    log.event(msg, level="ERROR")
                    print(f"   ! {msg}")
                else:
                    log.event(f"{tag}: returned to start ({np.degrees(q_final - start_q):.3f}deg off)")

        print()
        for m in config.motors:
            samples = ibus_samples.get(m.node_id, [])
            if samples:
                ibus_arr = np.array([s[0] for s in samples])
                vbus_arr = np.array([s[1] for s in samples])
                mean_ibus, mean_vbus = float(np.mean(ibus_arr)), float(np.mean(vbus_arr))
                mean_bus_power_w = mean_ibus * mean_vbus
            else:
                mean_bus_power_w = float("nan")
            print(f"  node{m.node_id}: I2t={i2t.get(m.node_id, 0.0):.4f}A2.s  "
                  f"bus_power mean={mean_bus_power_w:.2f}W  "
                  f"(crude thermal-budget / dissipation indicators, whole process)")
        print("\nBREAKAWAY_JSON " + json.dumps(results))
        log.event("breakaway_json " + json.dumps(results))
    except DriveDisarmed as exc:
        aborted = True
        disarm_summary = str(exc)
        print(f"\n! {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
    except (Aborted, JointLimitViolation) as exc:
        aborted = True
        print(f"\n! aborted: {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
    except KeyboardInterrupt:
        aborted = True
        print("\n! interrupted")
        log.event("interrupted", level="WARN")
    except Exception as exc:  # noqa: BLE001
        aborted = True
        print(f"\n! {type(exc).__name__}: {exc}")
        log.event(f"{type(exc).__name__}: {exc}", level="ERROR")
    finally:
        print("\nzero torque / relax + IDLE")
        if armed:
            try:
                for m in config.motors:
                    link.set_input_torque(m.node_id, 0.0)
                    link.set_pos_gain(m.node_id, 0.0)
            except Exception:  # noqa: BLE001
                pass
            if default_vel_gain is not None:
                for m in config.motors:
                    try:
                        link.set_vel_gains(m.node_id, default_vel_gain, 0.0)
                        log.event(f"restored node {m.node_id} vel_gain -> {default_vel_gain}")
                    except Exception as exc:  # noqa: BLE001
                        log.event(f"restore vel_gain node {m.node_id}: {exc}", level="ERROR")
        idle_all()
        time.sleep(0.2)
        if disarm_summary is not None:
            print(f"\n  *** {disarm_summary} ***")
        link.close()
        log.close()
        print(f"closed. log: {log.dir}")

    if aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
