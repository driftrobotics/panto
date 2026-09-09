"""Open-loop torque step, for hand-driven bring-up checks. One joint or both.

    python -m scripts.torque_step --joint 0 --torque-mnm max --duration 2
    python -m scripts.torque_step --joint 0 --torque-mnm -max --duration 2
    python -m scripts.torque_step --joint both --torque-mnm max --duration 4
    python -m scripts.torque_step --joint both --torque-mnm max,-max --duration 2

Single-joint (`--joint 0` / `--joint 1`): that joint in torque mode, the other
held in position mode at its current angle (`--no-hold-other` to skip).
Both-joint (`--joint both`): BOTH nodes in torque mode from this one process,
no position hold -- `--torque-mnm` takes `T0,T1` (each may be `max`/`-max`;
a single value with no comma applies to both nodes the same).

Same schedule either way: 0.5s at zero, `--duration`s at the commanded
value(s), 0.5s back at zero, then IDLE. Reuses scripts/breakaway.py's
torque-mode plumbing -- ODrive's `enable_torque_mode_vel_limit` clamps
*effective* torque to `vel_gain*(vel_limit-|vel|)` even in TORQUE_CONTROL
(see panto/breakaway_logic.py's 2026-09-04 note), so this script raises
vel_gain to 0.01 and vel_limit to 20rad/s on every node in torque mode right
after arming (restored to the config default in `finally`) so the plateau
(~32mN.m) sits above the current cap and the cap does the limiting, not the
plateau.

`--torque-mnm max` / `-max` resolve to +/- (current cap * torque_constant),
i.e. the largest torque the current cap allows.

Guards, every tick, per active joint: joint-limit trip wire
(panto.limits.check_runtime), --max-deg motion cap, --max-excursion-mm from
config.test_pose (hardcoded 50mm outer bound per the safety brief), and stale
feedback (>100ms). Any trip zeroes torque on EVERY active joint and goes to
IDLE immediately -- no ramp-back (this is a diagnostic step, not a repeatable
trial like breakaway.py).

Logged via panto.telemetry.RunLogger: every tick's t, q (deg, both joints),
qd, tau_cmd (per active joint), currents (Iq, both nodes), ibus/vbus (both
nodes, via Get_Bus_Voltage_Current -- see CanLink.bus_voltage_current(); not
all firmware configs broadcast this by default, see dbc/README.md), tip pose;
events for start/step-on/step-off/abort. Summary at exit: mean/peak Iq per
node, mean/peak ibus per node, and the sum of the two ibus means (what a
supply ammeter should read, plus board quiescent draw).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.kinematics import forward
from panto.config import Config
from panto.limits import JointLimitViolation, check_armable, check_runtime, format_limits_deg, q_deg
from panto.presets import resolve as resolve_preset
from panto.telemetry import RunLogger
from panto.breakaway_logic import plateau_vel_limit_rad_s
from panto.torque_step_logic import check_abort, schedule, total_duration_s

MAX_FEEDBACK_AGE_S = 0.1
MAX_EXCURSION_MM = 50.0     # hard bound, not a CLI flag -- see the safety brief
TORQUE_VEL_GAIN = 0.01      # raises the torque-mode-vel-limit plateau; see breakaway_logic.py
HOLD_VEL_LIMIT_RAD_S = 10.0
JOINT_NAMES = {0: "shoulder", 1: "elbow"}


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    """A node under test left CLOSED_LOOP_CONTROL mid-run (e.g. a protective
    trip like DC_BUS_OVER_CURRENT) -- torque kept being commanded into thin
    air otherwise, and Get_Iq freezes at its last value once a node is IDLE,
    which silently corrupts any mean/peak-current statistic computed over
    that period. See the 2026-09-04 incident: node 0 disarmed 11ms after
    step-on, the script kept commanding torque for the full 4s, and the
    summary reported a bogus 'Iq mean 1.49A' that was really Get_Iq's frozen
    last-good reading."""


def _resolve_torque(spec: str, current_cap: float, torque_constant: float) -> float:
    s = spec.strip().lower()
    if s in ("max", "+max"):
        return current_cap * torque_constant
    if s == "-max":
        return -current_cap * torque_constant
    return float(spec) * 1e-3


def main() -> None:
    p = argparse.ArgumentParser(prog="torque_step", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--joint", type=str, required=True, choices=("0", "1", "both"),
                   help="which joint(s) to step")
    p.add_argument("--torque-mnm", type=str, required=True,
                   help="commanded torque, mN.m, signed, joint frame, CCW positive. "
                        "'max'/'-max' = +/-(current cap * torque_constant). "
                        "For --joint both: 'T0,T1' (per node), or a single value for both.")
    p.add_argument("--duration", type=float, default=2.0, help="seconds at the commanded torque")
    p.add_argument("--preset", type=str, default=None,
                   help="named preset from presets.json; only its 'current' field applies here "
                        "(this is a torque-control script, not position -- stiffness/vel-gain/etc "
                        "don't apply). Explicit --current overrides it")
    p.add_argument("--current", type=float, default=None,
                   help="A, current cap (default 0.8, or preset's 'current'). No hard ceiling "
                        "(user has accepted destructive-testing risk) -- values above the 1.0A "
                        "datasheet stall max print a warning and proceed; values above the "
                        "drive's current_hard_max (see --hard-max) will be silently clamped by "
                        "the drive itself, also warned")
    p.add_argument("--hard-max", type=float, default=2.5,
                   help="A, the drive's currently-configured axis0.config.motor.current_hard_max "
                        "(both drives) -- used only to warn if --current exceeds it. Update this "
                        "if the drives' stored current_hard_max changes; this script does not read "
                        "it from the drive")
    p.add_argument("--hold-other", dest="hold_other", action="store_true", default=True,
                   help="(single-joint only) hold the other joint in position mode at its "
                        "current angle (default on)")
    p.add_argument("--no-hold-other", dest="hold_other", action="store_false")
    p.add_argument("--hold-stiffness", type=float, default=15.0, help="N.m/rad, the held joint's gain")
    p.add_argument("--max-deg", type=float, default=30.0, help="abort if a joint moves more than this")
    p.add_argument("--ignore-start-pose", action="store_true",
                   help="skip the 'within 50mm of test_pose' pre-flight AND the 50mm excursion "
                        "guard for this run. Joint limits, --max-deg, and feedback-age guards "
                        "stay active regardless. For destructive/characterisation testing where "
                        "the arm is intentionally not at test_pose.")
    p.add_argument("--rest", type=float, default=0.0,
                   help="seconds to idle-wait after the step, before this process exits -- lets "
                        "back-to-back runs be scripted from a shell loop with a cool-down")
    p.add_argument("--rate", type=float, default=100.0, help="control loop Hz")
    args = p.parse_args()

    resolved = resolve_preset({"current": args.current}, args.preset)
    args.current = 0.8 if resolved["current"] is None else resolved["current"]

    DATASHEET_STALL_MAX_A = 1.0
    DRIVE_CURRENT_HARD_MAX_A = args.hard_max  # both drives -- odrivetool-only to raise; see --hard-max
    if args.current > DATASHEET_STALL_MAX_A:
        print(f"! WARNING: --current {args.current}A exceeds the EM3215 datasheet stall max "
              f"({DATASHEET_STALL_MAX_A}A) -- proceeding (destructive testing accepted)")
    if args.current > DRIVE_CURRENT_HARD_MAX_A:
        print(f"! WARNING: --current {args.current}A exceeds the drive's current_hard_max "
              f"({DRIVE_CURRENT_HARD_MAX_A}A) -- Set_Limits writes current_soft_max, which the "
              f"drive itself will silently clamp to current_hard_max. Raising current_hard_max "
              f"requires an odrivetool write to axis0.config.motor.current_hard_max; this script "
              f"will not do that. Check the peak-Iq-vs-requested-cap line in the summary below "
              f"to see whether that clamp actually engaged.")

    both = args.joint == "both"
    active_joints = [0, 1] if both else [int(args.joint)]

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    for m in config.motors:
        m.current_soft_max = args.current

    specs = args.torque_mnm.split(",")
    if both:
        if len(specs) == 1:
            specs = specs * 2
        elif len(specs) != 2:
            raise SystemExit("--torque-mnm for --joint both must be 'T' or 'T0,T1'")
    elif len(specs) != 1:
        raise SystemExit("--torque-mnm for a single joint must be one value")

    tau_nm = {}  # joint_idx -> N.m
    for j in active_joints:
        spec = specs[j] if both else specs[0]
        tau_nm[j] = _resolve_torque(spec, args.current, config.motors[j].torque_constant)

    other_idx = None if both else (1 - active_joints[0])

    test_pose_xy = config.test_pose_xy_m
    log = RunLogger("torque_step", interface=config.can.interface, channel=config.can.channel,
                    joint=args.joint, torque_mnm_cmd={j: tau_nm[j] * 1e3 for j in active_joints},
                    duration=args.duration, current_cap=args.current,
                    hold_other=(args.hold_other and not both), hold_stiffness=args.hold_stiffness,
                    max_deg=args.max_deg, max_excursion_mm=MAX_EXCURSION_MM,
                    ignore_start_pose=args.ignore_start_pose, rest_s=args.rest,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=False)
    tau_str = ", ".join(f"{JOINT_NAMES[j]}={tau_nm[j]*1e3:.2f}mN.m" for j in active_joints)
    print(f"opening {config.can.interface}/{config.can.channel}  joints={active_joints}  "
          f"{tau_str}  duration={args.duration}s  cap={args.current}A")
    link.start()

    def idle_all() -> None:
        for m in config.motors:
            try:
                link.set_idle(m.node_id)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {m.node_id}: {exc}", level="ERROR")

    armed = False
    aborted = False
    disarm_summary = None
    default_vel_gain: dict[int, float] = {}
    qd_peak = {j: 0.0 for j in active_joints}
    dq_total = {j: 0.0 for j in active_joints}
    currents_during_step: dict[int, list] = {0: [], 1: []}
    ibus_during_step: dict[int, list] = {0: [], 1: []}
    vbus_during_step: dict[int, list] = {0: [], 1: []}
    fet_temp_during_step: dict[int, list] = {0: [], 1: []}
    i2t: dict[int, float] = {0: 0.0, 1: 0.0}   # A^2.s, integrated over the WHOLE process, not just the step
    fet_temp_start = None
    moved_first_200ms = {j: None for j in active_joints}
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")
        excursion_guard_on = not args.ignore_start_pose
        if excursion_guard_on and test_pose_xy is None:
            raise SystemExit("no test_pose configured in calibration.json -- refusing to run "
                             "without a known excursion reference (or pass --ignore-start-pose "
                             "to explicitly skip this check)")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        if excursion_guard_on:
            excursion0_mm = float(np.linalg.norm(pose0 - test_pose_xy) * 1e3)
            print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
                  f"{excursion0_mm:.1f}mm from test_pose  limits={format_limits_deg(config.motors)}")
            log.event(f"start q_deg={q_deg(q0)} excursion_mm={excursion0_mm:.2f}")
            if excursion0_mm > MAX_EXCURSION_MM:
                raise SystemExit(f"start pose is {excursion0_mm:.1f}mm from test_pose, exceeds "
                                 f"the {MAX_EXCURSION_MM}mm cap -- move the arm back to "
                                 f"test_pose first, or pass --ignore-start-pose")
        else:
            print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
                  f"limits={format_limits_deg(config.motors)}")
            print("  ! --ignore-start-pose: the 50mm excursion guard from test_pose is OFF for "
                  "this run. Joint limits / --max-deg / feedback-age guards remain active.")
            log.event("--ignore-start-pose: excursion guard OFF for this run "
                     f"(start q_deg={q_deg(q0)})", level="WARN")

        problems = check_armable(q0, config.motors)
        if problems:
            raise SystemExit("refusing to arm -- " + "; ".join(problems))

        for i, m in enumerate(config.motors):
            link.set_controller_mode(m.node_id, "position")
            link.set_limits(m.node_id, HOLD_VEL_LIMIT_RAD_S, args.current)
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
        log.event("cleared errors on both nodes")

        for j in active_joints:
            m = config.motors[j]
            default_vel_gain[j] = m.vel_gain
            vel_limit_rad_s, plateau_a = plateau_vel_limit_rad_s(
                args.current, m.torque_constant, TORQUE_VEL_GAIN)
            link.set_vel_gains(m.node_id, TORQUE_VEL_GAIN, 0.0)
            link.set_controller_mode(m.node_id, "torque")
            link.set_limits(m.node_id, vel_limit_rad_s, args.current)
            print(f"  node {m.node_id}: plateau = {plateau_a:.2f}A, cap = {args.current:.2f}A "
                  f"(vel_limit={vel_limit_rad_s:.2f}rad/s)")
            log.event(f"set node {m.node_id} vel_gain -> {TORQUE_VEL_GAIN}, vel_limit -> "
                     f"{vel_limit_rad_s:.3f}rad/s (plateau={plateau_a:.3f}A, cap={args.current:.3f}A); "
                     f"vel_gain restored to {default_vel_gain[j]} at exit")

        start_q = {j: float(q0[j]) for j in active_joints}
        hold_other = args.hold_other and not both
        other_q = float(q0[other_idx]) if hold_other else None

        time.sleep(0.1)  # give Get_Temperature a moment to arrive at least once
        fet_temp_start, motor_temp_start = link.temperatures()
        print(f"  FET temp at start: node0={fet_temp_start[0]:.1f}C node1={fet_temp_start[1]:.1f}C  "
              f"(motor thermistor disabled -> motor_temp NaN/0, expected)")
        log.event(f"fet_temp_start_c={fet_temp_start.tolist()} motor_temp_start_c={motor_temp_start.tolist()}")

        active_node_ids = {config.motors[j].node_id for j in active_joints}
        period = 1.0 / args.rate
        t0 = time.monotonic()
        stepped_on = stepped_off = False
        total_s = total_duration_s(args.duration)

        while True:
            elapsed = time.monotonic() - t0
            phases = {j: schedule(elapsed, args.duration, tau_nm[j]) for j in active_joints}
            if all(ph.name == "done" for ph in phases.values()):
                break

            age_s = link.feedback_age_s()
            if age_s > MAX_FEEDBACK_AGE_S:
                raise Aborted(f"feedback age {age_s*1e3:.1f}ms exceeds cap")

            status = link.node_status()
            node_armed = {s.node_id: s.axis_state == AXIS_STATE_CLOSED_LOOP_CONTROL for s in status}
            for s in status:
                if s.node_id in active_node_ids and not node_armed[s.node_id]:
                    reason = decode_error_flags(s.disarm_reason or 0)
                    msg = (f"drive disarmed at t={elapsed:.3f}s: node {s.node_id} axis_state "
                          f"{s.axis_state} (left CLOSED_LOOP) reason={reason}")
                    log.event(msg, level="ERROR")
                    raise DriveDisarmed(msg)

            q_all, qd_all = link.joint_state()
            check_runtime(q_all, config.motors)
            pose = forward(q_all, config.geo)
            if excursion_guard_on:
                exc_mm = float(np.linalg.norm(pose - test_pose_xy) * 1e3)
                if exc_mm > MAX_EXCURSION_MM:
                    raise Aborted(f"excursion {exc_mm:.1f}mm exceeds cap {MAX_EXCURSION_MM}mm")

            moved_deg = {j: float(np.degrees(q_all[j] - start_q[j])) for j in active_joints}
            for j in active_joints:
                abort_reason = check_abort(moved_deg[j], args.max_deg)
                if abort_reason is not None:
                    raise Aborted(f"joint {j} ({JOINT_NAMES[j]}): {abort_reason}")

            for j in active_joints:
                m = config.motors[j]
                link.set_input_torque(m.node_id, phases[j].torque_nm if phases[j].name != "done" else 0.0)
            if hold_other:
                vel_gain = config.motors[other_idx].vel_gain
                pos_gain = 0.0 if vel_gain <= 0 else min(
                    args.hold_stiffness / vel_gain, config.motors[other_idx].max_pos_gain)
                link.set_input_pos(config.motors[other_idx].node_id, other_q)
                link.set_pos_gain(config.motors[other_idx].node_id, pos_gain)

            any_step = any(ph.name == "step" for ph in phases.values())
            any_post = any(ph.name == "post" for ph in phases.values())
            if any_step and not stepped_on:
                stepped_on = True
                print(f"  -- step ON: {tau_str} --")
                log.event(f"step on: {tau_str}")
            if any_post and not stepped_off:
                stepped_off = True
                print("  -- step OFF --")
                log.event("step off")

            cur = link.motor_currents()
            vbus, ibus = link.bus_voltage_current()
            fet_temp, motor_temp = link.temperatures()
            for j in active_joints:
                qd_peak[j] = max(qd_peak[j], abs(float(qd_all[j])))
                dq_total[j] = moved_deg[j]
                if moved_first_200ms[j] is None and (elapsed - 0.5) >= 0.2:
                    moved_first_200ms[j] = abs(moved_deg[j]) > 0.05  # deg, above encoder noise floor
            for node_i in (0, 1):
                node_id = config.motors[node_i].node_id
                if node_armed.get(node_id, True):
                    # I^2t since process start -- crude thermal budget indicator, not gated to
                    # the step phase (friction/holding current before/after the step still heats
                    # the motor). Skipped while disarmed for the same frozen-Get_Iq reason as
                    # the step-phase stats below.
                    i2t[node_i] += float(cur[node_i]) ** 2 * period
            if any_step:
                for node_i in (0, 1):
                    node_id = config.motors[node_i].node_id
                    if not node_armed.get(node_id, True):
                        continue  # disarmed nodes freeze Get_Iq at its last value -- don't count it
                    currents_during_step[node_i].append(float(cur[node_i]))
                    ibus_during_step[node_i].append(float(ibus[node_i]))
                    vbus_during_step[node_i].append(float(vbus[node_i]))
                    fet_temp_during_step[node_i].append(float(fet_temp[node_i]))

            log.sample(t=elapsed, phase={j: phases[j].name for j in active_joints},
                      q_deg=np.degrees(q_all).tolist(), qd=qd_all.tolist(),
                      tau_cmd_nm={j: phases[j].torque_nm for j in active_joints},
                      currents=cur.tolist(), vbus=vbus.tolist(), ibus=ibus.tolist(),
                      fet_temp_c=fet_temp.tolist(), motor_temp_c=motor_temp.tolist(),
                      pose=pose.tolist(), node_status=link.node_status())
            time.sleep(period)

        print(f"\ndone: {total_s:.1f}s schedule complete")
        log.event("schedule complete")

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
                for j in active_joints:
                    link.set_input_torque(config.motors[j].node_id, 0.0)
                    link.set_pos_gain(config.motors[j].node_id, 0.0)
                if other_idx is not None:
                    link.set_pos_gain(config.motors[other_idx].node_id, 0.0)
            except Exception:  # noqa: BLE001
                pass
            for j, dvg in default_vel_gain.items():
                try:
                    link.set_vel_gains(config.motors[j].node_id, dvg, 0.0)
                    log.event(f"restored node {config.motors[j].node_id} vel_gain -> {dvg}")
                except Exception as exc:  # noqa: BLE001
                    log.event(f"restore vel_gain node {config.motors[j].node_id}: {exc}", level="ERROR")
        idle_all()
        time.sleep(0.2)

        fet_temp_end = motor_temp_end = None
        try:
            fet_temp_end, motor_temp_end = link.temperatures()
        except Exception:  # noqa: BLE001
            pass

        summary_bits = []
        ibus_means = {}
        peak_i_per_node = {}
        for node_i in (0, 1):
            if currents_during_step[node_i]:
                arr = np.abs(np.asarray(currents_during_step[node_i]))
                mean_i, peak_i = float(arr.mean()), float(arr.max())
            else:
                mean_i = peak_i = float("nan")
            peak_i_per_node[node_i] = peak_i
            if ibus_during_step[node_i]:
                barr = np.abs(np.asarray(ibus_during_step[node_i]))
                mean_ib, peak_ib = float(barr.mean()), float(barr.max())
            else:
                mean_ib = peak_ib = float("nan")
            ibus_means[node_i] = mean_ib
            fet_peak = (max(fet_temp_during_step[node_i]) if fet_temp_during_step[node_i]
                       else float("nan"))
            if vbus_during_step[node_i]:
                mean_vb = float(np.mean(vbus_during_step[node_i]))
            else:
                mean_vb = float("nan")
            mean_bus_power_w = mean_ib * mean_vb
            # implied copper loss 1.5*Iq^2*R (3-phase, Iq is the peak-equivalent q-axis
            # current -- 1.5x factor per ODrive's own power accounting) at two candidate
            # per-phase resistances, to see which one the measured bus power matches.
            loss_1p86 = 1.5 * mean_i ** 2 * 1.86
            loss_4p0 = 1.5 * mean_i ** 2 * 4.0
            summary_bits.append(
                f"node{node_i}: Iq mean={mean_i:.3f}A peak={peak_i:.3f}A  "
                f"ibus mean={mean_ib:.3f}A peak={peak_ib:.3f}A  FET peak={fet_peak:.1f}C  "
                f"bus_power mean={mean_bus_power_w:.2f}W  "
                f"copper_loss@1.86ohm={loss_1p86:.2f}W @4.0ohm={loss_4p0:.2f}W  "
                f"I2t={i2t[node_i]:.4f}A2.s")
        ibus_sum = sum(v for v in ibus_means.values() if not np.isnan(v)) if ibus_means else float("nan")
        for j in active_joints:
            print(f"  joint {j} ({JOINT_NAMES[j]}): peak|qd|={qd_peak[j]:.3f}rad/s  "
                  f"total_dq={dq_total[j]:.3f}deg  moved_in_first_200ms={moved_first_200ms[j]}")
            requested_cap = args.current
            print(f"    peak measured Iq={peak_i_per_node[j]:.3f}A vs requested cap={requested_cap:.2f}A"
                  + ("  -- did NOT reach the requested cap (nothing to see re: a silent "
                     "current_hard_max clamp)" if peak_i_per_node[j] < requested_cap * 0.95 else
                     "  -- reached (or exceeded) the requested cap"))
        for line in summary_bits:
            print(f"  {line}")
        if fet_temp_start is not None and fet_temp_end is not None:
            print(f"  FET temp: start=[{fet_temp_start[0]:.1f},{fet_temp_start[1]:.1f}]C  "
                  f"end=[{fet_temp_end[0]:.1f},{fet_temp_end[1]:.1f}]C  "
                  f"(motor thermistor disabled on both drives -> motor_temp NaN/0, expected)")
        print(f"  summed ibus mean (both nodes) = {ibus_sum:.3f}A  "
              f"(supply ammeter should read ~this + board quiescent)")
        if disarm_summary is not None:
            print(f"\n  *** {disarm_summary} ***")
        log.event("summary: " + " | ".join(summary_bits) + f" | ibus_sum={ibus_sum:.4f}"
                 + (f" | fet_temp_end_c={fet_temp_end.tolist()}" if fet_temp_end is not None else "")
                 + (f" | {disarm_summary}" if disarm_summary is not None else ""))

        if args.rest > 0:
            print(f"\nresting {args.rest:.1f}s (idle) before exit...")
            time.sleep(args.rest)

        link.close()
        log.close()
        print(f"log: {log.dir}")

    if aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
