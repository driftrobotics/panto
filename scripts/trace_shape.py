"""Move the impedance anchor smoothly along a closed 2D path (box/circle/line)
in the calibrated tip frame, and log tracking accuracy.

    python -m scripts.trace_shape --shape box --size-mm 25 --speed-mm-s 10 \\
        --preset step-lin-K10
    python -m scripts.trace_shape --shape circle --size-mm 25 --speed-mm-s 15 \\
        --stiffness 15 --vel-gain 0.03
    python -m scripts.trace_shape --sim --shape box --size-mm 25 --speed-mm-s 10 \\
        --preset step-lin-K10
    python -m scripts.trace_shape --backend torque --shape box --size-mm 25 \\
        --speed-mm-s 10 --stiffness 25 --damping 0.7 --vel-lpf-hz 50 --current 0.8

Arms at the current pose (must be within 60mm of `config.test_pose`, same
guard as scripts/step_response.py), ramps the anchor from there to the path's
start point over `--lead-in-s`, then follows `panto.shapes.path_points` once
per tick (`--laps` repeats), logging every sample via
panto.telemetry.RunLogger (t, anchor, pose, q, q_target, currents, tracking
error, path phase). Aborts on joint-limit approach, drive disarm/loss, a
`--max-excursion-mm` excursion from the path centre, or feedback age >30ms --
same guard family as step_response.py/goto_pose.py. Relaxes + IDLEs on any
exit path, then a mandatory `--cooldown-s` thermal pause.

``--backend`` selects the impedance backend: ``position`` (default) drives
the ODrive's own position cascade the way it always has (vel_gain/vel_limit/
pos_gain/cap schedule all apply); ``torque`` instead drives
``panto.backends.torque.TorqueBackend`` -- host-side Cartesian impedance,
mirroring scripts/impedance_step.py (``--damping``, ``--vel-lpf-hz``,
``--notch-hz``/``--notch-q``, current cap from ``--current``). In torque
mode the position-only knobs (``--vel-gain``, ``--vel-limit``, cap-schedule,
``--max-pos-gain``, ``--ff-scale``) are not applied. Default ``--rate`` is
500 Hz in torque mode (250 Hz in position mode) to match impedance_step.py's
host-loop-vs-2ms-encoder-broadcast rationale.

Post-processes with panto.trace_logic.analyze_trace over the whole traced
path (RMS/max tracking error, mean lag via x/y cross-correlation, per-side
RMS + corner overshoot for the box, peak/RMS current, I2t) and prints one
SUMMARY_JSON line (also written to summary.json in the log dir).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from panto.backends import PositionBackend, TorqueBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.constraints import Point
from panto.kinematics import forward
from panto.limits import JointLimitViolation, format_limits_deg, q_deg
from panto.presets import apply_to_config, drive_defaults, resolve as resolve_preset
from panto.shapes import SHAPES, path_length_m, path_points
from panto.step_logic import parse_per_joint, to_per_joint
from panto.telemetry import RunLogger
from panto.trace_logic import analyze_trace

MAX_FEEDBACK_AGE_S = 0.03
MAX_START_OFFSET_MM = 60.0
MAX_HEARTBEAT_AGE_S = 2.0


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    pass


class DriveLost(Aborted):
    pass


def _parse_centre(spec: str | None, config: Config) -> np.ndarray:
    if spec is None:
        xy = config.test_pose_xy_m
        if xy is None:
            raise SystemExit("no test_pose configured and --centre not given")
        return xy
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--centre must be 'x,y' in mm, got {spec!r}")
    return np.array([float(parts[0]), float(parts[1])]) * 1e-3


def main() -> None:
    p = argparse.ArgumentParser(prog="trace_shape", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--sim", action="store_true", help="use the in-process sim bus instead of hardware")
    p.add_argument("--backend", choices=("position", "torque"), default="position",
                   help="impedance backend: position (ODrive position cascade, default) or "
                        "torque (host-side Cartesian impedance, panto.backends.torque.TorqueBackend)")
    p.add_argument("--preset", type=str, default="step-lin-K10",
                   help="named preset from presets.json; supplies any of stiffness/vel-gain/"
                        "vel-limit/current/cap-slope/cap-min/ff-scale/max-pos-gain not given "
                        "explicitly below")
    p.add_argument("--stiffness", type=float, default=None)
    p.add_argument("--vel-gain", type=str, default=None,
                   help="'V' (both joints) or 'V0,V1' (shoulder,elbow)")
    p.add_argument("--vel-limit", type=float, default=None)
    p.add_argument("--current", type=float, default=None)
    p.add_argument("--cap-slope", type=str, default=None)
    p.add_argument("--cap-min", type=str, default=None)
    p.add_argument("--ff-scale", type=float, default=None)
    p.add_argument("--max-pos-gain", type=float, default=None)
    p.add_argument("--damping", type=float, default=0.0,
                   help="torque backend only: isotropic tip damping B, N.s/m")
    p.add_argument("--vel-lpf-hz", type=float, default=20.0,
                   help="torque backend only: low-pass cutoff (Hz) on qd before the damping "
                        "term; <=0 disables")
    p.add_argument("--notch-hz", type=float, default=0.0,
                   help="torque backend only: optional notch centre (Hz) on filtered qd; 0=off")
    p.add_argument("--notch-q", type=float, default=4.0, help="torque backend only: notch quality factor")
    p.add_argument("--slew", type=float, default=0.0,
                   help="torque backend only: per-joint torque slew limit, N.m/s; 0=off")
    p.add_argument("--torque-vel-gain", type=float, default=0.01,
                   help="torque backend only: vel_gain pushed on both nodes in torque mode, to "
                        "raise the vel_limit*vel_gain plateau above the current cap (see "
                        "panto.breakaway_logic.plateau_vel_limit_rad_s)")
    p.add_argument("--shape", choices=sorted(SHAPES), required=True)
    p.add_argument("--size-mm", type=float, default=25.0,
                   help="box side / circle diameter / line length, mm")
    p.add_argument("--speed-mm-s", type=float, default=10.0, help="tangential anchor speed, mm/s")
    p.add_argument("--centre", type=str, default=None,
                   help="'x,y' mm in the calibrated tip frame; default config.test_pose")
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--corner-dwell-s", type=float, default=0.3, help="box only")
    p.add_argument("--lead-in-s", type=float, default=2.0,
                   help="ramp the anchor from the current pose to the path start over this long")
    p.add_argument("--rate", type=float, default=None,
                   help="control loop Hz; default 250 (position) / 500 (torque)")
    p.add_argument("--cooldown-s", type=float, default=30.0)
    p.add_argument("--max-excursion-mm", type=float, default=60.0,
                   help="hard bound on distance from --centre")
    args = p.parse_args()
    if args.rate is None:
        args.rate = 500.0 if args.backend == "torque" else 250.0

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    if args.backend == "torque":
        # torque backend: no position-cascade knobs (vel_gain/vel_limit/
        # pos_gain/cap-schedule) -- mirrors scripts/impedance_step.py, which
        # doesn't use presets/resolve_preset either.
        if args.stiffness is None:
            raise SystemExit("--stiffness is required for --backend torque")
        resolved = {"stiffness": args.stiffness,
                   "vel_gain": None, "vel_limit": None,
                   "current": 0.8 if args.current is None else args.current,
                   "cap_slope": None, "cap_min": None,
                   "ff_scale": None, "max_pos_gain": None}
        cap_slope_per_joint = (0.0, 0.0)
        cap_min_per_joint = (0.8, 0.8)
    else:
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
        resolved["current"] = 0.8 if resolved["current"] is None else resolved["current"]
        cap_slope_per_joint = to_per_joint(resolved["cap_slope"]) or (0.0, 0.0)
        resolved["cap_slope"] = list(cap_slope_per_joint)
        cap_min_per_joint = tuple(resolved["cap_min"]) if resolved["cap_min"] is not None else (0.8, 0.8)
        vel_gain_per_joint = to_per_joint(resolved["vel_gain"])
        resolved["vel_gain"] = list(vel_gain_per_joint)

    for m in config.motors:
        m.current_soft_max = resolved["current"]
    default_vel_gain = drive_defaults(config)
    if args.backend != "torque":
        apply_to_config(config, resolved)

    nodes = [m.node_id for m in config.motors]
    K = resolved["stiffness"] * np.eye(2)
    centre_xy = _parse_centre(args.centre, config)
    size_m = args.size_mm * 1e-3
    speed_m_s = args.speed_mm_s * 1e-3
    dt = 1.0 / args.rate

    test_pose_xy = config.test_pose_xy_m

    log_kwargs = dict(preset=args.preset,
                      stiffness=resolved["stiffness"], current=resolved["current"],
                      shape=args.shape, size_mm=args.size_mm, speed_mm_s=args.speed_mm_s,
                      centre_mm=(centre_xy * 1e3).tolist(), laps=args.laps,
                      corner_dwell_s=args.corner_dwell_s, lead_in_s=args.lead_in_s, rate=args.rate,
                      joint_limits_deg=format_limits_deg(config.motors),
                      backend=args.backend)
    if args.backend == "torque":
        log_kwargs.update(damping=args.damping, vel_lpf_hz=args.vel_lpf_hz,
                          notch_hz=args.notch_hz, notch_q=args.notch_q,
                          slew_nm_s=args.slew, torque_vel_gain=args.torque_vel_gain)
    else:
        log_kwargs.update(vel_gain=resolved["vel_gain"], vel_limit=resolved["vel_limit"],
                          cap_slope=resolved["cap_slope"], cap_min=list(cap_min_per_joint),
                          ff_scale=resolved["ff_scale"], max_pos_gain=resolved["max_pos_gain"])
    log = RunLogger("trace_shape", interface=config.can.interface, channel=config.can.channel,
                    **log_kwargs)
    link = CanLink(config, sim=args.sim)
    if args.backend == "torque":
        backend = TorqueBackend(link, config)
        backend.torque_vel_gain = args.torque_vel_gain
        backend.damping = args.damping
        backend.vel_lpf_hz = args.vel_lpf_hz
        backend.notch_hz = args.notch_hz
        backend.notch_q = args.notch_q
        backend.slew_nm_s = args.slew
    else:
        backend = PositionBackend(link, config)
        backend.vel_limit_rad_s = resolved["vel_limit"]
    est_duration_s = path_length_m(args.shape, size_m, args.laps) / speed_m_s
    if args.backend == "torque":
        print(f"opening {config.can.interface}/{config.can.channel} (sim={args.sim})  backend=torque  "
              f"K={resolved['stiffness']}N/m  B={args.damping}N.s/m  vel_lpf={args.vel_lpf_hz}Hz  "
              f"cap={resolved['current']}A  "
              f"shape={args.shape} size={args.size_mm}mm speed={args.speed_mm_s}mm/s laps={args.laps}  "
              f"est. path duration~{est_duration_s:.1f}s + lead-in {args.lead_in_s:.1f}s")
    else:
        print(f"opening {config.can.interface}/{config.can.channel} (sim={args.sim})  preset={args.preset}  "
              f"K={resolved['stiffness']}N/m  vel_gain={resolved['vel_gain']}  "
              f"vel_limit={resolved['vel_limit']}rad/s  cap={resolved['current']}A  "
              f"shape={args.shape} size={args.size_mm}mm speed={args.speed_mm_s}mm/s laps={args.laps}  "
              f"est. path duration~{est_duration_s:.1f}s + lead-in {args.lead_in_s:.1f}s")
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
    rows: list[dict] = []
    i2t_a2s = {0: 0.0, 1: 0.0}
    period = dt
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        if args.backend == "torque":
            print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
                  f"limits={format_limits_deg(config.motors)}")
        else:
            pos_gain0 = PositionBackend.pos_gains_for(K, q0, config)
            print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm  "
                  f"pos_gain@start={[round(g, 1) for g in pos_gain0]}  "
                  f"limits={format_limits_deg(config.motors)}")
        log.event(f"start q_deg={q_deg(q0)}")

        if test_pose_xy is not None:
            start_offset_mm = float(np.linalg.norm(pose0 - test_pose_xy) * 1e3)
            print(f"  {start_offset_mm:.1f}mm from test_pose (cap {MAX_START_OFFSET_MM}mm)")
            if start_offset_mm > MAX_START_OFFSET_MM:
                raise SystemExit(f"start pose is {start_offset_mm:.1f}mm from test_pose, exceeds "
                                 f"the {MAX_START_OFFSET_MM}mm cap -- move the arm back first")

        path = path_points(args.shape, size_m, centre_xy, speed_m_s, dt,
                           laps=args.laps, corner_dwell=args.corner_dwell_s, start_xy=pose0)
        path_t = path[:, 0]
        path_xy = path[:, 1:3]
        path_end_s = float(path_t[-1])
        total_s = args.lead_in_s + path_end_s

        # phase labels: "lead_in" during ramp-in, then for box "side_N"/"corner"
        # (corner = near-zero displacement runs at the start/end of each side,
        # identified by matching each path sample to its nearest corner), else
        # just the shape name.
        phase_labels: list[str] = []
        if args.shape == "box":
            half = size_m / 2.0
            corners = np.array([[half, -half], [half, half], [-half, half], [-half, -half]]) + centre_xy
            corner_tol_m = max(1e-4, 0.15 * (speed_m_s * args.corner_dwell_s + 1e-6))
            for xy in path_xy:
                dists = np.linalg.norm(corners - xy, axis=1)
                nearest = int(np.argmin(dists))
                if dists[nearest] <= corner_tol_m:
                    phase_labels.append("corner")
                else:
                    # side = the side departing from `nearest`'s predecessor;
                    # approximate by nearest side segment
                    seg_d = []
                    for i in range(4):
                        a, b = corners[i], corners[(i + 1) % 4]
                        ab = b - a
                        t_proj = np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0.0, 1.0)
                        proj = a + t_proj * ab
                        seg_d.append(np.linalg.norm(xy - proj))
                    phase_labels.append(f"side_{int(np.argmin(seg_d))}")
        else:
            phase_labels = [args.shape] * len(path_xy)

        if args.backend != "torque":
            # set gains BEFORE arming so the very first closed-loop tick
            # already uses the resolved vel_gain/vel_limit, not the config
            # default.
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
        if args.backend != "torque":
            for i, nid in enumerate(nodes):
                link.set_vel_gains(nid, resolved["vel_gain"][i], 0.0)

        path_start_xy = path_xy[0]
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
                          f"{s.age_s:.2f}s exceeds {MAX_HEARTBEAT_AGE_S:.1f}s cap")
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

            if t < args.lead_in_s:
                phase = "lead_in"
                frac = (t / args.lead_in_s) if args.lead_in_s > 0 else 1.0
                anchor = pose0 + frac * (path_start_xy - pose0)
            else:
                tp = t - args.lead_in_s
                idx = int(np.searchsorted(path_t, tp, side="left"))
                idx = min(idx, len(path_xy) - 1)
                anchor = path_xy[idx]
                phase = phase_labels[idx]

            if phase != last_phase:
                print(f"  -- t={t:5.2f}s phase={phase} --")
                log.event(f"phase -> {phase} at t={t:.3f}s")
                last_phase = phase

            constraint = Point(at=anchor)
            q, qd = link.joint_state()
            pose = forward(q, config.geo)

            exc_mm = float(np.linalg.norm(pose - centre_xy) * 1e3)
            if exc_mm > args.max_excursion_mm:
                raise Aborted(f"excursion {exc_mm:.1f}mm exceeds cap {args.max_excursion_mm}mm")

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
            err_mm = float(np.linalg.norm(pose - anchor) * 1e3)

            rows.append({"t": t, "phase": phase, "cmd_xy": anchor.tolist(),
                        "pose_xy": pose.tolist(), "err_mm": err_mm, "currents": cur.tolist()})

            log.sample(t=t, phase=phase, q=q, qd=qd, pose=pose, anchor=proj.anchor,
                      currents=cur, sent=sent, node_status=status,
                      tracking_error_mm=err_mm, feedback_age_ms=age_s * 1e3)

            time.sleep(period)

        print(f"\ndone: {total_s:.1f}s schedule complete")
        log.event("schedule complete")

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
            print("\n! bus lost -- skipping relax/idle/vel-gain-restore writes")
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

        summary_config: dict = {"stiffness": resolved["stiffness"], "current": resolved["current"],
                               "shape": args.shape, "size_mm": args.size_mm,
                               "speed_mm_s": args.speed_mm_s, "laps": args.laps,
                               "corner_dwell_s": args.corner_dwell_s, "lead_in_s": args.lead_in_s}
        if args.backend == "torque":
            summary_config.update(damping=args.damping, vel_lpf_hz=args.vel_lpf_hz,
                                  notch_hz=args.notch_hz, notch_q=args.notch_q,
                                  slew_nm_s=args.slew, torque_vel_gain=args.torque_vel_gain)
        else:
            summary_config.update(vel_gain=resolved["vel_gain"], vel_limit=resolved["vel_limit"],
                                  cap_slope=resolved["cap_slope"], cap_min=list(cap_min_per_joint),
                                  ff_scale=resolved["ff_scale"], max_pos_gain=resolved["max_pos_gain"])
        summary: dict = {"preset": args.preset, "backend": args.backend,
                        "config": summary_config,
                        "aborted": aborted, "abort_reason": abort_reason, "bus_lost": bus_lost,
                        "i2t_a2s": [i2t_a2s[0], i2t_a2s[1]]}

        trace_rows = [r for r in rows if r["phase"] != "lead_in"]
        if trace_rows:
            t_arr = np.array([r["t"] for r in trace_rows])
            t_arr = t_arr - t_arr[0]
            cmd_arr = np.array([r["cmd_xy"] for r in trace_rows])
            pose_arr = np.array([r["pose_xy"] for r in trace_rows])
            cur_arr = np.array([r["currents"] for r in trace_rows])
            phase_arr = [r["phase"] for r in trace_rows]
            metrics = analyze_trace(t_arr, cmd_arr, pose_arr, cur_arr, period,
                                    shape=args.shape, phase=phase_arr, aborted=aborted)
            summary.update({
                "rms_error_mm": metrics.rms_error_mm,
                "max_error_mm": metrics.max_error_mm,
                "mean_lag_s": metrics.mean_lag_s,
                "lag_x_s": metrics.lag_x_s,
                "lag_y_s": metrics.lag_y_s,
                "per_side_rms_mm": metrics.per_side_rms_mm,
                "corner_overshoot_mm": metrics.corner_overshoot_mm,
                "peak_current_a": metrics.peak_current_a,
                "rms_current_a": metrics.rms_current_a,
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
