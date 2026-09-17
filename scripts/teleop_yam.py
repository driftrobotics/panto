"""Force-feedback teleop POC: panto j0/j1 (leader) -> i2rt YAM joint2/joint3 (follower).

Follower joints are --yam-joints, ZERO-indexed (default 1,2 = i2rt joint2 shoulder pitch +
joint3 elbow: two parallel hinge axes, like panto). Help text below that says "J1,J2" means
"first,second follower joint".

    python -m scripts.teleop_yam --sim                  # both robots simulated, no CAN
    python -m scripts.teleop_yam --observe              # YAM only: grav-comp idle, print pose + effort noise
    python -m scripts.teleop_yam --no-reflect           # position-position coupling only
    python -m scripts.teleop_yam                        # coupling + effort reflection
    python -m scripts.teleop_yam --stop                 # E-STOP a running instance from another shell

One process, one ~200 Hz loop, both buses (panto: config CAN channel; YAM:
--yam-channel). Control law and fault policy live in ``panto/teleop.py``.
Needs i2rt importable (``pip install -e ~/code/i2rt`` into this venv).

--observe ENERGISES the YAM (i2rt enables the motors into gravity-comp idle --
the DM motors do not report position otherwise) but commands no motion. Move
the arm by hand through the poses you consider safe; the min/max J1/J2 swept is
printed at exit as a ready-made --yam-box. NOTE: with --gripper linear_4310
i2rt runs its gripper limit calibration at startup (the gripper opens/closes)
unless --gripper-limits is given.

E-stop (every path ends in the same ``_safe_stop``): Ctrl-C / SIGTERM, Enter on
stdin, ``--stop`` from another shell (SIGTERM via logs/teleop_yam.pid), any
``TeleopMonitor`` fault, a panto drive leaving CLOSED_LOOP, joint-limit
violation, I²t trip, oscillation guard, or any exception. Safe stop = YAM ->
gravity-comp idle (NOT zero torque: J2 carries the arm against gravity), panto
-> relax + IDLE. The YAM then keeps floating in grav-comp idle until you have
lowered it and typed 'release' (i2rt's close() zeroes torques -> the arm drops,
so it is never called on a timer).

Every tick is logged via ``panto.telemetry.RunLogger`` (logs/teleop_yam-*/).
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from panto.backends import PositionBackend
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.limits import JointLimitViolation, check_armable, check_runtime, clamp_targets, q_deg
from panto.telemetry import LOG_ROOT, RunLogger
from panto.teleop import BuzzDetector, EffortReflector, JointMap, RateLimiter, TeleopLimits, TeleopMonitor

PID_FILE = LOG_ROOT / "teleop_yam.pid"
_VEL_GAIN_MAX = 0.05     # same ceilings as Runtime.set_tuning
_CURRENT_CAP_MAX = 2.0
_GUARD_GRACE_S = 1.0


class EStop(Exception):
    """Operator or monitor requested a stop; the message is the reason."""


def _pair(text: str) -> np.ndarray:
    parts = [float(x) for x in text.split(",")]
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'a' or 'a,b', got {text!r}")
    return np.array(parts)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="teleop_yam", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stop", action="store_true", help="E-STOP the running instance and exit")
    p.add_argument("--sim", action="store_true", help="simulate both robots (no CAN)")
    p.add_argument("--observe", action="store_true", help="YAM only, grav-comp idle, no panto, no motion")
    p.add_argument("--wiggle", action="store_true",
                   help="YAM only: slow sine on J1/J2 about the hand-placed pose (breakaway / friction ID)")
    p.add_argument("--wiggle-deg", type=float, default=5.0)
    p.add_argument("--wiggle-hz", type=float, default=0.2)
    p.add_argument("--no-reflect", action="store_true", help="position-position coupling only")
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until e-stop")
    p.add_argument("--rate", type=float, default=250.0,
                   help="loop + log Hz. 250 = i2rt's own motor-chain rate (CONTROL_FREQ), the fastest new YAM\n"
                        "data arrives; every tick is logged with both robots' feedback stamps")
    p.add_argument("--config", help="panto live-override config json")
    p.add_argument("--interface")
    p.add_argument("--channel")
    # follower
    p.add_argument("--yam-channel", default="can_yam")
    p.add_argument("--yam-joints", type=lambda t: [int(x) for x in t.split(",")], default=[1, 2],
                   help="ZERO-indexed follower joints for panto j0,j1. Default 1,2 = i2rt joint2 (shoulder "
                        "pitch) + joint3 (elbow); 0 is the base yaw")
    p.add_argument("--arm", default="yam")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--gripper-limits", type=_pair, default=None,
                   help="'closed,open' motor rad: skips i2rt's startup gripper calibration")
    p.add_argument("--scale", type=_pair, default=np.array([-1.0, 1.0]),
                   help="YAM rad per panto rad, signed (rig 2026-09-17: shoulder reversed): 's' or 's1,s2'")
    p.add_argument("--yam-range-deg", type=_pair, default=np.array([45.0, 45.0]),
                   help="J1,J2 box half-width around the engage pose (ignored with --yam-box)")
    p.add_argument("--yam-abs-box-deg", type=str, default="10,150,10,135",
                   help="always-on absolute box 'lo1,hi1,lo2,hi2' deg, inside the 2026-09-17 hand sweep\n"
                        "(joint2 0..174, joint3 0..145); the range/box above is intersected with it")
    p.add_argument("--no-box", action="store_true",
                   help="drop the range/abs software boxes; only i2rt's joint limits (less a 0.2 rad margin:\n"
                        "i2rt KILLS its control thread on a limit violation and the arm goes limp) remain")
    p.add_argument("--yam-box-deg", type=str, default=None,
                   help="absolute J1/J2 box 'lo1,hi1,lo2,hi2' deg (e.g. from --observe)")
    p.add_argument("--yam-kp", type=_pair, default=np.array([80.0, 80.0]),
                   help="J1,J2 MIT kp (i2rt default 80; softer does not break J2 stiction)")
    p.add_argument("--yam-kd", type=_pair, default=np.array([5.0, 5.0]))
    p.add_argument("--yam-rate-deg-s", type=float, default=120.0, help="follower target slew limit")
    # leader
    p.add_argument("--couple-k", type=float, default=0.3, help="leader coupling spring, N.m/rad")
    p.add_argument("--current", type=float, default=0.8, help="panto per-axis current cap, A")
    p.add_argument("--vel-gain", type=_pair, default=None, help="override panto vel_gain 'v' or 'v0,v1'")
    # reflection
    p.add_argument("--alpha", type=_pair, default=np.array([0.01, 0.01]),
                   help="panto N.m per YAM N.m of external torque")
    p.add_argument("--cutoff-hz", type=float, default=8.0)
    p.add_argument("--deadband-nm", type=_pair, default=np.array([2.3, 2.7]),
                   help="joint2,joint3: covers the free-motion friction envelope (2026-09-17 wiggle at 65/65 deg:\n"
                        "2.13 / 2.48 N.m about the bias)")
    # 2026-09-17 --wiggle at J2~70 deg, kp 80: free-motion tau_ext J1 +0.43/-0.40, J2 +0.56/-2.06 N.m
    p.add_argument("--yam-friction-nm", type=_pair, default=np.array([0.0, 0.0]),
                   help="J1,J2 Coulomb friction removed from tau_ext (x tanh(qd/0.05)); off: stick phases at\n"
                        "reversals make the residual worse than a plain deadband (wiggle: 0.42 / 1.3 N.m)")
    p.add_argument("--yam-bias-nm", type=_pair, default=np.array([-0.7, 0.95]),
                   help="J1,J2 constant tau_ext bias under PD (gravity-model error; pose dependent)")
    p.add_argument("--sim-tau", type=_pair, default=None, help="--sim only: fake YAM external torque, N.m")
    # faults
    p.add_argument("--max-err-deg", type=float, default=20.0)
    p.add_argument("--max-vel-deg-s", type=float, default=200.0)
    p.add_argument("--max-eff-nm", type=float, default=16.0,
                   help="|tau_ext| fault. kp 80 x 8 deg of ordinary dynamic lag is already 11 N.m (8 N.m\n"
                        "false-tripped 2026-09-17); DM4340 peak is 27 N.m")
    return p.parse_args()


def _send_stop() -> None:
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, signal.SIGTERM)
        print(f"E-STOP sent to teleop_yam pid {pid}")
    except (FileNotFoundError, ValueError):
        raise SystemExit("no running teleop_yam (no pid file)")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        raise SystemExit("stale pid file removed; no running teleop_yam")


class _StopFlag:
    """SIGINT/SIGTERM + Enter-on-stdin -> one reason string."""

    def __init__(self) -> None:
        self.reason: str | None = None
        signal.signal(signal.SIGINT, lambda *_: self.set("operator:SIGINT"))
        signal.signal(signal.SIGTERM, lambda *_: self.set("operator:SIGTERM"))

    def watch_stdin(self) -> None:
        """Enter = e-stop from here on (started after the engage prompt)."""
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._watch_stdin, daemon=True).start()

    def set(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason

    def _watch_stdin(self) -> None:
        while self.reason is None:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                sys.stdin.readline()
                self.set("operator:enter")


class _Follower:
    """Thin wrapper over the i2rt robot: J1/J2 commanded, everything else held."""

    def __init__(self, args: argparse.Namespace) -> None:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.robot = get_yam_robot(
            channel=args.yam_channel,
            arm_type=ArmType.from_string_name(args.arm),
            gripper_type=GripperType.from_string_name(args.gripper),
            zero_gravity_mode=True,
            gripper_limits_override=args.gripper_limits,
            sim=args.sim,
        )
        self._sim = args.sim
        self.n = self.robot.num_dofs()
        self.hold = np.asarray(self.robot.get_joint_pos(), float).copy()
        kp = np.asarray(getattr(self.robot, "_kp", np.zeros(self.n)), float).copy()
        kd = np.asarray(getattr(self.robot, "_kd", np.zeros(self.n)), float).copy()
        self.idx = list(args.yam_joints)
        if len(self.idx) != 2 or len(set(self.idx)) != 2 or not all(0 <= i < 6 for i in self.idx):
            raise SystemExit("--yam-joints needs two distinct arm joints in 0..5")
        self.held = [i for i in range(6) if i not in self.idx]
        self.names = "/".join(f"joint{i + 1}" for i in self.idx)
        kp[self.idx], kd[self.idx] = args.yam_kp, args.yam_kd
        kp[6:], kd[6:] = 0.0, 0.0   # gripper stays limp: never drive it toward a hold value
        self._kp, self._kd = kp, kd
        info = self.robot.get_robot_info() if hasattr(self.robot, "get_robot_info") else {}
        self.joint_limits = np.asarray(info.get("joint_limits"), float) if info.get("joint_limits") is not None \
            else None

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """-> (q[n], qd[n], tau_ext[n], age_s). tau_ext = reported effort minus the
        feedforward (gravity comp) torque i2rt last sent."""
        obs = self.robot.get_observations()
        q = np.concatenate([obs["joint_pos"], obs.get("gripper_pos", [])]).astype(float)
        qd = np.concatenate([obs["joint_vel"], obs.get("gripper_vel", [])]).astype(float)
        eff = np.concatenate([obs["joint_eff"], obs.get("gripper_eff", [])]).astype(float)
        ff = self.robot.get_motor_torques()
        tau_ext = eff - (np.asarray(ff, float) if ff is not None else 0.0)
        stamp = getattr(getattr(self.robot, "_joint_state", None), "timestamp", None)
        self.stamp = stamp
        age = max(0.0, time.time() - float(stamp)) if stamp else 0.0
        return q, qd, tau_ext, age

    def alive(self) -> bool:
        thread = getattr(self.robot, "_server_thread", None)
        return thread is None or thread.is_alive()

    def command(self, q12: np.ndarray) -> None:
        pos = self.hold.copy()
        pos[self.idx] = q12
        self.robot.command_joint_state({"pos": pos, "vel": np.zeros(self.n), "kp": self._kp, "kd": self._kd})

    def idle(self) -> None:
        """Gravity-comp idle: the arm floats instead of dropping."""
        if hasattr(self.robot, "enter_gravity_comp_idle"):
            self.robot.enter_gravity_comp_idle()
        elif hasattr(self.robot, "enable_gravity_comp"):
            self.robot.enable_gravity_comp()

    def close(self) -> None:
        self.robot.close()


def _hold_until_released(follower: "_Follower", log: RunLogger) -> None:
    """i2rt's close() zeroes all torques and the arm drops, so never close on a
    timer: float in gravity-comp idle until the operator has lowered the arm and
    says so. Signals are ignored here on purpose -- a second Ctrl-C must not
    drop the arm. No tty -> hold until the rest pose is reached by hand."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    log.event("YAM is FLOATING in gravity-comp idle and will stay so. Lower it to the table by hand, "
              "then type 'release' + Enter to de-energise.", level="WARN")
    tty = sys.stdin is not None and sys.stdin.isatty()
    while True:
        if not follower.alive():
            log.event("i2rt control thread is dead -- YAM is NOT being held", level="ERROR")
            return
        follower.idle()
        if tty:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready and sys.stdin.readline().strip().lower() == "release":
                return
        else:
            q, qd, _, _ = follower.read()
            if abs(q[1]) < np.radians(5.0) and np.max(np.abs(qd[:6])) < 0.02:
                return
            time.sleep(0.2)


def _wiggle(args: argparse.Namespace, stop: _StopFlag) -> None:
    """One joint at a time (J1 then J2), 2 periods each, amplitude ramped in."""
    log = RunLogger("teleop_yam_wiggle", **vars(args))
    follower = _Follower(args)
    monitor = TeleopMonitor(TeleopLimits(follower_err_rad=np.radians(10.0),
                                         follower_vel_rad_s=np.radians(args.max_vel_deg_s),
                                         follower_eff_nm=args.max_eff_nm))
    reason = "done"
    try:
        follower.idle()
        if sys.stdin is not None and sys.stdin.isatty():
            input(">>> YAM floating. Hand-place it (clear of the table by > wiggle amplitude), LET GO, "
                  "then press Enter to start the wiggle: ")
        stop.watch_stdin()
        follower.hold = np.asarray(follower.robot.get_joint_pos(), float).copy()
        q0 = follower.hold[follower.idx].copy()
        amp, w = np.radians(args.wiggle_deg), 2 * np.pi * args.wiggle_hz
        seg = 2.0 / args.wiggle_hz
        log.event(f"wiggle +-{args.wiggle_deg} deg @ {args.wiggle_hz} Hz about {np.round(np.degrees(q0), 1).tolist()}")
        t0 = last = time.monotonic()
        next_print = 0.0
        while True:
            now = time.monotonic()
            t, dt, last = now - t0, now - last, now
            if stop.reason is not None:
                raise EStop(stop.reason)
            if t >= 2 * seg:
                break
            if not follower.alive():
                raise EStop("follower:i2rt control thread died")
            j = 0 if t < seg else 1
            yj = follower.idx[j]
            ts = t - j * seg
            cmd = q0.copy()
            cmd[j] += amp * min(1.0, ts / 2.0) * np.sin(w * ts)
            follower.command(cmd)
            q, qd, tau_ext, age = follower.read()
            fault = monitor.check(q_cmd=cmd, q_meas=q[follower.idx], qd_meas=qd[follower.idx],
                                  tau_ext=tau_ext[follower.idx],
                                  held_err=q[follower.held] - follower.hold[follower.held], leader_age_s=0.0,
                                  follower_age_s=age, tick_s=dt)
            log.sample(t=t, dt=dt, joint=yj, yam_q=q, yam_qd=qd, yam_q_cmd=cmd, yam_tau_ext=tau_ext, fault=fault)
            if fault is not None:
                raise EStop(f"monitor:{fault}")
            if t >= next_print:
                next_print = t + 0.5
                print(f"  t={t:5.1f}s joint{yj+1} cmd={np.degrees(cmd[j]):7.2f} meas={np.degrees(q[yj]):7.2f} "
                      f"tau_ext={tau_ext[yj]:6.2f} N.m")
            time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - now)))
    except EStop as exc:
        reason = str(exc)
    finally:
        stop.set(reason)
        log.event(f"E-STOP: {reason}", level="WARN")
        follower.idle()
        if not args.sim:
            _hold_until_released(follower, log)
        follower.close()
        log.close()


def _observe(args: argparse.Namespace, stop: _StopFlag) -> None:
    log = RunLogger("teleop_yam_observe", yam_channel=args.yam_channel, arm=args.arm, gripper=args.gripper)
    follower = _Follower(args)
    stop.watch_stdin()
    lo, hi = np.full(follower.n, np.inf), np.full(follower.n, -np.inf)
    ext: list[np.ndarray] = []
    try:
        follower.idle()
        log.event("YAM in gravity-comp idle; move it by hand through safe poses. Enter / Ctrl-C to finish.")
        t0, next_print = time.monotonic(), 0.0
        while stop.reason is None and follower.alive():
            t = time.monotonic() - t0
            if args.duration and t >= args.duration:
                break
            q, qd, tau_ext, age = follower.read()
            lo, hi = np.minimum(lo, q), np.maximum(hi, q)
            ext.append(tau_ext[follower.idx].copy())
            log.sample(t=t, yam_q=q, yam_qd=qd, yam_tau_ext=tau_ext, yam_age_s=age)
            if t >= next_print:
                next_print = t + 0.5
                print(f"  t={t:6.1f}s  q_deg={np.round(np.degrees(q[:6]), 1).tolist()}  "
                      f"tau_ext {follower.names}={np.round(tau_ext[follower.idx], 2).tolist()} N.m")
            time.sleep(1.0 / args.rate)
    finally:
        stop.set('done')
        follower.idle()
        e = np.array(ext) if ext else np.zeros((1, 2))
        a, b = follower.idx
        box = ",".join(f"{v:.1f}" for v in np.degrees([lo[a], hi[a], lo[b], hi[b]]))
        log.event(f"stop: {stop.reason or 'done'}")
        log.event(f"swept q_deg lo={np.round(np.degrees(lo), 1).tolist()} hi={np.round(np.degrees(hi), 1).tolist()}")
        log.event(f"tau_ext {follower.names} mean={np.round(e.mean(0), 3).tolist()} std={np.round(e.std(0), 3).tolist()} "
                  f"absmax={np.round(np.abs(e).max(0), 3).tolist()} N.m")
        log.event(f"suggested: --yam-box-deg {box}")
        if not args.sim:
            _hold_until_released(follower, log)
        follower.close()
        log.close()


def main() -> None:
    args = _parse()
    if args.stop:
        _send_stop()
        return
    if not 0.0 < args.current <= _CURRENT_CAP_MAX:
        raise SystemExit(f"--current must be in (0, {_CURRENT_CAP_MAX}] A")
    if args.vel_gain is not None and np.any((args.vel_gain <= 0) | (args.vel_gain > _VEL_GAIN_MAX)):
        raise SystemExit(f"--vel-gain must be in (0, {_VEL_GAIN_MAX}]")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        try:
            os.kill(int(PID_FILE.read_text()), 0)
            raise SystemExit(f"teleop_yam already running (pid {PID_FILE.read_text().strip()}); --stop it first")
        except (ProcessLookupError, ValueError):
            PID_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    stop = _StopFlag()
    try:
        if args.wiggle:
            _wiggle(args, stop)
        elif args.observe:
            _observe(args, stop)
        else:
            _teleop(args, stop)
    finally:
        PID_FILE.unlink(missing_ok=True)


def _teleop(args: argparse.Namespace, stop: _StopFlag) -> None:
    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    for i, m in enumerate(config.motors):
        m.current_soft_max = args.current
        if args.vel_gain is not None:
            m.vel_gain = float(args.vel_gain[i])
    nodes = [m.node_id for m in config.motors]
    tau_max = args.current * min(float(m.torque_constant) for m in config.motors)

    log = RunLogger("teleop_yam", **{k: v for k, v in vars(args).items()}, tau_max_nm=tau_max)
    link = CanLink(config, sim=args.sim)
    backend = PositionBackend(link, config)
    follower: _Follower | None = None
    armed = False

    def _safe_stop(reason: str) -> None:
        """The one exit path. Follower first: it is the one that can hurt."""
        log.event(f"E-STOP: {reason}", level="WARN")
        if follower is not None:
            try:
                follower.idle()
                log.event("YAM -> gravity-comp idle")
            except Exception as exc:  # noqa: BLE001
                log.event(f"YAM idle failed: {exc!r}", level="ERROR")
        try:
            backend.relax()
        except Exception as exc:  # noqa: BLE001
            log.event(f"panto relax failed: {exc!r}", level="ERROR")
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                log.event(f"panto idle node {nid}: {exc!r}", level="ERROR")
        log.event("panto -> IDLE")

    reason = "done"
    try:
        link.start()
        link.wait_for_feedback(timeout=5.0, wait_for_errors=not args.sim)
        q_p0, _ = link.joint_state()
        problems = check_armable(q_p0, config.motors)
        if problems:
            raise EStop("refusing to arm panto: " + "; ".join(problems))

        follower = _Follower(args)
        if not args.sim and sys.stdin is not None and sys.stdin.isatty():
            follower.idle()
            input(">>> YAM is floating in gravity-comp idle. Hand-place it at the pose that should "
                  "correspond to panto's current pose (mid-box), hold panto, then press Enter to ENGAGE: ")
            follower.hold = np.asarray(follower.robot.get_joint_pos(), float).copy()
            q_p0, _ = link.joint_state()
        stop.watch_stdin()
        q_y, _, _, _ = follower.read()
        q_y0 = q_y[follower.idx].copy()
        if args.yam_box_deg:
            b = np.radians([float(x) for x in args.yam_box_deg.split(",")])
            box_lo, box_hi = b[[0, 2]], b[[1, 3]]
        else:
            half = np.radians(args.yam_range_deg)
            box_lo, box_hi = q_y0 - half, q_y0 + half
        if args.no_box:
            box_lo, box_hi = np.full(2, -np.inf), np.full(2, np.inf)
        else:
            ab = np.radians([float(x) for x in args.yam_abs_box_deg.split(",")])
            box_lo, box_hi = np.maximum(box_lo, ab[[0, 2]]), np.minimum(box_hi, ab[[1, 3]])
        if follower.joint_limits is not None:
            box_lo = np.maximum(box_lo, follower.joint_limits[follower.idx, 0] + 0.2)
            box_hi = np.minimum(box_hi, follower.joint_limits[follower.idx, 1] - 0.2)
        elif args.no_box:
            raise EStop("--no-box needs i2rt joint limits and none were reported")
        try:
            jmap = JointMap(scale=args.scale, leader_zero=q_p0, follower_zero=q_y0,
                            follower_lo=box_lo, follower_hi=box_hi)
        except ValueError as exc:
            raise EStop(f"refusing to engage: {exc} (yam {np.round(np.degrees(q_y0), 1).tolist()} deg, box "
                        f"{np.round(np.degrees([box_lo, box_hi]), 1).tolist()})")
        limiter = RateLimiter(np.radians(args.yam_rate_deg_s), q_y0)
        reflector = EffortReflector(args.alpha, args.scale, cutoff_hz=args.cutoff_hz,
                                    deadband_nm=args.deadband_nm, tau_max_nm=tau_max)
        monitor = TeleopMonitor(TeleopLimits(follower_err_rad=np.radians(args.max_err_deg),
                                             follower_vel_rad_s=np.radians(args.max_vel_deg_s),
                                             follower_eff_nm=args.max_eff_nm))
        # No OscillationGuard here: both of its checks assume a stationary hold. A moving
        # leader is pose variance, and a leader pushed ahead of a sticky follower sits at the
        # current cap by design (2026-09-17 false trips). BuzzDetector + I2t cover the real faults.
        buzz = BuzzDetector()
        log.event(f"engage: panto q_deg={q_deg(q_p0)}  yam q_deg={np.round(np.degrees(q_y), 1).tolist()}  "
                  f"box_deg={np.round(np.degrees([box_lo, box_hi]), 1).tolist()}")

        # Baseline: tau_ext while stationary in grav-comp idle is model error, not contact.
        samples = []
        for _ in range(int(0.5 * args.rate)):
            samples.append(follower.read()[2][follower.idx])
            time.sleep(1.0 / args.rate)
        tau_bias = np.mean(samples, axis=0)
        log.event(f"tau_ext baseline {follower.names} = {np.round(tau_bias, 3).tolist()} N.m "
                  f"(std {np.round(np.std(samples, axis=0), 3).tolist()})")

        backend.enter()
        for i, nid in enumerate(nodes):
            link.set_input_pos(nid, float(q_p0[i]))
        log.event(f">>> panto entering CLOSED_LOOP_CONTROL; YAM {follower.names} under PD <<<")
        try:
            link.enter_closed_loop(timeout=5.0)
        except CanLinkError as exc:
            raise EStop(f"refusing to arm: {exc}")
        armed = True
        for i, nid in enumerate(nodes):
            link.set_vel_gains(nid, float(config.motors[i].vel_gain), 0.0)
        pos_gains = [PositionBackend._pos_gain(m, args.couple_k) for m in config.motors]
        vel_limit = backend.vel_limit_rad_s
        follower.command(q_y0)

        period = 1.0 / args.rate
        t0 = last = time.monotonic()
        next_print = 0.0
        i2t = np.zeros(2)
        i_cont2, budget = float(config.thermal.i_continuous) ** 2, float(config.thermal.budget_a2s)
        while True:
            now = time.monotonic()
            t, dt, last = now - t0, now - last, now
            if stop.reason is not None:
                raise EStop(stop.reason)
            if args.duration and t >= args.duration:
                break
            if not follower.alive():
                raise EStop("follower:i2rt control thread died")
            for s in link.node_status():
                if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    raise EStop(f"drive:axis{s.node_id}:{decode_error_flags(s.disarm_reason or 0)}")

            q_p, qd_p = link.joint_state()
            check_runtime(q_p, config.motors)
            q_y, qd_y, tau_ext_all, y_age = follower.read()
            tau_raw = tau_ext_all[follower.idx] - tau_bias
            tau_ext = tau_raw - args.yam_bias_nm - args.yam_friction_nm * np.tanh(qd_y[follower.idx] / 0.05)
            if args.sim and args.sim_tau is not None:
                tau_ext = tau_ext + args.sim_tau

            # follower: mapped, boxed, slew-limited
            q_y_target, boxed = jmap.to_follower(q_p)
            q_y_cmd = limiter.step(q_y_target, dt)
            follower.command(q_y_cmd)

            # leader: spring to the mapped-back measured follower + reflected torque
            q_p_target = clamp_targets(jmap.to_leader(q_y[follower.idx]), config.motors)
            tau_reflect = np.zeros(2) if args.no_reflect else reflector.step(tau_ext, dt)
            tau_ff = []
            for i, motor in enumerate(config.motors):
                ff = float(tau_reflect[i]) + PositionBackend._hold_ff(motor, q_p)
                link.set_input_pos(motor.node_id, float(q_p_target[i]), torque_ff_nm=ff)
                link.set_pos_gain(motor.node_id, pos_gains[i])
                link.set_limits(motor.node_id, vel_limit, args.current)
                tau_ff.append(ff)

            cur = link.motor_currents()
            i2t = np.clip(i2t + (np.asarray(cur, float) ** 2 - i_cont2) * dt, 0.0, None)
            if budget > 0 and np.any(i2t > budget):
                raise EStop(f"over_current_i2t:{np.round(i2t, 1).tolist()}>{budget}")
            if t >= _GUARD_GRACE_S:
                buzzing = buzz.step(t, q_p)
                if buzzing is not None:
                    raise EStop(f"osc_guard:{buzzing}")
            fault = monitor.check(q_cmd=q_y_cmd, q_meas=q_y[follower.idx], qd_meas=qd_y[follower.idx],
                                  tau_ext=tau_ext,
                                  held_err=q_y[follower.held] - follower.hold[follower.held],
                                  leader_age_s=link.feedback_age_s(), follower_age_s=y_age,
                                  tick_s=dt)
            log.sample(node_status=link.node_status(), t=t, dt=dt, panto_q=q_p, panto_qd=qd_p,
                       panto_q_target=q_p_target, panto_tau_ff=tau_ff, panto_iq=cur, i2t=i2t,
                       yam_q=q_y, yam_qd=qd_y, yam_q_cmd=q_y_cmd, yam_boxed=boxed,
                       yam_tau_ext=tau_ext, yam_tau_raw=tau_raw, yam_stamp=follower.stamp,
                       panto_stamps=link.feedback_stamps(), mono=now, tau_reflect=tau_reflect, yam_age_s=y_age, fault=fault)
            if fault is not None:
                raise EStop(f"monitor:{fault}")

            if t >= next_print:
                next_print = t + 0.5
                print(f"  t={t:6.1f}s  panto={np.round(np.degrees(q_p), 1).tolist()}  "
                      f"yam {follower.names}={np.round(np.degrees(q_y[follower.idx]), 1).tolist()}"
                      f"{' BOX' if boxed.any() else ''}  tau_ext={np.round(tau_ext, 2).tolist()} N.m  "
                      f"reflect={np.round(tau_reflect * 1e3, 1).tolist()} mN.m  "
                      f"iq={np.round(cur, 2).tolist()} A")
            sleep = period - (time.monotonic() - now)
            if sleep > 0:
                time.sleep(sleep)
    except EStop as exc:
        reason = str(exc)
    except JointLimitViolation as exc:
        reason = f"workspace:{exc}"
    except BaseException as exc:  # noqa: BLE001 -- KeyboardInterrupt included: always stop both robots
        reason = f"loop:{type(exc).__name__}:{exc}"
        raise
    finally:
        stop.set(reason)   # retires the Enter-watcher so the release prompt owns stdin
        _safe_stop(reason)
        if follower is not None:
            if not args.sim:
                _hold_until_released(follower, log)
            try:
                follower.close()
            except Exception as exc:  # noqa: BLE001
                log.event(f"YAM close failed: {exc!r}", level="ERROR")
        link.close()
        log.close()


if __name__ == "__main__":
    main()
