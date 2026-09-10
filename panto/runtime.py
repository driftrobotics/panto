"""The runtime: the one process that owns the control loop and the CAN link.

Responsibilities (nothing else touches the drives):
  * fixed-rate loop at ``config.control_rate_hz`` (~200 Hz), ported from
    ``reference/odrive_knob/skodrive/knob.py`` (``next_tick += period``, jitter
    p95, overruns -> "resync rather than spiral")
  * mode state machine: TRANSPARENT / PLOTTER / INTERACTIVE
  * constraint solve: project(pose) per active constraint -> one combined
    stiffness + anchor -> ImpedanceCommand -> backend
  * per-motor I²t budget -> force-limit cutback (mechanism now, trip policy later)
  * heartbeat watchdog: no UI heartbeat for config.heartbeat_timeout_s -> idle
  * telemetry out (schema in CONTRACTS.md)

The UI is a separate websocket client (see web.py) and only sends intent.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import constraints as _c
from .backends import ImpedanceBackend, ImpedanceCommand
from .can_link import CanLink
from .config import Config
from .kinematics import Unreachable, forward, inverse, jacobian, min_singular_value
from .shapes import SHAPES, path_points

try:  # B's canonical gate; falls back until constraints.py lands in this tree
    from .constraints import WorkspaceBoundary, is_active as _is_active
except ImportError:  # pragma: no cover
    WorkspaceBoundary = None

    def _is_active(proj) -> bool:
        return (not getattr(proj, "unilateral", False)) or (
            float(getattr(proj, "penetration", 0.0)) > 0.0
        )

log = logging.getLogger(__name__)


class Mode(enum.Enum):
    TRANSPARENT = "transparent"   # relax(), pure idle
    PLOTTER = "plotter"           # follow a recorded trajectory (stub)
    INTERACTIVE = "interactive"   # haptics from active constraints


# --- I²t cutback curve knobs (why: 0.8 A is a *transient* rating) -------------
_I2T_KNEE = 0.5     # start cutting back once the budget is half spent
_I2T_FLOOR = 0.05   # never scale force below this, even fully depleted

_LEAD_IN_S = 2.0    # ramp from the current pose before any played-back trajectory

_VEL_GAIN_MAX = 0.05  # shoulder chatters at 41 Hz from 0.1 (2026-09-09); never push more
_CURRENT_CAP_MAX = 2.0  # A; drive current_hard_max is 2.5, 2 A bursts accepted 2026-09-10
_ELBOW_HYST_RAD = np.radians(3.0)

# Sustained-oscillation guard (user decision 2026-09-10: no K clamp, guard instead).
_OSC_GUARD_K = 25.0   # N/m: free-air-stable tip stiffness the guard falls back to
_OSC_WINDOW_S = 0.5
_OSC_MM = 3.0         # tip-error std over the window that counts as oscillating


def _scan_workspace_radii(geo, elbow: str, threshold: float, *, samples: int = 512):
    """Once-at-init radius scan for the reachable, well-conditioned annulus
    (``min_singular_value(J) >= threshold``) -> ``(r_min, r_max)`` metres.

    The 2R arm's reach and conditioning are rotationally symmetric about the
    base (rigidly rotating q1 doesn't change J's singular values), so scanning
    one ray is enough -- same technique as
    ``constraints.WorkspaceBoundary._nearest_valid``, but purely kinematic
    (no workspace_polygon) since this is a display quantity, not the live
    constraint.
    """
    reach = geo.l1 + geo.l2
    radii = np.linspace(1e-4, reach, samples)
    ok = np.zeros(samples, dtype=bool)
    for i, r in enumerate(radii):
        try:
            q = inverse(np.array([r, 0.0]), geo, elbow=elbow)
        except Unreachable:
            continue
        ok[i] = min_singular_value(q, geo) >= threshold
    idx = np.where(ok)[0]
    if idx.size == 0:
        return 0.0, 0.0
    return float(radii[idx[0]]), float(radii[idx[-1]])


@dataclass
class LoopStats:
    """Loop-health metrics — they diagnose most feel problems."""

    rate_hz: float = 0.0
    jitter_p95_ms: float = 0.0
    feedback_age_ms: float = 0.0
    overruns: int = 0
    tx: int = 0
    rx: int = 0


class _I2tBudget:
    """Per-motor I²t accumulator -> a single soft force-cutback factor.

    Integrate ``i² - i_continuous²`` per motor and hold the accumulator at >= 0.
    Above continuous current it fills; below it, it leaks (recovery). The cutback
    factor is 1.0 until the budget is ``_I2T_KNEE`` spent, then ramps linearly to
    ``_I2T_FLOOR`` as the fill fraction reaches 1.0.
    """

    def __init__(self, config: Config) -> None:
        n = len(config.motors)
        self._i_cont2 = float(config.thermal.i_continuous) ** 2
        self._budget = float(config.thermal.budget_a2s)
        self._accum = np.zeros(n)

    def update(self, currents: np.ndarray, dt: float) -> float:
        i2 = np.asarray(currents, dtype=float) ** 2
        self._accum += (i2 - self._i_cont2) * max(dt, 0.0)
        np.clip(self._accum, 0.0, None, out=self._accum)
        return float(np.min(self._cutback(self.fractions())))

    def fractions(self) -> np.ndarray:
        return self._accum / self._budget if self._budget > 0 else self._accum * 0.0

    @staticmethod
    def _cutback(frac: np.ndarray) -> np.ndarray:
        span = 1.0 - _I2T_KNEE
        ramp = 1.0 - (1.0 - _I2T_FLOOR) * (frac - _I2T_KNEE) / span
        return np.clip(ramp, _I2T_FLOOR, 1.0)


class Runtime:
    """Owns the control loop, the mode SM, the I²t budget and telemetry."""

    def __init__(
        self,
        config: Config,
        link: CanLink,
        backend: ImpedanceBackend,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config
        self._link = link
        self._backend = backend
        self._clock = clock

        self._lock = threading.Lock()
        self._mode = Mode.TRANSPARENT
        self._mode_entered = False
        self._constraints: list = []
        self._trajectory: list[tuple[float, np.ndarray]] = []      # Cartesian (t, xy)
        self._trajectory_q: list[tuple[float, np.ndarray]] = []    # joint-space (t, q)
        self._traj_t0 = 0.0

        self._i2t = _I2tBudget(config)
        self._stats = LoopStats()
        self._closed_loop = False
        self._watchdog_tripped = False
        self._idled = False
        self._last_heartbeat = clock()
        self._last_step_t = clock()

        self._recording = False
        self._record_buf: list[dict] = []
        self._record_t0 = 0.0
        self._trajectories: dict[str, list] = {}

        # always-on workspace boundary (never removable via set_constraints)
        self._workspace = (
            WorkspaceBoundary.from_config(config) if WorkspaceBoundary is not None
            else None
        )
        self._ws_r_min, self._ws_r_max = _scan_workspace_radii(
            config.geo, config.elbow, config.sigma_min_threshold
        )

        # I²t hard-trip latch (distinct from the soft cutback the budget
        # already applies every tick)
        self._tripped = False
        self._trip_idled = False
        self._loop_error: str | None = None
        self._osc_hist: deque = deque()
        self._osc_tripped = False
        self._elbow_mode = "auto"
        self._elbow_now = str(config.elbow)

        self._tick_listeners: list[Callable[[dict], None]] = []

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tel_lock = threading.Lock()
        self._tel: dict = self._blank_telemetry()

    # ------------------------------------------------------------------ intent

    def set_mode(self, mode: Mode | str) -> None:
        m = mode if isinstance(mode, Mode) else Mode(mode)
        with self._lock:
            if m is not self._mode:
                self._mode = m
                self._mode_entered = False

    def set_constraints(self, constraints: Sequence) -> None:
        with self._lock:
            self._constraints = list(constraints)

    def set_plotter_trajectory(self, points: Sequence) -> None:
        """``points`` is an ordered list of ``(t_seconds, [x, y])``. Stub: the
        loop just walks it; it does not use trajectory-control mode to reach the
        start pose, and there is no playback-by-id lookup yet."""
        with self._lock:
            self._trajectory = [
                (float(t), np.asarray(xy, dtype=float)) for t, xy in points
            ]
            self._trajectory_q = []
            self._traj_t0 = self._clock()

    def note_heartbeat(self) -> None:
        self._last_heartbeat = self._clock()
        if self._watchdog_tripped:
            self._watchdog_tripped = False
            self._idled = False

    def set_idle(self) -> None:
        """UI idle button: force TRANSPARENT and drop the drives to idle."""
        self.set_mode(Mode.TRANSPARENT)
        self._backend.relax()
        self._idle_motors()
        with self._lock:
            self._closed_loop = False

    def note_closed_loop(self, value: bool) -> None:
        self._closed_loop = bool(value)

    # ------------------------------------------------------------------ elbow branch

    def set_elbow(self, mode: str) -> None:
        """IK branch policy: "up" / "down" fixed, or "auto" = follow the arm's
        measured branch each tick (so flipping the elbow through straight in
        interactive mode never yields mirror-image targets)."""
        if mode not in ("up", "down", "auto"):
            raise ValueError(f"elbow mode must be up/down/auto, got {mode!r}")
        with self._lock:
            self._elbow_mode = mode

    def _elbow_branch(self, q: np.ndarray) -> str:
        mode = self._elbow_mode
        if mode != "auto":
            self._elbow_now = mode
            return mode
        # hysteresis band around straight so a wobble at q2≈0 doesn't flap
        if q[1] > _ELBOW_HYST_RAD:
            self._elbow_now = "up"
        elif q[1] < -_ELBOW_HYST_RAD:
            self._elbow_now = "down"
        return self._elbow_now

    _TUNING_FIELDS = ("stiffness_n_per_m", "wall_stiffness_n_per_m", "force_limit_n")

    def set_tuning(self, *, vel_gain=None, current_cap_a=None, **fields: float) -> None:
        """Live feel presets from the UI: any of ``stiffness_n_per_m``,
        ``wall_stiffness_n_per_m``, ``force_limit_n`` (positive floats),
        ``vel_gain`` = per-motor list (pushed to the drives if armed) and
        ``current_cap_a`` (all motors' ``current_soft_max``; the backend re-reads
        it every tick). K sets crispness, the cap sets strength (2026-09-10)."""
        bad = set(fields) - set(self._TUNING_FIELDS)
        if bad:
            raise ValueError(f"unknown tuning field(s): {sorted(bad)}")
        if current_cap_a is not None:
            cap = float(current_cap_a)
            if not 0.0 < cap <= _CURRENT_CAP_MAX:
                raise ValueError(f"current_cap_a must be in (0, {_CURRENT_CAP_MAX}], got {cap}")
            with self._lock:
                for m in self._cfg.motors:
                    m.current_soft_max = cap
        if vel_gain is not None:
            gains = [float(v) for v in vel_gain]
            if len(gains) != len(self._cfg.motors):
                raise ValueError(f"vel_gain needs {len(self._cfg.motors)} values")
            for v in gains:
                if not 0.0 < v <= _VEL_GAIN_MAX:
                    raise ValueError(f"vel_gain must be in (0, {_VEL_GAIN_MAX}], got {v}")
        with self._lock:
            for name, value in fields.items():
                v = float(value)
                if not v > 0.0:
                    raise ValueError(f"{name} must be > 0, got {value!r}")
                setattr(self._cfg.control, name, v)
            self._osc_tripped = False       # a new preset re-arms the guard
            self._osc_hist.clear()
            if vel_gain is not None:
                for m, v in zip(self._cfg.motors, gains):
                    m.vel_gain = v
                armed = self._closed_loop
        if vel_gain is not None and armed:
            self._push_vel_gains()

    def _push_vel_gains(self) -> None:
        """Config is the source of truth for vel_gain: the position backend
        derives pos_gain from it, so the drives must hold the same value."""
        fn = getattr(self._link, "set_vel_gains", None)
        if not callable(fn):
            return
        for m in self._cfg.motors:
            v = float(m.vel_gain)
            if v > _VEL_GAIN_MAX:
                log.warning("motor %d vel_gain %.4f clamped to %.3f", m.node_id, v, _VEL_GAIN_MAX)
                v = _VEL_GAIN_MAX
            fn(m.node_id, v, float(getattr(m, "vel_integrator_gain", 0.0)))

    def _tuning(self) -> dict:
        t = {name: float(getattr(self._cfg.control, name)) for name in self._TUNING_FIELDS}
        t["vel_gain"] = [float(m.vel_gain) for m in self._cfg.motors]
        t["current_cap_a"] = float(max(m.current_soft_max for m in self._cfg.motors))
        t["elbow"] = self._elbow_mode
        t["elbow_now"] = self._elbow_now
        return t

    # ------------------------------------------------------------------ record

    def record_start(self) -> None:
        """Begin buffering ``{t, pose, q}`` samples once per tick, regardless
        of armed state -- meant to capture a hand-guided pass while unarmed."""
        with self._lock:
            self._record_buf = []
            self._record_t0 = self._clock()
            self._recording = True

    def record_stop(self) -> dict:
        """Stop buffering and stash the take under ``"last"`` for playback."""
        with self._lock:
            self._recording = False
            buf = self._record_buf
            self._trajectories["last"] = buf
            samples = len(buf)
            duration_s = float(buf[-1]["t"]) if buf else 0.0
        return {"id": "last", "samples": samples, "duration_s": duration_s}

    def playback(self, id: str = "last") -> None:
        """Replay a previously recorded take via the PLOTTER trajectory
        follower. Raises ``KeyError`` if ``id`` was never recorded."""
        with self._lock:
            buf = self._trajectories[id]
        # Joint space: a take is replayed on the joint angles it was recorded at
        # (no IK, no reach/branch questions), ramped in from the current q.
        q0, _ = (np.asarray(a, dtype=float) for a in self._link.joint_state())
        traj = [(0.0, q0)] + [(_LEAD_IN_S + float(s["t"]), np.asarray(s["q"], dtype=float))
                              for s in buf]
        with self._lock:
            self._trajectory_q = traj
            self._trajectory = []
            self._traj_t0 = self._clock()
        self.set_mode(Mode.PLOTTER)

    # ------------------------------------------------------------------ trace

    def trace_shape(
        self,
        shape: str,
        size_m: float,
        centre,
        speed: float,
        laps: int = 1,
    ) -> None:
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
        pose0 = self._current_pose()
        dt = 1.0 / self._cfg.control_rate_hz
        path = path_points(shape, size_m, np.asarray(centre, dtype=float), speed, dt,
                           laps=laps, start_xy=pose0)
        self._follow([(float(t), np.array([x, y])) for t, x, y in path])

    def _current_pose(self) -> np.ndarray:
        q, _ = (np.asarray(a, dtype=float) for a in self._link.joint_state())
        return forward(q, self._cfg.geo)

    def _follow(self, points: list[tuple[float, np.ndarray]], *, validate: bool = True) -> None:
        if validate:
            self._validate_path(points)
        # An instantaneous anchor step saturates the position loop into a relay
        # cycle (2026-09-04), so every trajectory gets a ramp from the current pose.
        pose0 = self._current_pose()
        ramped = [(0.0, pose0)] + [(_LEAD_IN_S + t, p) for t, p in points]
        self.set_plotter_trajectory(ramped)
        self.set_mode(Mode.PLOTTER)

    def _validate_path(self, points: list[tuple[float, np.ndarray]]) -> None:
        # Strided (paths are smooth) -- a full per-point IK+SVD pass over a long
        # path hogs the GIL long enough to overrun the control loop.
        geo, thr = self._cfg.geo, self._cfg.sigma_min_threshold
        stride = max(1, len(points) // 200)
        for t, p in points[::stride] + points[-1:]:
            try:
                q = inverse(p, geo, elbow=self._elbow_now)
            except Unreachable:
                raise ValueError(f"path leaves reach at t={t:.2f}s: ({p[0]:.3f}, {p[1]:.3f})")
            if min_singular_value(q, geo) < thr:
                raise ValueError(
                    f"path too close to a singularity at t={t:.2f}s: ({p[0]:.3f}, {p[1]:.3f})"
                )

    # ------------------------------------------------------------------ props

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def constraints(self) -> list:
        return list(self._constraints)

    @property
    def stats(self) -> LoopStats:
        return self._stats

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        # Best-effort link bring-up; a minimal fake link needn't provide these.
        # Motors stay IDLE until the UI explicitly calls engage() -- do not
        # commutate on process start.
        for name in ("start", "wait_for_feedback"):
            fn = getattr(self._link, name, None)
            if callable(fn):
                fn()
        self._last_heartbeat = self._clock()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, name="panto-loop", daemon=True
        )
        self._thread.start()

    def engage(self) -> None:
        """UI-initiated arm. Exceptions from the link propagate uncaught --
        the caller (web.py) reports them. Refuses while the I²t trip latch is
        set; call ``clear_errors()`` first."""
        if self._tripped:
            raise RuntimeError("tripped: over_torque")
        if self._loop_error:
            raise RuntimeError(f"clear errors first: {self._loop_error}")
        # Same order as the bring-up scripts: gains, limits + controller mode,
        # park the anchor on the current angle at zero gain, *then* commutate --
        # entering closed loop against a stale input_pos lunges toward it.
        self._push_vel_gains()
        enter = getattr(self._backend, "enter", None)
        if callable(enter):
            enter()
        self._backend.relax()
        fn = getattr(self._link, "enter_closed_loop", None)
        if callable(fn):
            fn()
        with self._lock:
            self._closed_loop = True
            self._idled = False
            self._mode_entered = False

    def clear_errors(self) -> None:
        """Release the I²t trip latch and ask the link to clear any latched
        drive-side axis faults, per motor (tolerant ``getattr``, like the
        rest of this file). Does *not* reset the I²t accumulator itself --
        it keeps leaking/recovering as always, so a still-over-budget motor
        can re-trip on the very next tick."""
        with self._lock:
            self._tripped = False
            self._trip_idled = False
            self._loop_error = None
            self._osc_tripped = False
        clear = getattr(self._link, "clear_errors", None)
        if callable(clear):
            for m in self._cfg.motors:
                try:
                    clear(m.node_id)
                except Exception:  # noqa: BLE001
                    pass

    def add_tick_listener(self, fn: Callable[[dict], None]) -> None:
        """Register ``fn`` to be called with the telemetry dict after every
        ``_publish``. Must be cheap: called from the control-loop thread.
        Exceptions are logged and swallowed -- a listener can never crash or
        stall the loop."""
        self._tick_listeners.append(fn)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._backend.relax()
        self._idle_motors()
        stop = getattr(self._link, "stop", None)
        if callable(stop):
            stop()

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        period = 1.0 / self._cfg.control_rate_hz
        next_tick = time.perf_counter()
        last = next_tick
        jitter: list[float] = []
        window_start = next_tick
        window_iters = 0

        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now

            try:
                self.step(dt)
            except Exception as exc:  # noqa: BLE001
                # A haptic device must never be left with a dead supervisor:
                # idle the drives, go passive, keep ticking (telemetry only).
                log.exception("control loop error; drives idled")
                self._backend.relax()
                self._idle_motors()
                with self._lock:
                    self._closed_loop = False
                    self._mode = Mode.TRANSPARENT
                    self._loop_error = f"loop:{type(exc).__name__}: {exc}"[:120]

            window_iters += 1
            if now - window_start >= 1.0:
                self._stats.rate_hz = window_iters / (now - window_start)
                if jitter:
                    jitter.sort()
                    self._stats.jitter_p95_ms = (
                        jitter[int(len(jitter) * 0.95)] * 1e3
                    )
                    jitter.clear()
                window_start = now
                window_iters = 0

            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
                jitter.append(abs(time.perf_counter() - next_tick))
            else:
                # Fell behind: resync rather than spiral.
                self._stats.overruns += 1
                next_tick = time.perf_counter()

    # ------------------------------------------------------------------ tick

    def step(self, dt: float | None = None) -> None:
        """One control tick. See module docstring for the algorithm."""
        if dt is None:
            now = self._clock()
            dt = min(max(now - self._last_step_t, 0.0), 1.0)
            self._last_step_t = now

        with self._lock:
            mode = self._mode
            constraints = list(self._constraints)
            entered = self._mode_entered
            self._mode_entered = True

        # 1. read state
        q, q_dot = (np.asarray(a, dtype=float) for a in self._link.joint_state())
        age = float(self._link.feedback_age_s())
        currents = np.asarray(self._link.motor_currents(), dtype=float)

        # 2. FK + latency compensation (cap the extrapolation horizon)
        geo = self._cfg.geo
        pose = forward(q, geo)
        J = jacobian(q, geo)
        pose_dot = J @ q_dot
        horizon = min(age, self._cfg.latency_compensation_s)
        pose_c = pose + horizon * pose_dot

        sigma = min_singular_value(q, geo)
        cutback = self._i2t.update(currents, dt)
        if bool(np.any(self._i2t.fractions() >= 1.0)):
            self._tripped = True
        self._stats.feedback_age_ms = age * 1e3
        self._stats.tx, self._stats.rx = self._counters()

        # always-on workspace boundary: active whenever pose_c leaves the
        # reachable, well-conditioned annulus -- checked every tick,
        # independent of mode/arming, so the error is always honest.
        workspace_active = False
        if self._workspace is not None:
            workspace_active = _is_active(self._workspace.project(pose_c))
        unsolvable = False

        anchor = pose.copy()
        force_limit = 0.0

        # recording: buffer once per tick, regardless of armed state (meant
        # to capture a hand-guided pass while unarmed/passive).
        with self._lock:
            if self._recording:
                self._record_buf.append({
                    "t": self._clock() - self._record_t0,
                    "pose": pose.tolist(),
                    "q": q.tolist(),
                })

        # 2.5 I²t hard trip -> behave like set_idle() and latch. Checked
        # ahead of the heartbeat watchdog: over-budget current is the more
        # urgent cutoff and shouldn't wait on a UI heartbeat lapsing too.
        if self._tripped:
            with self._lock:
                self._mode = Mode.TRANSPARENT
                self._mode_entered = True
                self._closed_loop = False
            self._backend.relax()
            if not self._trip_idled:
                self._idle_motors()
                self._trip_idled = True
            self._publish(Mode.TRANSPARENT, pose, q, q_dot, anchor, currents,
                          force_limit, sigma, unsolvable=unsolvable,
                          workspace_active=workspace_active)
            return

        # 3. heartbeat watchdog -> force TRANSPARENT + idle
        if self._clock() - self._last_heartbeat > self._cfg.heartbeat_timeout_s:
            self._watchdog_tripped = True
            with self._lock:
                self._mode = Mode.TRANSPARENT
                self._mode_entered = True
                self._closed_loop = False
            self._backend.relax()
            if not self._idled:
                self._idle_motors()
                self._idled = True
            self._publish(Mode.TRANSPARENT, pose, q, q_dot, anchor, currents,
                          force_limit, sigma, unsolvable=unsolvable,
                          workspace_active=workspace_active)
            return

        # unarmed: publish telemetry only, no backend calls at all.
        if not self._closed_loop:
            self._publish(mode, pose, q, q_dot, anchor, currents, force_limit, sigma,
                          unsolvable=unsolvable, workspace_active=workspace_active)
            return

        if not entered:
            self._enter_mode(mode)
        if hasattr(self._backend, "elbow"):
            self._backend.elbow = self._elbow_branch(q)

        # 4-6. per mode
        if mode is Mode.TRANSPARENT:
            self._backend.relax()

        elif mode is Mode.PLOTTER:
            q_target = self._interp(self._trajectory_q)
            target = forward(q_target, geo) if q_target is not None else self._interp(self._trajectory)
            if target is None:
                self._backend.relax()
            else:
                anchor = target
                K = np.eye(2) * self._cfg.control.stiffness_n_per_m
                force_limit = self._force_limit(sigma, cutback)
                cmd = ImpedanceCommand(pose=pose_c, q=q, anchor=anchor,
                                       stiffness=K, force_limit=force_limit)
                try:
                    apply_joint = getattr(self._backend, "apply_joint", None)
                    if q_target is not None and callable(apply_joint):
                        apply_joint(q_target, cmd)
                    else:
                        self._backend.apply(cmd)
                except Unreachable:
                    self._backend.relax()
                    unsolvable = True

        else:  # INTERACTIVE
            solve_set = list(constraints)
            if self._workspace is not None:
                solve_set.append(self._workspace)
            K, pull, active = self._combine(solve_set, pose_c)
            if not active:
                self._backend.relax()
            else:
                # K is a sum of isotropic k_i·I, always invertible; the effective
                # anchor is the stiffness-weighted mean of the per-constraint
                # targets, and K·(anchor-pose) == Σ K_i·(anchor_i-pose) == ΣF_i.
                anchor = np.linalg.solve(K, pull)
                force_limit = self._force_limit(sigma, cutback)
                try:
                    self._backend.apply(ImpedanceCommand(
                        pose=pose_c, q=q, anchor=anchor,
                        stiffness=K, force_limit=force_limit,
                    ))
                except Unreachable:
                    self._backend.relax()
                    unsolvable = True

        # force_limit > 0 only on ticks where the backend rendered an anchor
        self._guard_oscillation(pose, anchor, applied=force_limit > 0.0)
        self._publish(mode, pose, q, q_dot, anchor, currents, force_limit, sigma,
                      unsolvable=unsolvable, workspace_active=workspace_active)

    def _guard_oscillation(self, pose: np.ndarray, anchor: np.ndarray, *, applied: bool) -> None:
        """Above K25 the free-air arm relay-cycles at the current cap (hand let
        go -> ~3 Hz swing); the hand is the damping. If the tip error keeps
        swinging > _OSC_MM for a window, fall back to K25 and flag the UI."""
        if not applied or self._cfg.control.stiffness_n_per_m <= _OSC_GUARD_K:
            self._osc_hist.clear()
            return
        t = self._clock()
        self._osc_hist.append((t, np.asarray(pose - anchor, dtype=float)))
        while self._osc_hist and self._osc_hist[0][0] < t - _OSC_WINDOW_S:
            self._osc_hist.popleft()
        if len(self._osc_hist) < 8 or t - self._osc_hist[0][0] < 0.8 * _OSC_WINDOW_S:
            return
        err = np.array([e for _, e in self._osc_hist])
        std_mm = err.std(axis=0) * 1e3
        if np.any(std_mm > _OSC_MM):
            with self._lock:
                self._cfg.control.stiffness_n_per_m = _OSC_GUARD_K
                self._cfg.control.wall_stiffness_n_per_m = min(
                    self._cfg.control.wall_stiffness_n_per_m, 2.0 * _OSC_GUARD_K)
                self._osc_tripped = True
            self._osc_hist.clear()
            log.warning("oscillation guard: tip error std %s mm over %.1fs -> K%.0f",
                        np.round(std_mm, 2).tolist(), _OSC_WINDOW_S, _OSC_GUARD_K)

    # ------------------------------------------------------------------ solve

    def _combine(self, constraints: Sequence, pose: np.ndarray):
        """Multi-constraint combination policy (per coordinator, uniform form).

        Every active constraint contributes an isotropic stiffness ``K_i = k_i·I``
        and its target ``anchor_i``; the restoring direction is carried entirely
        by ``anchor_i - pose`` (Wall returns ``anchor == pose`` on the free side,
        so we never touch ``normal``). Bilateral terms always contribute;
        unilateral terms only when ``is_active`` (``penetration > 0``). We sum
        ``K = ΣK_i`` and ``pull = ΣK_i·anchor_i``; the backend gets a single
        ImpedanceCommand whose anchor ``K⁻¹·pull`` is the stiffness-weighted mean
        — a stiff wall dominates a soft snap.
        """
        K = np.zeros((2, 2))
        pull = np.zeros(2)
        active = False
        # Points are "hold here" targets: only the most recently added one acts
        # (averaging several into a centroid is not what anyone means by it).
        last_point = next((c for c in reversed(constraints) if isinstance(c, _c.Point)), None)
        for c in constraints:
            if isinstance(c, _c.Point) and c is not last_point:
                continue
            proj = c.project(pose)
            if not _is_active(proj):
                continue
            k_i = (
                self._cfg.control.wall_stiffness_n_per_m
                if bool(getattr(proj, "unilateral", False))
                else self._cfg.control.stiffness_n_per_m
            )
            K_i = np.eye(2) * k_i
            K += K_i
            pull += K_i @ np.asarray(proj.anchor, dtype=float)
            active = True
        return K, pull, active

    def _force_limit(self, sigma: float, cutback: float) -> float:
        sigma_scale = float(
            np.clip(sigma / self._cfg.sigma_min_threshold, 0.0, 1.0)
        )
        return self._cfg.control.force_limit_n * sigma_scale * cutback

    def _interp(self, traj: list[tuple[float, np.ndarray]]) -> np.ndarray | None:
        """Piecewise-linear sample of a (t, vector) trajectory at loop time;
        holds the last point once past the end."""
        if not traj:
            return None
        t = self._clock() - self._traj_t0
        prev = traj[0]
        for pt in traj:
            if pt[0] >= t:
                if pt is prev or pt[0] == prev[0]:
                    return pt[1].copy()
                frac = (t - prev[0]) / (pt[0] - prev[0])
                return prev[1] + frac * (pt[1] - prev[1])
            prev = pt
        return traj[-1][1].copy()

    # ------------------------------------------------------------------ helpers

    def _enter_mode(self, mode: Mode) -> None:
        if mode is Mode.TRANSPARENT:
            self._backend.relax()
        else:
            enter = getattr(self._backend, "enter", None)
            if callable(enter):
                enter()
        if mode is Mode.PLOTTER:
            self._traj_t0 = self._clock()

    def _idle_motors(self) -> None:
        set_idle = getattr(self._link, "set_idle", None)
        if not callable(set_idle):
            return
        for m in self._cfg.motors:
            try:
                set_idle(m.node_id)
            except Exception:  # noqa: BLE001
                pass

    def _counters(self) -> tuple[int, int]:
        fn = getattr(self._link, "counters", None)
        if callable(fn):
            try:
                tx, rx = fn()
                return int(tx), int(rx)
            except Exception:  # noqa: BLE001
                pass
        return self._stats.tx, self._stats.rx

    def _drive_errors(self) -> list:
        fn = getattr(self._link, "axis_errors", None)
        if not callable(fn):
            return []
        try:
            errs = fn()
        except Exception:  # noqa: BLE001
            return []
        return [
            f"drive:axis{i}:0x{int(e):x}" for i, e in enumerate(errs) if int(e) != 0
        ]

    def _build_errors(self, *, unsolvable: bool, workspace_active: bool) -> list:
        """Stable, duplicate-free error taxonomy: drive faults first (by axis
        index), then the latched over-current trip, then this tick's solver/
        workspace state -- always in this order regardless of the order the
        conditions were detected in, so telemetry ordering is deterministic."""
        errs = self._drive_errors()
        if self._loop_error:
            errs.append(self._loop_error)
        if self._osc_tripped:
            errs.append("osc_guard")
        if self._tripped:
            errs.append("over_torque")
        if unsolvable:
            errs.append("unsolvable")
        if workspace_active:
            errs.append("workspace")
        return errs

    # ------------------------------------------------------------------ telemetry

    def _blank_telemetry(self) -> dict:
        return {
            "type": "state",
            "mode": self._mode.value,
            "closed_loop": self._closed_loop,
            "pose": [0.0, 0.0],
            "q": [0.0, 0.0],
            "q_dot": [0.0, 0.0],
            "anchor": [0.0, 0.0],
            "currents": [0.0, 0.0],
            "i2t_frac": [0.0, 0.0],
            "force_limit": 0.0,
            "sigma_min": 0.0,
            "errors": [],
            "tripped": False,
            "tuning": self._tuning(),
            "workspace": {"r_min": self._ws_r_min, "r_max": self._ws_r_max},
            "recording": False,
            "recorded_samples": 0,
            "stats": {
                "rate_hz": 0.0, "jitter_p95_ms": 0.0, "feedback_age_ms": 0.0,
                "overruns": 0, "tx": 0, "rx": 0,
            },
        }

    def _publish(self, mode, pose, q, q_dot, anchor, currents,
                 force_limit, sigma, *, unsolvable: bool = False,
                 workspace_active: bool = False) -> None:
        tel = {
            "type": "state",
            "mode": mode.value,
            "closed_loop": bool(self._closed_loop),
            "pose": np.asarray(pose, dtype=float).tolist(),
            "q": np.asarray(q, dtype=float).tolist(),
            "q_dot": np.asarray(q_dot, dtype=float).tolist(),
            "anchor": np.asarray(anchor, dtype=float).tolist(),
            "currents": np.asarray(currents, dtype=float).tolist(),
            "i2t_frac": self._i2t.fractions().tolist(),
            "force_limit": float(force_limit),
            "sigma_min": float(sigma),
            "errors": self._build_errors(
                unsolvable=unsolvable, workspace_active=workspace_active
            ),
            "tripped": bool(self._tripped),
            "tuning": self._tuning(),
            "workspace": {"r_min": self._ws_r_min, "r_max": self._ws_r_max},
            "recording": bool(self._recording),
            "recorded_samples": len(self._record_buf),
            "stats": {
                "rate_hz": self._stats.rate_hz,
                "jitter_p95_ms": self._stats.jitter_p95_ms,
                "feedback_age_ms": self._stats.feedback_age_ms,
                "overruns": self._stats.overruns,
                "tx": self._stats.tx,
                "rx": self._stats.rx,
            },
        }
        with self._tel_lock:
            self._tel = tel
        for fn in self._tick_listeners:
            try:
                fn(tel)
            except Exception:  # noqa: BLE001 - a listener must never crash the loop
                log.exception("tick listener raised")

    def telemetry(self) -> dict:
        """Latest tick's state, matching the CONTRACTS.md schema."""
        with self._tel_lock:
            return dict(self._tel)
