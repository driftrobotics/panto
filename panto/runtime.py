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
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .backends import ImpedanceBackend, ImpedanceCommand
from .can_link import CanLink
from .config import Config
from .kinematics import forward, jacobian, min_singular_value
from .shapes import SHAPES, path_points

try:  # B's canonical gate; falls back until constraints.py lands in this tree
    from .constraints import is_active as _is_active
except ImportError:  # pragma: no cover
    def _is_active(proj) -> bool:
        return (not getattr(proj, "unilateral", False)) or (
            float(getattr(proj, "penetration", 0.0)) > 0.0
        )


class Mode(enum.Enum):
    TRANSPARENT = "transparent"   # relax(), pure idle
    PLOTTER = "plotter"           # follow a recorded trajectory (stub)
    INTERACTIVE = "interactive"   # haptics from active constraints


# --- I²t cutback curve knobs (why: 0.8 A is a *transient* rating) -------------
_I2T_KNEE = 0.5     # start cutting back once the budget is half spent
_I2T_FLOOR = 0.05   # never scale force below this, even fully depleted


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
        self._trajectory: list[tuple[float, np.ndarray]] = []
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
        points = [(sample["t"], np.asarray(sample["pose"], dtype=float)) for sample in buf]
        self.set_plotter_trajectory(points)
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
        """Build a lead-in ramp from the current pose to a shapes.path_points
        path and feed it to the PLOTTER trajectory follower."""
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")

        q, _q_dot = (np.asarray(a, dtype=float) for a in self._link.joint_state())
        pose0 = forward(q, self._cfg.geo)

        centre_xy = np.asarray(centre, dtype=float)
        dt = 1.0 / self._cfg.control_rate_hz
        path = path_points(shape, size_m, centre_xy, speed, dt, laps=laps, start_xy=pose0)

        lead_in_s = 2.0
        points: list[tuple[float, np.ndarray]] = [(0.0, pose0)]
        for t, x, y in path:
            points.append((lead_in_s + float(t), np.array([x, y])))

        self.set_plotter_trajectory(points)
        self.set_mode(Mode.PLOTTER)

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
        """UI-initiated arm: enter closed-loop control. Exceptions from the
        link propagate uncaught -- the caller (web.py) reports them."""
        fn = getattr(self._link, "enter_closed_loop", None)
        if callable(fn):
            fn()
        with self._lock:
            self._closed_loop = True
            self._idled = False
            self._mode_entered = False

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
            except Exception:  # noqa: BLE001 - a loop crash must idle the drives
                self._backend.relax()
                self._idle_motors()
                raise

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
        self._stats.feedback_age_ms = age * 1e3
        self._stats.tx, self._stats.rx = self._counters()

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
                          force_limit, sigma)
            return

        # unarmed: publish telemetry only, no backend calls at all.
        if not self._closed_loop:
            self._publish(mode, pose, q, q_dot, anchor, currents, force_limit, sigma)
            return

        if not entered:
            self._enter_mode(mode)

        # 4-6. per mode
        if mode is Mode.TRANSPARENT:
            self._backend.relax()

        elif mode is Mode.PLOTTER:
            target = self._trajectory_target()
            if target is None:
                self._backend.relax()
            else:
                anchor = target
                K = np.eye(2) * self._cfg.control.stiffness_n_per_m
                force_limit = self._force_limit(sigma, cutback)
                self._backend.apply(ImpedanceCommand(
                    pose=pose_c, q=q, anchor=anchor,
                    stiffness=K, force_limit=force_limit,
                ))

        else:  # INTERACTIVE
            K, pull, active = self._combine(constraints, pose_c)
            if not active:
                self._backend.relax()
            else:
                # K is a sum of isotropic k_i·I, always invertible; the effective
                # anchor is the stiffness-weighted mean of the per-constraint
                # targets, and K·(anchor-pose) == Σ K_i·(anchor_i-pose) == ΣF_i.
                anchor = np.linalg.solve(K, pull)
                force_limit = self._force_limit(sigma, cutback)
                self._backend.apply(ImpedanceCommand(
                    pose=pose_c, q=q, anchor=anchor,
                    stiffness=K, force_limit=force_limit,
                ))

        self._publish(mode, pose, q, q_dot, anchor, currents, force_limit, sigma)

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
        for c in constraints:
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

    def _trajectory_target(self) -> np.ndarray | None:
        if not self._trajectory:
            return None
        t = self._clock() - self._traj_t0
        prev = self._trajectory[0]
        for pt in self._trajectory:
            if pt[0] >= t:
                if pt is prev or pt[0] == prev[0]:
                    return pt[1].copy()
                frac = (t - prev[0]) / (pt[0] - prev[0])
                return prev[1] + frac * (pt[1] - prev[1])
            prev = pt
        return self._trajectory[-1][1].copy()

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

    def _errors(self) -> list:
        fn = getattr(self._link, "axis_errors", None)
        if not callable(fn):
            return []
        try:
            errs = fn()
        except Exception:  # noqa: BLE001
            return []
        return [
            f"axis{i}:0x{int(e):x}" for i, e in enumerate(errs) if int(e) != 0
        ]

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
            "recording": False,
            "recorded_samples": 0,
            "stats": {
                "rate_hz": 0.0, "jitter_p95_ms": 0.0, "feedback_age_ms": 0.0,
                "overruns": 0, "tx": 0, "rx": 0,
            },
        }

    def _publish(self, mode, pose, q, q_dot, anchor, currents,
                 force_limit, sigma) -> None:
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
            "errors": self._errors(),
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

    def telemetry(self) -> dict:
        """Latest tick's state, matching the CONTRACTS.md schema."""
        with self._tel_lock:
            return dict(self._tel)
