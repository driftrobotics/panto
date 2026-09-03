"""The outer control loop.

Runs at a fixed rate (default 200 Hz):

    read latest encoder estimate  ->  detent state machine  ->  Set_Input_Pos

The spring itself lives on the ODrive at 8 kHz; this loop only moves its anchor
and adjusts its stiffness. That split is what makes 200 Hz sufficient here --
see README "Why 200 Hz is enough (and where it isn't)".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .config import KnobConfig, MotorConfig, PRESETS, TuningConfig
from .haptics import DetentEngine, KnobState, TWO_PI
from .odrive_can import (
    CONTROL_MODE_POSITION,
    INPUT_MODE_PASSTHROUGH,
    ODriveAxis,
    ODriveError,
)

log = logging.getLogger(__name__)

StateCallback = Callable[[KnobState, "LoopStats"], None]


@dataclass
class LoopStats:
    """Health metrics. Worth watching -- they diagnose most feel problems."""

    rate_hz: float = 0.0
    #: Fraction of a period, p95. High jitter shows up as inconsistent clicks.
    jitter_p95_ms: float = 0.0
    #: Age of the newest encoder frame when we used it.
    feedback_age_ms: float = 0.0
    #: Detents crossed in one update. Persistent >1 means the loop is behind and
    #: click positions will be smeared.
    max_steps: int = 0
    overruns: int = 0
    tx: int = 0
    rx: int = 0


class SmartKnob:
    def __init__(
        self,
        axis: ODriveAxis,
        motor: MotorConfig,
        tuning: TuningConfig,
        config: Optional[KnobConfig] = None,
        rate_hz: float = 200.0,
        invert: bool = False,
    ) -> None:
        self._axis = axis
        self._motor = motor
        self._tuning = tuning
        self._rate_hz = rate_hz
        self._period = 1.0 / rate_hz

        self._engine = DetentEngine(
            config or PRESETS[0], motor=motor, tuning=tuning, invert=invert
        )
        self._pending_config: Optional[KnobConfig] = None
        self._config_lock = threading.Lock()

        self._stats = LoopStats()
        self._latest: Optional[KnobState] = None
        self._callbacks: List[StateCallback] = []

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ api

    def add_listener(self, cb: StateCallback) -> None:
        self._callbacks.append(cb)

    def set_config(self, config: KnobConfig) -> None:
        """Queue a config change; applied at the top of the next iteration."""
        with self._config_lock:
            self._pending_config = config

    @property
    def config(self) -> KnobConfig:
        return self._engine.config

    @property
    def latest(self) -> Optional[KnobState]:
        return self._latest

    @property
    def stats(self) -> LoopStats:
        return self._stats

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._axis.start()
        fb = self._axis.wait_for_feedback()
        log.info("encoder alive: pos=%.4f turns", fb.pos_turns)

        # Position mode. pos_gain is driven per-iteration by the detent engine;
        # vel_gain is the local 8 kHz damper and stays fixed.
        self._axis.set_controller_mode(CONTROL_MODE_POSITION, INPUT_MODE_PASSTHROUGH)
        self._axis.set_vel_gains(
            self._tuning.vel_gain, self._tuning.vel_integrator_gain, force=True
        )
        self._axis.set_limits(
            self._motor.velocity_limit, self._motor.max_current, force=True
        )
        # Anchor the spring where the knob currently is, so enabling closed loop
        # doesn't yank it.
        angle = fb.pos_turns * TWO_PI
        self._engine.set_config(self._engine.config, angle)
        self._axis.set_pos_gain(0.0, force=True)
        self._axis.set_input_pos(fb.pos_turns)
        self._axis.enter_closed_loop()

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="knob-loop", daemon=True)
        self._thread.start()
        log.info("control loop running at %.0f Hz", self._rate_hz)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._axis.idle()
        self._axis.stop()

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        next_tick = time.perf_counter()
        last = next_tick
        jitter: List[float] = []
        window_start = next_tick
        window_iters = 0

        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now

            try:
                self._iterate(dt)
            except ODriveError:
                log.exception("CAN error in control loop; stopping")
                break

            window_iters += 1
            if now - window_start >= 1.0:
                self._stats.rate_hz = window_iters / (now - window_start)
                if jitter:
                    jitter.sort()
                    self._stats.jitter_p95_ms = jitter[int(len(jitter) * 0.95)] * 1e3
                    jitter.clear()
                window_start = now
                window_iters = 0
                self._stats.max_steps = 0

            next_tick += self._period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
                jitter.append(abs(time.perf_counter() - next_tick))
            else:
                # Fell behind: resync rather than spiral.
                self._stats.overruns += 1
                next_tick = time.perf_counter()

    def _iterate(self, dt: float) -> None:
        with self._config_lock:
            pending, self._pending_config = self._pending_config, None

        fb = self._axis.feedback
        angle = fb.pos_turns * TWO_PI
        velocity = fb.vel_turns_s * TWO_PI

        if pending is not None:
            self._engine.set_config(pending, angle)

        state = self._engine.update(angle, velocity, dt)

        self._axis.set_pos_gain(state.pos_gain)
        self._axis.set_input_pos(state.setpoint_turns)

        self._stats.feedback_age_ms = fb.age * 1e3
        self._stats.max_steps = max(self._stats.max_steps, abs(state.steps))
        self._stats.tx, self._stats.rx = self._axis.counters
        self._latest = state

        for cb in self._callbacks:
            try:
                cb(state, self._stats)
            except Exception:
                log.exception("state listener raised")
