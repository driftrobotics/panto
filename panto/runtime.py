"""The runtime: the one process that owns the control loop and the CAN link.

Responsibilities (nothing else touches the drives):
  * fixed-rate loop at ``config.control_rate_hz`` (~200 Hz)
  * mode state machine: TRANSPARENT / PLOTTER / INTERACTIVE
  * constraint solve: for the active constraints, project(pose) -> anchor, then
    build an ImpedanceCommand and hand it to the backend
  * per-motor I²t budget -> force-limit cutback
  * heartbeat watchdog: no UI heartbeat for config.heartbeat_timeout_s -> idle
  * telemetry out (loop rate, jitter p95, feedback age, overruns, currents)

The UI is a separate websocket client (see web.py) and only sends intent:
set mode, set constraints, start/stop recording, playback.
"""

from __future__ import annotations

import enum

from .backends import ImpedanceBackend
from .can_link import CanLink
from .config import Config


class Mode(enum.Enum):
    TRANSPARENT = "transparent"   # relax(), pure idle
    PLOTTER = "plotter"           # follow a trajectory
    INTERACTIVE = "interactive"   # haptics from active constraints


class Runtime:
    def __init__(self, config: Config, link: CanLink, backend: ImpedanceBackend):
        self._cfg = config
        self._link = link
        self._backend = backend
        self._mode = Mode.TRANSPARENT
        self._constraints: list = []
        # self._i2t = I2tBudget(config.thermal)  # TODO
        # self._health = LoopHealth()             # TODO

    def set_mode(self, mode: Mode) -> None:
        raise NotImplementedError

    def set_constraints(self, constraints: list) -> None:
        raise NotImplementedError

    def step(self) -> None:
        """One control tick. Called at a fixed rate by :meth:`run`.

        1. read joint_state + feedback_age
        2. FK -> pose;  latency-compensate pose by feedback_age * pose_dot
        3. if TRANSPARENT: backend.relax(); return
        4. for each constraint: project(pose); combine (nearest / sum bilateral,
           gate unilateral on penetration > 0)
        5. scale force_limit by σ_min(J)/σ_min_threshold and by the I²t cutback
        6. backend.apply(ImpedanceCommand(...))
        """
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError
