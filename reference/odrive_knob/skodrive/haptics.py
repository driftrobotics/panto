"""Detent state machine.

Port of the haptic loop in ``firmware/src/motor_task.cpp`` (L216-298), with the
control law moved out of this process: instead of computing a torque every
iteration at 1 kHz, this decides *where the spring should be anchored* and *how
stiff it should be*, and the ODrive renders that spring at 8 kHz.

Two deliberate changes from the firmware:

1. **Multi-step snapping.** The firmware increments/decrements position by at
   most one detent per loop iteration (motor_task.cpp:240-248). At 1 kHz that
   caps tracking at ``rate * detent_width``; at our 200 Hz outer loop that would
   be 0.55 rev/s for 1 degree detents. We loop until the error is inside the
   snap point, so position stays correct at any speed.

2. **Latency compensation.** The snap decision is evaluated against a
   velocity-extrapolated angle, because our decision is late by the host + CAN +
   USB round trip. Click position error is ``omega * latency``; see
   ``TuningConfig.latency_compensation_s``.

Sign convention: internally, position *increases* with increasing knob angle.
The firmware uses the opposite convention plus a compile-time SK_INVERT_ROTATION
flag; here that collapses into the ``invert`` constructor argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import KnobConfig, MotorConfig, TuningConfig

TWO_PI = 2.0 * math.pi

# Firmware constants (motor_task.cpp:15-22). The EWMA alphas there are tuned for
# a 1 kHz loop; expressed as time constants they become rate-independent.
DEAD_ZONE_DETENT_PERCENT = 0.2
DEAD_ZONE_RAD = math.radians(1)
IDLE_VELOCITY_TAU_S = 1.0          # was alpha=0.001 @ 1 kHz
IDLE_VELOCITY_RAD_PER_SEC = 0.05
IDLE_CORRECTION_DELAY_S = 0.5
IDLE_CORRECTION_MAX_ANGLE_RAD = math.radians(5)
IDLE_CORRECTION_TAU_S = 2.0        # was alpha=0.0005 @ 1 kHz

#: Safety cap on snap iterations per update. At 200 Hz with 1 degree detents
#: this allows ~14 rev/s before positions are dropped.
MAX_STEPS_PER_UPDATE = 256


@dataclass
class KnobState:
    """Everything the UI and the ODrive command need for one iteration."""

    position: int
    sub_position_unit: float
    #: Sign-adjusted knob angle, radians.
    angle_rad: float
    velocity_rad_s: float
    #: Where the spring is anchored, in the same frame as ``angle_rad``.
    detent_center_rad: float
    out_of_bounds: bool
    #: Position setpoint to send to the ODrive, in motor turns.
    setpoint_turns: float
    #: Derived ODrive gain, (turn/s)/turn.
    pos_gain: float
    #: N.m ceiling for this iteration.
    torque_limit: float
    config: KnobConfig
    #: Detents crossed on this update. >1 means we were behind; useful as a
    #: health metric for whether the outer loop is keeping up.
    steps: int = 0


class DetentEngine:
    def __init__(
        self,
        config: KnobConfig,
        motor: MotorConfig,
        tuning: TuningConfig,
        invert: bool = False,
    ) -> None:
        self._motor = motor
        self._tuning = tuning
        self._sign = -1.0 if invert else 1.0

        self._config = config
        self._position = config.position
        self._detent_center = 0.0
        self._sub_position_unit = 0.0

        self._idle_velocity_ewma = 0.0
        #: Seconds spent below the idle velocity threshold. Accumulated from the
        #: caller's dt rather than a wall clock, so behaviour is identical at any
        #: loop rate and the engine stays testable without sleeping.
        self._idle_time = 0.0
        self._initialised = False

    # ------------------------------------------------------------------ config

    @property
    def config(self) -> KnobConfig:
        return self._config

    @property
    def position(self) -> int:
        return self._position

    def set_config(self, new: KnobConfig, angle_rad: float) -> None:
        """Apply a new config, re-anchoring the spring.

        Mirrors the validation and idempotency rules of the firmware's
        CommandType::CONFIG handler (motor_task.cpp:115-193): position is only
        applied when it actually changes, or when the nonce changes.
        """
        if new.detent_strength_unit < 0:
            raise ValueError("detent_strength_unit cannot be negative")
        if new.endstop_strength_unit < 0:
            raise ValueError("endstop_strength_unit cannot be negative")
        if new.snap_point < 0.5:
            raise ValueError("snap_point must be >= 0.5 for stability")
        if new.snap_point_bias < 0:
            raise ValueError("snap_point_bias cannot be negative (risks instability)")
        if new.position_width_radians <= 0:
            raise ValueError("position_width_radians must be positive")

        old = self._config
        theta = self._sign * angle_rad

        position_updated = (
            new.position != old.position
            or new.sub_position_unit != old.sub_position_unit
            or new.position_nonce != old.position_nonce
            or not self._initialised
        )
        if position_updated:
            self._position = new.position

        if new.max_position >= new.min_position:
            self._position = min(max(self._position, new.min_position), new.max_position)

        if position_updated or new.position_width_radians != old.position_width_radians:
            sub = new.sub_position_unit if position_updated else self._sub_position_unit
            self._detent_center = theta - sub * new.position_width_radians

        self._config = new
        self._initialised = True

    # ------------------------------------------------------------------ update

    def update(self, angle_rad: float, velocity_rad_s: float, dt: float) -> KnobState:
        cfg = self._config
        theta = self._sign * angle_rad
        omega = self._sign * velocity_rad_s

        if not self._initialised:
            self.set_config(cfg, angle_rad)

        width = cfg.position_width_radians
        bounded = cfg.max_position >= cfg.min_position

        self._update_idle_correction(theta, omega, dt)

        # Snap decision runs against a latency-extrapolated angle so clicks land
        # where the knob *is* by the time the ODrive sees the new setpoint.
        theta_pred = theta + omega * self._tuning.latency_compensation_s
        steps = self._snap(theta_pred, width, bounded, cfg)

        error = theta - self._detent_center
        self._sub_position_unit = error / width

        out_of_bounds = bounded and (
            (error < 0 and self._position == cfg.min_position)
            or (error > 0 and self._position == cfg.max_position)
        )

        strength = cfg.endstop_strength_unit if out_of_bounds else cfg.detent_strength_unit
        # Magnetic detents: positions not in the list spin freely.
        if cfg.detent_positions and not out_of_bounds and self._position not in cfg.detent_positions:
            strength = 0.0

        pos_gain = self._pos_gain_for(strength, width)

        setpoint_rad = self._detent_center
        if self._tuning.dead_zone_enabled:
            dead_zone = min(width * DEAD_ZONE_DETENT_PERCENT, DEAD_ZONE_RAD)
            if abs(error) < dead_zone:
                # Park the setpoint on the current angle => zero spring torque.
                setpoint_rad = theta

        return KnobState(
            position=self._position,
            sub_position_unit=self._sub_position_unit,
            angle_rad=theta,
            velocity_rad_s=omega,
            detent_center_rad=self._detent_center,
            out_of_bounds=out_of_bounds,
            setpoint_turns=(self._sign * setpoint_rad) / TWO_PI,
            pos_gain=pos_gain,
            torque_limit=self._motor.peak_torque,
            config=cfg,
            steps=steps,
        )

    # ----------------------------------------------------------------- internals

    def _update_idle_correction(self, theta: float, omega: float, dt: float) -> None:
        """Slowly re-centre the spring anchor when the knob is at rest.

        Without this, the anchor drifts a fraction of a detent away from where
        the knob actually settles and the motor holds a small standing torque.
        """
        alpha_v = 1.0 - math.exp(-dt / IDLE_VELOCITY_TAU_S)
        self._idle_velocity_ewma += alpha_v * (omega - self._idle_velocity_ewma)

        if abs(self._idle_velocity_ewma) > IDLE_VELOCITY_RAD_PER_SEC:
            self._idle_time = 0.0
        else:
            self._idle_time += dt

        idle_long_enough = self._idle_time > IDLE_CORRECTION_DELAY_S
        if idle_long_enough and abs(theta - self._detent_center) < IDLE_CORRECTION_MAX_ANGLE_RAD:
            alpha_c = 1.0 - math.exp(-dt / IDLE_CORRECTION_TAU_S)
            self._detent_center += alpha_c * (theta - self._detent_center)

    def _snap(self, theta: float, width: float, bounded: bool, cfg: KnobConfig) -> int:
        """Advance position past every snap point we've crossed.

        Re-evaluates thresholds each step rather than solving in closed form, so
        the position-dependent ``snap_point_bias`` keeps the exact firmware
        semantics even across a multi-detent jump.
        """
        snap = width * cfg.snap_point
        bias = width * cfg.snap_point_bias
        steps = 0

        for _ in range(MAX_STEPS_PER_UPDATE):
            error = theta - self._detent_center
            # Moving away from home costs an extra `bias`; moving toward home
            # is `bias` cheaper. Matches motor_task.cpp:236-237.
            up_thresh = snap + (bias if self._position >= 0 else -bias)
            down_thresh = -snap - (bias if self._position <= 0 else -bias)

            if error > up_thresh and (not bounded or self._position < cfg.max_position):
                self._detent_center += width
                self._position += 1
                steps += 1
            elif error < down_thresh and (not bounded or self._position > cfg.min_position):
                self._detent_center -= width
                self._position -= 1
                steps -= 1
            else:
                break

        return steps

    def _pos_gain_for(self, strength: float, width: float) -> float:
        """Map a detent strength unit onto an ODrive ``pos_gain``.

        ODrive's cascade is ``vel_cmd = pos_error * pos_gain`` then
        ``torque = (vel_cmd - vel_est) * vel_gain``, so the effective spring
        constant is ``pos_gain * vel_gain`` N.m per *turn* of error.

        We define ``detent_strength_unit == 1`` as "peak motor torque at
        ``detent_full_scale`` of a detent width of deflection", which makes the
        unit mean the same thing regardless of detent width.
        """
        if strength <= 0:
            return 0.0
        vel_gain = self._tuning.vel_gain
        if vel_gain <= 0:
            return 0.0

        deflection_rad = width * self._tuning.detent_full_scale
        k_rad = strength * self._motor.peak_torque / deflection_rad  # N.m/rad
        k_turn = k_rad * TWO_PI                                      # N.m/turn
        return min(k_turn / vel_gain, self._tuning.max_pos_gain)
