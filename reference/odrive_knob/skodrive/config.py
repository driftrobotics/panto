"""Knob configuration.

`KnobConfig` mirrors `PB_SmartKnobConfig` from ``proto/smartknob.proto`` so the
built-in presets port over from ``firmware/src/interface_task.cpp`` unchanged.

Fields that only made sense on the original hardware (``led_hue``) are kept so
configs stay comparable, even though nothing consumes them here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class KnobConfig:
    """One haptic "mode". Angles are radians, strengths are dimensionless units."""

    #: Integer position to jump to. Only applied when it changes vs. the previous
    #: config, or when ``position_nonce`` changes (matches firmware semantics).
    position: int = 0
    sub_position_unit: float = 0.0
    position_nonce: int = 0

    min_position: int = 0
    #: If ``max_position < min_position``, bounds are disabled (free rotation).
    max_position: int = -1

    #: Angular width of one detent/position.
    position_width_radians: float = math.radians(10)

    #: Detent spring strength. 1.0 => peak torque at ``DETENT_FULL_SCALE`` of a
    #: detent width of deflection (see haptics.stiffness_for).
    detent_strength_unit: float = 0.0
    #: Spring strength applied when pushed past min/max.
    endstop_strength_unit: float = 1.0

    #: Fraction of a detent width you must travel before snapping to the next
    #: position. > 0.5 gives hysteresis; typical 0.55 - 1.1.
    snap_point: float = 1.1

    text: str = ""

    #: "Magnetic" detents: if non-empty, only these positions get a spring and
    #: everything else spins freely.
    detent_positions: Tuple[int, ...] = ()

    #: Shifts the snap point so returning toward position 0 is easier than
    #: leaving it. 0 = symmetric.
    snap_point_bias: float = 0.0

    led_hue: int = 0

    def replace(self, **kwargs) -> "KnobConfig":
        return replace(self, **kwargs)


def _deg(d: float) -> float:
    return math.radians(d)


# Ported verbatim from firmware/src/interface_task.cpp so behaviour can be
# A/B'd against the original hardware.
PRESETS: Tuple[KnobConfig, ...] = (
    KnobConfig(
        min_position=0,
        max_position=-1,
        position_width_radians=_deg(10),
        detent_strength_unit=0,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Unbounded\nNo detents",
        led_hue=200,
    ),
    KnobConfig(
        position_nonce=1,
        min_position=0,
        max_position=10,
        position_width_radians=_deg(10),
        detent_strength_unit=0,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Bounded 0-10\nNo detents",
    ),
    KnobConfig(
        position_nonce=2,
        min_position=0,
        max_position=72,
        position_width_radians=_deg(10),
        detent_strength_unit=0,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Multi-rev\nNo detents",
        led_hue=73,
    ),
    KnobConfig(
        position_nonce=3,
        min_position=0,
        max_position=1,
        position_width_radians=_deg(60),
        detent_strength_unit=1,
        endstop_strength_unit=1,
        # Snap point just past the midpoint, unlike normal detents which snap
        # past the *next* value.
        snap_point=0.55,
        text="On/off\nStrong detent",
        led_hue=157,
    ),
    KnobConfig(
        position_nonce=4,
        min_position=0,
        max_position=0,
        position_width_radians=_deg(60),
        detent_strength_unit=0.01,
        endstop_strength_unit=0.6,
        snap_point=1.1,
        text="Return-to-center",
        led_hue=45,
    ),
    KnobConfig(
        position=127,
        position_nonce=5,
        min_position=0,
        max_position=255,
        position_width_radians=_deg(1),
        detent_strength_unit=0,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Fine values\nNo detents",
        led_hue=219,
    ),
    KnobConfig(
        position=127,
        position_nonce=5,
        min_position=0,
        max_position=255,
        position_width_radians=_deg(1),
        detent_strength_unit=1,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Fine values\nWith detents",
        led_hue=25,
    ),
    KnobConfig(
        position_nonce=6,
        min_position=0,
        max_position=31,
        position_width_radians=_deg(8.225806452),
        detent_strength_unit=2,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Coarse values\nStrong detents",
        led_hue=200,
    ),
    KnobConfig(
        position_nonce=6,
        min_position=0,
        max_position=31,
        position_width_radians=_deg(8.225806452),
        detent_strength_unit=0.2,
        endstop_strength_unit=1,
        snap_point=1.1,
        text="Coarse values\nWeak detents",
    ),
    KnobConfig(
        position_nonce=7,
        min_position=0,
        max_position=31,
        position_width_radians=_deg(7),
        detent_strength_unit=2.5,
        endstop_strength_unit=1,
        snap_point=0.7,
        text="Magnetic detents",
        detent_positions=(2, 10, 21, 22),
        led_hue=73,
    ),
    KnobConfig(
        position_nonce=8,
        min_position=-6,
        max_position=6,
        position_width_radians=_deg(60),
        detent_strength_unit=1,
        endstop_strength_unit=1,
        snap_point=0.55,
        text="Return-to-center\nwith detents",
        snap_point_bias=0.4,
        led_hue=157,
    ),
)


@dataclass
class MotorConfig:
    """Physical motor + drive parameters.

    Defaults are for the Wanzhida/Oncetop OT-EM3215D2450Y1R (SparkFun ROB-20441),
    the motor this project and the original SmartKnob both use.

    ``torque_constant`` is derived two ways from the SparkFun datasheet and they
    agree to ~25%:
      - start torque / start current = 320 g.cm / 0.8 A = 0.039 N.m/A
      - 8.27 / KV, KV = 2000 rpm / 7.4 V = 270           = 0.031 N.m/A
    """

    #: N.m per amp of Iq.
    torque_constant: float = 0.035
    #: Hard ceiling on commanded current. The datasheet's 0.8 A is a *starting*
    #: (transient) rating -- ~4.4 W of copper loss in a 32 mm motor -- so holding
    #: it continuously will cook the motor. See README "Thermal".
    max_current: float = 0.8
    #: Runaway guard, turns/s. The knob should never legitimately exceed this.
    velocity_limit: float = 20.0

    @property
    def peak_torque(self) -> float:
        """N.m available at ``max_current``."""
        return self.torque_constant * self.max_current


@dataclass
class TuningConfig:
    """ODrive controller tuning.

    ``vel_gain`` is the local 8 kHz damper and is the single most important
    number here -- it is what keeps the spring from buzzing, and it must live on
    the ODrive because damping through host latency becomes *negative* damping.
    Tune it once, by hand; everything else is derived from the knob config.
    """

    #: N.m per (turn/s). The local damper.
    vel_gain: float = 0.02
    #: ODrive's velocity integrator. Usually 0 for haptics -- integral action
    #: fights the user.
    vel_integrator_gain: float = 0.0
    #: Clamp on the derived pos_gain, in (turn/s)/turn.
    max_pos_gain: float = 500.0
    #: Deflection, as a fraction of one detent width, at which
    #: detent_strength_unit == 1 should produce peak motor torque.
    detent_full_scale: float = 0.5

    #: Seconds of velocity extrapolation applied before evaluating the snap
    #: point, to compensate host+CAN+USB latency. Click position error is
    #: omega * latency, so this is what keeps detent clicks landing in a
    #: consistent place. Measure your actual round-trip and set it to that.
    #: 0.0 disables (safe default -- overshoot causes early snaps).
    latency_compensation_s: float = 0.0

    #: Emulate the firmware's dead zone (motor_task.cpp:252) by parking the
    #: setpoint on the current angle when very close to the detent centre.
    #: Off by default: with a well-tuned vel_gain the zero-position buzz it
    #: exists to suppress doesn't occur.
    dead_zone_enabled: bool = False
