"""Tests for the detent state machine.

The original firmware has no tests (``firmware/test/`` is a README), which left
the subtle behaviours -- snap hysteresis, bias asymmetry, bounds, magnetic
detents -- protected by nothing. These pin them down before the port can drift.
"""

import math

import pytest

from skodrive.config import KnobConfig, MotorConfig, TuningConfig
from skodrive.haptics import TWO_PI, DetentEngine

MOTOR = MotorConfig()
TUNING = TuningConfig()


def engine(cfg: KnobConfig, **tune) -> DetentEngine:
    t = TuningConfig(**{**vars(TUNING), **tune})
    e = DetentEngine(cfg, motor=MOTOR, tuning=t)
    e.set_config(cfg, 0.0)
    return e


def rot(e: DetentEngine, angle: float, velocity: float = 0.0, dt: float = 0.005):
    return e.update(angle, velocity, dt)


# --------------------------------------------------------------------- snapping

def test_stays_put_inside_snap_point():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=10)
    e = engine(cfg)
    # 1.1 * 10deg = 11deg of travel needed; 10deg must not snap.
    assert rot(e, math.radians(10)).position == 0


def test_snaps_past_snap_point():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=10)
    e = engine(cfg)
    assert rot(e, math.radians(12)).position == 1


def test_snap_is_hysteretic():
    """Coming back from position 1 must not immediately drop to 0."""
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=10)
    e = engine(cfg)
    rot(e, math.radians(12))
    assert e.position == 1
    # Back to 10deg: only 2deg below the new centre, well inside the snap point.
    assert rot(e, math.radians(10)).position == 1


def test_direction_symmetry():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=-10, max_position=10)
    e = engine(cfg)
    assert rot(e, math.radians(-12)).position == -1


# ------------------------------------------------------- multi-step (the fix)

def test_multi_detent_jump_in_one_update():
    """The firmware moved one detent per iteration; at 200 Hz that aliases.

    A 90deg jump across 10deg detents must land near position 9, not at 1.
    With snap_point=1.1 the last detent isn't earned until 99deg, so 8 is
    correct -- the point is that we don't stop at 1.
    """
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=100)
    e = engine(cfg)
    assert rot(e, math.radians(90)).position == 8


def test_multi_step_reports_step_count():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=100)
    e = engine(cfg)
    assert rot(e, math.radians(90)).steps == 8


def test_fast_spin_tracks_fine_detents():
    """1deg detents at 3 rev/s: 1080 detents/s against a 200 Hz loop."""
    cfg = KnobConfig(position_width_radians=math.radians(1), snap_point=1.1,
                     min_position=0, max_position=100_000)
    e = engine(cfg)
    dt = 1 / 200
    angle = 0.0
    for _ in range(200):  # one second
        angle += 3 * TWO_PI * dt
        st = rot(e, angle, velocity=3 * TWO_PI, dt=dt)
    # ~1080 degrees travelled => ~1080 positions, allowing for the snap offset.
    assert 1070 <= st.position <= 1082


# ----------------------------------------------------------------- bounds

def test_clamps_to_max_position():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=3)
    e = engine(cfg)
    st = rot(e, math.radians(200))
    assert st.position == 3
    assert st.out_of_bounds


def test_clamps_to_min_position():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=3)
    e = engine(cfg)
    st = rot(e, math.radians(-200))
    assert st.position == 0
    assert st.out_of_bounds


def test_unbounded_when_max_below_min():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=-1)
    e = engine(cfg)
    st = rot(e, math.radians(500))
    assert st.position > 40
    assert not st.out_of_bounds


def test_endstop_uses_endstop_strength():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=1,
                     detent_strength_unit=0.0, endstop_strength_unit=1.0)
    e = engine(cfg)
    free = rot(e, math.radians(4))
    assert free.pos_gain == 0.0          # no detents configured
    pushed = rot(e, math.radians(-30))   # past the min bound
    assert pushed.out_of_bounds
    assert pushed.pos_gain > 0.0         # endstop spring engages


# ------------------------------------------------------------ magnetic detents

def test_magnetic_detents_only_spring_on_listed_positions():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=0.7,
                     min_position=0, max_position=30,
                     detent_strength_unit=1.0, detent_positions=(5,))
    e = engine(cfg)
    assert rot(e, math.radians(20)).pos_gain == 0.0     # position 2, not listed
    assert rot(e, math.radians(50)).pos_gain > 0.0      # position 5, listed


# ----------------------------------------------------------------- snap bias

def test_bias_makes_returning_home_easier_than_leaving():
    cfg = KnobConfig(position_width_radians=math.radians(60), snap_point=0.55,
                     min_position=-6, max_position=6, snap_point_bias=0.4)
    width = cfg.position_width_radians

    # Leaving home costs snap + bias = 0.95 width.
    leaving = engine(cfg)
    assert rot(leaving, width * 0.9).position == 0
    assert rot(leaving, width * 1.0).position == 1

    # Returning costs snap - bias = 0.15 width from the position-1 centre.
    returning = engine(cfg)
    rot(returning, width * 1.0)
    assert returning.position == 1
    assert rot(returning, width * 0.8).position == 0


# --------------------------------------------------------------- gain mapping

def test_pos_gain_scales_with_detent_strength():
    base = KnobConfig(position_width_radians=math.radians(10), detent_strength_unit=1.0,
                      min_position=0, max_position=10)
    weak = rot(engine(base.replace(detent_strength_unit=0.5)), 0.0).pos_gain
    strong = rot(engine(base), 0.0).pos_gain
    assert strong == pytest.approx(weak * 2, rel=1e-6)


def test_detent_strength_unit_is_width_independent():
    """strength=1 should mean the same peak torque regardless of detent width."""
    narrow = KnobConfig(position_width_radians=math.radians(5), detent_strength_unit=1.0,
                        min_position=0, max_position=10)
    wide = narrow.replace(position_width_radians=math.radians(20))

    k_narrow = rot(engine(narrow), 0.0).pos_gain * math.radians(5) * TUNING.detent_full_scale
    k_wide = rot(engine(wide), 0.0).pos_gain * math.radians(20) * TUNING.detent_full_scale
    assert k_narrow == pytest.approx(k_wide, rel=1e-6)


def test_pos_gain_is_clamped():
    cfg = KnobConfig(position_width_radians=math.radians(0.5),
                     detent_strength_unit=100.0, min_position=0, max_position=10)
    assert rot(engine(cfg), 0.0).pos_gain == TUNING.max_pos_gain


def test_zero_strength_gives_free_rotation():
    cfg = KnobConfig(position_width_radians=math.radians(10), detent_strength_unit=0.0,
                     min_position=0, max_position=-1)
    assert rot(engine(cfg), math.radians(3)).pos_gain == 0.0


# ------------------------------------------------------------ latency comp

def test_latency_compensation_snaps_earlier_when_moving():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=100)
    velocity = 2 * TWO_PI  # 2 rev/s

    plain = engine(cfg, latency_compensation_s=0.0)
    comped = engine(cfg, latency_compensation_s=0.01)

    angle = math.radians(9)
    assert rot(plain, angle, velocity).position == 0
    # 0.01 s at 2 rev/s = 7.2deg of lookahead, carrying us past the snap point.
    assert rot(comped, angle, velocity).position == 1


def test_latency_compensation_inert_at_rest():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=100)
    e = engine(cfg, latency_compensation_s=0.02)
    assert rot(e, math.radians(10), velocity=0.0).position == 0


# ------------------------------------------------------------------- config

def test_setpoint_tracks_detent_centre_in_turns():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=0, max_position=100)
    e = engine(cfg)
    st = rot(e, math.radians(12))
    assert st.setpoint_turns == pytest.approx(st.detent_center_rad / TWO_PI)


def test_invert_flips_position_direction():
    cfg = KnobConfig(position_width_radians=math.radians(10), snap_point=1.1,
                     min_position=-10, max_position=10)
    normal = DetentEngine(cfg, MOTOR, TUNING, invert=False)
    normal.set_config(cfg, 0.0)
    inverted = DetentEngine(cfg, MOTOR, TUNING, invert=True)
    inverted.set_config(cfg, 0.0)

    assert normal.update(math.radians(12), 0, 0.005).position == 1
    assert inverted.update(math.radians(12), 0, 0.005).position == -1


def test_config_change_reanchors_spring_without_jump():
    """Switching modes must not yank the knob to a new setpoint."""
    cfg = KnobConfig(position_width_radians=math.radians(10), min_position=0,
                     max_position=10, position=0)
    e = engine(cfg)
    rot(e, math.radians(35))

    angle = math.radians(35)
    e.set_config(cfg.replace(position=5, position_nonce=1,
                             position_width_radians=math.radians(20)), angle)
    st = rot(e, angle)
    assert st.position == 5
    # Spring anchored on the current angle => no torque at the moment of switch.
    assert st.detent_center_rad == pytest.approx(angle, abs=1e-9)


def test_config_position_is_idempotent_without_nonce_change():
    cfg = KnobConfig(position_width_radians=math.radians(10), min_position=0,
                     max_position=10, position=0)
    e = engine(cfg)
    rot(e, math.radians(35))
    assert e.position == 3
    e.set_config(cfg, math.radians(35))  # same config object
    assert e.position == 3               # not reset to 0


def test_config_clamps_position_into_new_bounds():
    cfg = KnobConfig(position_width_radians=math.radians(10), min_position=0,
                     max_position=100)
    e = engine(cfg)
    rot(e, math.radians(200))
    assert e.position > 5
    e.set_config(cfg.replace(min_position=0, max_position=5), math.radians(200))
    assert e.position == 5


@pytest.mark.parametrize("bad", [
    {"detent_strength_unit": -1},
    {"endstop_strength_unit": -1},
    {"snap_point": 0.4},
    {"snap_point_bias": -0.1},
    {"position_width_radians": 0},
])
def test_invalid_configs_rejected(bad):
    cfg = KnobConfig(**bad)
    e = DetentEngine(KnobConfig(), MOTOR, TUNING)
    with pytest.raises(ValueError):
        e.set_config(cfg, 0.0)


# ------------------------------------------------------------------- idle

def test_idle_correction_pulls_centre_toward_resting_angle():
    cfg = KnobConfig(position_width_radians=math.radians(10), min_position=0,
                     max_position=10, detent_strength_unit=1.0)
    e = engine(cfg)
    resting = math.radians(2)
    for _ in range(400):  # 2 s at 200 Hz, past IDLE_CORRECTION_DELAY_S
        st = rot(e, resting, velocity=0.0, dt=0.005)
    assert abs(st.detent_center_rad - resting) < abs(0 - resting)
