"""End-to-end tests: control loop <-> CANSimple codec <-> simulated ODrive.

These exercise the real CAN encode/decode path (over a python-can virtual bus),
so they catch protocol-layer mistakes that the pure-logic tests in
``test_haptics.py`` cannot.

They run in real time against the simulator's physics, so they're slow-ish by
unit test standards. Kept to ~1 s each.
"""

import math
import time

import can
import pytest

from skodrive.config import KnobConfig, MotorConfig, TuningConfig
from skodrive.knob import SmartKnob
from skodrive.odrive_can import ODriveAxis
from skodrive.sim import SimParams, SimulatedODrive

TWO_PI = 2 * math.pi


@pytest.fixture
def rig():
    """A knob wired to a simulated ODrive over a virtual CAN bus."""
    channel = f"test-{time.monotonic_ns()}"
    host_bus = can.Bus(interface="virtual", channel=channel)
    sim_bus = can.Bus(interface="virtual", channel=channel)

    sim = SimulatedODrive(sim_bus, node_id=0, params=SimParams())
    sim.start()

    axis = ODriveAxis(host_bus, node_id=0)
    knob = SmartKnob(
        axis,
        motor=MotorConfig(),
        tuning=TuningConfig(vel_gain=0.02),
        config=KnobConfig(
            position_width_radians=math.radians(20),
            detent_strength_unit=1.0,
            endstop_strength_unit=1.0,
            snap_point=0.6,
            min_position=0,
            max_position=10,
        ),
        rate_hz=200.0,
    )
    knob.start()
    try:
        yield knob, sim
    finally:
        knob.stop()
        sim.stop()
        host_bus.shutdown()
        sim_bus.shutdown()


def settle(seconds=0.4):
    time.sleep(seconds)


#: The motor saturates at max_current * torque_constant = 0.8 * 0.035 = 0.028 N.m,
#: so a "hand" torque must exceed that to push through a detent at all. Below it
#: the spring simply balances the hand at a partial deflection.
HAND_TORQUE = 0.05


def test_loop_runs_at_requested_rate(rig):
    knob, _ = rig
    settle(1.2)
    assert 150 <= knob.stats.rate_hz <= 230
    assert knob.stats.overruns == 0


def test_encoder_feedback_is_fresh(rig):
    knob, _ = rig
    settle(0.5)
    # Sim emits Get_Encoder_Estimates every 2 ms; anything stale means the RX
    # thread or the arbitration IDs are wrong.
    assert knob.stats.feedback_age_ms < 25


def test_can_traffic_flows_both_ways(rig):
    knob, _ = rig
    settle(0.5)
    tx, rx = knob.stats.tx, knob.stats.rx
    assert tx > 50   # Set_Input_Pos every iteration
    assert rx > 50   # cyclic encoder estimates


def test_external_torque_advances_position(rig):
    knob, sim = rig
    assert knob.latest.position == 0
    sim.apply_external_torque(HAND_TORQUE)
    settle(0.5)
    sim.apply_external_torque(0.0)
    settle(0.3)
    assert knob.latest.position > 0


def test_spring_holds_at_a_detent_after_release(rig):
    """Release mid-detent and the spring should pull back to a centre."""
    knob, sim = rig
    sim.apply_external_torque(HAND_TORQUE)
    settle(0.15)                      # short nudge, so we don't hit the endstop
    sim.apply_external_torque(0.0)
    settle(0.8)

    state = knob.latest
    # Settled within the snap point of a detent centre, not drifting.
    assert abs(state.sub_position_unit) < state.config.snap_point
    assert abs(state.velocity_rad_s) < 0.5


def test_endstop_blocks_travel_past_max(rig):
    knob, sim = rig
    sim.apply_external_torque(HAND_TORQUE)   # push hard, continuously
    settle(1.5)
    state = knob.latest
    assert state.position == state.config.max_position
    assert state.out_of_bounds
    sim.apply_external_torque(0.0)


def test_pos_gain_is_pushed_to_the_drive(rig):
    """Detent strength must actually reach the ODrive as a gain change."""
    knob, _ = rig
    settle(0.3)
    assert knob.latest.pos_gain > 0

    knob.set_config(knob.config.replace(detent_strength_unit=0.0, position_nonce=99))
    settle(0.3)
    assert knob.latest.pos_gain == 0.0


def test_config_switch_does_not_jolt_the_knob(rig):
    """Changing mode must re-anchor the spring, not yank the rotor."""
    knob, sim = rig
    sim.apply_external_torque(HAND_TORQUE)
    settle(0.15)
    sim.apply_external_torque(0.0)
    settle(0.3)

    before = knob.latest.angle_rad
    knob.set_config(
        KnobConfig(
            position_width_radians=math.radians(60),
            detent_strength_unit=1.0,
            min_position=-6,
            max_position=6,
            position=0,
            position_nonce=42,
            snap_point=0.55,
        )
    )
    settle(0.4)
    # The knob should stay roughly where it was rather than snapping elsewhere.
    assert abs(knob.latest.angle_rad - before) < math.radians(25)
