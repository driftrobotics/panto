"""CanLink over the virtual-bus simulator.

Style follows reference/odrive_knob/tests/test_integration.py: exercise the real
CANSimple encode/decode path against the sim's physics, so protocol-layer
mistakes (arbitration ids, signal names, scaling, the calibration transform)
surface without hardware. Real-time against the sim, kept to ~1 s each.
"""

from __future__ import annotations

import math
import time

from types import SimpleNamespace

import numpy as np
import pytest

from panto.can_link import (
    CanLink,
    joint_from_turns,
    jointvel_from_turns,
    motor_torque_from_joint,
    turns_from_joint,
)
from panto.config import Config

TWO_PI = 2 * math.pi


def _motor(node_id, flip=False, zero=0.0):
    # stand-in for the extended MotorConfig stream D still needs to ship
    # (adds flip / zero_offset_rad); CanLink only reads attributes.
    return SimpleNamespace(
        node_id=node_id, flip=flip, zero_offset_rad=zero,
        torque_constant=0.035, current_soft_max=0.8,
        vel_gain=0.0025, vel_integrator_gain=0.0, max_pos_gain=500.0,
    )


# --------------------------------------------------------------------------
# pure calibration transform — no bus
# --------------------------------------------------------------------------

@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("zero", [0.0, 0.37, -1.2])
@pytest.mark.parametrize("q", [-0.9, 0.0, 0.5, 2.1])
def test_calibration_position_roundtrips(flip, zero, q):
    turns = turns_from_joint(q, flip, zero)
    assert joint_from_turns(turns, flip, zero) == pytest.approx(q, abs=1e-12)


def test_flipped_motor_reverses_sign():
    # same encoder reading, opposite mounting -> opposite joint motion
    assert joint_from_turns(0.25, False, 0.0) == pytest.approx(TWO_PI * 0.25)
    assert joint_from_turns(0.25, True, 0.0) == pytest.approx(-TWO_PI * 0.25)
    assert jointvel_from_turns(1.0, True) == pytest.approx(-TWO_PI)
    assert motor_torque_from_joint(0.01, True) == pytest.approx(-0.01)


def test_zero_offset_is_additive_in_joint_space():
    assert joint_from_turns(0.0, False, 0.42) == pytest.approx(0.42)
    assert turns_from_joint(0.42, False, 0.42) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# against the sim
# --------------------------------------------------------------------------

def _config(flip=(False, True), zero=(0.0, 0.0)):
    return Config(motors=(
        _motor(0, flip=flip[0], zero=zero[0]),
        _motor(1, flip=flip[1], zero=zero[1]),
    ))


@pytest.fixture
def link():
    lk = CanLink(_config(), sim=True)
    lk.start()
    lk.wait_for_feedback(timeout=5.0)
    try:
        yield lk
    finally:
        lk.stop()


def _arm(lk, gain=200.0):
    for nid in (0, 1):
        lk.set_controller_mode(nid, "position")
        lk.set_limits(nid, 40.0, 0.8)
        lk.set_pos_gain(nid, gain)
    lk.enter_closed_loop(timeout=5.0)


def _hold(lk, targets, seconds=1.2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for nid, q in zip((0, 1), targets):
            lk.set_input_pos(nid, q)
        time.sleep(0.02)


def test_feedback_is_fresh(link):
    # sim emits Get_Encoder_Estimates every 2 ms; stale => rx thread / ids wrong
    assert link.feedback_age_s() < 0.03


def test_traffic_flows_both_ways(link):
    _arm(link)
    _hold(link, (0.0, 0.0), seconds=0.5)
    tx, rx = link.counters()
    assert tx > 20
    assert rx > 100


def test_closed_loop_entry(link):
    _arm(link)
    # heartbeat must report CLOSED_LOOP for both axes, no error
    assert link.axis_errors() == (0, 0)


def test_position_command_converges(link):
    _arm(link)
    target = (0.4, -0.3)
    _hold(link, target, seconds=1.5)
    q, qd = link.joint_state()
    assert q == pytest.approx(np.array(target), abs=0.03)
    assert np.all(np.abs(qd) < 0.2)


def test_calibration_applied_through_the_stack(link):
    # node 1 is flipped in _config(); commanding +q there must still land at +q
    _arm(link)
    _hold(link, (0.0, 0.5), seconds=1.5)
    q, _ = link.joint_state()
    assert q[1] == pytest.approx(0.5, abs=0.03)
    # ...and the sim's true rotor angle went the *other* way
    assert link._sim.true_angle(1) < -0.1


def test_zero_offset_shifts_reported_angle():
    lk = CanLink(_config(zero=(0.0, 0.6)), sim=True)
    lk.start()
    try:
        lk.wait_for_feedback(timeout=5.0)
        _arm(lk)
        # hold the elbow rotor near its encoder zero -> joint angle ~ offset
        _hold(lk, (0.0, 0.6), seconds=1.2)
        q, _ = lk.joint_state()
        assert q[1] == pytest.approx(0.6, abs=0.03)
        assert abs(lk._sim.true_angle(1)) < 0.05
    finally:
        lk.stop()


def test_external_joint_torque_backdrives_a_relaxed_axis(link):
    # torque mode at zero torque = transparent; a hand torque should move it
    for nid in (0, 1):
        link.set_controller_mode(nid, "torque")
        link.set_limits(nid, 40.0, 0.8)
    link.enter_closed_loop(timeout=5.0)
    for nid in (0, 1):
        link.set_input_torque(nid, 0.0)
    q0, _ = link.joint_state()
    link.inject_joint_torque([0.05, 0.0])
    time.sleep(0.5)
    link.inject_joint_torque([0.0, 0.0])
    time.sleep(0.2)
    q1, _ = link.joint_state()
    assert q1[0] - q0[0] > 0.1
    assert abs(q1[1] - q0[1]) < 0.05          # other axis undisturbed


def test_motor_currents_shape_and_errors(link):
    _arm(link)
    _hold(link, (0.2, -0.2), seconds=0.4)
    cur = link.motor_currents()
    assert cur.shape == (2,)
    assert np.all(np.abs(cur) <= 0.8 + 1e-6)
    assert link.axis_errors() == (0, 0)
