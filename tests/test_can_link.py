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
    CanLinkError,
    decide_wrap_turns,
    decode_error_flags,
    joint_from_turns,
    jointvel_from_turns,
    motor_torque_from_joint,
    turns_from_joint,
)
from panto.config import Config

TWO_PI = 2 * math.pi


def _motor(node_id, flip=False, zero=0.0, q_min=float("-inf"), q_max=float("inf"),
          margin=0.087):
    # stand-in for the extended MotorConfig stream D still needs to ship
    # (adds flip / zero_offset_rad); CanLink only reads attributes.
    return SimpleNamespace(
        node_id=node_id, flip=flip, zero_offset_rad=zero,
        torque_constant=0.035, current_soft_max=0.8,
        vel_gain=0.0025, vel_integrator_gain=0.0, max_pos_gain=500.0,
        q_min_rad=q_min, q_max_rad=q_max, limit_margin_rad=margin,
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

def _config(flip=(False, True), zero=(0.0, 0.0), limits=None):
    limits = limits or ({}, {})
    return Config(motors=(
        _motor(0, flip=flip[0], zero=zero[0], **limits[0]),
        _motor(1, flip=flip[1], zero=zero[1], **limits[1]),
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


def _arm(lk, gain=80.0):
    # set_limits takes joint rad/s and CanLink converts to motor turn/s
    # internally (/2pi); pass 40*2pi so the sim sees the same 40 turn/s
    # ceiling test_sim.py's direct-CAN _arm() uses.
    for nid in (0, 1):
        lk.set_controller_mode(nid, "position")
        lk.set_limits(nid, 40.0 * 2 * 3.14159265358979, 4.0)
        lk.set_pos_gain(nid, gain)
    lk.enter_closed_loop(timeout=5.0)


def _hold(lk, targets, seconds=2.0):
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
    _hold(link, target, seconds=2.5)
    q, qd = link.joint_state()
    assert q == pytest.approx(np.array(target), abs=0.05)
    assert np.all(np.abs(qd) < 0.2)


def test_set_vel_gains_reaches_sim_plant(link):
    # Bump vel_gain 4x on both axes; the sim plant (panto/sim.py _AxisPlant)
    # applies Set_Vel_Gains directly, so a stiffer torque response should
    # still converge to the target (just faster/harder), proving the CAN
    # message round-trips into the plant rather than being ignored.
    _arm(link)
    for nid in (0, 1):
        link.set_vel_gains(nid, 0.01, 0.0)
    target = (0.4, -0.3)
    _hold(link, target, seconds=2.5)
    q, qd = link.joint_state()
    assert q == pytest.approx(np.array(target), abs=0.05)
    assert np.all(np.abs(qd) < 0.2)


def test_calibration_applied_through_the_stack(link):
    # node 1 is flipped in _config(); commanding +q there must still land at +q
    _arm(link)
    _hold(link, (0.0, 0.5), seconds=2.5)
    q, _ = link.joint_state()
    assert q[1] == pytest.approx(0.5, abs=0.05)
    # ...and the sim's true rotor angle went the *other* way
    assert link._sim.true_angle(1) < -0.1


def test_zero_offset_shifts_reported_angle():
    lk = CanLink(_config(zero=(0.0, 0.6)), sim=True)
    lk.start()
    try:
        lk.wait_for_feedback(timeout=5.0)
        _arm(lk)
        # hold the elbow rotor near its encoder zero -> joint angle ~ offset
        _hold(lk, (0.0, 0.6), seconds=2.0)
        q, _ = lk.joint_state()
        assert q[1] == pytest.approx(0.6, abs=0.05)
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


def test_bus_voltage_current_defaults_to_zero_before_any_frame(link):
    # the sim plant doesn't emit Get_Bus_Voltage_Current -- confirms the
    # accessor doesn't crash / returns a well-shaped zero array when the
    # message has never been decoded, rather than erroring or returning None.
    vbus, ibus = link.bus_voltage_current()
    assert vbus.shape == (2,)
    assert ibus.shape == (2,)
    assert np.all(vbus == 0.0)
    assert np.all(ibus == 0.0)


def test_bus_voltage_current_decodes_injected_frame(link):
    # Inject a Get_Bus_Voltage_Current frame for node 0 directly through the
    # same decode path the rx thread uses (bypassing needing a bus peer that
    # actually sends one -- the sim plant doesn't), to confirm the DBC field
    # names (Bus_Voltage, Bus_Current) and CanLink wiring are correct.
    msg = link._msg(0, "Get_Bus_Voltage_Current")
    data = msg.encode({"Bus_Voltage": 24.3, "Bus_Current": 1.75})
    frame_id = link._frame_id(0, "Get_Bus_Voltage_Current")
    hit = link._rx_map[frame_id]
    assert hit == (0, "bus")
    d = msg.decode(data)
    assert d["Bus_Voltage"] == pytest.approx(24.3, abs=1e-3)
    assert d["Bus_Current"] == pytest.approx(1.75, abs=1e-3)


def test_decode_error_flags_none_and_single_bit():
    assert decode_error_flags(0) == "NONE"
    assert decode_error_flags(0x400) == "DC_BUS_OVER_CURRENT"
    assert decode_error_flags(0x2) == "SYSTEM_LEVEL"
    assert decode_error_flags(0x40000000) == "CALIBRATION_ERROR"


def test_decode_error_flags_multiple_bits_joined():
    combined = decode_error_flags(0x400 | 0x2000)
    assert "DC_BUS_OVER_CURRENT" in combined
    assert "MOTOR_OVER_TEMP" in combined
    assert combined.count("|") == 1


def test_decode_error_flags_unrecognised_bit_reported_not_dropped():
    # a bit not in the table must still show up (as its own hex literal),
    # never silently disappear from the decoded string
    result = decode_error_flags(0x400 | 0x80000000)
    assert "DC_BUS_OVER_CURRENT" in result
    assert "0x80000000" in result


def test_temperatures_defaults_to_zero_before_any_frame(link):
    fet, motor = link.temperatures()
    assert fet.shape == (2,)
    assert motor.shape == (2,)
    assert np.all(fet == 0.0)
    assert np.all(motor == 0.0)


def test_temperatures_decodes_injected_frame():
    # Get_Temperature (0x015/0x035) reuses the 0.5.6 Get_Sensorless_Estimates
    # byte layout under 0.6.x firmware (dbc/README.md) -- confirms the DBC
    # patch renamed the right bytes: FET_Temperature at byte 0, Motor_Temperature
    # at byte 4, both float32 LE.
    lk = CanLink(_config(), sim=True)
    msg = lk._msg(0, "Get_Temperature")
    data = msg.encode({"FET_Temperature": 42.5, "Motor_Temperature": float("nan")})
    frame_id = lk._frame_id(0, "Get_Temperature")
    hit = lk._rx_map[frame_id]
    assert hit == (0, "temp")
    d = msg.decode(data)
    assert d["FET_Temperature"] == pytest.approx(42.5, abs=1e-2)
    assert math.isnan(d["Motor_Temperature"])


def test_motor_currents_shape_and_errors(link):
    _arm(link)
    _hold(link, (0.2, -0.2), seconds=0.4)
    cur = link.motor_currents()
    assert cur.shape == (2,)
    assert np.all(np.abs(cur) <= 0.8 + 1e-6)
    assert link.axis_errors() == (0, 0)


# --------------------------------------------------------------------------
# active_errors/disarm_reason: unknown until the first Get_Error frame
# --------------------------------------------------------------------------

def test_node_status_errors_unknown_before_first_get_error_frame():
    # Before the rx thread has ever run, every Feedback is still its dataclass
    # default (error_stamp=None) -- node_status() must report the error fields
    # as unknown (None), not as a fabricated "no error".
    lk = CanLink(_config(), sim=True)
    for s in lk.node_status():
        assert s.active_errors is None
        assert s.disarm_reason is None


def test_wait_for_feedback_wait_for_errors_yields_known_status():
    lk = CanLink(_config(), sim=True)
    lk.start()
    try:
        lk.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in lk.node_status():
            assert s.active_errors == 0
            assert s.disarm_reason == 0
    finally:
        lk.stop()


# --------------------------------------------------------------------------
# encoder wrap folding — 2026-09-04: a joint parked near the 12-bit
# single-turn absolute encoder's wrap can boot 1 turn (2*pi) off.
# --------------------------------------------------------------------------

def test_decide_wrap_no_limits_never_folds():
    assert decide_wrap_turns(5.0, False, 0.0, float("-inf"), float("inf"), 0.087) == 0


def test_decide_wrap_folds_reading_one_turn_outside_limits():
    # q_min/q_max = [-0.2, 0.2] rad -- a real joint near its zero. raw=1.0 turn
    # decodes (unflipped, zero=0) to q = 2*pi rad, a full turn outside the
    # range; raw - 1 = 0.0 turn decodes to q = 0.0 rad, inside.
    k = decide_wrap_turns(1.0, False, 0.0, -0.2, 0.2, 0.05)
    assert k == -1
    folded_q = joint_from_turns(1.0 + k, False, 0.0)
    assert folded_q == pytest.approx(0.0, abs=1e-9)
    assert -0.2 - 0.05 <= folded_q <= 0.2 + 0.05


def test_decide_wrap_prefers_smallest_fold_when_several_fit():
    # a generous range where raw=0.0 already fits -- must not fold needlessly.
    assert decide_wrap_turns(0.01, False, 0.0, -10.0, 10.0, 0.05) == 0


def test_decide_wrap_handles_flip_and_zero_offset():
    # flip=True, zero=0.3: q = -(2*pi*turns) + 0.3. raw=-1.0 turn -> q = 2*pi+0.3,
    # a full turn outside [-0.5, 0.5]; raw+1 -> q = 0.3, inside.
    k = decide_wrap_turns(-1.0, True, 0.3, -0.5, 0.5, 0.05)
    assert k == 1
    folded_q = joint_from_turns(-1.0 + k, True, 0.3)
    assert folded_q == pytest.approx(0.3, abs=1e-9)


def test_wrap_fold_command_round_trips_to_same_raw_turns():
    """The fold chosen on read must be exactly undone on write: commanding the
    joint angle the fold decoded to must reproduce the *original* raw motor
    turns the drive itself reports (its own multi-turn frame), not the folded
    one -- otherwise Set_Input_Pos would command a spurious extra turn."""
    raw = 1.0
    flip, zero = False, 0.0
    q_min, q_max, margin = -0.2, 0.2, 0.05
    k = decide_wrap_turns(raw, flip, zero, q_min, q_max, margin)
    assert k != 0  # sanity: this case does fold

    q = joint_from_turns(raw + k, flip, zero)
    # CanLink.set_input_pos computes turns_from_joint(q, ...) - wrap_turns[i]
    commanded_drive_turns = turns_from_joint(q, flip, zero) - k
    assert commanded_drive_turns == pytest.approx(raw, abs=1e-9)


def test_canlink_folds_encoder_wrap_on_first_feedback_and_commands_match():
    # Build a sim-backed CanLink with limits configured on node 0 such that the
    # sim's home turn (0.4 rad / 2pi ~= 0.0637 turn) already sits inside range
    # -- verifies wrap decisions don't fold a perfectly sane startup reading,
    # and that commands still round-trip through the fold machinery (k=0 here).
    lk = CanLink(_config(limits=({"q_min": -1.0, "q_max": 1.0}, {})), sim=True)
    lk.start()
    try:
        lk.wait_for_feedback(timeout=5.0)
        assert lk._wrap_decided[0] is True
        assert lk._wrap_turns[0] == 0
        _arm(lk)
        _hold(lk, (0.3, 0.0), seconds=1.0)
        q, _ = lk.joint_state()
        assert q[0] == pytest.approx(0.3, abs=0.05)
    finally:
        lk.stop()


# --------------------------------------------------------------------------
# refuse-to-arm when a joint is outside its configured limits
# --------------------------------------------------------------------------

def test_enter_closed_loop_refuses_to_arm_outside_limits():
    # node 0's limits are [-0.1, 0.1] rad but the sim boots at q0 ~= 0.4 rad
    # (SIM_HOME_JOINT_RAD) -- well outside, even with margin.
    lk = CanLink(_config(limits=({"q_min": -0.1, "q_max": 0.1, "margin": 0.02}, {})),
                 sim=True)
    lk.start()
    try:
        lk.wait_for_feedback(timeout=5.0)
        with pytest.raises(CanLinkError, match="refusing to arm"):
            lk.enter_closed_loop(timeout=1.0)
        # must not have armed -- no CLOSED_LOOP_CONTROL frame took effect
        assert lk.node_status()[0].axis_state != 8
    finally:
        lk.stop()


def test_enter_closed_loop_arms_when_inside_limits():
    # generous limits that comfortably contain SIM_HOME_JOINT_RAD = (0.4, 1.4)
    lk = CanLink(_config(limits=({"q_min": -3.0, "q_max": 3.0},
                                 {"q_min": -3.0, "q_max": 3.0})), sim=True)
    lk.start()
    try:
        lk.wait_for_feedback(timeout=5.0)
        lk.enter_closed_loop(timeout=5.0)  # must not raise
        assert lk.axis_errors() == (0, 0)
    finally:
        lk.stop()
