"""The 2-axis simulator, exercised directly on a virtual CAN bus.

These talk raw CANSimple frames (no CanLink) to prove the sim *is* a faithful
bus peer: right arbitration ids, right signal names/scaling, a stable ODrive
cascade, 12-bit quantisation in the loop.
"""

from __future__ import annotations

import math
import time

import can
import cantools
import pytest

from panto.can_link import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CMD,
    CONTROL_MODE_POSITION,
    DEFAULT_DBC,
    INPUT_MODE_PASSTHROUGH,
)
from panto.sim import PantoSim, SimParams

TWO_PI = 2 * math.pi
DB = cantools.database.load_file(str(DEFAULT_DBC))


@pytest.fixture
def rig():
    channel = f"simtest-{time.monotonic_ns()}"
    host = can.Bus(interface="virtual", channel=channel)
    sim_bus = can.Bus(interface="virtual", channel=channel)
    sim = PantoSim(sim_bus, node_ids=(0, 1), params=SimParams())
    sim.start()
    try:
        yield host, sim
    finally:
        sim.stop()
        host.shutdown()
        sim_bus.shutdown()


def _send(host, node_id, base_name, signals=None):
    fid = (node_id << 5) | CMD[base_name]
    data = b"" if signals is None else DB.get_message_by_name(
        f"Axis{node_id}_{base_name}"
    ).encode(signals)
    host.send(can.Message(arbitration_id=fid, data=data, is_extended_id=False))


def _recv_decode(host, node_id, base_name, window=0.25):
    """Newest matching frame within a drain window.

    The virtual bus is an unbounded FIFO; after a test has been running a while
    there's a backlog, so 'read one frame' would hand back a stale value. Drain
    for `window` seconds and return the last match.
    """
    want = (node_id << 5) | CMD[base_name]
    msg = DB.get_message_by_name(f"Axis{node_id}_{base_name}")
    end = time.monotonic() + window + 1.0
    stop_draining = time.monotonic() + window
    last = None
    while time.monotonic() < end:
        frame = host.recv(timeout=0.05)
        if frame is None:
            if last is not None:
                return last
            continue
        if frame.arbitration_id == want:
            last = msg.decode(frame.data)
        if last is not None and time.monotonic() >= stop_draining:
            return last
    if last is not None:
        return last
    raise AssertionError(f"no Axis{node_id}_{base_name} seen")


def _arm(host, node_id, gain=80.0):
    _send(host, node_id, "Set_Controller_Mode",
          {"Control_Mode": CONTROL_MODE_POSITION, "Input_Mode": INPUT_MODE_PASSTHROUGH})
    _send(host, node_id, "Set_Limits", {"Velocity_Limit": 40.0, "Current_Limit": 4.0})
    _send(host, node_id, "Set_Pos_Gain", {"Pos_Gain": gain})
    _send(host, node_id, "Set_Axis_State",
          {"Axis_Requested_State": AXIS_STATE_CLOSED_LOOP_CONTROL})


def test_sim_emits_encoder_frames_for_both_nodes(rig):
    host, _ = rig
    d0 = _recv_decode(host, 0, "Get_Encoder_Estimates")
    d1 = _recv_decode(host, 1, "Get_Encoder_Estimates")
    assert set(d0) == {"Pos_Estimate", "Vel_Estimate"}
    assert d0["Pos_Estimate"] == pytest.approx(0.0, abs=1e-3)
    assert d1["Pos_Estimate"] == pytest.approx(0.0, abs=1e-3)


def test_sim_emits_heartbeat_and_iq(rig):
    host, _ = rig
    hb = _recv_decode(host, 0, "Heartbeat")
    assert "Axis_State" in hb
    iq = _recv_decode(host, 1, "Get_Iq")
    assert "Iq_Measured" in iq


def test_heartbeat_tracks_axis_state(rig):
    host, _ = rig
    _send(host, 0, "Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_CLOSED_LOOP_CONTROL})
    time.sleep(0.25)
    hb = _recv_decode(host, 0, "Heartbeat")
    assert int(getattr(hb["Axis_State"], "value", hb["Axis_State"])) == AXIS_STATE_CLOSED_LOOP_CONTROL


def test_external_torque_advances_rotor(rig):
    host, sim = rig
    assert sim.true_angle(0) == pytest.approx(0.0, abs=1e-6)
    sim.set_external_torque(0, 0.05)
    time.sleep(0.4)
    sim.set_external_torque(0, 0.0)
    assert sim.true_angle(0) > 0.1
    assert sim.true_angle(1) == pytest.approx(0.0, abs=1e-6)   # nodes independent


def test_position_loop_converges_and_holds(rig):
    host, sim = rig
    _arm(host, 0)
    target = 0.35
    end = time.monotonic() + 1.5
    while time.monotonic() < end:
        _send(host, 0, "Set_Input_Pos",
              {"Input_Pos": target / TWO_PI, "Vel_FF": 0.0, "Torque_FF": 0.0})
        time.sleep(0.02)
    d = _recv_decode(host, 0, "Get_Encoder_Estimates")
    assert d["Pos_Estimate"] * TWO_PI == pytest.approx(target, abs=0.03)
    assert abs(d["Vel_Estimate"]) < 0.2


def test_position_loop_resists_a_disturbance(rig):
    host, sim = rig
    _arm(host, 0)
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        _send(host, 0, "Set_Input_Pos", {"Input_Pos": 0.0, "Vel_FF": 0.0, "Torque_FF": 0.0})
        time.sleep(0.02)
    # This loop's max *static* holding torque is vel_gain * Velocity_Limit --
    # at the drives' current (2026-09-03, bumped 10x from the measured 2.5e-4)
    # vel_gain=2.5e-3 and this rig's 40 turn/s limit, that ceiling is 0.1 N.m.
    # 0.002 N.m is comfortably under it (steady-state offset ~ tau/(pos_gain*
    # vel_gain), well inside a small-signal, non-saturated regime).
    sim.set_external_torque(0, 0.002)
    t = time.monotonic() + 1.2
    while time.monotonic() < t:
        _send(host, 0, "Set_Input_Pos", {"Input_Pos": 0.0, "Vel_FF": 0.0, "Torque_FF": 0.0})
        time.sleep(0.02)
    assert abs(sim.true_angle(0)) < 0.1


def test_encoder_estimate_is_quantised(rig):
    host, sim = rig
    # nudge to a non-trivial angle, then let it settle open-loop
    sim.set_external_torque(0, 0.05)
    time.sleep(0.2)
    sim.set_external_torque(0, 0.0)
    time.sleep(0.5)
    quantum_turns = 1.0 / (1 << SimParams().encoder_bits)
    d = _recv_decode(host, 0, "Get_Encoder_Estimates")
    true_turns = sim.true_angle(0) / TWO_PI
    # estimate tracks truth to within a couple of LSBs (PLL + quantiser)
    assert abs(d["Pos_Estimate"] - true_turns) < 3 * quantum_turns


def test_two_nodes_are_independently_addressable(rig):
    host, sim = rig
    _arm(host, 1)
    end = time.monotonic() + 1.2
    while time.monotonic() < end:
        _send(host, 1, "Set_Input_Pos",
              {"Input_Pos": -0.25 / TWO_PI, "Vel_FF": 0.0, "Torque_FF": 0.0})
        time.sleep(0.02)
    assert sim.true_angle(1) == pytest.approx(-0.25, abs=0.03)
    assert sim.true_angle(0) == pytest.approx(0.0, abs=1e-6)
