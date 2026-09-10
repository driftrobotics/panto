"""The 2-axis simulator, exercised directly on a virtual CAN bus.

These talk raw CANSimple frames (no CanLink) to prove the sim *is* a faithful
bus peer: right arbitration ids, right signal names/scaling, a stable ODrive
cascade, 12-bit quantisation in the loop.
"""

from __future__ import annotations

import json
import math
import time

import can
import cantools
import pytest

from panto.can_link import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CMD,
    CONTROL_MODE_POSITION,
    CONTROL_MODE_TORQUE,
    DEFAULT_DBC,
    INPUT_MODE_PASSTHROUGH,
)
from panto.sim import PantoSim, SimParams, _AxisPlant, _inertia_matrix

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


# ---------------------------------------------------------------------------
# pure, no-I/O tests on _AxisPlant / _inertia_matrix directly -- fast and
# exact, no thread timing involved (house style: pure logic gets no-I/O
# tests). _AxisPlant is "private" only by convention; sim.py and this test
# file are both owned by the same stream.
# ---------------------------------------------------------------------------

def test_torsion_spring_holds_static_offset_proportional_to_disturbance():
    """A constant external ('hand') torque against a torsion spring settles
    to angle_home + tau_ext / (k * Kt) -- Hooke's law at equilibrium, net of
    the spring's own restoring torque -k*(angle-home)*Kt."""
    k_a_per_rad = 2.0
    kt = 0.035
    p = SimParams(torsion_a_per_rad=k_a_per_rad, torque_constant=kt, friction=0.0)
    plant = _AxisPlant(p, home_rad=0.3)
    tau_ext = 0.01  # N.m
    plant.set_external_torque(tau_ext)
    plant.step(40_000)  # 5s sim time -- underdamped 2nd order, settles well inside 1s
    expected_offset = tau_ext / (k_a_per_rad * kt)
    assert plant.true_angle() == pytest.approx(0.3 + expected_offset, rel=0.02)
    assert abs(plant.velocity) < 1e-3

    # zero disturbance -> spring holds exactly at home
    plant2 = _AxisPlant(SimParams(torsion_a_per_rad=k_a_per_rad, friction=0.0), home_rad=-0.5)
    plant2.step(40_000)
    assert plant2.true_angle() == pytest.approx(-0.5, abs=1e-6)


def test_delay_shifts_controller_torque_by_n_ticks():
    """A constant Input_Torque step's effect on Iq_Measured should be held at
    zero for exactly delay_ticks ticks, then appear in full -- a pure FIFO,
    not a filtered/smoothed lag."""
    dt = SimParams().control_dt
    delay_ticks = 5
    p = SimParams(delay_s=delay_ticks * dt, friction=0.0, damping=0.0)
    plant = _AxisPlant(p, home_rad=0.0)
    plant.apply_command("Set_Controller_Mode", {"Control_Mode": CONTROL_MODE_TORQUE})
    plant.apply_command("Set_Limits", {"Velocity_Limit": 1000.0, "Current_Limit": 100.0})
    plant.apply_command("Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_CLOSED_LOOP_CONTROL})
    plant.apply_command("Set_Input_Torque", {"Input_Torque": 0.35})  # 10A @ Kt=0.035

    seen = []
    for _ in range(10):
        plant.step(1)
        seen.append(plant.iq_measured())

    assert seen[:delay_ticks] == pytest.approx([0.0] * delay_ticks, abs=1e-9)
    assert all(v == pytest.approx(10.0, abs=1e-6) for v in seen[delay_ticks:])


def test_delay_zero_is_immediate():
    """delay_s=0.0 (the default) must not introduce even a 1-tick lag."""
    p = SimParams(delay_s=0.0, friction=0.0, damping=0.0)
    plant = _AxisPlant(p, home_rad=0.0)
    plant.apply_command("Set_Controller_Mode", {"Control_Mode": CONTROL_MODE_TORQUE})
    plant.apply_command("Set_Limits", {"Velocity_Limit": 1000.0, "Current_Limit": 100.0})
    plant.apply_command("Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_CLOSED_LOOP_CONTROL})
    plant.apply_command("Set_Input_Torque", {"Input_Torque": 0.35})
    plant.step(1)
    assert plant.iq_measured() == pytest.approx(10.0, abs=1e-6)


def test_coulomb_a_overrides_friction():
    """coulomb_a, when set, replaces `friction` (N.m) as the deadband via
    coulomb_a * torque_constant -- not additive with it."""
    kt = 0.02235
    p = SimParams(coulomb_a=0.5, torque_constant=kt, friction=999.0)  # friction would swamp everything if used
    plant = _AxisPlant(p, home_rad=0.0)
    # a torque just under the coulomb deadband should be fully absorbed
    plant.set_external_torque(0.5 * kt * 0.9)
    plant.step(100)
    assert plant.true_angle() == pytest.approx(0.0, abs=1e-9)
    # a torque just over it should produce motion
    plant2 = _AxisPlant(SimParams(coulomb_a=0.5, torque_constant=kt), home_rad=0.0)
    plant2.set_external_torque(0.5 * kt * 1.5)
    plant2.step(4000)
    assert abs(plant2.true_angle()) > 1e-6


def test_from_plant_model_maps_fields_and_skips_nulls(tmp_path):
    data = {
        "inertia_a_s2_per_rad": 0.003,
        "delay_s": 0.016,
        "viscous_a_per_rad_s": {"+": 0.02, "-": 0.01},
        "friction_kinetic_intercept_a": {"+": None, "-": None},
    }
    path = tmp_path / "plant_model.json"
    path.write_text(json.dumps(data))

    params = SimParams.from_plant_model(path, torque_constant=0.02235)
    defaults = SimParams()

    assert params.torque_constant == 0.02235
    assert params.inertia == pytest.approx(0.003 * 0.02235)
    assert params.damping == pytest.approx(0.015 * 0.02235)  # mean(0.02, 0.01)
    assert params.coulomb_a is None  # both directions null -> default kept
    assert params.delay_s == pytest.approx(0.016)
    assert params.torsion_a_per_rad == defaults.torsion_a_per_rad  # field absent -> default
    # untouched fields keep the class defaults
    assert params.friction == defaults.friction
    assert params.encoder_bandwidth == defaults.encoder_bandwidth


def test_from_plant_model_all_null_falls_back_to_defaults(tmp_path):
    data = {
        "inertia_a_s2_per_rad": None,
        "delay_s": None,
        "viscous_a_per_rad_s": {"+": None, "-": None},
        "friction_kinetic_intercept_a": {"+": None, "-": None},
    }
    path = tmp_path / "plant_model.json"
    path.write_text(json.dumps(data))

    params = SimParams.from_plant_model(path, torque_constant=0.05)
    defaults = SimParams()

    assert params.torque_constant == 0.05
    assert params.inertia == defaults.inertia
    assert params.damping == defaults.damping
    assert params.coulomb_a is None
    assert params.delay_s == defaults.delay_s


def test_golden_trajectory_defaults_are_deterministic():
    """Pins current default-parameter behaviour: a fixed command sequence
    against a fresh _AxisPlant always lands at the same (angle, velocity),
    bit-for-bit, since the sim has no randomness. Regressions to the shared
    `_tick_common` refactor (touched by the torsion/delay/coulomb additions)
    would show up here even if no other test catches them."""
    plant = _AxisPlant(SimParams(), home_rad=0.0)
    plant.apply_command("Set_Controller_Mode", {"Control_Mode": CONTROL_MODE_POSITION})
    plant.apply_command("Set_Limits", {"Velocity_Limit": 40.0, "Current_Limit": 4.0})
    plant.apply_command("Set_Pos_Gain", {"Pos_Gain": 80.0})
    plant.apply_command("Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_CLOSED_LOOP_CONTROL})
    plant.apply_command("Set_Input_Pos", {"Input_Pos": 0.1})
    plant.step(4000)
    plant.set_external_torque(0.01)
    plant.step(2000)
    plant.set_external_torque(0.0)
    plant.step(2000)

    # Recomputed from this exact implementation once, then pinned -- a
    # behaviour change to the shared default path (not just the new opt-in
    # features) will move these numbers.
    assert plant.true_angle() == pytest.approx(0.6226155563572194, abs=1e-9)
    assert plant.velocity == pytest.approx(-7.347225391313464e-06, abs=1e-9)


# ---------------------------------------------------------------------------
# coupled 2R dynamics
# ---------------------------------------------------------------------------

def test_inertia_matrix_is_positive_definite_across_q2():
    p = SimParams()
    for q2 in (-2.5, -1.0, 0.0, 0.5, 1.5, math.pi):
        m11, m12, m22 = _inertia_matrix(q2, p, p.inertia, p.inertia)
        assert m11 > 0.0
        assert m22 > 0.0
        assert m11 * m22 - m12 * m12 > 0.0  # positive-definite: a valid mass matrix


def test_inertia_matrix_m12_matches_hand_derivation_at_q2_zero():
    """At q2=0 (arm straight), M12 = m2*(com2^2 + l1*com2) with l1=2*com1 --
    checked by hand against the standard 2-link point-mass result rather than
    just re-deriving the same formula the implementation uses."""
    p = SimParams(m1=0.04, m2=0.02, com1=0.05, com2=0.03)
    l1 = 2 * p.com1
    _m11, m12, _m22 = _inertia_matrix(0.0, p, inertia0=0.0, inertia1=0.0)
    expected_m12 = p.m2 * (p.com2 ** 2 + l1 * p.com2)
    assert m12 == pytest.approx(expected_m12)


@pytest.fixture
def coupled_rig():
    channel = f"simtest-coupled-{time.monotonic_ns()}"
    host = can.Bus(interface="virtual", channel=channel)
    sim_bus = can.Bus(interface="virtual", channel=channel)
    sim = PantoSim(sim_bus, node_ids=(0, 1), params=SimParams(friction=0.0), coupled=True)
    sim.start()
    try:
        yield host, sim
    finally:
        sim.stop()
        host.shutdown()
        sim_bus.shutdown()


def test_coupled_mode_shoulder_motion_backdrives_free_elbow(coupled_rig):
    """The point of coupled dynamics: node 1 (elbow) is never commanded, but
    driving node 0 (shoulder) hard through the inertial+Coriolis coupling
    should move it anyway."""
    host, sim = coupled_rig
    assert sim.true_angle(1) == pytest.approx(0.0, abs=1e-9)
    _arm(host, 0, gain=200.0)
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        _send(host, 0, "Set_Input_Pos", {"Input_Pos": 1.0 / TWO_PI, "Vel_FF": 0.0, "Torque_FF": 0.0})
        time.sleep(0.01)
    assert abs(sim.true_angle(1)) > 1e-3


def test_coupled_mode_point_hold_still_converges(coupled_rig):
    """Coupling must not break the existing point-hold use case (both axes
    position-controlled, as tests/test_runtime.py-style callers rely on)."""
    host, sim = coupled_rig
    _arm(host, 0)
    _arm(host, 1)
    target0, target1 = 0.3, -0.2
    # Coupled effective inertia at the shoulder is much larger than a single
    # rotor's (it now carries the whole arm's mass), so convergence is slower
    # than the single-axis test_position_loop_converges_and_holds -- give it
    # more wall time rather than a looser tolerance.
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        _send(host, 0, "Set_Input_Pos", {"Input_Pos": target0 / TWO_PI, "Vel_FF": 0.0, "Torque_FF": 0.0})
        _send(host, 1, "Set_Input_Pos", {"Input_Pos": target1 / TWO_PI, "Vel_FF": 0.0, "Torque_FF": 0.0})
        time.sleep(0.02)
    assert sim.true_angle(0) == pytest.approx(target0, abs=0.05)
    assert sim.true_angle(1) == pytest.approx(target1, abs=0.05)


def test_coupled_defaults_to_false_matches_independent_behaviour():
    """coupled=False (the default) must integrate identically to two
    independent _AxisPlant.step calls -- no coupling terms sneak in."""
    p = SimParams()
    plant_a = _AxisPlant(p, home_rad=0.2)
    plant_b = _AxisPlant(p, home_rad=-0.4)
    plant_a.set_external_torque(0.01)
    plant_b.set_external_torque(-0.005)
    for _ in range(500):
        plant_a.step(1)
        plant_b.step(1)

    channel = f"simtest-uncoupled-{time.monotonic_ns()}"
    sim_bus = can.Bus(interface="virtual", channel=channel)
    sim = PantoSim(sim_bus, node_ids=(0, 1), params=p, home_rad=(0.2, -0.4), coupled=False)
    sim.set_external_torque(0, 0.01)
    sim.set_external_torque(1, -0.005)
    for plant in sim._plants.values():
        plant.step(500)
    sim_bus.shutdown()

    assert sim.true_angle(0) == pytest.approx(plant_a.true_angle(), abs=1e-12)
    assert sim.true_angle(1) == pytest.approx(plant_b.true_angle(), abs=1e-12)
