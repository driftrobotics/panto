"""Unit tests for the pure ramp/abort decision logic behind scripts/breakaway.py.
No CAN/sim needed -- see panto/breakaway_logic.py for why this is split out."""

from __future__ import annotations

import time

import numpy as np
import pytest

from panto.breakaway_logic import check_breakaway, commanded_torque, plateau_vel_limit_rad_s


def test_plateau_vel_limit_scales_with_cap_not_fixed():
    # 2026-09-04: a fixed vel_limit sized the plateau for one current cap and
    # silently re-capped Iq at higher caps. The plateau (in amps) must equal
    # margin * cap regardless of vel_gain/torque_constant -- those only
    # decide what vel_limit is needed to hit it.
    vl_08, plateau_08 = plateau_vel_limit_rad_s(0.8, 0.02235, 0.01)
    vl_25, plateau_25 = plateau_vel_limit_rad_s(2.5, 0.02235, 0.01)
    assert plateau_08 == pytest.approx(1.2)   # 1.5 * 0.8
    assert plateau_25 == pytest.approx(3.75)  # 1.5 * 2.5
    assert vl_25 > vl_08  # higher cap needs a higher vel_limit to hit its (higher) plateau


def test_plateau_vel_limit_matches_coordinator_worked_example():
    # vel_gain 0.01, cap 2.5A -> vel_limit_turns ~= 1.5*2.5*0.02235/0.01 = 8.38 turn/s
    # ~= 52.7 rad/s
    vel_limit_rad_s, plateau_a = plateau_vel_limit_rad_s(2.5, 0.02235, 0.01)
    assert vel_limit_rad_s == pytest.approx(52.66, abs=0.1)
    assert plateau_a == pytest.approx(3.75)


def test_plateau_vel_limit_independent_of_torque_constant_for_plateau_amps():
    # plateau in amps depends only on cap * margin; torque_constant only
    # affects the vel_limit needed to reach it
    _, plateau_a = plateau_vel_limit_rad_s(1.0, 0.035, 0.0025)
    assert plateau_a == pytest.approx(1.5)
from panto.can_link import CanLink
from panto.config import Config, MotorConfig


def test_commanded_torque_ramps_linearly_and_signed():
    assert commanded_torque(0.0, 0.002, 1) == 0.0
    assert commanded_torque(5.0, 0.002, 1) == 0.01
    assert commanded_torque(5.0, 0.002, -1) == -0.01


def test_keeps_ramping_when_nothing_has_happened():
    r = check_breakaway(1.0, moved_deg=0.1, current_a=0.1, rate_nm_s=0.002, sign=1,
                        current_cap_a=0.8, break_deg=2.0, abort_deg=5.0, abort_s=15.0)
    assert r is None


def test_breakaway_detected_at_break_deg():
    r = check_breakaway(3.0, moved_deg=2.05, current_a=0.3, rate_nm_s=0.002, sign=1,
                        current_cap_a=0.8, break_deg=2.0, abort_deg=5.0, abort_s=15.0)
    assert r is not None
    assert r.status == "breakaway"
    assert r.torque_nm == 0.006
    assert r.current_a == 0.3
    assert r.moved_deg == 2.05


def test_current_limit_before_movement():
    r = check_breakaway(5.0, moved_deg=0.3, current_a=0.8, rate_nm_s=0.002, sign=1,
                        current_cap_a=0.8, break_deg=2.0, abort_deg=5.0, abort_s=15.0)
    assert r is not None
    assert r.status == "current_limit"


def test_breakaway_takes_priority_over_current_limit_same_tick():
    # Joint crossed break_deg on the same tick current also happens to be at
    # cap -- report it as a breakaway (it DID move), not a stall.
    r = check_breakaway(6.0, moved_deg=2.1, current_a=0.85, rate_nm_s=0.002, sign=1,
                        current_cap_a=0.8, break_deg=2.0, abort_deg=5.0, abort_s=15.0)
    assert r.status == "breakaway"


def test_runaway_is_the_backstop_if_break_deg_never_caught_it():
    # Normally break_deg < abort_deg so "breakaway" always fires first: this
    # exercises the safety-net path for a misconfiguration (break_deg >
    # abort_deg) or any other reason the break_deg check didn't already stop
    # the ramp -- moving past abort_deg must never go unnoticed.
    r = check_breakaway(10.0, moved_deg=5.5, current_a=0.2, rate_nm_s=0.002, sign=1,
                        current_cap_a=0.8, break_deg=6.0, abort_deg=5.0, abort_s=15.0)
    assert r is not None
    assert r.status == "runaway"


def test_timeout():
    r = check_breakaway(15.0, moved_deg=0.05, current_a=0.05, rate_nm_s=0.002, sign=1,
                        current_cap_a=0.8, break_deg=2.0, abort_deg=5.0, abort_s=15.0)
    assert r is not None
    assert r.status == "timeout"


def test_negative_direction_uses_absolute_displacement():
    r = check_breakaway(3.0, moved_deg=-2.1, current_a=0.3, rate_nm_s=0.002, sign=-1,
                        current_cap_a=0.8, break_deg=2.0, abort_deg=5.0, abort_s=15.0)
    assert r.status == "breakaway"
    assert r.torque_nm < 0


# --------------------------------------------------------------------------
# sim integration: exercise the mixed torque/position wiring breakaway.py
# depends on (Set_Controller_Mode "torque" on one node while the other stays
# in "position", Set_Input_Torque, motor_currents / joint_state reads) end to
# end against the sim plant, without a real bus. The sim has no static
# friction, so a ramping torque should produce a "breakaway" (or, if the sim
# plant's own vel_gain damping resists it long enough, a "timeout" -- either
# is fine here; the point is the wiring runs cleanly and the pure decision
# function is fed real numbers from a real (simulated) control loop.
# --------------------------------------------------------------------------

def _sim_config():
    motors = (
        MotorConfig(0, flip=False, torque_constant=0.02235, current_soft_max=0.8,
                   vel_gain=0.0025, max_pos_gain=2000.0),
        MotorConfig(1, flip=True, torque_constant=0.02235, current_soft_max=0.8,
                   vel_gain=0.0025, max_pos_gain=2000.0),
    )
    return Config(motors=motors)


def test_breakaway_ramp_runs_against_sim_without_crashing():
    config = _sim_config()
    link = CanLink(config, sim=True)
    link.start()
    try:
        link.wait_for_feedback(timeout=5.0)
        for m in config.motors:
            link.set_controller_mode(m.node_id, "position")
            link.set_limits(m.node_id, 10.0, m.current_soft_max)
        q0, _ = link.joint_state()
        for i, m in enumerate(config.motors):
            link.set_input_pos(m.node_id, float(q0[i]))
        link.enter_closed_loop(timeout=5.0)

        test_node = config.motors[0].node_id
        other_node = config.motors[1].node_id
        other_q = float(q0[1])
        link.set_controller_mode(test_node, "torque")

        rate = 0.01  # N.m/s -- brisk, so the sim test finishes quickly
        t0 = time.monotonic()
        result = None
        deadline = time.monotonic() + 5.0
        while result is None and time.monotonic() < deadline:
            elapsed = time.monotonic() - t0
            tau = rate * elapsed
            link.set_input_torque(test_node, tau)
            link.set_input_pos(other_node, other_q)
            link.set_pos_gain(other_node, 200.0)
            q, _ = link.joint_state()
            cur = link.motor_currents()
            moved_deg = float(np.degrees(q[0] - q0[0]))
            result = check_breakaway(elapsed, moved_deg, float(cur[0]), rate_nm_s=rate, sign=1,
                                     current_cap_a=0.8, break_deg=2.0, abort_deg=10.0, abort_s=4.0)
            time.sleep(0.01)
        assert result is not None, "breakaway ramp against sim never terminated within 5s wall time"
        assert result.status in ("breakaway", "current_limit", "runaway", "timeout")
    finally:
        link.stop()
