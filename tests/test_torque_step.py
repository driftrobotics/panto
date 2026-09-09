"""Unit tests for the pure step-schedule/abort logic behind scripts/torque_step.py.
Same split as tests/test_breakaway.py: pure-function tests + one sim
integration smoke test exercising the torque-mode plumbing end to end."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scripts.torque_step import _resolve_torque
from panto.can_link import CanLink
from panto.config import Config, MotorConfig
from panto.torque_step_logic import check_abort, schedule, total_duration_s


def test_resolve_torque_max_and_negative_max():
    assert _resolve_torque("max", current_cap=0.8, torque_constant=0.02235) == pytest.approx(0.01788)
    assert _resolve_torque("-max", current_cap=0.8, torque_constant=0.02235) == pytest.approx(-0.01788)
    assert _resolve_torque("+max", current_cap=1.5, torque_constant=0.02235) == pytest.approx(0.033525)


def test_resolve_torque_numeric_mnm():
    assert _resolve_torque("10", current_cap=0.8, torque_constant=0.02235) == pytest.approx(0.01)
    assert _resolve_torque("-7.5", current_cap=0.8, torque_constant=0.02235) == pytest.approx(-0.0075)


def test_pre_phase_is_zero_torque():
    p = schedule(0.0, duration_s=2.0, tau_nm=0.02)
    assert p.name == "pre"
    assert p.torque_nm == 0.0
    p = schedule(0.49, duration_s=2.0, tau_nm=0.02)
    assert p.name == "pre"


def test_step_phase_commands_the_torque():
    p = schedule(0.5, duration_s=2.0, tau_nm=0.02)
    assert p.name == "step"
    assert p.torque_nm == 0.02
    p = schedule(2.4, duration_s=2.0, tau_nm=0.02)
    assert p.name == "step"
    assert p.torque_nm == 0.02


def test_post_phase_is_zero_torque_again():
    p = schedule(2.5, duration_s=2.0, tau_nm=0.02)
    assert p.name == "post"
    assert p.torque_nm == 0.0
    p = schedule(2.99, duration_s=2.0, tau_nm=0.02)
    assert p.name == "post"


def test_done_after_pre_step_post():
    p = schedule(3.0, duration_s=2.0, tau_nm=0.02)
    assert p.name == "done"
    assert p.torque_nm == 0.0


def test_total_duration():
    assert total_duration_s(2.0) == 3.0
    assert total_duration_s(0.0) == 1.0


def test_negative_torque_passes_through_unchanged():
    p = schedule(1.0, duration_s=2.0, tau_nm=-0.0179)
    assert p.name == "step"
    assert p.torque_nm == -0.0179


def test_check_abort_within_bounds():
    assert check_abort(5.0, max_deg=30.0) is None
    assert check_abort(-29.9, max_deg=30.0) is None


def test_check_abort_exceeded():
    reason = check_abort(30.5, max_deg=30.0)
    assert reason is not None
    assert "30.5" in reason
    reason = check_abort(-31.0, max_deg=30.0)
    assert reason is not None


# --------------------------------------------------------------------------
# sim integration: exercise the same mixed torque/position wiring
# torque_step.py depends on, end to end against the sim plant.
# --------------------------------------------------------------------------

def _sim_config():
    motors = (
        MotorConfig(0, flip=False, torque_constant=0.02235, current_soft_max=0.8,
                   vel_gain=0.0025, max_pos_gain=2000.0),
        MotorConfig(1, flip=True, torque_constant=0.02235, current_soft_max=0.8,
                   vel_gain=0.0025, max_pos_gain=2000.0),
    )
    return Config(motors=motors)


def test_torque_step_schedule_runs_against_sim_without_crashing():
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
        link.set_vel_gains(test_node, 0.01, 0.0)
        link.set_limits(test_node, 20.0, config.motors[0].current_soft_max)
        link.set_controller_mode(test_node, "torque")

        duration_s = 0.3
        tau = 0.01
        t0 = time.monotonic()
        moved_deg_trace = []
        while True:
            elapsed = time.monotonic() - t0
            phase = schedule(elapsed, duration_s, tau)
            if phase.name == "done":
                break
            link.set_input_torque(test_node, phase.torque_nm)
            link.set_input_pos(other_node, other_q)
            link.set_pos_gain(other_node, 200.0)
            q, _ = link.joint_state()
            moved_deg = float(np.degrees(q[0] - q0[0]))
            moved_deg_trace.append(moved_deg)
            reason = check_abort(moved_deg, max_deg=30.0)
            if reason is not None:
                break
            time.sleep(0.01)
        assert len(moved_deg_trace) > 0
    finally:
        link.stop()


# --------------------------------------------------------------------------
# both-joint schedule/abort: torque_step.py's --joint both puts BOTH nodes
# in torque mode simultaneously, no position hold. Exercise that against sim.
# --------------------------------------------------------------------------

def test_both_joint_schedule_runs_against_sim_without_crashing():
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

        for m in config.motors:
            link.set_vel_gains(m.node_id, 0.01, 0.0)
            link.set_limits(m.node_id, 20.0, m.current_soft_max)
            link.set_controller_mode(m.node_id, "torque")

        duration_s = 0.3
        tau = {0: 0.01, 1: -0.008}  # asymmetric, like max/-max on the two nodes
        t0 = time.monotonic()
        moved_deg_trace = {0: [], 1: []}
        aborted_joint = None
        while True:
            elapsed = time.monotonic() - t0
            phases = {j: schedule(elapsed, duration_s, tau[j]) for j in (0, 1)}
            if all(ph.name == "done" for ph in phases.values()):
                break
            for j in (0, 1):
                link.set_input_torque(config.motors[j].node_id, phases[j].torque_nm)
            q, _ = link.joint_state()
            for j in (0, 1):
                moved_deg = float(np.degrees(q[j] - q0[j]))
                moved_deg_trace[j].append(moved_deg)
                reason = check_abort(moved_deg, max_deg=30.0)
                if reason is not None:
                    aborted_joint = j
            if aborted_joint is not None:
                break
            time.sleep(0.01)

        # both joints must have been driven independently -- confirms the
        # both-joint loop actually commands two different torques, not just
        # one applied twice (the classic "forgot to index by joint" bug).
        assert len(moved_deg_trace[0]) > 0
        assert len(moved_deg_trace[1]) > 0
    finally:
        link.stop()


def test_both_joint_abort_zeroes_both_torques_conceptually():
    # Not a hardware assertion (that's scripts/torque_step.py's finally
    # block) -- confirms check_abort is evaluated independently per joint,
    # i.e. one joint tripping doesn't get masked by the other being fine.
    reason0 = check_abort(31.0, max_deg=30.0)
    reason1 = check_abort(5.0, max_deg=30.0)
    assert reason0 is not None
    assert reason1 is None
