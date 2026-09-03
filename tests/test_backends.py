"""Pure backend checks — no CAN, no drives. A fake CanLink records every call.

Covers the two mappings the runtime relies on: EE stiffness → per-joint pos_gain
(position backend) and Jᵀ F with clamping (torque backend), plus force_limit →
current cap, relax(), and enter() mode selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from panto.backends.base import DEFAULT_VEL_LIMIT_TURN_S, ImpedanceCommand
from panto.backends.position import PositionBackend
from panto.backends.torque import TorqueBackend
from panto.config import Config, MotorConfig
from panto.kinematics import forward, inverse, jacobian, min_singular_value

GEO = Config().geo
TWO_PI = 2.0 * np.pi


class FakeLink:
    """Records the last value pushed to each node for every setter."""

    def __init__(self, q=(0.4, 1.1), qd=(0.0, 0.0)):
        self._q = np.asarray(q, float)
        self._qd = np.asarray(qd, float)
        self.mode: dict[int, str] = {}
        self.input_pos: dict[int, float] = {}
        self.pos_gain: dict[int, float] = {}
        self.input_torque: dict[int, float] = {}
        self.limits: dict[int, tuple[float, float]] = {}
        self.idle: list[int] = []

    def set_controller_mode(self, node_id, mode):
        self.mode[node_id] = mode

    def set_input_pos(self, node_id, q_rad):
        self.input_pos[node_id] = q_rad

    def set_pos_gain(self, node_id, gain):
        self.pos_gain[node_id] = gain

    def set_input_torque(self, node_id, tau_nm):
        self.input_torque[node_id] = tau_nm

    def set_limits(self, node_id, vel_limit, current_limit):
        self.limits[node_id] = (vel_limit, current_limit)

    def joint_state(self):
        return self._q.copy(), self._qd.copy()

    def set_idle(self, node_id):
        self.idle.append(node_id)


def make_config(**motor_kw):
    max_pos_gain = motor_kw.pop("max_pos_gain", 1e12)
    kw = dict(torque_constant=0.035, current_soft_max=1000.0, vel_gain=0.02)
    kw.update(motor_kw)
    motors = (MotorConfig(0, **kw), MotorConfig(1, **kw))
    for m in motors:
        m.max_pos_gain = max_pos_gain
    return Config(motors=motors)


def cmd(anchor, pose=None, q=(0.4, 1.1), stiffness=None, force_limit=1e9):
    anchor = np.asarray(anchor, float)
    return ImpedanceCommand(
        pose=np.asarray(pose if pose is not None else anchor, float),
        q=np.asarray(q, float),
        anchor=anchor,
        stiffness=np.eye(2) * 1000.0 if stiffness is None else np.asarray(stiffness, float),
        force_limit=force_limit,
    )


# --------------------------------------------------------------------- position


def test_position_apply_iks_the_anchor():
    cfg = make_config()
    link = FakeLink()
    be = PositionBackend(link, cfg)
    anchor = forward(np.array([0.5, 0.9]), GEO)

    be.apply(cmd(anchor))

    expected = inverse(anchor, GEO, elbow="up")
    assert link.input_pos[0] == pytest.approx(expected[0])
    assert link.input_pos[1] == pytest.approx(expected[1])


def test_position_gain_matches_jt_k_j_diagonal():
    cfg = make_config()
    link = FakeLink()
    be = PositionBackend(link, cfg)
    q = np.array([0.4, 1.1])
    K = np.array([[1200.0, 0.0], [0.0, 800.0]])

    be.apply(cmd(forward(q, GEO), q=q, stiffness=K))

    J = jacobian(q, GEO)
    k_joint = np.diag(J.T @ K @ J)
    for i in (0, 1):
        want = (k_joint[i] * TWO_PI) / cfg.motors[i].vel_gain
        assert link.pos_gain[i] == pytest.approx(want)


def test_position_gain_scales_with_commanded_stiffness():
    cfg = make_config()
    q = np.array([0.4, 1.1])
    anchor = forward(q, GEO)

    soft = FakeLink()
    PositionBackend(soft, cfg).apply(cmd(anchor, q=q, stiffness=np.eye(2) * 500.0))
    stiff = FakeLink()
    PositionBackend(stiff, cfg).apply(cmd(anchor, q=q, stiffness=np.eye(2) * 1000.0))

    for i in (0, 1):
        assert stiff.pos_gain[i] == pytest.approx(2.0 * soft.pos_gain[i])
        assert soft.pos_gain[i] > 0.0


def test_position_gain_clamped_to_max_pos_gain():
    cfg = make_config(max_pos_gain=5.0)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         stiffness=np.eye(2) * 1e6))
    assert link.pos_gain[0] == 5.0
    assert link.pos_gain[1] == 5.0


def test_position_force_limit_maps_to_current_cap():
    cfg = make_config()
    link = FakeLink()
    be = PositionBackend(link, cfg)
    q = np.array([0.4, 1.1])
    force_limit = 0.4

    be.apply(cmd(forward(q, GEO), q=q, force_limit=force_limit))

    sigma = max(min_singular_value(q, GEO), cfg.sigma_min_threshold)
    want = (force_limit / sigma) / cfg.motors[0].torque_constant
    for i in (0, 1):
        vel_limit, current_limit = link.limits[i]
        assert vel_limit == DEFAULT_VEL_LIMIT_TURN_S
        assert current_limit == pytest.approx(want)
        assert current_limit < cfg.motors[i].current_soft_max


def test_position_current_cap_clamped_to_soft_max():
    cfg = make_config(current_soft_max=0.6)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         force_limit=1e6))
    assert link.limits[0][1] == 0.6
    assert link.limits[1][1] == 0.6


def test_position_relax_parks_on_current_angle_at_zero_gain():
    cfg = make_config()
    link = FakeLink(q=(0.33, 1.22))
    PositionBackend(link, cfg).relax()
    assert link.input_pos[0] == pytest.approx(0.33)
    assert link.input_pos[1] == pytest.approx(1.22)
    assert link.pos_gain[0] == 0.0
    assert link.pos_gain[1] == 0.0


def test_position_enter_sets_position_mode():
    cfg = make_config()
    link = FakeLink()
    PositionBackend(link, cfg).enter()
    assert link.mode == {0: "position", 1: "position"}
    assert link.limits[0][1] == cfg.motors[0].current_soft_max


# ----------------------------------------------------------------------- torque


def test_torque_apply_produces_jt_f():
    cfg = make_config()
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    q = np.array([0.4, 1.1])
    pose = forward(q, GEO)
    anchor = pose + np.array([0.002, -0.001])
    K = np.array([[900.0, 100.0], [100.0, 600.0]])

    be.apply(cmd(anchor, pose=pose, q=q, stiffness=K))

    F = K @ (anchor - pose)
    tau = jacobian(q, GEO).T @ F
    assert link.input_torque[0] == pytest.approx(tau[0])
    assert link.input_torque[1] == pytest.approx(tau[1])


def test_torque_clamps_force_magnitude():
    cfg = make_config()
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    q = np.array([0.4, 1.1])
    pose = forward(q, GEO)
    anchor = pose + np.array([0.05, 0.05])       # huge displacement
    K = np.eye(2) * 5000.0
    force_limit = 1.5

    be.apply(cmd(anchor, pose=pose, q=q, stiffness=K, force_limit=force_limit))

    F_raw = K @ (anchor - pose)
    F_clamped = F_raw * (force_limit / np.linalg.norm(F_raw))
    tau = jacobian(q, GEO).T @ F_clamped
    assert np.linalg.norm(F_clamped) == pytest.approx(force_limit)
    assert link.input_torque[0] == pytest.approx(tau[0])
    assert link.input_torque[1] == pytest.approx(tau[1])


def test_torque_relax_zeros_torque():
    cfg = make_config()
    link = FakeLink()
    TorqueBackend(link, cfg).relax()
    assert link.input_torque == {0: 0.0, 1: 0.0}


def test_torque_enter_sets_torque_mode():
    cfg = make_config()
    link = FakeLink()
    TorqueBackend(link, cfg).enter()
    assert link.mode == {0: "torque", 1: "torque"}
