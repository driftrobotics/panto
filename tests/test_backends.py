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
        self.torque_ff: dict[int, float] = {}
        self.pos_gain: dict[int, float] = {}
        self.input_torque: dict[int, float] = {}
        self.limits: dict[int, tuple[float, float]] = {}
        self.idle: list[int] = []
        self.vel_gains: dict[int, tuple[float, float]] = {}

    def set_controller_mode(self, node_id, mode):
        self.mode[node_id] = mode

    def set_vel_gains(self, node_id, vel_gain, vel_integrator_gain=0.0):
        self.vel_gains[node_id] = (vel_gain, vel_integrator_gain)

    def set_input_pos(self, node_id, q_rad, torque_ff_nm=0.0):
        self.input_pos[node_id] = q_rad
        self.torque_ff[node_id] = torque_ff_nm

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


def cmd(anchor, pose=None, q=(0.4, 1.1), qd=None, stiffness=None, force_limit=1e9):
    anchor = np.asarray(anchor, float)
    return ImpedanceCommand(
        pose=np.asarray(pose if pose is not None else anchor, float),
        q=np.asarray(q, float),
        qd=np.asarray(qd, float) if qd is not None else None,
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


def test_position_gain_uses_resolved_vel_gain_not_config_default():
    # Regression for the step_response.py bug: a script resolves an override
    # vel_gain (e.g. via panto.presets.resolve/apply_to_config) and pushes it
    # to the drive with link.set_vel_gains, but PositionBackend must derive
    # pos_gain from that *same* resolved vel_gain -- not silently fall back
    # to MotorConfig's 0.0025 default just because the config wasn't updated.
    from panto.presets import apply_to_config

    cfg = make_config(vel_gain=0.0025, max_pos_gain=1e12)
    resolved_vel_gain = 0.05
    apply_to_config(cfg, {"vel_gain": resolved_vel_gain})

    link = FakeLink()
    q = np.array([0.4, 1.1])
    K = 10.0 * np.eye(2)  # isotropic EE stiffness, N/m
    PositionBackend(link, cfg).apply(cmd(forward(q, GEO), q=q, stiffness=K))

    J = jacobian(q, GEO)
    k_joint = np.diag(J.T @ K @ J)
    wrong_pos_gain = [(k_joint[i] * TWO_PI) / 0.0025 for i in (0, 1)]
    for i in (0, 1):
        want = (k_joint[i] * TWO_PI) / resolved_vel_gain
        assert link.pos_gain[i] == pytest.approx(want)
        assert link.pos_gain[i] != pytest.approx(wrong_pos_gain[i])


def test_pos_gains_for_matches_apply():
    # PositionBackend.pos_gains_for() is the pure function scripts/*.py call
    # for pre-flight visibility -- it must equal what apply() actually sends
    # for the same q/K, not the naive (non-Jacobian-scaled) approximation the
    # old approx_pos_gain prints used.
    cfg = make_config()
    link = FakeLink()
    be = PositionBackend(link, cfg)
    q = np.array([0.4, 1.1])
    K = np.array([[1200.0, 0.0], [0.0, 800.0]])

    be.apply(cmd(forward(q, GEO), q=q, stiffness=K))
    predicted = PositionBackend.pos_gains_for(K, q, cfg)

    for i in (0, 1):
        assert predicted[i] == pytest.approx(link.pos_gain[i])


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


# ---------------------------------------------------- velocity-scheduled cap


def test_position_vel_schedule_off_by_default():
    """cap_vel_slope_a_per_rad_s defaults to 0 -> constant current_soft_max
    regardless of qd, i.e. today's (pre-2026-09-04-fix) behaviour."""
    cfg = make_config(current_soft_max=2.0)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         qd=(5.0, -5.0), force_limit=1e6))
    assert link.limits[0][1] == pytest.approx(2.0)
    assert link.limits[1][1] == pytest.approx(2.0)


def test_position_vel_schedule_falls_with_speed():
    cfg = make_config(current_soft_max=2.0, cap_vel_slope_a_per_rad_s=3.0, cap_min_a=0.5)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         qd=(0.5, 0.0), force_limit=1e6))
    # cap_max - k_v*|qd| = 2.0 - 3.0*0.5 = 0.5, quantised to 0.02A steps
    assert link.limits[0][1] == pytest.approx(0.5, abs=0.02)
    # qd=0 -> no reduction, full cap_max
    assert link.limits[1][1] == pytest.approx(2.0, abs=0.02)


def test_position_vel_schedule_floors_at_cap_min():
    cfg = make_config(current_soft_max=2.0, cap_vel_slope_a_per_rad_s=3.0, cap_min_a=0.5)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         qd=(50.0, 50.0), force_limit=1e6))
    assert link.limits[0][1] == pytest.approx(0.5, abs=0.02)
    assert link.limits[1][1] == pytest.approx(0.5, abs=0.02)


def test_position_vel_schedule_uses_speed_magnitude_sign_independent():
    cfg = make_config(current_soft_max=2.0, cap_vel_slope_a_per_rad_s=3.0, cap_min_a=0.5)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         qd=(-0.5, 0.5), force_limit=1e6))
    assert link.limits[0][1] == pytest.approx(0.5, abs=0.02)
    assert link.limits[1][1] == pytest.approx(0.5, abs=0.02)


def test_position_vel_schedule_no_qd_treated_as_zero():
    """cmd.qd is None (a caller that hasn't been plumbed for it yet) -> the
    schedule treats |qd| as 0, i.e. full cap_max, not an error."""
    cfg = make_config(current_soft_max=2.0, cap_vel_slope_a_per_rad_s=3.0, cap_min_a=0.5)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO), force_limit=1e6))
    assert link.limits[0][1] == pytest.approx(2.0, abs=0.02)


def test_position_vel_schedule_combines_with_force_limit_cap_takes_min():
    """The final cap is min(force-limit-derived cap, velocity-scheduled cap)
    -- whichever constraint is tighter wins."""
    cfg = make_config(current_soft_max=2.0, cap_vel_slope_a_per_rad_s=3.0, cap_min_a=0.5,
                      torque_constant=0.035)
    link = FakeLink()
    # a small force_limit drives the force-derived cap well below the
    # velocity-scheduled cap (qd=0 -> vel cap = current_soft_max = 2.0A)
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         qd=(0.0, 0.0), force_limit=0.0005))
    assert link.limits[0][1] < 2.0


def test_position_vel_schedule_slope_is_per_motor():
    """cap_vel_slope_a_per_rad_s is a per-motor field, like cap_min_a -- a
    shoulder/elbow pair with different slopes (including 0 on one joint)
    schedules independently rather than sharing one value."""
    motors = (
        MotorConfig(0, torque_constant=0.035, current_soft_max=2.0, vel_gain=0.02,
                    cap_vel_slope_a_per_rad_s=3.0, cap_min_a=0.5),
        MotorConfig(1, torque_constant=0.035, current_soft_max=2.0, vel_gain=0.02,
                    cap_vel_slope_a_per_rad_s=0.0, cap_min_a=0.5),
    )
    for m in motors:
        m.max_pos_gain = 1e12
    cfg = Config(motors=motors)
    link = FakeLink()
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO),
                                         qd=(0.5, 0.5), force_limit=1e6))
    # joint 0: slope 3.0 -> cap falls to 2.0 - 3.0*0.5 = 0.5
    assert link.limits[0][1] == pytest.approx(0.5, abs=0.02)
    # joint 1: slope 0.0 -> constant cap_max regardless of qd
    assert link.limits[1][1] == pytest.approx(2.0, abs=0.02)


# ---------------------------------------------------- Coulomb friction feedforward


def test_position_ff_disabled_by_default():
    """coulomb_pos_nm/coulomb_neg_nm default to 0 -> tau_ff always 0 even
    with a large q error, i.e. byte-identical to pre-feedforward behaviour."""
    cfg = make_config()
    link = FakeLink(q=(0.0, 1.1))
    PositionBackend(link, cfg).apply(cmd(forward(np.array([0.4, 1.1]), GEO), q=(0.0, 1.1)))
    assert link.torque_ff[0] == 0.0


def test_position_ff_sign_matches_move_direction():
    cfg = make_config(coulomb_pos_nm=0.03, coulomb_neg_nm=0.02, ff_scale=1.0)
    link = FakeLink(q=(0.0, 1.1))
    q = np.array([0.0, 1.1])
    # target well above q[0] -> moving in +q0 direction -> +coulomb_pos_nm
    target_up = forward(np.array([0.4, 1.1]), GEO)
    PositionBackend(link, cfg).apply(cmd(target_up, q=q))
    assert link.torque_ff[0] == pytest.approx(0.03)

    # target well below q[0] -> moving in -q0 direction -> -coulomb_neg_nm
    target_down = forward(np.array([-0.4, 1.1]), GEO)
    PositionBackend(link, cfg).apply(cmd(target_down, q=q))
    assert link.torque_ff[0] == pytest.approx(-0.02)


def test_position_ff_scale_attenuates():
    cfg = make_config(coulomb_pos_nm=0.03, coulomb_neg_nm=0.02, ff_scale=0.5)
    link = FakeLink(q=(0.0, 1.1))
    q = np.array([0.0, 1.1])
    target_up = forward(np.array([0.4, 1.1]), GEO)
    PositionBackend(link, cfg).apply(cmd(target_up, q=q))
    assert link.torque_ff[0] == pytest.approx(0.015)


def test_position_ff_zero_inside_deadband():
    cfg = make_config(coulomb_pos_nm=0.03, coulomb_neg_nm=0.02, ff_scale=1.0)
    link = FakeLink(q=(0.4, 1.1))
    q = np.array([0.4, 1.1])
    # target is q itself -> zero error -> inside deadband -> tau_ff = 0
    PositionBackend(link, cfg).apply(cmd(forward(q, GEO), q=q))
    assert link.torque_ff[0] == 0.0
    assert link.torque_ff[1] == 0.0


def test_position_relax_parks_on_current_angle_at_zero_gain():
    cfg = make_config()
    link = FakeLink(q=(0.33, 1.22))
    PositionBackend(link, cfg).relax()
    assert link.input_pos[0] == pytest.approx(0.33)
    assert link.input_pos[1] == pytest.approx(1.22)
    assert link.pos_gain[0] == 0.0
    assert link.pos_gain[1] == 0.0


def test_position_enter_selects_elbow_branch_from_measured_q1_sign():
    # q1 < 0 at enter() -> "down" branch must be selected and stick for the
    # session, so apply()'s IK solves toward the physically-calibrated arm
    # rather than defaulting to config.elbow (which could be the mirror).
    cfg = make_config()
    assert cfg.elbow == "up"  # default config says "up" -- must be overridden
    link = FakeLink(q=(0.4, -1.1))
    be = PositionBackend(link, cfg)
    be.enter()
    assert be.elbow == "down"

    # a target near the start pose must IK to something close to the measured
    # q, not the mirror-image "up" branch solution.
    start_q = np.array([0.4, -1.1])
    near_pose = forward(start_q, GEO)
    be.apply(cmd(near_pose, q=start_q))
    solved = np.array([link.input_pos[0], link.input_pos[1]])
    assert np.degrees(np.abs(solved - start_q)).max() < 5.0


def test_position_enter_selects_up_branch_for_positive_q1():
    cfg = make_config()
    link = FakeLink(q=(0.4, 1.1))
    be = PositionBackend(link, cfg)
    be.enter()
    assert be.elbow == "up"


def test_position_enter_sets_position_mode():
    cfg = make_config()
    link = FakeLink()
    PositionBackend(link, cfg).enter()
    assert link.mode == {0: "position", 1: "position"}
    assert link.limits[0][1] == cfg.motors[0].current_soft_max


# ------------------------------------------------------------------ joint limits

def test_position_apply_clamps_ik_target_to_limits():
    cfg = make_config()
    for m in cfg.motors:
        m.q_min_rad, m.q_max_rad = 0.0, 1.0
        m.limit_margin_rad = 0.1
    link = FakeLink(q=(0.5, 0.5))
    be = PositionBackend(link, cfg)
    # anchor whose IK solution is q=(1.5, 1.5) -- well past the upper limit
    far_anchor = forward(np.array([1.5, 1.5]), GEO)

    be.apply(cmd(far_anchor, q=(0.5, 0.5)))

    for i in (0, 1):
        assert link.input_pos[i] == pytest.approx(cfg.motors[i].q_max_rad
                                                   - cfg.motors[i].limit_margin_rad)


def test_position_apply_leaves_in_range_target_untouched():
    cfg = make_config()
    for m in cfg.motors:
        m.q_min_rad, m.q_max_rad = 0.0, 2.0
        m.limit_margin_rad = 0.1
    link = FakeLink(q=(0.5, 0.9))
    be = PositionBackend(link, cfg)
    q = np.array([0.5, 0.9])

    be.apply(cmd(forward(q, GEO), q=q))

    expected = inverse(forward(q, GEO), GEO, elbow="up")
    assert link.input_pos[0] == pytest.approx(expected[0])
    assert link.input_pos[1] == pytest.approx(expected[1])


def test_position_apply_raises_when_measured_q_near_limit():
    from panto.limits import JointLimitViolation

    cfg = make_config()
    cfg.motors[0].q_min_rad, cfg.motors[0].q_max_rad = 0.0, 1.0
    cfg.motors[0].limit_margin_rad = 0.2  # half-margin trip wire at 0.1 rad
    link = FakeLink(q=(0.05, 0.9))  # joint 0 is 0.05 rad from q_min -- inside half-margin
    be = PositionBackend(link, cfg)

    with pytest.raises(JointLimitViolation):
        be.apply(cmd(forward(np.array([0.5, 0.9]), GEO), q=(0.05, 0.9)))


def test_position_apply_does_not_raise_when_no_limits_configured():
    cfg = make_config()  # defaults: q_min/q_max unset (+-inf)
    link = FakeLink(q=(0.05, 0.9))
    be = PositionBackend(link, cfg)
    be.apply(cmd(forward(np.array([0.5, 0.9]), GEO), q=(0.05, 0.9)))  # no raise


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


def test_torque_enter_raises_vel_gain_and_sizes_plateau():
    """enter() must push torque_vel_gain and a vel_limit sized so the
    torque-mode plateau (margin * current_cap) sits above the current cap --
    see panto.breakaway_logic.plateau_vel_limit_rad_s. The 2026-09-04 bug was
    a fixed vel_limit that silently re-capped Iq below the requested current."""
    from panto.breakaway_logic import plateau_vel_limit_rad_s

    cfg = make_config(torque_constant=0.02235, current_soft_max=0.8, vel_gain=0.0025)
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    be.torque_vel_gain = 0.05
    be.enter()

    assert link.vel_gains[0] == (0.05, 0.0)
    assert link.vel_gains[1] == (0.05, 0.0)
    expected_vel_limit, plateau_a = plateau_vel_limit_rad_s(0.8, 0.02235, 0.05)
    assert link.limits[0][0] == pytest.approx(expected_vel_limit)
    assert link.limits[0][1] == pytest.approx(0.8)
    assert plateau_a > 0.8  # plateau above the cap -- the cap does the limiting


def test_torque_torque_cap_from_current_and_kt():
    """A huge commanded F must be clamped per-joint to current_soft_max *
    torque_constant, independent of the Cartesian force_limit."""
    cfg = make_config(torque_constant=0.02235, current_soft_max=0.5)
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    q = np.array([0.4, 1.1])
    pose = forward(q, GEO)
    anchor = pose + np.array([0.05, 0.05])
    K = np.eye(2) * 5000.0

    # generous Cartesian force_limit so the joint torque cap is the binding one
    be.apply(cmd(anchor, pose=pose, q=q, stiffness=K, force_limit=1e6))

    cap_nm = 0.5 * 0.02235
    assert abs(link.input_torque[0]) <= cap_nm + 1e-9
    assert abs(link.input_torque[1]) <= cap_nm + 1e-9
    # and at least one joint is actually pinned at the cap given how large K is
    assert max(abs(link.input_torque[0]), abs(link.input_torque[1])) == pytest.approx(cap_nm)


def test_torque_damping_opposes_velocity():
    """With K=0 and damping>0, tau must be -Jᵀ(B v_ee) -- pure damping, no
    spring term. dt=0 on the very first apply() means the qd filters pass
    qd through unfiltered (see panto.filters), so the check is exact."""
    cfg = make_config(current_soft_max=1000.0)
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    be.damping = 2.0
    q = np.array([0.4, 1.1])
    pose = forward(q, GEO)
    qd = np.array([0.1, -0.05])

    be.apply(cmd(pose, pose=pose, q=q, qd=qd, stiffness=np.zeros((2, 2)), force_limit=1e9))

    J = jacobian(q, GEO)
    v_ee = J @ qd
    F = -2.0 * v_ee
    tau = J.T @ F
    assert link.input_torque[0] == pytest.approx(tau[0])
    assert link.input_torque[1] == pytest.approx(tau[1])


def test_torque_slew_limits_step_change(monkeypatch):
    """A slew-limited backend must not jump the commanded torque by more
    than slew_nm_s * dt between two ticks."""
    cfg = make_config(current_soft_max=1000.0)
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    be.slew_nm_s = 0.01  # N.m/s, deliberately tiny
    q = np.array([0.4, 1.1])
    pose = forward(q, GEO)
    anchor = pose + np.array([0.01, 0.0])
    K = np.eye(2) * 5000.0

    # Pin the clock so dt between ticks is exactly 0.05s -- using the real
    # wall clock here makes the test's tolerance flaky on slower/loaded
    # machines, since apply() itself takes non-zero wall time.
    fake_times = iter([100.0, 100.05])
    monkeypatch.setattr(
        "panto.backends.torque.time.monotonic", lambda: next(fake_times)
    )

    be.apply(cmd(pose, pose=pose, q=q, stiffness=K, force_limit=1e9))  # tick 1: tau ~ 0
    first = dict(link.input_torque)
    be.apply(cmd(anchor, pose=pose, q=q, stiffness=K, force_limit=1e9))  # tick 2: big target jump
    second = dict(link.input_torque)

    max_step = be.slew_nm_s * 0.05 + 1e-9
    for node_id in (0, 1):
        assert abs(second[node_id] - first[node_id]) <= max_step


def test_torque_relax_resets_slew_state():
    cfg = make_config()
    link = FakeLink()
    be = TorqueBackend(link, cfg)
    be._last_tau = np.array([1.0, -1.0])
    be.relax()
    assert list(be._last_tau) == [0.0, 0.0]


def test_position_hold_ff_off_by_default_and_linear_when_on():
    cfg = make_config()
    from panto.backends.position import PositionBackend
    m = cfg.motors[0]
    assert PositionBackend._hold_ff(m, np.array([0.5, -1.0])) == 0.0
    m.hold_ff_const_a = -0.5
    m.hold_ff_per_deg = (0.03, 0.02)
    m.hold_ff_scale = 1.0
    q = np.array([np.radians(80.0), np.radians(-100.0)])
    expect = m.torque_constant * (-0.5 + 0.03 * 80.0 + 0.02 * -100.0)
    assert abs(PositionBackend._hold_ff(m, q) - expect) < 1e-12
    m.hold_ff_scale = 0.5
    assert abs(PositionBackend._hold_ff(m, q) - 0.5 * expect) < 1e-12


def test_position_apply_joint_sends_given_q_without_ik_and_shares_gain_path():
    from panto.backends.position import PositionBackend
    from panto.kinematics import forward
    cfg = make_config()
    link = FakeLink(q=(0.4, -1.1))
    be = PositionBackend(link, cfg)
    be.enter()
    q_t = np.array([0.5, -1.0])
    c = cmd(anchor=forward(q_t, cfg.geo), q=(0.4, -1.1), stiffness=25.0 * np.eye(2))
    be.apply_joint(q_t, c)
    assert abs(link.input_pos[0] - 0.5) < 1e-12 and abs(link.input_pos[1] + 1.0) < 1e-12
    assert link.pos_gain[0] > 0 and link.pos_gain[1] > 0
    assert be.last_command["q_target"] == [0.5, -1.0]
    # Cartesian apply() on the same anchor lands on the same joints (same branch)
    be.apply(c)
    assert np.allclose([link.input_pos[0], link.input_pos[1]], q_t, atol=1e-6)


def test_position_apply_joint_clamps_to_joint_limits():
    from panto.backends.position import PositionBackend
    from panto.kinematics import forward
    cfg = make_config(q_min_rad=-1.0, q_max_rad=1.0, limit_margin_rad=0.05)
    link = FakeLink(q=(0.4, -0.5))
    be = PositionBackend(link, cfg)
    be.enter()
    q_t = np.array([1.5, -0.5])
    be.apply_joint(q_t, cmd(anchor=forward(q_t, cfg.geo), q=(0.4, -0.5), stiffness=25.0 * np.eye(2)))
    assert link.input_pos[0] <= 1.0
