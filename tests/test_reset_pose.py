import numpy as np
import pytest

from panto.config import Config, MotorConfig, TestPose
from scripts.reset_pose import (
    CONVERGED_TOL_MM, compute_duration_s, joint_limits_known, parse_target,
)


def test_compute_duration_scales_with_distance():
    assert compute_duration_s(100.0) == pytest.approx(8.0)
    assert compute_duration_s(200.0) == pytest.approx(16.0)
    assert compute_duration_s(50.0) == pytest.approx(4.0)  # 4.0s, right at the floor


def test_compute_duration_floors_at_min():
    assert compute_duration_s(1.0) == pytest.approx(4.0)
    assert compute_duration_s(0.0) == pytest.approx(4.0)


def _config_with_limits():
    cfg = Config()
    cfg.motors = (
        MotorConfig(0, q_min_rad=-1.0, q_max_rad=1.0),
        MotorConfig(1, q_min_rad=-1.0, q_max_rad=1.0),
    )
    return cfg


def _config_without_limits():
    cfg = Config()
    cfg.motors = (MotorConfig(0), MotorConfig(1))
    return cfg


def test_joint_limits_known():
    assert joint_limits_known(_config_with_limits())
    assert not joint_limits_known(_config_without_limits())


def test_parse_target_explicit_xy():
    cfg = _config_with_limits()
    target = parse_target("120,80", cfg)
    assert target == pytest.approx(np.array([0.120, 0.080]))


def test_parse_target_defaults_to_test_pose():
    cfg = _config_with_limits()
    cfg.test_pose = TestPose(tip_xy_mm=(110.0, 85.0), q_deg=(93.6, -112.3))
    target = parse_target(None, cfg)
    assert target == pytest.approx(np.array([0.110, 0.085]))


def test_parse_target_none_without_test_pose_raises():
    cfg = _config_with_limits()
    with pytest.raises(ValueError):
        parse_target(None, cfg)


def test_parse_target_bad_spec_raises():
    cfg = _config_with_limits()
    with pytest.raises(ValueError):
        parse_target("120", cfg)


def test_converged_tol_is_3mm():
    assert CONVERGED_TOL_MM == 3.0
