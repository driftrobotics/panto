"""Config: template load, live-override merge, validation, JSON round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panto.config import Config, ConfigError, MotorConfig

# The checked-in template, loaded explicitly and hermetically -- Config.load()
# with no path also picks up a live calibration.json/config.local.json from
# CWD or the repo root if one exists (by design, for the hand-cal workflow;
# see panto/config.py). On rig-host that file is real and its
# limit_margin_rad/etc. legitimately differ in the last few digits from the
# template's rounded defaults, so any test asserting *template* values must
# load the template explicitly rather than relying on the ambient CWD having
# no calibration.json.
_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "panto" / "config.template.json"


def test_template_loads():
    cfg = Config.load()
    assert cfg.geo.l1 == 0.125 and cfg.geo.l2 == 0.125
    assert len(cfg.motors) == 2
    assert cfg.motors[0].flip is True           # 2026-09-04 hardware cal: shoulder flipped
    assert cfg.motors[1].flip is False          # one motor is physically flipped
    assert cfg.thermal.budget_a2s == 4.0
    assert cfg.can.bitrate == 1_000_000
    assert cfg.control.rate_hz == 200.0


def test_flat_aliases_survive():
    cfg = Config.load()
    assert cfg.control_rate_hz == cfg.control.rate_hz == 200.0
    assert cfg.latency_compensation_s == 0.0
    assert cfg.can_interface == "socketcan"
    z = cfg.zero_offset_rad
    assert isinstance(z, np.ndarray) and z.shape == (2,)
    cfg.control_rate_hz = 150.0
    assert cfg.control.rate_hz == 150.0


def test_live_override_merges_per_motor(tmp_path):
    override = {
        "control": {"rate_hz": 100.0},
        "motors": [{}, {"zero_offset_rad": 0.5, "flip": False}],
    }
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(override))

    cfg = Config.load(p)
    assert cfg.control_rate_hz == 100.0
    assert cfg.control.latency_compensation_s == 0.0     # untouched by override
    assert cfg.motors[0].node_id == 0 and cfg.motors[0].flip is True  # untouched by override
    assert cfg.motors[1].zero_offset_rad == 0.5
    assert cfg.motors[1].flip is False                   # overridden from True


def test_test_pose_unset_by_default():
    cfg = Config.load(_TEMPLATE_PATH)
    assert cfg.test_pose is None
    assert cfg.test_pose_xy_m is None
    assert cfg.test_pose_q_rad is None


def test_test_pose_loaded_from_calibration_json(tmp_path):
    override = {"test_pose": {"tip_xy_mm": [110.5, 84.7], "q_deg": [93.6, -112.3]}}
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(override))

    cfg = Config.load(p)
    assert cfg.test_pose is not None
    assert cfg.test_pose.tip_xy_mm == (110.5, 84.7)
    assert cfg.test_pose.q_deg == (93.6, -112.3)
    assert np.allclose(cfg.test_pose_xy_m, [0.1105, 0.0847])
    assert np.allclose(cfg.test_pose_q_rad, np.radians([93.6, -112.3]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["geo"].__setitem__("l1", -1.0),
        lambda d: d.__setitem__("elbow", "sideways"),
        lambda d: d["control"].__setitem__("rate_hz", 0.0),
        lambda d: d.__setitem__("sigma_min_threshold", -0.01),
        lambda d: d["motors"][0].__setitem__("current_soft_max", 0.0),
        lambda d: d.__setitem__("workspace_polygon", [[0.0, 0.0], [1.0, 0.0]]),
    ],
)
def test_validation_rejects_bad_values(mutate):
    d = json.loads(Config.load().to_json())
    mutate(d)
    with pytest.raises(ConfigError):
        Config.from_dict(d)


def test_json_round_trip(tmp_path):
    cfg = Config.load()
    p = tmp_path / "config.local.json"
    p.write_text(cfg.to_json())
    again = Config.load(p)
    assert again.to_dict() == cfg.to_dict()


def test_round_trip_with_polygon(tmp_path):
    d = json.loads(Config.load().to_json())
    d["workspace_polygon"] = [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]
    cfg = Config.from_dict(d)
    p = tmp_path / "calibration.json"
    p.write_text(cfg.to_json())
    again = Config.load(p)
    assert np.allclose(again.workspace_polygon, cfg.workspace_polygon)


def test_default_construction_matches_template():
    assert Config().to_dict()["motors"] == Config.load(_TEMPLATE_PATH).to_dict()["motors"]
    assert MotorConfig(1, flip=True).flip is True


def test_template_hardware_calibrated_values():
    cfg = Config.load()
    assert cfg.motors[0].flip is True
    assert cfg.motors[1].flip is False
    for m in cfg.motors:
        assert m.torque_constant == pytest.approx(0.02235)


def test_motor_config_limit_defaults_unknown():
    m = MotorConfig(0)
    assert m.q_min_rad == float("-inf")
    assert m.q_max_rad == float("inf")
    assert m.limit_margin_rad == pytest.approx(0.087)


def test_template_leaves_limits_unset():
    cfg = Config.load(_TEMPLATE_PATH)
    for m in cfg.motors:
        assert m.q_min_rad == float("-inf")
        assert m.q_max_rad == float("inf")


def test_config_limit_arrays():
    cfg = Config.load(_TEMPLATE_PATH)
    cfg.motors[0].q_min_rad, cfg.motors[0].q_max_rad = -1.0, 1.0
    cfg.motors[1].q_min_rad, cfg.motors[1].q_max_rad = -2.0, 2.0
    assert np.array_equal(cfg.q_min_rad, np.array([-1.0, -2.0]))
    assert np.array_equal(cfg.q_max_rad, np.array([1.0, 2.0]))
    assert np.allclose(cfg.limit_margin_rad, [0.087, 0.087])


def test_validate_rejects_bad_limit_ordering():
    d = json.loads(Config.load().to_json())
    d["motors"][0]["q_min_rad"] = 1.0
    d["motors"][0]["q_max_rad"] = -1.0
    with pytest.raises(ConfigError):
        Config.from_dict(d)
