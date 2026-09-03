"""Config: template load, live-override merge, validation, JSON round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from panto.config import Config, ConfigError, MotorConfig


def test_template_loads():
    cfg = Config.load()
    assert cfg.geo.l1 == 0.125 and cfg.geo.l2 == 0.125
    assert len(cfg.motors) == 2
    assert cfg.motors[0].flip is False
    assert cfg.motors[1].flip is True          # one motor is physically flipped
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
    assert cfg.motors[0].node_id == 0 and cfg.motors[0].flip is False
    assert cfg.motors[1].zero_offset_rad == 0.5
    assert cfg.motors[1].flip is False                   # overridden from True


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
    assert Config().to_dict()["motors"] == Config.load().to_dict()["motors"]
    assert MotorConfig(1, flip=True).flip is True
