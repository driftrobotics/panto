import json

import pytest

from panto.config import Config, MotorConfig
from panto.presets import (
    Preset, PresetError, apply_to_config, drive_defaults, get_preset, list_presets, load_presets,
    resolve, restore_drive_defaults, save_presets,
)


def _write_presets(tmp_path, data):
    path = tmp_path / "presets.json"
    path.write_text(json.dumps(data))
    return path


def test_load_presets_seed_file_has_expected_names():
    presets = load_presets()  # default: repo-root presets.json
    for name in ("hold-0p6", "move-1p5-sched", "move-2A-sched", "step-2A-kv3"):
        assert name in presets
    assert presets["hold-0p6"].current == 0.6
    assert presets["move-2A-sched"].current == 2.0
    assert presets["move-2A-sched"].cap_min == [0.9, 0.5]


def test_resolve_uses_preset_when_no_cli_value():
    cli = {"stiffness": None, "current": None}
    out = resolve(cli, "hold-0p6")
    assert out["stiffness"] == 25
    assert out["current"] == 0.6


def test_resolve_explicit_cli_overrides_preset():
    cli = {"stiffness": None, "current": 1.23}
    out = resolve(cli, "hold-0p6")
    assert out["stiffness"] == 25
    assert out["current"] == 1.23


def test_resolve_no_preset_keeps_cli_values_only():
    cli = {"stiffness": 99.0, "current": None}
    out = resolve(cli, None)
    assert out == {"stiffness": 99.0, "current": None}


def test_resolve_unknown_preset_raises():
    with pytest.raises(PresetError):
        resolve({"stiffness": None}, "does-not-exist")


def test_resolve_field_preset_doesnt_set_stays_none():
    # step-2A-kv3 doesn't set max_pos_gain (null in presets.json)
    out = resolve({"max_pos_gain": None}, "step-2A-kv3")
    assert out["max_pos_gain"] is None


def test_add_show_round_trip(tmp_path):
    path = tmp_path / "presets.json"
    save_presets({}, path)
    presets = load_presets(path)
    assert presets == {}

    p = Preset(name="my-test", stiffness=10.0, current=0.5, notes="test", verified="2026-01-01")
    presets["my-test"] = p
    save_presets(presets, path)

    reloaded = load_presets(path)
    assert "my-test" in reloaded
    got = reloaded["my-test"]
    assert got.stiffness == 10.0
    assert got.current == 0.5
    assert got.notes == "test"
    assert got.verified == "2026-01-01"

    fetched = get_preset("my-test", path)
    assert fetched.stiffness == 10.0


def test_list_presets(tmp_path):
    path = _write_presets(tmp_path, {"a": {}, "b": {}})
    names = list_presets(path)
    assert names == ["a", "b"]


def _config():
    motors = (MotorConfig(0, vel_gain=0.0025), MotorConfig(1, vel_gain=0.0025))
    return Config(motors=motors)


def test_apply_to_config_writes_all_fields():
    cfg = _config()
    resolved = {
        "vel_gain": 0.05, "max_pos_gain": 123.0, "ff_scale": 0.7,
        "cap_min": [0.4, 0.6], "cap_slope": [1.0, 2.0],
    }
    apply_to_config(cfg, resolved)
    for i, m in enumerate(cfg.motors):
        assert m.vel_gain == 0.05
        assert m.max_pos_gain == 123.0
        assert m.ff_scale == 0.7
        assert m.cap_min_a == resolved["cap_min"][i]
        assert m.cap_vel_slope_a_per_rad_s == resolved["cap_slope"][i]


def test_apply_to_config_scalar_cap_fields_broadcast():
    cfg = _config()
    apply_to_config(cfg, {"cap_min": 0.5, "cap_slope": 0.0})
    for m in cfg.motors:
        assert m.cap_min_a == 0.5
        assert m.cap_vel_slope_a_per_rad_s == 0.0


def test_apply_to_config_leaves_none_fields_untouched():
    cfg = _config()
    before = [m.vel_gain for m in cfg.motors]
    apply_to_config(cfg, {"vel_gain": None, "max_pos_gain": None, "ff_scale": None,
                           "cap_min": None, "cap_slope": None})
    for m, v in zip(cfg.motors, before):
        assert m.vel_gain == v


def test_drive_defaults_captures_pre_override_values():
    # Regression test: drive_defaults() must be called *before*
    # apply_to_config() mutates config.motors, or "restore" just restores
    # the override back onto itself (the step_response.py/goto_pose.py bug).
    cfg = _config()
    before = {m.node_id: m.vel_gain for m in cfg.motors}
    defaults = drive_defaults(cfg)
    assert defaults == before

    apply_to_config(cfg, {"vel_gain": 0.2, "max_pos_gain": None, "ff_scale": None,
                           "cap_min": None, "cap_slope": None})
    # config now holds the override...
    for m in cfg.motors:
        assert m.vel_gain == 0.2
    # ...but the captured defaults are untouched (still the pre-override value).
    assert defaults == before


class _FakeLink:
    def __init__(self):
        self.calls = []
        self.fail_node = None

    def set_vel_gains(self, node_id, vel_gain, vel_integrator_gain=0.0):
        if node_id == self.fail_node:
            raise RuntimeError("bus error")
        self.calls.append((node_id, vel_gain))


class _FakeLog:
    def __init__(self):
        self.events = []

    def event(self, msg, level="INFO"):
        self.events.append((level, msg))


def test_restore_drive_defaults_writes_captured_values():
    cfg = _config()
    defaults = drive_defaults(cfg)
    apply_to_config(cfg, {"vel_gain": 0.2, "max_pos_gain": None, "ff_scale": None,
                           "cap_min": None, "cap_slope": None})

    link = _FakeLink()
    log = _FakeLog()
    restore_drive_defaults(link, defaults, log)

    assert set(link.calls) == {(m.node_id, 0.0025) for m in cfg.motors}
    assert all(level == "INFO" for level, _ in log.events)


def test_restore_drive_defaults_is_best_effort_on_failure():
    cfg = _config()
    defaults = drive_defaults(cfg)

    link = _FakeLink()
    link.fail_node = cfg.motors[0].node_id
    log = _FakeLog()
    restore_drive_defaults(link, defaults, log)

    # the failing node logged an error, but the other node still got restored
    assert (cfg.motors[1].node_id, 0.0025) in link.calls
    assert any(level == "ERROR" for level, _ in log.events)
