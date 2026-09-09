import json

from scripts import presets as presets_cli


def _make_log_dir(tmp_path, meta: dict):
    log_dir = tmp_path / "logs" / "step_response-fake"
    log_dir.mkdir(parents=True)
    (log_dir / "meta.json").write_text(json.dumps(meta))
    return log_dir


def test_add_from_log_round_trip(tmp_path, capsys):
    presets_path = tmp_path / "presets.json"
    presets_path.write_text("{}")

    meta = {
        "name": "step_response", "started_utc": "x",
        "stiffness": 25, "vel_gain": 0.09, "vel_limit": 20, "current": 2.0,
        "cap_slope": 3.0, "cap_min": "0.9,0.5", "ff_scale": 0.7,
    }
    log_dir = _make_log_dir(tmp_path, meta)

    presets_cli.main([
        "--path", str(presets_path), "add", "from-log-test",
        "--from-log", str(log_dir), "--notes", "added from a fake log",
    ])

    presets = json.loads(presets_path.read_text())
    assert "from-log-test" in presets
    p = presets["from-log-test"]
    assert p["stiffness"] == 25
    assert p["current"] == 2.0
    assert p["cap_min"] == [0.9, 0.5]
    assert p["notes"] == "added from a fake log"

    capsys.readouterr()
    presets_cli.main(["--path", str(presets_path), "show", "from-log-test"])
    out = capsys.readouterr().out
    shown = json.loads(out)
    assert shown["current"] == 2.0


def test_verify_stamps_today(tmp_path):
    presets_path = tmp_path / "presets.json"
    presets_path.write_text(json.dumps({"foo": {"stiffness": 1.0}}))

    presets_cli.main(["--path", str(presets_path), "verify", "foo", "--log", "logs/foo-bar"])

    presets = json.loads(presets_path.read_text())
    assert "logs/foo-bar" in presets["foo"]["verified"]


def test_list_no_presets(tmp_path, capsys):
    presets_path = tmp_path / "presets.json"
    presets_path.write_text("{}")
    presets_cli.main(["--path", str(presets_path), "list"])
    out = capsys.readouterr().out
    assert "no presets" in out
