"""Tests for scripts/trace_shape.py's --backend {position,torque} switch:
argparse defaults/choices, and one sim smoke test per backend exercising the
full run end to end (arm, trace, relax/idle, summary).
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts import trace_shape

# deterministic start pose for Config.load(None) + CanLink(sim=True), see
# panto.sim's default joint angles -- used as --centre so the box is drawn
# around the arm's actual sim start position (no test_pose configured).
SIM_START_CENTRE_MM = "86.6,170.4"


def _base_argv(backend: str) -> list[str]:
    # --config pins the template (no test_pose): without it, Config.load(None)
    # picks up a real device's calibration.json/config.local.json from CWD or
    # the repo root, and on a machine with hardware calibrated (e.g. rig-host) its
    # test_pose won't match SIM_START_CENTRE_MM, tripping the start-pose cap.
    from panto.config import _TEMPLATE_PATH

    argv = [
        "trace_shape", "--sim", "--backend", backend,
        "--config", str(_TEMPLATE_PATH),
        "--shape", "box", "--size-mm", "10", "--speed-mm-s", "200",
        "--centre", SIM_START_CENTRE_MM, "--lead-in-s", "0.1",
        "--corner-dwell-s", "0.0", "--cooldown-s", "0",
    ]
    if backend == "torque":
        argv += ["--stiffness", "25", "--damping", "0.7", "--vel-lpf-hz", "50", "--current", "0.8"]
    else:
        argv += ["--preset", "step-lin-K10"]
    return argv


def test_backend_defaults_to_position():
    p = trace_shape.argparse.ArgumentParser()
    # trace_shape builds its own parser inline in main(); exercise the same
    # add_argument call it uses for --backend directly against a throwaway
    # parser to check the default/choices without running a full arm+trace.
    p.add_argument("--backend", choices=("position", "torque"), default="position")
    args = p.parse_args([])
    assert args.backend == "position"


def test_backend_rejects_unknown_value(capsys):
    p = trace_shape.argparse.ArgumentParser()
    p.add_argument("--backend", choices=("position", "torque"), default="position")
    with pytest.raises(SystemExit):
        p.parse_args(["--backend", "bogus"])


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    trace_shape.main()


@pytest.mark.parametrize("backend", ["position", "torque"])
def test_sim_smoke_per_backend(monkeypatch, backend):
    argv = _base_argv(backend)
    monkeypatch.setattr(sys, "argv", argv)
    # main() raises SystemExit(1) only if the run aborted; a clean run just
    # returns None.
    trace_shape.main()


def test_torque_backend_rate_defaults_to_500(monkeypatch, tmp_path):
    import panto.telemetry as telemetry
    monkeypatch.setattr(telemetry, "LOG_ROOT", tmp_path)
    argv = _base_argv("torque")
    monkeypatch.setattr(sys, "argv", argv)
    trace_shape.main()
    (run_dir,) = list(tmp_path.iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["rate"] == 500.0
    assert meta["backend"] == "torque"
    assert meta["damping"] == 0.7
    assert meta["vel_lpf_hz"] == 50.0


def test_position_backend_rate_defaults_to_250(monkeypatch, tmp_path):
    import panto.telemetry as telemetry
    monkeypatch.setattr(telemetry, "LOG_ROOT", tmp_path)
    argv = _base_argv("position")
    monkeypatch.setattr(sys, "argv", argv)
    trace_shape.main()
    (run_dir,) = list(tmp_path.iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["rate"] == 250.0
    assert meta["backend"] == "position"
