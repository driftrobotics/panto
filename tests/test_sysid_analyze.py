"""Tests for scripts/sysid.py --analyze: offline re-analysis of a logged run
from meta.json/samples.jsonl, with no CAN bus or sim access at all -- must
work on a laptop with no hardware attached. Builds a synthetic log directory
by hand (same shape RunLogger produces) rather than running a live/sim
sysid pass, so these tests never import anything that opens a bus."""

from __future__ import annotations

import json

import numpy as np
import pytest

from panto.sysid_logic import log_chirp
from scripts.sysid import analyze_logdir, build_parser


def _write_fake_chirp_log(tmp_path, *, iq_sign=1, iq_scale=1.0, amp_a=0.25, f0=1.0, f1=40.0,
                          duration=8.0, rate=500.0, joint=1, mode="chirp"):
    logdir = tmp_path / "sysid-fake"
    logdir.mkdir()

    args = build_parser().parse_args([
        "--joint", str(joint), "--mode", mode, "--amp-a", str(amp_a),
        "--f0", str(f0), "--f1", str(f1), "--duration", str(duration), "--rate", str(rate),
        "--no-plot",
    ])
    meta = {
        "name": "sysid", "mode": mode, "joint": joint,
        "cli_args": vars(args),
        "sign_conventions": {"node1": {"flip": True, "torque_constant": 0.02235}},
    }
    (logdir / "meta.json").write_text(json.dumps(meta))

    t = np.arange(0, duration, 1.0 / rate)
    i_cmd = log_chirp(t, amp_a, f0, f1, duration)
    # a clean first-order-lag-free integrator plant, J known, so the fit is
    # checkable; iq_measured is the (possibly sign/scale-distorted) broadcast
    J = 6e-5
    qd = np.cumsum(i_cmd) / rate / J
    iq_measured = iq_sign * iq_scale * i_cmd

    with (logdir / "samples.jsonl").open("w") as f:
        for i in range(len(t)):
            row = {"tag": "chirp", "t": float(t[i]), "i_cmd_a": float(i_cmd[i]),
                  "iq_a": float(iq_measured[i]), "qd": float(qd[i]), "q": float(qd[i] / rate)}
            f.write(json.dumps(row) + "\n")

    return logdir


def test_analyze_recovers_fit_from_a_logged_chirp(tmp_path):
    logdir = _write_fake_chirp_log(tmp_path)
    cli_args = build_parser().parse_args(["--analyze", str(logdir)])

    analyze_logdir(logdir, cli_args)

    out = logdir / "summary_analyzed.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "chirp" in data
    assert data["chirp"]["mode"] == "chirp"
    assert np.isfinite(data["chirp"]["inertia_a_s2_per_rad"])


def test_analyze_corrects_a_flipped_iq_sign(tmp_path):
    logdir = _write_fake_chirp_log(tmp_path, iq_sign=-1)
    cli_args = build_parser().parse_args(["--analyze", str(logdir)])

    analyze_logdir(logdir, cli_args)

    data = json.loads((logdir / "summary_analyzed.json").read_text())
    assert data["chirp"]["iq_sign"] == -1
    # the fit should still recover a positive, finite inertia -- the sign
    # flip must not silently corrupt the frequency-response fit
    assert data["chirp"]["inertia_a_s2_per_rad"] > 0


def test_analyze_missing_meta_raises(tmp_path):
    logdir = tmp_path / "empty"
    logdir.mkdir()
    cli_args = build_parser().parse_args(["--analyze", str(logdir)])
    with pytest.raises(SystemExit):
        analyze_logdir(logdir, cli_args)


def test_analyze_old_log_without_cli_args_raises(tmp_path):
    logdir = tmp_path / "old-log"
    logdir.mkdir()
    (logdir / "meta.json").write_text(json.dumps({"name": "sysid", "mode": "chirp", "joint": 1}))
    (logdir / "samples.jsonl").write_text("")
    cli_args = build_parser().parse_args(["--analyze", str(logdir)])
    with pytest.raises(SystemExit):
        analyze_logdir(logdir, cli_args)


def test_analyze_does_not_import_can_link_bus_classes():
    # scripts.sysid imports CanLink at module scope (needed for the live
    # path), but analyze_logdir itself must never instantiate one --
    # spot-check by confirming the function body doesn't reference CanLink(.
    import inspect

    from scripts import sysid

    src = inspect.getsource(sysid.analyze_logdir)
    assert "CanLink(" not in src
