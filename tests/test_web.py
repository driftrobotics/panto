"""Tests for the settings/calibration REST API in panto/web.py.

No pytest-aiohttp / pytest-asyncio in this env (checked ``pip list``), so each
test wraps a small ``async def`` body in ``asyncio.run`` and drives the app
directly with ``aiohttp.test_utils.TestServer`` / ``TestClient`` -- no extra
deps beyond aiohttp itself (already a project dependency). Every request in a
given test happens inside a single ``asyncio.run`` call (one event loop): the
``web.Application`` binds to the loop it first starts under, so reusing the
same app across two separate ``asyncio.run`` calls raises "different loop".

Every test isolates its live-config file write to ``tmp_path``: both the CWD
(``monkeypatch.chdir``) and ``panto.web._REPO_ROOT`` (the fallback "create a
new calibration.json here" location) are pointed at ``tmp_path``, so nothing
here ever touches the real repo root's calibration.json.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from panto import web as web_mod
from panto.can_link import joint_from_turns
from panto.config import Config
from panto.web import WebServer


# --------------------------------------------------------------------- fakes

class FakeRuntime:
    """Minimal stand-in for Runtime: only what web.py's new endpoints touch."""

    def __init__(self, closed_loop: bool = False) -> None:
        self._closed_loop = closed_loop
        self.cleared = 0

    def telemetry(self) -> dict:
        return {"type": "state", "closed_loop": self._closed_loop, "mode": "transparent"}

    def clear_errors(self) -> None:
        self.cleared += 1


class FakeLink:
    """Minimal stand-in for CanLink: raw_turns() + joint_state() only."""

    def __init__(self, raw=(0.0, 0.0), q=(0.0, 0.0), qd=(0.0, 0.0)) -> None:
        self._raw = np.asarray(raw, dtype=float)
        self._q = np.asarray(q, dtype=float)
        self._qd = np.asarray(qd, dtype=float)

    def raw_turns(self) -> np.ndarray:
        return self._raw

    def joint_state(self):
        return self._q, self._qd


# --------------------------------------------------------------------- harness

@pytest.fixture()
def isolated_live_file(tmp_path, monkeypatch):
    """Confine every live-config write this suite makes to tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_mod, "_REPO_ROOT", tmp_path)
    return tmp_path


def make_server(*, closed_loop=False, link=None, config=None) -> tuple[WebServer, FakeRuntime]:
    rt = FakeRuntime(closed_loop=closed_loop)
    srv = WebServer(rt, link=link if link is not None else FakeLink(), config=config)
    return srv, rt


async def _get(client: TestClient, path: str):
    resp = await client.get(path)
    return resp.status, await resp.json()


async def _post(client: TestClient, path: str, body: dict):
    resp = await client.post(path, json=body)
    return resp.status, await resp.json()


def run_with_client(app, body):
    """Run ``body(client)`` (an async callable) against one TestClient/loop."""
    async def _main():
        async with TestClient(TestServer(app)) as client:
            return await body(client)
    return asyncio.run(_main())


# --------------------------------------------------------------------- /api/config

def test_config_get_shape(isolated_live_file):
    srv, _rt = make_server(config=Config())

    status, data = run_with_client(srv._app, lambda c: _get(c, "/api/config"))

    assert status == 200
    assert data["geo"]["l1"] == Config().geo.l1
    assert len(data["motors"]) == 2
    assert {"node_id", "flip", "zero_offset_rad", "current_soft_max",
            "vel_gain", "torque_constant"} <= data["motors"][0].keys()
    assert "control" in data and "stiffness_n_per_m" in data["control"]
    assert "sigma_min_threshold" in data
    assert "heartbeat_timeout_s" in data


def test_config_post_merge_updates_only_patched_keys(isolated_live_file):
    srv, _rt = make_server(config=Config())

    async def body(client):
        s1, d1 = await _post(client, "/api/config", {"heartbeat_timeout_s": 2.5})
        s2, d2 = await _get(client, "/api/config")
        return s1, d1, s2, d2

    status, data, status2, data2 = run_with_client(srv._app, body)

    assert status == 200
    assert data["restart_required"] is True
    assert data["config"]["heartbeat_timeout_s"] == 2.5
    # untouched fields survive round-trip through the template defaults
    assert data["config"]["sigma_min_threshold"] == Config().sigma_min_threshold

    # live file on disk holds only the patch, not the whole config
    live_path = isolated_live_file / "calibration.json"
    assert live_path.is_file()
    on_disk = json.loads(live_path.read_text())
    assert on_disk == {"heartbeat_timeout_s": 2.5}

    # a second GET reflects the now-updated in-memory config
    assert status2 == 200
    assert data2["heartbeat_timeout_s"] == 2.5


def test_config_post_validation_failure_returns_400(isolated_live_file):
    srv, _rt = make_server(config=Config())

    async def body(client):
        s1, d1 = await _post(client, "/api/config", {"heartbeat_timeout_s": -1})
        s2, d2 = await _get(client, "/api/config")
        return s1, d1, s2, d2

    status, data, status2, data2 = run_with_client(srv._app, body)

    assert status == 400
    assert "error" in data and data["error"]

    # rejected patch must not have mutated the live config or written a file
    assert data2["heartbeat_timeout_s"] == Config().heartbeat_timeout_s
    assert not (isolated_live_file / "calibration.json").is_file()


def test_config_post_preserves_existing_live_keys(isolated_live_file):
    # pre-seed a live file the way a previous calibration session would have
    (isolated_live_file / "calibration.json").write_text(
        json.dumps({"motors": [{"node_id": 0, "zero_offset_rad": 0.75}, {}]})
    )
    srv, _rt = make_server(config=Config.load())

    status, data = run_with_client(
        srv._app, lambda c: _post(c, "/api/config", {"sigma_min_threshold": 0.05})
    )
    assert status == 200

    on_disk = json.loads((isolated_live_file / "calibration.json").read_text())
    assert on_disk["sigma_min_threshold"] == 0.05
    assert on_disk["motors"][0]["zero_offset_rad"] == 0.75  # preserved, not clobbered


# --------------------------------------------------------------------- calibration/raw

def test_calibration_raw_readout(isolated_live_file):
    link = FakeLink(raw=(1.0, -0.5), q=(0.3, -0.6))
    srv, _rt = make_server(link=link, config=Config())

    status, data = run_with_client(srv._app, lambda c: _get(c, "/api/calibration/raw"))

    assert status == 200
    assert data["raw_turns"] == pytest.approx([1.0, -0.5])
    assert data["q_rad"] == pytest.approx([0.3, -0.6])
    assert data["q_deg"] == pytest.approx(np.degrees([0.3, -0.6]).tolist())
    assert len(data["tip_mm"]) == 2
    assert data["closed_loop"] is False


# --------------------------------------------------------------------- calibration/zero

def test_calibration_zero_matches_transform(isolated_live_file):
    cfg = Config()  # motor0 flip=True, motor1 flip=False (template defaults)
    link = FakeLink(raw=(1.25, -0.5))
    srv, _rt = make_server(link=link, config=cfg)

    status, data = run_with_client(
        srv._app,
        lambda c: _post(c, "/api/calibration/zero", {"q_target_deg": [10.0, -5.0]}),
    )

    assert status == 200
    zeros = {m["node_id"]: m["zero_offset_rad"] for m in data["motors"]}
    for i, motor in enumerate(cfg.motors):
        q = joint_from_turns(link._raw[i], motor.flip, zeros[motor.node_id])
        target = np.radians([10.0, -5.0][i])
        assert q == pytest.approx(target, abs=1e-9)


def test_calibration_zero_default_target_zeros_current_pose(isolated_live_file):
    cfg = Config()
    link = FakeLink(raw=(0.4, 2.2))
    srv, _rt = make_server(link=link, config=cfg)

    status, data = run_with_client(
        srv._app, lambda c: _post(c, "/api/calibration/zero", {})
    )

    assert status == 200
    zeros = {m["node_id"]: m["zero_offset_rad"] for m in data["motors"]}
    for i, motor in enumerate(cfg.motors):
        q = joint_from_turns(link._raw[i], motor.flip, zeros[motor.node_id])
        assert q == pytest.approx(0.0, abs=1e-9)


def test_calibration_zero_refused_while_armed(isolated_live_file):
    srv, _rt = make_server(closed_loop=True, config=Config())

    status, data = run_with_client(
        srv._app, lambda c: _post(c, "/api/calibration/zero", {})
    )

    assert status == 409
    assert "error" in data


# --------------------------------------------------------------------- calibration/limit

def test_calibration_limit_records_current_q(isolated_live_file):
    cfg = Config()
    link = FakeLink(q=(0.7, -0.3))
    srv, _rt = make_server(link=link, config=cfg)

    async def body(client):
        s1, d1 = await _post(client, "/api/calibration/limit", {"motor": 0, "bound": "min"})
        s2, d2 = await _post(client, "/api/calibration/limit", {"motor": 1, "bound": "max"})
        s3, d3 = await _get(client, "/api/config")
        return s1, d1, s2, d2, s3, d3

    status1, data1, status2, data2, status3, data3 = run_with_client(srv._app, body)

    assert status1 == 200
    assert data1["q_min_rad"] == pytest.approx(0.7)
    assert data1["q_max_rad"] == "inf"  # +-inf floats aren't valid JSON; see _json_safe

    assert status2 == 200
    assert data2["q_max_rad"] == pytest.approx(-0.3)

    # both survive together in the merged config
    assert status3 == 200
    assert data3["motors"][0]["q_min_rad"] == pytest.approx(0.7)
    assert data3["motors"][1]["q_max_rad"] == pytest.approx(-0.3)


def test_calibration_limit_refused_while_armed(isolated_live_file):
    srv, _rt = make_server(closed_loop=True, config=Config())

    status, data = run_with_client(
        srv._app,
        lambda c: _post(c, "/api/calibration/limit", {"motor": 0, "bound": "min"}),
    )

    assert status == 409


# --------------------------------------------------------------------- calibration/workspace

def test_calibration_workspace_save_and_validate(isolated_live_file):
    srv, _rt = make_server(config=Config())
    poly = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]

    status, data = run_with_client(
        srv._app, lambda c: _post(c, "/api/calibration/workspace", {"polygon": poly})
    )

    assert status == 200
    np.testing.assert_allclose(np.asarray(data["workspace_polygon"]), np.asarray(poly))


def test_calibration_workspace_rejects_short_polygon(isolated_live_file):
    srv, _rt = make_server(config=Config())

    status, data = run_with_client(
        srv._app,
        lambda c: _post(c, "/api/calibration/workspace", {"polygon": [[0.0, 0.0], [1.0, 1.0]]}),
    )

    assert status == 400
    assert "error" in data


def test_calibration_workspace_refused_while_armed(isolated_live_file):
    srv, _rt = make_server(closed_loop=True, config=Config())

    status, data = run_with_client(
        srv._app,
        lambda c: _post(
            c, "/api/calibration/workspace", {"polygon": [[0, 0], [1, 0], [1, 1]]}
        ),
    )

    assert status == 409
