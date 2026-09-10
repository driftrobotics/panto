"""WebSocket + static-file server bridging the UI and the runtime.

Low-latency, single active controller. The first client to connect becomes the
controller and may send intent (``set_mode`` / ``set_constraints`` / …); further
connections attach **read-only** — they still get the full telemetry stream, they
just can't send intent. A read-only client may ``take_control`` to take over.

Telemetry goes out as JSON at ``BROADCAST_HZ`` (~60), decoupled from the ~200 Hz
control tick: the UI doesn't need more, and pushing control-rate JSON per client
would add latency to the control thread. Nothing physical is computed here — the
control loop lives entirely in :mod:`panto.runtime`.

The runtime idles the motors if no ``heartbeat`` arrives for
``config.heartbeat_timeout_s``; the UI must send one periodically.

Outbound / inbound message shapes are frozen in ``CONTRACTS.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import constraints as _c
from .config import Config, _LIVE_CANDIDATES
from .kinematics import forward
from .runtime import Mode, Runtime

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_REPO_ROOT = UI_DIR.parent
BROADCAST_HZ = 60.0

#: message kinds a read-only client may still send
_SPECTATOR_OK = {"heartbeat", "take_control"}


# ---------------------------------------------------------------------- config
# Settings/calibration are offline for v0 (CONTRACTS.md): write the live JSON
# file, tell the user to restart. Live file discovery mirrors Config.load's
# search (same candidate names/roots) so we always update the file the running
# process would itself pick up next time.

def _live_file_path() -> Path:
    """Existing live-override file if one is found (CWD or repo root, same
    order as Config.load); otherwise the path a *new* one should be created
    at (repo-root calibration.json)."""
    for name in _LIVE_CANDIDATES:
        for root in (Path.cwd(), _REPO_ROOT):
            p = root / name
            if p.is_file():
                return p
    return _REPO_ROOT / "calibration.json"


def _json_safe(x):
    """Recursively replace +-inf/NaN floats with JSON-legal string sentinels.

    ``Config``'s q_min_rad/q_max_rad default to +-inf (MotorConfig, "unset,
    no limiting"). ``json.dumps`` happily emits the literal tokens
    ``Infinity``/``-Infinity`` for those (Python's own JSON extension), but
    that isn't valid JSON -- browsers' ``JSON.parse`` (and ``resp.json()``)
    reject it outright, which silently breaks *any* endpoint that ever
    round-trips a Config dict before every motor has both limits calibrated.
    """
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        if x == float("-inf"):
            return "-inf"
        if x != x:  # NaN
            return "nan"
        return x
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _json(payload: dict, *, status: int = 200) -> web.Response:
    """``web.json_response`` with inf/NaN sanitised first (see _json_safe)."""
    return web.json_response(_json_safe(payload), status=status)


def _deep_merge(base: dict, patch: dict) -> dict:
    """Deep-merge ``patch`` onto ``base``, preserving every key ``patch``
    doesn't mention. ``motors`` merges per-index (a patch can carry just the
    changed fields for one motor) -- same semantics as Config._merge, kept
    local since that helper is private to config.py."""
    out = dict(base)
    for k, v in patch.items():
        if k == "motors" and isinstance(v, list):
            merged = [dict(m) for m in out.get(k, [])]
            for i, m in enumerate(v):
                if i < len(merged):
                    merged[i] = {**merged[i], **m}
                else:
                    merged.append(dict(m))
            out[k] = merged
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class WebServer:
    def __init__(
        self,
        runtime: Runtime,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        link=None,
        config: Config | None = None,
    ) -> None:
        self._rt = runtime
        self._host = host
        self._port = port
        #: only set in --sim, enables the "hand" torque control
        self._link = link
        self._sim = bool(getattr(link, "_sim_mode", False))
        #: the Config this process was started with. Falls back to the
        #: runtime's own (if it happens to expose one) then a bare default --
        #: settings/calibration should still be *readable* even if nobody
        #: wired a config through explicitly (e.g. older callers, tests).
        self._config: Config = config if config is not None else (
            getattr(runtime, "_cfg", None) or Config()
        )

        self._clients: set[web.WebSocketResponse] = set()
        self._controller: web.WebSocketResponse | None = None

        self._app = web.Application()
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/settings", self._settings_page)
        self._app.router.add_get("/calibrate", self._calibrate_page)
        self._app.router.add_get("/ws", self._ws)
        self._app.router.add_get("/api/config", self._api_config_get)
        self._app.router.add_post("/api/config", self._api_config_post)
        self._app.router.add_get("/api/calibration/raw", self._api_calibration_raw)
        self._app.router.add_post("/api/calibration/zero", self._api_calibration_zero)
        self._app.router.add_post("/api/calibration/limit", self._api_calibration_limit)
        self._app.router.add_post("/api/calibration/workspace", self._api_calibration_workspace)
        self._app.router.add_static("/static", UI_DIR)
        self._app.on_startup.append(self._on_startup)
        self._app.on_cleanup.append(self._on_cleanup)
        self._broadcast: asyncio.Task | None = None

    # ------------------------------------------------------------------ serve

    def run(self) -> None:
        web.run_app(self._app, host=self._host, port=self._port, print=None)

    async def _on_startup(self, _app: web.Application) -> None:
        self._broadcast = asyncio.create_task(self._broadcast_loop())

    async def _on_cleanup(self, _app: web.Application) -> None:
        if self._broadcast is not None:
            self._broadcast.cancel()
            try:
                await self._broadcast
            except asyncio.CancelledError:
                pass

    async def _index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(UI_DIR / "index.html")

    async def _settings_page(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(UI_DIR / "settings.html")

    async def _calibrate_page(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(UI_DIR / "calibrate.html")

    # ------------------------------------------------------------------ ws

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        if self._controller is None:
            self._controller = ws
        await ws.send_json(
            {
                "type": "hello",
                "controller": ws is self._controller,
                "sim": self._sim,
                "modes": [m.value for m in Mode],
            }
        )
        try:
            async for msg in ws:
                if msg.type is WSMsgType.TEXT:
                    try:
                        self._dispatch(ws, json.loads(msg.data))
                    except Exception:
                        log.exception("bad client message: %s", msg.data)
                elif msg.type is WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
            if ws is self._controller:
                self._controller = next(iter(self._clients), None)
                self._broadcast_roles()
        return ws

    def _dispatch(self, ws: web.WebSocketResponse, msg: dict) -> None:
        kind = msg.get("type")
        if ws is not self._controller and kind not in _SPECTATOR_OK:
            return

        if kind == "heartbeat":
            self._rt.note_heartbeat()
        elif kind == "take_control":
            self._controller = ws
            self._broadcast_roles()
        elif kind == "set_mode":
            self._rt.set_mode(msg["mode"])
        elif kind == "set_idle":
            self._rt.set_idle()
        elif kind == "set_constraints":
            self._rt.set_constraints(_decode_constraints(msg["constraints"]))
        elif kind == "set_plotter_trajectory":
            self._rt.set_plotter_trajectory(
                [(float(t), np.asarray(p, float)) for t, p in msg["points"]]
            )
        elif kind == "engage":
            self._try(ws, self._rt.engage)
        elif kind == "record_start":
            self._rt.record_start()
        elif kind == "record_stop":
            self._rt.record_stop()
        elif kind == "playback":
            self._try(ws, self._rt.playback, msg.get("id", "last"))
        elif kind == "trace_shape":
            self._try(
                ws,
                self._rt.trace_shape,
                msg["shape"],
                float(msg["size_m"]),
                msg["centre"],
                float(msg["speed"]),
                laps=int(msg.get("laps", 1)),
            )
        elif kind == "perturb" and self._link is not None:
            self._link.inject_joint_torque(np.asarray(msg["tau"], float))
        elif kind == "clear_errors":
            fn = getattr(self._rt, "clear_errors", None)
            if callable(fn):
                fn()
        elif kind == "set_tuning":
            self._try(ws, self._rt.set_tuning,
                      **{k: v for k, v in msg.items() if k != "type"})
        else:
            log.debug("ignoring message kind=%r", kind)

    def _try(self, ws: web.WebSocketResponse, fn, *args, **kwargs) -> None:
        """Call ``fn``; on exception, report to the requesting ``ws`` only
        (not broadcast) rather than letting it propagate out of ``_dispatch``."""
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - reported to the client, not raised
            asyncio.create_task(ws.send_json({"type": "error", "message": str(exc)}))

    def _broadcast_roles(self) -> None:
        for ws in self._clients:
            asyncio.create_task(
                ws.send_json({"type": "role", "controller": ws is self._controller})
            )

    # ------------------------------------------------------------------ config API
    #
    # Settings/calibration are offline for v0 (CONTRACTS.md): every write here
    # updates the live JSON file and tells the caller a restart is required --
    # nothing here touches the running Config/Runtime's live behaviour.

    def _persist_patch(self, patch: dict) -> Config:
        """Deep-merge ``patch`` onto the live-override file's current raw
        content, validate the result by round-tripping it through Config's own
        loader (template + live merge, same as production), then write it.
        Raises on validation failure -- caller turns that into a 400."""
        live_path = _live_file_path()
        live = json.loads(live_path.read_text()) if live_path.is_file() else {}
        merged_live = _deep_merge(live, patch)

        fd, tmp_name = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(merged_live, f)
            cfg = Config.load(tmp_name)  # raises on validation failure
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

        live_path.write_text(json.dumps(merged_live, indent=2))
        self._config = cfg
        return cfg

    def _refuse_if_armed(self) -> web.Response | None:
        """409 while the runtime reports closed_loop=True; calibration writes
        require the arm to be passive (motors IDLE, operator holding it)."""
        tel = self._rt.telemetry()
        if bool(tel.get("closed_loop", False)):
            return _json(
                {"error": "refusing calibration write while armed (closed_loop)"},
                status=409,
            )
        return None

    async def _api_config_get(self, _request: web.Request) -> web.Response:
        return _json(self._config.to_dict())

    async def _api_config_post(self, request: web.Request) -> web.Response:
        try:
            patch = await request.json()
        except Exception as exc:  # noqa: BLE001 - reported as 400, not raised
            return _json({"error": f"invalid JSON: {exc}"}, status=400)
        try:
            cfg = self._persist_patch(patch)
        except Exception as exc:  # noqa: BLE001 - validation failure -> 400
            return _json({"error": str(exc)}, status=400)
        return _json({"config": cfg.to_dict(), "restart_required": True})

    async def _api_calibration_raw(self, _request: web.Request) -> web.Response:
        if self._link is None:
            return _json({"error": "no CAN link"}, status=400)
        raw = np.asarray(self._link.raw_turns(), dtype=float)
        q, _qd = self._link.joint_state()
        q = np.asarray(q, dtype=float)
        tip_m = forward(q, self._config.geo)
        tel = self._rt.telemetry()
        return _json({
            "raw_turns": raw.tolist(),
            "q_rad": q.tolist(),
            "q_deg": np.degrees(q).tolist(),
            "tip_mm": (np.asarray(tip_m, dtype=float) * 1000.0).tolist(),
            "closed_loop": bool(tel.get("closed_loop", False)),
        })

    async def _api_calibration_zero(self, request: web.Request) -> web.Response:
        armed = self._refuse_if_armed()
        if armed is not None:
            return armed
        if self._link is None:
            return _json({"error": "no CAN link"}, status=400)
        try:
            body = await request.json()
        except Exception:
            body = {}
        q_target_deg = body.get("q_target_deg", [0.0, 0.0])

        raw = np.asarray(self._link.raw_turns(), dtype=float)
        motors = list(self._config.motors)
        motors_patch = []
        for i, m in enumerate(motors):
            s = -1.0 if m.flip else 1.0
            q_target_rad = np.radians(float(q_target_deg[i]))
            zero_offset_rad = q_target_rad - s * (2.0 * np.pi * raw[i])
            motors_patch.append({"node_id": m.node_id, "zero_offset_rad": zero_offset_rad})

        try:
            cfg = self._persist_patch({"motors": motors_patch})
        except Exception as exc:  # noqa: BLE001
            return _json({"error": str(exc)}, status=400)
        return _json({
            "motors": [
                {"node_id": m.node_id, "zero_offset_rad": m.zero_offset_rad}
                for m in cfg.motors
            ],
        })

    async def _api_calibration_limit(self, request: web.Request) -> web.Response:
        armed = self._refuse_if_armed()
        if armed is not None:
            return armed
        try:
            body = await request.json()
            motor_idx = int(body["motor"])
            bound = str(body["bound"])
        except Exception as exc:  # noqa: BLE001
            return _json({"error": f"bad request: {exc}"}, status=400)
        if bound not in ("min", "max"):
            return _json({"error": "bound must be 'min' or 'max'"}, status=400)
        motors = list(self._config.motors)
        if self._link is None or not (0 <= motor_idx < len(motors)):
            return _json({"error": "invalid motor index or no link"}, status=400)

        q, _qd = self._link.joint_state()
        q_val = float(np.asarray(q, dtype=float)[motor_idx])
        key = "q_min_rad" if bound == "min" else "q_max_rad"

        motors_patch = [{} for _ in motors]
        motors_patch[motor_idx] = {"node_id": motors[motor_idx].node_id, key: q_val}
        try:
            cfg = self._persist_patch({"motors": motors_patch})
        except Exception as exc:  # noqa: BLE001
            return _json({"error": str(exc)}, status=400)
        m = cfg.motors[motor_idx]
        return _json({
            "motor": motor_idx, "node_id": m.node_id,
            "q_min_rad": m.q_min_rad, "q_max_rad": m.q_max_rad,
        })

    async def _api_calibration_workspace(self, request: web.Request) -> web.Response:
        armed = self._refuse_if_armed()
        if armed is not None:
            return armed
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return _json({"error": f"invalid JSON: {exc}"}, status=400)
        polygon = body.get("polygon")
        try:
            cfg = self._persist_patch({"workspace_polygon": polygon})
        except Exception as exc:  # noqa: BLE001
            return _json({"error": str(exc)}, status=400)
        return _json({
            "workspace_polygon": (
                None if cfg.workspace_polygon is None
                else np.asarray(cfg.workspace_polygon).tolist()
            ),
        })

    # ------------------------------------------------------------------ broadcast

    async def _broadcast_loop(self) -> None:
        period = 1.0 / BROADCAST_HZ
        while True:
            await asyncio.sleep(period)
            if not self._clients:
                continue
            payload = json.dumps(self._rt.telemetry())
            for ws in list(self._clients):
                if ws.closed:
                    self._clients.discard(ws)
                    continue
                try:
                    await ws.send_str(payload)
                except Exception:
                    self._clients.discard(ws)


def _decode_constraints(items: list) -> list:
    """UI JSON constraint list -> constraint objects.

    The UI sends analytic primitives only (CONTRACTS.md: "analytic primitives are
    the source of truth"), never resampled points.
    """
    out: list = []
    for it in items:
        kind = it.get("kind")
        if kind == "point":
            out.append(_c.Point(at=np.asarray(it["at"], float)))
        elif kind == "line":
            out.append(_c.Line(a=np.asarray(it["a"], float), d=np.asarray(it["d"], float)))
        elif kind == "wall":
            out.append(
                _c.Wall(a=np.asarray(it["a"], float), normal=np.asarray(it["normal"], float))
            )
        elif kind == "grid":
            out.append(
                _c.SnapGrid(pitch=float(it["pitch"]), origin=np.asarray(it["origin"], float))
            )
        else:
            log.warning("unknown constraint kind: %r", kind)
    return out


async def serve(
    runtime: Runtime, host: str = "0.0.0.0", port: int = 8080, *,
    link=None, config: Config | None = None,
) -> None:
    """Async entry point used by ``python -m panto``. Runs until cancelled."""
    server = WebServer(runtime, host=host, port=port, link=link, config=config)
    runner = web.AppRunner(server._app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info("UI on http://%s:%d", host, port)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
