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
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import constraints as _c
from .runtime import Mode, Runtime

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
BROADCAST_HZ = 60.0

#: message kinds a read-only client may still send
_SPECTATOR_OK = {"heartbeat", "take_control"}


class WebServer:
    def __init__(
        self,
        runtime: Runtime,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        link=None,
    ) -> None:
        self._rt = runtime
        self._host = host
        self._port = port
        #: only set in --sim, enables the "hand" torque control
        self._link = link
        self._sim = bool(getattr(link, "_sim_mode", False))

        self._clients: set[web.WebSocketResponse] = set()
        self._controller: web.WebSocketResponse | None = None

        self._app = web.Application()
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/ws", self._ws)
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


async def serve(runtime: Runtime, host: str = "0.0.0.0", port: int = 8080, *, link=None) -> None:
    """Async entry point used by ``python -m panto``. Runs until cancelled."""
    server = WebServer(runtime, host=host, port=port, link=link)
    runner = web.AppRunner(server._app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info("UI on http://%s:%d", host, port)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
