"""Minimal web UI for jogging/e-stopping teleop from a browser.

Runs its own aiohttp server on a daemon thread with its own asyncio event
loop, independent of any other web server in the process (see ``panto/web.py``
for the project's main UI server, which this does not touch). The control
thread (250 Hz) only ever calls the small, lock-cheap, non-blocking surface:
``held``/``jog``/``stop_reason``/``publish``.

Keys are tracked per-websocket-connection and OR-ed across connections, so
one client's stuck key can't be silently cleared by another client's clean
disconnect. A connection's keys are released wholesale on disconnect or on an
explicit ``{"type": "blur"}"`` message (mirrors the browser losing focus).

``stop_reason`` latches "web:estop" forever once tripped -- the control loop
is expected to treat it like any other e-stop and require an out-of-band
restart, not clear it itself.

Interface frozen in CONTRACTS.md ("Teleop web keys (`/teleop`)").
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from pathlib import Path

from aiohttp import WSMsgType, web

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

_KEYS = ("space", "left", "right", "up", "down", "a", "d", "c")
BROADCAST_HZ = 30.0


class TeleopWeb:
    def __init__(self, *, host: str = "0.0.0.0", port: int = 8081) -> None:
        self._host = host
        self._port = port

        # Guards everything below -- polled from the control thread.
        self._lock = threading.Lock()
        self._keys_by_conn: dict[int, set[str]] = {}
        self._stop_reason: str | None = None
        self._state: dict = {}
        self._pending_map: dict | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._app: web.Application | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._broadcast_task: asyncio.Task | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="teleop-web", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)

    def stop(self) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(self._async_stop(), loop)
        try:
            fut.result(timeout=5.0)
        except Exception:  # noqa: BLE001 - best-effort shutdown
            log.exception("TeleopWeb.stop failed")
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._loop = None
        self._thread = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_start())
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _async_start(self) -> None:
        app = web.Application()
        app.router.add_get("/teleop", self._teleop_page)
        app.router.add_get("/ws", self._ws)
        self._app = app
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._broadcast_task = asyncio.ensure_future(self._broadcast_loop())
        log.info("teleop web on %s", self.url)

    async def _async_stop(self) -> None:
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._runner is not None:
            await self._runner.cleanup()
        # loop.stop() is scheduled from stop() *after* this future resolves -- stopping the
        # loop in here would strand the caller's fut.result() (it timed out at 5 s).

    @property
    def url(self) -> str:
        host = self._host if self._host != "0.0.0.0" else "localhost"
        return f"http://{host}:{self._port}/teleop"

    # ------------------------------------------------------------------ control-thread surface

    def held(self, key: str) -> bool:
        with self._lock:
            return any(key in keys for keys in self._keys_by_conn.values())

    def axis(self, neg: str, pos: str) -> float:
        """-1 / 0 / +1 from a pair of held keys (both -> 0)."""
        return float(self.held(pos)) - float(self.held(neg))

    def jog(self) -> float:
        """Base rotate: A = +1, D = -1 (rig convention: +yaw is to the operator's left)."""
        return self.axis("d", "a")

    def stop_reason(self) -> str | None:
        with self._lock:
            return self._stop_reason

    def publish(self, state: dict) -> None:
        with self._lock:
            self._state = dict(state)

    def take_map(self) -> dict | None:
        with self._lock:
            m = self._pending_map
            self._pending_map = None
            return m

    # ------------------------------------------------------------------ http/ws

    async def _teleop_page(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(UI_DIR / "teleop.html")

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        conn_id = id(ws)
        with self._lock:
            self._keys_by_conn[conn_id] = set()
        await ws.send_json({"type": "hello", "keys": []})
        try:
            async for msg in ws:
                if msg.type is WSMsgType.TEXT:
                    try:
                        error = self._dispatch(conn_id, json.loads(msg.data))
                    except Exception:
                        log.exception("bad client message: %s", msg.data)
                    else:
                        if error is not None:
                            await ws.send_json({"type": "error", "message": error})
                elif msg.type is WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
            with self._lock:
                self._keys_by_conn.pop(conn_id, None)
        return ws

    def _dispatch(self, conn_id: int, msg: dict) -> str | None:
        """Handle one client message. Returns an error string to send back, or None."""
        kind = msg.get("type")
        if kind == "key":
            key = msg.get("key")
            down = bool(msg.get("down"))
            if key not in _KEYS:
                return None
            with self._lock:
                keys = self._keys_by_conn.setdefault(conn_id, set())
                if down:
                    keys.add(key)
                else:
                    keys.discard(key)
        elif kind == "blur":
            with self._lock:
                self._keys_by_conn[conn_id] = set()
        elif kind == "estop":
            with self._lock:
                self._stop_reason = "web:estop"
        elif kind == "map":
            error = self._validate_map(msg)
            if error is not None:
                return error
            with self._lock:
                self._pending_map = {
                    "joints": [int(msg["joints"][0]), int(msg["joints"][1])],
                    "scale": [float(msg["scale"][0]), float(msg["scale"][1])],
                }
        else:
            log.debug("ignoring message kind=%r", kind)
        return None

    @staticmethod
    def _validate_map(msg: dict) -> str | None:
        joints = msg.get("joints")
        scale = msg.get("scale")
        if not isinstance(joints, list) or len(joints) != 2:
            return "map: 'joints' must be a list of two ints"
        if not isinstance(scale, list) or len(scale) != 2:
            return "map: 'scale' must be a list of two floats"
        try:
            i0, i1 = int(joints[0]), int(joints[1])
        except (TypeError, ValueError):
            return "map: joints must be ints"
        if isinstance(joints[0], bool) or isinstance(joints[1], bool):
            return "map: joints must be ints"
        if i0 == i1:
            return "map: joints must be distinct"
        if not (0 <= i0 <= 6 and 0 <= i1 <= 6):
            return "map: joints must be in 0..6"
        try:
            s0, s1 = float(scale[0]), float(scale[1])
        except (TypeError, ValueError):
            return "map: scale must be floats"
        if not (math.isfinite(s0) and math.isfinite(s1)):
            return "map: scale must be finite"
        if s0 == 0.0 or s1 == 0.0:
            return "map: scale must be non-zero"
        return None

    # ------------------------------------------------------------------ broadcast

    async def _broadcast_loop(self) -> None:
        period = 1.0 / BROADCAST_HZ
        while True:
            await asyncio.sleep(period)
            if not self._clients:
                continue
            with self._lock:
                state = dict(self._state)
            payload = json.dumps({"type": "state", **state})
            for ws in list(self._clients):
                if ws.closed:
                    self._clients.discard(ws)
                    continue
                try:
                    await ws.send_str(payload)
                except Exception:  # noqa: BLE001
                    self._clients.discard(ws)
