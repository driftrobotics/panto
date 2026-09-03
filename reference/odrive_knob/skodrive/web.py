"""WebSocket + static file server for the HTML UI.

Replaces the round TFT (``firmware/src/display_task.cpp``) and the
COBS/CRC-framed serial protocol: state goes out as JSON over a WebSocket, and
the browser draws it on a canvas.

State is pushed at a fixed rate rather than once per control iteration -- the UI
only needs ~60 Hz, and pushing 200 Hz of JSON per client just adds latency to
the control loop's thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Set

from aiohttp import WSMsgType, web

from .config import PRESETS, KnobConfig
from .knob import SmartKnob
from .sim import SimulatedODrive

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
BROADCAST_HZ = 60.0


class WebServer:
    def __init__(
        self,
        knob: SmartKnob,
        host: str = "127.0.0.1",
        port: int = 8080,
        sim: Optional[SimulatedODrive] = None,
    ) -> None:
        self._knob = knob
        self._host = host
        self._port = port
        self._sim = sim
        self._clients: Set[web.WebSocketResponse] = set()
        self._app = web.Application()
        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/", self._index)
        self._app.router.add_static("/static", UI_DIR)
        self._app.on_startup.append(self._start_broadcast)
        self._app.on_cleanup.append(self._stop_broadcast)
        self._broadcast_task: Optional[asyncio.Task] = None

    def run(self) -> None:
        web.run_app(self._app, host=self._host, port=self._port, print=None)
        log.info("serving UI on http://%s:%d", self._host, self._port)

    # ------------------------------------------------------------------ routes

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(UI_DIR / "index.html")

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        await ws.send_json({"type": "presets", "presets": [_config_json(c) for c in PRESETS]})
        await ws.send_json({"type": "capabilities", "sim": self._sim is not None})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        await self._on_message(json.loads(msg.data))
                    except Exception:
                        log.exception("bad client message: %s", msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "preset":
            index = int(msg["index"])
            self._knob.set_config(PRESETS[index % len(PRESETS)])
        elif kind == "config":
            self._knob.set_config(_config_from_json(msg["config"]))
        elif kind == "torque" and self._sim is not None:
            # Sim only: the UI drags the virtual knob.
            self._sim.apply_external_torque(float(msg["value"]))

    # ------------------------------------------------------------------ broadcast

    async def _start_broadcast(self, app: web.Application) -> None:
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def _stop_broadcast(self, app: web.Application) -> None:
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

    async def _broadcast_loop(self) -> None:
        period = 1.0 / BROADCAST_HZ
        while True:
            await asyncio.sleep(period)
            state = self._knob.latest
            if state is None or not self._clients:
                continue
            stats = self._knob.stats
            payload = json.dumps(
                {
                    "type": "state",
                    "position": state.position,
                    "subPositionUnit": state.sub_position_unit,
                    "angleRad": state.angle_rad,
                    "velocityRadS": state.velocity_rad_s,
                    "detentCenterRad": state.detent_center_rad,
                    "outOfBounds": state.out_of_bounds,
                    "posGain": state.pos_gain,
                    "steps": state.steps,
                    "config": _config_json(state.config),
                    "stats": {
                        "rateHz": round(stats.rate_hz, 1),
                        "jitterP95Ms": round(stats.jitter_p95_ms, 2),
                        "feedbackAgeMs": round(stats.feedback_age_ms, 2),
                        "maxSteps": stats.max_steps,
                        "overruns": stats.overruns,
                        "tx": stats.tx,
                        "rx": stats.rx,
                    },
                }
            )
            for ws in list(self._clients):
                if ws.closed:
                    self._clients.discard(ws)
                    continue
                try:
                    await ws.send_str(payload)
                except Exception:
                    self._clients.discard(ws)


def _config_json(c: KnobConfig) -> dict:
    d = asdict(c)
    d["detent_positions"] = list(c.detent_positions)
    return d


def _config_from_json(d: dict) -> KnobConfig:
    fields = {f for f in KnobConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in d.items() if k in fields}
    if "detent_positions" in kwargs:
        kwargs["detent_positions"] = tuple(int(x) for x in kwargs["detent_positions"])
    return KnobConfig(**kwargs)
