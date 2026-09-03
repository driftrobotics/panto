"""Websocket server bridging the UI and the runtime.

Low-latency, single active controller. A second connection is prompted to kick
the current one or attach read-only (it still receives the telemetry + motion
stream, it just can't send intent). No UI heartbeat for
``config.heartbeat_timeout_s`` -> runtime drops the motors to idle.

Outbound (to all clients): telemetry — mode, FOC/closed-loop state, loop rate,
jitter p95, feedback age, overruns, motor currents, pose, joint angles.
Inbound (from the controller): set_mode, set_constraints, record_start/stop,
playback, set_idle, heartbeat.
"""

from __future__ import annotations

from .runtime import Runtime


async def serve(runtime: Runtime, host: str = "0.0.0.0", port: int = 8080) -> None:
    raise NotImplementedError
