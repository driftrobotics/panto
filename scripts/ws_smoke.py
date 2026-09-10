"""Drive a running panto server over its websocket and print what happened.

    python -m panto --sim --port 8093 &
    python -m scripts.ws_smoke --port 8093            # full flow (arms the sim)
    python -m scripts.ws_smoke --port 8080 --passive  # hardware: never sends engage

Reads the *latest* broadcast at each step (drains the socket) so numbers are
never stale. Exit code 1 on any failed check. ``--passive`` is safe on the rig:
it only observes, records and exercises the error path; nothing commutates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def dist_mm(a, b) -> float:
    return 1e3 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class Client:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.ws = ws
        self.errors: list[str] = []
        self.hello: dict | None = None

    async def _recv(self, timeout: float) -> dict:
        msg = await self.ws.receive(timeout=timeout)
        if msg.type is not aiohttp.WSMsgType.TEXT:
            raise ConnectionError(f"socket {msg.type}")
        m = json.loads(msg.data)
        if m["type"] == "error":
            self.errors.append(m["message"])
        elif m["type"] == "hello":
            self.hello = m
        return m

    async def latest(self) -> dict:
        """Drain everything queued, then return the newest ``state``."""
        state = None
        while True:
            try:
                m = await asyncio.wait_for(self._recv(2.0), timeout=0.005)
            except asyncio.TimeoutError:
                if state is not None:
                    return state
                m = await self._recv(2.0)
            if m["type"] == "state":
                state = m

    async def until(self, pred, timeout: float = 3.0) -> dict | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            s = await self.latest()
            if pred(s):
                return s
            await asyncio.sleep(0.02)
        return None

    async def send(self, **msg) -> None:
        await self.ws.send_json(msg)

    async def pop_error(self, timeout: float = 2.0) -> str | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.errors:
                return self.errors.pop(0)
            await self.latest()
        return None


async def heartbeat(c: Client) -> None:
    while True:
        await c.send(type="heartbeat")
        await asyncio.sleep(1.0)


async def run(url: str, passive: bool, perturb: float) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            c = Client(ws)
            hb = asyncio.create_task(heartbeat(c))
            s = await c.latest()
            sim = bool(c.hello and c.hello.get("sim"))
            print(f"connected: controller={c.hello and c.hello['controller']} sim={sim}")
            if c.hello and not c.hello["controller"]:
                # A browser tab holds the controller role; spectator intents are
                # dropped silently, so take it (the tab can take it back).
                await c.send(type="take_control")
                print("  took control from the current controller")
            check(s["closed_loop"] is False, "starts unarmed (no auto-commutate)")
            check(s["stats"]["rate_hz"] > 0, f"loop running at {s['stats']['rate_hz']:.0f} Hz while unarmed")

            print("passive tracking")
            p0 = s["pose"]
            if sim:
                await c.send(type="perturb", tau=[perturb, perturb])
                await asyncio.sleep(0.3)
                await c.send(type="perturb", tau=[0.0, 0.0])
            else:
                print("  -> move the arm by hand for 3 s")
                await asyncio.sleep(3.0)
            s = await c.latest()
            check(dist_mm(s["pose"], p0) > 1.0, f"pose tracks while unarmed ({dist_mm(s['pose'], p0):.1f} mm)")
            check(s["closed_loop"] is False, "still unarmed after motion")

            print("record while passive")
            await c.send(type="record_start")
            s = await c.until(lambda s: s["recording"])
            check(s is not None, "recording flag set")
            if sim:
                await c.send(type="perturb", tau=[-perturb, -perturb * 0.5])
                await asyncio.sleep(0.3)
                await c.send(type="perturb", tau=[0.0, 0.0])
            await asyncio.sleep(0.5)
            await c.send(type="record_stop")
            s = await c.until(lambda s: not s["recording"])
            check(s is not None and s["recorded_samples"] > 50, f"take has {s and s['recorded_samples']} samples")

            print("error path")
            await c.send(type="trace_shape", shape="hexagon", size_m=0.02, centre=s["pose"], speed=0.01)
            err = await c.pop_error()
            check(err is not None and "hexagon" in err, f"bad shape reported to requester: {err!r}")

            if passive:
                hb.cancel()
                return

            print("engage")
            pre = (await c.latest())["pose"]
            await c.send(type="engage")
            s = await c.until(lambda s: s["closed_loop"], timeout=6.0)
            check(s is not None, "armed")
            if s is None:
                print("  engage error:", await c.pop_error())
                hb.cancel()
                return
            await asyncio.sleep(0.5)
            s = await c.latest()
            moved = dist_mm(s["pose"], pre)
            check(moved < 5.0, f"no lunge on arming ({moved:.2f} mm, currents {s['currents']})")
            check(s["mode"] == "transparent", "mode unchanged by engage")

            print("playback")
            await c.send(type="playback", id="last")
            s = await c.until(lambda s: s["mode"] == "plotter")
            check(s is not None, "plotter mode entered")
            if s is not None:
                check(dist_mm(s["anchor"], s["pose"]) < 5.0, f"anchor starts at the pose (ramp), {dist_mm(s['anchor'], s['pose']):.2f} mm")
                await asyncio.sleep(3.0)
                s = await c.latest()
                check(dist_mm(s["anchor"], s["pose"]) < 5.0, f"tracking the take, err {dist_mm(s['anchor'], s['pose']):.2f} mm")
            else:
                print("  playback error:", await c.pop_error())

            print("trace box")
            s = await c.latest()
            await c.send(type="trace_shape", shape="box", size_m=0.03, centre=s["pose"], speed=0.03, laps=1)
            await asyncio.sleep(2.5)
            a0 = (await c.latest())["anchor"]
            await asyncio.sleep(1.0)
            s = await c.latest()
            check(dist_mm(s["anchor"], a0) > 10.0, f"anchor moving along the box ({dist_mm(s['anchor'], a0):.1f} mm/s)")
            check(dist_mm(s["anchor"], s["pose"]) < 5.0, f"tracking the box, err {dist_mm(s['anchor'], s['pose']):.2f} mm")
            check(s["stats"]["overruns"] == 0, f"no overruns ({s['stats']['rate_hz']:.0f} Hz)")

            await c.send(type="trace_shape", shape="box", size_m=0.4, centre=s["pose"], speed=0.03)
            err = await c.pop_error()
            check(err is not None and ("reach" in err or "singularity" in err),
                  f"oversize box rejected: {err!r}")
            s = await c.latest()
            check(s["stats"]["rate_hz"] > 0 and s["mode"] == "plotter", "loop alive after rejection")

            print("go passive")
            await c.send(type="set_idle")
            s = await c.until(lambda s: not s["closed_loop"])
            check(s is not None and s["mode"] == "transparent", "unarmed + transparent after set_idle")
            hb.cancel()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--passive", action="store_true", help="never send engage (safe on hardware)")
    # sim rotor: J 6e-5, b 1.2e-3 -> terminal 2.5 rad/s at 3 mN·m, a gentle ~50 mm nudge
    p.add_argument("--perturb", type=float, default=0.003, help="sim hand torque, N·m")
    a = p.parse_args()
    asyncio.run(run(f"http://{a.host}:{a.port}/ws", a.passive, a.perturb))
    print(f"{len(FAILS)} failed" if FAILS else "all checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
