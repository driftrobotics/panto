"""TeleopWeb: key state, jog direction, estop latch, publish broadcast, /teleop route.

Runs a real TeleopWeb server on an ephemeral port (port 0 -> OS-assigned) on
its own daemon thread, and talks to it with a plain aiohttp ClientSession
from a throwaway asyncio.run() per test -- no pytest-asyncio plugin is
installed in this repo's venv, so each test is a normal sync function that
drives its own short-lived event loop.
"""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from panto.teleop_web import TeleopWeb


@pytest.fixture()
def server():
    tw = TeleopWeb(host="127.0.0.1", port=0)
    tw.start()
    sockets = tw._site._server.sockets
    port = sockets[0].getsockname()[1]
    tw._port = port
    yield tw
    tw.stop()


async def _connect(server):
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(f"http://127.0.0.1:{server._port}/ws")
    hello = await ws.receive_json()
    assert hello["type"] == "hello"
    return session, ws


async def _close(session, ws):
    await ws.close()
    await session.close()


def test_held_and_jog_follow_key_messages(server):
    async def run():
        session, ws = await _connect(server)
        try:
            assert server.held("left") is False
            assert server.jog() == 0.0

            await ws.send_json({"type": "key", "key": "right", "down": True})
            await asyncio.sleep(0.05)
            assert server.held("right") is True
            assert server.jog() == 1.0

            await ws.send_json({"type": "key", "key": "right", "down": False})
            await asyncio.sleep(0.05)
            assert server.held("right") is False
            assert server.jog() == 0.0

            await ws.send_json({"type": "key", "key": "space", "down": True})
            await asyncio.sleep(0.05)
            assert server.held("space") is True
        finally:
            await _close(session, ws)

    asyncio.run(run())


def test_disconnect_releases_keys(server):
    async def run():
        session, ws = await _connect(server)
        await ws.send_json({"type": "key", "key": "left", "down": True})
        await asyncio.sleep(0.05)
        assert server.held("left") is True
        await _close(session, ws)
        await asyncio.sleep(0.05)
        assert server.held("left") is False

    asyncio.run(run())


def test_blur_releases_only_that_connections_keys(server):
    async def run():
        session, ws = await _connect(server)
        await ws.send_json({"type": "key", "key": "left", "down": True})
        await asyncio.sleep(0.05)
        await ws.send_json({"type": "blur"})
        await asyncio.sleep(0.05)
        assert server.held("left") is False
        await _close(session, ws)

    asyncio.run(run())


def test_keys_or_across_two_clients(server):
    async def run():
        s1, ws1 = await _connect(server)
        s2, ws2 = await _connect(server)
        try:
            await ws1.send_json({"type": "key", "key": "left", "down": True})
            await asyncio.sleep(0.05)
            assert server.held("left") is True

            # client 2 disconnects without ever pressing left -- client 1's key
            # must survive (per-connection OR, not a global set client 2 can clear).
            await _close(s2, ws2)
            await asyncio.sleep(0.05)
            assert server.held("left") is True
        finally:
            await _close(s1, ws1)

    asyncio.run(run())


def test_estop_latches_stop_reason(server):
    async def run():
        session, ws = await _connect(server)
        try:
            assert server.stop_reason() is None
            await ws.send_json({"type": "estop"})
            await asyncio.sleep(0.05)
            assert server.stop_reason() == "web:estop"
            # latched: further key traffic doesn't clear it
            await ws.send_json({"type": "key", "key": "space", "down": True})
            await asyncio.sleep(0.05)
            assert server.stop_reason() == "web:estop"
        finally:
            await _close(session, ws)

    asyncio.run(run())


def test_publish_reaches_client_as_state_message(server):
    async def run():
        session, ws = await _connect(server)
        try:
            server.publish({"pose": [1.0, 2.0], "closed_loop": True})
            msg = None
            for _ in range(20):
                msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                if msg.get("type") == "state":
                    break
            assert msg["type"] == "state"
            assert msg["pose"] == [1.0, 2.0]
            assert msg["closed_loop"] is True
        finally:
            await _close(session, ws)

    asyncio.run(run())


def test_teleop_route_serves_html(server):
    async def run():
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{server._port}/teleop") as resp:
                assert resp.status == 200
                body = await resp.text()
                assert "<html" in body.lower()
                assert "E-STOP" in body or "estop" in body.lower()

    asyncio.run(run())
