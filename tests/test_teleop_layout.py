"""Layout check for /teleop: everything must fit in one viewport, no scroll.

Starts a real TeleopWeb (port=0), publishes a realistic state dict, and loads
/teleop in headless Chromium via Playwright at a few viewport sizes. Skips
cleanly if Playwright/Chromium isn't available in this environment.
"""

from __future__ import annotations

import pytest

playwright_sync_api = pytest.importorskip("playwright.sync_api")

from panto.teleop_web import TeleopWeb

STATE = {
    "t": 12.3,
    "panto_deg": [84.3, -93.8],
    "yam_deg": [88.6, 91.7],
    "lead_deg": [1.2, -0.4],
    "boxed": [False, False],
    "yam_tau_ext_nm": [0.31, -0.82],
    "panto_iq_a": [0.41, 0.12],
    "base_deg": 87.9,
    "grip": 1.0,
    "grip_close": False,
    "jog": 0.0,
    "ee_offset_cm": [0.0, 0.0],
    "nudge_rejected": 0,
    "map": {"joints": [1, 2], "scale": [-1.0, 1.0], "names": ["joint2", "joint3"]},
    "fault": "",
}

VIEWPORTS = [(1440, 900), (1280, 720), (1920, 1080)]


@pytest.fixture()
def server():
    tw = TeleopWeb(host="127.0.0.1", port=0)
    tw.start()
    port = tw._site._server.sockets[0].getsockname()[1]
    tw._port = port
    yield tw
    tw.stop()


def _launch_browser():
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = None
    errors = []
    for kwargs in ({}, {"channel": "chrome"}, {"executable_path": "/usr/bin/google-chrome"}):
        try:
            browser = pw.chromium.launch(**kwargs)
            break
        except Exception as e:  # pragma: no cover - environment dependent
            errors.append(str(e))
    if browser is None:
        pw.stop()
        pytest.skip(f"chromium unavailable: {errors}")
    return pw, browser


def test_teleop_layout_fits_viewport(server):
    pw, browser = _launch_browser()
    try:
        for width, height in VIEWPORTS:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(f"http://127.0.0.1:{server._port}/teleop")
            page.wait_for_selector("#status")

            # Push a state message the same way a live run would, then wait
            # for the DOM to reflect it (dl children come from renderState).
            server.publish(dict(STATE))
            page.wait_for_function(
                "document.querySelectorAll('#state dt').length > 5"
            )
            page.wait_for_timeout(150)

            scroll_h = page.evaluate("document.documentElement.scrollHeight")
            scroll_w = page.evaluate("document.documentElement.scrollWidth")
            inner_h = page.evaluate("window.innerHeight")
            inner_w = page.evaluate("window.innerWidth")

            assert scroll_h <= inner_h + 1, (
                f"{width}x{height}: page scrolls vertically "
                f"(scrollHeight={scroll_h} > innerHeight={inner_h})"
            )
            assert scroll_w <= inner_w + 1, (
                f"{width}x{height}: page scrolls horizontally "
                f"(scrollWidth={scroll_w} > innerWidth={inner_w})"
            )

            for selector in ["#estop", "#map-apply"]:
                rect = page.eval_on_selector(
                    selector, "el => el.getBoundingClientRect()"
                )
                assert rect["bottom"] <= inner_h + 1, (
                    f"{width}x{height}: {selector} bottom={rect['bottom']} "
                    f"exceeds viewport height {inner_h}"
                )

            last_dd_bottom = page.evaluate(
                "() => { const dds = document.querySelectorAll('#state dd'); "
                "const last = dds[dds.length - 1]; "
                "return last ? last.getBoundingClientRect().bottom : 0; }"
            )
            assert last_dd_bottom <= inner_h + 1, (
                f"{width}x{height}: last state dd bottom={last_dd_bottom} "
                f"exceeds viewport height {inner_h}"
            )

            if (width, height) in [(1440, 900), (1280, 720)]:
                page.screenshot(path=f"/tmp/teleop_{width}.png")

            page.close()
    finally:
        browser.close()
        pw.stop()
