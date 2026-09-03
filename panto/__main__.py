"""Entry point: wire config -> CAN link -> backend -> runtime -> web server.

    python -m panto --interface socketcan --channel can0
    python -m panto --sim            # once sim.py exists

Mirrors odrive_knob's CLI surface.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="panto")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.parse_args()
    raise NotImplementedError("see HANDOFF.md milestone 2")


if __name__ == "__main__":
    main()
