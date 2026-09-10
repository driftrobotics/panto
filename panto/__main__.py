"""Entry point: wire config -> CAN link -> backend -> runtime -> web server.

    python -m panto                       # hardware: socketcan/can0 from config
    python -m panto --sim                 # no hardware: in-process 2-axis sim
    python -m panto --backend torque      # milestone 6

Mirrors the odrive_knob CLI surface.
"""

from __future__ import annotations

import argparse
import logging
import socket

from .backends import PositionBackend, TorqueBackend
from .can_link import CanLink
from .config import Config
from .runtime import Runtime
from .web import WebServer

_BACKENDS = {"position": PositionBackend, "torque": TorqueBackend}


def main() -> None:
    p = argparse.ArgumentParser(prog="panto", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true", help="run against the in-process simulator")
    p.add_argument("--sim-plant", help="--sim only: plant_model.json (scripts/sysid.py) "
                                       "shaping both axes' SimParams")
    p.add_argument("--sim-coupled", action="store_true",
                   help="--sim only: integrate both axes as one 2R arm")
    p.add_argument("--interface", help="python-can interface (overrides config)")
    p.add_argument("--channel", help="CAN channel (overrides config)")
    p.add_argument("--config", help="path to a live-override config json")
    p.add_argument("--backend", choices=list(_BACKENDS), default="position")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("panto")

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    sim_params = None
    if args.sim_plant:
        from .sim import SimParams
        sim_params = SimParams.from_plant_model(args.sim_plant)
    link = CanLink(config, sim=args.sim, sim_params=sim_params, sim_coupled=args.sim_coupled)
    backend = _BACKENDS[args.backend](link, config)
    runtime = Runtime(config, link, backend)

    # Fail before touching the CAN bus if another instance holds the port --
    # a stale `python -m panto` is the usual cause (find it with `ss -ltnp`).
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((args.host, args.port))
    except OSError as exc:
        raise SystemExit(f"port {args.host}:{args.port} is busy ({exc.strerror}); another "
                         f"panto instance is probably running -- see `ss -ltnp | grep {args.port}`")
    finally:
        probe.close()

    log.info("starting runtime (%s backend, %s)", args.backend,
             "sim" if args.sim else f"{config.can.interface}/{config.can.channel}")
    runtime.start()
    try:
        WebServer(runtime, host=args.host, port=args.port, link=link, config=config).run()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
