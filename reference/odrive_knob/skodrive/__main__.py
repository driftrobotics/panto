"""CLI entry point.

    python -m skodrive --sim                       # no hardware
    python -m skodrive --interface socketcan --channel can0
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import can

from .config import PRESETS, MotorConfig, TuningConfig
from .knob import SmartKnob
from .odrive_can import DEFAULT_DBC, ODriveAxis, ODriveError, open_bus
from .sim import SimParams, SimulatedODrive
from .web import WebServer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skodrive", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    bus = p.add_argument_group("CAN")
    bus.add_argument("--sim", action="store_true",
                     help="run against a simulated ODrive on a virtual bus")
    bus.add_argument("--interface", default="socketcan",
                     help="python-can interface (socketcan, gs_usb, slcan, ...)")
    bus.add_argument("--channel", default="can0")
    bus.add_argument("--bitrate", type=int, default=1_000_000)
    bus.add_argument("--node-id", type=int, default=0)
    bus.add_argument("--dbc", type=Path, default=DEFAULT_DBC,
                     help="ODrive CANSimple .dbc matching your firmware")

    motor = p.add_argument_group("motor")
    motor.add_argument("--torque-constant", type=float, default=0.035,
                       help="N.m/A (default: SparkFun ROB-20441)")
    motor.add_argument("--max-current", type=float, default=0.8,
                       help="A. NB: 0.8 A is the datasheet *starting* rating; "
                            "~4.4 W continuous will cook a 32 mm motor")
    motor.add_argument("--velocity-limit", type=float, default=20.0, help="turns/s")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--vel-gain", type=float, default=0.02,
                      help="N.m/(turn/s): the local 8 kHz damper. Tune this first.")
    tune.add_argument("--vel-integrator-gain", type=float, default=0.0)
    tune.add_argument("--latency-compensation", type=float, default=0.0,
                      help="seconds of velocity extrapolation before the snap "
                           "decision; set to your measured round-trip")
    tune.add_argument("--dead-zone", action="store_true",
                      help="emulate the firmware dead zone (usually unnecessary "
                           "with a well-tuned vel-gain)")
    tune.add_argument("--invert", action="store_true")

    loop = p.add_argument_group("loop")
    loop.add_argument("--rate", type=float, default=200.0, help="outer loop Hz")
    loop.add_argument("--preset", type=int, default=0)

    srv = p.add_argument_group("ui")
    srv.add_argument("--host", default="0.0.0.0",
                     help="bind address for the UI (default: all interfaces)")
    srv.add_argument("--port", type=int, default=8080)
    srv.add_argument("--no-ui", action="store_true")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("skodrive")

    motor = MotorConfig(
        torque_constant=args.torque_constant,
        max_current=args.max_current,
        velocity_limit=args.velocity_limit,
    )
    tuning = TuningConfig(
        vel_gain=args.vel_gain,
        vel_integrator_gain=args.vel_integrator_gain,
        latency_compensation_s=args.latency_compensation,
        dead_zone_enabled=args.dead_zone,
    )

    sim = None
    if args.sim:
        log.info("simulator mode: virtual CAN bus, no hardware")
        bus = can.Bus(interface="virtual", channel="skodrive")
        sim_bus = can.Bus(interface="virtual", channel="skodrive")
        sim = SimulatedODrive(
            sim_bus,
            node_id=args.node_id,
            params=SimParams(torque_constant=motor.torque_constant),
            dbc_path=args.dbc,
        )
        sim.start()
    else:
        log.info("opening %s on %s", args.interface, args.channel)
        bus = open_bus(args.interface, args.channel, args.bitrate)

    axis = ODriveAxis(bus, node_id=args.node_id, dbc_path=args.dbc, owns_bus=True)
    knob = SmartKnob(
        axis,
        motor=motor,
        tuning=tuning,
        config=PRESETS[args.preset % len(PRESETS)],
        rate_hz=args.rate,
        invert=args.invert,
    )

    try:
        knob.start()
    except ODriveError as exc:
        log.error("%s", exc)
        if sim is not None:
            sim.stop()
        return 1

    def shutdown(*_):
        log.info("shutting down")
        knob.stop()
        if sim is not None:
            sim.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.no_ui:
        log.info("running headless; Ctrl-C to stop")
        signal.pause()
    else:
        log.info("UI at http://%s:%d", args.host, args.port)
        WebServer(knob, host=args.host, port=args.port, sim=sim).run()

    shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
