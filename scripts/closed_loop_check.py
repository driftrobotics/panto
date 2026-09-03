"""Minimal closed-loop bring-up check — FIRST COMMUTATION.

    python -m scripts.closed_loop_check           # 0.15 A cap, 5 s hold
    python -m scripts.closed_loop_check --current 0.2 --hold 8

Sequence, deliberately gentle:

  1. open the link, read current joint state
  2. set a LOW current limit on both axes (default 0.15 A vs the 0.8 A max)
  3. controller mode -> position / passthrough
  4. input position -> the *current* measured angle (so entering closed loop
     commands no motion), pos_gain -> 0 (no spring, ODrive vel damping only)
  5. enter CLOSED_LOOP_CONTROL on both axes
  6. hold and print state / current / errors for `--hold` seconds
  7. return both axes to IDLE

Any exception or Ctrl-C -> both axes commanded IDLE immediately, then close.
The arm should not visibly move at any point; currents should stay near zero
unless you push it by hand.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from panto.can_link import CanLink
from panto.config import Config
from panto.kinematics import forward


def _dump(link: CanLink, geo, tag: str) -> np.ndarray:
    q, qd = link.joint_state()
    xy = forward(q, geo)
    cur = link.motor_currents()
    errs = link.axis_errors()
    print(f"  [{tag:9}] q=({np.degrees(q[0]):7.2f}, {np.degrees(q[1]):7.2f})deg  "
          f"xy=({xy[0]*1e3:6.1f}, {xy[1]*1e3:6.1f})mm  "
          f"i=({cur[0]:.3f}, {cur[1]:.3f})A  "
          f"err={'ok' if not any(errs) else [hex(e) for e in errs]}  "
          f"age={link.feedback_age_s()*1e3:.1f}ms")
    return q


def main() -> None:
    p = argparse.ArgumentParser(prog="closed_loop_check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--current", type=float, default=0.15, help="per-axis current cap, A")
    p.add_argument("--vel-limit", type=float, default=5.0, help="per-axis runaway guard, rad/s")
    p.add_argument("--hold", type=float, default=5.0, help="seconds to hold closed loop")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    nodes = [m.node_id for m in config.motors]
    link = CanLink(config, sim=False)
    print(f"opening {config.can.interface}/{config.can.channel}")
    link.start()

    def idle_all() -> None:
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! failed to idle node {nid}: {exc}")

    try:
        link.wait_for_feedback(timeout=5.0)
        q0 = _dump(link, config.geo, "start")
        for i, nid in enumerate(nodes):
            if link.axis_errors()[i]:
                raise SystemExit(f"node {nid} has an axis error at rest; clear it first")

        print(f"\nsetting limits: {args.current} A, {args.vel_limit} rad/s per axis")
        for nid in nodes:
            link.set_limits(nid, args.vel_limit, args.current)
        print("controller mode -> position/passthrough; anchor -> current angle; pos_gain -> 0")
        for i, nid in enumerate(nodes):
            link.set_controller_mode(nid, "position")
            link.set_input_pos(nid, float(q0[i]))
            link.set_pos_gain(nid, 0.0)
        time.sleep(0.1)

        print("\n>>> entering CLOSED_LOOP_CONTROL on both axes <<<")
        link.enter_closed_loop(timeout=5.0)
        print("    both axes report CLOSED_LOOP_CONTROL\n")

        t_end = time.monotonic() + args.hold
        while time.monotonic() < t_end:
            _dump(link, config.geo, "hold")
            time.sleep(0.5)

        drift = np.degrees(link.joint_state()[0] - q0)
        print(f"\n  joint drift over hold: ({drift[0]:+.2f}, {drift[1]:+.2f}) deg")
    except KeyboardInterrupt:
        print("\n! interrupted")
    finally:
        print("\nreturning both axes to IDLE")
        idle_all()
        time.sleep(0.2)
        _dump(link, config.geo, "end")
        link.close()
        print("closed.")


if __name__ == "__main__":
    main()
