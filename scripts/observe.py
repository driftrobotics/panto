"""Passive hardware-in-the-loop observer — READ ONLY, never transmits.

    python -m scripts.observe --interface socketcan --channel can0

Opens the CAN link, listens to the two ODrives' cyclic broadcasts, and prints
calibrated joint state, the FK end-effector pose, per-motor current, feedback age
and any axis errors at a few Hz. It does **not** enter closed loop, send a
setpoint, or command an axis state — nothing on the bus changes.

Use it to sanity-check the CANSimple codec against the real drives and to watch
joint angles / pose while hand-positioning the linkage, before any calibration
or closed-loop bring-up.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from panto.can_link import CanLink
from panto.config import Config
from panto.kinematics import forward, min_singular_value
from panto.limits import format_limits_deg
from panto.telemetry import RunLogger


def main() -> None:
    p = argparse.ArgumentParser(prog="observe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", help="python-can interface (overrides config)")
    p.add_argument("--channel", help="CAN channel (overrides config)")
    p.add_argument("--config", help="live-override config json")
    p.add_argument("--hz", type=float, default=4.0, help="print rate")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    log = RunLogger("observe", interface=config.can.interface, channel=config.can.channel,
                    hz=args.hz)
    link = CanLink(config, sim=False)
    print(f"opening {config.can.interface}/{config.can.channel} (read-only)…")
    link.start()
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
    except Exception as exc:
        link.close()
        raise SystemExit(f"no feedback: {exc}")

    wrap = link.wrap_turns()
    limits_msg = f"limits (deg): {format_limits_deg(config.motors)}  wrap_turns={wrap}"
    print(f"connected. Ctrl-C to stop.\n{limits_msg}\n")
    log.event(limits_msg)
    print(f"{'q1 (deg)':>10} {'q2 (deg)':>10} {'x (mm)':>9} {'y (mm)':>9} "
          f"{'|v| rad/s':>10} {'i0':>6} {'i1':>6} {'age ms':>7} {'sigma_min':>10}  errors")
    period = 1.0 / args.hz
    try:
        while True:
            q, qd = link.joint_state()
            xy = forward(q, config.geo)
            cur = link.motor_currents()
            age = link.feedback_age_s() * 1e3
            status = link.node_status()
            sig = min_singular_value(q, config.geo)
            estr = "ok" if all(s.active_errors == 0 for s in status) else \
                " ".join(f"axis{s.node_id}:0x{s.active_errors:x}" for s in status)
            print(f"{np.degrees(q[0]):10.2f} {np.degrees(q[1]):10.2f} "
                  f"{xy[0]*1e3:9.1f} {xy[1]*1e3:9.1f} {np.linalg.norm(qd):10.3f} "
                  f"{cur[0]:6.2f} {cur[1]:6.2f} {age:7.1f} {sig:10.4f}  {estr}")
            log.sample(q=q, xy=xy, qd=qd, currents=cur, feedback_age_ms=age,
                      sigma_min=sig, node_status=status)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()   # TX-free teardown: the drives never hear from us
        log.event("closed (no commands sent)")
        log.close()
        print(f"\nclosed (no commands sent). log: {log.dir}")


if __name__ == "__main__":
    main()
