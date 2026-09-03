"""Closed-loop Cartesian point-hold via the real PositionBackend — step 3a.

    python -m scripts.point_hold                        # hold current pose, 8 s
    python -m scripts.point_hold --step 15              # then jog the anchor +15 mm in x
    python -m scripts.point_hold --current 0.3 --stiffness 400

Exercises the actual `PositionBackend` + `Point` constraint against hardware, in
a bare ~100 Hz loop (no Runtime, no web). Gentle by default: 0.25 A per-axis cap.

  1. read current joint state -> FK pose -> anchor a `Point` there
  2. backend.enter() (position mode + current cap), enter CLOSED_LOOP_CONTROL
  3. loop: project(pose) -> ImpedanceCommand -> backend.apply(); print periodically
  4. optional: after half the hold, jog the anchor by --step mm in +x
  5. backend.relax(), both axes -> IDLE

Any exception / Ctrl-C -> relax + IDLE + close. Push the arm by hand during the
hold to feel the spring; it should pull back toward the anchor.

NOTE: pos_gain is derived as k_rad*2pi / motor.vel_gain, using the *config*
vel_gain (default 2.5e-3, the sane not-buzzy start on EM3215). If the drives'
stored vel_gain differs, the rendered stiffness scales by that ratio — pass
--vel-gain to match, or nail it in the milestone-3 vel_gain sweep. The 0.25 A
cap bounds it either way. Also watch max_pos_gain (500): with vel_gain 2.5e-3 a
modest k_rad already saturates it, so effective stiffness may clamp — raise it
in a config.local.json if the hold is still soft at the current cap.

Every control tick is logged (not just the printed ones) via `panto.telemetry
.RunLogger`, including per-node `axis_state`/`active_errors`/`disarm_reason` —
so a fault or oscillation that happens between printouts is still on disk. This
is the first slice of the "architect a telemetry/logging stack" TODO; see the
Notion progress page.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from panto.backends import PositionBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import CanLink
from panto.config import Config
from panto.constraints import Point
from panto.kinematics import forward, min_singular_value
from panto.telemetry import RunLogger


def main() -> None:
    p = argparse.ArgumentParser(prog="point_hold", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--current", type=float, default=0.25, help="per-axis current cap, A")
    p.add_argument("--stiffness", type=float, default=300.0, help="isotropic EE stiffness, N/m")
    p.add_argument("--vel-gain", type=float, default=None, help="override motor.vel_gain")
    p.add_argument("--hold", type=float, default=8.0, help="seconds")
    p.add_argument("--step", type=float, default=0.0, help="jog anchor +x by this many mm at hold/2")
    p.add_argument("--rate", type=float, default=100.0, help="control loop Hz")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    for m in config.motors:
        m.current_soft_max = args.current
        if args.vel_gain is not None:
            m.vel_gain = args.vel_gain

    nodes = [m.node_id for m in config.motors]
    K = args.stiffness * np.eye(2)

    log = RunLogger("point_hold", interface=config.can.interface, channel=config.can.channel,
                    current=args.current, stiffness=args.stiffness, vel_gain=args.vel_gain,
                    hold=args.hold, step=args.step, rate=args.rate)
    link = CanLink(config, sim=False)
    backend = PositionBackend(link, config)
    print(f"opening {config.can.interface}/{config.can.channel}  "
          f"(cap {args.current} A, K {args.stiffness} N/m, vel_gain {config.motors[0].vel_gain})")
    link.start()

    def idle_all() -> None:
        for nid in nodes:
            try:
                link.set_idle(nid)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {nid}: {exc}", level="ERROR")

    try:
        link.wait_for_feedback(timeout=5.0)
        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        print(f"  start q=({np.degrees(q0[0]):.2f}, {np.degrees(q0[1]):.2f})deg  "
              f"pose=({pose0[0]*1e3:.1f}, {pose0[1]*1e3:.1f})mm")
        constraint = Point(at=pose0.copy())

        backend.enter()
        for i, nid in enumerate(nodes):
            link.set_input_pos(nid, float(q0[i]))
        print(">>> entering CLOSED_LOOP_CONTROL <<<")
        log.event(">>> entering CLOSED_LOOP_CONTROL <<<")
        link.enter_closed_loop(timeout=5.0)
        print("    closed loop; holding\n")

        period = 1.0 / args.rate
        t0 = time.monotonic()
        stepped = False
        next_print = 0.0
        while True:
            t = time.monotonic() - t0
            if t >= args.hold:
                break
            if args.step and not stepped and t >= args.hold / 2:
                tgt = pose0 + np.array([args.step * 1e-3, 0.0])
                constraint = Point(at=tgt)
                stepped = True
                msg = f"anchor -> ({tgt[0]*1e3:.1f}, {tgt[1]*1e3:.1f})mm"
                print(f"  -- {msg} --")
                log.event(msg)

            q, qd = link.joint_state()
            pose = forward(q, config.geo)
            proj = constraint.project(pose)
            cmd = ImpedanceCommand(pose=pose, q=q, anchor=proj.anchor,
                                   stiffness=K, force_limit=5.0)
            backend.apply(cmd)

            err = float(np.linalg.norm(proj.anchor - pose))
            cur = link.motor_currents()
            sig = min_singular_value(q, config.geo)
            status = link.node_status()
            sent = backend.last_command or {}
            # every tick, not just printed ones -- a fault between prints is
            # still on disk, not just visible on the LEDs for a few ms. Also log
            # what we *commanded* (pos_gain, current cap actually sent), not just
            # what we measured -- "it didn't move" is undiagnosable from the
            # measured side alone.
            log.sample(t=t, q=q, qd=qd, pose=pose, anchor=proj.anchor, err_m=err,
                      currents=cur, sigma_min=sig,
                      feedback_age_ms=link.feedback_age_s() * 1e3, node_status=status,
                      sent=sent)

            if t >= next_print:
                estr = "ok" if all(s.active_errors == 0 for s in status) else \
                    " ".join(f"ax{s.node_id}=0x{s.active_errors:x}" for s in status)
                pg = sent.get("pos_gain", ["?", "?"])
                cc = sent.get("current_cap_a", ["?", "?"])
                print(f"  t={t:4.1f}s  pose=({pose[0]*1e3:6.1f},{pose[1]*1e3:6.1f})mm  "
                      f"err={err*1e3:5.2f}mm  i=({cur[0]:+.3f},{cur[1]:+.3f})A  "
                      f"pos_gain=({pg[0]:.0f},{pg[1]:.0f})  cap=({cc[0]:.2f},{cc[1]:.2f})A  "
                      f"sig={sig:.3f}  age={link.feedback_age_s()*1e3:.1f}ms  err={estr}")
                next_print = t + 0.5
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n! interrupted")
        log.event("interrupted", level="WARN")
    except Exception as exc:  # noqa: BLE001
        print(f"\n! {type(exc).__name__}: {exc}")
        log.event(f"{type(exc).__name__}: {exc}", level="ERROR")
    finally:
        print("\nrelax + IDLE")
        try:
            backend.relax()
        except Exception:  # noqa: BLE001
            pass
        idle_all()
        time.sleep(0.2)
        link.close()
        log.close()
        print(f"closed. log: {log.dir}")


if __name__ == "__main__":
    main()
