"""Symmetric +/-x, +/-y offset-hold sweep — averaged authority measurement.

    python -m scripts.offset_sweep                         # 0.4 A, 200 N/m, +-15mm
    python -m scripts.offset_sweep --current 0.5 --stiffness 250 --step 20

For each of +x, -x, +y, -y (in that order): step the anchor `--step` mm off the
start pose, hold `--settle` s to reach steady state, record the settled error
and current, then return to the start pose and re-settle before the next
direction. Reports per-direction results plus the average and a +/- symmetry
check, rather than a single anecdotal direction.

Every tick is logged via panto.telemetry.RunLogger (pose, anchor, error,
currents, node_status, and PositionBackend.last_command — what was actually
sent, not just measured). Gentle defaults; current-capped, no I2t protection in
this script (bypasses Runtime), so keep --current modest for multi-direction
runs. relax + IDLE on any exit path.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from panto.backends import PositionBackend
from panto.backends.base import ImpedanceCommand
from panto.can_link import CanLink
from panto.config import Config
from panto.constraints import Point
from panto.kinematics import Unreachable, forward, min_singular_value
from panto.telemetry import RunLogger

DIRECTIONS = {
    "+x": np.array([1.0, 0.0]),
    "-x": np.array([-1.0, 0.0]),
    "+y": np.array([0.0, 1.0]),
    "-y": np.array([0.0, -1.0]),
}


@dataclass
class DirResult:
    name: str
    settle_err_mm: float
    settle_cur: tuple
    peak_cur: tuple


def _run_to(link, backend, config, log, K, force_limit, target, seconds, rate_hz, tag):
    """Drive toward `target` for `seconds`; return (final_err_m, final_cur, peak_cur)."""
    period = 1.0 / rate_hz
    constraint = Point(at=target)
    t_end = time.monotonic() + seconds
    peak = np.zeros(2)
    err = 0.0
    cur = np.zeros(2)
    while time.monotonic() < t_end:
        q, qd = link.joint_state()
        pose = forward(q, config.geo)
        proj = constraint.project(pose)
        cmd = ImpedanceCommand(pose=pose, q=q, anchor=proj.anchor, stiffness=K,
                               force_limit=force_limit)
        backend.apply(cmd)
        err = float(np.linalg.norm(proj.anchor - pose))
        cur = link.motor_currents()
        peak = np.maximum(peak, np.abs(cur))
        status = link.node_status()
        log.sample(tag=tag, q=q, pose=pose, anchor=proj.anchor, err_m=err,
                  currents=cur, node_status=status, sent=backend.last_command or {})
        time.sleep(period)
    return err, cur, peak


def main() -> None:
    p = argparse.ArgumentParser(prog="offset_sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--current", type=float, default=0.4, help="per-axis current cap, A")
    p.add_argument("--stiffness", type=float, default=200.0, help="isotropic EE stiffness, N/m")
    p.add_argument("--step", type=float, default=15.0, help="offset magnitude, mm")
    p.add_argument("--settle", type=float, default=3.0, help="seconds to settle at each offset")
    p.add_argument("--return-settle", type=float, default=2.0,
                   help="seconds to re-settle at the start pose between directions")
    p.add_argument("--rate", type=float, default=100.0, help="control loop Hz")
    args = p.parse_args()

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel
    for m in config.motors:
        m.current_soft_max = args.current

    K = args.stiffness * np.eye(2)
    force_limit = 5.0  # current_soft_max is the real limit; see point_hold.py

    log = RunLogger("offset_sweep", interface=config.can.interface, channel=config.can.channel,
                    current=args.current, stiffness=args.stiffness, step_mm=args.step,
                    settle=args.settle, return_settle=args.return_settle, rate=args.rate,
                    vel_gain=config.motors[0].vel_gain)
    link = CanLink(config, sim=False)
    backend = PositionBackend(link, config)
    print(f"opening {config.can.interface}/{config.can.channel}  "
          f"(cap {args.current} A, K {args.stiffness} N/m, vel_gain {config.motors[0].vel_gain}, "
          f"step {args.step} mm)")
    link.start()

    def idle_all() -> None:
        for m in config.motors:
            try:
                link.set_idle(m.node_id)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {m.node_id}: {exc}", level="ERROR")

    results: list[DirResult] = []
    try:
        link.wait_for_feedback(timeout=5.0)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        sig0 = min_singular_value(q0, config.geo)
        print(f"  start pose=({pose0[0]*1e3:.1f}, {pose0[1]*1e3:.1f})mm  sigma_min={sig0:.4f}")

        backend.enter()
        for i, m in enumerate(config.motors):
            link.set_input_pos(m.node_id, float(q0[i]))
        print(">>> entering CLOSED_LOOP_CONTROL <<<")
        log.event(">>> entering CLOSED_LOOP_CONTROL <<<")
        link.enter_closed_loop(timeout=5.0)

        if sig0 < config.sigma_min_threshold:
            # Too close to a singularity (e.g. sitting near full extension) to
            # step +/-x/y meaningfully -- every direction risks Unreachable and
            # force authority there is ~0 anyway. Pull inward along the same
            # ray to a better-conditioned pose before centring the sweep on it.
            inward = pose0 * 0.7
            msg = (f"start sigma_min={sig0:.4f} < threshold "
                  f"{config.sigma_min_threshold:.4f} -- pulling in to "
                  f"({inward[0]*1e3:.1f},{inward[1]*1e3:.1f})mm before sweeping")
            print(f"  ! {msg}")
            log.event(msg, level="WARN")
            _run_to(link, backend, config, log, K, force_limit, inward, 3.0, args.rate, "recentre")
            q0, _ = link.joint_state()
            pose0 = forward(q0, config.geo)
            sig0 = min_singular_value(q0, config.geo)
            print(f"  new centre pose=({pose0[0]*1e3:.1f}, {pose0[1]*1e3:.1f})mm  "
                  f"sigma_min={sig0:.4f}")

        print(f"    closed loop. Sweeping {list(DIRECTIONS)}\n")

        # settle at the start pose first
        _run_to(link, backend, config, log, K, force_limit, pose0, 1.0, args.rate, "center0")

        for name, unit in DIRECTIONS.items():
            target = pose0 + unit * (args.step * 1e-3)
            log.event(f"-> {name}: target=({target[0]*1e3:.1f},{target[1]*1e3:.1f})mm")
            try:
                err, cur, peak = _run_to(link, backend, config, log, K, force_limit, target,
                                          args.settle, args.rate, f"offset_{name}")
            except Unreachable as exc:
                print(f"  {name:>3}  UNREACHABLE from this pose ({exc}) -- skipped")
                log.event(f"{name} unreachable: {exc}", level="WARN")
                continue
            results.append(DirResult(name, err * 1e3, tuple(cur), tuple(peak)))
            print(f"  {name:>3}  settle_err={err*1e3:6.2f}mm  "
                  f"cur=({cur[0]:+.3f},{cur[1]:+.3f})A  peak=({peak[0]:.3f},{peak[1]:.3f})A")
            # return to centre before the next direction
            _run_to(link, backend, config, log, K, force_limit, pose0,
                    args.return_settle, args.rate, f"return_{name}")

        if not results:
            raise SystemExit("every direction was unreachable from this pose")

        by_name = {r.name: r for r in results}
        avg_err = np.mean([r.settle_err_mm for r in results])
        avg_cur_mag = np.mean([np.linalg.norm(r.settle_cur) for r in results])
        print(f"\n  avg settle_err = {avg_err:.2f} mm   avg |current| = {avg_cur_mag:.3f} A  "
              f"(n={len(results)}/4 directions)")
        for a, b in (("+x", "-x"), ("+y", "-y")):
            if a in by_name and b in by_name:
                asym = by_name[a].settle_err_mm - by_name[b].settle_err_mm
                print(f"  {a} - {b} = {asym:+.2f} mm  (symmetry check)")
                log.event(f"symmetry {a}-{b} = {asym:.3f}mm")
        log.event(f"summary: avg_err_mm={avg_err:.3f} avg_cur_a={avg_cur_mag:.3f} "
                 f"n={len(results)}/4")
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
