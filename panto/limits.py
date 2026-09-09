"""Joint-limit safety helpers shared by ``CanLink.enter_closed_loop``,
``PositionBackend.apply``, and the bring-up scripts' pre-flight checks.

2026-09-04: a run drove the arm into a mechanical joint stop because nothing
anywhere checked the IK target against the physical range of motion.
``MotorConfig.q_min_rad``/``q_max_rad`` (default +-inf = "unknown, no
limiting") plus ``limit_margin_rad`` (default 5 deg) are the source of truth;
this module is the one place that interprets them so the arm-refusal check,
the runtime clamp, and the scripts' pre-move check all agree on what "outside
the limits" means.

Three checks, three different tightnesses, on purpose:
  * ``armable_range`` / ``check_armable``: [q_min+margin, q_max-margin] -- the
    range a joint must already be inside before we command CLOSED_LOOP_CONTROL.
  * ``clamp_targets``: IK targets get pulled inside the same armable range
    every tick, silently (logged once per anchor, not every tick) -- a target
    a few mm past a limit degrades gracefully into a target at the limit
    rather than being sent to the drive verbatim.
  * ``check_runtime``: a tighter, margin/2 trip wire on the *measured* angle
    while armed -- if the arm gets this close to a limit despite the clamp
    (a fast push, an unmodelled anchor, a fault), stop rendering stiffness
    now rather than wait for the mechanical stop.
"""

from __future__ import annotations

import math

import numpy as np


class JointLimitViolation(RuntimeError):
    """Measured joint angle got within margin/2 of a configured limit."""


def has_limits(motor) -> bool:
    """True if this motor has at least one finite joint limit configured."""
    return motor.q_min_rad > float("-inf") or motor.q_max_rad < float("inf")


def armable_range(motor) -> tuple[float, float]:
    """[q_min+margin, q_max-margin] rad -- required before arming."""
    return motor.q_min_rad + motor.limit_margin_rad, motor.q_max_rad - motor.limit_margin_rad


def check_armable(q, motors) -> list[str]:
    """Return a list of human-readable problems (empty if OK to arm)."""
    problems = []
    for i, m in enumerate(motors):
        if not has_limits(m):
            continue
        lo, hi = armable_range(m)
        if not (lo <= q[i] <= hi):
            problems.append(
                f"node {m.node_id}: q={math.degrees(q[i]):.1f}deg outside armable range "
                f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]deg "
                f"(limits [{_deg(m.q_min_rad)}, {_deg(m.q_max_rad)}]deg, "
                f"margin {math.degrees(m.limit_margin_rad):.1f}deg)"
            )
    return problems


def clamp_targets(q_target, motors) -> np.ndarray:
    """Clamp joint targets to each motor's armable range. No-op for motors
    without configured limits."""
    out = np.array(q_target, dtype=float)
    for i, m in enumerate(motors):
        if not has_limits(m):
            continue
        lo, hi = armable_range(m)
        out[i] = min(max(out[i], lo), hi)
    return out


def check_runtime(q, motors) -> None:
    """Raise ``JointLimitViolation`` if any measured joint is within
    margin/2 of its configured limit. No-op for motors without limits."""
    for i, m in enumerate(motors):
        if not has_limits(m):
            continue
        half = m.limit_margin_rad / 2.0
        if q[i] <= m.q_min_rad + half or q[i] >= m.q_max_rad - half:
            raise JointLimitViolation(
                f"joint {i} (node {m.node_id}) at {math.degrees(q[i]):.1f}deg is within "
                f"{math.degrees(half):.1f}deg of its limit "
                f"[{_deg(m.q_min_rad)}, {_deg(m.q_max_rad)}]deg"
            )


def format_limits_deg(motors) -> str:
    """One-line summary of configured limits, for status logs / meta.json."""
    return " ".join(
        f"node{m.node_id}=[{_deg(m.q_min_rad)},{_deg(m.q_max_rad)}]deg" for m in motors
    )


def q_deg(q) -> list:
    return [round(math.degrees(float(v)), 2) for v in q]


def _deg(rad: float) -> str:
    if rad == float("-inf") or rad == float("inf"):
        return "inf" if rad > 0 else "-inf"
    return f"{math.degrees(rad):.1f}"
