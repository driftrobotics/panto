"""Pure ramp/abort decision logic for scripts/breakaway.py -- separated out so
it can be unit-tested without a CAN bus or sim (see tests/test_breakaway.py).

The measurement: ramp commanded torque linearly from 0 in one direction on one
joint (the other joint held in position mode) until the joint has moved
``break_deg`` from its start -- that's static-friction breakaway. Record the
commanded torque/current at that instant. Three ways this can end early
instead of a clean breakaway, each a distinct, loggable outcome:

  * ``current_limit`` -- current hit the cap before the joint moved
    ``break_deg``. Not a breakaway measurement -- it means the joint is
    torque-stalled against something (friction, a bind, a limit) harder than
    the current cap can push through, which is itself the finding.
  * ``runaway`` -- moved past ``abort_deg`` (should not happen if
    ``break_deg`` < ``abort_deg``, but guards against a control hiccup or an
    already-moving joint at the start of the ramp).
  * ``timeout`` -- neither happened within ``abort_s``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 2026-09-04: a *fixed* torque-mode vel_limit sized the plateau for whatever the
#: default current cap happened to be (e.g. 0.8A -> 1.42A plateau) but silently capped Iq
#: around that same figure in higher-cap runs (2.0A, 2.5A) -- the plateau must scale with
#: the requested current cap, not be a constant. See plateau_vel_limit_rad_s below.
PLATEAU_MARGIN = 1.5        # plateau sized to margin * requested current cap


def plateau_vel_limit_rad_s(current_cap_a: float, torque_constant: float, vel_gain: float,
                            margin: float = PLATEAU_MARGIN) -> tuple[float, float]:
    """ODrive vel_limit (joint rad/s) to set so TORQUE_CONTROL's
    vel_gain*(vel_limit-|vel|) plateau sits `margin`x above `current_cap_a`
    (near zero velocity) -- i.e. the current cap does the limiting, not the
    plateau. Returns (vel_limit_rad_s, plateau_a) where plateau_a is what the
    resulting plateau is in amps (== margin * current_cap_a by construction,
    independent of vel_gain/torque_constant -- they only decide what
    vel_limit achieves it)."""
    vel_limit_turns_s = margin * current_cap_a * torque_constant / vel_gain
    vel_limit_rad_s = vel_limit_turns_s * (2.0 * math.pi)
    plateau_a = margin * current_cap_a
    return vel_limit_rad_s, plateau_a


@dataclass(frozen=True)
class BreakawayResult:
    status: str          # "breakaway" | "current_limit" | "runaway" | "timeout"
    torque_nm: float      # commanded torque at the instant of the result
    current_a: float       # measured |current| at that instant
    moved_deg: float       # signed joint displacement from start at that instant
    elapsed_s: float


def commanded_torque(elapsed_s: float, rate_nm_s: float, sign: int) -> float:
    """Linear ramp: tau(t) = sign * rate * t."""
    return sign * rate_nm_s * elapsed_s


def check_breakaway(elapsed_s: float, moved_deg: float, current_a: float, *,
                    rate_nm_s: float, sign: int, current_cap_a: float,
                    break_deg: float, abort_deg: float, abort_s: float
                    ) -> BreakawayResult | None:
    """Evaluate one tick's state; return a terminal :class:`BreakawayResult`,
    or ``None`` to keep ramping. Order matters: breakaway is checked before
    the current cap so a breakaway that happens to coincide with a
    cap-brushing current tick is still reported as a breakaway, not a
    current-limit stall -- the joint DID move, which is the actual criterion.
    """
    tau = commanded_torque(elapsed_s, rate_nm_s, sign)
    moved = abs(moved_deg)
    if moved >= break_deg:
        return BreakawayResult("breakaway", tau, current_a, moved_deg, elapsed_s)
    if abs(current_a) >= current_cap_a:
        return BreakawayResult("current_limit", tau, current_a, moved_deg, elapsed_s)
    if moved >= abort_deg:
        return BreakawayResult("runaway", tau, current_a, moved_deg, elapsed_s)
    if elapsed_s >= abort_s:
        return BreakawayResult("timeout", tau, current_a, moved_deg, elapsed_s)
    return None
