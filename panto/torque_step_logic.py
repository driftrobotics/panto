"""Pure step-schedule + abort decision logic for scripts/torque_step.py --
split out so it's unit-testable without a CAN bus or sim, same pattern as
panto/breakaway_logic.py.

Schedule: PRE (0.5s at zero torque) -> STEP (``duration_s`` at the commanded
torque) -> POST (0.5s at zero torque) -> DONE. A single, deterministic
function of elapsed time so the control loop just calls it every tick.
"""

from __future__ import annotations

from dataclasses import dataclass

PRE_S = 0.5
POST_S = 0.5


@dataclass(frozen=True)
class Phase:
    name: str          # "pre" | "step" | "post" | "done"
    torque_nm: float


def schedule(elapsed_s: float, duration_s: float, tau_nm: float) -> Phase:
    """Commanded torque at ``elapsed_s`` into the run."""
    if elapsed_s < PRE_S:
        return Phase("pre", 0.0)
    if elapsed_s < PRE_S + duration_s:
        return Phase("step", tau_nm)
    if elapsed_s < PRE_S + duration_s + POST_S:
        return Phase("post", 0.0)
    return Phase("done", 0.0)


def total_duration_s(duration_s: float) -> float:
    return PRE_S + duration_s + POST_S


def check_abort(moved_deg: float, max_deg: float) -> str | None:
    """None if within bounds, else a human-readable abort reason."""
    if abs(moved_deg) >= max_deg:
        return f"moved {moved_deg:.2f}deg exceeds --max-deg={max_deg}"
    return None
