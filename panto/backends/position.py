"""Position-mode backend (v0).

Per tick:
  1. IK the anchor to joint targets (single elbow branch).
  2. Map the desired EE stiffness to per-joint ``pos_gain``. One scalar gain per
     joint is a joint-space-diagonal stiffness ellipse; it is *not* aligned with
     a diagonal wall. Scale the two gains using J at the contact config to get as
     close as the position loop allows (K_q ≈ Jᵀ K_x J, then take the diagonal).
  3. Push Set_Input_Pos + Set_Pos_Gain over CAN.

The ODrive renders the spring at 8 kHz; we only move the anchor at ~200 Hz.
Endstop / force limiting: cap current via Set_Limits, or detune gain, so pushing
past the limit makes the constraint yield rather than the drive fault.
"""

from __future__ import annotations

from .base import ImpedanceBackend, ImpedanceCommand


class PositionBackend(ImpedanceBackend):
    def __init__(self, link, geo):
        self._link = link      # can_link.CanLink
        self._geo = geo        # kinematics.LinkGeometry

    def apply(self, cmd: ImpedanceCommand) -> None:
        raise NotImplementedError

    def relax(self) -> None:
        # Park each anchor on the current angle at ~zero gain, or drop to idle.
        raise NotImplementedError
