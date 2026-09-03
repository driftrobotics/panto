"""Torque-mode backend (milestone 6).

Per tick:
  1. F = K_x · (anchor - pose)          (world frame, N)
  2. clamp |F| to cmd.force_limit
  3. tau = Jᵀ · F                        (N·m per joint)
  4. Set_Input_Torque over CAN

Gives a true Cartesian 2x2 stiffness — "stiff into the wall, free along it" for
walls at any angle, which the position backend cannot do with scalar per-joint
gains. Cost: the spring now runs at the ~200 Hz host rate + CAN delay instead of
8 kHz on the ODrive. Velocity damping still stays local (vel_gain); do not add
host-side damping here.
"""

from __future__ import annotations

from .base import ImpedanceBackend, ImpedanceCommand


class TorqueBackend(ImpedanceBackend):
    def __init__(self, link, geo):
        self._link = link
        self._geo = geo

    def apply(self, cmd: ImpedanceCommand) -> None:
        raise NotImplementedError

    def relax(self) -> None:
        raise NotImplementedError
