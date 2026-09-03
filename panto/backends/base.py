"""The backend interface.

The runtime computes an :class:`ImpedanceCommand` in end-effector space each tick
and hands it to a backend. The backend owns *only* the translation to ODrive
CANSimple messages and the choice of control mode. It must not read constraints,
modes, or UI state.

Velocity damping is deliberately NOT in the command: it stays local on the ODrive
(``vel_gain``) at 8 kHz. Rendering damping host-side through the ~200 Hz + CAN
delay injects negative damping near 1/(2T) and destabilises. See odrive_knob.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImpedanceCommand:
    pose: np.ndarray          # current EE [x, y] (m)
    q: np.ndarray             # current joint angles [q1, q2] (rad)
    anchor: np.ndarray        # impedance target [x, y] (m)
    stiffness: np.ndarray     # desired EE stiffness, 2x2 (N/m) in world frame
    force_limit: float        # N, after σ_min scaling + I²t cutback


class ImpedanceBackend(abc.ABC):
    @abc.abstractmethod
    def apply(self, cmd: ImpedanceCommand) -> None:
        """Render ``cmd`` on the drives for one tick."""

    @abc.abstractmethod
    def relax(self) -> None:
        """Command zero interaction force (transparent mode / fault)."""
