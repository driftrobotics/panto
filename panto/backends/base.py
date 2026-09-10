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

#: ODrive ``vel_limit`` (turn/s) the backends install alongside the current cap.
#: Not force-relevant — it only bounds runaway — so a fixed generous value is
#: fine until stream D grows a real config field (see report).
DEFAULT_VEL_LIMIT_TURN_S = 20.0


@dataclass(frozen=True)
class ImpedanceCommand:
    pose: np.ndarray          # current EE [x, y] (m)
    q: np.ndarray             # current joint angles [q1, q2] (rad)
    anchor: np.ndarray        # impedance target [x, y] (m)
    stiffness: np.ndarray     # desired EE stiffness, 2x2 (N/m) in world frame
    force_limit: float        # N, after σ_min scaling + I²t cutback
    qd: np.ndarray | None = None  # joint velocities [q1_dot, q2_dot] (rad/s), for
                                   # velocity-scheduled current caps (see
                                   # PositionBackend._current_cap). None -> no
                                   # velocity term (schedule off / unknown).


class ImpedanceBackend(abc.ABC):
    """Common ctor + lifecycle; subclasses implement the command mapping.

    ``enter`` runs once on mode entry (controller mode + limits), ``apply`` every
    tick, ``relax`` drops the interaction force to zero (transparent / fault).
    """

    def __init__(self, link, config) -> None:
        self._link = link          # can_link.CanLink
        self._config = config      # config.Config

    def enter(self) -> None:
        """Put the drives in this backend's control mode. Default: no-op."""

    @abc.abstractmethod
    def apply(self, cmd: ImpedanceCommand) -> None:
        """Render ``cmd`` on the drives for one tick."""

    @abc.abstractmethod
    def relax(self) -> None:
        """Command zero interaction force (transparent mode / fault)."""

    def apply_joint(self, q_target: np.ndarray, cmd: ImpedanceCommand) -> None:
        """Render a *joint-space* target (recorded playback runs on joint angles,
        spec "Deferred / TODO"). ``cmd.anchor`` is already ``forward(q_target)``.
        Default: choose the IK branch from the target itself -- never the mirror
        image of a pose the arm physically visited -- then render as Cartesian.
        Backends that can command joints directly override this."""
        if hasattr(self, "elbow"):
            self.elbow = "down" if float(q_target[1]) < 0.0 else "up"
        self.apply(cmd)
