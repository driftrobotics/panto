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

import numpy as np

from ..kinematics import jacobian
from .base import DEFAULT_VEL_LIMIT_TURN_S, ImpedanceBackend, ImpedanceCommand


class TorqueBackend(ImpedanceBackend):
    def enter(self) -> None:
        for motor in self._config.motors:
            self._link.set_controller_mode(motor.node_id, "torque")
            self._link.set_limits(
                motor.node_id, DEFAULT_VEL_LIMIT_TURN_S, motor.current_soft_max
            )

    def apply(self, cmd: ImpedanceCommand) -> None:
        F = np.asarray(cmd.stiffness, float) @ (
            np.asarray(cmd.anchor, float) - np.asarray(cmd.pose, float)
        )
        mag = float(np.linalg.norm(F))
        if mag > cmd.force_limit > 0.0:
            F = F * (cmd.force_limit / mag)

        tau = jacobian(cmd.q, self._config.geo).T @ F
        for i, motor in enumerate(self._config.motors):
            self._link.set_input_torque(motor.node_id, float(tau[i]))

    def relax(self) -> None:
        for motor in self._config.motors:
            self._link.set_input_torque(motor.node_id, 0.0)
