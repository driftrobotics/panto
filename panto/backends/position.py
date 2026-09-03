"""Position-mode backend (v0).

Per tick:
  1. IK the anchor to joint targets (single elbow branch).
  2. Map the desired EE stiffness to per-joint ``pos_gain``. One scalar gain per
     joint is a joint-space-diagonal stiffness ellipse; it is *not* aligned with
     a diagonal wall. We scale the two gains using J at the current config:
     ``K_q ≈ Jᵀ K_x J``, then take the diagonal → per-joint N·m/rad.
  3. Convert N·m/rad → ODrive ``pos_gain`` per motor with the odrive_knob mapping
     ``pos_gain = k_turn / vel_gain`` (``k_turn = k_rad · 2π``); clamp to
     ``motor.max_pos_gain``.
  4. Push Set_Input_Pos + Set_Pos_Gain over CAN.

Force limiting: convert ``cmd.force_limit`` (N) to a per-motor current cap and
push it via Set_Limits, so pushing past the limit makes the constraint yield
rather than the drive fault. No host-side damping term (stays on the ODrive).
"""

from __future__ import annotations

import numpy as np

from ..kinematics import inverse, jacobian, min_singular_value
from .base import DEFAULT_VEL_LIMIT_TURN_S, ImpedanceBackend, ImpedanceCommand

TWO_PI = 2.0 * np.pi


class PositionBackend(ImpedanceBackend):
    #: what the last `apply()` actually computed/sent, for logging/diagnosis.
    #: `None` until the first `apply()`. See scripts/point_hold.py.
    last_command: dict | None = None

    def enter(self) -> None:
        for motor in self._config.motors:
            self._link.set_controller_mode(motor.node_id, "position")
            self._link.set_limits(
                motor.node_id, DEFAULT_VEL_LIMIT_TURN_S, motor.current_soft_max
            )

    def apply(self, cmd: ImpedanceCommand) -> None:
        geo = self._config.geo
        q_target = inverse(cmd.anchor, geo, elbow=self._config.elbow)

        J = jacobian(cmd.q, geo)
        # Jᵀ K_x J is PSD for PSD K_x; its diagonal is the per-joint stiffness the
        # scalar pos_gain can render. Off-diagonal coupling is dropped — the cost
        # of scalar per-joint gains that the torque backend exists to avoid.
        k_joint = np.diag(J.T @ np.asarray(cmd.stiffness, float) @ J)

        # τ_max the drives may need to hit force_limit in the worst-conditioned
        # direction; floored σ_min keeps it finite near singularities.
        sigma = max(
            min_singular_value(cmd.q, geo),
            float(self._config.sigma_min_threshold),
        )
        tau_max = float(cmd.force_limit) / sigma

        pos_gains = []
        current_caps = []
        for i, motor in enumerate(self._config.motors):
            pos_gain = self._pos_gain(motor, k_joint[i])
            current_cap = self._current_cap(motor, tau_max)
            self._link.set_input_pos(motor.node_id, float(q_target[i]))
            self._link.set_pos_gain(motor.node_id, pos_gain)
            self._link.set_limits(motor.node_id, DEFAULT_VEL_LIMIT_TURN_S, current_cap)
            pos_gains.append(pos_gain)
            current_caps.append(current_cap)

        self.last_command = {
            "q_target": q_target.tolist(),
            "k_joint_nm_rad": k_joint.tolist(),
            "pos_gain": pos_gains,
            "current_cap_a": current_caps,
            "sigma_min": sigma,
            "tau_max_nm": tau_max,
            "vel_limit_turn_s": DEFAULT_VEL_LIMIT_TURN_S,
        }

    def relax(self) -> None:
        # Park each anchor on the current joint angle at zero gain → no torque.
        q, _ = self._link.joint_state()
        for i, motor in enumerate(self._config.motors):
            self._link.set_input_pos(motor.node_id, float(q[i]))
            self._link.set_pos_gain(motor.node_id, 0.0)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _pos_gain(motor, k_rad: float) -> float:
        """N·m/rad → ODrive (turn/s)/turn, via ``pos_gain·vel_gain`` = N·m/turn."""
        vel_gain = getattr(motor, "vel_gain", 0.0)
        if k_rad <= 0.0 or vel_gain <= 0.0:
            return 0.0
        pos_gain = (k_rad * TWO_PI) / vel_gain
        max_pos_gain = getattr(motor, "max_pos_gain", float("inf"))
        return float(min(pos_gain, max_pos_gain))

    @staticmethod
    def _current_cap(motor, tau_max: float) -> float:
        """τ_max (N·m) → Iq cap (A): ``I = τ / Kt``, clamped to current_soft_max."""
        kt = getattr(motor, "torque_constant", 0.035)
        cap = tau_max / kt if kt > 0.0 else motor.current_soft_max
        return float(min(cap, motor.current_soft_max))
