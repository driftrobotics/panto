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

import logging
import math

import numpy as np

from ..kinematics import inverse, jacobian, min_singular_value
from ..limits import check_runtime, clamp_targets
from .base import DEFAULT_VEL_LIMIT_TURN_S, ImpedanceBackend, ImpedanceCommand

TWO_PI = 2.0 * np.pi

#: below this |q_target - q| (rad), don't feed forward any Coulomb friction
#: torque -- avoids dithering the output at zero commanded motion (a fixed
#: tau_ff right at the target would just push the joint past it and back).
FF_DEADBAND_RAD = math.radians(0.5)

log = logging.getLogger(__name__)


class PositionBackend(ImpedanceBackend):
    #: what the last `apply()` actually computed/sent, for logging/diagnosis.
    #: `None` until the first `apply()`. See scripts/point_hold.py.
    last_command: dict | None = None

    #: ODrive vel_limit, joint rad/s. Public so callers (scripts/*.py) can
    #: override it for a sweep without threading a new ctor arg through;
    #: defaults to the historical fixed value.
    vel_limit_rad_s: float = DEFAULT_VEL_LIMIT_TURN_S

    #: anchor (rounded) last logged for an IK-target clamp, so the WARN fires
    #: once per anchor change rather than every tick it stays clamped.
    _last_clamp_anchor: tuple | None = None

    #: elbow branch actually used by apply()'s IK, chosen once in enter() from
    #: the *measured* q1 sign so it always matches how the arm is physically
    #: assembled/calibrated -- config.elbow is just the fallback for sim/tests
    #: that never call enter() (e.g. constructing a backend and calling
    #: apply() directly). See enter() below.
    elbow: str | None = None

    def enter(self) -> None:
        for motor in self._config.motors:
            self._link.set_controller_mode(motor.node_id, "position")
            self._link.set_limits(
                motor.node_id, self.vel_limit_rad_s, motor.current_soft_max
            )
        # Pick the elbow branch that matches how the arm is actually sitting
        # right now (q1 < 0 -> "down", q1 > 0 -> "up") and keep it for the
        # session -- IK must never solve for the mirror-image branch of a
        # physically calibrated arm.
        q, _ = self._link.joint_state()
        self.elbow = "down" if q[1] < 0.0 else "up"
        log.info("elbow branch selected: %s (measured q1=%.2fdeg)",
                 self.elbow, np.degrees(q[1]))

    def apply(self, cmd: ImpedanceCommand) -> None:
        geo = self._config.geo
        elbow = self.elbow if self.elbow is not None else self._config.elbow
        q_target_ik = inverse(cmd.anchor, geo, elbow=elbow)
        q_target = clamp_targets(q_target_ik, self._config.motors)
        if not np.allclose(q_target, q_target_ik):
            anchor_key = tuple(np.round(np.asarray(cmd.anchor, float), 6))
            if anchor_key != self._last_clamp_anchor:
                self._last_clamp_anchor = anchor_key
                log.warning(
                    "anchor (%.4f, %.4f): IK target clamped to joint limits %s -> %s",
                    cmd.anchor[0], cmd.anchor[1],
                    q_target_ik.tolist(), q_target.tolist(),
                )
        else:
            self._last_clamp_anchor = None

        # Measured q within margin/2 of a configured limit while armed -- stop
        # rendering stiffness now rather than let it reach the mechanical stop.
        check_runtime(cmd.q, self._config.motors)

        # Jᵀ K_x J is PSD for PSD K_x; its diagonal is the per-joint stiffness the
        # scalar pos_gain can render. Off-diagonal coupling is dropped — the cost
        # of scalar per-joint gains that the torque backend exists to avoid.
        k_joint = self._k_joint(cmd.q, cmd.stiffness, geo)

        # τ_max the drives may need to hit force_limit in the worst-conditioned
        # direction; floored σ_min keeps it finite near singularities.
        sigma = max(
            min_singular_value(cmd.q, geo),
            float(self._config.sigma_min_threshold),
        )
        tau_max = float(cmd.force_limit) / sigma

        qd = cmd.qd
        pos_gains = []
        current_caps = []
        vel_caps = []
        tau_ffs = []
        for i, motor in enumerate(self._config.motors):
            pos_gain = self._pos_gain(motor, k_joint[i])
            force_cap = self._current_cap(motor, tau_max)
            qd_i = float(qd[i]) if qd is not None else 0.0
            vel_cap = self._vel_scheduled_cap(motor, qd_i)
            current_cap = min(force_cap, vel_cap)
            tau_ff = self._coulomb_ff(motor, float(q_target[i]) - float(cmd.q[i]))
            self._link.set_input_pos(motor.node_id, float(q_target[i]), torque_ff_nm=tau_ff)
            self._link.set_pos_gain(motor.node_id, pos_gain)
            self._link.set_limits(motor.node_id, self.vel_limit_rad_s, current_cap)
            pos_gains.append(pos_gain)
            current_caps.append(current_cap)
            vel_caps.append(vel_cap)
            tau_ffs.append(tau_ff)

        self.last_command = {
            "q_target": q_target.tolist(),
            "k_joint_nm_rad": k_joint.tolist(),
            "pos_gain": pos_gains,
            "current_cap_a": current_caps,
            "vel_scheduled_cap_a": vel_caps,
            "tau_ff_nm": tau_ffs,
            "sigma_min": sigma,
            "tau_max_nm": tau_max,
            "vel_limit_turn_s": self.vel_limit_rad_s,
        }

    def relax(self) -> None:
        # Park each anchor on the current joint angle at zero gain → no torque.
        q, _ = self._link.joint_state()
        for i, motor in enumerate(self._config.motors):
            self._link.set_input_pos(motor.node_id, float(q[i]))
            self._link.set_pos_gain(motor.node_id, 0.0)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _k_joint(q, stiffness, geo) -> np.ndarray:
        """Per-joint stiffness (N·m/rad) apply() would render for EE stiffness
        matrix ``stiffness`` (N/m) at joint config ``q``: diag(Jᵀ K_x J)."""
        J = jacobian(np.asarray(q, float), geo)
        return np.diag(J.T @ np.asarray(stiffness, float) @ J)

    @classmethod
    def pos_gains_for(cls, stiffness, q, config) -> list[float]:
        """Per-joint ODrive ``pos_gain`` apply() would actually send for EE
        stiffness ``stiffness`` (N/m) at joint config ``q``, given
        ``config.motors``/``config.geo``. Pure function factored out of
        apply() so pre-flight prints show the real value instead of a naive
        (non-Jacobian-scaled) approximation -- see scripts/*.py."""
        k_joint = cls._k_joint(q, stiffness, config.geo)
        return [cls._pos_gain(m, k_joint[i]) for i, m in enumerate(config.motors)]

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

    #: cap values are quantised to this step before being sent, so tiny qd
    #: jitter near the schedule's operating point doesn't spam Set_Limits at
    #: full loop rate (this backend already sends Set_Limits every apply()
    #: regardless -- see the vel_cap note below -- but bus-bandwidth-wise a
    #: quantised value is friendlier to read off the wire/logs).
    CAP_QUANT_A = 0.02

    @classmethod
    def _vel_scheduled_cap(cls, motor, qd_rad_s: float) -> float:
        """Host-side soft saturation: ``cap = clamp(cap_max - k_v*|qd|, cap_min,
        cap_max)``. 2026-09-04 finding: at a fixed current cap, the saturated
        position cascade is an undamped relay oscillation whose amplitude
        scales with the cap -- letting the cap fall as the joint moves faster
        (i.e. exactly when/where saturation happens) gives the loop somewhere
        to bleed energy instead of bouncing off the cap at full authority.
        ``cap_vel_slope_a_per_rad_s`` == 0 (default) reproduces the old
        constant-cap behaviour exactly (returns ``current_soft_max``)."""
        cap_max = motor.current_soft_max
        slope = getattr(motor, "cap_vel_slope_a_per_rad_s", 0.0)
        if slope <= 0.0:
            return float(cap_max)
        cap_min = getattr(motor, "cap_min_a", 0.5)
        cap = cap_max - slope * abs(qd_rad_s)
        cap = min(cap_max, max(cap_min, cap))
        return round(cap / cls.CAP_QUANT_A) * cls.CAP_QUANT_A

    @staticmethod
    def _coulomb_ff(motor, q_err_rad: float) -> float:
        """Coulomb friction feedforward, joint-frame N.m, sent as Set_Input_Pos's
        Torque_FF. ``q_err_rad`` is q_target - q_measured; its *sign* is the
        direction the joint needs to move to reach the target, which is the
        direction breakaway torque should be fed forward in (helping the move,
        not opposing it -- this offsets static friction, it isn't a damping
        term). Zero inside +-FF_DEADBAND_RAD so a joint sitting on target
        doesn't get a step-function torque dithering it around the setpoint.
        ``coulomb_pos_nm``/``coulomb_neg_nm`` default 0 -> always returns 0,
        i.e. old (no-feedforward) behaviour is exact when unset."""
        if abs(q_err_rad) < FF_DEADBAND_RAD:
            return 0.0
        ff_scale = getattr(motor, "ff_scale", 0.0)
        if q_err_rad > 0:
            return ff_scale * getattr(motor, "coulomb_pos_nm", 0.0)
        return -ff_scale * getattr(motor, "coulomb_neg_nm", 0.0)
