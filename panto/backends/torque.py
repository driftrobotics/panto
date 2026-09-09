"""Torque-mode backend: host-side Cartesian impedance, tau = J^T . F.

Per tick:
  1. v_ee = J(q) . qd_filtered      (world frame, m/s) -- damping term only,
                                     the spring term uses pose/anchor raw.
  2. F = K . (anchor - pose) - B . v_ee
  3. clamp |F| to cmd.force_limit
  4. tau = J^T . F                  (N.m per joint, joint frame)
  5. clamp |tau_i| to the per-joint torque cap (current_soft_max * Kt)
  6. slew-limit tau_i vs. last tick's commanded value
  7. Set_Input_Torque over CAN (CanLink applies flip + the CAN encoding;
     joint-frame N.m in, no torque_constant conversion needed here -- the
     drive's own Kt config does A<->N.m on-board)

Gives a true Cartesian 2x2 stiffness -- "stiff into the wall, free along it"
for walls at any angle, which the position backend cannot do with scalar
per-joint gains. Cost: the spring now runs at the host loop rate (~200-500 Hz)
+ CAN delay instead of 8 kHz on the ODrive, so unlike PositionBackend this
backend *does* add host-side damping (``damping``, N.s/m) -- see the module
docstring in backends/base.py for why that's normally avoided; the torque
path has no local (ODrive-side) damping term to lean on instead, so leaving
damping out here means an undamped spring at CAN-delay + host-loop latency,
which rings even harder than adding host damping does.

Velocity source: ``cmd.qd``, i.e. whatever ``CanLink.joint_state()`` last
decoded from Get_Encoder_Estimates -- the drive's own encoder-PLL velocity
estimate (bandwidth set by ``encoder_bandwidth``, ~100 Hz on this rig),
broadcast at the drive's encoder message rate (2 ms / 500 Hz). It is NOT
independently estimated host-side. That estimate is noisy enough (12-bit
encoder underneath the PLL) that using it raw for B*v damping injects noise
into the torque command -- ``vel_lpf_hz`` (default 20 Hz, one-pole) filters
it first; an optional ``notch_hz``/``notch_q`` can additionally null out a
known structural mode (e.g. the ~12 Hz lightly-damped elbow mode from the
free-air stiffness investigation) before it reaches the damping term.

Torque-mode vel_limit plateau: TORQUE_CONTROL on the ODrive still clamps
*effective* torque to ``vel_gain*(vel_limit-|vel|)`` (see
``panto.breakaway_logic``'s note) even though vel_gain/vel_limit are nominally
position-loop parameters. ``enter()`` raises ``vel_gain`` to
``torque_vel_gain`` and sizes ``vel_limit`` via
``plateau_vel_limit_rad_s`` so that plateau sits comfortably above the
current cap -- otherwise the plateau, not the current cap, silently limits
authority (2026-09-04 incident: 30 mN.m commanded, 0.17 A drawn).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..breakaway_logic import plateau_vel_limit_rad_s
from ..filters import Notch, OnePoleLowPass
from ..kinematics import jacobian
from ..limits import check_runtime
from .base import ImpedanceBackend, ImpedanceCommand

log = logging.getLogger(__name__)


class TorqueBackend(ImpedanceBackend):
    #: ODrive vel_gain pushed on enter() (both nodes) to raise the
    #: torque-mode vel_limit plateau above the current cap -- see module
    #: docstring. Callers that want the pre-2026-09 default back can set this
    #: to the config motor's own vel_gain before calling enter(), but that
    #: reintroduces the plateau-caps-authority failure mode.
    torque_vel_gain: float = 0.01

    #: tip damping, N.s/m, isotropic, world frame: F_damp = -damping * v_ee.
    #: 0 (default) = no host-side damping (matches the pre-impedance-step
    #: TorqueBackend exactly).
    damping: float = 0.0

    #: one-pole low-pass cutoff (Hz) applied to each joint's qd before it's
    #: used for the damping term. <=0 disables filtering.
    vel_lpf_hz: float = 20.0

    #: optional notch on the (already low-passed) joint velocity; 0 = off.
    notch_hz: float = 0.0
    notch_q: float = 4.0

    #: per-joint torque slew limit, N.m/s. <=0 = off (no slew limiting).
    slew_nm_s: float = 0.0

    #: what the last apply() actually computed/sent, for logging -- mirrors
    #: PositionBackend.last_command. None until the first apply().
    last_command: dict | None = None

    def __init__(self, link, config) -> None:
        super().__init__(link, config)
        self._reset_filters()

    def _reset_filters(self) -> None:
        n = len(self._config.motors)
        self._lpf = [OnePoleLowPass(self.vel_lpf_hz) for _ in range(n)]
        self._notch = [Notch(self.notch_hz, self.notch_q) for _ in range(n)]
        self._last_tau = np.zeros(n)
        self._last_t: float | None = None

    def enter(self) -> None:
        self._reset_filters()
        for motor in self._config.motors:
            self._link.set_controller_mode(motor.node_id, "torque")
            self._link.set_vel_gains(motor.node_id, self.torque_vel_gain, 0.0)
            vel_limit_rad_s, plateau_a = plateau_vel_limit_rad_s(
                motor.current_soft_max, motor.torque_constant, self.torque_vel_gain
            )
            self._link.set_limits(motor.node_id, vel_limit_rad_s, motor.current_soft_max)
            log.info(
                "node %d: torque mode, vel_gain=%.4g -> plateau=%.3fA (cap=%.3fA, "
                "vel_limit=%.3frad/s)",
                motor.node_id, self.torque_vel_gain, plateau_a,
                motor.current_soft_max, vel_limit_rad_s,
            )

    def apply(self, cmd: ImpedanceCommand) -> None:
        check_runtime(cmd.q, self._config.motors)

        now = time.monotonic()
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now

        n = len(self._config.motors)
        qd_raw = np.asarray(cmd.qd, float) if cmd.qd is not None else np.zeros(n)
        qd_filt = np.array(
            [self._notch[i].update(self._lpf[i].update(float(qd_raw[i]), dt), dt)
             for i in range(n)]
        )

        J = jacobian(np.asarray(cmd.q, float), self._config.geo)
        v_ee = J @ qd_filt

        F = np.asarray(cmd.stiffness, float) @ (
            np.asarray(cmd.anchor, float) - np.asarray(cmd.pose, float)
        ) - self.damping * v_ee

        mag = float(np.linalg.norm(F))
        if mag > cmd.force_limit > 0.0:
            F = F * (cmd.force_limit / mag)

        tau = J.T @ F

        tau_caps = np.array([m.current_soft_max * m.torque_constant for m in self._config.motors])
        tau = np.clip(tau, -tau_caps, tau_caps)

        if self.slew_nm_s > 0.0 and dt > 0.0:
            step = self.slew_nm_s * dt
            tau = np.clip(tau, self._last_tau - step, self._last_tau + step)
        self._last_tau = tau

        for i, motor in enumerate(self._config.motors):
            self._link.set_input_torque(motor.node_id, float(tau[i]))

        self.last_command = {
            "F_n": F.tolist(),
            "tau_nm": tau.tolist(),
            "tau_cap_nm": tau_caps.tolist(),
            "qd_filtered": qd_filt.tolist(),
            "v_ee_m_s": v_ee.tolist(),
            "dt_s": dt,
        }

    def relax(self) -> None:
        for motor in self._config.motors:
            self._link.set_input_torque(motor.node_id, 0.0)
        self._last_tau = np.zeros(len(self._config.motors))
