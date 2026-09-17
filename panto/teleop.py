"""Force-feedback teleop logic: panto (leader) j0/j1 -> i2rt YAM (follower) J1/J2.

Pure numpy, no CAN, no i2rt import -- ``scripts/teleop_yam.py`` owns both buses
and calls into this. Scheme (per joint, i in {0, 1}):

  follower   q_yam_cmd[i] = clamp(q_yam0[i] + scale[i] * (q_panto[i] - q_panto0[i]))
             rate-limited, rendered by the DM drives' own MIT PD (kp/kd) on top
             of i2rt's gravity compensation.
  leader     position spring toward the mapped-back *measured* YAM angle (the
             position-position coupling, as i2rt's own bilateral gello) plus a
             reflected torque  -alpha * scale * lowpass(deadband(tau_ext)),
             tau_ext = joint_eff - feedforward (gravity comp) torque.

Relative mapping: both robots' poses at engage time are the correspondence
origin, so engaging never commands a jump.

``TeleopMonitor`` is the fault policy: any tripped check names itself in the
``drive:``/``loop:`` style of runtime telemetry and the script e-stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointMap:
    """Relative leader->follower joint map with a follower-side box clamp."""

    scale: np.ndarray            # follower rad per leader rad, signed, shape (2,)
    leader_zero: np.ndarray      # leader q at engage (rad)
    follower_zero: np.ndarray    # follower q at engage (rad)
    follower_lo: np.ndarray      # box clamp (rad), already inside the joint limits
    follower_hi: np.ndarray

    def __post_init__(self) -> None:
        self.scale = np.asarray(self.scale, float)
        if np.any(self.scale == 0.0):
            raise ValueError("JointMap scale must be non-zero")
        for name in ("leader_zero", "follower_zero", "follower_lo", "follower_hi"):
            setattr(self, name, np.asarray(getattr(self, name), float))
        if np.any(self.follower_lo > self.follower_hi):
            raise ValueError("JointMap box is inverted")
        if np.any(self.follower_zero < self.follower_lo) or np.any(self.follower_zero > self.follower_hi):
            raise ValueError("follower engage pose is outside the follower box")

    def to_follower(self, q_leader) -> tuple[np.ndarray, np.ndarray]:
        """-> (clamped follower target, bool[2] which joints hit the box)."""
        raw = self.follower_zero + self.scale * (np.asarray(q_leader, float) - self.leader_zero)
        clamped = np.clip(raw, self.follower_lo, self.follower_hi)
        return clamped, clamped != raw

    def to_leader(self, q_follower) -> np.ndarray:
        return self.leader_zero + (np.asarray(q_follower, float) - self.follower_zero) / self.scale


class RateLimiter:
    """Per-joint slew limit on the follower target (rad/s)."""

    def __init__(self, max_rate_rad_s: float, start) -> None:
        self._max = float(max_rate_rad_s)
        self._value = np.asarray(start, float).copy()

    def step(self, target, dt: float) -> np.ndarray:
        lim = self._max * max(dt, 0.0)
        self._value += np.clip(np.asarray(target, float) - self._value, -lim, lim)
        return self._value.copy()


class EffortReflector:
    """Follower external torque (N.m) -> leader joint torque (N.m).

    deadband -> one-pole low-pass -> ``-alpha * sign(scale)`` -> clamp. The sign
    makes the reflected torque oppose the leader motion that loads the follower.
    ``gain`` ramps 0->1 over ``ramp_s`` after engage so reflection fades in.
    """

    def __init__(self, alpha, scale, *, cutoff_hz: float, deadband_nm, tau_max_nm: float,
                 ramp_s: float = 1.0) -> None:
        self._alpha = np.broadcast_to(np.asarray(alpha, float), (2,)).copy()
        self._sign = np.sign(np.asarray(scale, float))
        self._cutoff = float(cutoff_hz)
        self._deadband = np.broadcast_to(np.asarray(deadband_nm, float), (2,)).copy()
        self._tau_max = float(tau_max_nm)
        self._ramp_s = float(ramp_s)
        self._t = 0.0
        self.filtered = np.zeros(2)

    def step(self, tau_ext, dt: float) -> np.ndarray:
        tau_ext = np.asarray(tau_ext, float)
        dead = np.sign(tau_ext) * np.maximum(np.abs(tau_ext) - self._deadband, 0.0)
        a = 1.0 - np.exp(-2.0 * np.pi * self._cutoff * max(dt, 0.0))
        self.filtered += a * (dead - self.filtered)
        self._t += max(dt, 0.0)
        gain = min(1.0, self._t / self._ramp_s) if self._ramp_s > 0 else 1.0
        out = -gain * self._alpha * self._sign * self.filtered
        return np.clip(out, -self._tau_max, self._tau_max)


class BuzzDetector:
    """Leader oscillation detector that tolerates deliberate hand motion (the
    pose-std OscillationGuard does not: a moving leader *is* pose variance).
    Works on joint *position* with hysteresis -- the drives' velocity estimate
    is too noisy (+-0.4 rad/s of jitter on a hand-held, 0.1 deg-still arm). A
    reversal counts only after the joint has swung >= ``min_swing_rad`` from
    its last turning point; ``flips`` of them within ``window_s`` trips, i.e.
    a sustained >= flips/(2*window) Hz oscillation of real amplitude."""

    def __init__(self, window_s: float = 0.5, flips: int = 6, min_swing_rad: float = np.radians(0.75)) -> None:
        self._window, self._flips, self._swing = float(window_s), int(flips), float(min_swing_rad)
        self._extreme: np.ndarray | None = None
        self._dir = np.zeros(2)
        self._events: list[list[float]] = [[], []]

    def step(self, t: float, q) -> str | None:
        q = np.asarray(q, float)
        if self._extreme is None:
            self._extreme = q.copy()
            return None
        for j in range(2):
            delta = q[j] - self._extreme[j]
            if self._dir[j] == 0:
                if abs(delta) >= self._swing:
                    self._dir[j], self._extreme[j] = np.sign(delta), q[j]
            elif delta * self._dir[j] > 0:
                self._extreme[j] = q[j]                       # still travelling: move the turning point
            elif abs(delta) >= self._swing:
                self._dir[j], self._extreme[j] = -self._dir[j], q[j]
                self._events[j].append(t)
            self._events[j] = [e for e in self._events[j] if e >= t - self._window]
            if len(self._events[j]) >= self._flips:
                return f"joint{j}:{len(self._events[j])}_reversals_in_{self._window}s"
        return None


@dataclass
class TeleopLimits:
    """Fault thresholds. Defaults are deliberately tight for a first POC."""

    follower_err_rad: float = 0.35      # |cmd - measured| on J1/J2: follower is blocked hard / lost
    follower_vel_rad_s: float = 2.0     # J1/J2 measured speed
    follower_eff_nm: float = 8.0        # |tau_ext| on J1/J2
    held_drift_rad: float = 0.15        # J3..J6 moved away from their hold pose
    leader_age_s: float = 0.1           # panto feedback staleness
    follower_age_s: float = 0.1         # YAM observation unchanged for this long
    loop_overrun_s: float = 0.05        # one tick took this long


@dataclass
class TeleopMonitor:
    """Latches the first fault; ``check`` returns its name or None."""

    limits: TeleopLimits = field(default_factory=TeleopLimits)
    fault: str | None = None

    def check(self, *, q_cmd, q_meas, qd_meas, tau_ext, held_err, leader_age_s: float,
              follower_age_s: float, tick_s: float) -> str | None:
        if self.fault is not None:
            return self.fault
        lim = self.limits
        checks = (
            ("follower_tracking", float(np.max(np.abs(np.asarray(q_cmd) - np.asarray(q_meas)))),
             lim.follower_err_rad),
            ("follower_overspeed", float(np.max(np.abs(qd_meas))), lim.follower_vel_rad_s),
            ("follower_over_effort", float(np.max(np.abs(tau_ext))), lim.follower_eff_nm),
            ("held_joint_drift", float(np.max(np.abs(held_err))) if len(held_err) else 0.0,
             lim.held_drift_rad),
            ("leader_stale", float(leader_age_s), lim.leader_age_s),
            ("follower_stale", float(follower_age_s), lim.follower_age_s),
            ("loop_overrun", float(tick_s), lim.loop_overrun_s),
        )
        for name, value, limit in checks:
            if not np.isfinite(value) or value > limit:
                self.fault = f"{name}:{value:.3f}>{limit:.3f}"
                return self.fault
        return None
