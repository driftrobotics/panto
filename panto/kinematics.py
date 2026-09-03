"""Serial 2R planar kinematics.

Textbook implementation for a shoulder + elbow arm with link lengths ``l1`` (base
to elbow) and ``l2`` (elbow to end-effector), both revolute, moving in a
horizontal plane so gravity is out-of-plane.

Angle convention here:
    q1  shoulder angle, measured from +x, absolute
    q2  elbow angle, measured from link 1, relative
    x = l1 cos(q1) + l2 cos(q1 + q2)
    y = l1 sin(q1) + l2 sin(q1 + q2)

⚠️  The "q(DD) monster math" Notion page has the worked math for the real
device. Reconcile sign conventions, angle zeros, and the base-frame origin with
that doc before trusting anything here on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkGeometry:
    l1: float  # metres, base axis -> elbow axis
    l2: float  # metres, elbow axis -> end-effector

    @classmethod
    def panto_v0(cls) -> "LinkGeometry":
        # 125 mm axis-to-axis, both links. Confirm l2 on the physical build.
        return cls(l1=0.125, l2=0.125)


def forward(q: np.ndarray, geo: LinkGeometry) -> np.ndarray:
    """Joint angles [q1, q2] (rad) -> end-effector [x, y] (m)."""
    q1, q2 = float(q[0]), float(q[1])
    x = geo.l1 * np.cos(q1) + geo.l2 * np.cos(q1 + q2)
    y = geo.l1 * np.sin(q1) + geo.l2 * np.sin(q1 + q2)
    return np.array([x, y])


def jacobian(q: np.ndarray, geo: LinkGeometry) -> np.ndarray:
    """d[x, y] / d[q1, q2] at ``q``. Columns are joint contributions.

    Used both for ``x_dot = J q_dot`` and for mapping an end-effector wrench to
    joint torques: ``tau = J.T @ F``.
    """
    q1, q2 = float(q[0]), float(q[1])
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array(
        [
            [-geo.l1 * s1 - geo.l2 * s12, -geo.l2 * s12],
            [geo.l1 * c1 + geo.l2 * c12, geo.l2 * c12],
        ]
    )


def min_singular_value(q: np.ndarray, geo: LinkGeometry) -> float:
    """σ_min(J) — the force/velocity authority in the worst direction.

    The workspace boundary is chosen so this stays above a threshold everywhere
    inside it; consistent EE force = motor_torque_limit * σ_min_threshold.
    """
    return float(np.linalg.svd(jacobian(q, geo), compute_uv=False)[-1])


class Unreachable(ValueError):
    """Target lies outside the annulus the linkage can reach."""


def inverse(
    xy: np.ndarray, geo: LinkGeometry, *, elbow: str = "up"
) -> np.ndarray:
    """End-effector [x, y] -> joint angles [q1, q2] for one elbow branch.

    ``elbow`` is "up" or "down". v0 stays on a single branch (mechanical limits +
    the workspace boundary keep us away from the branch-switching singularity);
    we never flip mid-motion. Raises :class:`Unreachable` outside the workspace.
    """
    x, y = float(xy[0]), float(xy[1])
    r2 = x * x + y * y
    cos_q2 = (r2 - geo.l1**2 - geo.l2**2) / (2 * geo.l1 * geo.l2)
    if not -1.0 <= cos_q2 <= 1.0:
        raise Unreachable(f"({x:.4f}, {y:.4f}) outside reach")
    q2 = np.arccos(cos_q2)
    if elbow == "down":
        q2 = -q2
    q1 = np.arctan2(y, x) - np.arctan2(
        geo.l2 * np.sin(q2), geo.l1 + geo.l2 * np.cos(q2)
    )
    return np.array([q1, q2])
