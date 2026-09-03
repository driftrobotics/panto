"""Haptic constraints.

Every constraint answers one question: given where the end-effector is, where
should the impedance anchor be, and how hard are we allowed to push toward it.

    project(pose) -> Projection(anchor, normal, penetration)

The runtime turns that into ``F = K·(anchor - pose)`` (+ local damping) and hands
it to whichever backend is active. Point / Line / Wall / WorkspaceBoundary /
SnapGrid all implement the same protocol so the runtime never special-cases them.

Analytic primitives are the source of truth for constraints — do not resample a
drawn line into points and snap to those (quantisation ripple in the force
field). Sampled points are only for recorded-trajectory playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Projection:
    anchor: np.ndarray      # [x, y] the impedance target pulls toward
    normal: np.ndarray      # unit vector, constraint's "push" direction
    penetration: float      # >0 means the EE is on the constrained side

    #: For a bilateral constraint (Point, Line snap) the anchor always attracts.
    #: For a unilateral one (Wall) the runtime zeroes the force when
    #: ``penetration <= 0`` so the free side stays free.
    unilateral: bool = False


class Constraint(Protocol):
    def project(self, pose: np.ndarray) -> Projection: ...


@dataclass(frozen=True)
class Point:
    """Bilateral: hold the EE at a fixed point. Milestone 2 / bring-up."""

    at: np.ndarray

    def project(self, pose: np.ndarray) -> Projection:
        raise NotImplementedError


@dataclass(frozen=True)
class Line:
    """Bilateral snap-to: attract toward the nearest point on an infinite line
    through ``a`` with unit direction ``d``. Milestone 4."""

    a: np.ndarray
    d: np.ndarray

    def project(self, pose: np.ndarray) -> Projection:
        raise NotImplementedError  # anchor = a + ((pose-a)·d) d


@dataclass(frozen=True)
class Wall:
    """Unilateral: free on one side, stiff into the surface. Milestone 5.

    Anchor is the per-tick reprojection of ``pose`` onto the surface, clamped to
    the penetration direction only. A fixed anchor at first contact would drag
    the user along the wall and pull them in from the free side.
    """

    a: np.ndarray          # a point on the wall
    normal: np.ndarray     # unit, points toward the *free* side

    def project(self, pose: np.ndarray) -> Projection:
        raise NotImplementedError


@dataclass(frozen=True)
class WorkspaceBoundary:
    """Unilateral: keeps the EE inside the reachable / well-conditioned region
    (σ_min(J) above threshold, away from extension + fold singularities)."""

    def project(self, pose: np.ndarray) -> Projection:
        raise NotImplementedError


@dataclass(frozen=True)
class SnapGrid:
    """Bilateral: attract toward the nearest grid intersection. Hotkey-toggled."""

    pitch: float
    origin: np.ndarray

    def project(self, pose: np.ndarray) -> Projection:
        raise NotImplementedError
