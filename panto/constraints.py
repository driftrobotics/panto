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

Sign convention used here (see Projection): ``normal`` is a unit vector and,
where a restoring pull exists, points *from the pose toward the anchor* so that
``F = K·penetration·normal`` agrees with ``F = K·(anchor - pose)``. The only
subtlety is Wall — documented on the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from .kinematics import LinkGeometry, Unreachable, inverse, min_singular_value

_EPS = 1e-12


@dataclass(frozen=True)
class Projection:
    anchor: np.ndarray      # [x, y] the impedance target pulls toward
    normal: np.ndarray      # unit vector, constraint's "push" direction
    penetration: float      # >0 means the EE is on the constrained side

    #: For a bilateral constraint (Point, Line snap) the anchor always attracts.
    #: For a unilateral one (Wall) the runtime zeroes the force when
    #: ``penetration <= 0`` so the free side stays free.
    unilateral: bool = False
    #: bilateral snap radius (m): inert until the tip is within it; None = always
    snap_m: Optional[float] = None


class Constraint(Protocol):
    def project(self, pose: np.ndarray) -> Projection: ...


def _unit(v: np.ndarray) -> np.ndarray:
    """Unit vector, or zero when the input has no length (degenerate pose on
    the constraint — no push direction is defined and none is needed)."""
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.zeros(2)


def _clamp_along(t: float, a: np.ndarray, b: np.ndarray, d: np.ndarray) -> float:
    """Clamp a parameter along unit ``d`` from ``a`` to the segment a..b."""
    lo, hi = sorted((0.0, float((b - a) @ d)))
    return min(max(t, lo), hi)


def within_snap(distance: float, snap_m: Optional[float]) -> bool:
    """Bilateral 'snap-to' gating: no radius means always active; otherwise the
    constraint is inert until the tip comes within ``snap_m`` of it."""
    return snap_m is None or distance <= float(snap_m)


def is_active(projection: Projection) -> bool:
    """The unilateral gate, factored out: a bilateral term always contributes,
    a unilateral one only once the EE is past the surface. The runtime still
    owns how it *combines* the active terms (sum bilateral pulls, etc.)."""
    if projection.unilateral:
        return projection.penetration > 0.0
    return within_snap(projection.penetration, projection.snap_m)


@dataclass(frozen=True)
class Point:
    """Bilateral: hold the EE at a fixed point. Milestone 2 / bring-up."""

    at: np.ndarray
    snap_m: Optional[float] = None

    def project(self, pose: np.ndarray) -> Projection:
        pose = np.asarray(pose, dtype=float)
        at = np.asarray(self.at, dtype=float)
        delta = at - pose
        dist = float(np.linalg.norm(delta))
        return Projection(
            anchor=at.copy(),
            normal=_unit(delta),          # from pose toward the anchor
            penetration=dist,
            unilateral=False,
            snap_m=self.snap_m,
        )


@dataclass(frozen=True)
class Line:
    """Bilateral snap-to: attract toward the nearest point on an infinite line
    through ``a`` with direction ``d`` (normalised here, need not be unit).
    Milestone 4."""

    a: np.ndarray
    d: np.ndarray
    b: Optional[np.ndarray] = None   # second endpoint -> finite segment a..b
    snap_m: Optional[float] = None

    def project(self, pose: np.ndarray) -> Projection:
        pose = np.asarray(pose, dtype=float)
        a = np.asarray(self.a, dtype=float)
        d = _unit(np.asarray(self.d, dtype=float))
        t = float((pose - a) @ d)
        if self.b is not None:
            t = _clamp_along(t, a, np.asarray(self.b, dtype=float), d)
        anchor = a + t * d
        delta = anchor - pose               # perpendicular, pose -> line
        return Projection(
            anchor=anchor,
            normal=_unit(delta),
            penetration=float(np.linalg.norm(delta)),
            unilateral=False,
            snap_m=self.snap_m,
        )


@dataclass(frozen=True)
class Wall:
    """Unilateral: free on one side, stiff into the surface. Milestone 5.

    Anchor is the per-tick reprojection of ``pose`` onto the surface, clamped to
    the penetration direction only. A fixed anchor at first contact would drag
    the user along the wall and pull them in from the free side.

    Sign convention:
      * ``self.normal`` — constructor input, unit, points to the *free* side.
      * ``Projection.normal`` — the same free-side unit normal; the restoring
        force is ``+normal`` (out of the surface).
      * ``penetration`` — signed depth past the surface: ``>0`` when the pose is
        on the blocked (non-free) side, ``<=0`` on the free side.
      * free-side case: ``anchor == pose``, ``penetration <= 0``,
        ``normal`` = free-side normal, ``unilateral == True`` — the runtime
        gates the term off (``is_active`` is False) so nothing pulls.
      * blocked-side case: ``anchor`` = foot of the perpendicular onto the
        surface = ``pose + penetration·normal``; ``F = K·(anchor - pose)`` pushes
        straight back out with no tangential component.
    """

    a: np.ndarray          # a point on the wall
    normal: np.ndarray     # unit, points toward the *free* side
    b: Optional[np.ndarray] = None   # second endpoint -> finite wall a..b

    def project(self, pose: np.ndarray) -> Projection:
        pose = np.asarray(pose, dtype=float)
        a = np.asarray(self.a, dtype=float)
        n = _unit(np.asarray(self.normal, dtype=float))
        signed = float((pose - a) @ n)      # >0 on the free side
        penetration = -signed               # >0 once past the surface
        if self.b is not None and penetration > 0.0:
            # Finite wall: past the ends there is no surface to push against.
            b = np.asarray(self.b, dtype=float)
            tangent = _unit(b - a)
            along = float((pose - a) @ tangent)
            if along < 0.0 or along > float(np.linalg.norm(b - a)):
                penetration = 0.0
        # Reproject onto the surface, but only ever pull along +n (out). On the
        # free side the clamp collapses the anchor onto the pose: no suck-in.
        anchor = pose + max(penetration, 0.0) * n
        return Projection(
            anchor=anchor,
            normal=n,
            penetration=penetration,
            unilateral=True,
        )


@dataclass(frozen=True)
class WorkspaceBoundary:
    """Unilateral: keeps the EE inside the reachable / well-conditioned region
    (``σ_min(J) >= sigma_min_threshold``, away from the extension + fold
    singularities), optionally intersected with a Cartesian ``workspace_polygon``.

    Inside the valid region: ``anchor == pose``, zero penetration, no push.
    Outside: ``anchor`` = nearest valid pose, ``normal`` points back inward,
    ``penetration`` = distance to that anchor.

    The valid set is, for this geometry, a single conditioned annulus about the
    base; ``_nearest_valid`` exploits that (radial clamp + bisection refine) and
    only falls back to a coarse 2-D search if a ``workspace_polygon`` removes the
    whole ray.
    """

    geo: LinkGeometry = field(default_factory=LinkGeometry.panto_v0)
    elbow: str = "up"
    sigma_min_threshold: float = 0.03
    workspace_polygon: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, config) -> "WorkspaceBoundary":
        """Pull the fields we need off a ``Config`` without importing it (keeps
        this module's only hard dependency on ``kinematics``)."""
        return cls(
            geo=config.geo,
            elbow=config.elbow,
            sigma_min_threshold=config.sigma_min_threshold,
            workspace_polygon=config.workspace_polygon,
        )

    # --- validity predicate -------------------------------------------------
    def _valid(self, pose: np.ndarray) -> bool:
        try:
            q = inverse(np.asarray(pose, dtype=float), self.geo, elbow=self.elbow)
        except Unreachable:
            return False
        if min_singular_value(q, self.geo) < self.sigma_min_threshold:
            return False
        if self.workspace_polygon is not None:
            poly = np.asarray(self.workspace_polygon, dtype=float)
            if not _point_in_polygon(poly, np.asarray(pose, dtype=float)):
                return False
        return True

    def project(self, pose: np.ndarray) -> Projection:
        pose = np.asarray(pose, dtype=float)
        if self._valid(pose):
            return Projection(
                anchor=pose.copy(),
                normal=np.zeros(2),
                penetration=0.0,
                unilateral=True,
            )
        anchor = self._nearest_valid(pose)
        delta = anchor - pose
        return Projection(
            anchor=anchor,
            normal=_unit(delta),           # back toward the valid region
            penetration=float(np.linalg.norm(delta)),
            unilateral=True,
        )

    # --- nearest valid pose ----------------------------------------------------
    def _nearest_valid(self, pose: np.ndarray) -> np.ndarray:
        reach = self.geo.l1 + self.geo.l2
        r = float(np.linalg.norm(pose))
        u = pose / r if r > _EPS else np.array([1.0, 0.0])

        radii = np.linspace(1e-4, reach, 257)
        valid = np.array([self._valid(u * rr) for rr in radii])
        if not valid.any():
            return self._grid_fallback(pose)

        idx = np.where(valid)[0]
        lo_i, hi_i = int(idx[0]), int(idx[-1])
        r_lo = _refine_edge(
            self._valid, u,
            r_valid=radii[lo_i],
            r_invalid=(radii[lo_i - 1] if lo_i > 0 else 0.0),
        )
        r_hi = _refine_edge(
            self._valid, u,
            r_valid=radii[hi_i],
            r_invalid=(radii[hi_i + 1] if hi_i + 1 < len(radii) else reach),
        )
        cand = u * min(max(r, r_lo), r_hi)

        if self.workspace_polygon is not None and not self._valid(cand):
            poly = np.asarray(self.workspace_polygon, dtype=float)
            cand = _nearest_point_on_polyline(poly, pose)
            rc = float(np.linalg.norm(cand))
            uc = cand / rc if rc > _EPS else u
            cand = uc * min(max(rc, r_lo), r_hi)
        return cand

    def _grid_fallback(self, pose: np.ndarray) -> np.ndarray:
        reach = self.geo.l1 + self.geo.l2
        axis = np.linspace(-reach, reach, 81)
        best, best_d = None, np.inf
        for xx in axis:
            for yy in axis:
                cand = np.array([xx, yy])
                if self._valid(cand):
                    d = float(np.linalg.norm(cand - pose))
                    if d < best_d:
                        best_d, best = d, cand
        return pose.copy() if best is None else best


@dataclass(frozen=True)
class SnapGrid:
    """Bilateral: attract toward the nearest grid intersection. Hotkey-toggled."""

    pitch: float
    origin: np.ndarray
    snap_m: Optional[float] = None

    def project(self, pose: np.ndarray) -> Projection:
        pose = np.asarray(pose, dtype=float)
        origin = np.asarray(self.origin, dtype=float)
        anchor = origin + np.round((pose - origin) / self.pitch) * self.pitch
        delta = anchor - pose
        return Projection(
            anchor=anchor,
            normal=_unit(delta),
            penetration=float(np.linalg.norm(delta)),
            unilateral=False,
            snap_m=self.snap_m,
        )


# --- geometry helpers -------------------------------------------------------
def _refine_edge(
    pred: Callable[[np.ndarray], bool],
    u: np.ndarray,
    *,
    r_valid: float,
    r_invalid: float,
    iters: int = 48,
) -> float:
    """Bisect the radius bracket ``[r_valid, r_invalid]`` along ray ``u`` and
    return the extreme radius that is still valid."""
    for _ in range(iters):
        mid = 0.5 * (r_valid + r_invalid)
        if pred(u * mid):
            r_valid = mid
        else:
            r_invalid = mid
    return r_valid


def _point_in_polygon(poly: np.ndarray, p: np.ndarray) -> bool:
    """Ray-cast even-odd test for a simple closed polygon (vertex list)."""
    x, y = float(p[0]), float(p[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _nearest_point_on_polyline(poly: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Closest point on the closed polygon boundary to ``p``."""
    p = np.asarray(p, dtype=float)
    best, best_d = p.copy(), np.inf
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], dtype=float)
        b = np.asarray(poly[(i + 1) % n], dtype=float)
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < _EPS else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d, best = d, q
    return best
