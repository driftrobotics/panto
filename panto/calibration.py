"""Two-point calibration derivation (pure logic, no I/O).

Companion to the single-point ``/api/calibration/zero`` recipe already in
``web.py``. That recipe needs the operator to already know ``flip`` -- it only
solves for ``zero_offset_rad`` given one held pose. This module derives
``flip`` *and* ``zero_offset_rad`` (plus the mechanical fold limits) from two
held poses and their raw turns, no prior calibration assumed.

Poses (see CONTRACTS.md's CanLink calibration transform,
``q = s*2*pi*turns + zero_offset``, ``s = -1 if flip else +1``):

  A = full extension -- by definition ``q = (0, 0)`` (tip at ``(2L, 0)``).
  B = "collision 2" -- links folded fully onto each other (elbow folded CW,
      shoulder pushed to its CCW stop). This is the mechanical fold limit on
      *both* joints, so the fold angle measured at B doubles as the joint
      limit to record (``q0_max`` for the shoulder, ``q1_min`` for the elbow).

Derivation, per motor ``i``:
  ``dt_i = wrap(t_B,i - t_A,i)`` into ``(-0.5, 0.5]`` (single-turn encoder --
  the raw delta must be unwrapped explicitly since a small physical motion can
  cross the 0/1 turn boundary).
  Required sign of travel A->B is known a priori from the arm's geometry:
  shoulder swings positive (``dq0 > 0``), elbow swings negative (``dq1 < 0``).
  So ``s_0 = sign(dt_0)``, ``s_1 = -sign(dt_1)``; ``flip_i = (s_i == -1)``.
  ``zero_offset_rad_i = 0 - s_i * 2*pi * t_A,i`` (makes extension read exactly
  ``(0, 0)``). Fold angle ``q_B,i = s_i * 2*pi * dt_i``.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

#: plausible range for |fold angle|, degrees -- collision 2 folds each link
#: most but not all of the way onto itself; well outside this is very likely
#: the wrong pose (or a missed wrap) rather than a real reading.
_FOLD_DEG_MIN = 120.0
_FOLD_DEG_MAX = 179.0

#: below this many turns of travel between the two poses, treat the capture
#: as suspect (arm barely moved -- likely the same pose captured twice).
_MIN_TRAVEL_TURNS = 0.05


def _wrap_half(x: np.ndarray) -> np.ndarray:
    """Wrap ``x`` into ``(-0.5, 0.5]`` (one encoder revolution's worth)."""
    return x - np.ceil(x - 0.5)


def two_point(t_ext, t_fold) -> dict:
    """Derive flip / zero_offset / fold-limits from two raw-turn captures.

    ``t_ext``  -- raw turns ``[t0, t1]`` at full extension (``q = (0, 0)``).
    ``t_fold`` -- raw turns ``[t0, t1]`` at collision 2 (both links folded
    fully onto themselves -- the mechanical fold limit on each joint).

    Returns a dict:
      ``flip``: [bool, bool]
      ``zero_offset_rad``: [float, float]
      ``fold_deg``: [float, float]  -- signed fold angle at collision 2
      ``limits``: {"q0_max_rad": float, "q1_min_rad": float}
      ``warnings``: [str, ...]  -- never raises; implausible input is flagged
      here instead so the caller (a dry-run preview) can show it to the
      operator before anything is written.
    """
    t_ext = np.asarray(t_ext, dtype=float).reshape(2)
    t_fold = np.asarray(t_fold, dtype=float).reshape(2)

    dt = _wrap_half(t_fold - t_ext)

    s0 = 1.0 if dt[0] >= 0 else -1.0
    s1 = -1.0 if dt[1] >= 0 else 1.0
    s = np.array([s0, s1])

    flip = [bool(si == -1.0) for si in s]
    zero_offset_rad = -s * TWO_PI * t_ext
    fold_rad = s * TWO_PI * dt
    fold_deg = np.degrees(fold_rad)

    warnings: list[str] = []
    for i in range(2):
        if not (_FOLD_DEG_MIN <= abs(fold_deg[i]) <= _FOLD_DEG_MAX):
            warnings.append(
                f"motor {i}: fold angle {fold_deg[i]:.1f} deg implausible "
                f"(expected {_FOLD_DEG_MIN:.0f}-{_FOLD_DEG_MAX:.0f} deg) -- "
                "wrong pose or wrap"
            )
        if abs(dt[i]) < _MIN_TRAVEL_TURNS:
            warnings.append(f"motor {i} barely moved between poses")

    return {
        "flip": flip,
        "zero_offset_rad": zero_offset_rad.tolist(),
        "fold_deg": fold_deg.tolist(),
        "limits": {
            "q0_max_rad": float(fold_rad[0]),
            "q1_min_rad": float(fold_rad[1]),
        },
        "warnings": warnings,
    }
