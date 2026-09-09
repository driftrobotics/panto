"""Oscillation guard for bring-up scripts (scripts/point_hold.py, scripts/offset_sweep.py).

Two independent trip conditions, either one aborts the run:

  1. tip pose is buzzing: std of either axis's pose (mm) over a rolling window
     exceeds ``osc_mm``.
  2. a joint is *stalled* pinned near its current cap: |current| > ``current_frac``
     * cap for more than half the samples in the window, on either joint, AND
     one of:
       a. the error to the anchor is not decreasing (``err_end > 0.9*err_start``
          and ``err_start > stall_err_floor_mm``) -- current is pinned but the
          tip isn't converging, i.e. a real stall/limit cycle, not
       b. the pinned current alternates sign >= ``alternation_count`` times in
          the window -- current-mode buzz, independent of whether err happens
          to be decreasing overall.

Condition 2 used to be "pinned near cap for >50% of the window", full stop --
that also fires on a perfectly ordinary large step: current saturates for the
whole slew while the tip is closing on the target, which is expected, not a
fault (see the 2026-09-04 sweep false-positive at K=25, tripped 0.56s into a
15mm -x step with no sign alternation). The error-progress / alternation split
distinguishes "saturated because we commanded a big move and it's still
closing" from "saturated and going nowhere" or "saturated and buzzing".

Kept as a small, deque-backed, dependency-free class so the detection math can
be unit-tested in isolation (no CAN bus / sim needed) -- see
tests/test_guard.py. The window is time-based (seconds), not sample-count
based, so it behaves the same regardless of loop rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OscillationGuard:
    window_s: float = 0.5
    osc_mm: float = 3.0
    current_frac: float = 0.9
    stall_err_floor_mm: float = 3.0        # below this, "not decreasing" doesn't count as a stall
    stall_err_ratio: float = 0.9           # err_end > ratio * err_start => not converging
    alternation_count: int = 3             # sign flips within the window => buzz
    _samples: deque = field(default_factory=deque, repr=False)

    def reset(self) -> None:
        self._samples.clear()

    def push(self, t: float, pose_mm, currents, current_cap, err_mm: float = 0.0) -> None:
        """Record one tick's tip pose (mm, len-2), joint currents (A, len-2),
        the current cap (A, scalar or len-2) in force at that tick, and the
        tip's distance to its current anchor (mm) -- needed to tell a
        saturated-but-converging slew apart from a stall."""
        self._samples.append((t, np.asarray(pose_mm, float), np.asarray(currents, float),
                              np.broadcast_to(np.asarray(current_cap, float), (2,)).copy(),
                              float(err_mm)))
        cutoff = t - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def check(self) -> str | None:
        """Return a human-readable trip reason, or None if the window is clean
        (or not yet full enough to judge)."""
        if len(self._samples) < 2:
            return None
        ts = [s[0] for s in self._samples]
        if ts[-1] - ts[0] < self.window_s * 0.5:
            return None  # not enough history yet -- don't false-trip at startup
        poses = np.array([s[1] for s in self._samples])
        currents = np.array([s[2] for s in self._samples])
        caps = np.array([s[3] for s in self._samples])
        errs = np.array([s[4] for s in self._samples])

        pose_std = poses.std(axis=0)
        if np.any(pose_std > self.osc_mm):
            return (f"pose std {pose_std.tolist()} mm exceeds osc_mm={self.osc_mm} "
                    f"over {ts[-1]-ts[0]:.2f}s window")

        near_cap = np.abs(currents) > (self.current_frac * caps)
        frac_near_cap = near_cap.mean(axis=0)
        pinned_axes = np.flatnonzero(frac_near_cap > 0.5)
        if pinned_axes.size == 0:
            return None

        err_start, err_end = errs[0], errs[-1]
        not_converging = (err_start > self.stall_err_floor_mm and
                          err_end > self.stall_err_ratio * err_start)

        for j in pinned_axes:
            signs = np.sign(currents[:, j])
            signs = signs[signs != 0]
            flips = int(np.sum(np.diff(signs) != 0)) if signs.size > 1 else 0
            if flips >= self.alternation_count:
                return (f"current on joint {j} alternates sign {flips} times while pinned "
                        f"near cap ({frac_near_cap[j]*100:.0f}% of window) -- buzz")
            if not_converging:
                return (f"current on joint {j} pinned near cap "
                        f"({frac_near_cap[j]*100:.0f}% of window) and tip error not decreasing "
                        f"(err {err_start:.2f}->{err_end:.2f}mm) -- stall")
        return None


def detect_oscillation(ts, poses_mm, currents, caps, *, errs_mm=None, osc_mm=3.0,
                        current_frac=0.9, min_window_s=0.25, **guard_kwargs) -> str | None:
    """Stateless version of OscillationGuard.check() over an explicit series,
    for direct unit testing without going through push()."""
    g = OscillationGuard(osc_mm=osc_mm, current_frac=current_frac, window_s=min_window_s * 2,
                         **guard_kwargs)
    if errs_mm is None:
        errs_mm = [0.0] * len(ts)
    for t, p, c, cap, e in zip(ts, poses_mm, currents, caps, errs_mm):
        g.push(t, p, c, cap, e)
    return g.check()
