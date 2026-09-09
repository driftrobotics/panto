"""Unit tests for the oscillation guard (panto/guard.py), detector-only --
no CAN/sim involved, per the milestone-3 sweep spec's guard requirement.

2026-09-04 update: the current criterion was "pinned near cap for >50% of
window", full stop -- that false-tripped on an ordinary large step (current
saturates for the whole slew while genuinely converging). It's now a stall
criterion: pinned AND (error not decreasing, OR alternating sign >=3 times).
"""

from __future__ import annotations

import numpy as np
import pytest

from panto.guard import OscillationGuard, detect_oscillation


def _fill(g, n, t0=0.0, dt=0.01, pose=lambda i: (0.0, 0.0), cur=lambda i: (0.0, 0.0),
          cap=(0.6, 0.6), err=lambda i: 0.0):
    for i in range(n):
        g.push(t0 + i * dt, pose(i), cur(i), cap, err(i))
    return g


def test_clean_run_does_not_trip():
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    _fill(g, 80, pose=lambda i: (1.0, 1.0), cur=lambda i: (0.1, 0.1))
    assert g.check() is None


def test_buzzing_pose_trips_on_std():
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    # +/-8mm square wave in x -> std well above 3mm
    _fill(g, 80, pose=lambda i: (8.0 if i % 2 == 0 else -8.0, 0.0),
          cur=lambda i: (0.1, 0.1))
    reason = g.check()
    assert reason is not None
    assert "pose std" in reason


def test_saturated_slew_with_decreasing_error_does_not_trip():
    # A normal big step: current pinned the whole window (saturated slew), but
    # the tip error is monotonically closing -- must NOT trip. This is the
    # 2026-09-04 false positive (K=25, -x step, tripped 0.56s in, no
    # alternation) that motivated the stall criterion.
    def err(i):
        return 15.0 * (1.0 - i / 79.0)  # 15mm -> ~0mm, steadily decreasing
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    _fill(g, 80, cur=lambda i: (0.58, 0.0), err=err)
    assert g.check() is None


def test_pinned_with_static_error_trips_as_stall():
    # Current pinned near cap the whole window, but the error barely moves --
    # a real stall (e.g. friction/quantization deadband), no sign alternation.
    def err(i):
        return 10.0 - 0.01 * i  # ~9.7 -> ~9.2mm over the window: barely decreasing, not converging
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    _fill(g, 80, cur=lambda i: (0.58, 0.0), err=err)
    reason = g.check()
    assert reason is not None
    assert "stall" in reason


def test_alternating_current_at_cap_trips_as_buzz_even_if_error_shrinks():
    # Sign-alternating current pinned near cap is buzz regardless of whether
    # the (noisy) error nominally trends down over the window.
    def cur(i):
        return (0.58 if i % 2 == 0 else -0.58, 0.0)
    def err(i):
        return 15.0 - 0.1 * i  # decreasing, but alternation should still trip
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    _fill(g, 80, cur=cur, err=err)
    reason = g.check()
    assert reason is not None
    assert "buzz" in reason


def test_brief_current_spike_does_not_trip():
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    # near-cap for only ~10% of the window -- below the 50% threshold
    def cur(i):
        return (0.58, 0.0) if i < 8 else (0.05, 0.0)
    _fill(g, 80, cur=cur, err=lambda i: 10.0)
    assert g.check() is None


def test_insufficient_history_does_not_false_trip():
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    g.push(0.0, (0.0, 0.0), (0.0, 0.0), (0.6, 0.6), 10.0)
    g.push(0.05, (20.0, 0.0), (0.0, 0.0), (0.6, 0.6), 10.0)
    assert g.check() is None


def test_small_static_error_below_floor_does_not_stall_trip():
    # err_start <= stall_err_floor_mm (3mm default): pinned-but-tiny-error is
    # just fine settling precision, not a stall.
    g = OscillationGuard(window_s=0.5, osc_mm=3.0, current_frac=0.9)
    _fill(g, 80, cur=lambda i: (0.58, 0.0), err=lambda i: 2.0)
    assert g.check() is None


def test_stateless_detector_matches_class():
    ts = np.arange(80) * 0.01
    poses = [(9.0 if i % 2 == 0 else -9.0, 0.0) for i in range(80)]
    currents = [(0.1, 0.1)] * 80
    caps = [(0.6, 0.6)] * 80
    reason = detect_oscillation(ts, poses, currents, caps, osc_mm=3.0, min_window_s=0.25)
    assert reason is not None
