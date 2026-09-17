"""Teleop logic: joint map, slew limit, effort reflection, fault monitor."""

from __future__ import annotations

import numpy as np
import pytest

from panto.teleop import BuzzDetector, EffortReflector, JointMap, RateLimiter, TeleopLimits, TeleopMonitor


def _map(scale=(1.0, -2.0)):
    return JointMap(scale=scale, leader_zero=[0.5, -1.0], follower_zero=[0.1, 1.0],
                    follower_lo=[-0.2, 0.8], follower_hi=[0.4, 1.2])


def test_map_is_identity_at_engage_and_round_trips():
    m = _map()
    tgt, boxed = m.to_follower([0.5, -1.0])
    assert np.allclose(tgt, [0.1, 1.0]) and not boxed.any()
    tgt, _ = m.to_follower([0.6, -1.05])
    assert np.allclose(tgt, [0.2, 1.1])                  # scale -2 on the second joint
    assert np.allclose(m.to_leader(tgt), [0.6, -1.05])


def test_map_clamps_to_box_and_reports_it():
    tgt, boxed = _map().to_follower([2.0, -1.0])
    assert np.allclose(tgt, [0.4, 1.0])
    assert boxed.tolist() == [True, False]


@pytest.mark.parametrize("kwargs", [
    {"scale": (0.0, 1.0)},
    {"follower_lo": [0.5, 0.8]},                         # engage pose outside the box
])
def test_map_rejects_bad_setup(kwargs):
    base = dict(scale=(1.0, 1.0), leader_zero=[0, 0], follower_zero=[0.1, 1.0],
                follower_lo=[-0.2, 0.8], follower_hi=[0.4, 1.2])
    base.update(kwargs)
    with pytest.raises(ValueError):
        JointMap(**base)


def test_rate_limiter_slews():
    r = RateLimiter(1.0, [0.0, 0.0])
    assert np.allclose(r.step([1.0, -0.001], 0.005), [0.005, -0.001])
    for _ in range(300):
        out = r.step([1.0, 0.0], 0.005)
    assert np.allclose(out, [1.0, 0.0])


def test_reflector_opposes_loading_and_respects_deadband_and_clamp():
    r = EffortReflector(0.01, (1.0, -1.0), cutoff_hz=1000.0, deadband_nm=0.3, tau_max_nm=0.018, ramp_s=0.0)
    assert np.allclose(r.step([0.2, -0.2], 0.005), 0.0)            # inside the deadband
    for _ in range(50):
        out = r.step([1.3, 1.3], 0.005)
    assert np.allclose(out, [-0.01, 0.01], atol=1e-4)             # -alpha*sign(scale)*(1.3-0.3)
    for _ in range(50):
        out = r.step([50.0, 0.0], 0.005)
    assert out[0] == pytest.approx(-0.018)                         # clamped


def test_reflector_ramps_in():
    r = EffortReflector(0.01, (1.0, 1.0), cutoff_hz=1000.0, deadband_nm=0.0, tau_max_nm=1.0, ramp_s=1.0)
    first = r.step([1.0, 1.0], 0.005)
    assert abs(first[0]) < 1e-3
    for _ in range(400):
        out = r.step([1.0, 1.0], 0.005)
    assert np.allclose(out, -0.01, atol=1e-4)


def _ok(**over):
    base = dict(q_cmd=[0.0, 1.0], q_meas=[0.01, 1.0], qd_meas=[0.1, 0.0], tau_ext=[0.5, -0.5],
                held_err=np.zeros(4), leader_age_s=0.005, follower_age_s=0.005, tick_s=0.005)
    base.update(over)
    return base


def test_monitor_passes_nominal():
    assert TeleopMonitor().check(**_ok()) is None


@pytest.mark.parametrize("over,name", [
    ({"q_meas": [0.5, 1.0]}, "follower_tracking"),
    ({"qd_meas": [0.0, 3.0]}, "follower_overspeed"),
    ({"tau_ext": [9.0, 0.0]}, "follower_over_effort"),
    ({"held_err": [0.0, 0.5, 0.0, 0.0]}, "held_joint_drift"),
    ({"leader_age_s": 0.5}, "leader_stale"),
    ({"follower_age_s": 0.5}, "follower_stale"),
    ({"tick_s": 0.2}, "loop_overrun"),
    ({"tau_ext": [float("nan"), 0.0]}, "follower_over_effort"),
])
def test_monitor_faults_and_latches(over, name):
    mon = TeleopMonitor(TeleopLimits())
    fault = mon.check(**_ok(**over))
    assert fault is not None and fault.startswith(name)
    assert mon.check(**_ok()) == fault                             # latched


def test_buzz_detector_ignores_hand_motion_and_catches_oscillation():
    ts = np.arange(0, 2.0, 0.005)
    hand = BuzzDetector()
    assert all(hand.step(t, [0.5 * np.sin(2 * np.pi * 1.0 * t), 0.0]) is None for t in ts)   # 1 Hz sweep
    buzz = BuzzDetector()
    trips = [buzz.step(t, [0.0, np.radians(2.0) * np.sin(2 * np.pi * 12.0 * t)]) for t in ts]  # 12 Hz, 4 deg p-p
    assert any(r is not None and r.startswith("joint1") for r in trips)
    rng = np.random.default_rng(0)
    jitter = BuzzDetector()                                     # 2026-09-17 false trip: +-0.15 deg tremor
    assert all(jitter.step(t, np.radians(rng.uniform(-0.15, 0.15, 2))) is None for t in ts)
