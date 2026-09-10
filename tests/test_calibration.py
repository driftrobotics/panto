"""Pure checks for panto.calibration.two_point -- no I/O. Fixtures are built by
forward-applying the calibration transform (never hand-asserted numbers), so a
correct implementation must recover exactly what was used to build them."""

from __future__ import annotations

import numpy as np
import pytest

from panto.calibration import TWO_PI, two_point


def _forward_fixture(t_ext, flip, fold_deg_target):
    """Build a (t_ext, t_fold) pair consistent with a chosen flip/fold-angle,
    by inverting the same transform two_point is expected to recover.
    Returns (t_fold, zero_offset_rad_expected, fold_rad_expected)."""
    t_ext = np.asarray(t_ext, dtype=float)
    s = np.array([-1.0 if f else 1.0 for f in flip])
    fold_rad = np.radians(np.asarray(fold_deg_target, dtype=float))
    dt = fold_rad / (s * TWO_PI)
    t_fold = t_ext + dt
    zero_offset_rad = -s * TWO_PI * t_ext
    return t_fold, zero_offset_rad, fold_rad


def test_2026_09_04_fixture_recovers_flip_zero_and_fold():
    # 2026-09-04 rig calibration: real extension-pose raw turns, real
    # recorded flips, and a fold angle close to the collision-2 report
    # (elbow folded CW, shoulder pushed to its CCW stop, q0~+163.8/q1~-163.6).
    t_ext = [0.16966, 0.25424]
    flip_expected = [True, False]
    fold_deg_target = [163.8, -163.6]

    t_fold, zero_expected, fold_rad_expected = _forward_fixture(
        t_ext, flip_expected, fold_deg_target
    )

    result = two_point(t_ext, t_fold)

    assert result["flip"] == flip_expected
    assert result["zero_offset_rad"] == pytest.approx(zero_expected.tolist(), abs=1e-9)
    assert result["fold_deg"] == pytest.approx(fold_deg_target, abs=1e-6)
    assert result["limits"]["q0_max_rad"] == pytest.approx(fold_rad_expected[0], abs=1e-9)
    assert result["limits"]["q1_min_rad"] == pytest.approx(fold_rad_expected[1], abs=1e-9)
    assert result["warnings"] == []


def test_flip_false_case_also_recovered():
    # Same recipe but both motors travelling the "unflipped" sign, to make
    # sure flip=False isn't just the default that happens to fall out.
    t_ext = [0.4, -0.1]
    flip_expected = [False, True]
    fold_deg_target = [150.0, -140.0]

    t_fold, zero_expected, fold_rad_expected = _forward_fixture(
        t_ext, flip_expected, fold_deg_target
    )

    result = two_point(t_ext, t_fold)

    assert result["flip"] == flip_expected
    assert result["zero_offset_rad"] == pytest.approx(zero_expected.tolist(), abs=1e-9)
    assert result["fold_deg"] == pytest.approx(fold_deg_target, abs=1e-6)
    assert result["warnings"] == []


def test_wrap_crossing_recovered():
    # Motor 0 physically travels +0.455 turns from t_ext=0.9, so the true
    # unwrapped position is 1.355 -- but a single-turn encoder reports that
    # as 0.355 (mod 1). The naive raw diff (0.355 - 0.9 = -0.545) lands
    # outside (-0.5, 0.5], so an implementation that skips the explicit wrap
    # would get this motor wrong. Motor 1 is an ordinary in-range case for
    # contrast (no rollover, naive diff already in range).
    t_ext = np.array([0.9, 0.1])
    flip_expected = [False, True]
    fold_deg_target = [163.8, -160.0]

    t_fold, zero_expected, fold_rad_expected = _forward_fixture(
        t_ext, flip_expected, fold_deg_target
    )
    t_fold = t_fold.copy()
    t_fold[0] = t_fold[0] % 1.0  # simulate the single-turn sensor's own rollover
    # sanity: this fixture really does cross the boundary for motor 0
    assert not (-0.5 < (t_fold[0] - t_ext[0]) <= 0.5)

    result = two_point(t_ext, t_fold)

    assert result["flip"] == flip_expected
    assert result["zero_offset_rad"] == pytest.approx(zero_expected.tolist(), abs=1e-9)
    assert result["fold_deg"] == pytest.approx(fold_deg_target, abs=1e-6)
    assert result["warnings"] == []


def test_warns_on_implausible_fold_angle():
    # Only ~72 deg of travel -- well outside the 120-179 deg plausible band,
    # but comfortably above the "barely moved" threshold.
    result = two_point([0.0, 0.0], [0.2, 0.2])

    assert any("implausible" in w and "motor 0" in w for w in result["warnings"])
    assert any("implausible" in w and "motor 1" in w for w in result["warnings"])
    assert not any("barely moved" in w for w in result["warnings"])


def test_warns_on_barely_moved():
    result = two_point([0.0, 0.0], [0.01, 0.01])

    assert any("motor 0 barely moved" in w for w in result["warnings"])
    assert any("motor 1 barely moved" in w for w in result["warnings"])


def test_no_warnings_for_plausible_capture():
    t_ext = [0.16966, 0.25424]
    t_fold, _zero, _fold_rad = _forward_fixture(
        t_ext, [True, False], [163.8, -163.6]
    )
    result = two_point(t_ext, t_fold)
    assert result["warnings"] == []
