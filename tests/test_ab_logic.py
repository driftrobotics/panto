"""No-I/O unit tests for panto/ab_logic.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panto.ab_logic import (
    analyze_line, analyze_wall, build_report_md, combine_constraints,
    i2t_a2s, line_direction, normal_compliance_mm_per_a, project_current,
    pull_away_time_s, rms,
)
from panto.constraints import Projection


def test_line_direction_is_unit_and_45deg():
    d = line_direction(45.0)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert d[0] == pytest.approx(d[1])
    assert d[0] == pytest.approx(math.cos(math.radians(45.0)))


def test_line_direction_default_matches_explicit_45():
    assert np.allclose(line_direction(), line_direction(45.0))


def test_rms_basic():
    assert rms([3.0, -4.0]) == pytest.approx(math.sqrt((9 + 16) / 2))
    assert math.isnan(rms([]))


def test_project_current_identity_jacobian():
    # J = I -> inv(J).T = I -> the projection is just a plain dot product.
    J = np.eye(2)
    i_joint = np.array([3.0, 4.0])
    assert project_current(i_joint, J, np.array([1.0, 0.0])) == pytest.approx(3.0)
    assert project_current(i_joint, J, np.array([0.0, 1.0])) == pytest.approx(4.0)


def test_project_current_rotated_jacobian_direction_matters():
    # A pure rotation J: inv(J).T is the same rotation (orthogonal), so the
    # projection picks out the *rotated* component, not the raw joint value.
    theta = math.pi / 2
    J = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    i_joint = np.array([1.0, 0.0])
    along_x = project_current(i_joint, J, np.array([1.0, 0.0]))
    along_y = project_current(i_joint, J, np.array([0.0, 1.0]))
    assert along_x == pytest.approx(0.0, abs=1e-9)
    assert along_y == pytest.approx(1.0, abs=1e-9)


def test_i2t_a2s():
    currents = [[1.0, 2.0], [1.0, 2.0]]
    got = i2t_a2s(currents, period_s=0.5)
    assert got == pytest.approx([1.0, 4.0])


def test_i2t_a2s_empty():
    assert i2t_a2s([], period_s=0.5) == [0.0, 0.0]


def test_normal_compliance_mm_per_a():
    penetration = [5.0] * 10
    current = [1.0] * 10
    assert normal_compliance_mm_per_a(penetration, current) == pytest.approx(5.0)


def test_normal_compliance_nan_when_no_restoring_current():
    assert math.isnan(normal_compliance_mm_per_a([5.0, 5.0], [0.0, 0.0]))


def test_normal_compliance_uses_tail_only():
    # first half is a transient at 20mm/0A (would blow up the ratio); tail
    # (last 30%) has settled to 2mm/1A -> compliance should reflect the tail.
    penetration = [20.0] * 7 + [2.0] * 3
    current = [0.0] * 7 + [1.0] * 3
    assert normal_compliance_mm_per_a(penetration, current, tail_frac=0.3) == pytest.approx(2.0)


def test_pull_away_time_s_finds_crossing():
    t = [0.0, 1.0, 2.0, 3.0]
    penetration = [5.0, 5.0, 2.0, -1.0]
    assert pull_away_time_s(t, penetration, release_t=1.0) == pytest.approx(2.0)


def test_pull_away_time_s_none_if_never_releases():
    t = [0.0, 1.0, 2.0]
    penetration = [5.0, 4.0, 3.0]
    assert pull_away_time_s(t, penetration, release_t=1.0) is None


def test_pull_away_time_s_ignores_samples_before_release():
    # penetration dips below threshold before release_t; must not count that.
    t = [0.0, 1.0, 2.0, 3.0]
    penetration = [-1.0, -1.0, 5.0, -1.0]
    assert pull_away_time_s(t, penetration, release_t=2.0) == pytest.approx(1.0)


def test_combine_constraints_weighted_mean_of_bilateral_anchors():
    p1 = Projection(anchor=np.array([0.0, 0.0]), normal=np.zeros(2), penetration=0.0, unilateral=False)
    p2 = Projection(anchor=np.array([10.0, 0.0]), normal=np.zeros(2), penetration=0.0, unilateral=False)
    K, pull, active = combine_constraints([(p1, 1.0), (p2, 3.0)])
    assert active
    anchor = np.linalg.solve(K, pull)
    # k-weighted mean: (1*0 + 3*10) / 4 = 7.5
    assert anchor[0] == pytest.approx(7.5)
    assert anchor[1] == pytest.approx(0.0)


def test_combine_constraints_gates_inactive_unilateral():
    free_side = Projection(anchor=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]),
                            penetration=-1.0, unilateral=True)
    K, pull, active = combine_constraints([(free_side, 5.0)])
    assert not active
    assert np.allclose(K, 0.0)
    assert np.allclose(pull, 0.0)


def test_combine_constraints_includes_active_unilateral():
    blocked = Projection(anchor=np.array([1.0, 2.0]), normal=np.array([0.0, 1.0]),
                          penetration=0.5, unilateral=True)
    K, pull, active = combine_constraints([(blocked, 2.0)])
    assert active
    assert np.linalg.solve(K, pull) == pytest.approx([1.0, 2.0])


def test_analyze_line_fields():
    tang = [1.0, -1.0, 1.0, -1.0]
    lateral = [0.0, 1.0, 2.0, 1.0]
    currents = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
    m = analyze_line(tang, lateral, currents, period_s=0.01)
    assert m.tangential_current_rms_a == pytest.approx(1.0)
    assert m.lateral_max_mm == pytest.approx(2.0)
    assert m.rms_current_a[0] == pytest.approx(0.1)
    assert m.i2t_a2s[0] == pytest.approx(4 * 0.1 ** 2 * 0.01)


def test_analyze_wall_fields():
    penetration_hold = [3.0] * 5
    restoring_hold = [1.5] * 5
    t_full = [0.0, 1.0, 2.0, 3.0]
    penetration_full = [3.0, 3.0, 1.0, -1.0]
    currents = [[0.1, 0.1]] * 4
    m = analyze_wall(penetration_hold, restoring_hold, t_full, penetration_full,
                     release_t=1.0, currents=currents, period_s=0.01)
    assert m.normal_compliance_mm_per_a == pytest.approx(2.0)
    assert m.pull_away_time_s == pytest.approx(2.0)


def test_build_report_md_contains_backends_and_tasks():
    meta = {"utc": "20260101-000000", "preset": "hover-K25-pj", "pose_mm": [100.0, 80.0],
            "speed_mm_s": 10.0, "line_distance_mm": 30.0, "wall_depth_mm": 5.0,
            "wall_stiffness_n_per_m": 2000.0, "sim": True}
    results = {
        "position": {"line": {"tangential_current_rms_a": 0.123, "verdict": "converged"},
                     "wall": {"normal_compliance_mm_per_a": 0.05, "verdict": "stall"}},
        "torque": {"line": {"tangential_current_rms_a": 0.045, "verdict": "converged"},
                   "wall": {"normal_compliance_mm_per_a": 0.02, "verdict": "stall"}},
    }
    md = build_report_md(meta, results)
    assert "position" in md and "torque" in md
    assert "## line" in md and "## wall" in md
    assert "0.123" in md
    assert "hover-K25-pj" in md


def test_build_report_md_marks_missing_values():
    meta = {"utc": "x"}
    results = {"position": {"line": {}, "wall": {}}}
    md = build_report_md(meta, results)
    assert "-" in md


def test_build_report_md_notes_abort_reason():
    meta = {"utc": "x"}
    results = {"position": {"aborted": True, "abort_reason": "drive disarmed", "line": {}, "wall": {}}}
    md = build_report_md(meta, results)
    assert "drive disarmed" in md
