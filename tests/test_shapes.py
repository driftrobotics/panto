"""Unit tests for panto/shapes.py -- closure, speed, and length checks, no
CAN/sim needed."""

from __future__ import annotations

import numpy as np
import pytest

from panto.shapes import path_length_m, path_points


def _speeds(path: np.ndarray) -> np.ndarray:
    t, xy = path[:, 0], path[:, 1:3]
    dt = np.diff(t)
    d = np.diff(xy, axis=0)
    dist = np.linalg.norm(d, axis=1)
    ok = dt > 1e-9
    return dist[ok] / dt[ok]


class TestBox:
    def test_closes(self):
        path = path_points("box", 0.025, (0.0, 0.15), speed=0.01, dt=0.004, laps=1)
        assert np.allclose(path[0, 1:3], path[-1, 1:3], atol=1e-6)

    def test_two_laps_close_at_lap_boundary_and_end(self):
        one = path_points("box", 0.025, (0.0, 0.15), speed=0.01, dt=0.004, laps=1)
        two = path_points("box", 0.025, (0.0, 0.15), speed=0.01, dt=0.004, laps=2)
        assert np.allclose(one[-1, 1:3], two[-1, 1:3], atol=1e-6)

    def test_speed_matches_nominal(self):
        path = path_points("box", 0.025, (0.0, 0.15), speed=0.012, dt=0.004, laps=1)
        speeds = _speeds(path)
        assert speeds.size > 0
        assert np.median(speeds) == pytest.approx(0.012, rel=0.05)

    def test_length_matches_helper(self):
        size, laps = 0.025, 2
        path = path_points("box", size, (0.0, 0.15), speed=0.01, dt=0.001, laps=laps)
        d = np.diff(path[:, 1:3], axis=0)
        total = float(np.linalg.norm(d, axis=1).sum())
        assert total == pytest.approx(path_length_m("box", size, laps), rel=0.02)

    def test_starts_at_corner_nearest_start_xy(self):
        centre = np.array([0.0, 0.15])
        half = 0.0125
        target_corner = centre + np.array([-half, half])  # top-left
        path = path_points("box", 0.025, centre, speed=0.01, dt=0.004, laps=1,
                            start_xy=target_corner + np.array([0.001, 0.001]))
        assert np.allclose(path[0, 1:3], target_corner, atol=1e-9)

    def test_corner_dwell_holds_position(self):
        path = path_points("box", 0.025, (0.0, 0.15), speed=0.01, dt=0.004, laps=1,
                            corner_dwell=0.1)
        # the first several samples (the initial dwell) should all sit at the
        # start corner.
        n_dwell = max(1, int(round(0.1 / 0.004)))
        first_leg = path[: n_dwell + 1, 1:3]
        assert np.allclose(first_leg, first_leg[0], atol=1e-9)


class TestCircle:
    def test_closes(self):
        path = path_points("circle", 0.025, (0.0, 0.15), speed=0.01, dt=0.004, laps=1)
        assert np.allclose(path[0, 1:3], path[-1, 1:3], atol=1e-3)

    def test_constant_radius(self):
        centre = np.array([0.0, 0.15])
        path = path_points("circle", 0.025, centre, speed=0.01, dt=0.004, laps=1)
        r = np.linalg.norm(path[:, 1:3] - centre, axis=1)
        assert np.allclose(r, 0.0125, atol=1e-9)

    def test_speed_matches_nominal(self):
        path = path_points("circle", 0.025, (0.0, 0.15), speed=0.015, dt=0.004, laps=1)
        speeds = _speeds(path)
        assert np.median(speeds) == pytest.approx(0.015, rel=0.05)

    def test_length_matches_helper(self):
        size, laps = 0.025, 1
        path = path_points("circle", size, (0.0, 0.15), speed=0.01, dt=0.001, laps=laps)
        d = np.diff(path[:, 1:3], axis=0)
        total = float(np.linalg.norm(d, axis=1).sum())
        assert total == pytest.approx(path_length_m("circle", size, laps), rel=0.02)


class TestLine:
    def test_round_trip_returns_to_start(self):
        path = path_points("line", 0.025, (0.0, 0.15), speed=0.01, dt=0.004, laps=1)
        assert np.allclose(path[0, 1:3], path[-1, 1:3], atol=1e-6)

    def test_reaches_both_ends(self):
        centre = np.array([0.0, 0.15])
        path = path_points("line", 0.025, centre, speed=0.01, dt=0.004, laps=1)
        xs = path[:, 1]
        assert xs.min() == pytest.approx(centre[0] - 0.0125, abs=1e-9)
        assert xs.max() == pytest.approx(centre[0] + 0.0125, abs=1e-9)

    def test_length_matches_helper(self):
        size, laps = 0.025, 3
        path = path_points("line", size, (0.0, 0.15), speed=0.01, dt=0.001, laps=laps)
        d = np.diff(path[:, 1:3], axis=0)
        total = float(np.linalg.norm(d, axis=1).sum())
        assert total == pytest.approx(path_length_m("line", size, laps), rel=0.02)


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        path_points("triangle", 0.025, (0.0, 0.15), speed=0.01, dt=0.004)


def test_bad_speed_raises():
    with pytest.raises(ValueError):
        path_points("box", 0.025, (0.0, 0.15), speed=0.0, dt=0.004)


def test_box_accepts_rectangle_extents():
    path = path_points("box", (0.04, 0.02), (0.0, 0.0), speed=0.01, dt=0.01)
    xy = path[:, 1:]
    assert xy[:, 0].max() == pytest.approx(0.02) and xy[:, 0].min() == pytest.approx(-0.02)
    assert xy[:, 1].max() == pytest.approx(0.01) and xy[:, 1].min() == pytest.approx(-0.01)
    assert path_length_m("box", (0.04, 0.02)) == pytest.approx(0.12)
    assert path_length_m("box", 0.03) == pytest.approx(0.12)
