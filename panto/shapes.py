"""Pure path generation for scripts/trace_shape.py -- no CAN/sim dependency,
same split as panto/step_logic.py.

``path_points`` returns an (N, 3) array of ``(t, x, y)`` samples (metres,
seconds) tracing a closed (box/circle) or back-and-forth (line) path in the
tip frame, at constant tangential speed. All shapes start at ``t=0`` with the
anchor already on the path (the caller is responsible for a lead-in ramp from
wherever the arm currently is to that first sample -- see
``scripts/trace_shape.py``'s ``--lead-in-s``).

Box: a square of side ``size_m`` centred on ``centre_xy``, traversed CCW.
Corner order is fixed (bottom-right, top-right, top-left, bottom-left) but
rotated so the path *starts* at whichever corner is nearest ``start_xy`` (if
given) -- this is what "starting at the corner nearest the start pose" means
in the brief; if ``start_xy`` is ``None`` the path just starts at the
bottom-right corner. Dwells ``corner_dwell`` seconds at each corner
(including the very first, so the anchor doesn't jerk the instant tracing
begins) before departing along the next side.

Circle: constant angular speed, radius ``size_m/2`` (``size_m`` is the
diameter), starting at the angle nearest ``start_xy`` (or angle 0, i.e. the
+x point on the circle, if not given).

Line: back-and-forth over a segment of length ``size_m`` centred on
``centre_xy`` along the x axis, starting at whichever end is nearest
``start_xy`` (or the -x end if not given). One ``lap`` is a full round trip
(there and back).
"""

from __future__ import annotations

import numpy as np

SHAPES = ("box", "circle", "line")


def _nearest_index(points: np.ndarray, ref: np.ndarray) -> int:
    d = np.linalg.norm(points - ref, axis=1)
    return int(np.argmin(d))


def _resample_segment(a: np.ndarray, b: np.ndarray, speed: float, dt: float,
                       t0: float) -> tuple[list[float], list[np.ndarray]]:
    """Uniform-speed samples strictly *after* ``a`` (exclusive) through ``b``
    (inclusive), at ``dt`` spacing, times starting at ``t0 + dt``."""
    length = float(np.linalg.norm(b - a))
    if length <= 0.0:
        return [], []
    duration = length / speed
    n_steps = max(1, int(round(duration / dt)))
    ts, pts = [], []
    for k in range(1, n_steps + 1):
        frac = min(1.0, k / n_steps)
        ts.append(t0 + k * dt)
        pts.append(a + frac * (b - a))
    return ts, pts


def _box_path(size_m: float, centre_xy: np.ndarray, speed: float, dt: float,
              laps: int, corner_dwell: float, start_xy: np.ndarray | None) -> np.ndarray:
    half = size_m / 2.0
    corners = np.array([
        [half, -half],
        [half, half],
        [-half, half],
        [-half, -half],
    ]) + centre_xy

    start_idx = _nearest_index(corners, start_xy) if start_xy is not None else 0
    order = [(start_idx + i) % 4 for i in range(4)]
    seq = corners[order]

    t = 0.0
    times = [0.0]
    pts = [seq[0].copy()]

    def dwell(at: np.ndarray) -> None:
        nonlocal t
        if corner_dwell <= 0.0:
            return
        n_dwell = max(1, int(round(corner_dwell / dt)))
        for _ in range(n_dwell):
            t += dt
            times.append(t)
            pts.append(at.copy())

    dwell(seq[0])  # settle before the first leg departs
    for _lap in range(laps):
        for i in range(4):
            a, b = seq[i], seq[(i + 1) % 4]
            ts, ps = _resample_segment(a, b, speed, dt, t)
            times.extend(ts)
            pts.extend(ps)
            t = times[-1]
            dwell(b)

    return np.column_stack([np.array(times), np.array(pts)])


def _circle_path(size_m: float, centre_xy: np.ndarray, speed: float, dt: float,
                  laps: int, start_xy: np.ndarray | None) -> np.ndarray:
    radius = size_m / 2.0
    if radius <= 0.0:
        return np.array([[0.0, centre_xy[0], centre_xy[1]]])
    if start_xy is not None:
        rel = np.asarray(start_xy, float) - centre_xy
        theta0 = float(np.arctan2(rel[1], rel[0]))
    else:
        theta0 = 0.0
    omega = speed / radius  # rad/s, CCW
    circumference = 2.0 * np.pi * radius
    total_duration = laps * circumference / speed
    n_steps = max(1, int(round(total_duration / dt)))
    t = dt * np.arange(n_steps + 1)
    theta = theta0 + omega * t
    x = centre_xy[0] + radius * np.cos(theta)
    y = centre_xy[1] + radius * np.sin(theta)
    return np.column_stack([t, x, y])


def _line_path(size_m: float, centre_xy: np.ndarray, speed: float, dt: float,
               laps: int, start_xy: np.ndarray | None) -> np.ndarray:
    half = size_m / 2.0
    ends = np.array([[-half, 0.0], [half, 0.0]]) + centre_xy
    start_idx = _nearest_index(ends, start_xy) if start_xy is not None else 0
    other_idx = 1 - start_idx

    t = 0.0
    times = [0.0]
    pts = [ends[start_idx].copy()]
    cur_idx, nxt_idx = start_idx, other_idx
    for _lap in range(laps):
        for _leg in range(2):  # there, then back
            ts, ps = _resample_segment(ends[cur_idx], ends[nxt_idx], speed, dt, t)
            times.extend(ts)
            pts.extend(ps)
            t = times[-1]
            cur_idx, nxt_idx = nxt_idx, cur_idx

    return np.column_stack([np.array(times), np.array(pts)])


def path_points(shape: str, size_m: float, centre_xy, speed: float, dt: float,
                laps: int = 1, corner_dwell: float = 0.0,
                start_xy=None) -> np.ndarray:
    """Return an (N, 3) array of ``(t, x, y)`` samples, metres/seconds.

    ``centre_xy`` is a 2-vector (metres); ``start_xy``, if given, is used to
    pick which corner/angle/end the path starts at (nearest one) so the lead-
    in from the arm's current pose is short. ``corner_dwell`` only applies to
    ``shape == "box"``.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
    if speed <= 0.0:
        raise ValueError("speed must be > 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if laps < 1:
        raise ValueError("laps must be >= 1")

    centre_xy = np.asarray(centre_xy, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float) if start_xy is not None else None

    if shape == "box":
        return _box_path(size_m, centre_xy, speed, dt, laps, corner_dwell, start_xy)
    if shape == "circle":
        return _circle_path(size_m, centre_xy, speed, dt, laps, start_xy)
    return _line_path(size_m, centre_xy, speed, dt, laps, start_xy)


def path_length_m(shape: str, size_m: float, laps: int = 1) -> float:
    """Total path length (metres) for one or more laps -- used by tests and
    by the CLI to sanity-print expected duration."""
    if shape == "box":
        return 4.0 * size_m * laps
    if shape == "circle":
        return np.pi * size_m * laps
    if shape == "line":
        return 2.0 * size_m * laps
    raise ValueError(f"unknown shape {shape!r}")
