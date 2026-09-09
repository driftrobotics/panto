"""Pure post-processing for scripts/trace_shape.py -- lag estimation and the
tracking-error summary. Kept dependency-free (numpy only), same split as
panto/step_logic.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _resample_uniform(t: np.ndarray, x: np.ndarray, dt: float | None = None):
    if dt is None:
        dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("non-positive dt")
    n = len(t)
    grid = t[0] + dt * np.arange(n)
    return grid, np.interp(grid, t, x), dt


def estimate_lag_s(t: np.ndarray, commanded: np.ndarray, actual: np.ndarray,
                    max_lag_s: float | None = None) -> float:
    """Estimate how long ``actual`` lags ``commanded`` (seconds; positive =
    actual arrives after commanded), by cross-correlating the two signals on
    a uniform time grid and locating the correlation peak.

    Both signals are de-meaned first so a constant offset doesn't bias the
    correlation. Returns 0.0 if there isn't enough data to estimate (< 8
    samples) or the signals are essentially flat.
    """
    t = np.asarray(t, float)
    commanded = np.asarray(commanded, float)
    actual = np.asarray(actual, float)
    if len(t) < 8:
        return 0.0
    grid, c, dt = _resample_uniform(t, commanded)
    _, a, _ = _resample_uniform(t, actual, dt)
    c = c - c.mean()
    a = a - a.mean()
    if np.std(c) < 1e-9 or np.std(a) < 1e-9:
        return 0.0
    corr = np.correlate(a, c, mode="full")
    lags = np.arange(-len(c) + 1, len(c))
    if max_lag_s is not None:
        keep = np.abs(lags * dt) <= max_lag_s
        corr = corr[keep]
        lags = lags[keep]
    best_lag = int(lags[int(np.argmax(corr))])
    return float(best_lag * dt)


@dataclass
class TraceMetrics:
    rms_error_mm: float
    max_error_mm: float
    mean_lag_s: float
    lag_x_s: float
    lag_y_s: float
    per_side_rms_mm: dict | None
    corner_overshoot_mm: float | None
    peak_current_a: list
    rms_current_a: list
    i2t_a2s: list
    verdict: str


def analyze_trace(t: np.ndarray, cmd_xy: np.ndarray, actual_xy: np.ndarray,
                  currents: np.ndarray, period_s: float, *,
                  shape: str = "circle", phase: list | None = None,
                  max_error_ok_mm: float = 5.0, aborted: bool = False,
                  max_lag_s: float | None = 1.0) -> TraceMetrics:
    """Summarize a tracking run.

    ``t`` seconds, ``cmd_xy``/``actual_xy`` (N, 2) metres, ``currents``
    (N, 2) amps. ``phase`` (optional, len N) is the path-side label
    (``side_0``..``side_3`` for a box) used for the per-side RMS breakdown.
    """
    t = np.asarray(t, float)
    cmd_xy = np.asarray(cmd_xy, float)
    actual_xy = np.asarray(actual_xy, float)
    currents = np.asarray(currents, float)

    err = actual_xy - cmd_xy
    err_mm = np.linalg.norm(err, axis=1) * 1e3
    rms_error_mm = float(np.sqrt(np.mean(err_mm ** 2))) if len(err_mm) else float("nan")
    max_error_mm = float(err_mm.max()) if len(err_mm) else float("nan")

    lag_x_s = estimate_lag_s(t, cmd_xy[:, 0], actual_xy[:, 0], max_lag_s=max_lag_s)
    lag_y_s = estimate_lag_s(t, cmd_xy[:, 1], actual_xy[:, 1], max_lag_s=max_lag_s)
    mean_lag_s = float((lag_x_s + lag_y_s) / 2.0)

    per_side_rms_mm = None
    corner_overshoot_mm = None
    if shape == "box" and phase is not None:
        phase = list(phase)
        sides = sorted({p for p in phase if p.startswith("side_")})
        if sides:
            per_side_rms_mm = {}
            for side in sides:
                mask = np.array([p == side for p in phase])
                if mask.any():
                    per_side_rms_mm[side] = float(np.sqrt(np.mean(err_mm[mask] ** 2)))
        corner_mask = np.array([p == "corner" for p in phase])
        corner_overshoot_mm = float(err_mm[corner_mask].max()) if corner_mask.any() else 0.0

    peak_current_a = np.abs(currents).max(axis=0).tolist() if len(currents) else [float("nan")] * 2
    rms_current_a = np.sqrt((currents ** 2).mean(axis=0)).tolist() if len(currents) else [float("nan")] * 2
    i2t_a2s = ((currents ** 2).sum(axis=0) * period_s).tolist() if len(currents) else [0.0, 0.0]

    if aborted:
        verdict = "aborted"
    elif len(err_mm) == 0:
        verdict = "no_data"
    elif max_error_mm < max_error_ok_mm:
        verdict = "ok"
    else:
        verdict = "poor"

    return TraceMetrics(
        rms_error_mm=rms_error_mm,
        max_error_mm=max_error_mm,
        mean_lag_s=mean_lag_s,
        lag_x_s=lag_x_s,
        lag_y_s=lag_y_s,
        per_side_rms_mm=per_side_rms_mm,
        corner_overshoot_mm=corner_overshoot_mm,
        peak_current_a=peak_current_a,
        rms_current_a=rms_current_a,
        i2t_a2s=i2t_a2s,
        verdict=verdict,
    )
