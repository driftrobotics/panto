"""Unit tests for panto/trace_logic.py -- lag estimation on synthetic data
with a known delay, and the summary metrics."""

from __future__ import annotations

import numpy as np
import pytest

from panto.trace_logic import analyze_trace, estimate_lag_s


def test_lag_estimate_matches_known_delay():
    dt = 0.004
    t = np.arange(0, 10, dt)
    freq = 0.3
    commanded = np.sin(2 * np.pi * freq * t)
    true_lag_s = 0.08
    shift = int(round(true_lag_s / dt))
    actual = np.concatenate([np.zeros(shift), commanded[:-shift]]) if shift else commanded.copy()

    lag = estimate_lag_s(t, commanded, actual)
    assert lag == pytest.approx(true_lag_s, abs=2 * dt)


def test_lag_estimate_zero_for_synchronous_signals():
    dt = 0.004
    t = np.arange(0, 5, dt)
    commanded = np.sin(2 * np.pi * 0.5 * t)
    lag = estimate_lag_s(t, commanded, commanded.copy())
    assert lag == pytest.approx(0.0, abs=1e-9)


def test_lag_estimate_handles_flat_signal():
    t = np.linspace(0, 1, 20)
    flat = np.zeros_like(t)
    assert estimate_lag_s(t, flat, flat) == 0.0


def test_lag_estimate_short_signal_returns_zero():
    t = np.array([0.0, 0.1, 0.2])
    assert estimate_lag_s(t, t, t) == 0.0


def test_analyze_trace_ok_verdict_for_small_error():
    n = 200
    t = np.linspace(0, 2, n)
    cmd = np.column_stack([np.linspace(0, 0.01, n), np.full(n, 0.15)])
    actual = cmd.copy()
    actual[:, 0] += 0.0005  # 0.5mm constant offset, well under 5mm
    currents = np.tile([0.3, 0.2], (n, 1))
    m = analyze_trace(t, cmd, actual, currents, period_s=t[1] - t[0])
    assert m.verdict == "ok"
    assert m.max_error_mm == pytest.approx(0.5, abs=1e-6)
    assert m.rms_error_mm == pytest.approx(0.5, abs=1e-6)


def test_analyze_trace_poor_verdict_for_large_error():
    n = 100
    t = np.linspace(0, 1, n)
    cmd = np.column_stack([np.zeros(n), np.full(n, 0.15)])
    actual = cmd.copy()
    actual[:, 0] += 0.01  # 10mm error, over the 5mm default threshold
    currents = np.zeros((n, 2))
    m = analyze_trace(t, cmd, actual, currents, period_s=t[1] - t[0])
    assert m.verdict == "poor"
    assert m.max_error_mm == pytest.approx(10.0, abs=1e-6)


def test_analyze_trace_aborted_verdict_overrides():
    n = 50
    t = np.linspace(0, 1, n)
    cmd = np.zeros((n, 2))
    actual = np.zeros((n, 2))
    currents = np.zeros((n, 2))
    m = analyze_trace(t, cmd, actual, currents, period_s=t[1] - t[0], aborted=True)
    assert m.verdict == "aborted"


def test_analyze_trace_per_side_rms_for_box():
    n = 40
    t = np.linspace(0, 1, n)
    cmd = np.column_stack([np.linspace(0, 0.025, n), np.full(n, 0.15)])
    actual = cmd.copy()
    actual[: n // 2, 0] += 0.001
    actual[n // 2 :, 0] += 0.003
    currents = np.zeros((n, 2))
    phase = ["side_0"] * (n // 2) + ["side_1"] * (n - n // 2)
    m = analyze_trace(t, cmd, actual, currents, period_s=t[1] - t[0], shape="box", phase=phase)
    assert m.per_side_rms_mm is not None
    assert m.per_side_rms_mm["side_0"] == pytest.approx(1.0, abs=1e-6)
    assert m.per_side_rms_mm["side_1"] == pytest.approx(3.0, abs=1e-6)


def test_analyze_trace_i2t_and_currents():
    n = 10
    t = np.linspace(0, 1, n)
    cmd = np.zeros((n, 2))
    actual = np.zeros((n, 2))
    currents = np.tile([1.0, 2.0], (n, 1))
    period = t[1] - t[0]
    m = analyze_trace(t, cmd, actual, currents, period_s=period)
    assert m.peak_current_a == pytest.approx([1.0, 2.0])
    assert m.rms_current_a == pytest.approx([1.0, 2.0])
    assert m.i2t_a2s[0] == pytest.approx(n * 1.0 ** 2 * period)
    assert m.i2t_a2s[1] == pytest.approx(n * 2.0 ** 2 * period)
