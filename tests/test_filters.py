"""Unit tests for panto/filters.py -- the host-side low-pass/notch used by
TorqueBackend's damping term."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panto.filters import Notch, OnePoleLowPass


def test_lowpass_passes_dc():
    lp = OnePoleLowPass(20.0)
    y = 0.0
    for _ in range(200):
        y = lp.update(3.0, 1.0 / 500.0)
    assert y == pytest.approx(3.0, abs=1e-3)


def test_lowpass_first_call_passthrough():
    lp = OnePoleLowPass(20.0)
    assert lp.update(5.0, 1.0 / 500.0) == pytest.approx(5.0)


def test_lowpass_disabled_when_fc_nonpositive():
    lp = OnePoleLowPass(0.0)
    assert lp.update(1.0, 0.01) == pytest.approx(1.0)
    assert lp.update(2.0, 0.01) == pytest.approx(2.0)  # no lag at all


def test_lowpass_attenuates_high_frequency():
    """A tone well above the cutoff should end up with much smaller
    amplitude than a tone well below it, after the filter settles."""
    fs = 1000.0
    dt = 1.0 / fs
    lp_lo = OnePoleLowPass(20.0)
    lp_hi = OnePoleLowPass(20.0)
    n = 2000
    t = np.arange(n) * dt

    low_freq_in = np.sin(2 * math.pi * 1.0 * t)     # 1 Hz, well below 20 Hz cutoff
    high_freq_in = np.sin(2 * math.pi * 200.0 * t)  # 200 Hz, well above cutoff

    low_out = np.array([lp_lo.update(x, dt) for x in low_freq_in])
    high_out = np.array([lp_hi.update(x, dt) for x in high_freq_in])

    tail = slice(n // 2, n)
    low_amp = low_out[tail].max() - low_out[tail].min()
    high_amp = high_out[tail].max() - high_out[tail].min()
    assert high_amp < 0.3 * low_amp


def test_notch_disabled_when_f0_nonpositive():
    nf = Notch(0.0, 4.0)
    assert nf.update(1.23, 0.01) == pytest.approx(1.23)


def test_notch_attenuates_center_frequency():
    fs = 1000.0
    dt = 1.0 / fs
    f0 = 50.0
    nf = Notch(f0, q=8.0)
    n = 4000
    t = np.arange(n) * dt
    x = np.sin(2 * math.pi * f0 * t)
    y = np.array([nf.update(v, dt) for v in x])

    tail = slice(n // 2, n)
    in_amp = x[tail].max() - x[tail].min()
    out_amp = y[tail].max() - y[tail].min()
    assert out_amp < 0.2 * in_amp


def test_notch_passes_far_off_center_frequency():
    fs = 1000.0
    dt = 1.0 / fs
    nf = Notch(50.0, q=8.0)
    n = 4000
    t = np.arange(n) * dt
    x = np.sin(2 * math.pi * 5.0 * t)  # far from the 50 Hz notch
    y = np.array([nf.update(v, dt) for v in x])

    tail = slice(n // 2, n)
    in_amp = x[tail].max() - x[tail].min()
    out_amp = y[tail].max() - y[tail].min()
    assert out_amp > 0.7 * in_amp


def test_notch_skips_when_above_nyquist():
    """A notch requested above Nyquist for the current dt must pass the
    sample through unmodified rather than compute nonsense coefficients."""
    nf = Notch(600.0, q=4.0)  # would need fs > 1200 Hz
    dt = 1.0 / 500.0          # fs = 500 Hz
    assert nf.update(2.5, dt) == pytest.approx(2.5)
