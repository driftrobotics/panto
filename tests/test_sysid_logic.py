"""Unit tests for panto/sysid_logic.py -- pure estimator functions on
synthetic data, no CAN/sim needed (same split as test_breakaway.py /
test_torque_step_logic.py)."""

from __future__ import annotations

import numpy as np
import pytest

from panto.sysid_logic import (
    amp_profile_scale,
    cogging_spectrum,
    estimate_latency,
    fit_delay,
    fit_inertia,
    fit_inertia_delay,
    fit_kinetic_friction,
    infer_current_sign,
    instantaneous_freq,
    log_chirp,
    noise_threshold_crossing_time,
    predicted_low_freq_excursion_deg,
    profiled_chirp,
    select_coherent_band,
    step_crossing_time,
    welch_csd,
)


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------

def test_step_crossing_time_simple_rise():
    t = np.linspace(0, 1, 1001)
    y = np.where(t < 0.3, 0.0, 1.0)
    lat = step_crossing_time(t, y, 0.5)
    assert lat == pytest.approx(0.3, abs=0.002)


def test_step_crossing_time_no_crossing_returns_none():
    t = np.linspace(0, 1, 100)
    y = np.zeros_like(t)
    assert step_crossing_time(t, y, 0.5) is None


def test_noise_threshold_crossing_time():
    t = np.linspace(0, 1, 1001)
    y = np.where(t < 0.25, 0.0, 0.5)
    lat = noise_threshold_crossing_time(t, y, threshold=0.01)
    assert lat == pytest.approx(0.25, abs=0.002)


def test_noise_threshold_crossing_time_never_crosses():
    t = np.linspace(0, 1, 100)
    y = np.full_like(t, 0.001)
    assert noise_threshold_crossing_time(t, y, threshold=0.01) is None


def test_estimate_latency_synthetic_trials():
    # t is relative to t_cmd (t_cmd=0.0 here) -- negative times are the
    # pre-pulse window the new adaptive thresholds need: the Iq baseline/
    # settled-step median, and the velocity noise std.
    rng = np.random.RandomState(0)
    trials = []
    for _ in range(3):
        t = np.linspace(-0.2, 0.5, 701)
        iq = np.where(t < 0.02, 0.0, 0.25)
        vel = np.where(t < 0.03, 0.0, 0.5) + rng.normal(0, 0.001, size=t.shape)
        trials.append({
            "t_cmd": 0.0,
            "t_iq": t.tolist(), "iq": iq.tolist(),
            "t_vel": t.tolist(), "vel": vel.tolist(),
            "feedback_age_s": [0.002, 0.003, 0.0025],
        })
    est = estimate_latency(trials, pulse_s=0.1, vel_noise_sigma=5.0)
    assert est.iq_latency_s == pytest.approx(0.02, abs=0.002)
    assert est.vel_latency_s == pytest.approx(0.03, abs=0.002)
    assert est.n_trials == 3
    assert est.feedback_age_mean_s == pytest.approx(0.0025, abs=1e-6)
    assert est.iq_reason is None
    assert est.vel_reason is None


def test_estimate_latency_reports_reason_when_no_step():
    trials = [{
        "t_cmd": 0.0,
        "t_iq": np.linspace(-0.2, 0.3, 100).tolist(), "iq": [0.0] * 100,
        "t_vel": np.linspace(-0.2, 0.3, 100).tolist(), "vel": [0.0] * 100,
        "feedback_age_s": [0.001],
    }]
    est = estimate_latency(trials, pulse_s=0.1, vel_noise_sigma=5.0)
    assert est.iq_latency_s is None
    assert est.vel_latency_s is None
    assert est.iq_reason is not None
    assert est.vel_reason is not None


def test_infer_current_sign_detects_flip():
    active = np.ones(10, dtype=bool)
    i_cmd = np.linspace(0.05, 0.5, 10)
    same_sign_iq = i_cmd * 0.6
    flipped_iq = -i_cmd * 0.6
    assert infer_current_sign(i_cmd, same_sign_iq, active) == 1
    assert infer_current_sign(i_cmd, flipped_iq, active) == -1


def test_infer_current_sign_needs_variation():
    active = np.ones(5, dtype=bool)
    assert infer_current_sign(np.zeros(5), np.zeros(5), active) == 1


# --------------------------------------------------------------------------
# chirp waveform + frequency response fit
# --------------------------------------------------------------------------

def test_log_chirp_hits_endpoints_frequency():
    duration = 10.0
    freq = instantaneous_freq(np.array([0.0, duration]), f0=1.0, f1=40.0, duration=duration)
    assert freq[0] == pytest.approx(1.0)
    assert freq[1] == pytest.approx(40.0)


def test_log_chirp_amplitude_bounded():
    t = np.linspace(0, 5, 5000)
    y = log_chirp(t, amp=0.3, f0=1.0, f1=40.0, duration=5.0)
    assert np.max(np.abs(y)) <= 0.3 + 1e-9


def _simulate_rigid_body(t, i_cmd, J, damping, delay_s):
    """1/(J*s + damping) plant, driven by i_cmd delayed by delay_s, integrated
    with simple forward Euler -- good enough ground truth for a fit-recovery
    test at these frequencies/timesteps."""
    dt = t[1] - t[0]
    delay_samples = int(round(delay_s / dt))
    i_delayed = np.concatenate([np.zeros(delay_samples), i_cmd])[: len(t)]
    vel = np.zeros_like(t)
    for k in range(1, len(t)):
        accel = (i_delayed[k - 1] - damping * vel[k - 1]) / J
        vel[k] = vel[k - 1] + accel * dt
    return vel


def test_fit_inertia_and_delay_recover_known_plant():
    fs = 500.0
    duration = 12.0
    t = np.arange(0, duration, 1.0 / fs)
    i_cmd = log_chirp(t, amp=0.25, f0=1.0, f1=40.0, duration=duration)
    J_true = 6.0e-5
    damping_true = 1.2e-3
    delay_true = 0.004
    # b/J corner is damping/(2*pi*J) ~= 3.2 Hz here -- the inertia asymptote
    # |H|~=1/(J*omega) only holds above that, so the fit window must sit
    # above it (see panto.sysid_logic.fit_inertia's docstring for why an
    # earlier, low-frequency version of this fit was off by >1 order of
    # magnitude).
    qd = _simulate_rigid_body(t, i_cmd, J=J_true, damping=damping_true, delay_s=delay_true)

    fr = welch_csd(i_cmd, qd, fs=fs)
    fit = fit_inertia_delay(fr, f_min=10.0, f_max=40.0)

    assert fit.inertia_kg_m2 == pytest.approx(J_true, rel=0.2)
    assert abs(fit.delay_s - delay_true) < 0.003  # a few samples at fs=500


def test_fit_inertia_empty_mask_returns_nan():
    freq = np.array([100.0, 200.0])  # nothing <= f_max=5
    mag = np.array([1.0, 1.0])
    j, n = fit_inertia(freq, mag, f_max=5.0)
    assert n == 0
    assert np.isnan(j)


def test_fit_delay_needs_two_points():
    j, n = fit_delay(np.array([1.0]), np.array([0.1]))
    assert n < 2
    assert np.isnan(j)


def test_find_resonances_flags_a_peak():
    from panto.sysid_logic import find_resonances

    freq = np.linspace(1, 40, 400)
    mag = 1.0 / (2 * np.pi * freq)  # smooth rolloff
    # inject a resonance bump around 12 Hz
    mag = mag * (1.0 + 3.0 * np.exp(-((freq - 12.0) ** 2) / (2 * 0.5 ** 2)))
    res, anti = find_resonances(freq, mag, prominence_db=3.0)
    assert any(abs(f - 12.0) < 1.0 for f in res)


# --------------------------------------------------------------------------
# friction
# --------------------------------------------------------------------------

def test_fit_kinetic_friction_recovers_line():
    speeds = [0.1, 0.3, 0.6, 1.0]
    true_intercept, true_slope = 0.15, 0.05
    currents = [true_intercept + true_slope * s for s in speeds]
    intercept, slope = fit_kinetic_friction(speeds, currents)
    assert intercept == pytest.approx(true_intercept, abs=1e-6)
    assert slope == pytest.approx(true_slope, abs=1e-6)


def test_fit_kinetic_friction_needs_two_points():
    intercept, slope = fit_kinetic_friction([0.1], [0.2])
    assert np.isnan(intercept)
    assert np.isnan(slope)


# --------------------------------------------------------------------------
# cogging
# --------------------------------------------------------------------------

def test_cogging_spectrum_recovers_known_period_and_amplitude():
    angle_deg = np.linspace(-30, 30, 2000)
    period_deg = 12.0
    amplitude_a = 0.02
    iq = amplitude_a * np.sin(2 * np.pi * angle_deg / period_deg)
    fit = cogging_spectrum(np.radians(angle_deg), iq)
    assert fit.period_deg == pytest.approx(period_deg, rel=0.15)
    assert fit.amplitude_a == pytest.approx(amplitude_a, rel=0.25)


def test_cogging_spectrum_near_zero_for_flat_current():
    angle_deg = np.linspace(-30, 30, 500)
    iq = np.full_like(angle_deg, 0.1) + np.random.RandomState(0).normal(0, 1e-4, size=angle_deg.shape)
    fit = cogging_spectrum(np.radians(angle_deg), iq)
    assert fit.amplitude_a < 0.005


def test_cogging_spectrum_too_few_samples():
    fit = cogging_spectrum(np.array([0.0, 0.01]), np.array([0.0, 0.01]))
    assert np.isnan(fit.amplitude_a)


# --------------------------------------------------------------------------
# coherence gating (2026-09-08 hardware: a low-SNR chirp fit a nonsense
# negative delay from noise-dominated bins near the fit window's edge)
# --------------------------------------------------------------------------

def test_fit_delay_coherence_gate_excludes_noisy_bins():
    freq = np.linspace(10, 40, 30)
    omega = 2 * np.pi * freq
    true_delay = 0.004
    clean_phase = -np.pi / 2 - true_delay * omega
    coherence = np.full_like(freq, 0.95)

    # corrupt a couple of bins near the edge with garbage phase and mark them
    # low-coherence, like a noise-dominated bin would be
    noisy_phase = clean_phase.copy()
    noisy_phase[0] = 5.0
    noisy_phase[-1] = -8.0
    coherence[0] = 0.1
    coherence[-1] = 0.05

    delay_ungated, _ = fit_delay(freq, noisy_phase, f_min=10.0, f_max=40.0)
    delay_gated, n = fit_delay(freq, noisy_phase, f_min=10.0, f_max=40.0,
                              coherence=coherence, coherence_min=0.8)
    assert abs(delay_gated - true_delay) < abs(delay_ungated - true_delay)
    assert abs(delay_gated - true_delay) < 0.001
    assert n == 28


def test_fit_inertia_coherence_gate():
    freq = np.linspace(10, 40, 20)
    omega = 2 * np.pi * freq
    J_true = 6e-5
    mag = 1.0 / (J_true * omega)
    coherence = np.full_like(freq, 0.9)
    coherence[5] = 0.2  # excluded
    j, n = fit_inertia(freq, mag, f_min=10.0, f_max=40.0, coherence=coherence, coherence_min=0.8)
    assert j == pytest.approx(J_true, rel=1e-6)
    assert n == 19


# --------------------------------------------------------------------------
# amp-profile / pre-arm excursion guard
# --------------------------------------------------------------------------

def test_amp_profile_flat_is_constant():
    assert amp_profile_scale(1.0, 40.0, "flat") == 1.0
    assert amp_profile_scale(40.0, 40.0, "flat") == 1.0


def test_amp_profile_const_vel_scales_with_frequency():
    assert amp_profile_scale(1.0, 40.0, "const-vel") == pytest.approx(0.025)
    assert amp_profile_scale(40.0, 40.0, "const-vel") == pytest.approx(1.0)


def test_profiled_chirp_flat_matches_log_chirp():
    t = np.linspace(0, 5, 200)
    flat = profiled_chirp(t, 0.25, 1.0, 40.0, 5.0, "flat")
    plain = log_chirp(t, 0.25, 1.0, 40.0, 5.0)
    np.testing.assert_allclose(flat, plain)


def test_profiled_chirp_const_vel_tapers_low_frequency_amplitude():
    t = np.linspace(0.001, 5, 200)
    scaled = profiled_chirp(t, 1.0, 1.0, 40.0, 5.0, "const-vel")
    plain = log_chirp(t, 1.0, 1.0, 40.0, 5.0)
    # near the start (low instantaneous frequency), the scaled chirp must be
    # much smaller in amplitude than the flat one
    assert np.max(np.abs(scaled[:20])) < 0.3 * np.max(np.abs(plain[:20]))


def test_predicted_low_freq_excursion_flags_a_dangerous_amplitude():
    # 2026-09-08 hardware incident: a flat 0.25A chirp at f0=1Hz threw a free
    # (very low inertia) elbow ~69deg in 0.3s -- a conservative J guess
    # should predict a large excursion for that combination.
    deg = predicted_low_freq_excursion_deg(0.25, 1.0, 40.0, "flat", j_guess_a_s2_per_rad=0.001)
    assert deg > 25.0


def test_predicted_low_freq_excursion_const_vel_is_smaller():
    flat = predicted_low_freq_excursion_deg(0.25, 1.0, 40.0, "flat", 0.001)
    const_vel = predicted_low_freq_excursion_deg(0.25, 1.0, 40.0, "const-vel", 0.001)
    assert const_vel < flat


# --------------------------------------------------------------------------
# select_coherent_band / find_resonances coherence gating, and the full
# chirp pipeline on a synthetic noisy plant (2026-09-08 hardware: a fixed
# "top half of the sweep" fit window landed entirely in a noise-dominated
# region while a good contiguous coherent band sat lower in the sweep and
# the fit silently returned NaN)
# --------------------------------------------------------------------------

def test_select_coherent_band_finds_longest_contiguous_run():
    freq = np.arange(0, 20, dtype=float)
    coh = np.zeros_like(freq)
    coh[2:6] = 0.9      # a 3Hz-wide run
    coh[10:17] = 0.85   # a 6Hz-wide run -- should win
    band = select_coherent_band(freq, coh, coherence_min=0.8, f_floor=0.0)
    assert band == (10.0, 16.0)


def test_select_coherent_band_none_when_nothing_qualifies():
    freq = np.arange(0, 20, dtype=float)
    coh = np.full_like(freq, 0.3)
    assert select_coherent_band(freq, coh, coherence_min=0.8) is None


def test_find_resonances_excludes_low_coherence_peak():
    from panto.sysid_logic import find_resonances

    freq = np.linspace(1, 40, 400)
    mag = 1.0 / (2 * np.pi * freq)
    mag = mag * (1.0 + 3.0 * np.exp(-((freq - 12.0) ** 2) / (2 * 0.5 ** 2)))  # real bump @12Hz
    mag = mag * (1.0 + 5.0 * np.exp(-((freq - 30.0) ** 2) / (2 * 0.5 ** 2)))  # noise bump @30Hz
    coherence = np.where(freq < 20, 0.95, 0.1)  # only the 12Hz bump is trustworthy

    res_ungated, _ = find_resonances(freq, mag, prominence_db=3.0)
    assert any(abs(f - 30.0) < 1.0 for f in res_ungated)  # sanity: the fake bump IS a "peak"

    res_gated, _ = find_resonances(freq, mag, prominence_db=3.0, coherence=coherence,
                                   coherence_min=0.6)
    assert any(abs(f - 12.0) < 1.0 for f in res_gated)
    assert not any(abs(f - 30.0) < 1.0 for f in res_gated)


def test_full_chirp_pipeline_recovers_known_plant_with_noisy_high_band():
    """End-to-end regression for the 2026-09-08 fix: welch_csd ->
    select_coherent_band -> fit_inertia_delay on a synthetic response whose
    high-frequency content is noise-dominated (mimicking the real hardware
    log) must still recover J/delay from whatever coherent band survives,
    with coherence > 0.95 there -- not silently return NaN because a fixed
    high-frequency window missed the good data."""
    fs = 500.0
    duration = 12.0
    t = np.arange(0, duration, 1.0 / fs)
    i_cmd = log_chirp(t, amp=0.25, f0=1.0, f1=40.0, duration=duration)
    J_true = 6.0e-5
    delay_true = 0.004
    qd_clean = _simulate_rigid_body(t, i_cmd, J=J_true, damping=1.0e-5, delay_s=delay_true)

    # noise big enough to swamp the (shrinking, ~1/omega) high-frequency
    # signal but not the low/mid-frequency one -- exactly the real-hardware
    # shape reported 2026-09-08 (good coherence low, garbage above ~15-20Hz)
    rng = np.random.RandomState(0)
    qd = qd_clean + rng.normal(0, 0.05, size=qd_clean.shape)

    fr = welch_csd(i_cmd, qd, fs=fs)
    band = select_coherent_band(fr.freq_hz, fr.coherence, coherence_min=0.95, f_floor=1.0, f_ceil=40.0)
    assert band is not None, "expected a surviving high-coherence band"
    f_min, f_max = band
    mask = (fr.freq_hz >= f_min) & (fr.freq_hz <= f_max)
    assert np.all(fr.coherence[mask] > 0.95)

    fit = fit_inertia_delay(fr, f_min=f_min, f_max=f_max, coherence_min=0.95)
    assert np.isfinite(fit.inertia_kg_m2)
    assert fit.inertia_kg_m2 == pytest.approx(J_true, rel=0.3)
    assert abs(fit.delay_s - delay_true) < 0.003


def test_cogging_spectrum_removes_torsion_trend_and_reports_it():
    angle_deg = np.linspace(60.0, 90.0, 900)
    angle = np.radians(angle_deg)
    torsion = 0.9 * (angle - angle.mean()) + 0.2          # A/rad spring + bias
    iq = torsion + 0.05 * np.sin(2 * np.pi * angle_deg / 4.3)
    fit = cogging_spectrum(angle, iq, cap_a=0.8)
    assert abs(fit.torsion_a_per_rad - 0.9) < 0.05   # partial sine cycles bias the line fit slightly
    assert abs(fit.torsion_offset_a - 0.2) < 0.02
    assert abs(fit.period_deg - 4.3) < 0.3
    assert abs(fit.amplitude_a - 0.05) < 0.01
    assert fit.saturated_frac == 0.0 and not fit.rejected


def test_cogging_spectrum_rejects_bang_bang():
    angle = np.radians(np.linspace(60.0, 90.0, 900))
    iq = 0.8 * np.sign(np.sin(2 * np.pi * np.linspace(0, 15, 900)))   # pinned at +/-cap
    fit = cogging_spectrum(angle, iq, cap_a=0.8)
    assert fit.saturated_frac > 0.9
    assert fit.rejected
