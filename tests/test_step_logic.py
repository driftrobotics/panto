import numpy as np

from panto.step_logic import (
    PRE_HOLD_S, STEP_HOLD_S, BACK_HOLD_S, DIRS,
    total_duration_s, anchor_offset_m, phase_name, analyze_step,
    parse_per_joint, to_per_joint,
)


def test_total_duration():
    assert total_duration_s() == PRE_HOLD_S + STEP_HOLD_S + BACK_HOLD_S


def test_anchor_offset_instant_step():
    assert anchor_offset_m(0.5, 0.015) == 0.0
    assert anchor_offset_m(PRE_HOLD_S + 0.001, 0.015) == 0.015
    assert anchor_offset_m(PRE_HOLD_S + STEP_HOLD_S - 0.001, 0.015) == 0.015
    assert anchor_offset_m(PRE_HOLD_S + STEP_HOLD_S + 0.001, 0.015) == 0.0
    assert anchor_offset_m(total_duration_s() - 0.001, 0.015) == 0.0


def test_anchor_offset_ramp():
    ramp = 0.2
    off_mid = anchor_offset_m(PRE_HOLD_S + ramp / 2, 0.015, ramp_s=ramp)
    assert 0.0 < off_mid < 0.015
    assert anchor_offset_m(PRE_HOLD_S + ramp + 0.001, 0.015, ramp_s=ramp) == 0.015


def test_phase_name_sequence():
    assert phase_name(0.5) == "pre"
    assert phase_name(PRE_HOLD_S + 0.001) == "step_hold"
    assert phase_name(PRE_HOLD_S + STEP_HOLD_S + 0.001) == "back_hold"
    assert phase_name(total_duration_s() + 0.001) == "done"
    assert phase_name(PRE_HOLD_S + 0.05, ramp_s=0.2) == "ramp_up"


def test_dirs_unit_vectors():
    for v in DIRS.values():
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_parse_per_joint_single_value():
    assert parse_per_joint("3.0") == (3.0, 3.0)


def test_parse_per_joint_two_values():
    assert parse_per_joint("0.9,0.5") == (0.9, 0.5)


def test_to_per_joint_none_passes_through():
    assert to_per_joint(None) is None


def test_to_per_joint_scalar_broadcasts():
    assert to_per_joint(3.0) == (3.0, 3.0)


def test_to_per_joint_list_pairs():
    assert to_per_joint([0.9, 0.5]) == (0.9, 0.5)


def test_to_per_joint_tuple_pairs():
    assert to_per_joint((0.9, 0.5)) == (0.9, 0.5)


def test_analyze_step_converged():
    t = np.linspace(0, 2.0, 400)
    step_mm = 5.0
    disp = step_mm * (1 - np.exp(-t / 0.1))  # clean first-order response, no overshoot
    m = analyze_step(t, disp, step_mm)
    assert m.verdict == "converged"
    assert m.overshoot_mm < 0.5
    assert m.settling_time_s is not None
    assert m.steady_state_error_mm < 1.0


def test_analyze_step_limit_cycle():
    t = np.linspace(0, 2.0, 400)
    step_mm = 5.0
    # never converges, buzzes at constant amplitude the whole time (undamped)
    disp = step_mm + 5.5 * np.sign(np.sin(2 * np.pi * 14 * t))
    m = analyze_step(t, disp, step_mm)
    assert m.verdict == "limit_cycle"
    assert m.osc_freq_hz is not None
    assert abs(m.osc_freq_hz - 14.0) < 2.0


def test_analyze_step_damped_oscillation():
    t = np.linspace(0, 2.0, 400)
    step_mm = 5.0
    disp = step_mm + 4.0 * np.exp(-t / 0.3) * np.sin(2 * np.pi * 10 * t)
    m = analyze_step(t, disp, step_mm)
    assert m.verdict in ("damped_oscillation", "converged")
    assert m.decay_ratio is not None
    assert m.decay_ratio < 0.7


def test_analyze_step_stall():
    t = np.linspace(0, 2.0, 400)
    step_mm = 15.0
    disp = np.full_like(t, 1.0)  # barely moved
    m = analyze_step(t, disp, step_mm)
    assert m.verdict == "stall"
