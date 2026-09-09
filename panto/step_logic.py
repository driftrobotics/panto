"""Pure logic for scripts/step_response.py -- schedule + post-processing.

Kept dependency-free (numpy only, no CAN/sim) so it's unit-testable without
hardware, same split as breakaway_logic.py / torque_step_logic.py.

Schedule (all times seconds, measured from run start t=0):

    [0, pre_s)                    hold at the anchor (0 offset along `dir`)
    [pre_s, pre_s+ramp_s)         ramp offset 0 -> +step_m (or instant if ramp_s==0)
    [pre_s+ramp_s, step_end)      hold at +step_m           (step_end = pre_s+step_hold_s)
    [step_end, step_end+ramp_s)   ramp offset +step_m -> 0
    [step_end+ramp_s, total_s)    hold at 0                 (total_s = step_end+back_hold_s)

`anchor_offset_m` returns the *scalar* offset along the step direction at time
t; multiply by the unit direction vector to get the 2-vector anchor offset.
`phase_name` returns a human label for printing/logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRE_HOLD_S = 1.0
STEP_HOLD_S = 2.0
BACK_HOLD_S = 2.0

def parse_per_joint(spec: str) -> tuple[float, float]:
    """Parse a CLI value that may be per-joint (`"0.9,0.5"` -> shoulder,elbow)
    or a single value applied to both (`"0.5"` -> (0.5, 0.5)). Used by
    --cap-min on scripts/step_response.py and scripts/goto_pose.py."""
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) == 1:
        v = float(parts[0])
        return (v, v)
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"expected 'V' or 'V0,V1', got {spec!r}")


def to_per_joint(v) -> tuple[float, float] | None:
    """Normalize an already-parsed per-joint value -- as stored in a preset
    (scalar or 2-list) or returned by :func:`parse_per_joint` (tuple) -- into
    a ``(shoulder, elbow)`` tuple. ``None`` passes through unchanged. Used for
    fields like ``cap_slope``/``cap_min`` that may be a single number applied
    to both joints or a per-joint pair."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"expected 2 values for per-joint spec, got {v!r}")
        return (float(v[0]), float(v[1]))
    return (float(v), float(v))


DIRS = {
    "+x": np.array([1.0, 0.0]),
    "-x": np.array([-1.0, 0.0]),
    "+y": np.array([0.0, 1.0]),
    "-y": np.array([0.0, -1.0]),
}


def total_duration_s(pre_s: float = PRE_HOLD_S, step_hold_s: float = STEP_HOLD_S,
                     back_hold_s: float = BACK_HOLD_S) -> float:
    return pre_s + step_hold_s + back_hold_s


def anchor_offset_m(t: float, step_m: float, ramp_s: float = 0.0,
                    pre_s: float = PRE_HOLD_S, step_hold_s: float = STEP_HOLD_S,
                    back_hold_s: float = BACK_HOLD_S) -> float:
    step_start = pre_s
    step_end = pre_s + step_hold_s
    back_end = step_end + back_hold_s
    if t < step_start:
        return 0.0
    if ramp_s > 0.0 and t < step_start + ramp_s:
        return step_m * (t - step_start) / ramp_s
    if t < step_end:
        return step_m
    if ramp_s > 0.0 and t < step_end + ramp_s:
        return step_m * (1.0 - (t - step_end) / ramp_s)
    if t < back_end:
        return 0.0
    return 0.0


def phase_name(t: float, ramp_s: float = 0.0, pre_s: float = PRE_HOLD_S,
              step_hold_s: float = STEP_HOLD_S, back_hold_s: float = BACK_HOLD_S) -> str:
    step_start = pre_s
    step_end = pre_s + step_hold_s
    back_end = step_end + back_hold_s
    if t < step_start:
        return "pre"
    if t < step_start + ramp_s:
        return "ramp_up"
    if t < step_end:
        return "step_hold"
    if t < step_end + ramp_s:
        return "ramp_down"
    if t < back_end:
        return "back_hold"
    return "done"


# --------------------------------------------------------------------- post-processing


@dataclass
class StepMetrics:
    rise_time_s: float | None
    overshoot_mm: float
    settling_time_s: float | None
    steady_state_error_mm: float
    osc_freq_hz: float | None
    osc_amplitude_mm: float
    decay_ratio: float | None
    verdict: str


def _dominant_freq_hz(t: np.ndarray, x_mm: np.ndarray) -> tuple[float | None, float]:
    """FFT of the (already step-onset-relative) error signal `x_mm` sampled at
    (uneven) times `t`. Resamples onto a uniform grid at the median dt first.
    Returns (dominant freq Hz or None if <8 samples / all-DC, amplitude mm)."""
    if len(t) < 8:
        return None, float(np.std(x_mm)) if len(x_mm) else 0.0
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None, float(np.std(x_mm))
    n = len(t)
    grid = t[0] + dt * np.arange(n)
    x = np.interp(grid, t, x_mm)
    x = x - x.mean()
    amp = float(np.std(x))
    if amp < 1e-6:
        return None, amp
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(spec) < 2:
        return None, amp
    peak_i = 1 + int(np.argmax(spec[1:]))  # skip DC bin
    return float(freqs[peak_i]), amp


def analyze_step(t: np.ndarray, disp_mm: np.ndarray, step_mm: float,
                 currents: np.ndarray | None = None, *, settle_tol_mm: float = 1.0,
                 stall_frac: float = 0.5, limit_cycle_decay: float = 0.7,
                 osc_noise_floor_mm: float = 0.3) -> StepMetrics:
    """`t` (s, relative to step onset) and `disp_mm` (signed displacement
    along the step direction, relative to the pre-step anchor) cover from
    step onset through the end of the step-hold phase (before stepping back).
    `currents` optional (n, 2) array over the same window, for RMS/peak
    (returned separately by the caller -- this function only classifies the
    motion). `step_mm` is the commanded (signed) step size."""
    t = np.asarray(t, float)
    disp_mm = np.asarray(disp_mm, float)
    target = step_mm

    rise_time_s = None
    thresh = 0.9 * target
    if target != 0:
        crossed = np.flatnonzero(np.sign(target) * disp_mm >= np.sign(target) * thresh)
        if crossed.size:
            rise_time_s = float(t[crossed[0]])

    if target >= 0:
        overshoot_mm = float(max(0.0, disp_mm.max() - target)) if len(disp_mm) else 0.0
    else:
        overshoot_mm = float(max(0.0, target - disp_mm.min())) if len(disp_mm) else 0.0

    err = disp_mm - target
    settling_time_s = None
    within = np.abs(err) <= settle_tol_mm
    for i in range(len(within)):
        if within[i:].all():
            settling_time_s = float(t[i])
            break

    tail_frac = 0.3
    tail_n = max(1, int(len(err) * tail_frac))
    steady_state_error_mm = float(np.mean(np.abs(err[-tail_n:]))) if len(err) else float("nan")

    # Oscillation is judged over the tail window, not the whole post-step
    # trace -- a clean first-order settle has plenty of "AC content" during
    # its initial transient (that's just the step itself decaying), which
    # would otherwise get misread as oscillation. The tail is what's left
    # once any legitimate transient has had time to die out.
    osc_freq_hz, osc_amplitude_mm = _dominant_freq_hz(t[-tail_n:], err[-tail_n:])

    decay_ratio = None
    if len(t) and t[-1] - t[0] >= 1.0:
        first_mask = t <= t[0] + 0.5
        last_mask = t >= t[-1] - 0.5
        std_first = float(np.std(err[first_mask])) if first_mask.any() else 0.0
        std_last = float(np.std(err[last_mask])) if last_mask.any() else 0.0
        decay_ratio = std_last / std_first if std_first > 1e-6 else (0.0 if std_last < 1e-6 else float("inf"))

    reached = (abs(float(np.mean(disp_mm[-tail_n:]))) >= stall_frac * abs(target)
              if (target != 0 and len(disp_mm)) else True)
    oscillating = osc_amplitude_mm is not None and osc_amplitude_mm > osc_noise_floor_mm

    if not reached and settling_time_s is None:
        verdict = "stall"
    elif oscillating and (decay_ratio is None or decay_ratio >= limit_cycle_decay):
        verdict = "limit_cycle"
    elif oscillating:
        verdict = "damped_oscillation"
    elif settling_time_s is not None:
        verdict = "converged"
    else:
        verdict = "damped_oscillation" if reached else "stall"

    return StepMetrics(
        rise_time_s=rise_time_s,
        overshoot_mm=overshoot_mm,
        settling_time_s=settling_time_s,
        steady_state_error_mm=steady_state_error_mm,
        osc_freq_hz=osc_freq_hz,
        osc_amplitude_mm=osc_amplitude_mm,
        decay_ratio=decay_ratio,
        verdict=verdict,
    )
