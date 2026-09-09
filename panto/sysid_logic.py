"""Pure signal-processing / estimator logic behind scripts/sysid.py, split out
so it's unit-testable on synthetic data without a CAN bus or sim (same pattern
as panto/breakaway_logic.py and panto/torque_step_logic.py).

Two independent pieces:

  * latency estimators (``mode latency``): given a logged torque-step trial
    (host command time, then a stream of stamped Iq / velocity samples),
    find command->Iq and command->velocity latency, plus the feedback-age
    distribution.
  * chirp / frequency-response estimators (``mode chirp``): build the
    logarithmic chirp waveform, and turn a (t, i_cmd, qd) trace into a
    complex frequency response H(f) = qd(f)/i_cmd(f) via Welch/CSD, then fit
    a rigid-body inertia (from |H| ~= 1/(J*omega) at low frequency, J in
    A.s^2/rad since the input is amps not N.m) and a pure delay (from the
    phase slope) to it.
  * friction estimators (``mode friction``): breakaway (static) current from
    a torque ramp, and kinetic current vs. commanded speed from a set of
    constant-velocity segments.
  * cogging estimator (``mode cogging``): the periodic (position-locked)
    component of Iq during a slow constant-velocity sweep, as an amplitude
    and a dominant spatial period in degrees.

Everything here is expressed in **current (A)**, never torque -- the drive's
torque_constant is unknown/unverified, so baking it into a fit would silently
scale every derived number by that error. ``scripts/sysid.py`` converts a
requested current to N.m only at the CAN-encoding boundary (``Set_Input_Torque``
wants N.m on the wire); everything upstream and downstream of that one call,
including this module, stays in amps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# latency mode
# --------------------------------------------------------------------------

def step_crossing_time(t: np.ndarray, y: np.ndarray, target_frac: float,
                       *, y0: float | None = None, y1: float | None = None) -> float | None:
    """First time ``y`` crosses ``target_frac`` of the way from its pre-step
    value ``y0`` (defaults to ``y[0]``) to its settled value ``y1`` (defaults
    to the mean of the last 20% of samples). Linear interpolation between the
    bracketing samples. ``None`` if the crossing never happens (e.g. no
    response at all -- a real latency failure, not a numerical one)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 2:
        return None
    if y0 is None:
        y0 = float(y[0])
    if y1 is None:
        tail = y[int(0.8 * len(y)):]
        y1 = float(tail.mean()) if len(tail) else float(y[-1])
    if y1 == y0:
        return None
    target = y0 + target_frac * (y1 - y0)
    rising = y1 > y0
    for i in range(1, len(y)):
        a, b = y[i - 1], y[i]
        crossed = (a <= target <= b) if rising else (b <= target <= a)
        if crossed and b != a:
            frac = (target - a) / (b - a)
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def noise_threshold_crossing_time(t: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """First time ``|y|`` exceeds ``threshold`` (a fixed noise floor, not a
    fraction of the step -- used for the velocity latency estimate, since a
    freshly-zeroed velocity has no clean "pre-step value" the way Iq does)."""
    t = np.asarray(t, float)
    y = np.abs(np.asarray(y, float))
    idx = np.argmax(y > threshold) if np.any(y > threshold) else None
    if idx is None or (idx == 0 and not y[0] > threshold):
        if not np.any(y > threshold):
            return None
    hits = np.flatnonzero(y > threshold)
    if len(hits) == 0:
        return None
    return float(t[hits[0]])


@dataclass(frozen=True)
class LatencyEstimate:
    iq_latency_s: float | None       # command -> Iq reaches 50% of its step
    vel_latency_s: float | None      # command -> |qd| exceeds the noise floor
    feedback_age_mean_s: float
    feedback_age_p50_s: float
    feedback_age_p95_s: float
    feedback_age_max_s: float
    n_trials: int
    iq_reason: str | None = None     # why iq_latency_s is None, if it is
    vel_reason: str | None = None


def _trial_iq_latency(t_iq: np.ndarray, iq: np.ndarray, pulse_s: float) -> tuple[float | None, str | None]:
    """Command->Iq latency for one trial. Threshold is 50% of the MEASURED
    step (median Iq over the last 50ms of the pulse, not an assumed
    commanded amplitude) -- a free joint's actual current during a small
    pulse can differ a lot from the commanded value (friction, voltage
    limits, a wrong sign on the wire), and a fixed-fraction-of-command
    threshold silently mis-measures or never crosses in that case."""
    if len(t_iq) < 2:
        return None, "too few samples"
    tail_mask = (t_iq >= max(0.0, pulse_s - 0.05)) & (t_iq < pulse_s)
    if not np.any(tail_mask):
        tail_mask = t_iq >= t_iq[-1] - 0.05
    settled = float(np.median(iq[tail_mask])) if np.any(tail_mask) else float(iq[-1])
    pre_mask = t_iq < 0.0
    baseline = float(np.median(iq[pre_mask])) if np.any(pre_mask) else float(iq[0])
    if abs(settled - baseline) < 1e-6:
        return None, "no measurable Iq step (settled == baseline)"
    lat = step_crossing_time(t_iq, iq, 0.5, y0=baseline, y1=settled)
    if lat is None or lat < 0:
        return None, "never crossed 50% of the measured step"
    return lat, None


def _trial_vel_latency(t_vel: np.ndarray, vel: np.ndarray, n_sigma: float = 5.0
                       ) -> tuple[float | None, str | None]:
    """Command->velocity latency, threshold = n_sigma * pre-pulse velocity
    noise std (not a fixed rad/s constant) -- a stiff/loaded joint and a
    free-swinging one have very different noise floors."""
    pre_mask = t_vel < 0.0
    if np.count_nonzero(pre_mask) < 4:
        return None, "too few pre-pulse samples to estimate noise floor"
    noise_std = float(np.std(vel[pre_mask]))
    threshold = max(n_sigma * noise_std, 1e-6)
    lat = noise_threshold_crossing_time(t_vel, vel, threshold)
    if lat is None or lat < 0:
        return None, f"never exceeded {n_sigma}-sigma noise floor ({threshold:.4g} rad/s)"
    return lat, None


def estimate_latency(trials: list[dict], *, pulse_s: float = 0.1,
                     vel_noise_sigma: float = 5.0) -> LatencyEstimate:
    """``trials``: one dict per repeated pulse, each with keys ``t_cmd``
    (host monotonic pulse-on time), ``t_iq``/``iq`` (arrays, Iq
    receive-stamped trace spanning pre-pulse through post-pulse, current
    already sign-corrected to the commanded direction -- see
    ``infer_current_sign``), ``t_vel``/``vel`` (velocity trace, same span),
    and ``feedback_age_s`` (array, sampled every tick over the whole trial).
    Both ``t_iq``/``t_vel`` are relative to ``t_cmd`` (negative = pre-pulse).
    Returns the across-trial median latency (only over trials that produced
    one) plus pooled feedback-age percentiles."""
    iq_lat, vel_lat, ages = [], [], []
    iq_reason = vel_reason = None
    for tr in trials:
        t_iq = np.asarray(tr["t_iq"], float) - tr["t_cmd"]
        iq = np.asarray(tr["iq"], float)
        lat, reason = _trial_iq_latency(t_iq, iq, pulse_s)
        if lat is not None:
            iq_lat.append(lat)
        else:
            iq_reason = reason

        t_vel = np.asarray(tr["t_vel"], float) - tr["t_cmd"]
        vel = np.asarray(tr["vel"], float)
        vlat, vreason = _trial_vel_latency(t_vel, vel, vel_noise_sigma)
        if vlat is not None:
            vel_lat.append(vlat)
        else:
            vel_reason = vreason

        ages.extend(np.asarray(tr.get("feedback_age_s", []), float).tolist())

    ages_arr = np.asarray(ages, float) if ages else np.array([float("nan")])
    return LatencyEstimate(
        iq_latency_s=float(np.median(iq_lat)) if iq_lat else None,
        vel_latency_s=float(np.median(vel_lat)) if vel_lat else None,
        feedback_age_mean_s=float(np.nanmean(ages_arr)),
        feedback_age_p50_s=float(np.nanpercentile(ages_arr, 50)),
        feedback_age_p95_s=float(np.nanpercentile(ages_arr, 95)),
        feedback_age_max_s=float(np.nanmax(ages_arr)),
        n_trials=len(trials),
        iq_reason=None if iq_lat else (iq_reason or "no trials"),
        vel_reason=None if vel_lat else (vel_reason or "no trials"),
    )


def infer_current_sign(i_cmd: np.ndarray, iq_measured: np.ndarray, active_mask: np.ndarray) -> int:
    """+1 or -1: whether the drive's *measured* Iq comes back with the same
    sign as our commanded current, or flipped. 2026-09-08 hardware finding:
    the elbow (node 1) reports Iq opposite in sign to the commanded joint
    current -- CanLink's flip convention cancels between the command and
    encoder-velocity paths (by construction, see can_link.py's module
    docstring), but Iq is a separate broadcast with its own on-drive sign
    that isn't guaranteed to agree. Anything downstream that wants "current
    that actually produced this motion" (a frequency-response input, an
    Iq-latency threshold) needs the MEASURED, sign-corrected Iq, not the
    command -- correlate the two over the active (pulse/chirp-on) samples and
    flip if they're anti-correlated."""
    i_cmd = np.asarray(i_cmd, float)[active_mask]
    iq = np.asarray(iq_measured, float)[active_mask]
    if len(i_cmd) < 2 or np.std(i_cmd) == 0 or np.std(iq) == 0:
        return 1
    corr = float(np.corrcoef(i_cmd, iq)[0, 1])
    return 1 if corr >= 0 else -1


# --------------------------------------------------------------------------
# chirp waveform
# --------------------------------------------------------------------------

def log_chirp(t: np.ndarray, amp: float, f0: float, f1: float, duration: float) -> np.ndarray:
    """Logarithmic ("exponential") chirp, amplitude ``amp``, instantaneous
    frequency sweeping f0 -> f1 over ``duration`` seconds. Phase is the
    analytic integral of the log-swept instantaneous frequency, so
    instantaneous frequency is exact (not a finite-difference approximation).
    ``t`` outside [0, duration] is clamped to the endpoints' rate (silence is
    handled by the caller, not here)."""
    t = np.asarray(t, float)
    tc = np.clip(t, 0.0, duration)
    if f0 <= 0 or f1 <= 0:
        raise ValueError("f0/f1 must be > 0 for a logarithmic chirp")
    k = (f1 / f0) ** (1.0 / duration)
    if abs(k - 1.0) < 1e-9:
        phase = 2.0 * math.pi * f0 * tc
    else:
        phase = 2.0 * math.pi * f0 * (k ** tc - 1.0) / math.log(k)
    return amp * np.sin(phase)


def profiled_chirp(t: np.ndarray, amp: float, f0: float, f1: float, duration: float,
                   profile: str = "const-vel") -> np.ndarray:
    """:func:`log_chirp` with an amplitude envelope applied per
    :func:`amp_profile_scale` -- ``flat`` reproduces ``log_chirp`` exactly;
    ``const-vel`` (default) tapers the low-frequency amplitude down so the
    joint doesn't get thrown at the sweep's low end (see
    ``amp_profile_scale``'s docstring for the 2026-09-08 hardware incident
    this fixes)."""
    t = np.asarray(t, float)
    base = log_chirp(t, amp, f0, f1, duration)
    freq = instantaneous_freq(t, f0, f1, duration)
    scale = np.array([amp_profile_scale(f, f1, profile) for f in np.atleast_1d(freq)])
    return base * scale.reshape(base.shape) if profile != "flat" else base


def instantaneous_freq(t: np.ndarray, f0: float, f1: float, duration: float) -> np.ndarray:
    t = np.asarray(t, float)
    tc = np.clip(t, 0.0, duration)
    k = (f1 / f0) ** (1.0 / duration)
    return f0 * (k ** tc)


def amp_profile_scale(freq_hz: float, f1: float, profile: str) -> float:
    """Amplitude multiplier at instantaneous frequency ``freq_hz`` for a
    chirp's ``--amp-profile``. ``flat``: 1.0 always (constant current
    amplitude across the sweep, the original behaviour). ``const-vel``
    (default): scales current proportional to ``freq_hz/f1`` so that, for a
    roughly inertia-dominated plant where velocity amplitude ~= current
    amplitude/(J*omega), the induced velocity (and hence excursion) stays
    roughly constant across the sweep instead of blowing up at low
    frequency -- 2026-09-08 hardware: a flat 0.25A chirp swung a free elbow
    69 degrees in 0.3s at its low-frequency start."""
    if profile == "flat":
        return 1.0
    if profile == "const-vel":
        return max(freq_hz, 1e-6) / f1
    raise ValueError(f"unknown amp_profile {profile!r}")


def predicted_low_freq_excursion_deg(amp_a: float, f0: float, f1: float, profile: str,
                                     j_guess_a_s2_per_rad: float) -> float:
    """Conservative worst-case peak excursion (deg) from a chirp's
    lowest-frequency content, assuming a pure-inertia plant (no damping/
    friction to slow it down -- worst case) driven at ``f0`` with whatever
    amplitude ``amp_profile`` applies there. For a sinusoidal current
    i(t)=A*sin(omega t), a pure integrator qdd=i/J gives steady-state angle
    amplitude A/(J*omega^2); used as a pre-arm safety check, not a plant
    model. ``j_guess_a_s2_per_rad`` should be a deliberately LOW estimate
    (stiffer/heavier than expected) so this stays conservative -- an
    underestimated J under-predicts the swing."""
    a0 = amp_a * amp_profile_scale(f0, f1, profile)
    omega0 = 2.0 * math.pi * f0
    if j_guess_a_s2_per_rad <= 0 or omega0 <= 0:
        return float("inf")
    angle_rad = a0 / (j_guess_a_s2_per_rad * omega0 ** 2)
    return math.degrees(angle_rad)


# --------------------------------------------------------------------------
# frequency response: tau -> qd
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FreqResponse:
    freq_hz: np.ndarray
    mag: np.ndarray          # |H(f)|, qd per N.m
    phase_rad: np.ndarray    # unwrapped
    coherence: np.ndarray | None = None


def welch_csd(x: np.ndarray, y: np.ndarray, fs: float, *, nperseg: int | None = None) -> FreqResponse:
    """H(f) = Pxy(f)/Pxx(f) (x = tau_cmd, y = qd) via Welch-averaged
    cross/auto spectral density, ~1 Hz bins by default at fs=500."""
    from scipy import signal  # local import: keep sysid_logic importable without scipy for pure-python callers

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if nperseg is None:
        nperseg = min(n, max(64, int(round(fs))))  # ~1 Hz bins at fs samples/s per bin
    f, Pxy = signal.csd(x, y, fs=fs, nperseg=nperseg)
    _, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    _, Pyy = signal.welch(y, fs=fs, nperseg=nperseg)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = Pxy / Pxx
        coh = (np.abs(Pxy) ** 2) / (Pxx * Pyy)
    mag = np.abs(H)
    phase = np.unwrap(np.angle(H))
    return FreqResponse(freq_hz=f, mag=mag, phase_rad=phase, coherence=coh)


@dataclass(frozen=True)
class InertiaDelayFit:
    inertia_kg_m2: float
    delay_s: float
    low_freq_mask_n: int
    resonances_hz: list[float]
    antiresonances_hz: list[float]
    coherence_at_fmin_fmax: tuple[float, float] | None = None


def fit_inertia(freq_hz: np.ndarray, mag: np.ndarray, *, f_min: float = 10.0,
                f_max: float = 40.0, coherence: np.ndarray | None = None,
                coherence_min: float = 0.8) -> tuple[float, int]:
    """Rigid-body inertia from the HIGH-frequency asymptote |H(f)| ~= 1/(J*omega).

    A real joint is ``H(omega) = 1/(b + j*omega*J)`` (viscous damping b, inertia
    J) -- at low frequency that's damping-dominated and flat (``|H| ~= 1/b``);
    inertia only takes over, and ``|H| ~= 1/(J*omega)`` only holds, once
    ``omega*J >> b``, i.e. *above* the b/J corner, not below it. (An earlier
    version of this fit used the low-frequency band and silently recovered
    ``1/b``-ish numbers, off by 1-2 orders of magnitude -- see the corner
    frequency check in tests/test_sysid_logic.py.) Default window
    ``[10, 40] Hz`` assumes the corner sits below ~10 Hz, true for every joint
    inertia/damping combination seen on this rig so far; pass a higher
    ``f_min`` if a fit still looks damping-flat (check the Bode plot).

    Fits J as the median of ``1/(|H|*omega)`` per-point (robust to a few
    noisy bins) over ``f_min <= f <= f_max``. If ``coherence`` is given, bins
    with coherence below ``coherence_min`` are excluded first -- a low
    coherence means Pxy/Pxx isn't a reliable linear-system estimate at that
    frequency (noise floor, off-axis excitation, a saturating current cap),
    and including it silently corrupts the median."""
    freq_hz = np.asarray(freq_hz, float)
    mag = np.asarray(mag, float)
    mask = (freq_hz >= f_min) & (freq_hz <= f_max) & np.isfinite(mag) & (mag > 0)
    if coherence is not None:
        mask &= np.asarray(coherence, float) >= coherence_min
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    omega = 2.0 * np.pi * freq_hz[mask]
    j_per_point = 1.0 / (mag[mask] * omega)
    return float(np.median(j_per_point)), n


def fit_delay(freq_hz: np.ndarray, phase_rad: np.ndarray, *, f_min: float = 10.0,
             f_max: float = 40.0, coherence: np.ndarray | None = None,
             coherence_min: float = 0.8) -> tuple[float, int]:
    """Pure transport delay from the phase slope: phase(omega) ~= -pi/2 -
    delay*omega once inertia dominates (the -pi/2 intercept a linear fit's
    intercept absorbs); at low frequency (damping-dominated) phase curves
    toward 0 instead, which would bias a slope fit that included it -- same
    ``[f_min, f_max]`` high-frequency window as :func:`fit_inertia`, and for
    the same reason. Low-coherence bins are excluded the same way -- a
    single noise-dominated bin near the fit window's edge can flip the sign
    of the fitted slope (2026-09-08 hardware: a 0.05A chirp on a near-silent
    channel fit a nonsense -70ms "delay" from bins that were mostly noise;
    coherence gating is the fix, not a smarter slope estimator)."""
    freq_hz = np.asarray(freq_hz, float)
    phase_rad = np.asarray(phase_rad, float)
    mask = (freq_hz >= f_min) & (freq_hz <= f_max) & np.isfinite(phase_rad)
    if coherence is not None:
        mask &= np.asarray(coherence, float) >= coherence_min
    n = int(mask.sum())
    if n < 2:
        return float("nan"), n
    omega = 2.0 * np.pi * freq_hz[mask]
    A = np.vstack([omega, np.ones_like(omega)]).T
    slope, _intercept = np.linalg.lstsq(A, phase_rad[mask], rcond=None)[0]
    return float(-slope), n


def find_resonances(freq_hz: np.ndarray, mag: np.ndarray, *, prominence_db: float = 3.0,
                    f_min: float = 1.0, f_max: float = 40.0,
                    coherence: np.ndarray | None = None,
                    coherence_min: float = 0.6) -> tuple[list[float], list[float]]:
    """Local peaks (resonance) / dips (anti-resonance) in 20*log10|H| beyond
    ``prominence_db`` relative to their immediate neighbourhood, restricted
    to ``[f_min, f_max]``. A thin wrapper on scipy.signal.find_peaks so the
    prominence threshold is the one tunable.

    2026-09-08 hardware: without a coherence floor this reported "resonances"
    up in the noise-dominated tail of a chirp (coherence ~0.001-0.4), which
    are just noise bumps, not real structural features -- excluded here by
    requiring ``coherence >= coherence_min`` at a candidate peak/dip's own
    bin (a peak found in the masked, gap-containing array is re-checked
    against the *unmasked* index so the coherence lookup lines up)."""
    from scipy.signal import find_peaks

    freq_hz = np.asarray(freq_hz, float)
    mag = np.asarray(mag, float)
    mask = (freq_hz >= f_min) & (freq_hz <= f_max) & np.isfinite(mag) & (mag > 0)
    if coherence is not None:
        mask &= np.asarray(coherence, float) >= coherence_min
    f = freq_hz[mask]
    db = 20.0 * np.log10(mag[mask])
    if len(f) < 3:
        return [], []
    peak_idx, _ = find_peaks(db, prominence=prominence_db)
    dip_idx, _ = find_peaks(-db, prominence=prominence_db)
    return [float(f[i]) for i in peak_idx], [float(f[i]) for i in dip_idx]


def select_coherent_band(freq_hz: np.ndarray, coherence: np.ndarray, *, coherence_min: float = 0.8,
                         f_floor: float = 1.0, f_ceil: float | None = None
                         ) -> tuple[float, float] | None:
    """Longest contiguous run of frequency bins with ``coherence >=
    coherence_min`` (and ``f_floor <= f <= f_ceil``), as a ``(f_min, f_max)``
    fit window. ``None`` if no bin qualifies.

    2026-09-08 hardware finding: a fixed "top half of the swept band" fit
    window (the sim-derived heuristic, chosen to dodge the low-frequency
    damping-flat asymptote) can land entirely in a noise-dominated region on
    real hardware while a perfectly good, contiguous high-coherence band
    (here: ~1-14 Hz, coherence 0.8-0.95) sits lower in the sweep -- the fixed
    window silently produced NaN (zero bins survived) while a working fit was
    sitting right there. Picking the fit window FROM the coherence data
    itself, rather than guessing a frequency range up front, is the fix;
    :func:`fit_inertia`/:func:`fit_delay`'s own coherence gate still applies
    on top of whatever window this returns, as a second check."""
    freq_hz = np.asarray(freq_hz, float)
    coherence = np.asarray(coherence, float)
    mask = (freq_hz >= f_floor) & np.isfinite(coherence) & (coherence >= coherence_min)
    if f_ceil is not None:
        mask &= freq_hz <= f_ceil
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    best = max(runs, key=lambda r: freq_hz[r[1]] - freq_hz[r[0]])
    return float(freq_hz[best[0]]), float(freq_hz[best[1]])


def fit_inertia_delay(fr: FreqResponse, *, f_min: float = 10.0, f_max: float = 40.0,
                      resonance_prominence_db: float = 3.0,
                      coherence_min: float = 0.8,
                      resonance_coherence_min: float = 0.6) -> InertiaDelayFit:
    j, n = fit_inertia(fr.freq_hz, fr.mag, f_min=f_min, f_max=f_max,
                       coherence=fr.coherence, coherence_min=coherence_min)
    delay, _ = fit_delay(fr.freq_hz, fr.phase_rad, f_min=f_min, f_max=f_max,
                         coherence=fr.coherence, coherence_min=coherence_min)
    # resonance/anti-resonance peaks are also gated on coherence (2026-09-08:
    # otherwise noise-dominated bins report fake "resonances" -- see
    # find_resonances's docstring)
    res, anti = find_resonances(fr.freq_hz, fr.mag, prominence_db=resonance_prominence_db,
                                coherence=fr.coherence, coherence_min=resonance_coherence_min)
    coh_range = None
    if fr.coherence is not None:
        band = (fr.freq_hz >= f_min) & (fr.freq_hz <= f_max)
        if np.any(band):
            coh_range = (float(np.min(fr.coherence[band])), float(np.max(fr.coherence[band])))
    return InertiaDelayFit(inertia_kg_m2=j, delay_s=delay, low_freq_mask_n=n,
                           resonances_hz=res, antiresonances_hz=anti,
                           coherence_at_fmin_fmax=coh_range)


# --------------------------------------------------------------------------
# friction mode: static (breakaway) + kinetic (vs. speed)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FrictionFit:
    static_a: float                 # |current| at breakaway
    kinetic_a: list                 # per commanded speed, mean |Iq| once settled
    speeds_rad_s: list
    kinetic_intercept_a: float       # linear fit Iq(|speed|) -> intercept at v=0
    kinetic_slope_a_per_rad_s: float  # viscous-like slope, A per rad/s


def fit_kinetic_friction(speeds_rad_s: list, mean_currents_a: list) -> tuple[float, float]:
    """Linear least-squares fit of steady-state |Iq| vs. |speed| across the
    constant-velocity segments: intercept ~= Coulomb kinetic friction (A),
    slope ~= viscous term (A per rad/s). Needs >=2 points; returns (nan, nan)
    otherwise."""
    speeds = np.asarray(speeds_rad_s, float)
    currents = np.asarray(mean_currents_a, float)
    if len(speeds) < 2:
        return float("nan"), float("nan")
    A = np.vstack([speeds, np.ones_like(speeds)]).T
    slope, intercept = np.linalg.lstsq(A, currents, rcond=None)[0]
    return float(intercept), float(slope)


# --------------------------------------------------------------------------
# cogging mode: periodic component of Iq vs. angle
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CoggingFit:
    amplitude_a: float        # half the peak-to-peak of the dominant harmonic
    period_deg: float         # spatial period of the dominant harmonic
    n_samples: int


def cogging_spectrum(angle_rad: np.ndarray, iq_a: np.ndarray, *,
                     n_resample: int = 720, min_period_deg: float = 1.0,
                     max_period_deg: float = 60.0) -> CoggingFit:
    """Resample the (angle, Iq) trace onto a uniform angle grid (constant
    velocity in the raw trace makes this close to a no-op, but a uniform grid
    is what an angle-domain FFT assumes), detrend, FFT in the angle domain,
    and report the dominant spatial harmonic within
    [min_period_deg, max_period_deg] as an amplitude + period. Angle need not
    be monotonic increasing -- sorted internally so both sweep directions
    work the same way."""
    angle = np.asarray(angle_rad, float)
    iq = np.asarray(iq_a, float)
    order = np.argsort(angle)
    angle, iq = angle[order], iq[order]
    # drop duplicate angle samples (np.interp needs strictly increasing x)
    keep = np.concatenate([[True], np.diff(angle) > 1e-9])
    angle, iq = angle[keep], iq[keep]
    if len(angle) < 8:
        return CoggingFit(float("nan"), float("nan"), len(angle))

    span_deg = math.degrees(angle[-1] - angle[0])
    grid = np.linspace(angle[0], angle[-1], n_resample)
    iq_grid = np.interp(grid, angle, iq)
    iq_grid = iq_grid - np.mean(iq_grid)

    spacing_deg = span_deg / (n_resample - 1)
    spectrum = np.fft.rfft(iq_grid)
    freqs_per_deg = np.fft.rfftfreq(n_resample, d=spacing_deg)  # cycles/deg
    with np.errstate(divide="ignore"):
        period_deg_axis = np.where(freqs_per_deg > 0, 1.0 / freqs_per_deg, np.inf)
    mask = (period_deg_axis >= min_period_deg) & (period_deg_axis <= max_period_deg)
    if not np.any(mask):
        return CoggingFit(0.0, float("nan"), len(angle))

    mags = np.abs(spectrum)
    idx_local = np.argmax(mags[mask])
    idx = np.flatnonzero(mask)[idx_local]
    amplitude_a = float(2.0 * mags[idx] / n_resample)  # single-sided amplitude
    period_deg = float(period_deg_axis[idx])
    return CoggingFit(amplitude_a, period_deg, len(angle))
