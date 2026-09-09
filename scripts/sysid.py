"""Rerunnable system-identification suite, one joint at a time (the other
joint held in position mode, low stiffness -- same pattern as breakaway.py).

    python -m scripts.sysid --joint 1 --mode latency
    python -m scripts.sysid --joint 1 --mode chirp --amp-a 0.25 --f0 1 --f1 40 --duration 12
    python -m scripts.sysid --joint 0 --mode all --label "links v1, harness B"

Everything here is expressed in **current (A)**, not torque -- torque_constant
is unverified, so baking it into a fit would silently scale every derived
number by that error. ``Set_Input_Torque`` still wants N.m on the wire (the
drive's own Kt does A<->N.m on-board); this script converts current -> N.m
only immediately before that one call, and logs/reports amps everywhere else.
See ``panto/sysid_logic.py`` for the pure estimators (unit-tested on
synthetic data, no CAN/sim needed).

Modes
-----
``latency``   torque(current) step, --amp-a for 0.3s x5 (1s gaps). Estimates
              command->Iq and command->velocity latency, and the feedback-age
              distribution.
``chirp``     torque(current) chirp, --amp-a, log f0->f1 over --duration,
              superimposed on zero. Estimates the frequency response
              I->qd (Welch/CSD), fits inertia (A.s^2/rad) + pure delay, flags
              resonances. Writes summary.json + sysid_<joint>.png.
``friction``  per direction: (1) slow current ramp to breakaway (small motion
              threshold to ignore drift, like breakaway.py) for static
              current; (2) constant-velocity position-mode ramps (this
              firmware only exposes position/torque control modes, so
              "velocity mode" here is a constant-rate position ramp) at 3-4
              speeds for kinetic current vs. speed.
``cogging``   constant-rate position ramp (default 0.05 turn/s) over +/-30deg
              from the current position (clamped to joint limits), both
              directions. Logs Iq vs. angle; reports the periodic
              component's amplitude (A) and dominant spatial period (deg).
``velnoise``  HARDWARE ONLY (raises under --sim): sweeps encoder_bandwidth
              via /tmp/odrive_set_encbw.py on this host (a standalone
              odrivetool script -- see its docstring), sampling vel_estimate
              noise (idle) at each setting, then restores 300. Does not open
              a CanLink of its own while that helper runs (its own docstring
              warns against a concurrent bus owner).
``all``       runs latency, chirp, friction, cogging (+ velnoise if not
              --sim) per joint in that order with cool-downs, and writes one
              versioned ``plant_model.json`` (timestamped, ``--label``) plus
              a markdown report and the chirp plots.

Guards active throughout: panto.limits joint-limit trip wire, --max-deg
motion cap, --max-excursion-mm excursion cap from config.test_pose, stale
feedback abort, and drive-disarm abort -- same as breakaway.py/torque_step.py.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from panto.breakaway_logic import check_breakaway, plateau_vel_limit_rad_s
from panto.can_link import AXIS_STATE_CLOSED_LOOP_CONTROL, CanLink, CanLinkError, decode_error_flags
from panto.config import Config
from panto.kinematics import forward
from panto.limits import JointLimitViolation, check_armable, check_runtime, format_limits_deg, q_deg
from panto.sysid_logic import (
    CoggingFit,
    FreqResponse,
    InertiaDelayFit,
    LatencyEstimate,
    amp_profile_scale,
    cogging_spectrum,
    estimate_latency,
    fit_inertia_delay,
    fit_kinetic_friction,
    infer_current_sign,
    log_chirp,
    predicted_low_freq_excursion_deg,
    profiled_chirp,
    select_coherent_band,
    welch_csd,
)
from panto.telemetry import RunLogger

MAX_FEEDBACK_AGE_S = 0.1
JOINT_NAMES = {0: "shoulder", 1: "elbow"}
ENCBW_HELPER = Path("/tmp/odrive_set_encbw.py")
DEFAULT_ENCBW = 300.0   # setting of record since 2026-09-08 (saved on both drives)


class Aborted(RuntimeError):
    pass


class DriveDisarmed(Aborted):
    pass


def _hold_other(link, config, other_idx, other_q, hold_k_nm_rad):
    """One tick of position-mode holding for the non-test joint. Same
    scalar-stiffness mapping as scripts/breakaway.py's helper of the same
    name (duplicated, not imported, to keep sysid.py independent of
    breakaway.py's own CLI/main entrypoint)."""
    motor = config.motors[other_idx]
    vel_gain = motor.vel_gain
    pos_gain = 0.0 if vel_gain <= 0 else min(hold_k_nm_rad / vel_gain, motor.max_pos_gain)
    link.set_input_pos(motor.node_id, float(other_q))
    link.set_pos_gain(motor.node_id, pos_gain)


def _guard(link, config, log, joint_idx, start_q, max_deg, excursion_ref_m, max_excursion_mm):
    """One tick's worth of abort checks, common to every mode. Returns the
    current (q_all, qd_all, moved_deg) so callers don't re-read joint_state."""
    age_s = link.feedback_age_s()
    if age_s > MAX_FEEDBACK_AGE_S:
        raise Aborted(f"feedback age {age_s*1e3:.1f}ms exceeds cap")
    for s in link.node_status():
        if s.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
            reason = decode_error_flags(s.disarm_reason or 0)
            raise DriveDisarmed(f"node {s.node_id} left CLOSED_LOOP reason={reason}")
    q_all, qd_all = link.joint_state()
    check_runtime(q_all, config.motors)
    pose = forward(q_all, config.geo)
    if excursion_ref_m is not None:
        exc_mm = float(np.linalg.norm(pose - excursion_ref_m) * 1e3)
        if exc_mm > max_excursion_mm:
            raise Aborted(f"excursion {exc_mm:.1f}mm exceeds cap {max_excursion_mm}mm")
    moved_deg = float(np.degrees(q_all[joint_idx] - start_q))
    if abs(moved_deg) > max_deg:
        raise Aborted(f"joint moved {moved_deg:.2f}deg exceeds --max-deg={max_deg}")
    return q_all, qd_all, moved_deg


def _current_to_nm(current_a: float, torque_constant: float) -> float:
    return current_a * torque_constant


# --------------------------------------------------------------------------
# mode: latency
# --------------------------------------------------------------------------

def run_latency(link, config, log, joint_idx, other_idx, start_q, other_q, excursion_ref_m,
                args) -> dict:
    motor = config.motors[joint_idx]
    tau_nm = _current_to_nm(args.amp_a, motor.torque_constant)
    period = 1.0 / args.rate
    trials = []
    n_steps = 5
    pulse_s = args.pulse_s
    # 0.2s pre-pulse (for the noise-floor/baseline estimate), pulse_s at amp,
    # 0.2s tail for decay
    pre_s = 0.2
    for trial_i in range(n_steps):
        t0 = time.monotonic()
        t_cmd = None
        rows = []
        pulse_on_at = pre_s
        pulse_off_at = pulse_on_at + pulse_s
        end_at = pulse_off_at + 0.2
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= end_at:
                break
            q_all, qd_all, _moved = _guard(link, config, log, joint_idx, start_q[joint_idx],
                                           args.max_deg, excursion_ref_m, args.max_excursion_mm)
            active = pulse_on_at <= elapsed < pulse_off_at
            if active and t_cmd is None:
                t_cmd = time.monotonic()
            tau = tau_nm if active else 0.0
            link.set_input_torque(motor.node_id, tau)
            _hold_other(link, config, other_idx, other_q, args.hold_stiffness)
            t_now = time.monotonic()
            cur = link.motor_currents()
            age_s = link.feedback_age_s()
            rows.append((t_now, args.amp_a if active else 0.0, float(cur[joint_idx]),
                        float(qd_all[joint_idx]), age_s))
            log.sample(tag="latency", trial=trial_i, t=elapsed, i_cmd_a=(args.amp_a if active else 0.0),
                      iq_a=float(cur[joint_idx]), qd=float(qd_all[joint_idx]), age_s=age_s)
            time.sleep(period)
        link.set_input_torque(motor.node_id, 0.0)
        if t_cmd is None:
            t_cmd = t0 + pulse_on_at
        arr = np.array(rows)
        trials.append({
            "t_cmd": t_cmd, "t": arr[:, 0].tolist(), "i_cmd": arr[:, 1].tolist(),
            "t_iq": arr[:, 0].tolist(), "iq": arr[:, 2].tolist(),
            "t_vel": arr[:, 0].tolist(), "vel": arr[:, 3].tolist(),
            "feedback_age_s": arr[:, 4].tolist(),
        })
        time.sleep(1.0)

    return _finish_latency(trials, joint_idx, args, pulse_s)


def _finish_latency(trials: list[dict], joint_idx: int, args, pulse_s: float) -> dict:
    """Shared by the live run and --analyze: sign-correct Iq per trial (the
    drive's Iq broadcast can disagree in sign with the commanded current --
    see infer_current_sign's docstring), then estimate latency on the
    corrected traces."""
    corrected = []
    signs = []
    for tr in trials:
        i_cmd = np.asarray(tr["i_cmd"], float)
        iq = np.asarray(tr["iq"], float)
        active = i_cmd != 0.0
        sign = infer_current_sign(i_cmd, iq, active) if np.any(active) else 1
        signs.append(sign)
        tr2 = dict(tr)
        tr2["iq"] = (sign * iq).tolist()
        corrected.append(tr2)
    iq_sign = int(np.sign(np.sum(signs))) if signs else 1

    est = estimate_latency(corrected, pulse_s=pulse_s, vel_noise_sigma=args.vel_noise_sigma)
    print(f"  iq_latency={_fmt_ms(est.iq_latency_s)}"
          + (f" ({est.iq_reason})" if est.iq_latency_s is None else "")
          + f"  vel_latency={_fmt_ms(est.vel_latency_s)}"
          + (f" ({est.vel_reason})" if est.vel_latency_s is None else "")
          + f"  feedback_age: mean={est.feedback_age_mean_s*1e3:.2f}ms p50={est.feedback_age_p50_s*1e3:.2f}ms "
          f"p95={est.feedback_age_p95_s*1e3:.2f}ms max={est.feedback_age_max_s*1e3:.2f}ms  "
          f"(n={est.n_trials} trials)")
    if iq_sign < 0:
        print(f"  ! measured Iq sign is FLIPPED vs. commanded current on this node "
              f"(iq_sign={iq_sign}) -- corrected before estimating latency")
    return {"mode": "latency", "joint": JOINT_NAMES[joint_idx], "amp_a": args.amp_a,
            "pulse_s": pulse_s, "iq_sign": iq_sign,
            "trials": trials,
            **{k: v for k, v in asdict(est).items()}}


def _fmt_ms(x):
    return "n/a" if x is None else f"{x*1e3:.2f}ms"


# --------------------------------------------------------------------------
# mode: chirp
# --------------------------------------------------------------------------

def run_chirp(link, config, log, joint_idx, other_idx, start_q, other_q, excursion_ref_m, args):
    motor = config.motors[joint_idx]
    period = 1.0 / args.rate
    t0 = time.monotonic()
    rows = []
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= args.duration:
            break
        q_all, qd_all, _moved = _guard(link, config, log, joint_idx, start_q[joint_idx],
                                       args.max_deg, excursion_ref_m, args.max_excursion_mm)
        i_cmd = float(profiled_chirp(np.array([elapsed]), args.amp_a, args.f0, args.f1,
                                     args.duration, args.amp_profile)[0])
        tau_nm = _current_to_nm(i_cmd, motor.torque_constant)
        link.set_input_torque(motor.node_id, tau_nm)
        _hold_other(link, config, other_idx, other_q, args.hold_stiffness)
        cur = link.motor_currents()
        rows.append((elapsed, i_cmd, float(cur[joint_idx]), float(qd_all[joint_idx]),
                    float(q_all[joint_idx]), link.feedback_age_s()))
        log.sample(tag="chirp", t=elapsed, i_cmd_a=i_cmd, iq_a=float(cur[joint_idx]),
                  qd=float(qd_all[joint_idx]), q=float(q_all[joint_idx]))
        time.sleep(period)
    link.set_input_torque(motor.node_id, 0.0)

    arr = np.array(rows)
    t, i_cmd, iq_a, qd, q, age = arr.T
    return _finish_chirp(t, i_cmd, iq_a, qd, q, joint_idx, args)


def _finish_chirp(t, i_cmd, iq_a, qd, q, joint_idx, args) -> dict:
    """Post-processing shared by the live run and --analyze: sign-correct
    the measured Iq against the command, use THAT (not the command) as the
    frequency-response input -- 2026-09-08 hardware finding: the elbow's
    measured Iq comes back opposite in sign to the commanded current, so
    identifying against the command conflates the command-path sign with the
    actual plant and (combined with low-coherence bins) produced a nonsense
    negative delay. Identifying Iq->velocity instead is independent of
    whatever sign convention the command path happens to use."""
    t = np.asarray(t, float)
    i_cmd = np.asarray(i_cmd, float)
    iq_a = np.asarray(iq_a, float)
    qd = np.asarray(qd, float)
    q = np.asarray(q, float)

    active = np.abs(i_cmd) > 1e-9
    iq_sign = infer_current_sign(i_cmd, iq_a, active) if np.any(active) else 1
    iq_corrected = iq_sign * iq_a

    # sample at whatever rate the loop actually achieved, not the requested
    # --rate -- CAN I/O + guard checks make the realized loop rate slower
    # than nominal, and Welch/CSD's frequency axis is only correct if fs
    # matches the real sample spacing. Median dt, not (n-1)/(t[-1]-t[0]) --
    # a single stalled tick (a print, a GC pause) skews the mean interval
    # but not the median, and this fs feeds directly into the Hz axis.
    dt = np.diff(t)
    fs_actual = float(1.0 / np.median(dt)) if len(dt) > 0 else args.rate
    fr = welch_csd(iq_corrected, qd, fs=fs_actual)

    # Fit window: prefer the data's own longest contiguous high-coherence
    # band over a fixed "top half of the sweep" guess -- 2026-09-08 hardware:
    # that fixed guess landed entirely in a noise-dominated region on a real
    # joint while a perfectly good ~1-14Hz coherent band sat lower in the
    # sweep, and the fit silently returned NaN (zero surviving bins) instead
    # of using it. --fit-fmin still overrides explicitly when given.
    if args.fit_fmin is not None:
        fit_fmin, fit_fmax = args.fit_fmin, args.f1
    else:
        band = select_coherent_band(fr.freq_hz, fr.coherence, coherence_min=args.coherence_min,
                                    f_floor=max(args.f0, 1.0), f_ceil=args.f1)
        if band is not None:
            fit_fmin, fit_fmax = band
        else:
            fit_fmin, fit_fmax = max(args.f0, 0.5 * (args.f0 + args.f1)), args.f1
    fit = fit_inertia_delay(fr, f_min=fit_fmin, f_max=fit_fmax,
                            resonance_prominence_db=args.resonance_prominence_db,
                            coherence_min=args.coherence_min)
    print(f"  fit window: [{fit_fmin:.2f}, {fit_fmax:.2f}] Hz "
          f"(auto from coherence)" if args.fit_fmin is None else
          f"  fit window: [{fit_fmin:.2f}, {fit_fmax:.2f}] Hz (--fit-fmin override)")
    print(f"  fit: J={fit.inertia_kg_m2:.4g} A.s^2/rad  delay={fit.delay_s*1e3:.3f}ms  "
          f"(n_lowfreq={fit.low_freq_mask_n}, coherence range in fit band="
          f"{fit.coherence_at_fmin_fmax})")
    if iq_sign < 0:
        print(f"  ! measured Iq sign is FLIPPED vs. commanded current on this node "
              f"(iq_sign={iq_sign}) -- corrected before fitting")
    if fit.resonances_hz:
        print(f"  resonances (Hz): {[round(f,1) for f in fit.resonances_hz]}")
    if fit.antiresonances_hz:
        print(f"  anti-resonances (Hz): {[round(f,1) for f in fit.antiresonances_hz]}")

    trace = {"t": t.tolist(), "i_cmd_a": i_cmd.tolist(), "iq_a": iq_a.tolist(),
             "iq_corrected_a": iq_corrected.tolist(), "qd": qd.tolist(), "q": q.tolist()}
    plot_path = None
    if not args.no_plot:
        plot_path = _plot_chirp(joint_idx, trace, fr, args)

    return {
        "mode": "chirp", "joint": JOINT_NAMES[joint_idx], "amp_a": args.amp_a,
        "amp_profile": args.amp_profile,
        "f0_hz": args.f0, "f1_hz": args.f1, "duration_s": args.duration,
        "iq_sign": iq_sign, "coherence_min": args.coherence_min,
        "inertia_a_s2_per_rad": fit.inertia_kg_m2, "delay_s": fit.delay_s,
        "resonances_hz": fit.resonances_hz, "antiresonances_hz": fit.antiresonances_hz,
        "coherence_at_fmin_fmax": fit.coherence_at_fmin_fmax,
        "freq_hz": fr.freq_hz.tolist(), "mag": fr.mag.tolist(), "phase_rad": fr.phase_rad.tolist(),
        "coherence": fr.coherence.tolist() if fr.coherence is not None else None,
        "plot": str(plot_path) if plot_path else None,
    }


def _plot_chirp(joint_idx, trace, fr: FreqResponse, args) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # palette per the dataviz skill's validated categorical set (light mode),
    # reused from scripts/plot_step_response.py for visual consistency
    BLUE = "#2a78d6"
    ORANGE = "#eb6834"
    AQUA = "#1baf7a"
    TEXT_PRIMARY = "#0b0b0b"
    TEXT_SECONDARY = "#52514e"
    GRID = "#e4e2dc"

    fig, axes = plt.subplots(4, 1, figsize=(8, 11))
    ax_t, ax_mag, ax_phase, ax_coh = axes

    ax_t.plot(trace["t"], trace["i_cmd_a"], color=BLUE, lw=1.5, label="I cmd (A)")
    ax_t.plot(trace["t"], trace.get("iq_corrected_a", trace["iq_a"]), color=ORANGE, lw=1.0,
             label="Iq measured, sign-corrected (A)")
    ax_t.set_xlabel("t (s)", color=TEXT_SECONDARY)
    ax_t.set_ylabel("A", color=TEXT_SECONDARY)
    ax_t.set_title(f"{JOINT_NAMES[joint_idx]}: chirp time trace", color=TEXT_PRIMARY)
    ax_t.legend(frameon=False)
    ax_t.grid(color=GRID, lw=0.5)

    mask = fr.freq_hz > 0
    ax_mag.plot(fr.freq_hz[mask], 20 * np.log10(np.clip(fr.mag[mask], 1e-12, None)), color=BLUE, lw=1.5)
    ax_mag.set_xscale("log")
    ax_mag.set_ylabel("|H| (dB re 1 rad/s/A)", color=TEXT_SECONDARY)
    ax_mag.set_title("magnitude", color=TEXT_PRIMARY)
    ax_mag.grid(color=GRID, lw=0.5)

    ax_phase.plot(fr.freq_hz[mask], np.degrees(fr.phase_rad[mask]), color=ORANGE, lw=1.5)
    ax_phase.set_xscale("log")
    ax_phase.set_ylabel("phase (deg)", color=TEXT_SECONDARY)
    ax_phase.set_title("phase", color=TEXT_PRIMARY)
    ax_phase.grid(color=GRID, lw=0.5)

    if fr.coherence is not None:
        ax_coh.plot(fr.freq_hz[mask], fr.coherence[mask], color=AQUA, lw=1.5)
        ax_coh.axhline(args.coherence_min, color="#e34948", lw=1.0, ls="--",
                       label=f"fit cutoff ({args.coherence_min})")
        ax_coh.legend(frameon=False)
    ax_coh.set_xscale("log")
    ax_coh.set_xlabel("Hz", color=TEXT_SECONDARY)
    ax_coh.set_ylabel("coherence", color=TEXT_SECONDARY)
    ax_coh.set_ylim(0, 1.05)
    ax_coh.set_title("Pxy/Pxx coherence (fit only uses bins above the cutoff)", color=TEXT_PRIMARY)
    ax_coh.grid(color=GRID, lw=0.5)

    fig.tight_layout()
    out_dir = Path(args.plot_dir) if args.plot_dir else Path("/tmp/panto_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sysid_{JOINT_NAMES[joint_idx]}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


DEFAULT_J_GUESS_A_S2_PER_RAD = 0.001  # deliberately stiff/low; see --j-guess help


def _preflight_chirp_excursion(args) -> None:
    """Abort BEFORE arming if the chirp's predicted low-frequency excursion
    exceeds --max-deg -- 2026-09-08: a flat 0.25A chirp threw a free elbow
    69deg in 0.3s at the sweep's low-frequency start; --max-deg only catches
    that mid-run, after the joint has already moved. This is a cheap,
    deliberately pessimistic check (see
    panto.sysid_logic.predicted_low_freq_excursion_deg), not a substitute for
    the runtime guard."""
    j_guess = args.j_guess if args.j_guess is not None else DEFAULT_J_GUESS_A_S2_PER_RAD
    predicted_deg = predicted_low_freq_excursion_deg(args.amp_a, args.f0, args.f1,
                                                     args.amp_profile, j_guess)
    print(f"  pre-arm check: predicted low-freq excursion ~= {predicted_deg:.1f}deg "
          f"(amp={args.amp_a}A, f0={args.f0}Hz, profile={args.amp_profile}, "
          f"j_guess={j_guess}A.s^2/rad)")
    if predicted_deg > args.max_deg:
        raise SystemExit(
            f"refusing to arm: predicted low-freq chirp excursion {predicted_deg:.1f}deg "
            f"exceeds --max-deg={args.max_deg}deg. Lower --amp-a, raise --f0, switch "
            f"--amp-profile const-vel, or pass a larger --j-guess if you know this joint "
            f"is heavier than the conservative default ({DEFAULT_J_GUESS_A_S2_PER_RAD} A.s^2/rad).")


# --------------------------------------------------------------------------
# mode: friction
# --------------------------------------------------------------------------

def _ramp_to_breakaway(link, config, log, joint_idx, other_idx, start_q, other_q, sign,
                       excursion_ref_m, args) -> float:
    """Same shape as breakaway.py's ramp -- returns |Iq| at breakaway (small
    --break-deg motion threshold so slow drift doesn't false-trigger)."""
    motor = config.motors[joint_idx]
    period = 1.0 / args.rate
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        q_all, qd_all, moved_deg = _guard(link, config, log, joint_idx, start_q, args.max_deg,
                                          excursion_ref_m, args.max_excursion_mm)
        i_cmd = args.friction_ramp_a_s * elapsed * sign
        tau_nm = _current_to_nm(i_cmd, motor.torque_constant)
        link.set_input_torque(motor.node_id, tau_nm)
        _hold_other(link, config, other_idx, other_q, args.hold_stiffness)
        cur = link.motor_currents()
        result = check_breakaway(elapsed, moved_deg, float(cur[joint_idx]),
                                 rate_nm_s=_current_to_nm(args.friction_ramp_a_s, motor.torque_constant),
                                 sign=sign, current_cap_a=motor.current_soft_max,
                                 break_deg=args.break_deg, abort_deg=args.max_deg, abort_s=15.0)
        log.sample(tag="friction_static", joint=joint_idx, sign=sign, t=elapsed,
                  i_cmd_a=i_cmd, iq_a=float(cur[joint_idx]), moved_deg=moved_deg)
        if result is not None:
            link.set_input_torque(motor.node_id, 0.0)
            return abs(result.current_a), result.status
        time.sleep(period)


def _kinetic_segment(link, config, log, joint_idx, other_idx, start_q, other_q, sign, speed_rad_s,
                     excursion_ref_m, args) -> float:
    """Constant-rate position ramp ("velocity mode" substitute -- this
    firmware only exposes position/torque control). Returns the mean |Iq|
    over the settled (second-half) portion of the ramp."""
    motor = config.motors[joint_idx]
    period = 1.0 / args.rate
    duration = args.friction_seg_deg / max(1e-6, math.degrees(speed_rad_s))
    duration = min(duration, args.friction_seg_max_s)
    pos_gain = 0.0 if motor.vel_gain <= 0 else min(40.0 / motor.vel_gain, motor.max_pos_gain)
    t0 = time.monotonic()
    currents = []
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration:
            break
        q_all, qd_all, _moved = _guard(link, config, log, joint_idx, start_q, args.max_deg,
                                       excursion_ref_m, args.max_excursion_mm)
        q_cmd = start_q + sign * speed_rad_s * elapsed
        link.set_input_pos(motor.node_id, float(q_cmd))
        link.set_pos_gain(motor.node_id, pos_gain)
        _hold_other(link, config, other_idx, other_q, args.hold_stiffness)
        cur = link.motor_currents()
        if elapsed >= duration / 2.0:
            currents.append(float(cur[joint_idx]))
        log.sample(tag="friction_kinetic", joint=joint_idx, sign=sign, speed_rad_s=speed_rad_s,
                  t=elapsed, iq_a=float(cur[joint_idx]), q=float(q_all[joint_idx]))
        time.sleep(period)
    # settle back toward start won't be done here -- caller re-homes between segments
    return float(np.mean(currents)) if currents else float("nan")


def run_friction(link, config, log, joint_idx, other_idx, start_q, other_q, excursion_ref_m, args):
    motor = config.motors[joint_idx]
    per_dir = {}
    speeds = args.friction_speeds_rad_s
    for sign, dirname in ((+1, "+"), (-1, "-")):
        link.set_controller_mode(motor.node_id, "torque")
        vel_limit_rad_s, _ = plateau_vel_limit_rad_s(motor.current_soft_max, motor.torque_constant,
                                                      args.torque_vel_gain)
        link.set_vel_gains(motor.node_id, args.torque_vel_gain, 0.0)
        link.set_limits(motor.node_id, vel_limit_rad_s, motor.current_soft_max)
        static_a, status = _ramp_to_breakaway(link, config, log, joint_idx, other_idx,
                                              start_q[joint_idx], other_q, sign, excursion_ref_m, args)
        print(f"  {JOINT_NAMES[joint_idx]} {dirname}: static={static_a:.3f}A ({status})")

        # back to position mode, return to start, before the kinetic segments
        link.set_controller_mode(motor.node_id, "position")
        link.set_limits(motor.node_id, args.vel_limit, motor.current_soft_max)
        _ramp_position(link, config, log, joint_idx, other_idx, start_q[joint_idx], other_q,
                       excursion_ref_m, args)

        kinetic_currents = []
        for speed in speeds:
            i_a = _kinetic_segment(link, config, log, joint_idx, other_idx, start_q[joint_idx],
                                   other_q, sign, speed, excursion_ref_m, args)
            kinetic_currents.append(abs(i_a))
            _ramp_position(link, config, log, joint_idx, other_idx, start_q[joint_idx], other_q,
                           excursion_ref_m, args)
        intercept, slope = fit_kinetic_friction(speeds, kinetic_currents)
        print(f"    kinetic vs speed (A): {[round(x,3) for x in kinetic_currents]}  "
              f"-> intercept={intercept:.3f}A slope={slope:.4f}A/(rad/s)")
        per_dir[dirname] = {
            "static_a": static_a, "static_status": status,
            "speeds_rad_s": speeds, "kinetic_a": kinetic_currents,
            "kinetic_intercept_a": intercept, "kinetic_slope_a_per_rad_s": slope,
        }
    link.set_input_torque(motor.node_id, 0.0)
    return {"mode": "friction", "joint": JOINT_NAMES[joint_idx], **per_dir}


def _ramp_position(link, config, log, joint_idx, other_idx, target_q, other_q, excursion_ref_m,
                   args, ramp_s=2.0):
    motor = config.motors[joint_idx]
    period = 1.0 / args.rate
    q_all, _ = link.joint_state()
    q0 = float(q_all[joint_idx])
    pos_gain = 0.0 if motor.vel_gain <= 0 else min(40.0 / motor.vel_gain, motor.max_pos_gain)
    t0 = time.monotonic()
    while time.monotonic() - t0 < ramp_s:
        elapsed = time.monotonic() - t0
        frac = min(1.0, elapsed / ramp_s)
        q_cmd = q0 + frac * (target_q - q0)
        link.set_input_pos(motor.node_id, float(q_cmd))
        link.set_pos_gain(motor.node_id, pos_gain)
        _guard(link, config, log, joint_idx, target_q, args.max_deg, excursion_ref_m,
              args.max_excursion_mm)
        _hold_other(link, config, other_idx, other_q, args.hold_stiffness)
        time.sleep(period)


# --------------------------------------------------------------------------
# mode: cogging
# --------------------------------------------------------------------------

def run_cogging(link, config, log, joint_idx, other_idx, start_q, other_q, excursion_ref_m, args):
    motor = config.motors[joint_idx]
    period = 1.0 / args.rate
    span_rad = math.radians(args.cogging_span_deg)
    lo = start_q[joint_idx] - span_rad
    hi = start_q[joint_idx] + span_rad
    if motor.q_min_rad > float("-inf"):
        lo = max(lo, motor.q_min_rad + motor.limit_margin_rad)
    if motor.q_max_rad < float("inf"):
        hi = min(hi, motor.q_max_rad - motor.limit_margin_rad)
    speed = math.radians(args.cogging_speed_deg_s)
    pos_gain = 0.0 if motor.vel_gain <= 0 else min(40.0 / motor.vel_gain, motor.max_pos_gain)

    fits = {}
    for dirname, (q_from, q_to) in {"+": (lo, hi), "-": (hi, lo)}.items():
        duration = abs(q_to - q_from) / max(1e-6, speed)
        angles, currents = [], []
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= duration:
                break
            frac = elapsed / duration
            q_cmd = q_from + frac * (q_to - q_from)
            link.set_input_pos(motor.node_id, float(q_cmd))
            link.set_pos_gain(motor.node_id, pos_gain)
            _guard(link, config, log, joint_idx, start_q[joint_idx], args.max_deg + args.cogging_span_deg,
                  excursion_ref_m, args.max_excursion_mm)
            _hold_other(link, config, other_idx, other_q, args.hold_stiffness)
            q_all, _ = link.joint_state()
            cur = link.motor_currents()
            angles.append(float(q_all[joint_idx]))
            currents.append(float(cur[joint_idx]))
            log.sample(tag="cogging", joint=joint_idx, direction=dirname, t=elapsed,
                      q=float(q_all[joint_idx]), iq_a=float(cur[joint_idx]))
            time.sleep(period)
        fit = cogging_spectrum(np.array(angles), np.array(currents))
        print(f"  {JOINT_NAMES[joint_idx]} cogging {dirname}: amplitude={fit.amplitude_a:.4f}A "
              f"period={fit.period_deg:.2f}deg (n={fit.n_samples})")
        fits[dirname] = asdict(fit)
        _ramp_position(link, config, log, joint_idx, other_idx, start_q[joint_idx], other_q,
                       excursion_ref_m, args)
    return {"mode": "cogging", "joint": JOINT_NAMES[joint_idx], **fits}


# --------------------------------------------------------------------------
# mode: velnoise (hardware only, no CanLink of its own -- shells out)
# --------------------------------------------------------------------------

def run_velnoise(args) -> dict:
    if args.sim:
        raise SystemExit("--mode velnoise is hardware-only (no cogging/encoder model in --sim); "
                         "refusing under --sim")
    if not ENCBW_HELPER.exists():
        raise SystemExit(f"{ENCBW_HELPER} not found -- this mode shells out to that standalone "
                         "odrivetool script; see its docstring")
    bandwidths = args.velnoise_bandwidths
    results = {}
    for bw in bandwidths:
        print(f"  setting encoder_bandwidth -> {bw} ...")
        out = subprocess.run(
            ["uv", "run", "--with", "odrive==0.6.11.post1", "--offline", "python",
            str(ENCBW_HELPER), str(bw)],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            raise SystemExit(f"encbw helper failed at bw={bw}: {out.stderr}")
        print(out.stdout)
        stds = _parse_encbw_stdout(out.stdout)
        results[str(bw)] = stds
    if DEFAULT_ENCBW not in bandwidths:
        subprocess.run(["uv", "run", "--with", "odrive==0.6.11.post1", "--offline", "python",
                       str(ENCBW_HELPER), str(DEFAULT_ENCBW)], capture_output=True, text=True, timeout=60)
        print(f"  restored encoder_bandwidth -> {DEFAULT_ENCBW}")
    return {"mode": "velnoise", "bandwidths": bandwidths, "std_by_bandwidth": results,
            "restored_to": DEFAULT_ENCBW}


def _parse_encbw_stdout(text: str) -> dict:
    """Pull {serial: std} out of /tmp/odrive_set_encbw.py's printed table
    (last two whitespace-separated columns of each data row)."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] not in ("serial",):
            try:
                serial, mean, std = parts[0], float(parts[1]), float(parts[2])
            except ValueError:
                continue
            out[serial] = std
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sysid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface")
    p.add_argument("--channel")
    p.add_argument("--config")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--analyze", type=str, default=None, metavar="LOGDIR",
                  help="offline mode: re-run post-processing (fits + plots) from an existing "
                       "run's meta.json/samples.jsonl, no bus/sim access at all -- safe on a "
                       "laptop. Ignores --joint/--mode/--sim (read from meta.json)")
    p.add_argument("--joint", type=int, choices=(0, 1))
    p.add_argument("--mode", choices=("latency", "chirp", "friction", "cogging", "velnoise", "all"))
    p.add_argument("--label", type=str, default="", help="free-text tag for plant_model.json (e.g. "
                                                          "'links v1, harness B')")
    p.add_argument("--amp-a", type=float, default=None, help="commanded current amplitude, A "
                   "(default 0.25 for chirp, 6mN.m-equivalent legacy default for latency -- "
                   "pass explicitly to be sure)")
    p.add_argument("--f0", type=float, default=1.0, help="chirp start freq, Hz")
    p.add_argument("--f1", type=float, default=40.0, help="chirp end freq, Hz")
    p.add_argument("--duration", type=float, default=12.0, help="chirp duration, s")
    p.add_argument("--amp-profile", choices=("flat", "const-vel"), default="const-vel",
                  help="chirp amplitude envelope. 'flat': constant current amplitude across the "
                       "sweep (the original behaviour -- 2026-09-08: threw a free elbow 69deg in "
                       "0.3s at the chirp's low-frequency start). 'const-vel' (default): scales "
                       "amplitude by f/f1 so a roughly-inertial plant sees roughly constant "
                       "velocity amplitude across the sweep instead of blowing up at low f")
    p.add_argument("--j-guess", type=float, default=None,
                  help="A.s^2/rad, a deliberately LOW (stiff) guess at the joint's current-"
                       "referenced inertia, used only for the pre-arm excursion check below -- "
                       "not a plant model. Default: 0.001 A.s^2/rad (roughly the lightest joint "
                       "seen on this rig; override with a smaller number to be more conservative "
                       "if a joint has ever felt lighter than that)")
    p.add_argument("--pulse-s", type=float, default=0.1, help="latency mode: pulse width, s "
                   "(the drive-to-Iq/vel latency chain settles well inside 0.1s; a free joint "
                   "at a small --amp-a does not need a long pulse to respond measurably)")
    p.add_argument("--vel-noise-sigma", type=float, default=5.0, help="latency mode: velocity-"
                   "latency crossing threshold, multiples of the pre-pulse velocity noise std "
                   "(not a fixed rad/s -- a free joint and a loaded one have very different "
                   "noise floors)")
    p.add_argument("--max-deg", type=float, default=25.0, help="abort if the joint moves more than this")
    p.add_argument("--max-excursion-mm", type=float, default=50.0)
    p.add_argument("--hold-stiffness", type=float, default=15.0, help="N.m/rad, held joint's gain")
    p.add_argument("--vel-limit", type=float, default=10.0, help="ODrive vel_limit, position moves")
    p.add_argument("--torque-vel-gain", type=float, default=0.01)
    p.add_argument("--rate", type=float, default=500.0, help="control/log loop Hz")
    p.add_argument("--fit-fmin", type=float, default=None, help="Hz, lower bound of the "
                   "high-frequency window used to fit inertia/delay (default: midpoint of "
                   "[f0, f1] -- the fit needs frequencies where inertia, not damping, "
                   "dominates the response; see panto.sysid_logic.fit_inertia)")
    p.add_argument("--coherence-min", type=float, default=0.8, help="chirp mode: minimum "
                   "Pxy/Pxx coherence for a frequency bin to count toward the inertia/delay fit "
                   "-- excludes noise-dominated bins that would otherwise bias (even flip the "
                   "sign of) the delay estimate; see panto.sysid_logic.fit_delay's docstring "
                   "for the 2026-09-08 hardware incident this fixes")
    p.add_argument("--resonance-prominence-db", type=float, default=3.0)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-dir", type=str, default=None)
    p.add_argument("--break-deg", type=float, default=2.0, help="friction mode: motion threshold "
                   "for static breakaway (ignores drift)")
    p.add_argument("--friction-ramp-a-s", type=float, default=0.09, help="A/s current ramp rate "
                   "for the static-friction ramp (~2mN.m/s at Kt=0.02235)")
    p.add_argument("--friction-speeds-rad-s", type=float, nargs="+",
                  default=[0.1, 0.3, 0.6, 1.0], help="kinetic-friction test speeds, rad/s")
    p.add_argument("--friction-seg-deg", type=float, default=10.0, help="deg travelled per "
                   "kinetic-friction speed segment")
    p.add_argument("--friction-seg-max-s", type=float, default=8.0)
    p.add_argument("--cogging-span-deg", type=float, default=30.0)
    p.add_argument("--cogging-speed-deg-s", type=float, default=0.05 * 360.0,
                  help="deg/s (default: 0.05 turn/s)")
    p.add_argument("--velnoise-bandwidths", type=float, nargs="+", default=[1000.0, 300.0, 100.0, 30.0])
    p.add_argument("--rest-s", type=float, default=2.0, help="cool-down between --mode all stages")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.analyze:
        try:
            analyze_logdir(Path(args.analyze), args)
        except SystemExit:
            raise
        except Exception:
            # --analyze runs offline, often re-triggered interactively while
            # staring at a plot -- a bare "exit 1, no output" (2026-09-08
            # report: "exits 2 with no output") is useless for debugging a
            # post-processing bug. Always show the real traceback here.
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
        return

    if args.joint is None or args.mode is None:
        raise SystemExit("--joint and --mode are required unless --analyze is given")

    if args.amp_a is None:
        args.amp_a = 0.25

    if args.mode == "velnoise":
        result = run_velnoise(args)
        print("\nSYSID_JSON " + json.dumps(result))
        return

    config = Config.load(args.config)
    if args.interface:
        config.can.interface = args.interface
    if args.channel:
        config.can.channel = args.channel

    joint_idx = args.joint
    other_idx = 1 - joint_idx
    test_pose_xy = config.test_pose_xy_m

    if args.mode in ("chirp", "all"):
        _preflight_chirp_excursion(args)

    # self-describing: everything downstream (--analyze, a human reading the
    # log dir cold) needs the sign conventions in play, not just the numbers
    # -- 2026-09-08: a node's Iq sign vs. commanded current turned out to
    # disagree, and figuring that out after the fact required re-deriving it
    # from raw samples. Record what we know up front; run_latency/run_chirp
    # additionally record the per-run INFERRED iq_sign in their own results.
    sign_conventions = {
        f"node{m.node_id}": {"flip": bool(m.flip), "torque_constant": m.torque_constant}
        for m in config.motors
    }
    log = RunLogger("sysid", interface=config.can.interface, channel=config.can.channel,
                    joint=args.joint, mode=args.mode, amp_a=args.amp_a, label=args.label,
                    sim=args.sim, cli_args=vars(args),
                    config=asdict(config), sign_conventions=sign_conventions,
                    joint_limits_deg=format_limits_deg(config.motors))
    link = CanLink(config, sim=args.sim)
    print(f"opening {config.can.interface}/{config.can.channel} (sim={args.sim})  "
          f"joint={joint_idx} ({JOINT_NAMES[joint_idx]})  mode={args.mode}")
    link.start()

    def idle_all():
        for m in config.motors:
            try:
                link.set_idle(m.node_id)
            except Exception as exc:  # noqa: BLE001
                log.event(f"idle node {m.node_id}: {exc}", level="ERROR")

    armed = False
    aborted = False
    disarm_summary = None
    default_vel_gain = None
    results = []
    try:
        link.wait_for_feedback(timeout=5.0, wait_for_errors=True)
        for s in link.node_status():
            if s.active_errors:
                raise SystemExit(f"node {s.node_id} has an active error at rest "
                                 f"(0x{s.active_errors:x}); clear it first")

        q0, _ = link.joint_state()
        pose0 = forward(q0, config.geo)
        print(f"  start q_deg={q_deg(q0)} pose=({pose0[0]*1e3:.1f},{pose0[1]*1e3:.1f})mm")
        log.event(f"start q_deg={q_deg(q0)}")

        problems = check_armable(q0, config.motors)
        if problems:
            raise SystemExit("refusing to arm -- " + "; ".join(problems))

        excursion_ref_m = test_pose_xy if test_pose_xy is not None else pose0.copy()

        for i, m in enumerate(config.motors):
            link.set_controller_mode(m.node_id, "position")
            link.set_limits(m.node_id, args.vel_limit, m.current_soft_max)
            link.set_input_pos(m.node_id, float(q0[i]))

        print(">>> entering CLOSED_LOOP_CONTROL <<<")
        try:
            link.enter_closed_loop(timeout=5.0)
        except CanLinkError as exc:
            raise SystemExit(f"refusing to arm: {exc}")
        armed = True
        for m in config.motors:
            link.clear_errors(m.node_id)
        default_vel_gain = config.motors[0].vel_gain

        start_q = {i: float(q0[i]) for i in (0, 1)}
        other_q = float(q0[other_idx])
        motor = config.motors[joint_idx]

        modes = [args.mode] if args.mode != "all" else (
            ["latency", "chirp", "friction", "cogging"] + ([] if args.sim else ["velnoise"]))
        for m in modes:
            print(f"\n== mode: {m} ==")
            link.set_vel_gains(motor.node_id, args.torque_vel_gain, 0.0)
            if m in ("latency", "chirp"):
                vel_limit_rad_s, _ = plateau_vel_limit_rad_s(motor.current_soft_max,
                                                              motor.torque_constant, args.torque_vel_gain)
                link.set_controller_mode(motor.node_id, "torque")
                link.set_limits(motor.node_id, vel_limit_rad_s, motor.current_soft_max)
                if m == "latency":
                    results.append(run_latency(link, config, log, joint_idx, other_idx, start_q,
                                               other_q, excursion_ref_m, args))
                else:
                    results.append(run_chirp(link, config, log, joint_idx, other_idx, start_q,
                                             other_q, excursion_ref_m, args))
                link.set_controller_mode(motor.node_id, "position")
                link.set_limits(motor.node_id, args.vel_limit, motor.current_soft_max)
                _ramp_position(link, config, log, joint_idx, other_idx, start_q[joint_idx], other_q,
                               excursion_ref_m, args)
            elif m == "friction":
                results.append(run_friction(link, config, log, joint_idx, other_idx, start_q,
                                            other_q, excursion_ref_m, args))
            elif m == "cogging":
                link.set_controller_mode(motor.node_id, "position")
                link.set_limits(motor.node_id, args.vel_limit, motor.current_soft_max)
                results.append(run_cogging(link, config, log, joint_idx, other_idx, start_q,
                                           other_q, excursion_ref_m, args))
            elif m == "velnoise":
                idle_all()
                results.append(run_velnoise(args))
                link.enter_closed_loop(timeout=5.0)
            if args.rest_s > 0 and m != modes[-1]:
                time.sleep(args.rest_s)

        summary = {"joint": JOINT_NAMES[joint_idx], "results": results}
        (log.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        # the full chirp freq/mag/phase arrays live in summary.json -- keep
        # stdout/events.log readable by trimming them from the printed copy
        trimmed = {
            "joint": summary["joint"],
            "results": [{k: v for k, v in r.items()
                       if k not in ("freq_hz", "mag", "phase_rad", "coherence", "trials")}
                       for r in results],
        }
        print("\nSYSID_JSON " + json.dumps(trimmed, default=str))
        log.event("sysid_json " + json.dumps(trimmed, default=str))

        if args.mode == "all":
            _write_plant_model(log.dir, joint_idx, results, args)

    except DriveDisarmed as exc:
        aborted = True
        disarm_summary = str(exc)
        print(f"\n! {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
    except (Aborted, JointLimitViolation) as exc:
        aborted = True
        print(f"\n! aborted: {exc}")
        log.event(f"aborted: {exc}", level="ERROR")
    except KeyboardInterrupt:
        aborted = True
        print("\n! interrupted")
        log.event("interrupted", level="WARN")
    except Exception as exc:  # noqa: BLE001
        aborted = True
        print(f"\n! {type(exc).__name__}: {exc}")
        log.event(f"{type(exc).__name__}: {exc}", level="ERROR")
    finally:
        print("\nzero torque / relax + IDLE")
        if armed:
            try:
                for m in config.motors:
                    link.set_input_torque(m.node_id, 0.0)
                    link.set_pos_gain(m.node_id, 0.0)
            except Exception:  # noqa: BLE001
                pass
            if default_vel_gain is not None:
                for m in config.motors:
                    try:
                        link.set_vel_gains(m.node_id, default_vel_gain, 0.0)
                    except Exception as exc:  # noqa: BLE001
                        log.event(f"restore vel_gain node {m.node_id}: {exc}", level="ERROR")
        idle_all()
        time.sleep(0.2)
        if disarm_summary is not None:
            print(f"\n  *** {disarm_summary} ***")
        link.close()
        log.close()
        print(f"closed. log: {log.dir}")

    if aborted:
        raise SystemExit(1)


def _write_plant_model(log_dir: Path, joint_idx: int, results: list, args) -> None:
    joint_name = JOINT_NAMES[joint_idx]
    by_mode = {r["mode"]: r for r in results}
    model = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "joint": joint_name,
        "inertia_a_s2_per_rad": by_mode.get("chirp", {}).get("inertia_a_s2_per_rad"),
        "delay_s": by_mode.get("chirp", {}).get("delay_s"),
        "resonances_hz": by_mode.get("chirp", {}).get("resonances_hz"),
        "antiresonances_hz": by_mode.get("chirp", {}).get("antiresonances_hz"),
        "viscous_a_per_rad_s": {
            d: by_mode.get("friction", {}).get(d, {}).get("kinetic_slope_a_per_rad_s")
            for d in ("+", "-")
        },
        "friction_static_a": {
            d: by_mode.get("friction", {}).get(d, {}).get("static_a") for d in ("+", "-")
        },
        "friction_kinetic_intercept_a": {
            d: by_mode.get("friction", {}).get(d, {}).get("kinetic_intercept_a") for d in ("+", "-")
        },
        "cogging": {d: by_mode.get("cogging", {}).get(d) for d in ("+", "-")},
        "velocity_noise_std_by_bandwidth": by_mode.get("velnoise", {}).get("std_by_bandwidth"),
        "latency": {
            "iq_latency_s": by_mode.get("latency", {}).get("iq_latency_s"),
            "vel_latency_s": by_mode.get("latency", {}).get("vel_latency_s"),
            "feedback_age_p95_s": by_mode.get("latency", {}).get("feedback_age_p95_s"),
        },
    }
    (log_dir / "plant_model.json").write_text(json.dumps(model, indent=2, default=str))

    lines = [f"# Plant model: {joint_name} ({args.label or 'unlabeled'})",
            f"generated {model['generated_utc']}", ""]
    lines.append(f"- Inertia: {model['inertia_a_s2_per_rad']} A.s^2/rad")
    lines.append(f"- Delay: {model['delay_s']} s")
    lines.append(f"- Resonances (Hz): {model['resonances_hz']}")
    lines.append(f"- Anti-resonances (Hz): {model['antiresonances_hz']}")
    lines.append(f"- Static friction (A): + {model['friction_static_a']['+']}  "
                f"- {model['friction_static_a']['-']}")
    lines.append(f"- Kinetic friction intercept (A): + {model['friction_kinetic_intercept_a']['+']}  "
                f"- {model['friction_kinetic_intercept_a']['-']}")
    lines.append(f"- Viscous term (A/(rad/s)): + {model['viscous_a_per_rad_s']['+']}  "
                f"- {model['viscous_a_per_rad_s']['-']}")
    lines.append(f"- Cogging: {model['cogging']}")
    lines.append(f"- Velocity noise std by encoder_bandwidth: "
                f"{model['velocity_noise_std_by_bandwidth']}")
    (log_dir / "plant_model.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# offline analysis: re-run post-processing from a logged run, no bus/sim
# --------------------------------------------------------------------------

def analyze_logdir(logdir: Path, cli_args: argparse.Namespace) -> None:
    """Re-run fits + plots from an existing run's meta.json/samples.jsonl.
    Deliberately imports nothing that touches a bus -- this must work on a
    laptop with no CAN hardware attached (2026-09-08 ask: the crash on rig-host
    was a missing matplotlib, discovered only after the hardware run had
    already happened; being able to re-plot/re-fit offline means a plotting
    bug doesn't cost another hardware run)."""
    meta_path = logdir / "meta.json"
    samples_path = logdir / "samples.jsonl"
    if not meta_path.exists() or not samples_path.exists():
        raise SystemExit(f"{logdir} doesn't look like a sysid run (missing meta.json/samples.jsonl)")
    meta = json.loads(meta_path.read_text())
    stored_cli = meta.get("cli_args")
    if stored_cli:
        ns = argparse.Namespace(**stored_cli)
    else:
        # pre-cli_args log (2026-09-08 and earlier): fall back to this
        # parser's defaults, overlaid with whatever meta.json's flatter,
        # older schema does have (mode/joint/amp_a/label -- see
        # RunLogger's meta kwargs in the pre-cli_args version of main()).
        # Best-effort, not exact -- f0/f1/duration/etc. for that run are
        # simply unknown and take today's defaults; --fit-fmin and friends
        # can be passed explicitly on this --analyze invocation to correct
        # for it. Still strictly better than refusing to analyze the log.
        print(f"  ! {meta_path} has no cli_args (log predates --analyze support) -- "
              "falling back to parser defaults overlaid with meta.json's top-level fields; "
              "pass --f0/--f1/--duration/--fit-fmin etc. explicitly if these don't match "
              "the original run")
        ns = build_parser().parse_args([])
        for key in ("mode", "joint", "amp_a", "label"):
            if key in meta and meta[key] is not None:
                setattr(ns, key, meta[key])
    ns.mode = ns.mode or meta.get("mode")
    if ns.joint is None:
        ns.joint = meta.get("joint")
    # a handful of post-processing knobs may be usefully overridden for a
    # given analysis pass without re-running hardware (e.g. tightening
    # --coherence-min after seeing the plot) -- anything actually typed on
    # this invocation's command line wins over the stored value.
    override_flags = ("no_plot", "plot_dir", "fit_fmin", "coherence_min",
                      "resonance_prominence_db", "pulse_s", "vel_noise_sigma", "amp_profile")
    typed = {a.lstrip("-").replace("-", "_") for a in sys.argv if a.startswith("--")}
    for flag in override_flags:
        if flag in typed:
            setattr(ns, flag, getattr(cli_args, flag))

    mode = meta.get("mode")
    joint_idx = meta.get("joint")
    print(f"analyzing {logdir}  mode={mode}  joint={joint_idx} ({JOINT_NAMES.get(joint_idx)})")
    if meta.get("sign_conventions"):
        print(f"  sign conventions (from meta.json): {meta['sign_conventions']}")

    rows_by_tag: dict[str, list[dict]] = {}
    with samples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows_by_tag.setdefault(d.get("tag"), []).append(d)

    analyzed: dict[str, dict] = {}

    if mode in ("chirp", "all") and rows_by_tag.get("chirp"):
        rows = rows_by_tag["chirp"]
        t = np.array([r["t"] for r in rows], float)
        i_cmd = np.array([r["i_cmd_a"] for r in rows], float)
        iq_a = np.array([r["iq_a"] for r in rows], float)
        qd = np.array([r["qd"] for r in rows], float)
        q = np.array([r["q"] for r in rows], float)
        analyzed["chirp"] = _finish_chirp(t, i_cmd, iq_a, qd, q, joint_idx, ns)

    if mode in ("latency", "all") and rows_by_tag.get("latency"):
        by_trial: dict[int, list[dict]] = {}
        for r in rows_by_tag["latency"]:
            by_trial.setdefault(r.get("trial", 0), []).append(r)
        trials = []
        for _trial_i, trial_rows in sorted(by_trial.items()):
            trial_rows.sort(key=lambda r: r["t"])
            t_arr = np.array([r["t"] for r in trial_rows], float)
            i_cmd = np.array([r["i_cmd_a"] for r in trial_rows], float)
            on_idx = np.flatnonzero(i_cmd != 0.0)
            # samples.jsonl's "t" is elapsed-since-trial-start (see
            # run_latency's log.sample call), so the first active sample's t
            # IS t_cmd in that same relative frame -- no absolute wall-clock
            # reconstruction needed, and estimate_latency only ever uses
            # differences against t_cmd anyway.
            t_cmd_rel = float(t_arr[on_idx[0]]) if len(on_idx) else 0.0
            trials.append({
                "t_cmd": t_cmd_rel,
                "t_iq": t_arr.tolist(), "iq": [r["iq_a"] for r in trial_rows],
                "i_cmd": i_cmd.tolist(),
                "t_vel": t_arr.tolist(), "vel": [r["qd"] for r in trial_rows],
                "feedback_age_s": [r.get("age_s", 0.0) for r in trial_rows],
            })
        analyzed["latency"] = _finish_latency(trials, joint_idx, ns, ns.pulse_s)

    if not analyzed:
        raise SystemExit(f"nothing to analyze in {logdir} for mode={mode} "
                         "(no matching tagged samples found)")

    out_path = logdir / "summary_analyzed.json"
    out_path.write_text(json.dumps(analyzed, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
