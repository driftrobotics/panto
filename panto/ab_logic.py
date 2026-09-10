"""Pure post-processing for scripts/ab_bench.py -- the position-vs-torque
backend A/B comparison on a diagonal line-drag task and a one-sided wall task.

No CAN, no sim, no I/O: everything here takes sample arrays the bench script
already collected and returns numbers/strings, same split as
panto/step_logic.py / panto/trace_logic.py. The bench reuses
``panto.step_logic.analyze_step`` directly for the "along the commanded
axis" overshoot/settling-time/verdict on both tasks (the line's tangential
channel and the wall's penetration channel are each just a 1-D step
response); what's here is what analyze_step doesn't already cover: the
Jacobian-projected drag current, the line's lateral deviation, the wall's
compliance and pull-away time, the multi-constraint combine used to drive
the wall task, and the side-by-side report table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def line_direction(angle_deg: float = 45.0) -> np.ndarray:
    """Unit vector at ``angle_deg`` from +x -- the line task's slide axis."""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def project_current(i_joint, J, direction) -> float:
    """Per-motor Iq (A) -> a signed, Cartesian-direction-projected current (A).

    Both backends turn a Cartesian wrench into joint torque via
    ``tau = J.T @ F`` (see backends/base.py's module docstring and
    backends/torque.py). Iq is proportional to torque by each motor's
    ``torque_constant``, which is a fixed positive scalar per motor and so
    does not change *direction* -- treating measured ``i_joint`` as a stand-in
    for joint torque and inverting that map, ``F_hat = inv(J).T @ i_joint``,
    gives a Cartesian force-like vector; its component along the unit
    ``direction`` is what's returned, in amps. This is exactly what the
    contract calls "Iq projected along the line/normal direction via the
    Jacobian."
    """
    F_hat = np.linalg.inv(np.asarray(J, float)).T @ np.asarray(i_joint, float)
    return float(np.dot(F_hat, np.asarray(direction, float)))


def i2t_a2s(currents, period_s: float) -> list[float]:
    """Per-motor I^2t (A^2.s) for a run: ``sum(i^2) * dt`` per motor.
    ``currents`` is (N, 2) amps, one row per tick at a fixed ``period_s``."""
    currents = np.asarray(currents, float)
    if len(currents) == 0:
        return [0.0, 0.0]
    return ((currents ** 2).sum(axis=0) * period_s).tolist()


def _tail_mean(x: np.ndarray, tail_frac: float) -> float:
    n = max(1, int(len(x) * tail_frac))
    return float(np.mean(x[-n:]))


def normal_compliance_mm_per_a(penetration_mm, restoring_current_a, *,
                                tail_frac: float = 0.3) -> float:
    """Wall task's "normal stiffness" metric, reported as its inverse
    (compliance, easier to read on a wall that barely moves): mean
    steady-state penetration (mm) / mean steady-state restoring current (A),
    averaged over the last ``tail_frac`` of the given samples (the caller
    windows these to the push step-hold phase first). NaN if the steady-state
    current is too small to define a ratio.
    """
    pen = np.asarray(penetration_mm, float)
    cur = np.asarray(restoring_current_a, float)
    if len(pen) == 0 or len(cur) == 0:
        return float("nan")
    pen_tail = _tail_mean(pen, tail_frac)
    cur_tail = _tail_mean(np.abs(cur), tail_frac)
    if cur_tail < 1e-4:
        return float("nan")
    return pen_tail / cur_tail


def pull_away_time_s(t, penetration_mm, release_t: float,
                      threshold_mm: float = 0.0):
    """Seconds from ``release_t`` (when the push target starts retreating off
    the wall) until ``penetration_mm`` first drops to <= ``threshold_mm``
    (the EE actually leaves the wall / the unilateral constraint goes
    inactive again). ``None`` if it never does within the given samples."""
    t = np.asarray(t, float)
    pen = np.asarray(penetration_mm, float)
    mask = t >= release_t
    if not mask.any():
        return None
    idx = np.flatnonzero(mask & (pen <= threshold_mm))
    if idx.size == 0:
        return None
    return float(t[idx[0]] - release_t)


def combine_constraints(terms) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sum active constraint terms into one ``(K, pull, active)``, mirroring
    ``Runtime._combine``'s "uniform form" policy (sum bilateral pulls, gate
    unilateral ones on ``is_active``) without needing a live ``Runtime``.

    ``terms`` is a sequence of ``(Projection, k_i)`` pairs. Returns the
    summed isotropic stiffness ``K = sum(k_i * I)`` (over active terms only),
    ``pull = sum(k_i * anchor_i)``, and whether any term was active. The
    caller solves ``anchor = inv(K) @ pull`` when active.
    """
    from .constraints import is_active  # local import: keep this module numpy-only otherwise

    K = np.zeros((2, 2))
    pull = np.zeros(2)
    active = False
    for proj, k_i in terms:
        if not is_active(proj):
            continue
        K_i = np.eye(2) * float(k_i)
        K = K + K_i
        pull = pull + K_i @ np.asarray(proj.anchor, float)
        active = True
    return K, pull, active


@dataclass
class LineMetrics:
    """Everything analyze_step doesn't already give us for the line task."""

    tangential_current_rms_a: float
    lateral_rms_mm: float
    lateral_max_mm: float
    peak_current_a: list
    rms_current_a: list
    i2t_a2s: list


def analyze_line(tangential_current_a, lateral_mm, currents, period_s) -> LineMetrics:
    currents = np.asarray(currents, float)
    lateral_mm = np.asarray(lateral_mm, float)
    if len(currents):
        peak = np.abs(currents).max(axis=0).tolist()
        rms_cur = [rms(currents[:, 0]), rms(currents[:, 1])]
    else:
        peak = [float("nan"), float("nan")]
        rms_cur = [float("nan"), float("nan")]
    return LineMetrics(
        tangential_current_rms_a=rms(tangential_current_a),
        lateral_rms_mm=rms(lateral_mm),
        lateral_max_mm=float(np.max(np.abs(lateral_mm))) if len(lateral_mm) else float("nan"),
        peak_current_a=peak,
        rms_current_a=rms_cur,
        i2t_a2s=i2t_a2s(currents, period_s),
    )


@dataclass
class WallMetrics:
    """Everything analyze_step doesn't already give us for the wall task."""

    normal_compliance_mm_per_a: float
    pull_away_time_s: float | None
    peak_current_a: list
    rms_current_a: list
    i2t_a2s: list


def analyze_wall(penetration_mm_hold, restoring_current_a_hold, t_full,
                  penetration_mm_full, release_t: float, currents, period_s) -> WallMetrics:
    currents = np.asarray(currents, float)
    if len(currents):
        peak = np.abs(currents).max(axis=0).tolist()
        rms_cur = [rms(currents[:, 0]), rms(currents[:, 1])]
    else:
        peak = [float("nan"), float("nan")]
        rms_cur = [float("nan"), float("nan")]
    return WallMetrics(
        normal_compliance_mm_per_a=normal_compliance_mm_per_a(
            penetration_mm_hold, restoring_current_a_hold),
        pull_away_time_s=pull_away_time_s(t_full, penetration_mm_full, release_t),
        peak_current_a=peak,
        rms_current_a=rms_cur,
        i2t_a2s=i2t_a2s(currents, period_s),
    )


# --------------------------------------------------------------------- report

LINE_FIELDS = (
    ("tangential_current_rms_a", "tangential drag I RMS (A)"),
    ("lateral_rms_mm", "lateral dev RMS (mm)"),
    ("lateral_max_mm", "lateral dev max (mm)"),
    ("overshoot_mm", "overshoot (mm)"),
    ("settling_time_s", "settle (s)"),
    ("steady_state_error_mm", "ss error (mm)"),
    ("peak_current_a", "peak I (A)"),
    ("rms_current_a", "RMS I (A)"),
    ("i2t_a2s", "I2t (A2.s)"),
    ("verdict", "verdict"),
)

WALL_FIELDS = (
    ("normal_compliance_mm_per_a", "compliance (mm/A)"),
    ("overshoot_mm", "overshoot (mm)"),
    ("settling_time_s", "settle (s)"),
    ("steady_state_error_mm", "ss error (mm)"),
    ("pull_away_time_s", "pull-away (s)"),
    ("peak_current_a", "peak I (A)"),
    ("rms_current_a", "RMS I (A)"),
    ("i2t_a2s", "I2t (A2.s)"),
    ("verdict", "verdict"),
)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        return "/".join(_fmt(x) for x in v)
    return str(v)


def _task_table(task: str, fields, backends: list[str], results: dict) -> list[str]:
    lines = [f"## {task}", "",
             "| metric | " + " | ".join(backends) + " |",
             "|---" * (len(backends) + 1) + "|"]
    for key, label in fields:
        row = [label]
        for b in backends:
            row.append(_fmt(results.get(b, {}).get(task, {}).get(key)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def build_report_md(meta: dict, results: dict) -> str:
    """``results`` is ``{backend_name: {"line": {...}, "wall": {...}}}``,
    each inner dict a flat metric-name -> value mapping (as produced by
    scripts/ab_bench.py merging ``analyze_step`` with ``analyze_line``/
    ``analyze_wall``). Renders one side-by-side table per task."""
    backends = list(results.keys())
    lines = [f"# A/B torque-vs-position bench {meta.get('utc', '')}", ""]
    preset_line = f"preset `{meta.get('preset')}`"
    if meta.get("preset_fallback_reason"):
        preset_line += f" ({meta['preset_fallback_reason']})"
    lines.append(preset_line)
    lines.append(
        f"pose {meta.get('pose_mm')} mm, speed {meta.get('speed_mm_s')} mm/s, "
        f"line {meta.get('line_distance_mm')} mm, wall depth {meta.get('wall_depth_mm')} mm, "
        f"wall_stiffness {meta.get('wall_stiffness_n_per_m')} N/m, sim={meta.get('sim')}"
    )
    for b in backends:
        r = results.get(b, {})
        if r.get("aborted"):
            lines.append(f"**{b} aborted:** {r.get('abort_reason')}")
    lines.append("")
    lines += _task_table("line", LINE_FIELDS, backends, results)
    lines += _task_table("wall", WALL_FIELDS, backends, results)
    return "\n".join(lines)
