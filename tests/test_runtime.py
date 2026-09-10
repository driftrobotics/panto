"""Runtime: mode SM, constraint solve, sigma / I²t force scaling, watchdog,
telemetry schema. Driven with a fake CanLink + fake backend + fake constraints
(the real constraints module is still a stub in a sibling stream)."""

from __future__ import annotations

import numpy as np
import pytest

from panto.backends import ImpedanceCommand
from panto.config import Config
from panto.constraints import Projection
from panto.kinematics import forward
from panto.runtime import Mode, Runtime

GOOD_Q = np.array([0.4, 1.1])          # well-conditioned
SINGULAR_Q = np.array([0.35, 0.01])    # near full extension


class FakeLink:
    def __init__(self, q=GOOD_Q, q_dot=(0.0, 0.0), currents=(0.05, 0.05)):
        self.q = np.asarray(q, dtype=float)
        self.q_dot = np.asarray(q_dot, dtype=float)
        self.currents = np.asarray(currents, dtype=float)
        self.age = 0.005
        self.idle_calls = 0

    def joint_state(self):
        return self.q.copy(), self.q_dot.copy()

    def feedback_age_s(self):
        return self.age

    def motor_currents(self):
        return self.currents.copy()

    def axis_errors(self):
        return (0, 0)

    def counters(self):
        return (7, 11)

    def set_idle(self, node_id):
        self.idle_calls += 1

    def stop(self):
        pass


class FakeBackend:
    def __init__(self):
        self.applied: list[ImpedanceCommand] = []
        self.relaxed = 0
        self.entered = 0

    def apply(self, cmd):
        self.applied.append(cmd)

    def relax(self):
        self.relaxed += 1

    def enter(self):
        self.entered += 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakePoint:
    """Bilateral attractor toward ``at`` (stand-in for constraints.Point)."""

    def __init__(self, at):
        self.at = np.asarray(at, dtype=float)

    def project(self, pose):
        d = self.at - pose
        dist = float(np.linalg.norm(d))
        n = d / dist if dist else np.array([1.0, 0.0])
        return Projection(anchor=self.at, normal=n, penetration=dist)


class FakeWall:
    """Unilateral: pushes toward ``anchor`` only when ``pen`` > 0."""

    def __init__(self, anchor, normal, pen):
        self.anchor = np.asarray(anchor, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.pen = pen

    def project(self, pose):
        return Projection(anchor=self.anchor, normal=self.normal,
                          penetration=self.pen, unilateral=True)


def make(**link_kw):
    cfg = Config.load()
    link = FakeLink(**link_kw)
    backend = FakeBackend()
    clock = Clock()
    rt = Runtime(cfg, link, backend, clock=clock)
    rt.note_heartbeat()
    return rt, cfg, link, backend, clock


# --------------------------------------------------------------------- modes

def test_mode_transitions():
    rt, _, _, backend, _ = make()
    rt.engage()
    assert rt.mode is Mode.TRANSPARENT
    rt.set_mode("interactive")
    assert rt.mode is Mode.INTERACTIVE
    rt.step(dt=0.005)
    assert rt.mode is Mode.INTERACTIVE
    assert backend.entered == 1
    rt.set_mode(Mode.TRANSPARENT)
    rt.step(dt=0.005)
    assert rt.mode is Mode.TRANSPARENT


def test_transparent_relaxes():
    rt, _, _, backend, _ = make()
    rt.engage()
    rt.set_mode(Mode.TRANSPARENT)
    rt.step(dt=0.005)
    rt.step(dt=0.005)
    assert backend.relaxed >= 2
    assert not backend.applied


def test_interactive_point_produces_command():
    rt, cfg, link, backend, _ = make()
    rt.engage()
    target = np.array([0.12, 0.06])
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint(target)])
    rt.step(dt=0.005)

    assert backend.applied, "expected an ImpedanceCommand"
    cmd = backend.applied[-1]
    assert isinstance(cmd, ImpedanceCommand)
    assert np.allclose(cmd.anchor, target, atol=1e-9)
    assert np.allclose(cmd.stiffness, np.eye(2) * cfg.control.stiffness_n_per_m)
    assert cmd.force_limit > 0.0


def test_interactive_no_active_constraint_relaxes():
    rt, _, _, backend, _ = make()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakeWall([0.1, 0.0], [0.0, 1.0], pen=-0.01)])  # free side
    rt.step(dt=0.005)
    assert backend.relaxed >= 1
    assert not backend.applied


def test_wall_and_point_combine_into_one_anchor():
    rt, cfg, _, backend, _ = make()
    rt.engage()
    pose = forward(GOOD_Q, cfg.geo)
    kw = cfg.control.wall_stiffness_n_per_m
    kp = cfg.control.stiffness_n_per_m
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([
        FakePoint(pose + np.array([0.02, 0.0])),
        FakeWall(pose + np.array([0.0, 0.01]), [0.0, 1.0], pen=0.01),
    ])
    rt.step(dt=0.005)
    cmd = backend.applied[-1]
    # combined stiffness is isotropic (sum of k_i·I); anchor is the
    # stiffness-weighted mean of the two targets, so the stiff wall dominates.
    assert cmd.stiffness[0, 0] == pytest.approx(cmd.stiffness[1, 1])
    assert cmd.stiffness[0, 0] == pytest.approx(kw + kp)
    assert cmd.anchor[0] - pose[0] == pytest.approx(0.02 * kp / (kw + kp))
    assert cmd.anchor[1] - pose[1] == pytest.approx(0.01 * kw / (kw + kp))


# ------------------------------------------------------------- force scaling

def test_force_limit_shrinks_near_singularity():
    rt_g, cfg, _, bg, _ = make(q=GOOD_Q)
    rt_s, _, _, bs, _ = make(q=SINGULAR_Q)
    for rt in (rt_g, rt_s):
        rt.engage()
        rt.set_mode(Mode.INTERACTIVE)
        rt.set_constraints([FakePoint([0.1, 0.05])])
        rt.step(dt=0.005)
    fl_good = bg.applied[-1].force_limit
    fl_singular = bs.applied[-1].force_limit
    assert fl_good == pytest.approx(cfg.control.force_limit_n, rel=1e-6)
    assert fl_singular < fl_good


def test_i2t_accumulates_cuts_back_then_recovers():
    # template thermal: i_continuous=0.2 A, budget_a2s=4.0 A²·s
    rt, cfg, link, backend, _ = make(q=GOOD_Q, currents=(0.8, 0.8))
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])

    rt.step(dt=0.005)
    initial = backend.applied[-1].force_limit

    for _ in range(120):
        rt.step(dt=0.05)
    trough = backend.applied[-1].force_limit
    assert trough < 0.5 * initial          # sustained high current cut it back

    link.currents = np.zeros(2)
    for _ in range(400):
        rt.step(dt=0.05)
    partial = backend.applied[-1].force_limit
    assert partial > 1.8 * trough              # leaks back once current drops

    for _ in range(3000):
        rt.step(dt=0.05)                        # accumulator fully drains to 0
    recovered = backend.applied[-1].force_limit
    assert recovered == pytest.approx(initial, rel=1e-9)


# ---------------------------------------------------------------- watchdog

def test_heartbeat_timeout_forces_transparent_and_idles():
    rt, cfg, link, backend, clock = make()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])
    rt.step(dt=0.005)
    assert backend.applied

    clock.t = cfg.heartbeat_timeout_s + 1.0
    rt.step(dt=0.005)
    assert rt.mode is Mode.TRANSPARENT
    assert link.idle_calls > 0
    assert rt.telemetry()["closed_loop"] is False    # watchdog clears the arm
    n_applied = len(backend.applied)

    clock.t += 0.01
    rt.step(dt=0.005)
    assert len(backend.applied) == n_applied      # stays relaxed
    assert link.idle_calls == 2                   # idled once per motor, not spammed

    rt.note_heartbeat()
    assert rt.mode is Mode.TRANSPARENT             # recovers safe, not to INTERACTIVE


# ---------------------------------------------------------------- telemetry

def test_telemetry_matches_schema():
    rt, _, _, _, _ = make()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.12, 0.03])])
    rt.step(dt=0.005)
    tel = rt.telemetry()

    assert set(tel) == {
        "type", "mode", "closed_loop", "pose", "q", "q_dot", "anchor",
        "currents", "i2t_frac", "force_limit", "sigma_min", "errors",
        "recording", "recorded_samples", "stats",
    }
    assert tel["type"] == "state"
    assert tel["mode"] == "interactive"
    assert set(tel["stats"]) == {
        "rate_hz", "jitter_p95_ms", "feedback_age_ms", "overruns", "tx", "rx",
    }
    assert tel["stats"]["tx"] == 7 and tel["stats"]["rx"] == 11
    assert len(tel["pose"]) == 2 and len(tel["i2t_frac"]) == 2
    assert tel["errors"] == []
    assert tel["stats"]["feedback_age_ms"] == pytest.approx(5.0)


def test_start_stop_thread_runs():
    rt, _, _, backend, _ = make()
    rt.start()
    try:
        import time
        time.sleep(0.1)
    finally:
        rt.stop()
    assert backend.relaxed >= 1


# ------------------------------------------------------------------ arming

def test_unarmed_step_publishes_but_calls_no_backend_methods():
    rt, _, _, backend, _ = make()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])
    rt.step(dt=0.005)

    assert backend.relaxed == 0
    assert not backend.applied
    tel = rt.telemetry()
    assert tel["closed_loop"] is False
    assert tel["mode"] == "interactive"        # mode tracked even while unarmed


def test_engage_arms_and_step_then_reaches_backend():
    rt, _, _, backend, _ = make()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])

    rt.engage()
    assert rt.telemetry()["closed_loop"] is False  # not published until a step runs
    rt.step(dt=0.005)

    assert backend.applied
    assert rt.telemetry()["closed_loop"] is True


def test_set_idle_and_watchdog_both_clear_closed_loop():
    rt, cfg, link, backend, clock = make()
    rt.engage()
    rt.step(dt=0.005)
    assert rt.telemetry()["closed_loop"] is True

    rt.set_idle()
    rt.step(dt=0.005)
    assert rt.telemetry()["closed_loop"] is False

    rt.engage()
    rt.step(dt=0.005)
    assert rt.telemetry()["closed_loop"] is True

    clock.t = cfg.heartbeat_timeout_s + 1.0
    rt.step(dt=0.005)
    assert rt.telemetry()["closed_loop"] is False


def test_watchdog_idles_even_when_never_armed():
    """The heartbeat watchdog runs ahead of the arm-gate, so it still forces
    TRANSPARENT + idle even if engage() was never called (the point being:
    losing the UI heartbeat is a safety net independent of arming state)."""
    rt, cfg, link, _, clock = make()
    clock.t = cfg.heartbeat_timeout_s + 1.0
    rt.step(dt=0.005)
    assert rt.mode is Mode.TRANSPARENT
    assert link.idle_calls > 0
    assert rt.telemetry()["closed_loop"] is False


# ---------------------------------------------------------------- recording

def test_record_start_stop_and_playback_round_trip():
    rt, _, link, backend, clock = make()
    rt.record_start()
    rt.step(dt=0.005)
    clock.t += 0.005
    rt.step(dt=0.005)
    result = rt.record_stop()

    assert result["id"] == "last"
    assert result["samples"] == 2
    assert result["duration_s"] == pytest.approx(0.005)

    rt.playback("last")
    assert rt.mode is Mode.PLOTTER


def test_playback_unknown_id_raises_keyerror():
    rt, _, _, _, _ = make()
    with pytest.raises(KeyError):
        rt.playback("last")


# ------------------------------------------------------------- shape trace

def test_trace_shape_bad_name_raises_valueerror():
    rt, _, _, _, _ = make()
    with pytest.raises(ValueError):
        rt.trace_shape("hexagon", 0.02, [0.1, 0.05], 0.01)
