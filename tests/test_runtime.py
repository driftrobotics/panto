"""Runtime: mode SM, constraint solve, sigma / I²t force scaling, watchdog,
telemetry schema. Driven with a fake CanLink + fake backend + fake constraints
(the real constraints module is still a stub in a sibling stream)."""

from __future__ import annotations

import numpy as np
import pytest

from panto.backends import ImpedanceCommand
from panto.config import Config
from panto.constraints import Projection
from panto.kinematics import Unreachable, forward
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
        self.cleared: list[int] = []

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

    def clear_errors(self, node_id):
        self.cleared.append(node_id)

    def stop(self):
        pass


class FakeBackend:
    def __init__(self):
        self.applied: list[ImpedanceCommand] = []
        self.relaxed = 0
        self.entered = 0
        self.raise_unreachable = False

    def apply(self, cmd):
        if self.raise_unreachable:
            raise Unreachable("no IK solution")
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
    # The I²t tests reason about these exact numbers; the template's rig
    # values (0.8 A / 40 A²·s) are deliberately not what's under test.
    cfg.thermal.i_continuous, cfg.thermal.budget_a2s = 0.2, 4.0
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
    assert backend.entered == 2      # once by engage(), once on mode entry
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
        "tripped", "tuning", "workspace", "recording", "recorded_samples", "stats",
    }
    assert tel["type"] == "state"
    assert tel["mode"] == "interactive"
    assert set(tel["stats"]) == {
        "rate_hz", "jitter_p95_ms", "feedback_age_ms", "overruns", "tx", "rx",
    }
    assert tel["stats"]["tx"] == 7 and tel["stats"]["rx"] == 11
    assert len(tel["pose"]) == 2 and len(tel["i2t_frac"]) == 2
    assert tel["errors"] == []
    assert tel["tripped"] is False
    assert set(tel["workspace"]) == {"r_min", "r_max"}
    assert 0.0 <= tel["workspace"]["r_min"] < tel["workspace"]["r_max"]
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


def test_played_back_trajectories_ramp_in_from_the_current_pose():
    rt, cfg, link, _, clock = make()
    rt.record_start()
    rt.step(dt=0.005)
    rt.record_stop()
    link.q = np.array([0.6, 1.3])                       # arm moved since the take
    rt.playback("last")
    traj = rt._trajectory
    assert traj[0][0] == 0.0
    assert np.allclose(traj[0][1], forward(link.q, cfg.geo))
    assert traj[1][0] == pytest.approx(2.0)             # first recorded sample after the ramp
    assert np.allclose(traj[1][1], forward(GOOD_Q, cfg.geo))

    rt.trace_shape("circle", 0.02, forward(link.q, cfg.geo), 0.01)
    traj = rt._trajectory
    assert traj[0][0] == 0.0 and np.allclose(traj[0][1], forward(link.q, cfg.geo))
    assert traj[1][0] == pytest.approx(2.0)


def test_loop_exception_idles_goes_passive_and_keeps_ticking():
    import threading
    import time

    class BoomBackend(FakeBackend):
        def apply(self, cmd):
            raise ZeroDivisionError("boom")

    link = FakeLink()
    rt = Runtime(Config.load(), link, BoomBackend())
    rt.note_heartbeat()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])
    t = threading.Thread(target=rt.run, daemon=True)
    t.start()
    for _ in range(200):
        tel = rt.telemetry()
        if any(e.startswith("loop:ZeroDivisionError") for e in tel["errors"]):
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"loop error never surfaced: {rt.telemetry()['errors']}")
    assert t.is_alive() and tel["closed_loop"] is False and link.idle_calls >= 1
    with pytest.raises(RuntimeError, match="clear errors first"):
        rt.engage()
    rt.clear_errors()
    for _ in range(200):                     # next passive tick republishes
        if "loop:ZeroDivisionError" not in " ".join(rt.telemetry()["errors"]):
            break
        time.sleep(0.01)
    rt._stop.set()
    t.join(timeout=2.0)
    assert "loop:ZeroDivisionError" not in " ".join(rt.telemetry()["errors"])


def test_set_tuning_changes_live_stiffness_and_rejects_bad_values():
    rt, cfg, _, backend, _ = make()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.12, 0.06])])
    rt.set_tuning(stiffness_n_per_m=25.0, wall_stiffness_n_per_m=50.0)
    rt.step(dt=0.005)
    assert backend.applied[-1].stiffness[0, 0] == pytest.approx(25.0)
    assert rt.telemetry()["tuning"]["wall_stiffness_n_per_m"] == 50.0
    with pytest.raises(ValueError):
        rt.set_tuning(stiffness_n_per_m=0.0)
    with pytest.raises(ValueError):
        rt.set_tuning(pos_gain=5.0)


def test_only_the_most_recent_point_acts():
    from panto.constraints import Point

    rt, _, _, backend, _ = make()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([Point(at=np.array([0.10, 0.02])), Point(at=np.array([0.14, 0.08]))])
    rt.step(dt=0.005)
    assert np.allclose(backend.applied[-1].anchor, [0.14, 0.08])


def test_playback_of_a_take_near_a_singularity_is_not_refused():
    # Recorded poses were physically reached; only analytic shapes get validated.
    rt, _, link, _, _ = make(q=SINGULAR_Q)
    rt.record_start()
    rt.step(dt=0.005)
    rt.record_stop()
    rt.playback("last")
    assert rt.mode is Mode.PLOTTER


def test_engage_parks_the_anchor_before_commutating():
    """Entering closed loop against a stale input_pos lunges toward it, so
    engage() must configure + relax the backend *before* enter_closed_loop."""
    order: list[str] = []

    class ArmingLink(FakeLink):
        def set_input_pos(self, node_id, q):
            order.append("park")

        def set_pos_gain(self, node_id, gain):
            order.append("gain")

        def enter_closed_loop(self, timeout=5.0):
            order.append("commutate")

    class OrderedBackend(FakeBackend):
        def enter(self):
            order.append("enter")

        def relax(self):
            order.append("relax")

    rt = Runtime(Config.load(), ArmingLink(), OrderedBackend())
    rt.engage()
    assert order.index("enter") < order.index("relax") < order.index("commutate")


def test_engage_failure_leaves_runtime_unarmed():
    class RefusingLink(FakeLink):
        def enter_closed_loop(self, timeout=5.0):
            raise RuntimeError("refusing to arm")

    rt = Runtime(Config.load(), RefusingLink(), FakeBackend())
    with pytest.raises(RuntimeError):
        rt.engage()
    assert rt.telemetry()["closed_loop"] is False


def test_trajectories_are_validated_before_they_are_followed():
    rt, cfg, _, _, _ = make()
    reach = cfg.geo.l1 + cfg.geo.l2
    with pytest.raises(ValueError, match="leaves reach"):
        rt.trace_shape("line", 0.04, [reach + 0.05, 0.0], 0.01)
    with pytest.raises(ValueError, match="singularity"):
        rt.trace_shape("circle", 0.01, [reach * 0.999, 0.0], 0.01)
    assert rt.mode is Mode.TRANSPARENT
    assert rt._trajectory == []


def test_playback_unknown_id_raises_keyerror():
    rt, _, _, _, _ = make()
    with pytest.raises(KeyError):
        rt.playback("last")


# ------------------------------------------------------------- shape trace

def test_trace_shape_bad_name_raises_valueerror():
    rt, _, _, _, _ = make()
    with pytest.raises(ValueError):
        rt.trace_shape("hexagon", 0.02, [0.1, 0.05], 0.01)


# -------------------------------------------------------- workspace boundary

def test_workspace_boundary_always_active_near_reach_limit_and_not_removable():
    # SINGULAR_Q's sigma (~6e-4) is well under the default sigma_min_threshold
    # (0.03) -- outside the reachable/well-conditioned annulus.
    rt, _, _, backend, _ = make(q=SINGULAR_Q)
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([])  # nothing user-supplied: the boundary alone must engage
    rt.step(dt=0.005)

    assert backend.applied, "workspace boundary alone should produce a command"
    assert "workspace" in rt.telemetry()["errors"]

    # set_constraints can't remove it either.
    rt.set_constraints([FakePoint([0.05, 0.05])])
    rt.step(dt=0.005)
    assert "workspace" in rt.telemetry()["errors"]


def test_workspace_inactive_for_well_conditioned_pose():
    rt, _, _, backend, _ = make(q=GOOD_Q)
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([])
    rt.step(dt=0.005)

    assert not backend.applied
    assert "workspace" not in rt.telemetry()["errors"]


def test_workspace_telemetry_annulus_radii():
    rt, cfg, _, _, _ = make()
    tel = rt.telemetry()
    r_min, r_max = tel["workspace"]["r_min"], tel["workspace"]["r_max"]
    reach = cfg.geo.l1 + cfg.geo.l2
    assert 0.0 < r_min < r_max < reach


# ------------------------------------------------------------------- I2t trip

def test_i2t_trip_latches_blocks_engage_and_clear_errors_releases():
    rt, cfg, link, backend, clock = make(q=GOOD_Q, currents=(2.0, 2.0))
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])
    rt.step(dt=0.005)
    assert backend.applied  # rendering fine before the trip

    # (2^2 - 0.2^2) * 1.1 s = 4.356 A^2*s > the template's 4.0 A^2*s budget.
    rt.step(dt=1.1)
    tel = rt.telemetry()
    assert tel["tripped"] is True
    assert "over_torque" in tel["errors"]
    assert rt.mode is Mode.TRANSPARENT
    assert tel["closed_loop"] is False
    n_applied = len(backend.applied)
    idled = link.idle_calls
    assert idled > 0

    rt.step(dt=0.005)  # stays tripped: no new commands, no idle-call spam
    assert len(backend.applied) == n_applied
    assert link.idle_calls == idled

    with pytest.raises(RuntimeError, match="tripped: over_torque"):
        rt.engage()
    assert rt.telemetry()["closed_loop"] is False

    # Let the accumulator leak back under budget (clear_errors doesn't reset
    # it -- only the latch): 300 * 0.05s at zero current drains 0.6 A^2*s,
    # comfortably under the ~0.36 A^2*s the trip overshot by.
    link.currents = np.zeros(2)
    for _ in range(300):
        rt.step(dt=0.05)

    rt.clear_errors()
    assert link.cleared == [m.node_id for m in cfg.motors]

    rt.set_mode(Mode.INTERACTIVE)  # trip path forced TRANSPARENT while latched
    rt.engage()
    rt.step(dt=0.005)  # telemetry only updates on a step -- check it here
    assert rt.telemetry()["tripped"] is False
    assert len(backend.applied) > n_applied
    assert rt.telemetry()["closed_loop"] is True


# --------------------------------------------------------------- unsolvable

def test_unsolvable_interactive_relaxes_and_keeps_ticking():
    rt, _, _, backend, _ = make(q=GOOD_Q)
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])

    backend.raise_unreachable = True
    rt.step(dt=0.005)  # backend.apply raises Unreachable -- must not crash the loop
    assert not backend.applied
    assert backend.relaxed >= 1
    tel = rt.telemetry()
    assert "unsolvable" in tel["errors"]
    assert rt.mode is Mode.INTERACTIVE  # loop kept ticking, mode unchanged

    backend.raise_unreachable = False
    rt.step(dt=0.005)  # recovers next tick once the backend can solve again
    assert backend.applied
    assert "unsolvable" not in rt.telemetry()["errors"]


def test_unsolvable_plotter_relaxes_and_keeps_ticking():
    rt, _, _, backend, _ = make(q=GOOD_Q)
    rt.engage()
    rt.set_plotter_trajectory([(0.0, [0.1, 0.05]), (1.0, [0.12, 0.05])])
    rt.set_mode(Mode.PLOTTER)

    backend.raise_unreachable = True
    rt.step(dt=0.005)
    assert backend.relaxed >= 1
    assert "unsolvable" in rt.telemetry()["errors"]

    backend.raise_unreachable = False
    rt.step(dt=0.005)
    assert backend.applied


# ------------------------------------------------------------- tick listener

def test_tick_listener_called_with_telemetry_and_exception_swallowed():
    rt, _, _, _, _ = make()
    seen = []

    def bad(_tel):
        raise RuntimeError("boom")

    def good(tel):
        seen.append(tel)

    rt.add_tick_listener(bad)   # registered first -- must not block `good`
    rt.add_tick_listener(good)
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_constraints([FakePoint([0.1, 0.05])])
    rt.step(dt=0.005)  # must not raise despite `bad`

    assert len(seen) == 1
    assert seen[0]["type"] == "state"
    assert seen[0] == rt.telemetry()


def test_engage_pushes_config_vel_gains_clamped():
    pushed: list[tuple] = []

    class GainLink(FakeLink):
        def set_vel_gains(self, node_id, vel_gain, vel_integrator_gain=0.0):
            pushed.append((node_id, vel_gain, vel_integrator_gain))

    cfg = Config.load()
    cfg.motors[0].vel_gain = 0.01
    cfg.motors[1].vel_gain = 0.3                      # above the shoulder-chatter ceiling
    rt = Runtime(cfg, GainLink(), FakeBackend())
    rt.engage()
    assert pushed == [(0, 0.01, 0.0), (1, 0.05, 0.0)]

    pushed.clear()
    rt.set_tuning(vel_gain=[0.02, 0.03])
    assert pushed == [(0, 0.02, 0.0), (1, 0.03, 0.0)]  # live push while armed
    rt.note_heartbeat()
    rt.step(dt=0.005)
    assert rt.telemetry()["tuning"]["vel_gain"] == [0.02, 0.03]
    with pytest.raises(ValueError):
        rt.set_tuning(vel_gain=[0.1, 0.03])
    with pytest.raises(ValueError):
        rt.set_tuning(vel_gain=[0.01])


def test_oscillation_guard_drops_to_k25_and_flags():
    from panto.kinematics import inverse

    rt, cfg, link, backend, clock = make()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_tuning(stiffness_n_per_m=100.0, wall_stiffness_n_per_m=100.0)
    centre = forward(GOOD_Q, cfg.geo)
    rt.set_constraints([FakePoint(centre)])
    q_hi = inverse(centre + np.array([0.0, 0.005]), cfg.geo, elbow="up")
    q_lo = inverse(centre - np.array([0.0, 0.005]), cfg.geo, elbow="up")
    for i in range(160):                                   # 0.8 s at 200 Hz, +-5 mm square wave
        link.q = q_hi if (i // 10) % 2 == 0 else q_lo
        clock.t += 0.005
        rt.step(dt=0.005)
    assert cfg.control.stiffness_n_per_m == 25.0
    assert cfg.control.wall_stiffness_n_per_m == 50.0
    assert "osc_guard" in rt.telemetry()["errors"]
    rt.set_tuning(stiffness_n_per_m=100.0)                 # re-arming clears the flag
    clock.t += 0.005
    rt.step(dt=0.005)
    assert "osc_guard" not in rt.telemetry()["errors"]


def test_oscillation_guard_ignores_steady_tracking_and_low_k():
    rt, cfg, link, _, clock = make()
    rt.engage()
    rt.set_mode(Mode.INTERACTIVE)
    rt.set_tuning(stiffness_n_per_m=100.0)
    rt.set_constraints([FakePoint(forward(GOOD_Q, cfg.geo) + np.array([0.02, 0.0]))])
    for _ in range(160):                                   # 20 mm static error, no swing
        clock.t += 0.005
        rt.step(dt=0.005)
    assert cfg.control.stiffness_n_per_m == 100.0


def test_set_tuning_current_cap_applies_to_all_motors():
    rt, cfg, _, _, _ = make()
    rt.set_tuning(current_cap_a=2.0)
    assert [m.current_soft_max for m in cfg.motors] == [2.0, 2.0]
    rt.note_heartbeat()
    rt.step(dt=0.005)
    assert rt.telemetry()["tuning"]["current_cap_a"] == 2.0
    with pytest.raises(ValueError):
        rt.set_tuning(current_cap_a=2.5)
