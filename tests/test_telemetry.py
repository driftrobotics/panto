"""RunLogger's transition detection over CanLink.node_status().

Focus: active_errors/disarm_reason are None (unknown) until the first
Get_Error frame is decoded (see panto/can_link.py). RunLogger must not treat
that unknown -> known step as a real transition -- only WARN on a change
between two *known* values.
"""

from __future__ import annotations

from panto.can_link import NodeStatus
from panto.telemetry import RunLogger


def _status(node_id=0, axis_state=8, axis_error=0, active_errors=None,
            disarm_reason=None, age_s=0.01):
    return NodeStatus(
        node_id=node_id, axis_state=axis_state, axis_error=axis_error,
        active_errors=active_errors, disarm_reason=disarm_reason, age_s=age_s,
    )


def _events(log: RunLogger) -> list[str]:
    return (log.dir / "events.log").read_text().splitlines()


def test_unknown_to_known_no_error_is_not_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr("panto.telemetry.LOG_ROOT", tmp_path)
    log = RunLogger("t")
    # First sample: errors not yet known (mirrors the real gap between
    # Heartbeat and the first Get_Error frame).
    log.sample(node_status=[_status(active_errors=None, disarm_reason=None)])
    # Second sample: the first Get_Error frame arrived, reporting no fault.
    log.sample(node_status=[_status(active_errors=0, disarm_reason=0)])
    lines = _events(log)
    log.close()

    assert not any("[WARN " in l for l in lines)
    assert any("initial errors" in l and "active=0x0" in l for l in lines)


def test_unknown_to_known_latched_disarm_is_not_a_warning(tmp_path, monkeypatch):
    # This is the bug: a drive that already has a latched disarm_reason before
    # we ever connect must not produce a fake WARN transition on first sight.
    monkeypatch.setattr("panto.telemetry.LOG_ROOT", tmp_path)
    log = RunLogger("t")
    log.sample(node_status=[_status(active_errors=None, disarm_reason=None)])
    log.sample(node_status=[_status(active_errors=0, disarm_reason=0x2)])
    lines = _events(log)
    log.close()

    assert not any("[WARN " in l for l in lines)
    initial = [l for l in lines if "initial errors" in l]
    assert len(initial) == 1
    assert "disarm=0x2" in initial[0]
    assert "latched" in initial[0]


def test_real_transition_between_known_values_warns(tmp_path, monkeypatch):
    monkeypatch.setattr("panto.telemetry.LOG_ROOT", tmp_path)
    log = RunLogger("t")
    log.sample(node_status=[_status(active_errors=0, disarm_reason=0)])
    log.sample(node_status=[_status(active_errors=0, disarm_reason=0x2)])
    lines = _events(log)
    log.close()

    warns = [l for l in lines if "[WARN " in l]
    assert len(warns) == 1
    assert "disarm_reason 0x0 -> 0x2" in warns[0]


def test_no_change_between_known_values_is_silent(tmp_path, monkeypatch):
    monkeypatch.setattr("panto.telemetry.LOG_ROOT", tmp_path)
    log = RunLogger("t")
    log.sample(node_status=[_status(active_errors=0, disarm_reason=0)])
    n_events_after_first = len(_events(log))
    log.sample(node_status=[_status(active_errors=0, disarm_reason=0)])
    n_events_after_second = len(_events(log))
    log.close()

    assert n_events_after_second == n_events_after_first
