"""Structured logging for bring-up scripts (and, later, the runtime).

First slice of the "architect a telemetry/logging stack" TODO (see the Notion
progress page) — scoped to what `scripts/*.py` need today: **never lose an
axis-error transition again**. 2026-09-03's bring-up session had to reconstruct
a node's fault after the fact from a raw `candump`, because nothing during the
run logged `CanLink.node_status()` anywhere but a terminal that had already
scrolled past it.

Each run gets its own directory, `logs/<name>-<UTC timestamp>/`:

    meta.json       run metadata: name, start time, freeform kwargs (CLI args,
                     firmware/gain config — the spec wants this for milestone-3
                     experiment logs)
    samples.jsonl   one JSON object per `.sample()` call — everything you know
                     about the system that tick, including per-node status
    events.log      human-readable: every state / error transition, plus any
                     explicit `.event()` call. Also echoed to stdout so it is
                     visible *live*, not just in the file after the fact.

Schema is intentionally flat JSON so it's greppable and loads straight into
pandas/duckdb for the milestone-3 sweep scripts. `Runtime.telemetry()` already
returns a dict close to this shape; when the runtime grows a file sink, it
should reuse `RunLogger` rather than inventing a second schema.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

LOG_ROOT = Path(__file__).resolve().parent.parent / "logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):  # numpy arrays/scalars
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class RunLogger:
    """One instance per script invocation. Cheap; flushes every write."""

    def __init__(self, name: str, **meta: Any) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.dir = LOG_ROOT / f"{name}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)

        (self.dir / "meta.json").write_text(
            json.dumps({"name": name, "started_utc": _now_iso(), **_jsonable(meta)}, indent=2)
        )
        self._samples = (self.dir / "samples.jsonl").open("a", buffering=1)
        self._events = (self.dir / "events.log").open("a", buffering=1)
        self._t0 = time.monotonic()
        #: last seen NodeStatus per node_id, for transition detection
        self._last_status: dict[int, dict[str, Any]] = {}

        self.event(f"logging to {self.dir}")

    # ------------------------------------------------------------------ api

    def sample(self, node_status: Sequence[Any] = (), **fields: Any) -> None:
        """Write one structured sample. Pass `CanLink.node_status()` as
        `node_status=` to also get automatic transition detection -> events."""
        row = {"t": round(time.monotonic() - self._t0, 4), "ts": _now_iso()}
        row.update({k: _jsonable(v) for k, v in fields.items()})
        if node_status:
            row["nodes"] = [_jsonable(s) for s in node_status]
            for s in node_status:
                self._check_transition(s)
        self._samples.write(json.dumps(row) + "\n")

    def event(self, message: str, *, level: str = "INFO") -> None:
        line = f"{_now_iso()} [{level:5}] {message}"
        self._events.write(line + "\n")
        print(line, file=sys.stderr if level in ("WARN", "ERROR") else sys.stdout)

    def close(self) -> None:
        self._samples.close()
        self._events.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _check_transition(self, status: Any) -> None:
        nid = getattr(status, "node_id", None)
        if nid is None:
            return
        prev = self._last_status.get(nid)
        cur = {
            "axis_state": status.axis_state,
            "active_errors": status.active_errors,
            "disarm_reason": status.disarm_reason,
        }
        self._last_status[nid] = cur
        if prev is None:
            self.event(f"node {nid}: initial status {cur}")
            return
        if cur["axis_state"] != prev["axis_state"]:
            self.event(f"node {nid}: axis_state {prev['axis_state']} -> {cur['axis_state']}")
        if cur["active_errors"] != prev["active_errors"]:
            level = "WARN" if cur["active_errors"] else "INFO"
            self.event(
                f"node {nid}: active_errors 0x{prev['active_errors']:x} -> "
                f"0x{cur['active_errors']:x}",
                level=level,
            )
        if cur["disarm_reason"] != prev["disarm_reason"]:
            self.event(
                f"node {nid}: disarm_reason 0x{prev['disarm_reason']:x} -> "
                f"0x{cur['disarm_reason']:x}",
                level="WARN",
            )
