# panto — build contracts

Frozen interfaces every parallel work-stream codes against. If you need to change
something here, stop and flag it — a contract change ripples to other agents.

Progress source of truth: the Notion page **"panto — implementation progress"**
(child of the panto spec). Do not track status in this file.

## Ground rules

- **SI units everywhere.** Angles radians, positions metres, forces newtons,
  torques N·m, time seconds. Joint vector order is always `[q1, q2]` (shoulder,
  elbow), world vector order `[x, y]`.
- **Kinematics convention is fixed** by `panto/kinematics.py` and matches the
  Notion "(q)DD Monster Math" doc: `q1` absolute from world +x, `q2` relative to
  link 1, elbow-up default (`q2 = +arccos(...)`).
- **Calibration lives in exactly one place: `CanLink`.** It applies per-motor
  sign flip + per-joint zero offset. Everything above `CanLink` (runtime,
  backends, constraints, UI) works in *calibrated joint radians* and *world
  metres* only. No other module touches raw encoder turns.
- **Velocity damping is never host-side.** `vel_gain` stays on the ODrive. No
  module computes a damping force from host-rate velocity.
- **File ownership is exclusive** (see matrix below). Do not edit a file another
  stream owns. If you need a change in a shared/foreign file, put the exact diff
  in your final report and the integrator applies it. Shared files nobody edits
  directly: `requirements.txt`, `pytest.ini`, `panto/__init__.py`,
  `panto/__main__.py`, `CONTRACTS.md`.
- Match the house style: `from __future__ import annotations`, terse module
  docstrings that explain *why*, pure logic gets no-I/O unit tests in the style
  of `tests/test_kinematics.py`. Keep new deps minimal and add them to your
  report, not to `requirements.txt`.
- `reference/odrive_knob/` is the predecessor (SmartKnob on one ODrive). Read the
  relevant module before porting; it is reference only, never imported.

## Ownership matrix

| stream | owns (create/edit) | may import (interface only) |
|---|---|---|
| A — CAN + sim | `panto/can_link.py`, `panto/sim.py`, `dbc/`, `tests/test_can_link.py`, `tests/test_sim.py` | `config` |
| B — constraints | `panto/constraints.py`, `tests/test_constraints.py` | `kinematics` |
| C — backends | `panto/backends/base.py`, `panto/backends/position.py`, `panto/backends/torque.py`, `tests/test_backends.py` | `kinematics`, `can_link` (as `CanLink` type), `config` |
| D — runtime + config | `panto/runtime.py`, `panto/config.py`, `panto/config.template.json`, `tests/test_runtime.py` | everything (integration spine) |

Web/UI (`panto/web.py`, `ui/index.html`) is integrated by the coordinator after
A–D land, against the telemetry schema below.

## CanLink (stream A owns; C and D depend on this shape)

Two ODrive Micro nodes (ids 0, 1) on one SocketCAN bus at 1 Mbit/s. Port the rx
thread / DBC-validation / change-detected-setter design from
`reference/odrive_knob/skodrive/odrive_can.py`. The vendored DBC has explicit
`Axis0_*` and `Axis1_*` messages — address them directly by node id. Currents
come from `Get_Iq` (`Iq_Measured`).

```python
class CanLink:
    def __init__(self, config: Config, *, sim: bool = False) -> None: ...

    # lifecycle
    def start(self) -> None: ...
    def wait_for_feedback(self, timeout: float = 5.0) -> None: ...   # raises on silence
    def enter_closed_loop(self, timeout: float = 5.0) -> None: ...
    def stop(self) -> None: ...                                      # idles both axes

    # reads — CALIBRATED joint space (flip + zero offset already applied)
    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:          # ([q1,q2] rad, [q1d,q2d] rad/s)
        ...
    def feedback_age_s(self) -> float: ...                           # newest of the two frames
    def motor_currents(self) -> np.ndarray: ...                      # [i0, i1] A, Iq_Measured
    def axis_errors(self) -> tuple[int, int]: ...
    def counters(self) -> tuple[int, int]: ...                       # tx, rx

    # writes — node_id in {0,1}; joint-space inputs, CanLink does rad<->turn + flip
    def set_controller_mode(self, node_id: int, mode: str) -> None:  # "position" | "torque"
        ...
    def set_input_pos(self, node_id: int, q_rad: float) -> None: ...
    def set_pos_gain(self, node_id: int, gain: float) -> None:       # ODrive native (turn/s)/turn
        ...
    def set_input_torque(self, node_id: int, tau_nm: float) -> None: ...
    def set_limits(self, node_id: int, vel_limit: float, current_limit: float) -> None: ...
    def set_idle(self, node_id: int) -> None: ...
    def clear_errors(self, node_id: int) -> None: ...
```

Calibration transform (nail exact form in implementation, document it):
`q_joint = s * (2*pi * pos_turns) + zero_offset_rad`, with `s = -1 if flip else 1`;
command inverse `pos_turns = s * (q_rad - zero_offset_rad) / (2*pi)`.

`sim=True` constructs the in-process 2-axis simulator on a `virtual` bus and wires
`CanLink` to it, so the same code path runs sim and hardware (as
`reference/odrive_knob` does). Sim models per-joint rigid rotor + ODrive cascade
at 8 kHz + 12-bit encoder quantisation; coupled 2R arm dynamics are a stretch
goal, independent rotors are fine for bring-up. Expose a way for the UI to inject
an external joint torque ("the user's hand") in sim.

## Constraint / Projection (stream B owns)

Already sketched in `panto/constraints.py`. Fill in every `project`.

```python
@dataclass(frozen=True)
class Projection:
    anchor: np.ndarray       # [x,y] the impedance target pulls toward
    normal: np.ndarray       # unit, constraint push direction
    penetration: float       # >0 = EE on the constrained side
    unilateral: bool = False

class Constraint(Protocol):
    def project(self, pose: np.ndarray) -> Projection: ...
```

- `Point.project`: anchor = `at`, penetration = distance, normal toward pose.
- `Line.project`: anchor = `a + ((pose-a)·d) d`, bilateral.
- `Wall.project`: **per-tick reprojection** — anchor = projection of pose onto the
  surface, clamped so it only pushes along `-normal` (into the surface);
  `unilateral=True`; penetration = signed depth past the surface.
- `WorkspaceBoundary.project`: unilateral; keeps EE where reach is valid and
  `min_singular_value(inverse(pose)) >= config.sigma_min_threshold`. Constructor
  takes what it needs from `Config` (geo, threshold, optional polygon).
- `SnapGrid.project`: bilateral, nearest grid intersection of `pitch`/`origin`.

The runtime combines multiple constraints: sum bilateral anchor pulls, gate each
unilateral term on `penetration > 0`. Provide a pure helper if convenient but the
runtime owns the combination policy.

## ImpedanceBackend (stream C owns)

```python
@dataclass(frozen=True)
class ImpedanceCommand:
    pose: np.ndarray        # EE [x,y] m
    q: np.ndarray           # joint [q1,q2] rad
    anchor: np.ndarray      # target [x,y] m
    stiffness: np.ndarray   # 2x2 world-frame EE stiffness, N/m
    force_limit: float      # N, after sigma_min scaling + I2t cutback

class ImpedanceBackend(abc.ABC):
    def __init__(self, link: CanLink, config: Config) -> None: ...
    @abc.abstractmethod
    def apply(self, cmd: ImpedanceCommand) -> None: ...
    @abc.abstractmethod
    def relax(self) -> None: ...          # zero interaction force
    def enter(self) -> None: ...          # set controller mode etc. on mode entry
```

- `PositionBackend`: `q_target = inverse(anchor, geo, elbow=config.elbow)`;
  map `stiffness` (2x2 world) to per-joint `pos_gain` via `K_q ≈ Jᵀ K_x J` then
  take the diagonal, convert N·m/rad → ODrive `pos_gain` using each motor's
  `vel_gain` (`pos_gain = k_turn / vel_gain`, `k_turn = k_rad * 2π`); clamp to a
  configured `max_pos_gain`; push `set_input_pos` + `set_pos_gain`. `relax()`
  parks each anchor on the current angle at ~zero gain. Enforce `force_limit` via
  `set_limits` current cap so pushing past it yields instead of faulting.
- `TorqueBackend`: `F = stiffness @ (anchor - pose)`, clamp `|F|` to
  `force_limit`, `tau = jacobian(q, geo).T @ F`, `set_input_torque`. `relax()`
  commands zero torque.

## Config (stream D owns) — JSON template

`panto/config.template.json`, checked in. Live config is `calibration.json` /
`*.local.json` (already gitignored). `config.py` loads/merges/validates it into
dataclasses; calibration UI writes the live file back.

```json
{
  "geo": { "l1": 0.125, "l2": 0.125 },
  "motors": [
    { "node_id": 0, "flip": false, "zero_offset_rad": 0.0,
      "torque_constant": 0.035, "current_soft_max": 0.8,
      "vel_gain": 0.02, "vel_integrator_gain": 0.0, "max_pos_gain": 500.0 },
    { "node_id": 1, "flip": true, "zero_offset_rad": 0.0,
      "torque_constant": 0.035, "current_soft_max": 0.8,
      "vel_gain": 0.02, "vel_integrator_gain": 0.0, "max_pos_gain": 500.0 }
  ],
  "thermal": { "i_continuous": 0.2, "budget_a2s": 4.0 },
  "can": { "interface": "socketcan", "channel": "can0", "bitrate": 1000000 },
  "control": { "rate_hz": 200.0, "latency_compensation_s": 0.0 },
  "elbow": "up",
  "sigma_min_threshold": 0.03,
  "heartbeat_timeout_s": 5.0,
  "workspace_polygon": null
}
```

Note: one motor is physically flipped — hence per-motor `flip`. Keep the existing
`Config` dataclass importable (other streams already import it); extend it, don't
rename fields others use (`geo`, `elbow`, `sigma_min_threshold`,
`control_rate_hz` → may become `control.rate_hz`, coordinate).

## Runtime (stream D owns)

Port the fixed-rate loop from `reference/odrive_knob/skodrive/knob.py`
(`next_tick += period`, jitter p95, overruns → "resync rather than spiral").

- Modes: `TRANSPARENT` (backend.relax), `PLOTTER` (follow trajectory), `INTERACTIVE`
  (constraint solve → ImpedanceCommand → backend).
- `step()`: read joint_state + feedback_age → FK pose, latency-compensate by
  `feedback_age * pose_dot` (or `config latency_compensation_s`) → per mode.
- I²t: per motor integrate `(i² - i_continuous²)`, accumulate against `budget_a2s`,
  soft-cutback `force_limit` as budget depletes. Build it now (hooks + behaviour);
  user-facing trip policy later.
- Heartbeat watchdog: no UI heartbeat for `heartbeat_timeout_s` → TRANSPARENT/idle.
- Telemetry: expose a `telemetry()` dict matching the schema below.

## Telemetry schema (web integration; D exposes, coordinator serves)

Outbound `state` message, broadcast ~60 Hz (decoupled from control loop):

```json
{
  "type": "state",
  "mode": "transparent|plotter|interactive",
  "closed_loop": true,
  "pose": [0.12, 0.03],
  "q": [0.4, 1.1],
  "q_dot": [0.0, 0.0],
  "anchor": [0.12, 0.03],
  "currents": [0.05, 0.05],
  "i2t_frac": [0.0, 0.0],
  "force_limit": 2.0,
  "sigma_min": 0.08,
  "errors": [],
  "stats": { "rate_hz": 200.0, "jitter_p95_ms": 0.3,
             "feedback_age_ms": 6.0, "overruns": 0, "tx": 0, "rx": 0 }
}
```

Inbound (from the single active controller): `set_mode {mode}`,
`set_constraints {constraints:[...]}`, `record_start`, `record_stop`,
`playback {id}`, `set_idle`, `heartbeat`, and sim-only `perturb {tau:[t0,t1]}`.

## Passive/live arming + trajectory record/playback + shape trace (2026-09-10)

Two parallel streams against this frozen interface:

- **Runtime + web** (owns `panto/runtime.py`, `panto/web.py`, their tests) — arming
  semantics, recording buffer, `trace_shape`.
- **UI** (owns `ui/index.html`) — engage/passive toggle, record/playback panel,
  shape panel, against the JSON below only. Does not touch Python.

### Arming (fixes a latent bug: `closed_loop` telemetry never went false)

`Runtime.start()` no longer calls `enter_closed_loop()` — it only brings up the
link (`start()`, `wait_for_feedback()`) and the control-loop thread. Motors stay
IDLE (unarmed) until the UI engages. `step()` still runs every tick while
unarmed: it computes pose/q/telemetry from passive reads and publishes them
(this **is** the live-tracking feature — backdrive the arm by hand, watch the
canvas), but calls neither `backend.relax()` nor `backend.apply()`, so nothing
is ever transmitted toward an IDLE axis.

```python
def engage(self) -> None:
    """Commutate both axes. Raises CanLinkError (e.g. outside joint limits) —
    caller (web.py) catches and reports to the requesting client only."""

def set_idle(self) -> None:
    """Existing method, now also clears closed_loop — this is 'go passive.'"""
```

`closed_loop` in telemetry is the armed/passive flag the UI toggles on.

### Recording (in-memory, single slot, id always `"last"` — "simple" per spec)

```python
def record_start(self) -> None: ...   # begin appending {t, pose, q} every tick
def record_stop(self) -> dict: ...    # -> {"id": "last", "samples": N, "duration_s": T}
def playback(self, id: str = "last") -> None: ...  # KeyError if none recorded
```
Recording works regardless of armed state (hand-guide the passive arm through a
path, then engage + play it back). `playback` reuses the existing
`set_plotter_trajectory` + `Mode.PLOTTER` path verbatim.

Telemetry gains `"recording": bool`, `"recorded_samples": int`.

### Shape trace (reuses `panto/shapes.py` verbatim — box/circle/line only)

```python
def trace_shape(self, shape: str, size_m: float, centre: tuple[float, float],
                 speed: float, laps: int = 1) -> None: ...
```
Builds a 2-second lead-in ramp from the current pose to `shapes.path_points(...)`'s
first sample (same technique as `scripts/trace_shape.py`), splices it in front
time-shifted, feeds the combined list to `set_plotter_trajectory`, switches mode
to `PLOTTER`. Raises `ValueError` for an unknown shape (`shapes.SHAPES` is the
source of truth) — web.py reports it the same way as an `engage` failure.

### New outbound message: one-shot error, to the requesting client only

```json
{ "type": "error", "message": "refusing to arm -- joint(s) outside limits: ..." }
```
Sent (not broadcast) in response to a failed `engage`, `playback`, or `trace_shape`.

### New inbound messages

`engage` (no payload), `trace_shape {shape, size_m, centre:[x,y], speed, laps}`.
`record_start` / `record_stop` / `playback {id}` / `set_idle` already existed as
contract names above but were never dispatched — this is where they land.
