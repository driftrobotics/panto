# panto — build handoff

You are picking up a greenfield build. This repo is a **scaffold**: interfaces and
stubs that encode the architecture decisions, no working control loop yet.

## Context

- **Spec (source of truth):**
  <REDACTED-PRIVATE-NOTION-LINK> —
  read the "Resolved decisions", "Architecture implications", "Revised
  milestones", and "Deferred / TODO" sections. The material above those is the
  original brainstorm; the decision log supersedes it.
- **FK / IK / Jacobian:** the Notion page **"q(DD) monster math"** in the same
  workspace has the worked math for this specific linkage. `panto/kinematics.py`
  has a standard textbook serial-2R implementation — **reconcile its sign
  conventions and zero references against that doc before trusting it.**
- **Predecessor:** `software/odrive_knob/` on an unmerged branch of the
  `smartknob` repo. Same host↔CAN, same ODrive Micro, same motor. Read its README
  end to end — the loop-health metrics, latency analysis, encoder-resolution
  ceiling, and "damping must stay local" all carry over unchanged.

## Hardware facts

| | |
|---|---|
| linkage | serial 2R, shoulder + elbow, 125 mm axis-to-axis links |
| motors | 2× EM3215 gimbal, ODrive Micro each |
| encoder | onboard MA702, 12-bit, single-turn **absolute**; motor's own diametric magnet |
| transport | host ↔ CAN direct, 1 Mbit/s, no intermediate MCU |
| bus | 15 V now (≈2× nominal), headroom to 20 V |
| firmware | ODrive 0.6.x (Micro) — DBC in odrive_knob is 0.5.6, see its compat note |
| wiring | no slip ring → each joint stays < 1 turn, never 360° |

## The one architecture decision that matters

Render every haptic effect as an **impedance target** `F = K·(x_anchor − x)` plus
local velocity damping, computed by the constraint solver in the runtime. A
**backend** turns that into ODrive commands:

- **position backend (v0):** IK the anchor to joint targets, push `Set_Input_Pos`
  + `Set_Pos_Gain` at ~200 Hz. The ODrive position loop renders the stiffness at
  8 kHz; the host only walks the anchor.
- **torque backend (later):** `τ = Jᵀ·F`, push `Set_Input_Torque`.

Keep them swappable. The constraint solver, IR, UI, recording, and calibration
must not know which backend is active.

**Walls = per-tick anchor reprojection.** Every tick, set the anchor to the
nearest point on the constraint to the current EE pose, clamped so it only pushes
in the penetration direction. A fixed anchor at the contact point is a bilateral
spring to a point — it drags you along the wall and sucks you in from the free
side. Reprojection is what makes it a wall.

## Carry-over gotchas from odrive_knob (do not relearn these)

- **Host-side damping at 200 Hz destabilises** — a damper through delay `T` adds
  negative damping near `1/(2T)`. Velocity damping (`vel_gain`) stays on the
  ODrive. Host only moves anchors and adjusts stiffness.
- **Encoder resolution, not loop rate, caps stiffness** (Colgate–Brown: max
  stiffness ∝ available damping; 12-bit @ 8 kHz is ~12 rad/s noise per LSB). If
  detents/walls top out, an external encoder buys more than a faster loop.
- **Latency compensation:** contact/click position error ≈ `ω × T_roundtrip`.
  Measure the actual round trip (loop-health "feedback age"), set comp to it,
  don't guess high.
- **Thermal:** 0.8 A is the datasheet's *transient* start rating. Wall-holding is
  sustained. Build an I²t accumulator + soft torque cutback into the runtime now
  (stub in `panto/runtime.py`); user-facing trip policy can come later.

## Build order (from the spec's Revised milestones)

1. **Kinematics + calibration you trust.** Wire up FK/IK/J, reconcile with q(DD)
   monster math. Verify joint↔cartesian on hardware. Confirm the MA702 is usable
   in absolute mode (power-cycle-without-moving test — see below). Pick the
   workspace boundary + `σ_min(J)` threshold (sane default, iterate).
2. **Bring-up + UI + telemetry.** Constrain to a *point* (both joints
   position-hold). Loop-health panel (rate, jitter p95, feedback age, overruns).
   Heartbeat → idle after 5 s. UI idle button. Single active controller; 2nd
   connection prompts to kick or attaches read-only.
3. **Control-mode experiments.** Bootstrapping scripts in `scripts/`, log results
   back into the Notion spec: latency histogram; `pos_gain` sweep to
   hiss/instability; `vel_gain` sweep; encoder velocity-noise floor at the chosen
   rate; snap-to-line stiffness sweep for feel.
4. **Snap-to line** via per-tick anchor reprojection (position backend).
5. **Wall** (unilateral line): reprojection + penetration-direction clamp +
   per-joint gain scaling via J at the contact config.
6. **Torque backend** behind the same `ImpedanceBackend` interface; A/B against 4
   and 5 for tangential drag on diagonal walls.
7. **Shapes** (circle/bezier/poly), snap-to-grid, trajectory record/playback.
   Record at 100 Hz (configurable): cartesian points + joint angles + timestamp +
   velocity; user picks the export subset. Playback uses trajectory-control mode
   to reach the start pose.

## Open questions to close early

- **MA702 absolute?** Confirm ODrive 0.6.x can run the onboard encoder in
  absolute mode so `pos_estimate` survives a power cycle. If yes and joint travel
  stays < 1 turn (it does), calibration is one-time: store the zero offset. If
  no, per-boot homing move. This also decides whether "no 360°" needs separate
  turn-counting.
- **`σ_min(J)` threshold.** Consistent EE force = `motor_torque_limit ×
  σ_min_threshold`. Larger workspace → weaker consistent stiffness. Nail this
  number experimentally.
- **Kinematics conventions.** Elbow-up vs elbow-down sign, angle zero references,
  base frame — must match q(DD) monster math.
- **Snap-to-line feel target** (N/mm, breakaway N) — spec says this comes down to
  user feel; capture whatever you land on.

## Repo layout

```
panto/
  runtime.py        control loop, mode SM, I²t budget, constraint solve → backend
  kinematics.py     serial-2R FK / IK / Jacobian  (verify vs q(DD) monster math)
  constraints.py    Constraint protocol + Point / Line / Wall / WorkspaceBoundary
  backends/
    base.py         ImpedanceBackend interface
    position.py     ODrive position-mode backend (v0)
    torque.py       ODrive torque-mode backend (later)
  can_link.py       CANSimple codec + cyclic encoder estimates
  config.py         link lengths, gains, limits, snap params
  web.py            websocket server for the UI
ui/index.html       single-page UI
scripts/            bootstrapping / calibration experiments
tests/              start with kinematics (pure, no I/O)
```

## Conventions

- Match the style of `software/odrive_knob/` (stdlib venv + `requirements.txt`,
  `python -m panto`, terse module docstrings that explain *why*).
- Kinematics and constraint math get unit tests with no I/O, like
  `odrive_knob/tests/test_haptics.py`.
- Log experiment results into the Notion spec with firmware version + gain config
  so runs stay comparable.
