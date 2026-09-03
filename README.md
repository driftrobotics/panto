# panto

A 2-DOF planar haptic linkage. Takes the SmartKnob-on-ODrive control approach
(push the spring into the ODrive, host only walks the anchor) from one rotational
axis to 2D planar motion.

Three modes:

1. **transparent** — pure backdrive, motors idle (no gravity/friction comp in v0)
2. **plotter** — play back a recorded or drawn trajectory
3. **interactive** — haptics: snap-to, virtual walls, force feedback

Spec (source of truth, with resolved-decision log):
<REDACTED-PRIVATE-NOTION-LINK>

Start here if you're picking up the build: [`HANDOFF.md`](HANDOFF.md).

## Hardware

- Serial 2R linkage (shoulder + elbow), **125 mm** axis-to-axis links
- 2× EM3215 gimbal motors, ODrive Micro each (onboard MA702 encoder, motor's own
  diametric magnet)
- **host ↔ CAN direct**, no intermediate MCU
- 15 V bus (headroom to 20 V)
- No slip ring → joint travel stays < 1 turn, no 360° rotations

## Architecture

```
             UI (websocket client)
                     │  set constraints, mode, record/playback
                     ▼
┌─────────────────────────────────────────────┐
│ runtime  (~200 Hz, owns the CAN link)        │
│   mode state machine                         │
│   constraint solver:  project(pose) →        │
│       (anchor, normal, penetration)          │
│   impedance target:   F = K·(x_anchor − x)   │
│   per-motor I²t budget                       │
└───────────────┬─────────────────────────────┘
                │  ImpedanceBackend
        ┌───────┴────────┐
        ▼                ▼
  position backend   torque backend
  IK anchor→joints   τ = Jᵀ·F
  Set_Input_Pos      Set_Input_Torque
  Set_Pos_Gain
                \        /
                 ▼      ▼
              ODrive (8 kHz) — velocity damping stays here
```

The control law is written **once** as an impedance target. Swapping position ↔
torque backend does not touch the constraint solver, the IR, the UI, recording,
or calibration. v0 runs the position backend; the torque backend exists so we can
A/B it for tangential-drag feel on non-axis-aligned walls.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

Running against hardware / sim is not wired up yet — see `HANDOFF.md`.

## Prior art

The SmartKnob-on-ODrive implementation (`software/odrive_knob/` on an unmerged
branch of the `smartknob` repo) is the direct predecessor. Its README's "Not this
architecture" section is why the impedance target is backend-abstracted here.
