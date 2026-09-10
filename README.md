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

## Control presets

`presets.json` (repo root) is a registry of named, known-good tuning bundles
(`stiffness`, `vel_gain`, `vel_limit`, `current`, `cap_slope`, `cap_min`,
`ff_scale`, `max_pos_gain`), each with `notes` and `verified` provenance.
Every bring-up script (`step_response.py`, `goto_pose.py`, `point_hold.py`,
`offset_sweep.py`, `torque_step.py` for `current` only) takes `--preset NAME`;
explicit flags always override the preset's values field-by-field, and the
resolved parameters + preset name are written into the run's `meta.json`/
`summary.json`.

Current presets:

- `hold-0p6` — gentle hold, 0.6 A, no feedforward. The only config that
  converged cleanly on a 5 mm step; low force.
- `move-1p5-sched` — slow ramped moves (`goto_pose`, ≥8 s per 100 mm). Can
  stall the shoulder CCW near the end of a long recentre.
- `move-2A-sched` — same as `move-1p5-sched` but 2.0 A; use this if
  `move-1p5-sched` stalls. **Default preset for `scripts/reset_pose.py`.**
- `step-2A-kv3` / `step-2A-kv3-ff1` — 2 A step-response tuning, with and
  without full Coulomb feedforward (`ff_scale` 0.7 vs 1.0); `-ff1` is the
  best 2 A config found so far (settles ±1 mm in 2.0 s).

To reset the arm to a known-good pose before/between tuning runs:

```bash
python -m scripts.reset_pose                      # preset move-2A-sched, target = config.test_pose
python -m scripts.reset_pose --target 120,80       # explicit xy, mm
python -m scripts.reset_pose --preset move-1p5-sched --passes 3
```

It refuses to run without `calibration.json`'s `test_pose` and joint limits
configured, reports the start pose/distance/per-pass result, confirms both
drives end IDLE and error-free, and exits non-zero if it's still >3 mm off
target.

To add a preset from a run's log directory (reads `meta.json`):

```bash
python -m scripts.presets add my-preset --from-log logs/step_response-<stamp> --notes "..."
python -m scripts.presets list
python -m scripts.presets show my-preset
python -m scripts.presets verify my-preset --log logs/step_response-<stamp>
```

## CAN link (rig-host)

See [`docs/can-link.md`](docs/can-link.md): the ODrive USB-CAN adapter is auto-named
`can_odrive` and brought up at 1 Mbit by systemd-networkd; if the interface is
missing, the adapter is unplugged from USB.

## Rig cameras (rig-host)

Two RealSense cameras are attached to the rig host. They are independent:

- **D405 workspace camera** (serial `111111111111`): still frames and per-run
  video via `scripts/record_cam.py` (`--serial 111111111111 --out x.mp4
  --duration 0`, stop with `kill -TERM <pid>`). Leave it alone for streaming.
- **D435 stream camera** (serial `222222222222`): an RTSP server, published by
  `scripts/d435_publish.py` -> ffmpeg -> mediamtx (`scripts/mediamtx_d435.yml`,
  user-space binary in `~/bin`, no system install).

```bash
# on rig-host (or via ssh admin@rig-host '...')
~/bin/d435_rtsp.sh start     # background; logs to ~/d435_rtsp.log
~/bin/d435_rtsp.sh status    # server + publisher PIDs, stream probe
~/bin/d435_rtsp.sh restart   # after unplugging/replugging the camera
~/bin/d435_rtsp.sh stop

# view from anywhere on the LAN
ffplay -rtsp_transport tcp rtsp://rig-host:8554/d435
# or open rtsp://rig-host:8554/d435 in VLC
```

640x480 @ 30 fps, H.264 (x264 zerolatency; the Tegra HW encoder is not reachable
from ffmpeg on this box). Expect ~150-300 ms latency. Port 8554 only; RTMP/HLS/
WebRTC are disabled in the mediamtx config. `~/bin/d435_rtsp.sh` is a symlink to
the copy in `scripts/`, so edit it in the repo and rsync.

## Prior art

The SmartKnob-on-ODrive implementation (`software/odrive_knob/` on an unmerged
branch of the `smartknob` repo) is the direct predecessor. Its README's "Not this
architecture" section is why the impedance target is backend-abstracted here.
