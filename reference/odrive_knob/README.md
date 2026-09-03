# SmartKnob on an ODrive

A reimplementation of the SmartKnob haptics with an ODrive as the motor
controller, no TFT, and the UI rendered in HTML.

The original firmware computes a torque every iteration at ~1 kHz on an ESP32
(`firmware/src/motor_task.cpp`). Here the control law is split:

```
Host, 200 Hz                      ODrive, 8 kHz              Motor
-----------                       -------------              -----
read pos/vel  ──────────────┐
detent state machine        │     position P loop     ┐
where is the spring         │       + velocity PI     ├──▶  ~31 mN.m
how stiff is it   ──────────┴──▶  (the spring)        ┘
     Set_Input_Pos + Set_Pos_Gain over CAN
```

**Why this works at 200 Hz.** A detent is a *bilateral spring* — it attracts
from both sides toward its centre — which is exactly what ODrive's position
controller is. So the host doesn't need to close a loop; it only moves the
spring's anchor when the knob crosses a snap point, and adjusts stiffness when
the mode changes. The latency-critical part (velocity damping) stays on the
ODrive at 8 kHz, where it belongs: a damper rendered through a delay `T`
contributes *negative* damping near `1/(2T)`, so host-side damping at 200 Hz
would actively destabilise the knob.

This does **not** generalise to Cartesian virtual walls on a multi-DOF arm — a
wall is a *unilateral* constraint and needs coordinated `Jᵀ·F` across joints.
See the "Not this architecture" section.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# No hardware: simulated ODrive on a virtual CAN bus. Drag the dial to turn it.
.venv/bin/python -m skodrive --sim

# Real hardware
.venv/bin/python -m skodrive --interface socketcan --channel can0 --node-id 0
```

Then open <http://127.0.0.1:8080>.

## Hardware prerequisites

Configure and calibrate the ODrive with `odrivetool` **before** running this —
it does not perform calibration, it assumes a working closed-loop axis.

```python
odrv0.axis0.config.motor.torque_constant   = 0.035   # 8.27/KV, see below
odrv0.axis0.config.motor.pole_pairs        = 7
odrv0.axis0.config.motor.current_soft_max  = 0.8

odrv0.axis0.config.can.node_id             = 0
odrv0.axis0.config.can.encoder_msg_rate_ms = 2       # must be non-zero
odrv0.axis0.config.can.heartbeat_msg_rate_ms = 100
odrv0.can.config.baud_rate                 = 1000000

odrv0.axis0.requested_state = AxisState.FULL_CALIBRATION_SEQUENCE
odrv0.axis0.requested_state = AxisState.ANTICOGGING_CALIBRATION   # ~6 min, worth it
odrv0.axis0.config.anticogging.enabled = True
odrv0.save_configuration()
```

Bring the interface up first: `sudo ip link set can0 up type can bitrate 1000000`.

### Why CAN and not USB

Use USB for bring-up and calibration; use CAN to run.

⚠️ On the Micro, **USB and DC input may not be connected at the same time unless
you use a USB isolator.** Since driving the motor requires DC power, USB is not
a viable runtime transport without extra hardware and ground-loop care.

CAN also pushes `Get_Encoder_Estimates` cyclically, whereas the USB (fibre)
protocol is request/response — you'd be polling for position, doubling the round
trips. And USB is one cable per drive, which doesn't scale past this knob.

### Adapter choice

ODrive's own [USB-CAN Adapter](https://shop.odriverobotics.com/products/usb-can-adapter)
is the straightforward option: CAN 2.0B to 1 Mbit/s, and it enumerates as a
**native SocketCAN device** on Linux via the `gs_usb` kernel driver (USB PID
`0x606f`). `odrivetool` auto-detects it and reaches every ODrive on the bus, so
one adapter covers both tuning and the control loop.

Prefer any SocketCAN-backed adapter (`gs_usb` / candleLight) over an `slcan`
serial tunnel — the latter adds multi-millisecond jitter, and snap decisions are
latency-sensitive.

Note: one ODrive **cannot** bridge USB to CAN for other ODrives. The firmware has
no gateway function; a board's USB interface exposes only its own object tree.

Gotcha: on Linux the adapter binds as SocketCAN by default, which blocks ODrive's
Web GUI from using it. Unbind `gs_usb` to use the GUI, rebind for `can0` — one at
a time.

### Motor defaults

Defaults target the Wanzhida/Oncetop OT-EM3215D2450Y1R (SparkFun ROB-20441), the
same motor the original SmartKnob uses. From the SparkFun datasheet, two
independent routes to `Kt` agree within ~25%:

| | |
|---|---|
| start torque / start current | 320 g·cm / 0.8 A = **0.039 N·m/A** |
| 8.27 / KV, KV = 2000 rpm / 7.4 V | **0.031 N·m/A** |

Default is 0.035, giving **~31 mN·m** at 0.8 A — about 1.6× what the stock
SmartKnob gets from voltage-mode SimpleFOC at 5 V.

⚠️ **Thermal.** 0.8 A is the datasheet's *starting* (transient) rating, not a
continuous one — it's ~4.4 W of copper loss in a 32 mm motor with no airflow.
Detents are intermittent so normal use is fine, but **holding against an endstop
will cook it**. If you add continuous gravity compensation later, re-derive this.

## Tuning

Two knobs, in order.

**1. `--vel-gain` (N·m per turn/s).** The local 8 kHz damper. Turn it up until
the motor starts to hiss, then back off ~30%. Everything else derives from this.

You will hit a ceiling here, and it isn't the loop rate — it's the ODrive
Micro's 12-bit onboard encoder. 4096 counts/rev differentiated at 8 kHz is
12.3 rad/s of velocity noise per LSB; the PLL filters it at the cost of phase
lag, and `vel_gain` multiplies whatever survives. Per Colgate & Brown, damping
dominates achievable stiffness, so **encoder resolution is what caps how stiff a
detent you can render.** If you outgrow it, an external higher-resolution
encoder buys more than a faster loop would.

**2. `--latency-compensation` (seconds).** Snap decisions are late by the host +
CAN + USB round trip, so click position error is `ω × T`:

| rotation speed | error at T = 7 ms | vs. an 8.2° detent |
|---|---|---|
| 0.3 rev/s | 0.8° | ~9% — fine |
| 1 rev/s | 2.5° | ~30% — noticeable |

Measure your actual round trip (watch `feedback age` in the UI) and set this to
it. Default 0 — overshoot causes early snaps, so don't guess high.

Design rule: **`detent_width >> ω_max × T_latency`.** Coarse detents at
deliberate speeds feel good; **fine detents (the 1° presets) will not work** in
this architecture. Judge it on "Coarse values / Strong detents", not "Fine
values". If you need fine detents at speed, the state machine has to move back
onto an MCU on the CAN bus at 1 kHz.

`detent_strength_unit` is width-independent: 1.0 means "peak motor torque at
half a detent width of deflection", so the same value feels equivalent whether
detents are 5° or 60° wide.

## Loop health

The UI's health panel is the diagnostic for most "feel" complaints:

| metric | meaning |
|---|---|
| `rate` | should sit at your `--rate` |
| `jitter p95` | > 2 ms and clicks land inconsistently |
| `feedback age` | round trip; also what to set `--latency-compensation` to |
| `max steps` | detents crossed per update. Persistently > 1 = the loop is behind and clicks are smearing |
| `overruns` | iterations that missed their deadline |

## What's ported, and what isn't

Ported from `motor_task.cpp` / `interface_task.cpp`, with all 11 presets:
snap-point hysteresis, bounds and endstops, magnetic detents, `snap_point_bias`
asymmetry, idle re-centring, config idempotency.

Two deliberate deviations:

- **Multi-step snapping.** The firmware moves at most one detent per iteration
  (`motor_task.cpp:240-248`), capping tracking at `rate × detent_width`. At
  200 Hz that would be 0.55 rev/s for 1° detents. We snap by absolute angle
  instead, so position stays correct at any speed. This was a latent bug in the
  original at 1 kHz too.
- **Rate-independent time constants.** The firmware's EWMA alphas are tuned for
  1 kHz; they're expressed here as time constants so behaviour doesn't change
  with `--rate`.

Not ported: strain-gauge press detection, LEDs, ambient light sensor (all needed
an MCU, none touch the ODrive), and the dead zone — available via `--dead-zone`
but off by default, since a well-tuned `vel_gain` makes the buzz it suppresses a
non-issue.

## Not this architecture

Pushing the spring into the ODrive works for a knob because a detent is a
bilateral, joint-space spring. It does **not** extend to 6-DOF Cartesian walls:

- ODrive's position controller is bilateral; a wall is unilateral. Anchoring at
  a wall surface makes the wall *suck you in* from the free side.
- A Cartesian wall needs `τ = Jᵀ·F` — coupled and configuration-dependent.
  ODrive gives one scalar `pos_gain` per axis, i.e. a diagonal joint-space
  stiffness, which cannot produce "stiff into the wall, free along it".
- Switching control modes at runtime produces a torque discontinuity you feel at
  exactly the worst moment — contact transition.

For that, stay in torque mode permanently and do impedance control host-side,
keeping only velocity damping local.

## Firmware / DBC compatibility

`dbc/odrive-cansimple-0.5.6.dbc` is vendored from ODrive's repo. It is
**VERSION 0.5.6**; the Micro runs 0.6.x, where Heartbeat gained
`Procedure_Result`, `Get_Motor_Error`/`Get_Encoder_Error` became `Get_Error`, and
`Get_Torques`/`Get_Powers` were added.

Every message this code uses is unchanged across both — `Set_Axis_State` (0x07),
`Get_Encoder_Estimates` (0x09), `Set_Controller_Mode` (0x0B), `Set_Input_Pos`
(0x0C), `Set_Limits` (0x0F), `Set_Pos_Gain` (0x1A), `Set_Vel_Gains` (0x1B),
`Clear_Errors` (0x18) — and Heartbeat is decoded defensively. Required messages
are validated at startup, so a mismatched DBC fails loudly.

To use the 0.6.x file: download it from the ODrive CAN Protocol docs and pass
`--dbc path/to/odrive-cansimple.dbc`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

- `tests/test_haptics.py` — the detent state machine, no I/O. Pins down the
  behaviours the original firmware had no tests for (`firmware/test/` is a
  README only).
- `tests/test_integration.py` — control loop through the real CANSimple codec to
  a simulated ODrive over a python-can virtual bus. Runs in real time.

The simulator (`skodrive/sim.py`) models a rigid rotor with viscous damping and
Coulomb friction, ODrive's cascaded controller at 8 kHz, and 12-bit encoder
quantisation. It validates the architecture — snapping, endstops, multi-detent
jumps, latency effects — but **it will not tell you how the knob feels.** Only
hardware does that.
