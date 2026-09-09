# CANSimple DBC

Two files here:

- **`odrive-cansimple-0.5.6.dbc`** — vendored verbatim from ODrive's repo (via the
  `reference/odrive_knob` predecessor). `VERSION "0.5.6"`. Kept for provenance.
- **`odrive-cansimple-0.6.x.dbc`** — the 0.5.6 file with **`Heartbeat`** and
  **`Get_Motor_Error` → `Get_Error`** patched to the firmware 0.6.x layout (all
  8 axis copies of each). This is the default `CanLink` / `PantoSim` load
  (`DEFAULT_DBC` in `panto/can_link.py`).

## Why the patches

The panto ODrive Micros run firmware **0.6.11**. Two messages panto touches
changed byte layout between 0.5.6 and 0.6.x:

**`Heartbeat` (0x001):**

| bytes | 0.5.6 | 0.6.x (this file) |
|---|---|---|
| 0–3 | `Axis_Error` u32 | `Axis_Error` u32 *(unchanged — active errors \| disarm reason, combined)* |
| 4 | `Axis_State` u8 | `Axis_State` u8 *(unchanged)* |
| 5 | `Motor_Error_Flag` bit | **`Procedure_Result` u8** |
| 6 | `Encoder_Error_Flag` bit | **`Trajectory_Done_Flag` u8** |
| 7 | `Controller_Error_Flag` / `Trajectory_Done_Flag` bits | reserved |

With the unpatched 0.5.6 DBC, a normal idle heartbeat (`00 00 00 00 01 00 01 00`)
mis-decodes as `Encoder_Error_Flag=1`. `CanLink` reads `Axis_State` and
`Axis_Error` (both unchanged), so this was cosmetic for control — but matters
for diagnostics, which is exactly what this file is for.

**`Get_Motor_Error` (0x003) → `Get_Error`:** 0.5.6 has one `Motor_Error` u32
(bytes 0–3); 0.6.x splits it into **`Active_Errors`** (bytes 0–3, what's wrong
*right now*) and **`Disarm_Reason`** (bytes 4–7, what tripped the axis into
IDLE, latched until `Clear_Errors`). `CanLink` now decodes both — this is how
we told a live fault from a benign "it disarmed once, sitting safely idle now"
during hardware bring-up (2026-09-03: node 0 read `Active_Errors=0,
Disarm_Reason=2` after a control-loop oscillation — a latched protective trip,
not an ongoing problem).

Verified by decoding live captures from the rig (`dbc/README.md` git history /
commit messages have the exact bytes).

Regenerate both patches with the substitution script in the commit that
introduced this file: `VERSION` string; the 4 flag-bit `SG_` lines in each
`Heartbeat` → `Trajectory_Done_Flag : 48|8` + `Procedure_Result : 40|8`; each
`BO_ ... Get_Motor_Error` → `Get_Error` with `SG_ Motor_Error : 0|32` →
`Active_Errors : 0|32` + `Disarm_Reason : 32|32`.

**`Get_Sensorless_Estimates` (0x015/0x035, third patch, 2026-09-04):** 0.5.6's
`Get_Sensorless_Estimates` (`Sensorless_Pos_Estimate` bytes 0–3,
`Sensorless_Vel_Estimate` bytes 4–7) is reused by 0.6.x firmware for
`Get_Temperature` (`FET_Temperature` bytes 0–3, `Motor_Temperature` bytes
4–7) — same cmd ids, same byte layout (both float32 LE), only the message and
signal names changed. Renamed for all 8 axis copies; `CanLink` now decodes
`FET_Temperature`/`Motor_Temperature` from it (`scripts/torque_step.py`'s
thermal reporting). The motor thermistor is disabled on both panto drives
(`motor.motor_thermistor.config.enabled = False`, confirmed 2026-09-04 during
the breakaway-torque investigation) so `Motor_Temperature` reads NaN/0 — that
is expected, not a decode bug.

## Messages panto uses — all byte-compatible 0.5.6 ↔ 0.6.x except the three above

| cmd id | message | note |
|---|---|---|
| 0x001 | `Heartbeat` | patched, see above; we read `Axis_State`, `Axis_Error` |
| 0x003 | `Get_Error` | patched, see above; we read `Active_Errors`, `Disarm_Reason` |
| 0x007 | `Set_Axis_State` | unchanged |
| 0x009 | `Get_Encoder_Estimates` | unchanged (`Pos_Estimate`/`Vel_Estimate`, rev, rev/s) |
| 0x00B | `Set_Controller_Mode` | unchanged (`Control_Mode`, `Input_Mode`) |
| 0x00C | `Set_Input_Pos` | unchanged; `Vel_FF`/`Torque_FF` int16·1e-3 |
| 0x00E | `Set_Input_Torque` | unchanged |
| 0x015 | `Get_Temperature` | patched, see above; we read `FET_Temperature`, `Motor_Temperature` |
| 0x017 | `Get_Bus_Voltage_Current` | unchanged (`Bus_Voltage` V, `Bus_Current` A) |
| 0x00F | `Set_Limits` | unchanged (`Velocity_Limit` rev/s, `Current_Limit` A) |
| 0x014 | `Get_Iq` | unchanged (`Iq_Measured`, A) — our current source for I²t |
| 0x018 | `Clear_Errors` | unchanged (empty payload) |
| 0x01A | `Set_Pos_Gain` | unchanged ((rev/s)/rev) |
| 0x01B | `Set_Vel_Gains` | unchanged |

0.6.x changes panto does **not** use (left as their stale 0.5.6 definitions):
`Get_Encoder_Error`/`Get_Sensorless_Error` (folded into `Get_Error` upstream,
we don't decode the old per-subsystem ones), new `Get_Torques`/`Get_Powers`.
(`Get_Sensorless_Estimates` → `Get_Temperature`, 0x015, is now patched and
used -- see above.)

## Addressing

The DBC enumerates messages per axis (`Axis0_*` = node 0, `Axis1_*` = node 1, …).
panto's drives are node 0 (shoulder) and node 1 (elbow); we resolve `Axis0_*` /
`Axis1_*` by name — no arbitration-id arithmetic. `CanLink` validates
`REQUIRED_MESSAGES` for **both** nodes at load and raises on a mismatch rather
than silently mis-decoding.
