# CANSimple DBC

Two files here:

- **`odrive-cansimple-0.5.6.dbc`** — vendored verbatim from ODrive's repo (via the
  `reference/odrive_knob` predecessor). `VERSION "0.5.6"`. Kept for provenance.
- **`odrive-cansimple-0.6.x.dbc`** — the 0.5.6 file with the **`Heartbeat` message
  patched to the firmware 0.6.x byte layout** (all 8 axis copies). This is the
  default `CanLink` / `PantoSim` load (`DEFAULT_DBC` in `panto/can_link.py`).

## Why the patch

The panto ODrive Micros run firmware **0.6.11**. Only one message panto touches
changed its byte layout between 0.5.6 and 0.6.x: **`Heartbeat` (0x001)**.

| bytes | 0.5.6 | 0.6.x (this file) |
|---|---|---|
| 0–3 | `Axis_Error` u32 | `Axis_Error` u32 *(unchanged)* |
| 4 | `Axis_State` u8 | `Axis_State` u8 *(unchanged)* |
| 5 | `Motor_Error_Flag` bit | **`Procedure_Result` u8** |
| 6 | `Encoder_Error_Flag` bit | **`Trajectory_Done_Flag` u8** |
| 7 | `Controller_Error_Flag` / `Trajectory_Done_Flag` bits | reserved |

With the unpatched 0.5.6 DBC, a normal idle heartbeat (`00 00 00 00 01 00 01 00`)
mis-decodes as `Encoder_Error_Flag=1`. `CanLink` only ever reads `Axis_State` and
`Axis_Error` (both unchanged), so this was cosmetic — but the patched file makes
diagnostic dumps correct and matches the official 0.6.x layout. Verified by
decoding live captures from the rig.

Regenerate with the substitution in the commit that introduced this file
(`VERSION` string + the 4 flag-bit `SG_` lines → `Trajectory_Done_Flag : 48|8` +
`Procedure_Result : 40|8`).

## Messages panto uses — all byte-compatible 0.5.6 ↔ 0.6.x

| cmd id | message | note |
|---|---|---|
| 0x001 | `Heartbeat` | see patch above; we read `Axis_State`, `Axis_Error` |
| 0x007 | `Set_Axis_State` | unchanged |
| 0x009 | `Get_Encoder_Estimates` | unchanged (`Pos_Estimate`/`Vel_Estimate`, rev, rev/s) |
| 0x00B | `Set_Controller_Mode` | unchanged (`Control_Mode`, `Input_Mode`) |
| 0x00C | `Set_Input_Pos` | unchanged; `Vel_FF`/`Torque_FF` int16·1e-3 |
| 0x00E | `Set_Input_Torque` | unchanged |
| 0x00F | `Set_Limits` | unchanged (`Velocity_Limit` rev/s, `Current_Limit` A) |
| 0x014 | `Get_Iq` | unchanged (`Iq_Measured`, A) — our current source for I²t |
| 0x018 | `Clear_Errors` | unchanged (empty payload) |
| 0x01A | `Set_Pos_Gain` | unchanged ((rev/s)/rev) |
| 0x01B | `Set_Vel_Gains` | unchanged |

0.6.x changes panto does **not** use (left as their stale 0.5.6 definitions in the
file): `Get_Motor_Error`/`Get_Encoder_Error` → `Get_Error` (`Active_Errors` +
`Disarm_Reason`), `Get_Sensorless_Estimates` (0x015) → `Get_Temperature`
(`FET_Temperature` + `Motor_Temperature`), new `Get_Torques`/`Get_Powers`. If we
later want `Disarm_Reason` for fault diagnosis, patch `Get_Error` the same way.

## Addressing

The DBC enumerates messages per axis (`Axis0_*` = node 0, `Axis1_*` = node 1, …).
panto's drives are node 0 (shoulder) and node 1 (elbow); we resolve `Axis0_*` /
`Axis1_*` by name — no arbitration-id arithmetic. `CanLink` validates
`REQUIRED_MESSAGES` for **both** nodes at load and raises on a mismatch rather
than silently mis-decoding.
