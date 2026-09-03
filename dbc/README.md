# CANSimple DBC

`odrive-cansimple-0.5.6.dbc` is vendored verbatim from ODrive's repo, via the
`reference/odrive_knob` predecessor. **VERSION "0.5.6".**

## Why this file, on 0.6.x hardware

The panto ODrive Micros run firmware **0.6.x**. Every message `panto/can_link.py`
and `panto/sim.py` touch is byte-compatible between 0.5.6 and 0.6.x:

| cmd id | message | note |
|---|---|---|
| 0x001 | `Heartbeat` | 0.6.x adds `Procedure_Result`; we decode defensively (`.get`) |
| 0x007 | `Set_Axis_State` | unchanged |
| 0x009 | `Get_Encoder_Estimates` | unchanged (`Pos_Estimate`/`Vel_Estimate`, rev / rev·s⁻¹) |
| 0x00B | `Set_Controller_Mode` | unchanged (`Control_Mode`, `Input_Mode`) |
| 0x00C | `Set_Input_Pos` | unchanged; `Vel_FF`/`Torque_FF` are int16·1e-3 |
| 0x00E | `Set_Input_Torque` | unchanged |
| 0x00F | `Set_Limits` | unchanged |
| 0x014 | `Get_Iq` | unchanged (`Iq_Measured`, A) — our current source |
| 0x018 | `Clear_Errors` | unchanged (empty payload) |
| 0x01A | `Set_Pos_Gain` | unchanged |
| 0x01B | `Set_Vel_Gains` | unchanged |

0.6.x-only changes we deliberately ignore: `Get_Motor_Error`/`Get_Encoder_Error`
folded into `Get_Error`, new `Get_Torques`/`Get_Powers`, extra Heartbeat fields.

The DBC enumerates messages **per axis** (`Axis0_*` = node 0, `Axis1_*` = node 1,
…). panto's two drives are node 0 (shoulder) and node 1 (elbow), so we address
`Axis0_*` / `Axis1_*` by name directly — no arbitration-id arithmetic.

`CanLink` validates `REQUIRED_MESSAGES` for **both** nodes at load; a DBC that
can't resolve `Axis0_Set_Input_Pos` *and* `Axis1_Set_Input_Pos` (etc.) raises
immediately rather than silently mis-decoding.

To move to a genuine 0.6.x DBC: download it from the ODrive CAN Protocol docs,
drop it in this directory, and point `CanLink(..., dbc_path=...)` at it.
