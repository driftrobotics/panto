"""CANSimple link to the two ODrives.

Same transport as odrive_knob: SocketCAN at 1 Mbit/s, cyclic
``Get_Encoder_Estimates`` pushed by the drives, request/response avoided on the
hot path. Reuse its DBC (``odrive-cansimple-0.5.6.dbc``) — every message we need
(Set_Axis_State, Get_Encoder_Estimates, Set_Controller_Mode, Set_Input_Pos,
Set_Input_Torque, Set_Limits, Set_Pos_Gain, Set_Vel_Gains, Clear_Errors,
Heartbeat) is unchanged from 0.5.6 to the Micro's 0.6.x. Validate required
messages at startup and fail loudly on a DBC mismatch.

Two nodes (id 0, id 1) on one bus. Decode Heartbeat defensively.
"""

from __future__ import annotations

import numpy as np


class CanLink:
    def __init__(self, interface: str, channel: str, node_ids: tuple[int, int]):
        raise NotImplementedError

    # --- reads (from the cyclic stream) ---
    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Latest ([q1, q2] rad, [q1_dot, q2_dot] rad/s)."""
        raise NotImplementedError

    def feedback_age_s(self) -> float:
        """Age of the newest encoder frame — feeds latency compensation."""
        raise NotImplementedError

    def motor_currents(self) -> np.ndarray:
        raise NotImplementedError

    # --- writes ---
    def set_input_pos(self, node_id: int, pos_rev: float) -> None: ...
    def set_pos_gain(self, node_id: int, gain: float) -> None: ...
    def set_input_torque(self, node_id: int, torque_nm: float) -> None: ...
    def set_idle(self, node_id: int) -> None: ...
    def clear_errors(self, node_id: int) -> None: ...
