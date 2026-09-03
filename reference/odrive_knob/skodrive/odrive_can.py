"""ODrive CANSimple client, built on python-can + cantools.

Uses ODrive's published ``odrive-cansimple.dbc`` so message layouts, scaling and
endianness come from their spec rather than hand-rolled struct packing.

Arbitration IDs are ``node_id << 5 | cmd_id`` (11-bit standard frames). The DBC
enumerates messages per axis (``Axis0_Set_Input_Pos`` = 12, ``Axis1_`` = 44,
...), so for node IDs it covers we use ``frame_id`` directly; beyond that we
reuse the Axis0 signal layout and compute the ID ourselves.

Firmware compatibility: the DBC vendored here is VERSION 0.5.6. Every message
this module uses is unchanged in 0.6.x -- only Heartbeat gained fields, and we
decode it defensively. ``REQUIRED_MESSAGES`` is validated at connect time so a
mismatched DBC fails loudly instead of silently mis-decoding.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import can
import cantools

log = logging.getLogger(__name__)

DEFAULT_DBC = Path(__file__).resolve().parent.parent / "dbc" / "odrive-cansimple-0.5.6.dbc"

# CANSimple command IDs (Firmware/communication/can/can_simple.hpp).
CMD_HEARTBEAT = 0x001
CMD_SET_AXIS_STATE = 0x007
CMD_GET_ENCODER_ESTIMATES = 0x009
CMD_SET_CONTROLLER_MODE = 0x00B
CMD_SET_INPUT_POS = 0x00C
CMD_SET_INPUT_TORQUE = 0x00E
CMD_SET_LIMITS = 0x00F
CMD_CLEAR_ERRORS = 0x018
CMD_SET_POS_GAIN = 0x01A
CMD_SET_VEL_GAINS = 0x01B

# Base (Axis0) message names we require in the DBC.
REQUIRED_MESSAGES: Dict[str, int] = {
    "Heartbeat": CMD_HEARTBEAT,
    "Set_Axis_State": CMD_SET_AXIS_STATE,
    "Get_Encoder_Estimates": CMD_GET_ENCODER_ESTIMATES,
    "Set_Controller_Mode": CMD_SET_CONTROLLER_MODE,
    "Set_Input_Pos": CMD_SET_INPUT_POS,
    "Set_Limits": CMD_SET_LIMITS,
    "Set_Pos_Gain": CMD_SET_POS_GAIN,
    "Set_Vel_Gains": CMD_SET_VEL_GAINS,
}

AXIS_STATE_IDLE = 1
AXIS_STATE_FULL_CALIBRATION_SEQUENCE = 3
AXIS_STATE_MOTOR_CALIBRATION = 4
AXIS_STATE_ENCODER_OFFSET_CALIBRATION = 7
AXIS_STATE_CLOSED_LOOP_CONTROL = 8

CONTROL_MODE_TORQUE = 1
CONTROL_MODE_POSITION = 3

INPUT_MODE_PASSTHROUGH = 1


class ODriveError(RuntimeError):
    pass


def _as_int(value) -> int:
    """Coerce a decoded signal to int.

    cantools returns ``NamedSignalValue`` for signals that carry a VAL_ table
    (Axis_State does), which is not directly int()-able.
    """
    return int(getattr(value, "value", value))


@dataclass
class Feedback:
    """Latest cyclic feedback from the axis."""

    pos_turns: float = 0.0
    vel_turns_s: float = 0.0
    #: monotonic timestamp of the encoder frame, or None if never received.
    stamp: Optional[float] = None
    axis_state: int = 0
    axis_error: int = 0
    heartbeat_stamp: Optional[float] = None

    @property
    def age(self) -> float:
        if self.stamp is None:
            return float("inf")
        return time.monotonic() - self.stamp


class ODriveAxis:
    """One ODrive axis addressed over CANSimple.

    Gain and limit setters are change-detected: resending an unchanged value
    every cycle would double bus load for no benefit.
    """

    def __init__(
        self,
        bus: can.BusABC,
        node_id: int = 0,
        dbc_path: Path = DEFAULT_DBC,
        owns_bus: bool = False,
    ) -> None:
        self._bus = bus
        self._owns_bus = owns_bus
        self._node_id = node_id
        self._db = cantools.database.load_file(str(dbc_path))
        self._msg_cache: Dict[str, object] = {}
        self._validate_dbc(dbc_path)

        self._feedback = Feedback()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None

        # Change detection for low-rate parameter messages.
        self._last_pos_gain: Optional[float] = None
        self._last_vel_gains: Optional[Tuple[float, float]] = None
        self._last_limits: Optional[Tuple[float, float]] = None

        self._tx_count = 0
        self._rx_count = 0

    # ------------------------------------------------------------------ dbc

    def _validate_dbc(self, dbc_path: Path) -> None:
        missing = []
        for name in REQUIRED_MESSAGES:
            try:
                self._db.get_message_by_name(f"Axis0_{name}")
            except KeyError:
                try:
                    self._db.get_message_by_name(name)
                except KeyError:
                    missing.append(name)
        if missing:
            raise ODriveError(
                f"DBC at {dbc_path} is missing required messages: {', '.join(missing)}.\n"
                "Download the .dbc matching your firmware from the ODrive CAN Protocol "
                "docs and pass --dbc."
            )

    def _message(self, name: str):
        """Resolve a base message name to its cantools definition (Axis0 layout)."""
        if name not in self._msg_cache:
            try:
                msg = self._db.get_message_by_name(f"Axis0_{name}")
            except KeyError:
                msg = self._db.get_message_by_name(name)
            self._msg_cache[name] = msg
        return self._msg_cache[name]

    def _frame_id(self, name: str) -> int:
        return (self._node_id << 5) | REQUIRED_MESSAGES[name]

    def _send(self, name: str, signals: Optional[dict] = None) -> None:
        if signals is None:
            data = b""
        else:
            data = self._message(name).encode(signals)
        frame = can.Message(
            arbitration_id=self._frame_id(name),
            data=data,
            is_extended_id=False,
        )
        try:
            self._bus.send(frame)
            self._tx_count += 1
        except can.CanError as exc:
            raise ODriveError(f"failed to send {name}: {exc}") from exc

    # ------------------------------------------------------------------ rx

    def start(self) -> None:
        if self._rx_thread is not None:
            return
        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="odrive-rx", daemon=True)
        self._rx_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._owns_bus:
            self._bus.shutdown()

    def _rx_loop(self) -> None:
        encoder_id = self._frame_id("Get_Encoder_Estimates")
        heartbeat_id = self._frame_id("Heartbeat")
        while not self._stop.is_set():
            try:
                frame = self._bus.recv(timeout=0.2)
            except Exception:  # bus torn down under us during shutdown
                if self._stop.is_set():
                    return
                raise
            if frame is None:
                continue
            now = time.monotonic()
            if frame.arbitration_id == encoder_id:
                decoded = self._message("Get_Encoder_Estimates").decode(frame.data)
                with self._lock:
                    self._feedback.pos_turns = float(decoded["Pos_Estimate"])
                    self._feedback.vel_turns_s = float(decoded["Vel_Estimate"])
                    self._feedback.stamp = now
                    self._rx_count += 1
            elif frame.arbitration_id == heartbeat_id:
                # Decode defensively: 0.6.x reuses the ID but adds fields.
                try:
                    decoded = self._message("Heartbeat").decode(frame.data)
                except Exception:
                    continue
                with self._lock:
                    self._feedback.axis_state = _as_int(decoded.get("Axis_State", 0))
                    self._feedback.axis_error = _as_int(decoded.get("Axis_Error", 0))
                    self._feedback.heartbeat_stamp = now

    @property
    def feedback(self) -> Feedback:
        with self._lock:
            return Feedback(**vars(self._feedback))

    @property
    def counters(self) -> Tuple[int, int]:
        return self._tx_count, self._rx_count

    # ------------------------------------------------------------------ commands

    def clear_errors(self) -> None:
        frame = can.Message(
            arbitration_id=(self._node_id << 5) | CMD_CLEAR_ERRORS,
            data=b"",
            is_extended_id=False,
        )
        self._bus.send(frame)

    def set_axis_state(self, state: int) -> None:
        self._send("Set_Axis_State", {"Axis_Requested_State": state})

    def set_controller_mode(self, control_mode: int, input_mode: int) -> None:
        self._send(
            "Set_Controller_Mode",
            {"Control_Mode": control_mode, "Input_Mode": input_mode},
        )

    def set_limits(self, velocity_limit: float, current_limit: float, force: bool = False) -> None:
        key = (round(velocity_limit, 4), round(current_limit, 4))
        if not force and key == self._last_limits:
            return
        self._send(
            "Set_Limits",
            {"Velocity_Limit": velocity_limit, "Current_Limit": current_limit},
        )
        self._last_limits = key

    def set_pos_gain(self, pos_gain: float, force: bool = False) -> None:
        key = round(pos_gain, 4)
        if not force and key == self._last_pos_gain:
            return
        self._send("Set_Pos_Gain", {"Pos_Gain": pos_gain})
        self._last_pos_gain = key

    def set_vel_gains(self, vel_gain: float, vel_integrator_gain: float, force: bool = False) -> None:
        key = (round(vel_gain, 6), round(vel_integrator_gain, 6))
        if not force and key == self._last_vel_gains:
            return
        self._send(
            "Set_Vel_Gains",
            {"Vel_Gain": vel_gain, "Vel_Integrator_Gain": vel_integrator_gain},
        )
        self._last_vel_gains = key

    def set_input_pos(self, pos_turns: float, vel_ff: float = 0.0, torque_ff: float = 0.0) -> None:
        """Position setpoint plus feedforwards, in one frame.

        Note the DBC scales Vel_FF and Torque_FF as int16 * 0.001, so torque
        feedforward quantises to 1 mN.m -- coarse relative to this motor's
        ~31 mN.m peak. The knob doesn't use it; an arm doing gravity comp would
        want to check that resolution is adequate.
        """
        self._send(
            "Set_Input_Pos",
            {"Input_Pos": pos_turns, "Vel_FF": vel_ff, "Torque_FF": torque_ff},
        )

    # ------------------------------------------------------------------ lifecycle

    def wait_for_feedback(self, timeout: float = 5.0) -> Feedback:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fb = self.feedback
            if fb.stamp is not None:
                return fb
            time.sleep(0.01)
        raise ODriveError(
            f"no Get_Encoder_Estimates from node {self._node_id} within {timeout}s. "
            "Check node_id, bitrate, termination, and that encoder_msg_rate_ms != 0."
        )

    def enter_closed_loop(self, timeout: float = 5.0) -> None:
        self.set_axis_state(AXIS_STATE_CLOSED_LOOP_CONTROL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fb = self.feedback
            if fb.axis_state == AXIS_STATE_CLOSED_LOOP_CONTROL:
                return
            time.sleep(0.02)
        state = self.feedback.axis_state
        raise ODriveError(
            f"axis {self._node_id} did not reach CLOSED_LOOP_CONTROL (state={state}, "
            f"error=0x{self.feedback.axis_error:08x}). Is the motor calibrated?"
        )

    def idle(self) -> None:
        try:
            self.set_axis_state(AXIS_STATE_IDLE)
        except ODriveError:
            log.warning("failed to command IDLE on shutdown", exc_info=True)


def open_bus(interface: str, channel: str, bitrate: int) -> can.BusABC:
    """Open a python-can bus.

    For a USB-CAN adapter prefer a SocketCAN-backed device (``gs_usb`` /
    candleLight) over an slcan serial tunnel -- the latter adds multi-millisecond
    jitter, which matters because our snap decisions are latency-sensitive.
    """
    kwargs = {"interface": interface, "channel": channel}
    if interface not in ("socketcan", "virtual"):
        kwargs["bitrate"] = bitrate
    return can.Bus(**kwargs)
