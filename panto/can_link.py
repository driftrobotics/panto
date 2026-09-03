"""CANSimple link to the two ODrive Micros (node 0 = shoulder, node 1 = elbow).

Same transport as `reference/odrive_knob`: SocketCAN at 1 Mbit/s, drives push
`Get_Encoder_Estimates` / `Get_Iq` cyclically, request/response is kept off the
hot path. One bus, one rx thread, one `Feedback` per node. Gain/limit setters are
change-detected — re-sending an unchanged value every control tick just doubles
bus load. See `dbc/README.md` for the 0.5.6-DBC-on-0.6.x-firmware rationale.

Calibration lives here and nowhere else
=======================================
Everything above `CanLink` speaks *calibrated joint radians*; the ODrive speaks
motor turns off its own 12-bit single-turn absolute encoder. Per motor `i`
(`config.motors[i]`): `flip` (bool, one motor is mounted mirrored) and
`zero_offset_rad` (joint angle when the encoder reads 0). With `s = -1 if flip
else 1`:

    forward (read):   q_rad   = s * (2*pi * pos_turns) + zero_offset_rad
                      qd_rad_s = s * (2*pi * vel_turns_s)
    inverse (command): pos_turns = s * (q_rad - zero_offset_rad) / (2*pi)
                      motor_torque_nm = s * joint_torque_nm

`pos_gain` and the limits are magnitudes — the flip cancels between the position
error and the velocity command, so they pass through untouched (only the rad->
turn unit change applies to `vel_limit`).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import can
import cantools
import numpy as np

TWO_PI = 2.0 * np.pi

DEFAULT_DBC = Path(__file__).resolve().parent.parent / "dbc" / "odrive-cansimple-0.5.6.dbc"

#: Joint angles (rad, elbow-up) the --sim rig powers up at. Well inside the
#: workspace and away from the extension singularity so the arm has force
#: authority to move from the start; hardware homes wherever it physically sits.
SIM_HOME_JOINT_RAD = (0.4, 1.4)

# CANSimple command ids (Firmware/communication/can/can_simple.hpp).
CMD = {
    "Heartbeat": 0x001,
    "Set_Axis_State": 0x007,
    "Get_Encoder_Estimates": 0x009,
    "Set_Controller_Mode": 0x00B,
    "Set_Input_Pos": 0x00C,
    "Set_Input_Torque": 0x00E,
    "Set_Limits": 0x00F,
    "Get_Iq": 0x014,
    "Clear_Errors": 0x018,
    "Set_Pos_Gain": 0x01A,
    "Set_Vel_Gains": 0x01B,
}

# Base message names required in the DBC, validated for *both* nodes at load.
REQUIRED_MESSAGES = (
    "Heartbeat",
    "Set_Axis_State",
    "Get_Encoder_Estimates",
    "Set_Controller_Mode",
    "Set_Input_Pos",
    "Set_Input_Torque",
    "Set_Limits",
    "Get_Iq",
    "Set_Pos_Gain",
    "Set_Vel_Gains",
)

AXIS_STATE_IDLE = 1
AXIS_STATE_CLOSED_LOOP_CONTROL = 8

CONTROL_MODE_TORQUE = 1
CONTROL_MODE_POSITION = 3
INPUT_MODE_PASSTHROUGH = 1

_MODE_TO_CONTROL = {"position": CONTROL_MODE_POSITION, "torque": CONTROL_MODE_TORQUE}


class CanLinkError(RuntimeError):
    pass


def as_int(value) -> int:
    """Coerce a decoded signal (maybe a cantools NamedSignalValue) to int."""
    return int(getattr(value, "value", value))


# --- calibration transform (module-level so tests can hit it directly) -------

def joint_from_turns(pos_turns: float, flip: bool, zero_offset_rad: float) -> float:
    s = -1.0 if flip else 1.0
    return s * (TWO_PI * pos_turns) + zero_offset_rad


def jointvel_from_turns(vel_turns_s: float, flip: bool) -> float:
    s = -1.0 if flip else 1.0
    return s * TWO_PI * vel_turns_s


def turns_from_joint(q_rad: float, flip: bool, zero_offset_rad: float) -> float:
    s = -1.0 if flip else 1.0
    return s * (q_rad - zero_offset_rad) / TWO_PI


def motor_torque_from_joint(tau_nm: float, flip: bool) -> float:
    return (-1.0 if flip else 1.0) * tau_nm


@dataclass
class Feedback:
    """Latest cyclic feedback for one axis (raw motor frame)."""

    pos_turns: float = 0.0
    vel_turns_s: float = 0.0
    iq_measured: float = 0.0
    enc_stamp: float | None = None       # monotonic time of last encoder frame
    iq_stamp: float | None = None
    axis_state: int = 0
    axis_error: int = 0
    heartbeat_stamp: float | None = None


def _open_bus(interface: str, channel: str, bitrate: int) -> can.BusABC:
    kwargs = {"interface": interface, "channel": channel}
    if interface not in ("socketcan", "virtual"):
        kwargs["bitrate"] = bitrate
    return can.Bus(**kwargs)


class CanLink:
    """Two ODrive Micro axes on one CAN bus. Calibrated joint space in/out."""

    def __init__(self, config, *, sim: bool = False, dbc_path: Path = DEFAULT_DBC) -> None:
        self._config = config
        self._sim_mode = sim
        self._db = cantools.database.load_file(str(dbc_path))

        motors = list(config.motors)
        if len(motors) != 2:
            raise CanLinkError(f"expected 2 motors in config, got {len(motors)}")
        self._node_ids: tuple[int, int] = tuple(int(m.node_id) for m in motors)  # type: ignore[assignment]
        self._idx = {nid: i for i, nid in enumerate(self._node_ids)}

        # calibration, per node index. Prefer per-motor zero_offset_rad; fall
        # back to a legacy Config.zero_offset_rad array if the motor lacks it.
        fallback_zero = np.asarray(getattr(config, "zero_offset_rad", np.zeros(2)), float)
        self._flip = [bool(getattr(m, "flip", False)) for m in motors]
        self._zero = []
        for i, m in enumerate(motors):
            z = getattr(m, "zero_offset_rad", None)
            self._zero.append(float(z) if z is not None else float(fallback_zero[i]))

        self._validate_dbc(dbc_path)

        self._feedback = [Feedback(), Feedback()]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx_thread: threading.Thread | None = None

        # change detection for low-rate parameter messages, keyed by node id
        self._last_pos_gain: dict[int, float] = {}
        self._last_limits: dict[int, tuple[float, float]] = {}

        self._tx_count = 0
        self._rx_count = 0

        self._bus: can.BusABC | None = None
        self._owns_bus = False
        self._sim = None  # PantoSim, when sim=True
        self._sim_bus: can.BusABC | None = None

        # arbitration-id -> (node index, frame kind) lookup for the rx thread
        self._rx_map: dict[int, tuple[int, str]] = {}
        for nid in self._node_ids:
            i = self._idx[nid]
            self._rx_map[self._frame_id(nid, "Get_Encoder_Estimates")] = (i, "encoder")
            self._rx_map[self._frame_id(nid, "Heartbeat")] = (i, "heartbeat")
            self._rx_map[self._frame_id(nid, "Get_Iq")] = (i, "iq")

    # ------------------------------------------------------------------ dbc

    def _validate_dbc(self, dbc_path: Path) -> None:
        missing = []
        for nid in self._node_ids:
            for name in REQUIRED_MESSAGES:
                try:
                    self._db.get_message_by_name(f"Axis{nid}_{name}")
                except KeyError:
                    missing.append(f"Axis{nid}_{name}")
        if missing:
            raise CanLinkError(
                f"DBC at {dbc_path} is missing required messages: {', '.join(missing)}.\n"
                "Download the .dbc matching your ODrive firmware from the ODrive CAN "
                "Protocol docs and pass dbc_path=."
            )

    def _msg(self, node_id: int, base_name: str):
        return self._db.get_message_by_name(f"Axis{node_id}_{base_name}")

    def _frame_id(self, node_id: int, base_name: str) -> int:
        return (node_id << 5) | CMD[base_name]

    # ------------------------------------------------------------------ tx

    def _send(self, node_id: int, base_name: str, signals: dict | None = None) -> None:
        data = b"" if signals is None else self._msg(node_id, base_name).encode(signals)
        frame = can.Message(
            arbitration_id=self._frame_id(node_id, base_name),
            data=data,
            is_extended_id=False,
        )
        assert self._bus is not None, "call start() first"
        try:
            self._bus.send(frame)
            self._tx_count += 1
        except can.CanError as exc:
            raise CanLinkError(f"failed to send {base_name} to node {node_id}: {exc}") from exc

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._rx_thread is not None:
            return

        if self._sim_mode:
            from .sim import PantoSim  # local import: sim pulls in this module

            channel = f"panto-sim-{time.monotonic_ns()}"
            self._bus = can.Bus(interface="virtual", channel=channel)
            self._sim_bus = can.Bus(interface="virtual", channel=channel)
            self._owns_bus = True
            home = tuple(
                turns_from_joint(SIM_HOME_JOINT_RAD[i], self._flip[i], self._zero[i]) * TWO_PI
                for i in range(2)
            )
            self._sim = PantoSim(self._sim_bus, node_ids=self._node_ids, home_rad=home)
            self._sim.start()
        else:
            can_if, can_ch, can_br = self._can_settings()
            self._bus = _open_bus(can_if, can_ch, can_br)
            self._owns_bus = True

        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="canlink-rx", daemon=True)
        self._rx_thread.start()

    def _can_settings(self) -> tuple[str, str, int]:
        c = getattr(self._config, "can", None)
        if c is not None and not isinstance(c, (str, bytes)):
            return (
                getattr(c, "interface", "socketcan"),
                getattr(c, "channel", "can0"),
                int(getattr(c, "bitrate", 1_000_000)),
            )
        return (
            getattr(self._config, "can_interface", "socketcan"),
            getattr(self._config, "can_channel", "can0"),
            int(getattr(self._config, "can_bitrate", 1_000_000)),
        )

    def wait_for_feedback(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if all(fb.enc_stamp is not None for fb in self._feedback):
                    return
            time.sleep(0.01)
        raise CanLinkError(
            f"no Get_Encoder_Estimates from nodes {self._node_ids} within {timeout}s. "
            "Check node ids, bitrate, bus termination, and encoder_msg_rate_ms != 0."
        )

    def enter_closed_loop(self, timeout: float = 5.0) -> None:
        for nid in self._node_ids:
            self._send(nid, "Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_CLOSED_LOOP_CONTROL})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                states = [fb.axis_state for fb in self._feedback]
            if all(s == AXIS_STATE_CLOSED_LOOP_CONTROL for s in states):
                return
            time.sleep(0.02)
        with self._lock:
            states = [fb.axis_state for fb in self._feedback]
            errs = [fb.axis_error for fb in self._feedback]
        raise CanLinkError(
            f"axes {self._node_ids} did not reach CLOSED_LOOP_CONTROL "
            f"(states={states}, errors={[hex(e) for e in errs]}). Motor calibrated?"
        )

    def stop(self) -> None:
        for nid in self._node_ids:
            try:
                self._send(nid, "Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_IDLE})
            except CanLinkError:
                pass
        self._stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._sim is not None:
            self._sim.stop()
            self._sim = None
        if self._owns_bus and self._bus is not None:
            self._bus.shutdown()
        if self._sim_bus is not None:
            self._sim_bus.shutdown()
            self._sim_bus = None
        self._bus = None

    # ------------------------------------------------------------------ rx

    def _rx_loop(self) -> None:
        assert self._bus is not None
        enc_msgs = {i: self._msg(nid, "Get_Encoder_Estimates") for nid, i in self._idx.items()}
        hb_msgs = {i: self._msg(nid, "Heartbeat") for nid, i in self._idx.items()}
        iq_msgs = {i: self._msg(nid, "Get_Iq") for nid, i in self._idx.items()}
        while not self._stop.is_set():
            try:
                frame = self._bus.recv(timeout=0.2)
            except Exception:
                if self._stop.is_set():
                    return
                raise
            if frame is None:
                continue
            hit = self._rx_map.get(frame.arbitration_id)
            if hit is None:
                continue
            i, kind = hit
            now = time.monotonic()
            try:
                if kind == "encoder":
                    d = enc_msgs[i].decode(frame.data)
                    with self._lock:
                        fb = self._feedback[i]
                        fb.pos_turns = float(d["Pos_Estimate"])
                        fb.vel_turns_s = float(d["Vel_Estimate"])
                        fb.enc_stamp = now
                        self._rx_count += 1
                elif kind == "iq":
                    d = iq_msgs[i].decode(frame.data)
                    with self._lock:
                        fb = self._feedback[i]
                        fb.iq_measured = float(d["Iq_Measured"])
                        fb.iq_stamp = now
                        self._rx_count += 1
                elif kind == "heartbeat":
                    # decode defensively: 0.6.x reuses the id but adds fields
                    d = hb_msgs[i].decode(frame.data)
                    with self._lock:
                        fb = self._feedback[i]
                        fb.axis_state = as_int(d.get("Axis_State", 0))
                        fb.axis_error = as_int(d.get("Axis_Error", 0))
                        fb.heartbeat_stamp = now
            except Exception:
                continue

    # ------------------------------------------------------------------ reads

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated ([q1, q2] rad, [q1d, q2d] rad/s)."""
        with self._lock:
            pos = [fb.pos_turns for fb in self._feedback]
            vel = [fb.vel_turns_s for fb in self._feedback]
        q = np.array([
            joint_from_turns(pos[i], self._flip[i], self._zero[i]) for i in (0, 1)
        ])
        qd = np.array([
            jointvel_from_turns(vel[i], self._flip[i]) for i in (0, 1)
        ])
        return q, qd

    def feedback_age_s(self) -> float:
        """Age of the *newer* of the two encoder frames — feeds latency comp."""
        with self._lock:
            stamps = [fb.enc_stamp for fb in self._feedback]
        if any(s is None for s in stamps):
            return float("inf")
        return time.monotonic() - max(stamps)

    def motor_currents(self) -> np.ndarray:
        with self._lock:
            return np.array([fb.iq_measured for fb in self._feedback])

    def axis_errors(self) -> tuple[int, int]:
        with self._lock:
            return self._feedback[0].axis_error, self._feedback[1].axis_error

    def counters(self) -> tuple[int, int]:
        with self._lock:
            return self._tx_count, self._rx_count

    # ------------------------------------------------------------------ writes

    def set_controller_mode(self, node_id: int, mode: str) -> None:
        try:
            control_mode = _MODE_TO_CONTROL[mode]
        except KeyError:
            raise CanLinkError(f"unknown controller mode {mode!r}") from None
        self._send(node_id, "Set_Controller_Mode", {
            "Control_Mode": control_mode,
            "Input_Mode": INPUT_MODE_PASSTHROUGH,
        })

    def set_input_pos(self, node_id: int, q_rad: float) -> None:
        i = self._idx[node_id]
        pos_turns = turns_from_joint(q_rad, self._flip[i], self._zero[i])
        self._send(node_id, "Set_Input_Pos", {
            "Input_Pos": pos_turns, "Vel_FF": 0.0, "Torque_FF": 0.0,
        })

    def set_pos_gain(self, node_id: int, gain: float) -> None:
        key = round(gain, 4)
        if key == self._last_pos_gain.get(node_id):
            return
        self._send(node_id, "Set_Pos_Gain", {"Pos_Gain": gain})
        self._last_pos_gain[node_id] = key

    def set_input_torque(self, node_id: int, tau_nm: float) -> None:
        i = self._idx[node_id]
        motor_tau = motor_torque_from_joint(tau_nm, self._flip[i])
        self._send(node_id, "Set_Input_Torque", {"Input_Torque": motor_tau})

    def set_limits(self, node_id: int, vel_limit: float, current_limit: float) -> None:
        vel_turns_s = abs(vel_limit) / TWO_PI  # joint rad/s -> motor turn/s
        key = (round(vel_turns_s, 4), round(current_limit, 4))
        if key == self._last_limits.get(node_id):
            return
        self._send(node_id, "Set_Limits", {
            "Velocity_Limit": vel_turns_s, "Current_Limit": current_limit,
        })
        self._last_limits[node_id] = key

    def set_idle(self, node_id: int) -> None:
        self._send(node_id, "Set_Axis_State", {"Axis_Requested_State": AXIS_STATE_IDLE})

    def clear_errors(self, node_id: int) -> None:
        self._send(node_id, "Clear_Errors", None)

    # ------------------------------------------------------------------ sim hook

    def inject_joint_torque(self, tau_joint) -> None:
        """Sim only: apply an external ('hand') torque in *joint* space.

        Wires the telemetry `perturb {tau:[t0,t1]}` message to the sim, applying
        the same per-motor flip as a real hand would see through the linkage.
        """
        if self._sim is None:
            raise CanLinkError("inject_joint_torque is sim-only")
        tau = np.asarray(tau_joint, float)
        for nid in self._node_ids:
            i = self._idx[nid]
            self._sim.set_external_torque(nid, motor_torque_from_joint(float(tau[i]), self._flip[i]))
