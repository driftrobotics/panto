"""A simulated ODrive axis on a virtual CAN bus.

Exists so the detent engine, the CAN codec and the UI can all be exercised
without hardware -- and so the *same* code path runs in both cases: the sim
speaks real CANSimple frames over ``can.Bus(interface="virtual")``.

The physics is deliberately simple (rigid rotor, viscous damping, ODrive's
cascaded P/PI controller at 8 kHz). It reproduces the things that matter for
validating the architecture -- snapping, endstops, multi-detent jumps, the
effect of host latency -- but it will not tell you how the knob *feels*. Only
hardware does that.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import can
import cantools

from .odrive_can import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CMD_CLEAR_ERRORS,
    CMD_GET_ENCODER_ESTIMATES,
    CMD_HEARTBEAT,
    CMD_SET_AXIS_STATE,
    CMD_SET_CONTROLLER_MODE,
    CMD_SET_INPUT_POS,
    CMD_SET_LIMITS,
    CMD_SET_POS_GAIN,
    CMD_SET_VEL_GAINS,
    DEFAULT_DBC,
    _as_int,
)

TWO_PI = 2.0 * math.pi


@dataclass
class SimParams:
    """Plant parameters, roughly a 32 mm gimbal motor with a knob on it."""

    #: kg.m^2. Rotor + knob cap.
    inertia: float = 1.5e-5
    #: N.m/(rad/s). Bearing + eddy losses. This is Colgate & Brown's `b`, the
    #: term that dominates how stiff a spring you can render.
    damping: float = 5e-4
    #: N.m. Static friction, applied as a deadband on net torque.
    friction: float = 2e-4
    torque_constant: float = 0.035
    #: Physics/controller tick. Matches ODrive's real loop rate.
    control_dt: float = 1.0 / 8000.0
    #: How often to emit Get_Encoder_Estimates, mirroring encoder_msg_rate_ms.
    encoder_period: float = 0.002
    heartbeat_period: float = 0.1
    #: 12-bit MA702, as on the ODrive Micro. Quantisation here is not cosmetic:
    #: it is what limits usable vel_gain on real hardware.
    encoder_bits: int = 12
    #: Encoder PLL bandwidth (rad/s) for the velocity estimate.
    encoder_bandwidth: float = 1000.0


class SimulatedODrive:
    """Answers CANSimple frames for one node_id on a virtual bus."""

    def __init__(
        self,
        bus: can.BusABC,
        node_id: int = 0,
        params: Optional[SimParams] = None,
        dbc_path: Path = DEFAULT_DBC,
    ) -> None:
        self._bus = bus
        self._node_id = node_id
        self._p = params or SimParams()
        self._db = cantools.database.load_file(str(dbc_path))

        # Plant state
        self._angle = 0.0        # rad, true
        self._velocity = 0.0     # rad/s, true
        self._external_torque = 0.0  # N.m, "the user's hand"

        # Controller state
        self._axis_state = AXIS_STATE_IDLE
        self._input_pos = 0.0
        self._pos_gain = 0.0
        self._vel_gain = 0.02
        self._vel_integrator_gain = 0.0
        self._vel_integrator = 0.0
        self._current_limit = 0.8
        self._velocity_limit = 20.0

        # Encoder estimator state
        self._est_pos = 0.0
        self._est_vel = 0.0

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ api

    def start(self) -> None:
        self._stop.clear()
        for target, name in (
            (self._physics_loop, "sim-physics"),
            (self._rx_loop, "sim-rx"),
            (self._tx_loop, "sim-tx"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    def apply_external_torque(self, torque_nm: float) -> None:
        """Simulate a hand on the knob."""
        with self._lock:
            self._external_torque = torque_nm

    @property
    def angle(self) -> float:
        with self._lock:
            return self._angle

    def _message(self, name: str):
        try:
            return self._db.get_message_by_name(f"Axis0_{name}")
        except KeyError:
            return self._db.get_message_by_name(name)

    # ------------------------------------------------------------------ physics

    def _physics_loop(self) -> None:
        dt = self._p.control_dt
        p = self._p
        quantum = TWO_PI / (1 << p.encoder_bits)
        next_t = time.perf_counter()
        # Run in batches: 8 kHz of Python would burn a core for nothing.
        batch = 40

        while not self._stop.is_set():
            with self._lock:
                for _ in range(batch):
                    measured = round(self._angle / quantum) * quantum

                    # Encoder PLL: tracks quantised position, produces a
                    # smoothed velocity. Same structure ODrive uses.
                    err = measured - self._est_pos
                    self._est_pos += p.encoder_bandwidth * err * dt
                    self._est_vel += (p.encoder_bandwidth ** 2) * err * dt
                    self._est_pos += self._est_vel * dt

                    torque = self._external_torque
                    if self._axis_state == AXIS_STATE_CLOSED_LOOP_CONTROL:
                        torque += self._controller_torque(dt)

                    # Coulomb friction as a deadband on net torque.
                    if abs(self._velocity) < 1e-4 and abs(torque) < p.friction:
                        torque = 0.0
                    else:
                        torque -= math.copysign(p.friction, self._velocity)

                    accel = (torque - p.damping * self._velocity) / p.inertia
                    self._velocity += accel * dt
                    self._angle += self._velocity * dt

            next_t += dt * batch
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def _controller_torque(self, dt: float) -> float:
        """ODrive's cascade: pos P -> vel PI -> torque, then current clamp."""
        pos_turns = self._est_pos / TWO_PI
        vel_turns = self._est_vel / TWO_PI

        vel_cmd = (self._input_pos - pos_turns) * self._pos_gain
        vel_cmd = max(-self._velocity_limit, min(self._velocity_limit, vel_cmd))

        vel_err = vel_cmd - vel_turns
        torque = vel_err * self._vel_gain
        if self._vel_integrator_gain:
            self._vel_integrator += self._vel_integrator_gain * vel_err * dt
            torque += self._vel_integrator

        max_torque = self._current_limit * self._p.torque_constant
        return max(-max_torque, min(max_torque, torque))

    # ------------------------------------------------------------------ can

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._bus.recv(timeout=0.2)
            except Exception:
                if self._stop.is_set():
                    return
                raise
            if frame is None or (frame.arbitration_id >> 5) != self._node_id:
                continue
            cmd = frame.arbitration_id & 0x1F
            try:
                self._handle(cmd, frame)
            except Exception:
                continue

    def _handle(self, cmd: int, frame: can.Message) -> None:
        if cmd == CMD_SET_INPUT_POS:
            d = self._message("Set_Input_Pos").decode(frame.data)
            with self._lock:
                self._input_pos = float(d["Input_Pos"])
        elif cmd == CMD_SET_POS_GAIN:
            d = self._message("Set_Pos_Gain").decode(frame.data)
            with self._lock:
                self._pos_gain = float(d["Pos_Gain"])
        elif cmd == CMD_SET_VEL_GAINS:
            d = self._message("Set_Vel_Gains").decode(frame.data)
            with self._lock:
                self._vel_gain = float(d["Vel_Gain"])
                self._vel_integrator_gain = float(d["Vel_Integrator_Gain"])
                self._vel_integrator = 0.0
        elif cmd == CMD_SET_LIMITS:
            d = self._message("Set_Limits").decode(frame.data)
            with self._lock:
                self._velocity_limit = float(d["Velocity_Limit"])
                self._current_limit = float(d["Current_Limit"])
        elif cmd == CMD_SET_AXIS_STATE:
            d = self._message("Set_Axis_State").decode(frame.data)
            with self._lock:
                self._axis_state = _as_int(d["Axis_Requested_State"])
                self._vel_integrator = 0.0
        elif cmd == CMD_SET_CONTROLLER_MODE:
            pass  # position/passthrough is all the sim implements
        elif cmd == CMD_CLEAR_ERRORS:
            pass

    def _tx_loop(self) -> None:
        enc_msg = self._message("Get_Encoder_Estimates")
        hb_msg = self._message("Heartbeat")
        enc_id = (self._node_id << 5) | CMD_GET_ENCODER_ESTIMATES
        hb_id = (self._node_id << 5) | CMD_HEARTBEAT

        next_enc = time.perf_counter()
        next_hb = next_enc

        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= next_enc:
                with self._lock:
                    pos, vel = self._est_pos / TWO_PI, self._est_vel / TWO_PI
                data = enc_msg.encode({"Pos_Estimate": pos, "Vel_Estimate": vel})
                self._bus.send(can.Message(arbitration_id=enc_id, data=data, is_extended_id=False))
                next_enc = now + self._p.encoder_period
            if now >= next_hb:
                with self._lock:
                    state = self._axis_state
                data = hb_msg.encode(
                    {
                        "Axis_Error": 0,
                        "Axis_State": state,
                        "Motor_Error_Flag": 0,
                        "Encoder_Error_Flag": 0,
                        "Controller_Error_Flag": 0,
                        "Trajectory_Done_Flag": 0,
                    }
                )
                self._bus.send(can.Message(arbitration_id=hb_id, data=data, is_extended_id=False))
                next_hb = now + self._p.heartbeat_period
            time.sleep(0.0005)
