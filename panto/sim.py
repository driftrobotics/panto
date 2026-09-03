"""In-process 2-axis ODrive simulator that speaks real CANSimple frames.

Why this exists: `CanLink(config, sim=True)` must exercise the *identical* code
path as hardware — same DBC encode/decode, same rx thread, same arbitration ids —
so protocol bugs surface without a bench. The sim therefore talks over a
`can.Bus(interface="virtual")` rather than exposing a Python API to `CanLink`.

Physics per node (milestone-2 fidelity, not "feel" fidelity):
  * rigid rotor + viscous damping + Coulomb friction deadband
  * ODrive's cascade — pos P -> vel PI -> torque, then current clamp — at 8 kHz
  * 12-bit MA702 quantisation + an encoder PLL for the velocity estimate
Quantisation is not cosmetic: on hardware it is what caps usable `vel_gain`
(Colgate-Brown), so it must be in the loop the host tunes against.

TODO (stretch, spec "coupled 2R inertia-matrix dynamics"): the two rotors are
currently independent. Real panto has an off-diagonal inertia term M12(q2) and
Coriolis coupling, so fast shoulder motion should back-drive the elbow. Add a
shared `_couple(q, qdot)` step in `_physics_loop` that solves `M(q) qddot = tau`
before this is used for anything past a point-hold. Independent rotors are fine
for bring-up (milestone 2 = static point hold).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import can
import cantools

from .can_link import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CMD,
    CONTROL_MODE_POSITION,
    CONTROL_MODE_TORQUE,
    DEFAULT_DBC,
    as_int,
)

TWO_PI = 2.0 * math.pi


@dataclass
class SimParams:
    """Per-axis plant + estimator parameters.

    Defaults are a rough EM3215 gimbal carrying one 125 mm carbon link; they are
    only meant to give a stable, convergent position loop for tests, not to
    match the real device.
    """

    inertia: float = 6.0e-5          # kg·m^2, rotor + link about the joint axis
    damping: float = 1.2e-3          # N·m/(rad/s) — Colgate-Brown `b`
    friction: float = 8.0e-4         # N·m Coulomb, applied as a net-torque deadband
    torque_constant: float = 0.035   # N·m/A, EM3215
    control_dt: float = 1.0 / 8000.0
    encoder_period: float = 0.002    # mirrors encoder_msg_rate_ms
    heartbeat_period: float = 0.1
    iq_period: float = 0.005         # Get_Iq cyclic rate
    encoder_bits: int = 12           # onboard MA702
    encoder_bandwidth: float = 1000.0  # PLL bw, rad/s


class _AxisPlant:
    """Pure plant + ODrive controller for one node. No threads, no bus."""

    def __init__(self, params: SimParams, home_rad: float = 0.0) -> None:
        self._p = params
        self._lock = threading.Lock()

        # true plant state. `home_rad` is the motor-shaft angle the axis powers
        # up at; defaulting both joints to 0 puts the arm straight out at its
        # extension singularity, so CanLink seeds a non-singular pose in --sim.
        self.angle = home_rad   # rad
        self.velocity = 0.0     # rad/s
        self._external_torque = 0.0  # N·m — "the user's hand"

        # controller state
        self.axis_state = AXIS_STATE_IDLE
        self._control_mode = CONTROL_MODE_POSITION
        self._input_pos = home_rad / TWO_PI  # turns
        self._input_torque = 0.0     # N·m
        self._pos_gain = 20.0        # (turn/s)/turn
        self._vel_gain = 0.02        # N·m/(turn/s)
        self._vel_integrator_gain = 0.0
        self._vel_integrator = 0.0
        self._current_limit = 0.8    # A
        self._velocity_limit = 40.0  # turn/s
        self._last_current = 0.0     # A, for Get_Iq

        # encoder estimator (PLL) state, rad
        self._est_pos = home_rad
        self._est_vel = 0.0

    # -- commands (called from the sim rx thread) --------------------------

    def apply_command(self, base_name: str, signals: dict) -> None:
        with self._lock:
            if base_name == "Set_Input_Pos":
                self._input_pos = float(signals["Input_Pos"])
            elif base_name == "Set_Input_Torque":
                self._input_torque = float(signals["Input_Torque"])
            elif base_name == "Set_Pos_Gain":
                self._pos_gain = float(signals["Pos_Gain"])
            elif base_name == "Set_Vel_Gains":
                self._vel_gain = float(signals["Vel_Gain"])
                self._vel_integrator_gain = float(signals["Vel_Integrator_Gain"])
                self._vel_integrator = 0.0
            elif base_name == "Set_Limits":
                self._velocity_limit = float(signals["Velocity_Limit"])
                self._current_limit = float(signals["Current_Limit"])
            elif base_name == "Set_Controller_Mode":
                self._control_mode = as_int(signals["Control_Mode"])
                self._vel_integrator = 0.0
            elif base_name == "Set_Axis_State":
                self.axis_state = as_int(signals["Axis_Requested_State"])
                self._vel_integrator = 0.0
            # Clear_Errors: nothing to clear in the sim

    def set_external_torque(self, torque_nm: float) -> None:
        with self._lock:
            self._external_torque = float(torque_nm)

    # -- physics ----------------------------------------------------------

    def step(self, n_ticks: int) -> None:
        p = self._p
        dt = p.control_dt
        quantum = TWO_PI / (1 << p.encoder_bits)
        with self._lock:
            for _ in range(n_ticks):
                measured = round(self.angle / quantum) * quantum

                # Encoder PLL: 2nd-order tracker on the quantised angle. Same
                # structure ODrive runs; gives a smoothed velocity estimate.
                err = measured - self._est_pos
                self._est_pos += p.encoder_bandwidth * err * dt
                self._est_vel += (p.encoder_bandwidth ** 2) * err * dt
                self._est_pos += self._est_vel * dt

                torque = self._external_torque
                self._last_current = 0.0
                if self.axis_state == AXIS_STATE_CLOSED_LOOP_CONTROL:
                    ct = self._controller_torque(dt)
                    torque += ct
                    self._last_current = ct / p.torque_constant

                # Coulomb friction as a deadband on net torque.
                if abs(self.velocity) < 1e-4 and abs(torque) < p.friction:
                    torque = 0.0
                else:
                    torque -= math.copysign(p.friction, self.velocity)

                accel = (torque - p.damping * self.velocity) / p.inertia
                self.velocity += accel * dt
                self.angle += self.velocity * dt

    def _controller_torque(self, dt: float) -> float:
        p = self._p
        max_torque = self._current_limit * p.torque_constant

        if self._control_mode == CONTROL_MODE_TORQUE:
            return max(-max_torque, min(max_torque, self._input_torque))

        # POSITION_CONTROL cascade: pos P -> vel PI -> torque.
        pos_turns = self._est_pos / TWO_PI
        vel_turns = self._est_vel / TWO_PI

        vel_cmd = (self._input_pos - pos_turns) * self._pos_gain
        vel_cmd = max(-self._velocity_limit, min(self._velocity_limit, vel_cmd))

        vel_err = vel_cmd - vel_turns
        torque = vel_err * self._vel_gain
        if self._vel_integrator_gain:
            self._vel_integrator += self._vel_integrator_gain * vel_err * dt
            torque += self._vel_integrator
        return max(-max_torque, min(max_torque, torque))

    # -- readbacks (called from the sim tx thread) -----------------------

    def encoder_estimates(self) -> tuple[float, float]:
        with self._lock:
            return self._est_pos / TWO_PI, self._est_vel / TWO_PI

    def iq_measured(self) -> float:
        with self._lock:
            return self._last_current

    def true_angle(self) -> float:
        with self._lock:
            return self.angle


# cmd id -> base message name the sim understands from the host
_HOST_COMMANDS = {
    CMD["Set_Axis_State"]: "Set_Axis_State",
    CMD["Set_Controller_Mode"]: "Set_Controller_Mode",
    CMD["Set_Input_Pos"]: "Set_Input_Pos",
    CMD["Set_Input_Torque"]: "Set_Input_Torque",
    CMD["Set_Limits"]: "Set_Limits",
    CMD["Set_Pos_Gain"]: "Set_Pos_Gain",
    CMD["Set_Vel_Gains"]: "Set_Vel_Gains",
    CMD["Clear_Errors"]: "Clear_Errors",
}


@dataclass
class PantoSim:
    """Two `_AxisPlant`s answering CANSimple on one virtual bus.

    One rx/physics/tx thread set for the pair (not per-axis) so the shared bus is
    only `recv`'d from one place.
    """

    bus: can.BusABC
    node_ids: tuple[int, int] = (0, 1)
    params: SimParams = field(default_factory=SimParams)
    dbc_path: Path = DEFAULT_DBC
    #: motor-shaft power-up angle per node (rad); CanLink sets this in --sim
    home_rad: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        self._db = cantools.database.load_file(str(self.dbc_path))
        self._msg_cache: dict[tuple[int, str], object] = {}
        self._plants = {
            nid: _AxisPlant(self.params, self.home_rad[i])
            for i, nid in enumerate(self.node_ids)
        }
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
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

    # -- test / UI hooks ----------------------------------------------

    def set_external_torque(self, node_id: int, torque_nm: float) -> None:
        """Inject a torque on one rotor — the sim-side of 'the user's hand'.

        Raw motor frame (N·m at the motor). `CanLink.inject_joint_torque`
        applies the per-motor flip so callers can think in joint space.
        """
        self._plants[node_id].set_external_torque(torque_nm)

    def true_angle(self, node_id: int) -> float:
        """True (un-quantised) rotor angle, rad. Tests only."""
        return self._plants[node_id].true_angle()

    # -- internals ----------------------------------------------------

    def _msg(self, node_id: int, base_name: str):
        key = (node_id, base_name)
        if key not in self._msg_cache:
            self._msg_cache[key] = self._db.get_message_by_name(
                f"Axis{node_id}_{base_name}"
            )
        return self._msg_cache[key]

    def _physics_loop(self) -> None:
        # 8 kHz of Python would burn a core; step in batches and pace to wall.
        batch = 40
        dt = self.params.control_dt
        next_t = time.perf_counter()
        while not self._stop.is_set():
            for plant in self._plants.values():
                plant.step(batch)
            next_t += dt * batch
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.bus.recv(timeout=0.2)
            except Exception:
                if self._stop.is_set():
                    return
                raise
            if frame is None:
                continue
            node_id = frame.arbitration_id >> 5
            plant = self._plants.get(node_id)
            if plant is None:
                continue
            base_name = _HOST_COMMANDS.get(frame.arbitration_id & 0x1F)
            if base_name is None:
                continue
            if base_name == "Clear_Errors":
                continue
            try:
                signals = self._msg(node_id, base_name).decode(frame.data)
            except Exception:
                continue
            plant.apply_command(base_name, dict(signals))

    def _tx_loop(self) -> None:
        p = self.params
        next_enc = next_hb = next_iq = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= next_enc:
                for nid, plant in self._plants.items():
                    pos, vel = plant.encoder_estimates()
                    self._emit(nid, "Get_Encoder_Estimates",
                              {"Pos_Estimate": pos, "Vel_Estimate": vel})
                next_enc = now + p.encoder_period
            if now >= next_iq:
                for nid, plant in self._plants.items():
                    iq = plant.iq_measured()
                    self._emit(nid, "Get_Iq",
                              {"Iq_Setpoint": iq, "Iq_Measured": iq})
                next_iq = now + p.iq_period
            if now >= next_hb:
                for nid, plant in self._plants.items():
                    self._emit(nid, "Heartbeat", {
                        "Axis_Error": 0,
                        "Axis_State": plant.axis_state,
                        "Procedure_Result": 0,
                        "Trajectory_Done_Flag": 1,
                    })
                next_hb = now + p.heartbeat_period
            time.sleep(0.0005)

    def _emit(self, node_id: int, base_name: str, signals: dict) -> None:
        msg = self._msg(node_id, base_name)
        try:
            data = msg.encode(signals)
        except Exception:
            return
        self.bus.send(
            can.Message(arbitration_id=msg.frame_id, data=data, is_extended_id=False)
        )
