"""In-process 2-axis ODrive simulator that speaks real CANSimple frames.

Why this exists: `CanLink(config, sim=True)` must exercise the *identical* code
path as hardware — same DBC encode/decode, same rx thread, same arbitration ids —
so protocol bugs surface without a bench. The sim therefore talks over a
`can.Bus(interface="virtual")` rather than exposing a Python API to `CanLink`.

Physics per node (milestone-2 fidelity, not "feel" fidelity):
  * rigid rotor + viscous damping + Coulomb friction deadband, optionally a
    torsion spring (models the shoulder's cable-harness spring found in
    2026-09 sysid) and a pure FIFO delay on the controller's torque output
    (models the measured drive-side estimator + transport latency)
  * ODrive's cascade — pos P -> vel PI -> torque, then current clamp — at 8 kHz
  * 12-bit MA702 quantisation + an encoder PLL for the velocity estimate
Quantisation is not cosmetic: on hardware it is what caps usable `vel_gain`
(Colgate-Brown), so it must be in the loop the host tunes against.
`SimParams.from_plant_model` builds a `SimParams` straight from a
`scripts/sysid.py::_write_plant_model` output (amps -> N.m via a supplied
`torque_constant`, nulls fall back to the class defaults).

Coupled 2R dynamics (`PantoSim(..., coupled=True)`, opt-in — independent
rotors, unchanged, remain the default): the per-axis controller/estimator/
encoder/friction code path (`_AxisPlant._tick_common`) is shared between both
modes; only the last torque -> acceleration step differs, swapping the
per-axis scalar divide for a `M(q) qddot + C(q, qdot) qdot = tau` solve
(`_inertia_matrix` + `PantoSim._step_coupled`). Point-mass links (mass `m1`/
`m2` concentrated at `com1`/`com2` from their own joint) — link 1's physical
length isn't its own field; COM is "mid-link" by convention, so `2*com1`
stands in for it, exact at the 30 g/62.5 mm defaults (125 mm links, matching
the real arm). Each axis's own `inertia` (rotor) stays on the matrix diagonal
only, per the "rotor inertia kept as-is" spec — no off-diagonal rotor term.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
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

    #: N.m/rad-equivalent torsion spring, expressed as A/rad (x torque_constant
    #: to get N.m/rad) -- 0 disables it. Models the shoulder cable-harness
    #: spring found in 2026-09 sysid (~3 A/rad). Torque = -k*(angle-home)*Kt,
    #: applied on the plant (i.e. even while IDLE -- it's mechanical).
    torsion_a_per_rad: float = 0.0
    #: When set (A), overrides `friction` as the Coulomb deadband via
    #: coulomb_a * torque_constant. None (default) -> use `friction` as-is.
    coulomb_a: float | None = None
    #: FIFO delay applied to the controller's torque output only (not the
    #: external "hand" torque), rounded to the nearest control_dt tick.
    delay_s: float = 0.0

    # -- coupled 2R arm dynamics (only read when PantoSim(coupled=True)) ----
    m1: float = 0.03    # kg, link 1 mass
    m2: float = 0.03    # kg, link 2 mass
    com1: float = 0.0625  # m, link-1 COM distance from joint 1 (mid-link default)
    com2: float = 0.0625  # m, link-2 COM distance from joint 2 (mid-link default)

    @classmethod
    def from_plant_model(cls, path: str | Path, *, torque_constant: float = 0.02235) -> SimParams:
        """Build a `SimParams` from a `scripts/sysid.py::_write_plant_model`
        JSON file (amps -> N.m via `torque_constant`; that script deliberately
        works in amps since the drive's own Kt is unverified). Any field the
        sysid run didn't produce (or landed as JSON `null`, e.g. a direction
        that never ran) is skipped and the class default is kept."""
        data = json.loads(Path(path).read_text())
        kwargs: dict = {"torque_constant": torque_constant}

        inertia_a = data.get("inertia_a_s2_per_rad")
        if inertia_a is not None:
            kwargs["inertia"] = float(inertia_a) * torque_constant

        viscous = [v for v in (data.get("viscous_a_per_rad_s") or {}).values() if v is not None]
        if viscous:
            kwargs["damping"] = (sum(viscous) / len(viscous)) * torque_constant

        kinetic = [v for v in (data.get("friction_kinetic_intercept_a") or {}).values()
                  if v is not None]
        if kinetic:
            kwargs["coulomb_a"] = sum(kinetic) / len(kinetic)

        delay_s = data.get("delay_s")
        if delay_s is not None:
            kwargs["delay_s"] = float(delay_s)

        torsion = data.get("torsion_a_per_rad")
        if torsion is not None:
            kwargs["torsion_a_per_rad"] = float(torsion)

        return cls(**kwargs)


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
        self._home_rad = home_rad    # torsion spring's rest angle

        # controller-torque FIFO delay, in whole control_dt ticks (0 = none,
        # the default -- see _delayed_controller_torque)
        self._delay_ticks = (
            round(params.delay_s / params.control_dt) if params.control_dt else 0
        )
        self._torque_queue: deque[float] = deque()

        # controller state
        self.axis_state = AXIS_STATE_IDLE
        self._control_mode = CONTROL_MODE_POSITION
        self._input_pos = home_rad / TWO_PI  # turns
        self._input_torque = 0.0     # N·m
        self._pos_gain = 20.0        # (turn/s)/turn
        self._vel_gain = 0.0025      # N·m/(turn/s) — matches config default (drives bumped 10x, 2026-09-03)
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
        """Independent-rotor integration: shared per-tick torque computation
        (`_tick_common`), then a scalar torque -> acceleration divide. This is
        the only place `coupled=False` differs from `PantoSim._step_coupled`
        (a matrix solve) -- everything upstream of the divide is identical."""
        with self._lock:
            for _ in range(n_ticks):
                torque = self._tick_common()
                accel = (torque - self._p.damping * self.velocity) / self._p.inertia
                self.velocity += accel * self._p.control_dt
                self.angle += self.velocity * self._p.control_dt

    def _tick_common(self) -> float:
        """One tick's encoder PLL update + torsion spring + controller torque
        (through its FIFO delay) + Coulomb deadband. Must be called with
        `self._lock` already held (both `step` and `PantoSim._step_coupled`
        do). Returns the net torque, before damping/inertia -- shared by the
        independent and coupled integration paths (see module docstring)."""
        p = self._p
        dt = p.control_dt
        quantum = TWO_PI / (1 << p.encoder_bits)
        measured = round(self.angle / quantum) * quantum

        # Encoder PLL: 2nd-order tracker on the quantised angle. Same
        # structure ODrive runs; gives a smoothed velocity estimate.
        err = measured - self._est_pos
        self._est_pos += p.encoder_bandwidth * err * dt
        self._est_vel += (p.encoder_bandwidth ** 2) * err * dt
        self._est_pos += self._est_vel * dt

        torque = self._external_torque
        if p.torsion_a_per_rad:
            torque += -p.torsion_a_per_rad * (self.angle - self._home_rad) * p.torque_constant
        self._last_current = 0.0
        if self.axis_state == AXIS_STATE_CLOSED_LOOP_CONTROL:
            ct = self._delayed_controller_torque(dt)
            torque += ct
            self._last_current = ct / p.torque_constant

        # Coulomb friction as a deadband on net torque. `coulomb_a`, when
        # set, overrides the fixed N.m `friction` default.
        friction = p.coulomb_a * p.torque_constant if p.coulomb_a is not None else p.friction
        if abs(self.velocity) < 1e-4 and abs(torque) < friction:
            torque = 0.0
        else:
            torque -= math.copysign(friction, self.velocity)
        return torque

    def _delayed_controller_torque(self, dt: float) -> float:
        """`_controller_torque`'s output, shifted `self._delay_ticks` ticks
        via a FIFO. `delay_ticks <= 0` (the default) bypasses the queue
        entirely so the zero-delay path is arithmetically identical to before
        this existed."""
        ct = self._controller_torque(dt)
        if self._delay_ticks <= 0:
            return ct
        self._torque_queue.append(ct)
        if len(self._torque_queue) > self._delay_ticks:
            return self._torque_queue.popleft()
        return 0.0

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


def _inertia_matrix(q2: float, p: SimParams, inertia0: float, inertia1: float) -> tuple[float, float, float]:
    """(M11, M12, M22) for a serial 2R arm — point-mass links (`m1`/`m2`
    concentrated at `com1`/`com2` from their own joint) plus each axis's own
    rotor `inertia` on the diagonal only (no off-diagonal rotor term, per the
    "rotor inertia kept as-is" spec). Standard 2-link-planar-manipulator
    result (e.g. Spong, *Robot Modeling and Control*); M12 == M21 always, by
    construction of a Lagrangian mass matrix.

    Link 1's full length has no dedicated field -- COM is "mid-link" by
    convention, so `2*com1` stands in for it. Exact at the class defaults
    (30 g / 62.5 mm, i.e. 125 mm links, matching the real arm).
    """
    l1 = 2.0 * p.com1
    m1, m2, lc1, lc2 = p.m1, p.m2, p.com1, p.com2
    cos_q2 = math.cos(q2)
    m11 = inertia0 + m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * cos_q2)
    m12 = m2 * (lc2 * lc2 + l1 * lc2 * cos_q2)
    m22 = inertia1 + m2 * lc2 * lc2
    return m11, m12, m22


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
    #: One `SimParams` for both axes, or a `(shoulder, elbow)` 2-tuple so they
    #: can differ. Single-params call sites keep working unchanged.
    params: SimParams | tuple[SimParams, SimParams] = field(default_factory=SimParams)
    dbc_path: Path = DEFAULT_DBC
    #: motor-shaft power-up angle per node (rad); CanLink sets this in --sim
    home_rad: tuple[float, float] = (0.0, 0.0)
    #: When True, integrate both plants as one 2R arm (M(q) + Coriolis) instead
    #: of two independent rotors. See module docstring / `_step_coupled`.
    coupled: bool = False

    def __post_init__(self) -> None:
        self._db = cantools.database.load_file(str(self.dbc_path))
        self._msg_cache: dict[tuple[int, str], object] = {}
        axis_params = self.params if isinstance(self.params, tuple) else (self.params, self.params)
        self._axis_params: tuple[SimParams, SimParams] = axis_params  # type: ignore[assignment]
        self._plants = {
            nid: _AxisPlant(axis_params[i], self.home_rad[i])
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
        # Timing cadence (control_dt) is shared across axes even when their
        # SimParams differ -- taken from node_ids[0]'s params.
        batch = 40
        dt = self._axis_params[0].control_dt
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if self.coupled:
                self._step_coupled(batch)
            else:
                for plant in self._plants.values():
                    plant.step(batch)
            next_t += dt * batch
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def _step_coupled(self, n_ticks: int) -> None:
        """2R-arm integration: `_AxisPlant._tick_common` (shared with the
        independent path) supplies each axis's net torque; the difference is
        the last step, a `M(q) qddot + C(q,qdot) qdot = tau` solve instead of
        a per-axis scalar divide. `node_ids[0]` is link 1 (shoulder, q1),
        `node_ids[1]` is link 2 (elbow, q2), matching `home_rad`'s order.
        Arm geometry (`m1/m2/com1/com2`) is read from `node_ids[0]`'s
        SimParams only -- see the `params` field docstring: pass one SimParams
        for both axes (the common case) and this is unambiguous."""
        p0 = self._axis_params[0]
        dt = p0.control_dt
        plant0, plant1 = (self._plants[nid] for nid in self.node_ids)
        l1 = 2.0 * p0.com1
        m2, lc2 = p0.m2, p0.com2
        d0, d1 = self._axis_params[0].damping, self._axis_params[1].damping
        i0, i1 = self._axis_params[0].inertia, self._axis_params[1].inertia
        for _ in range(n_ticks):
            with plant0._lock, plant1._lock:
                tau0 = plant0._tick_common()
                tau1 = plant1._tick_common()
                q2 = plant1.angle
                qd1, qd2 = plant0.velocity, plant1.velocity

                m11, m12, m22 = _inertia_matrix(q2, p0, i0, i1)
                h = -m2 * l1 * lc2 * math.sin(q2)
                rhs0 = tau0 - h * (2.0 * qd1 * qd2 + qd2 * qd2) - d0 * qd1
                rhs1 = tau1 + h * qd1 * qd1 - d1 * qd2
                det = m11 * m22 - m12 * m12
                qdd1 = (rhs0 * m22 - m12 * rhs1) / det
                qdd2 = (m11 * rhs1 - m12 * rhs0) / det

                plant0.velocity += qdd1 * dt
                plant0.angle += plant0.velocity * dt
                plant1.velocity += qdd2 * dt
                plant1.angle += plant1.velocity * dt

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
        # Cyclic message cadence is shared across axes even when their
        # SimParams differ -- taken from node_ids[0]'s params (same rationale
        # as _physics_loop's dt).
        p = self._axis_params[0]
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
                    # The sim plant never faults, but Get_Error must still be on
                    # the bus so REQUIRED_MESSAGES validation and CanLink's
                    # node_status() exercise the real code path in --sim too.
                    self._emit(nid, "Get_Error",
                              {"Active_Errors": 0, "Disarm_Reason": 0})
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
