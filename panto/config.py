"""Static configuration: template + optional live (calibration) overrides.

Why this shape:
  * ``config.template.json`` is the checked-in default. ``calibration.json`` /
    ``*.local.json`` (gitignored) hold per-device calibration the UI writes back.
  * Other work-streams (CAN link, backends, constraints) already import ``Config``
    and reach for flat attributes (``config.geo``, ``config.control_rate_hz`` …).
    Nested sections (``config.control.rate_hz``) are the new home; every old flat
    name is kept as a property alias so nothing downstream breaks.
  * Calibration lives in exactly one place (``CanLink``); this module only *holds*
    the per-motor ``flip`` + ``zero_offset_rad`` it applies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .kinematics import LinkGeometry

_TEMPLATE_PATH = Path(__file__).with_name("config.template.json")
_REPO_ROOT = _TEMPLATE_PATH.parent.parent
#: gitignored live-override filenames, in priority order.
_LIVE_CANDIDATES = ("calibration.json", "config.local.json")


class ConfigError(ValueError):
    """Raised when a merged config fails validation."""


def _kw(cls, d: dict) -> dict:
    """Keep only keys that are fields of ``cls`` (forward-compat with new keys)."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class MotorConfig:
    """Per-motor physical + drive parameters. One motor is physically flipped."""

    node_id: int = 0
    flip: bool = False                  # sign of encoder vs. joint (CanLink applies)
    zero_offset_rad: float = 0.0        # joint angle at raw encoder zero
    torque_constant: float = 0.035      # N·m/A of Iq (EM3215)
    current_soft_max: float = 0.8       # A — datasheet *transient* rating
    vel_gain: float = 0.0025           # local 8 kHz damper, stays on the ODrive.
    #                                   # 2026-09-03: measured 2.5e-4 on both
    #                                   # drives via `odrivetool --can can0`
    #                                   # (10x below the assumed "not-buzzy"
    #                                   # figure -- explained two bring-up
    #                                   # anomalies). Deliberately bumped 10x
    #                                   # on the drives (odrivetool, persisted
    #                                   # with save_configuration()) as the
    #                                   # first step of a real vel_gain sweep;
    #                                   # this default now matches. Re-verify
    #                                   # against the drives before trusting.
    vel_integrator_gain: float = 0.0   # usually 0 for haptics
    max_pos_gain: float = 500.0        # clamp on derived pos_gain, (turn/s)/turn
    vel_limit: float = 20.0            # ODrive turn/s runaway guard (backend uses)

    # --- velocity-scheduled current cap (host-side soft saturation), 2026-09-04
    # relay-oscillation fix: at the current cap, the position cascade is
    # bang-bang (torque = vel_gain*clamp(pos_gain*err, +-vel_limit) then
    # current-capped) with no damping once saturated -- an undamped ~10-14Hz
    # relay whose amplitude scales with the cap. Scheduling the cap down as
    # |qd| grows lets it act like a physical current limit during the fast
    # part of the swing (where saturation happens) while still allowing the
    # full cap_max near zero velocity (holding stiffness). Off by default
    # (slope 0 -> constant current_soft_max, today's behaviour).
    cap_vel_slope_a_per_rad_s: float = 0.0   # k_v: cap_max -> cap_min per rad/s of |qd|
    cap_min_a: float = 0.5                   # floor the schedule never drops below, PER MOTOR
                                              # (shoulder/elbow have very different breakaway --
                                              # a shared floor either starves the shoulder or
                                              # over-currents the elbow)

    # --- Coulomb friction feedforward (2026-09-04 breakaway-informed), sent via
    # Set_Input_Pos's Torque_FF signal (see can_link.set_input_pos). Off by
    # default (0 N.m -> tau_ff always 0, byte-identical to pre-feedforward
    # behaviour). Values are joint-frame breakaway torque, N.m, for +/- motion
    # (measured asymmetric per direction -- see calibration.json's
    # "coulomb_pos_nm"/"coulomb_neg_nm" per motor and the breakaway note in
    # panto-hardware memory). ff_scale is a knob to under-drive the measured
    # breakaway (1.0 = feed forward the full measured value; 0 disables).
    coulomb_pos_nm: float = 0.0        # breakaway torque, +q direction, N.m
    coulomb_neg_nm: float = 0.0        # breakaway torque, -q direction, N.m (positive magnitude)
    ff_scale: float = 0.7

    # --- joint travel limits (2026-09-04: hit a mechanical stop with none
    # configured). Defaults are "unknown, no limiting" -- unset until
    # calibration.json supplies real values for this arm. ---
    q_min_rad: float = float("-inf")   # joint angle at the low mechanical stop
    q_max_rad: float = float("inf")    # joint angle at the high mechanical stop
    limit_margin_rad: float = 0.087    # ~5 deg keep-out inside [q_min, q_max]


@dataclass
class ThermalConfig:
    """I²t accumulator params: integrate ``i² - i_continuous²`` against a budget."""

    i_continuous: float = 0.2          # A, conservative continuous current
    budget_a2s: float = 4.0            # A²·s before force cutback bottoms out

    @property
    def budget(self) -> float:         # legacy alias
        return self.budget_a2s


@dataclass
class CanConfig:
    interface: str = "socketcan"
    channel: str = "can0"
    bitrate: int = 1_000_000


@dataclass
class TestPose:
    """A known-good, hand-verified bring-up pose, set once by the operator
    parking the arm and captured into calibration.json under the ``test_pose``
    key (this module never writes it). ``tip_xy_mm`` is the FK-consistent
    Cartesian check value, ``q_deg`` the joint angles at that pose -- both are
    stored so scripts can sanity-check FK/IK agreement against a physically
    verified reference, not just against themselves.

    2026-09-04: a run centred each new invocation on wherever the *previous*
    run's arm ended up (``--centre-here``); every failed direction left the
    elbow a little more folded, so repeated runs walked the arm toward the
    fold limit / cable harness without any single run's excursion check
    catching it (each excursion was measured from that run's own drifted
    start, not a fixed reference). ``test_pose`` is the fixed reference now:
    offset_sweep's default centre, and the point every run returns to before
    IDLE, so drift can't accumulate across invocations.
    """

    tip_xy_mm: tuple[float, float] = (0.0, 0.0)
    q_deg: tuple[float, float] = (0.0, 0.0)


@dataclass
class ControlConfig:
    rate_hz: float = 200.0
    latency_compensation_s: float = 0.0   # cap on feedback-age pose extrapolation
    # --- not in the JSON template; tune via a live-override file ---
    stiffness_n_per_m: float = 800.0      # isotropic EE stiffness for bilateral
    wall_stiffness_n_per_m: float = 2000.0  # along-normal stiffness for walls
    force_limit_n: float = 3.0            # nominal EE force cap (pre sigma / I²t)


@dataclass
class Config:
    geo: LinkGeometry = field(default_factory=LinkGeometry.panto_v0)
    motors: tuple[MotorConfig, MotorConfig] = field(
        default_factory=lambda: (
            MotorConfig(0, flip=True, torque_constant=0.02235),
            MotorConfig(1, flip=False, torque_constant=0.02235),
        )
    )
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    can: CanConfig = field(default_factory=CanConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

    elbow: str = "up"
    sigma_min_threshold: float = 0.03
    heartbeat_timeout_s: float = 5.0
    workspace_polygon: np.ndarray | None = None
    #: hand-verified bring-up reference pose; None until calibration.json sets
    #: it (never written here -- see TestPose docstring).
    test_pose: TestPose | None = None

    # ---------------------------------------------------------- flat aliases
    # Kept so A/B/C keep importing the names they already use.

    @property
    def control_rate_hz(self) -> float:
        return self.control.rate_hz

    @control_rate_hz.setter
    def control_rate_hz(self, v: float) -> None:
        self.control.rate_hz = v

    @property
    def latency_compensation_s(self) -> float:
        return self.control.latency_compensation_s

    @latency_compensation_s.setter
    def latency_compensation_s(self, v: float) -> None:
        self.control.latency_compensation_s = v

    @property
    def can_interface(self) -> str:
        return self.can.interface

    @property
    def can_channel(self) -> str:
        return self.can.channel

    @property
    def zero_offset_rad(self) -> np.ndarray:
        """[q1, q2] joint offsets, derived from the per-motor calibration."""
        return np.array([m.zero_offset_rad for m in self.motors], dtype=float)

    @zero_offset_rad.setter
    def zero_offset_rad(self, v) -> None:
        v = np.asarray(v, dtype=float).reshape(-1)
        # ``field`` default runs through here at construction (np.zeros default);
        # ignore anything that isn't a real per-joint pair.
        if v.size == len(self.motors):
            for m, off in zip(self.motors, v):
                m.zero_offset_rad = float(off)

    @property
    def q_min_rad(self) -> np.ndarray:
        """[q1, q2] lower joint limits, per-motor. -inf where unknown."""
        return np.array([m.q_min_rad for m in self.motors], dtype=float)

    @property
    def q_max_rad(self) -> np.ndarray:
        """[q1, q2] upper joint limits, per-motor. +inf where unknown."""
        return np.array([m.q_max_rad for m in self.motors], dtype=float)

    @property
    def limit_margin_rad(self) -> np.ndarray:
        """[q1, q2] keep-out margin inside [q_min, q_max], per-motor."""
        return np.array([m.limit_margin_rad for m in self.motors], dtype=float)

    @property
    def test_pose_xy_m(self) -> np.ndarray | None:
        """Hand-verified reference tip pose, metres. None if unset."""
        if self.test_pose is None:
            return None
        return np.asarray(self.test_pose.tip_xy_mm, dtype=float) * 1e-3

    @property
    def test_pose_q_rad(self) -> np.ndarray | None:
        """Hand-verified reference joint angles, radians. None if unset."""
        if self.test_pose is None:
            return None
        return np.radians(np.asarray(self.test_pose.q_deg, dtype=float))

    # ---------------------------------------------------------- (de)serialise

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        geo = LinkGeometry(l1=float(d["geo"]["l1"]), l2=float(d["geo"]["l2"]))
        motors = tuple(MotorConfig(**_kw(MotorConfig, m)) for m in d["motors"])
        poly = d.get("workspace_polygon")
        tp = d.get("test_pose")
        test_pose = None
        if tp is not None:
            test_pose = TestPose(
                tip_xy_mm=tuple(float(v) for v in tp["tip_xy_mm"]),
                q_deg=tuple(float(v) for v in tp["q_deg"]),
            )
        cfg = cls(
            geo=geo,
            motors=motors,
            thermal=ThermalConfig(**_kw(ThermalConfig, d.get("thermal", {}))),
            can=CanConfig(**_kw(CanConfig, d.get("can", {}))),
            control=ControlConfig(**_kw(ControlConfig, d.get("control", {}))),
            elbow=str(d.get("elbow", "up")),
            sigma_min_threshold=float(d.get("sigma_min_threshold", 0.03)),
            heartbeat_timeout_s=float(d.get("heartbeat_timeout_s", 5.0)),
            workspace_polygon=(
                None if poly is None else np.asarray(poly, dtype=float)
            ),
            test_pose=test_pose,
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return {
            "geo": {"l1": self.geo.l1, "l2": self.geo.l2},
            "motors": [asdict(m) for m in self.motors],
            "thermal": asdict(self.thermal),
            "can": asdict(self.can),
            "control": asdict(self.control),
            "elbow": self.elbow,
            "sigma_min_threshold": self.sigma_min_threshold,
            "heartbeat_timeout_s": self.heartbeat_timeout_s,
            "workspace_polygon": (
                None if self.workspace_polygon is None
                else np.asarray(self.workspace_polygon).tolist()
            ),
            "test_pose": None if self.test_pose is None else asdict(self.test_pose),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise for the calibration UI to write back to the live file."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Template + one live-override file, deep-merged and validated.

        ``path`` forces a specific override file; otherwise the first of
        ``calibration.json`` / ``config.local.json`` found in CWD or repo root.
        """
        base = json.loads(_TEMPLATE_PATH.read_text())
        live = None
        if path is not None:
            live = json.loads(Path(path).read_text())
        else:
            for name in _LIVE_CANDIDATES:
                for root in (Path.cwd(), _REPO_ROOT):
                    p = root / name
                    if p.is_file():
                        live = json.loads(p.read_text())
                        break
                if live is not None:
                    break
        merged = _merge(base, live) if live is not None else base
        return cls.from_dict(merged)

    # ---------------------------------------------------------- validation

    def validate(self) -> None:
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        check(self.geo.l1 > 0 and self.geo.l2 > 0, "link lengths must be > 0")
        check(len(self.motors) == 2, "exactly two motors required")
        check(
            self.motors[0].node_id != self.motors[1].node_id,
            "motor node_ids must be distinct",
        )
        for m in self.motors:
            check(m.torque_constant > 0, f"motor {m.node_id}: torque_constant > 0")
            check(m.current_soft_max > 0, f"motor {m.node_id}: current_soft_max > 0")
            check(m.vel_gain >= 0, f"motor {m.node_id}: vel_gain >= 0")
            check(m.max_pos_gain > 0, f"motor {m.node_id}: max_pos_gain > 0")
            check(m.limit_margin_rad >= 0, f"motor {m.node_id}: limit_margin_rad >= 0")
            check(m.q_min_rad < m.q_max_rad, f"motor {m.node_id}: q_min_rad < q_max_rad")
        check(self.thermal.i_continuous >= 0, "thermal.i_continuous >= 0")
        check(self.thermal.budget_a2s > 0, "thermal.budget_a2s > 0")
        check(self.control.rate_hz > 0, "control.rate_hz > 0")
        check(
            self.control.latency_compensation_s >= 0,
            "control.latency_compensation_s >= 0",
        )
        check(self.elbow in ("up", "down"), "elbow must be 'up' or 'down'")
        check(self.sigma_min_threshold > 0, "sigma_min_threshold > 0")
        check(self.heartbeat_timeout_s > 0, "heartbeat_timeout_s > 0")
        if self.workspace_polygon is not None:
            poly = np.asarray(self.workspace_polygon)
            check(
                poly.ndim == 2 and poly.shape[1] == 2 and poly.shape[0] >= 3,
                "workspace_polygon must be (N>=3, 2) or null",
            )


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge ``over`` onto ``base``. ``motors`` merges per-index so a live
    file can carry just ``{flip, zero_offset_rad}`` for one motor."""
    out = dict(base)
    for k, v in over.items():
        if k == "motors" and isinstance(v, list) and isinstance(out.get(k), list):
            merged = [dict(m) for m in out[k]]
            for i, m in enumerate(v):
                if i < len(merged):
                    merged[i].update(m)
                else:
                    merged.append(dict(m))
            out[k] = merged
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
