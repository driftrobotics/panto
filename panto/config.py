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
    vel_gain: float = 0.00025          # local 8 kHz damper, stays on the ODrive;
    #                                   # read directly off both drives via
    #                                   # `odrivetool --can can0` on 2026-09-03
    #                                   # (controller.config.vel_gain); NOT the
    #                                   # 2.5e-3 "sane not-buzzy" figure -- that
    #                                   # was never actually on the drives.
    #                                   # Low damping -> low max stable
    #                                   # stiffness (Colgate-Brown); needs the
    #                                   # milestone-3 vel_gain sweep to raise
    #                                   # for real (host can't change it, only
    #                                   # odrivetool / Set_Vel_Gains can).
    vel_integrator_gain: float = 0.0   # usually 0 for haptics
    max_pos_gain: float = 500.0        # clamp on derived pos_gain, (turn/s)/turn
    vel_limit: float = 20.0            # ODrive turn/s runaway guard (backend uses)


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
        default_factory=lambda: (MotorConfig(0), MotorConfig(1, flip=True))
    )
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    can: CanConfig = field(default_factory=CanConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

    elbow: str = "up"
    sigma_min_threshold: float = 0.03
    heartbeat_timeout_s: float = 5.0
    workspace_polygon: np.ndarray | None = None

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

    # ---------------------------------------------------------- (de)serialise

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        geo = LinkGeometry(l1=float(d["geo"]["l1"]), l2=float(d["geo"]["l2"]))
        motors = tuple(MotorConfig(**_kw(MotorConfig, m)) for m in d["motors"])
        poly = d.get("workspace_polygon")
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
