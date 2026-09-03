"""Static configuration. Offline-editable for v0; online settings UI is deferred.

Values here are placeholders — the control-mode experiments (milestone 3) fill in
the gain numbers, and calibration fills in zero offsets + the workspace polygon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import LinkGeometry


@dataclass
class MotorConfig:
    node_id: int
    torque_constant: float = 0.035     # N·m/A, EM3215, see odrive_knob README
    current_soft_max: float = 0.8      # A — datasheet *transient* rating
    vel_gain: float = 0.0              # set from the vel_gain sweep


@dataclass
class ThermalConfig:
    # I²t accumulator: integrate (I² - i_cont²), trip when the budget is spent,
    # then soft-cutback the force limit. i_cont near the 0.11–0.13 A no-load
    # current is very conservative; tune against measured motor temp.
    i_continuous: float = 0.2          # A
    budget: float = 4.0               # A²·s before cutback begins


@dataclass
class Config:
    geo: LinkGeometry = field(default_factory=LinkGeometry.panto_v0)
    motors: tuple[MotorConfig, MotorConfig] = field(
        default_factory=lambda: (MotorConfig(0), MotorConfig(1))
    )
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    can_interface: str = "socketcan"
    can_channel: str = "can0"

    control_rate_hz: float = 200.0
    latency_compensation_s: float = 0.0   # set from the round-trip histogram

    elbow: str = "up"
    sigma_min_threshold: float = 0.03     # m, workspace boundary conditioning
    heartbeat_timeout_s: float = 5.0

    # Calibration artefacts, populated by the calibration UI / scripts.
    zero_offset_rad: np.ndarray = field(default_factory=lambda: np.zeros(2))
    workspace_polygon: np.ndarray | None = None
