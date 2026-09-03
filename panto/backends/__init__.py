"""Impedance backends: turn ``F = K·(x_anchor - x)`` into ODrive commands.

``position`` is the v0 backend. ``torque`` exists so we can A/B tangential-drag
feel on non-axis-aligned walls without rewriting anything above this layer.
"""

from .base import ImpedanceBackend, ImpedanceCommand
from .position import PositionBackend
from .torque import TorqueBackend

__all__ = [
    "ImpedanceBackend",
    "ImpedanceCommand",
    "PositionBackend",
    "TorqueBackend",
]
