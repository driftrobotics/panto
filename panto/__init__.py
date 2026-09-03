"""panto — 2-DOF planar haptic linkage.

See HANDOFF.md and the Notion spec. The load-bearing idea: every haptic effect is
an impedance target ``F = K·(x_anchor - x)`` (+ local damping), rendered by a
swappable backend (position or torque mode). Nothing above the backend knows
which one is active.
"""

__version__ = "0.0.0"
