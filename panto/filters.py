"""Small, dependency-free discrete filters shared by the host-side control
loops (currently just :mod:`panto.backends.torque`). Kept separate from the
backend so they're unit-testable without a CAN bus or sim -- same split as
``breakaway_logic.py``/``step_logic.py``.

Both filters are *stateful*: construct one per signal (e.g. one per joint)
and call ``update(x, dt)`` once per tick. ``dt`` is measured wall time
between ticks (the host loop is not hard-real-time), not a fixed constant --
passing a wildly different ``dt`` from call to call is fine, it just changes
the effective cutoff for that one step.
"""

from __future__ import annotations

import math


class OnePoleLowPass:
    """First-order (RC) low-pass, cutoff ``fc_hz``. ``fc_hz <= 0`` disables
    filtering (``update`` returns ``x`` unchanged, no state kept)."""

    def __init__(self, fc_hz: float) -> None:
        self.fc_hz = fc_hz
        self._y: float | None = None

    def reset(self, x0: float = 0.0) -> None:
        self._y = x0

    def update(self, x: float, dt: float) -> float:
        if self.fc_hz <= 0.0 or dt <= 0.0:
            self._y = x
            return x
        if self._y is None:
            self._y = x
            return x
        rc = 1.0 / (2.0 * math.pi * self.fc_hz)
        alpha = dt / (rc + dt)
        self._y = self._y + alpha * (x - self._y)
        return self._y


class Notch:
    """Digital biquad notch (band-stop) filter, centre ``f0_hz`` / quality
    ``q``. ``f0_hz <= 0`` disables filtering (``update`` returns ``x``
    unchanged). Coefficients are recomputed each call from the *current*
    ``dt`` (the sample rate implied by the host loop isn't fixed), which is
    cheap relative to a CAN round-trip and keeps the notch centred correctly
    even if the loop rate drifts a bit tick to tick.
    """

    def __init__(self, f0_hz: float, q: float) -> None:
        self.f0_hz = f0_hz
        self.q = q
        self._x1 = self._x2 = 0.0
        self._y1 = self._y2 = 0.0

    def reset(self) -> None:
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0

    def update(self, x: float, dt: float) -> float:
        if self.f0_hz <= 0.0 or dt <= 0.0:
            return x
        fs = 1.0 / dt
        # Below Nyquist the standard RBJ notch design is well-conditioned;
        # above it (a too-slow loop for the requested notch), skip filtering
        # for this tick rather than fold the frequency and notch the wrong
        # thing.
        if self.f0_hz >= 0.5 * fs:
            return x
        w0 = 2.0 * math.pi * self.f0_hz / fs
        alpha = math.sin(w0) / (2.0 * self.q)
        cos_w0 = math.cos(w0)

        b0, b1, b2 = 1.0, -2.0 * cos_w0, 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha

        y = (b0 / a0) * x + (b1 / a0) * self._x1 + (b2 / a0) * self._x2 \
            - (a1 / a0) * self._y1 - (a2 / a0) * self._y2

        self._x2, self._x1 = self._x1, x
        self._y2, self._y1 = self._y1, y
        return y
