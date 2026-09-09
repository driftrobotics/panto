"""Registry of "known-good" control presets, for gain tuning to always fall
back to a working state.

A preset is a named bundle of the tuning parameters ``scripts/*.py`` take
(``stiffness``, ``vel_gain`` (scalar or per-joint list), ``vel_limit``,
``current``, ``cap_slope`` (scalar or per-joint list), ``cap_min`` (per-joint
list), ``ff_scale``, ``max_pos_gain``), plus
informational fields (``encoder_bandwidth``, ``drive`` -- drive-side settings
that aren't sent by these scripts, just documentation of what the drives were
set to when the preset was verified) and provenance (``notes``, ``verified``).

Stored in ``presets.json`` (repo root, falling back to ``panto/presets.json``
if the repo-root file is absent) as ``{"name": {...fields...}}``.

Scripts call :func:`resolve` with their parsed CLI args (a plain dict of
``field -> value``, ``None`` for anything the user didn't pass) and a preset
name (possibly ``None``): preset values are the base, any non-``None`` CLI
value overrides them field-by-field. The resolved dict plus the preset name
belong in the run's ``meta.json``/summary JSON (item 2 of the request) --
callers just do ``resolved["preset"] = preset_name``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = (_REPO_ROOT / "presets.json", _REPO_ROOT / "panto" / "presets.json")

#: fields a preset may carry that scripts actually consume (in resolve/override
#: order). ``cap_min`` is a 2-list [shoulder, elbow]; everything else scalar.
TUNING_FIELDS = (
    "stiffness", "vel_gain", "vel_limit", "current", "cap_slope", "cap_min",
    "ff_scale", "max_pos_gain",
)

#: fields whose value may be a scalar (applied to both joints) or a 2-list
#: ``[shoulder, elbow]``; normalize via ``panto.step_logic.to_per_joint``.
PER_JOINT_FIELDS = ("cap_slope", "cap_min", "vel_gain")


class PresetError(ValueError):
    pass


@dataclass
class Preset:
    name: str
    stiffness: float | None = None
    vel_gain: float | list[float] | None = None
    vel_limit: float | None = None
    current: float | None = None
    cap_slope: float | list[float] | None = None
    cap_min: list[float] | None = None
    ff_scale: float | None = None
    max_pos_gain: float | None = None
    encoder_bandwidth: float | None = None   # informational -- drives are set separately
    drive: dict[str, Any] = field(default_factory=dict)  # informational drive settings
    notes: str = ""
    verified: str = ""  # free text: date + evidence (e.g. a log dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        del d["name"]
        return d

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Preset":
        known = {f.name for f in fields(cls)} - {"name"}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(name=name, **kwargs)

    def tuning_dict(self) -> dict:
        """Just the fields scripts resolve/override -- no notes/provenance."""
        return {k: getattr(self, k) for k in TUNING_FIELDS}


def _presets_path() -> Path:
    for p in _CANDIDATES:
        if p.exists():
            return p
    return _CANDIDATES[0]


def load_presets(path: str | Path | None = None) -> dict[str, Preset]:
    p = Path(path) if path is not None else _presets_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {name: Preset.from_dict(name, d) for name, d in data.items()}


def save_presets(presets: dict[str, Preset], path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else _presets_path()
    data = {name: preset.to_dict() for name, preset in presets.items()}
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return p


def get_preset(name: str, path: str | Path | None = None) -> Preset:
    presets = load_presets(path)
    if name not in presets:
        raise PresetError(f"unknown preset {name!r}; known: {sorted(presets)}")
    return presets[name]


def resolve(cli_values: dict[str, Any], preset_name: str | None,
           path: str | Path | None = None) -> dict[str, Any]:
    """Merge a preset's tuning values with explicit CLI overrides.

    ``cli_values`` is ``{field: value_or_None}`` for whatever subset of
    ``TUNING_FIELDS`` the calling script supports; ``None`` means "not passed
    on the command line". Explicit (non-``None``) CLI values always win over
    the preset. Returns a plain dict over the same keys as ``cli_values``
    (fields the preset doesn't set stay ``None`` unless overridden).
    """
    base: dict[str, Any] = {k: None for k in cli_values}
    if preset_name is not None:
        preset = get_preset(preset_name, path)
        for k in cli_values:
            v = getattr(preset, k, None)
            if v is not None:
                base[k] = v
    for k, v in cli_values.items():
        if v is not None:
            base[k] = v
    return base


def list_presets(path: str | Path | None = None) -> list[str]:
    return sorted(load_presets(path))


def apply_to_config(config: Any, resolved: dict[str, Any]) -> None:
    """Write a ``resolve()``-d tuning dict into every motor of ``config``.

    This is the single place that pushes resolved tuning fields onto
    ``config.motors`` (``vel_gain``, ``max_pos_gain``, ``cap_min_a``,
    ``cap_vel_slope_a_per_rad_s``, ``ff_scale``). It exists so scripts can't
    silently diverge -- send a resolved ``vel_gain`` to the drive (via
    ``link.set_vel_gains``) *and* to ``PositionBackend``'s config without
    forgetting one of the two.  ``current`` (soft max) is intentionally not
    written here since not every caller wants to overwrite it the same way;
    callers that do should set ``motor.current_soft_max`` themselves.

    ``resolved["cap_min"]``, ``resolved["cap_slope"]``, and
    ``resolved["vel_gain"]`` may be scalars or 2-lists ``[shoulder, elbow]``
    (see ``panto.step_logic.to_per_joint``); ``None`` fields are left
    untouched on the motors.
    """
    from .step_logic import to_per_joint  # local import: avoid an import cycle

    cap_slope = to_per_joint(resolved.get("cap_slope")) if resolved.get("cap_slope") is not None else None
    cap_min = to_per_joint(resolved.get("cap_min")) if resolved.get("cap_min") is not None else None
    vel_gain = to_per_joint(resolved.get("vel_gain")) if resolved.get("vel_gain") is not None else None

    for i, motor in enumerate(config.motors):
        if vel_gain is not None:
            motor.vel_gain = vel_gain[i]
        if resolved.get("max_pos_gain") is not None:
            motor.max_pos_gain = resolved["max_pos_gain"]
        if resolved.get("ff_scale") is not None:
            motor.ff_scale = resolved["ff_scale"]
        if cap_min is not None:
            motor.cap_min_a = cap_min[i]
        if cap_slope is not None:
            motor.cap_vel_slope_a_per_rad_s = cap_slope[i]


def drive_defaults(config: Any) -> dict[int, float]:
    """Capture each motor's pre-override ``vel_gain`` by ``node_id``.

    Call this *before* ``apply_to_config`` mutates ``config.motors`` with a
    resolved preset/CLI ``vel_gain`` override, so the values captured here
    are what the drive should be restored to on exit/abort -- not whatever
    override was just pushed. Scripts that override ``vel_gain`` (directly
    or via a preset) and later restore it with ``link.set_vel_gains`` should
    build this dict first and pass it to ``restore_drive_defaults``.
    """
    return {m.node_id: m.vel_gain for m in config.motors}


def restore_drive_defaults(link: Any, defaults: dict[int, float], log: Any = None) -> None:
    """Restore each node's ``vel_gain`` to its captured pre-override default.

    Best-effort: a failure restoring one node is logged (if ``log`` is
    given, via ``log.event(..., level="ERROR")``) and doesn't stop the rest
    from being attempted -- this runs on abort/disarm paths where the bus
    may already be in a bad state.
    """
    for node_id, vel_gain in defaults.items():
        try:
            link.set_vel_gains(node_id, vel_gain, 0.0)
            if log is not None:
                log.event(f"restored node {node_id} vel_gain -> {vel_gain}")
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log.event(f"restore vel_gain node {node_id}: {exc}", level="ERROR")


if __name__ == "__main__":  # pragma: no cover -- `python -m panto.presets`, same CLI as scripts.presets
    from scripts.presets import main

    main()
