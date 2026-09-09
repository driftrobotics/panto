"""CLI for the control-preset registry (panto.presets / presets.json).

    python -m scripts.presets list
    python -m scripts.presets show hold-0p6
    python -m scripts.presets add my-preset --from-log logs/step_response-20260904-210916 \\
        --notes "explanation"
    python -m scripts.presets verify hold-0p6 --log logs/step_response-20260904-210916

``add``/``verify`` read a run's ``meta.json`` (written by ``panto.telemetry
.RunLogger`` -- every script's CLI args, including the resolved tuning
values) to fill in the preset's tuning fields; they don't guess or re-derive
anything the run didn't already record.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from panto.presets import Preset, TUNING_FIELDS, load_presets, save_presets


def _read_meta(log_dir: str | Path) -> dict:
    path = Path(log_dir) / "meta.json"
    if not path.exists():
        raise SystemExit(f"no meta.json in {log_dir}")
    return json.loads(path.read_text())


def _tuning_from_meta(meta: dict) -> dict:
    """Pull the fields scripts.presets.resolve()-shaped keys out of a run's
    meta.json. Most scripts log flat kwargs; step_response.py's summary.json
    nests them under "config" -- meta.json itself is always flat (it's just
    the CLI args each script passed to RunLogger), so this only has to
    handle the flat case, plus renaming ``cap_slope``/``cap_min`` in from
    the CLI's own flat names."""
    out = {}
    for k in TUNING_FIELDS:
        if k in meta:
            out[k] = meta[k]
    # cap_min may be logged as the raw CLI string ("0.9,0.5") -- normalise to
    # a 2-list so it round-trips the same way a preset's JSON does.
    cap_min = out.get("cap_min")
    if isinstance(cap_min, str):
        parts = [p.strip() for p in cap_min.split(",")]
        if len(parts) == 1:
            out["cap_min"] = [float(parts[0]), float(parts[0])]
        elif len(parts) == 2:
            out["cap_min"] = [float(parts[0]), float(parts[1])]
    return out


def cmd_list(args: argparse.Namespace) -> None:
    presets = load_presets(args.path)
    if not presets:
        print("(no presets)")
        return
    for name in sorted(presets):
        p = presets[name]
        print(f"{name:20s} K={p.stiffness} vel_gain={p.vel_gain} vel_limit={p.vel_limit} "
              f"current={p.current} cap_slope={p.cap_slope} cap_min={p.cap_min} "
              f"ff_scale={p.ff_scale}  verified={p.verified or '(none)'}")


def cmd_show(args: argparse.Namespace) -> None:
    presets = load_presets(args.path)
    if args.name not in presets:
        raise SystemExit(f"unknown preset {args.name!r}; known: {sorted(presets)}")
    print(json.dumps(presets[args.name].to_dict(), indent=2))


def cmd_add(args: argparse.Namespace) -> None:
    presets = load_presets(args.path)
    meta = _read_meta(args.from_log)
    tuning = _tuning_from_meta(meta)
    preset = Preset(name=args.name, notes=args.notes or "", **tuning)
    presets[args.name] = preset
    out = save_presets(presets, args.path)
    print(f"wrote {args.name!r} to {out} from {args.from_log}")
    print(json.dumps(preset.to_dict(), indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    presets = load_presets(args.path)
    if args.name not in presets:
        raise SystemExit(f"unknown preset {args.name!r}; known: {sorted(presets)}")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    presets[args.name].verified = f"{stamp}, log {args.log}"
    save_presets(presets, args.path)
    print(f"{args.name!r} verified: {presets[args.name].verified}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="presets")
    p.add_argument("--path", type=str, default=None, help="presets.json path (default: repo root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list all presets")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="show one preset's full definition")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("add", help="add/overwrite a preset from a run's log dir")
    sp.add_argument("name")
    sp.add_argument("--from-log", required=True, help="a scripts.*.py run's log directory")
    sp.add_argument("--notes", default=None)
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("verify", help="stamp an existing preset as verified today")
    sp.add_argument("name")
    sp.add_argument("--log", required=True, help="log dir that is the evidence for this verification")
    sp.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
