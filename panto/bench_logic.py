"""Pure logic for scripts/stiffness_bench.py: grid construction, the
escalation stop rule, and the report table. No CAN, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field

STOP_VERDICTS = frozenset({"limit_cycle", "stall", "aborted", "no_data"})


@dataclass(frozen=True)
class GridPoint:
    stiffness: float
    current: float
    direction: str
    step_mm: float


@dataclass
class BenchResult:
    point: GridPoint
    summary: dict = field(default_factory=dict)
    log_dir: str | None = None
    returncode: int = 0

    @property
    def verdict(self) -> str:
        return str(self.summary.get("verdict", "no_data"))


def build_grid(stiffnesses: list[float], currents: list[float], dirs: list[str],
               step_mm: float) -> list[GridPoint]:
    """Order: current-major, then stiffness ascending, then direction. Ascending
    K within a cap lets the stop rule skip the rest of that cap's ladder."""
    return [GridPoint(k, i, d, step_mm)
            for i in currents for k in sorted(stiffnesses) for d in dirs]


def should_skip(point: GridPoint, results: list[BenchResult]) -> str | None:
    """Skip a point if any lower-or-equal stiffness at the same cap already
    produced a STOP verdict in the same direction. Returns the reason, or None."""
    for r in results:
        p = r.point
        if p.current == point.current and p.direction == point.direction \
                and p.stiffness <= point.stiffness and r.verdict in STOP_VERDICTS:
            return f"K={p.stiffness:g} {p.direction} was {r.verdict}"
    return None


def parse_summary_line(stdout: str) -> dict | None:
    import json
    for line in reversed(stdout.splitlines()):
        if line.startswith("SUMMARY_JSON "):
            return json.loads(line[len("SUMMARY_JSON "):])
    return None


def _fmt(v, nd=2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, list):
        return "/".join(_fmt(x, nd) for x in v)
    return str(v)


def report_table(results: list[BenchResult], skipped: list[tuple[GridPoint, str]]) -> str:
    hdr = ("| cap A | K N/m | dir | verdict | rise s | overshoot mm | settle s | ss err mm "
           "| osc Hz | peak I A | I2t A2s |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for r in results:
        s, p = r.summary, r.point
        rows.append("| " + " | ".join([
            _fmt(p.current, 1), _fmt(p.stiffness, 0), p.direction, r.verdict,
            _fmt(s.get("rise_time_s"), 3), _fmt(s.get("overshoot_mm")),
            _fmt(s.get("settling_time_s"), 3), _fmt(s.get("steady_state_error_mm")),
            _fmt(s.get("osc_freq_hz"), 1), _fmt(s.get("peak_current_a")),
            _fmt(s.get("i2t_a2s"), 1)]) + " |")
    for p, why in skipped:
        rows.append(f"| {p.current:.1f} | {p.stiffness:g} | {p.direction} | skipped | "
                    f"{why} |||||||")
    return "\n".join(rows)


def best_converged(results: list[BenchResult]) -> BenchResult | None:
    """Highest stiffness that converged in every direction tested at its cap."""
    ok = [r for r in results if r.verdict == "converged"]
    if not ok:
        return None
    by_key: dict[tuple[float, float], list[BenchResult]] = {}
    for r in ok:
        by_key.setdefault((r.point.current, r.point.stiffness), []).append(r)
    dirs_tested = {r.point.direction for r in results}
    full = [(k, v) for k, v in by_key.items()
            if {r.point.direction for r in v} == dirs_tested]
    if not full:
        return None
    key, v = max(full, key=lambda kv: kv[0][1])
    return v[0]
