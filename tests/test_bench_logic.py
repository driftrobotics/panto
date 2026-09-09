from panto.bench_logic import (
    BenchResult, GridPoint, best_converged, build_grid, parse_summary_line, report_table,
    should_skip,
)


def _r(k, i, d, verdict):
    return BenchResult(GridPoint(k, i, d, 5.0), {"verdict": verdict})


def test_grid_order_current_major_k_ascending():
    g = build_grid([200, 50], [0.8, 2.0], ["+x", "+y"], 5.0)
    assert [(p.current, p.stiffness, p.direction) for p in g] == [
        (0.8, 50, "+x"), (0.8, 50, "+y"), (0.8, 200, "+x"), (0.8, 200, "+y"),
        (2.0, 50, "+x"), (2.0, 50, "+y"), (2.0, 200, "+x"), (2.0, 200, "+y")]


def test_skip_after_trip_same_cap_and_dir_only():
    res = [_r(100, 2.0, "+x", "limit_cycle"), _r(100, 2.0, "+y", "converged")]
    assert should_skip(GridPoint(200, 2.0, "+x", 5.0), res)
    assert should_skip(GridPoint(200, 2.0, "+y", 5.0), res) is None
    assert should_skip(GridPoint(200, 0.8, "+x", 5.0), res) is None
    assert should_skip(GridPoint(50, 2.0, "+x", 5.0), res) is None


def test_abort_counts_as_trip():
    assert should_skip(GridPoint(400, 2.0, "+x", 5.0), [_r(200, 2.0, "+x", "aborted")])


def test_parse_summary_line_takes_last():
    out = "noise\nSUMMARY_JSON {\"verdict\": \"stall\"}\nlog: x\n"
    assert parse_summary_line(out) == {"verdict": "stall"}
    assert parse_summary_line("nothing") is None


def test_best_converged_requires_all_dirs():
    res = [_r(100, 2.0, "+x", "converged"), _r(100, 2.0, "+y", "converged"),
           _r(200, 2.0, "+x", "converged"), _r(200, 2.0, "+y", "limit_cycle")]
    assert best_converged(res).point.stiffness == 100
    assert best_converged([_r(100, 2.0, "+x", "stall")]) is None


def test_report_table_renders_skips_and_lists():
    r = _r(100, 2.0, "+x", "converged")
    r.summary.update({"peak_current_a": [1.2, 0.4], "i2t_a2s": [3.0, 1.0], "rise_time_s": None})
    md = report_table([r], [(GridPoint(200, 2.0, "+x", 5.0), "K=100 tripped")])
    assert "| 2.0 | 100 | +x | converged | - |" in md
    assert "1.20/0.40" in md
    assert "skipped" in md
