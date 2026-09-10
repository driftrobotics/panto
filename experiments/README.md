# experiments/

Record-keeping for panto rig runs. Nothing in this directory touches
hardware; it summarizes runs that already happened on rig-host
(`admin@rig-host:~/code/panto/logs/<script>-<stamp>/{meta.json,summary.json,events.log,samples.jsonl}`).

## Files

- **`log.md`** — hand-written narrative, newest first, one section per
  session/date. This is the primary source of *conclusions* — what was
  tried, what it showed, what's still open. Extend it by hand when you learn
  something; cite the backing run stamp(s) inline (`logs/<script>-<stamp>`)
  so a conclusion can be traced back to data.
- **`runs.md`** — machine-generated index of every run since 2026-09-04:
  stamp, script, preset, K/vel_gain/vel_limit/cap, verdict, key metrics,
  I2t. This is the row-level ground truth `log.md` refers to; don't hand-edit
  it, regenerate it (see below).
- **`run_notes.md`** — machine-assisted batch-level pass over `runs.md`:
  consecutive runs (gap ≤15 min) grouped into the experiment batch they
  belong to, with one line on what the batch tested and what it showed,
  cross-checked against `log.md` and the dated memory notes. Anything not
  directly supported by `log.md`/memory notes is marked **uncertain** rather
  than inferred. Use this to find which batch a given stamp belongs to, or
  to spot stamp ranges nobody has written up yet.
- **`plots/`** — PNGs referenced from `log.md` (joint-error traces, box
  traces, chirp/FRF plots, etc.), named after the run or comparison they
  came from.

## Regenerating runs.md

`runs.md` is built by `scripts/experiment_log.py`, which walks
`logs/<script>-<stamp>/`, reads `meta.json` + `summary.json`, and emits one
table row per run. It must run **on rig-host** (it reads the log tree there),
then get copied back:

```
ssh admin@rig-host 'cd ~/code/panto && .venv/bin/python -m scripts.experiment_log --since 2026-09-04 --out /tmp/runs.md'
scp admin@rig-host:/tmp/runs.md experiments/runs.md
```

Use `--since 2026-09-04` to regenerate the full record this repo covers
(not `--since 2026-09-08` — that only covers the estimator-fix era onward
and silently drops the 09-04 bring-up runs; `log.md`'s own header comment
is stale on this point as of 2026-09-10 and should be fixed to match
whatever `--since` you actually use to produce the checked-in file).

`experiment_log.py` reads `cfg.get(...)`/`meta.get(...)` defensively and
did not crash when regenerated 2026-09-10 for `--since 2026-09-04` (500
runs, breakaway/impedance_step/observe/offset_sweep/point_hold/step_response/
stiffness_bench/sysid/test_sysid_chirp/test_sysid_cogging/test_sysid_friction/
test_sysid_latency/torque_step/trace_shape all represented). If it does
crash in the future, report the traceback rather than patching
`scripts/experiment_log.py` from here — that file is out of scope for
record-keeping edits.

## Frame and unit conventions (from `log.md`)

Tip frame: **+x** along the table edge toward the camera, **+y** into the
table, **CCW positive** (right-handed, viewed from above). Zero pose = both
joints fully extended toward the camera.

- `K` — isotropic tip stiffness, N/m.
- `vg` — ODrive `vel_gain`, N·m/(turn/s); written `shoulder/elbow` when the
  two joints are given different gains.
- `vl` — the `--vel-limit` flag, joint rad/s. Position-mode torque plateau
  is `vel_gain × vel_limit / (2π)` N·m — a small `vl` is a hidden current
  cap, easy to mistake for a stability win.
- `cap` — per-axis current limit, amps at the drive.
- All currents are amps at the drive. Torque constant label: `Kt = 0.02235`
  N·m/A (per `panto-calibration-2026-09-04`, unmeasured — don't bake it into
  anything that needs calibrated newtons without measuring it).

## Known caveats when reading old runs

- Every `step_response`/`goto_pose` run from **2026-09-04 evening through
  2026-09-08 17:08 UTC** has a labelled `K` that is ~30x the actual
  commanded stiffness (config-default `vel_gain` fed `pos_gain` instead of
  the CLI value — see `log.md` 2026-09-08 and 2026-09-04 sections). Treat
  the qualitative shape of those results as informative, not the K number.
  `offset_sweep` runs in that window were unaffected.
- `encoder_bandwidth` 100 (used through most of 09-04–09-08) adds 40–85° of
  velocity-estimate lag at 10–20 Hz and explains most limit cycles seen
  before 300 was adopted 2026-09-08 — don't re-diagnose those as gain or
  mechanical problems.
- Stability results are **pose-dependent** (`panto-stiffness-bench-2026-09-09`):
  always read the pose a run started at, not just K/cap, before comparing
  it to another run.
