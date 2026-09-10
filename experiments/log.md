# panto experiment log

Narrative record of what was tried and learned, newest first. Run names are
`logs/<script>-<stamp>` on rig-host (`admin@rig-host:~/code/panto/logs`); the
machine-generated index of every run with its config and metrics is
`experiments/runs.md` (`python -m scripts.experiment_log --since 2026-09-04 --out experiments/runs.md`
on rig-host). Plots referenced here live in `experiments/plots/`.

Conventions: tip frame +x along the table edge toward the camera, +y into the
table, CCW positive. `K` is isotropic tip stiffness N/m; `vg` is ODrive
`vel_gain` N·m/(turn/s), given as `shoulder/elbow` when split; `vl` is the
`--vel-limit` flag in joint rad/s (position-mode torque plateau =
vel_gain × vel_limit/(2π) N·m, so small `vl` is a hidden current cap); cap is the
per-axis current limit. All currents are amps at the drive; Kt label 0.02235.

## 2026-09-10 — vel_gain decimation, per-joint gains, hold feedforward

Rig: shoulder hovering in free space, harness rerouted (user verified rigid
coupling at every joint; drop the two-mass hypothesis from 09-09). j0 cool to
the touch; no mandatory cooldowns while inside motor spec.

- `step_response-20260910-144942/145054` K100 vg0.02/0.01 vl50: 18 mm and
  28 mm shoulder relay cycles at ±0.8 A, elbow joins once the vel clamp is
  lifted. **K100 cannot be held linearly at 0.8 A for any vel_gain.**
  Plot `jerr_K100_vg_down.png`.
- `145130/145243/145354` K25 vg0.01/0.005/0.002 vl50: shoulder cleanest at
  0.005–0.01 (±1.5° ring decaying ~0.3 s, no dither); elbow rings at 12 Hz
  below 0.02 and goes unstable at 0.002 (±5° at cap). Plot
  `jerr_K25_decimate.png`.
- `145510/145621/145732` K10: quiet at every vg, ss 0.8–1.1 mm (too soft
  to hold the harness bias).
- `145845..150005` per-joint vg 0.01/0.05: **K25 works** (+y os 2.2 mm,
  ss 0.49; +x ss 0.18, settle 0.56 s); **K50 relay-cycles** even with the
  split (linear band too narrow at 0.8 A). Plot `jerr_perjoint.png`.
- `trace_shape-20260910-150117/150134` box 25 mm, K25 vg0.01/0.05 vl50:
  RMS 1.14 mm @10 mm/s, 1.25 mm @25 mm/s, shoulder 0.2 A RMS (was 2.7 mm).
  Joint-space trace shows the remaining error is spring sag: θ0 error tracks
  Iq0 one-for-one (−1.3° at −0.45 A on the far side). Plot
  `box_k25pj_0910.png`. Preset `hover-K25-pj`.
- Hold-torque model fit on that box (quasi-static samples, joint frame):
  shoulder I = −0.539 + 0.0317·q0° + 0.0181·q1° A (resid 0.105 A; q0 alone
  0.137 A), elbow I = −0.416 + 0.0033·q0° − 0.0016·q1° (resid 0.053 A).
  Implemented as `hold_ff_*` in calibration.json + `--hold-ff SCALE`
  (`scripts/fit_hold_ff.py`).
- **Hold-ff A/B** (`trace_shape-20260910-151351` on / `-151413` off, back to
  back, K25 vg0.01/0.05): box RMS **0.84 mm vs 1.10 mm**, max 2.5 vs 3.0.
  Side_1 (the q0~88 deg side) 1.80 → 0.73 mm; side_2 got worse 0.68 → 1.41,
  i.e. the linear-in-(q0,q1) model over-corrects there — a 2-D map would do
  better. On 5 mm steps at test pose the ff is a wash (hold current ~0
  there): +y ss 0.56 → 0.51, overshoot 2.5 → 3.8 (`151303`/`151317`).
- **Cap ladder with per-joint gains** (`151436` K50 1.5 A, `151450` K50 2 A,
  `151504` K100 2 A, all +y, hold-ff 1): K50 relay-cycles at 3.4 Hz at BOTH
  caps (os 12.8 / 13.8 mm, I2t 9.6 / 14.7 A2s), K100 at 2 A stalls. **Raising
  the cap does not raise the K ceiling** — the earlier "linear band scales
  with cap" reading is falsified; the K25 ceiling is a damping limit.
- **sysid cogging mode rebuilt** (handoff from the UI session): bounded ramp
  loop (K_j 0.5 N·m/rad, per-joint vg 0.01/0.05, vel_limit 50, 5°/s, pre-ramp
  to the sweep start), linear Iq(q) detrend reported as `torsion_a_per_rad`,
  sweeps with >5 % of samples at the cap are rejected. First bounded runs
  (`sysid-20260910-152520/152533`) still pinned 36–68 % because the old code
  stepped the target by the full span; with the pre-ramp
  (`sysid-20260910-152738` shoulder, `-152801` elbow): shoulder 0 % pinned,
  **no visible cogging** (ripple < 0.02 A over 68–88°), Iq(q) slope −20 mA/deg
  going + and −8 mA/deg going − (≈0.25 A hysteresis band = harness friction);
  elbow flat and clean from −118° to −92° then **bang-bang chatter from −92°
  to −77° in both directions** (8 % pinned) — a pose-dependent elbow
  instability not seen in the step tests (those sit near q1 ≈ −104°). Plot
  `cogging_bounded_0910.png`.
- Elbow chatter region is elbow-vel_gain-dependent (`sysid-20260910-152918`
  vg 0.02: 0.1–0.3 % pinned; `-152801` vg 0.05: 8 %; `-152953` vg 0.1: 14 %).
  Box A/B with hold-ff (`trace_shape-20260910-153056` elbow 0.03: RMS 0.82,
  max 2.25; `-153117` elbow 0.05: 0.84 / 2.50). **Default preset is now
  `hover-K25-pj` = K25, vel_gain 0.01/0.03, vel_limit 50, max_pos_gain 1e5,
  0.8 A, hold_ff 1** (bare-default box `-153413`: RMS 0.82). `hold_ff` is a
  preset field; step_response/trace_shape/offset_sweep/point_hold/
  stiffness_bench default to it.
- **In-hand K ladder** (user at the handle, `point_hold-20260910-161033..161324`,
  guard off, 20 s holds, vg 0.01/0.03, hold_ff 1; the guarded attempt
  `161004..161018` tripped within 1–8 s on every point and is void): buzz
  band (>30 Hz tip motion) < 1 mm rms at every K/cap (K50→400: 0.20→0.44 mm
  @0.8 A, 0.39→0.73 mm @2 A); shoulder >30 Hz current 0.03→0.28 A rms;
  10–30 Hz band 0.3→1.7 mm @0.8 A. Force is cap-limited: 43–83 % of samples
  pinned at 0.8 A for K≥50, ~0 % at 2 A for K≥200. I2t per hold 6–12 A²s
  @0.8 A, 19–48 A²s @2 A; FETs 33/37 °C after. Conclusion: with a hand on the
  handle there is no velocity-loop buzz up to K400 at these gains; the
  hands-off 3 Hz relay swing is the only failure mode to guard.
  **Subjective feel (user):** 0.8 A — K50 good, not buzzy; K100 slightly more
  oscillation, fine when held; K200 oscillates a lot, must hold on, not
  noticeably stiffer than K100; K400 buzzy at idle, elbow tip oscillates
  (shoulder fine), disliked. 2 A — K50 nice; K100 clearly more holding power;
  K200 little extra, I2t high, shoulder gets hot; K400 buzzy at rest, not
  worth it. Verdict: K100/K200 @0.8 A handheld are fine; 2 A only at K≤100
  and briefly. Open: try elbow vel_gain 0.02 for the high-K elbow buzz.
  Cooling/rest pose recorded in calibration.json `cooling_pose`
  (77.5, 167.5) mm, q (107.5°, −84.9°). Thermal block set: i_continuous 0.8 A,
  budget 40 A²s.
- Reset paradigm: with no friction the arm drifts after `reset_pose` goes
  IDLE. `step_response --start-at-test-pose` now arms where the arm is, ramps
  the anchor to test_pose (1.5 s + 0.5 s hold), then steps.

## 2026-09-09 — hover rework, re-ID, chatter threshold

- Stiffness bench (`stiffness_bench-20260909-193350`, `-194714`): stability
  is pose-dependent; the 09-08 "converged K50–200" reference runs started at
  (100, 94) mm, not test pose. 2 A cap did not buy stiffness (K200 +y tripped
  at 2 A where 0.8 A held K400 at (108,100)).
- Re-ID after rework (`sysid-20260909-2231..2242`): shoulder breakaway
  0.106/0.007 A (was 0.68/0.47); elbow 0.057/0.004; Iq latency 2–3 ms;
  elbow J 0.0028 A·s²/rad clean; shoulder chirp amplitude-dependent and
  damper-like 5–30 Hz (J fit 0.012–0.023, not trustworthy). Cogging mode
  bang-bangs at the cap → its numbers are dither (`cogging_ramps_0909.png`).
- With friction gone, vg ≥ 0.1 gives a 41 Hz shoulder chatter at the cap
  (I2t ~2 A²s/run) independent of K and of encoder bw (150/200 worse than
  300). vg 0.02–0.05 chatter-free. Idle velocity-estimate noise σ:
  0.027/0.039 turn/s at bw 300, 0.009/0.012 at 150, 0.005/0.007 at 100.
  Plots `lin_1mm_0909.png`, `k25_vg02.png`, `jerr_K100_vg_up.png`.
- vel_limit clamp runs (`231526..231859`) looked clean partly because
  vl=1 caps the shoulder at 0.36 A (plateau); box with vl=0.5 → 14 mm RMS.
  Presets `hover-K100-vl1`, `hover-K25-vl1` carry that caveat.

## 2026-09-08 — estimator root cause

- Re-zero + breakaway after rework: `breakaway-20260908-152516/152617` (raw
  re-zero), low-friction confirmation shoulder CCW 0.37 A, CW 0.16, elbow
  0.10 A. Retune-from-bottom step_responses K10/25 straddle the pos_gain-bug
  fix boundary (~17:08 UTC): `step_response-20260908-165904..173256`
  (boundary not resolved to individual runs — see `experiments/run_notes.md`).
- Post-fix gain ladder, batch `20260908-203007..214956` (93 runs: step_response,
  impedance_step, point_hold, trace_shape): K=10 clean (os 0, ss 1.9 mm);
  K=25 rings 13.5 Hz at the cap regardless of vel_gain; vel_gain ≤0.03 rule;
  `point_hold-20260908-203034` tap test (no sustained structural mode);
  vel_limit A/B (clean at vl=1, limit-cycles at vl=5); hand-on-handle test
  clean K=25/50/100 at vg 0.03 (preset `hand-K100-vg03`, in-use only);
  first `trace_shape` box traces (preset `step-lin-K10` plateau-starved;
  vel_gain 0.1 traces but buzzes ±8 mm at 12 Hz).
- `encoder_bandwidth` 100 added 40–85° of velocity-estimate lag at 10–20 Hz →
  every limit cycle seen before. 300 saved on both drives. K50/100/200 steps
  converged (at pose (100,94)); box RMS 2.7–2.8 mm limited by shoulder
  friction 0.5–0.7 A (harness torsion, pose-dependent). Preset
  `pos-bw300-K100`, videos `box_K100_bw300.mp4`, `offset_K100_bw300.mp4`.
  Sysid suite batch `20260908-221334..232532` (107 runs): chirp FRFs
  `sysid-20260908-224420` (elbow), `-224509` (shoulder) identified the bw100
  lag as root cause; offset_sweep runs in this batch are the ones unaffected
  by the pos_gain bug. Reference-video runs `trace_shape-20260909-003643`
  and `offset_sweep-20260909-003732` (box RMS 2.7 mm; offsets ±15 mm,
  residual 2.1–4.1 mm) back the preset above.

## 2026-09-04 — bring-up

- disarm_reason=2 on node 0 is a cosmetic firmware quirk; RunLogger artifact
  fixed. pos_gain bug (config-default vel_gain → ~30× stated K) voided all
  step results 09-04 evening → 09-08 17:08 UTC. Torque-mode 4 mN·m plateau
  gotcha (`enable_torque_mode_vel_limit`). Bus raised to 20 V, hard max 2.5 A,
  session cap 2 A. Bring-up sweep batch `20260904-201012..211540` (31 runs:
  breakaway, offset_sweep, step_response, torque_step) — labelled K in this
  window is ~30× actual per the pos_gain bug above; the 21:00 UTC cap-sweep
  root-cause finding below falls inside it (qualitative shape only, not the
  K label). Evening-close retest `step_response-20260904-215823..220514`
  (post velocity-scheduled-cap fix, still inside the void-K window).
  Isolated `torque_step-20260905-042752..043142` on 09-05 is uncovered by
  any note (see `experiments/run_notes.md`).
