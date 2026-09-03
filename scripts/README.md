# Bootstrapping / calibration experiments

Milestone 3 in `HANDOFF.md`. Each script runs one experiment against hardware and
prints a result block to paste into the Notion spec (with firmware version + gain
config so runs stay comparable).

Planned:

| script | measures | sets |
|---|---|---|
| `latency_histogram.py` | host + CAN round-trip, p50/p95/max | `config.latency_compensation_s` |
| `sweep_pos_gain.py` | `pos_gain` at which the motor hisses / goes unstable | position-backend gain ceiling |
| `sweep_vel_gain.py` | `vel_gain` ceiling before instability | `MotorConfig.vel_gain` |
| `encoder_noise_floor.py` | velocity noise (rad/s RMS) at the control rate | stiffness ceiling estimate |
| `snap_line_stiffness.py` | subjective sweep of snap-to-line stiffness | v0 snap defaults |

Not yet written — see the predecessor `odrive_knob` for the measurement
techniques (its README "Tuning" and "Loop health" sections).
