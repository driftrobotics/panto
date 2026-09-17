# CAN link on rig-host

The two ODrive Micros (node 0 shoulder, node 1 elbow) hang off the **ODrive
USB-CAN adapter** (gs_usb / candleLight family, USB id `1d50:606f`, serial
`AAAAAAAAAAAAAAAAAAAAAAAA`). A second, identical-looking adapter (serial
`BBBBBBBBBBBBBBBB`, "CANable 2.5") belongs to a sibling project's rig and is
named `can_yam` — do not use it for panto.

Naming and bring-up are fully automatic via systemd-networkd:

- `/etc/systemd/network/10-can-odrive.link` matches `Driver=gs_usb` +
  `Property=ID_SERIAL_SHORT=AAAAAAAAAAAAAAAAAAAAAAAA` and names it `can_odrive`
  (tx queue 1000).
- `/etc/systemd/network/10-can-odrive.network` sets `BitRate=1M`,
  `SamplePoint=87.5%` and brings it UP on plug-in.

`config.template.json` (`can.channel`) defaults to `can_odrive`; the onboard
Tegra `can0..can3` are unused by panto.

## Health check

```bash
ip -details -brief link show type can          # expect: can_odrive UP
networkctl status can_odrive                   # bitrate 1 M, state routable/carrier
timeout 1 candump -n 10 can_odrive             # heartbeats from 0x001/0x021 even when idle
journalctl -k --since -30min | grep -i gs_usb  # "usb xmit fail" / "USB disconnect" = cable/hub
```

`scripts/observe.py` is the passive, non-commutating check from the panto side
(prints node status + pose at `--hz`).

## If `can_odrive` does not exist

The adapter is not enumerated on USB — this is a cable/hub/power problem, not
a config one. `journalctl -k` will show `usb 1-2.x: USB disconnect` for it.
Re-seat the adapter's USB cable on the hub (it lived on hub port `1-2.4`, next
to `can_yam` on `1-2.1`); on re-enumeration the link is named and brought up
within a second or two. Check with the health commands above. Never rename
or re-bitrate by hand; if you must, `sudo networkctl reconfigure can_odrive`
re-applies the .network file.

## If the link is present but silent

- Bus power: the drives need the 20 V bus to emit heartbeats.
- Bus-off / error state: `ip -details link show can_odrive` shows
  `state ERROR-ACTIVE` when healthy; `BUS-OFF` needs
  `sudo ip link set can_odrive down && sudo ip link set can_odrive up`
  (or `networkctl reconfigure`).
- Termination: the ODrive Micro has an onboard 120 Ω termination switch;
  both ends of the trunk must be terminated once (adapter end + far drive).
- Nothing panto-side should be running while `odrivetool` (via
  `uv run --with odrive==0.6.11.post1 --offline python /tmp/odrive_*.py`) has
  the bus; check `pgrep -af "python -m scripts"` first.

## What we changed on the drives (saved, survives power cycle)

`encoder_bandwidth` 300, CAN broadcast rates encoder/Iq 2 ms and error/bus
20 ms, `current_hard_max` 2.5 A, `dc_max_positive_current` 3 A, OV 23 V /
UV 16 V for the 20 V bus, `enable_overspeed_error` off. `vel_gain` on the
drives is pushed per run by the scripts (and restored to the config value,
0.01 / 0.03, at exit).
