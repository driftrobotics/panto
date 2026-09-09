#!/usr/bin/env python3
"""Read colour frames from the D435(I) RealSense camera and write raw rgb24
frames to stdout, for piping into ffmpeg.

Binds to the D435 by serial number so it never opens the D405 workspace
camera, which is already in use by other tooling.

Usage:
    LIBREALSENSE_PYTHON_DIR=... python3 d435_publish.py [--serial SERIAL]
        [--width 640] [--height 480] [--fps 30]

Output: raw rgb24 frames (width*height*3 bytes each) on stdout, back to back,
suitable for `ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r FPS -i -`.
"""
from __future__ import annotations

import argparse
import os
import sys

# Mirror capture_frame.py's approach to locating the on-robot pyrealsense2
# build (no arm64 wheels from Intel; built from a librealsense checkout).
_LRS_DIR = os.environ.get(
    "LIBREALSENSE_PYTHON_DIR",
    os.path.expanduser("~/code/librealsense/build_native/release"),
)
if _LRS_DIR and _LRS_DIR not in sys.path:
    sys.path.insert(0, _LRS_DIR)

import pyrealsense2 as rs  # noqa: E402

DEFAULT_D435_SERIAL = "222222222222"  # D435I -- do NOT default to the D405 (111111111111)


def find_device_serial(requested: str | None) -> str:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    names = [d.get_info(rs.camera_info.name) for d in devices]
    for name, serial in zip(names, serials):
        print(f"found device: {name} serial={serial}", file=sys.stderr)

    target = requested or DEFAULT_D435_SERIAL
    if target not in serials:
        raise SystemExit(
            f"D435 serial {target!r} not found among connected devices: {serials}"
        )
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None, help="RealSense serial number (default: D435I)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    serial = find_device_serial(args.serial)
    print(f"binding to serial {serial}", file=sys.stderr)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    pipeline.start(config)

    out = sys.stdout.buffer
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            out.write(bytes(bytearray(color.get_data())))
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
