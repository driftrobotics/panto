#!/usr/bin/env python3
"""Record a video clip from a RealSense D4xx color stream.

Standalone tool, meant to run under the box's general-purpose `~/.venv`
(same interpreter as `sibling_project/src/sibling_project/realsense/capture_frame.py`).
Color stream only -- no depth.

## Importing our pyrealsense2 build

Same mechanism as `capture_frame.py` (see that file's docstring for the
full story): Intel doesn't publish arm64 wheels, so pyrealsense2 is built
from a librealsense source checkout. This script checks
`LIBREALSENSE_PYTHON_DIR`, then a couple of default build locations, before
giving up.

## Encoder selection

Prefers `cv2.VideoWriter` (mp4v fourcc) if `cv2` is importable in the
running interpreter. If not (e.g. no opencv wheel installed), frames are
written as JPEGs to a temp directory and muxed into an mp4 with `ffmpeg`
at the end -- so at least one of `cv2` or `ffmpeg` must be available.

The ffmpeg path records the wall-clock capture time of every frame and
muxes with real per-frame durations (an ffmpeg concat-demuxer list, `-vsync
vfr`) rather than assuming the nominal `--fps` was actually achieved. Under
CPU contention `wait_for_frames` can come in far slower than the requested
rate; muxing at the nominal fps would make the resulting mp4 play back
faster than it was recorded and end early. With real per-frame durations,
the mp4's duration matches wall-clock capture duration regardless of
achieved fps. (The `cv2.VideoWriter` path is fixed-rate by construction and
doesn't have this concat option -- not exercised on this box since no `cv2`
build is installed here.)

## Overlay

By default burns a top-left overlay onto every frame: wall-clock UTC
timestamp with milliseconds, and elapsed seconds since the recording
started. This lets a human line up the video against panto's
`samples.jsonl` `ts` field without needing exact frame-rate bookkeeping.
Disable with `--no-overlay`.

## Stopping a background recording

`--duration 0` (the default) records until SIGINT or SIGTERM. Both are
handled the same way: stop pulling frames, flush/close the writer (or run
ffmpeg over the JPEG dump), print the result, exit 0.

    nohup ~/.venv/bin/python3 record_cam.py --serial 111111111111 \\
        --out /tmp/run1.mp4 --duration 0 > /tmp/run1.log 2>&1 &
    echo $!                     # save the PID
    ...
    kill -TERM $PID             # stop cleanly; check /tmp/run1.log for the result
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _locate_pyrealsense2() -> None:
    """Best-effort: add a source-built pyrealsense2 to sys.path if it's not already importable."""
    try:
        import pyrealsense2  # noqa: F401

        return
    except ImportError:
        pass

    candidates = []
    env_dir = os.environ.get("LIBREALSENSE_PYTHON_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates += [
        Path.home() / "code/librealsense/build_native/release",
        Path.home() / "code/librealsense/build/release",
    ]
    for candidate in candidates:
        if list(candidate.glob("pyrealsense2*.so")):
            sys.path.insert(0, str(candidate))
            break

    try:
        import pyrealsense2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyrealsense2 not importable. It has no arm64 PyPI wheel -- build it from a "
            "librealsense checkout and either set LIBREALSENSE_PYTHON_DIR to the build's "
            "release/ dir, or add a .pth file pointing at it to this interpreter's "
            "site-packages. See sibling_project's capture_frame.py module docstring and the repo "
            "README's 'RealSense (on-robot)' section."
        ) from e


_locate_pyrealsense2()

import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2

    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default=None, help="Camera serial number. Omit to auto-pick the only connected device.")
    parser.add_argument("--out", required=True, help="Output .mp4 path.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record. 0 = until SIGINT/SIGTERM.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--timeout-ms", type=int, default=5000, help="Per-frame wait_for_frames timeout.")
    parser.add_argument(
        "--warmup-timeout-ms",
        type=int,
        default=5000,
        help="How long to wait for the first frame before treating the pipeline as cold-started "
             "and retrying (see _start_pipeline_with_warmup).",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Burn UTC timestamp (ms) + elapsed seconds into the top-left of each frame.",
    )
    return parser


def resolve_serial(serial_number: str | None) -> str:
    devices = rs.context().query_devices()
    connected = [d.get_info(rs.camera_info.serial_number) for d in devices]
    if not connected:
        raise RuntimeError("no RealSense device connected")
    if serial_number is not None:
        if serial_number not in connected:
            raise RuntimeError(f"no device with serial {serial_number!r} found (connected: {connected})")
        return serial_number
    if len(connected) > 1:
        raise RuntimeError(f"multiple devices connected, pass --serial to pick one: {connected}")
    return connected[0]


class _StopFlag:
    """Set on SIGINT/SIGTERM; checked in the capture loop."""

    def __init__(self) -> None:
        self.stop = False

    def handler(self, signum, frame) -> None:  # noqa: ARG002
        self.stop = True


def _overlay_frame(image: np.ndarray, start_wall: float) -> np.ndarray:
    """Burn a UTC timestamp (ms) + elapsed-seconds line into the top-left corner. RGB in/out."""
    now = time.time()
    ts_str = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    elapsed_str = f"+{now - start_wall:7.3f}s"
    text = f"{ts_str}  {elapsed_str}"

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Black outline + white fill for legibility against any background.
    x, y = 4, 2
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 0), font=font)
    return np.asarray(img)


def _build_config(serial: str, args: argparse.Namespace, *, with_depth: bool) -> "rs.config":
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    if with_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    return config


def _start_pipeline_with_warmup(
    serial: str, args: argparse.Namespace, start_wall: float
) -> tuple["rs.pipeline", float]:
    """Start the pipeline and wait for a first frame, retrying on a cold-start stall.

    2026-09-04: on this box a D405 (serial 111111111111) sitting idle for a while (e.g.
    ~20 min) sometimes accepts `pipeline.start()` but then delivers zero frames for tens of
    seconds -- `pipeline.start()` itself doesn't fail, so the old code just hung in
    `wait_for_frames` until the caller's outer `--timeout-ms`/`--duration` gave up. Adding a
    depth stream alongside color reliably "woke" the sensor in manual testing, so: try
    color-only first (the configuration actually wanted), and if no frame arrives within
    `--warmup-timeout-ms`, tear down and retry with color+depth enabled. Up to 3 attempts
    total before giving up. Returns the pipeline (already past its first frame, primed to
    keep streaming) and the wall-clock time that first frame arrived at.
    """
    max_attempts = 3
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        with_depth = attempt > 1  # attempt 1: color-only (as configured); retries: +depth to wake the sensor
        config = _build_config(serial, args, with_depth=with_depth)
        pipeline = rs.pipeline()
        attempt_start = time.monotonic()
        try:
            pipeline.start(config)
        except Exception as e:
            print(f"[record_cam] attempt {attempt}/{max_attempts} pipeline.start() FAILED "
                  f"(with_depth={with_depth}): {e!r}", file=sys.stderr, flush=True)
            last_exc = e
            continue
        try:
            frames = pipeline.wait_for_frames(timeout_ms=args.warmup_timeout_ms)
            if not frames.get_color_frame():
                raise RuntimeError("first frameset had no color frame")
        except Exception as e:
            elapsed = time.monotonic() - attempt_start
            print(f"[record_cam] attempt {attempt}/{max_attempts} no frame within "
                  f"{args.warmup_timeout_ms}ms (with_depth={with_depth}, waited {elapsed:.2f}s): {e!r}",
                  file=sys.stderr, flush=True)
            pipeline.stop()
            last_exc = e
            continue
        first_frame_wall = time.time()
        print(f"[record_cam] attempt {attempt}/{max_attempts} succeeded "
              f"(with_depth={with_depth}); first frame at +{first_frame_wall - start_wall:.2f}s", flush=True)
        return pipeline, first_frame_wall

    print(f"[record_cam] pipeline failed to deliver a frame after {max_attempts} attempts", file=sys.stderr, flush=True)
    raise SystemExit(1) from last_exc


def record(args: argparse.Namespace) -> tuple[Path, int, float, float]:
    serial = resolve_serial(args.serial)
    start_wall = time.time()

    pipeline, _first_frame_wall = _start_pipeline_with_warmup(serial, args, start_wall)

    stopflag = _StopFlag()
    signal.signal(signal.SIGINT, stopflag.handler)
    signal.signal(signal.SIGTERM, stopflag.handler)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    start_mono = time.monotonic()
    deadline = start_mono + args.duration if args.duration > 0 else None

    writer = None
    jpeg_dir: Path | None = None
    frame_times: list[float] = []  # monotonic capture time of each frame, ffmpeg path only

    try:
        if HAVE_CV2:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
            if not writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
        else:
            jpeg_dir = Path(tempfile.mkdtemp(prefix="record_cam_"))

        print(f"[record_cam] serial={serial}: recording to {out_path} "
              f"(encoder={'cv2' if HAVE_CV2 else 'ffmpeg/jpeg'}, overlay={args.overlay})", flush=True)

        # Poll wait_for_frames with a short timeout (not --timeout-ms directly) so the
        # SIGINT/SIGTERM handler gets checked promptly -- a stop signal that lands while
        # blocked inside a 5s wait_for_frames call used to take up to 5s to be honoured
        # (2026-09-04: measured ~13s worst case with queued frames). --timeout-ms is still
        # honoured as the real "camera hung" threshold, just via a running total across
        # short polls rather than a single blocking call.
        POLL_TIMEOUT_MS = 100
        no_frame_since = time.monotonic()
        while not stopflag.stop:
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                frames = pipeline.wait_for_frames(timeout_ms=POLL_TIMEOUT_MS)
                no_frame_since = time.monotonic()
            except RuntimeError:
                # Short poll timeout -- normal at 100ms granularity, not necessarily a
                # camera hiccup. Only treat it as a real timeout (and stop) once no frame
                # has arrived for the full --timeout-ms.
                if (time.monotonic() - no_frame_since) * 1000.0 >= args.timeout_ms:
                    print(f"[record_cam] no frame for {args.timeout_ms}ms -- stopping", flush=True)
                    break
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())  # RGB
            if args.overlay:
                image = _overlay_frame(image, start_wall)

            if HAVE_CV2:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            else:
                assert jpeg_dir is not None
                Image.fromarray(image).save(jpeg_dir / f"frame_{frame_count:08d}.jpg", quality=90)
                frame_times.append(time.monotonic())

            frame_count += 1
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()

    end_mono = time.monotonic()
    elapsed = end_mono - start_mono
    achieved_fps = frame_count / elapsed if elapsed > 0 else 0.0

    if jpeg_dir is not None:
        # end_mono (capture-loop exit, i.e. deadline/signal time) is the true end of the
        # recorded interval -- more accurate than the last frame's own timestamp, since
        # frames can arrive well before the loop actually stops.
        _mux_with_ffmpeg(jpeg_dir, out_path, frame_times, start_mono, end_mono)
        shutil.rmtree(jpeg_dir, ignore_errors=True)

    wall_duration = elapsed
    # The mp4's own clock starts at the first captured frame, not at process start -- pipeline
    # start-up/first-frame latency (device init, USB negotiation) happens before any frame
    # exists to encode. Compare against that same window, not the full process elapsed time,
    # or a perfectly correct mux looks like it's "missing" the start-up seconds.
    capture_start = frame_times[0] if frame_times else start_mono
    capture_duration = end_mono - capture_start
    mp4_duration = _ffprobe_duration(out_path)
    if mp4_duration is not None:
        rel_err = abs(mp4_duration - capture_duration) / capture_duration if capture_duration > 0 else 0.0
        print(f"[record_cam] wall_duration={wall_duration:.3f}s capture_duration={capture_duration:.3f}s "
              f"mp4_duration={mp4_duration:.3f}s rel_err={rel_err * 100:.2f}%")
        assert rel_err <= 0.02, (
            f"mp4 duration ({mp4_duration:.3f}s) diverges from actual capture-window duration "
            f"({capture_duration:.3f}s) by {rel_err * 100:.2f}% (>2%)"
        )
    else:
        print(f"[record_cam] wall_duration={wall_duration:.3f}s mp4_duration=<ffprobe unavailable>")

    return out_path, frame_count, achieved_fps, wall_duration


def _mux_with_ffmpeg(
    jpeg_dir: Path,
    out_path: Path,
    frame_times: list[float],
    start_mono: float,
    end_mono: float,
) -> None:
    """Mux JPEG frames into an mp4 using each frame's real wall-clock duration.

    Under CPU contention, frames can arrive far slower than the requested --fps; muxing at
    the nominal fps would make playback faster than the actual capture and the mp4 would be
    much shorter than the real recording. Instead we build an ffmpeg concat-demuxer list with
    an explicit `duration` per frame (time until the *next* frame, or until end_mono for the
    last one) and encode with `-vsync vfr` so the container duration matches wall-clock time.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"cv2 is unavailable and ffmpeg is not on PATH -- cannot encode. "
            f"Raw JPEG frames were left in {jpeg_dir}."
        )
    if not frame_times:
        raise RuntimeError("no frames captured -- nothing to mux")

    boundaries = frame_times + [end_mono]
    list_path = jpeg_dir / "list.txt"
    with open(list_path, "w") as f:
        for i, t in enumerate(frame_times):
            dt = max(boundaries[i + 1] - t, 1e-3)
            f.write(f"file 'frame_{i:08d}.jpg'\n")
            f.write(f"duration {dt:.6f}\n")
        # The concat demuxer ignores the `duration` on the final entry unless followed by
        # another `file` line, so repeat the last frame per the documented workaround.
        f.write(f"file 'frame_{len(frame_times) - 1:08d}.jpg'\n")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-vsync", "vfr",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _ffprobe_duration(path: Path) -> float | None:
    """Container `format=duration` is unreliable for the VFR mp4s this script
    produces (2026-09-04: measured 4.08s container duration on a clip whose
    frames actually run to pts=4.84s -- the concat-demuxer mux holds the last
    JPEG frame for ~0.8s via its `duration` directive, which the mp4 muxer's
    metadata does not reflect, even though the frame's own presentation
    timestamp is correct). The last frame's presentation timestamp is the
    reliable number -- ask for that directly instead."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v",
             "-show_entries", "frame=best_effort_timestamp_time",
             "-of", "csv=p=0", str(path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ).stdout.strip().splitlines()
        if out:
            return float(out[-1])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass
    # fall back to the (less reliable) container-level duration tag
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path, frame_count, achieved_fps, wall_duration = record(args)
    print(f"[record_cam] wrote {out_path} frames={frame_count} achieved_fps={achieved_fps:.2f} "
          f"wall_duration={wall_duration:.2f}s")


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
