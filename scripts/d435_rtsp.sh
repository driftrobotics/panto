#!/usr/bin/env bash
# Start/stop/status an RTSP stream of the D435(I) RealSense colour camera.
#
# Architecture: mediamtx (RTSP server, port 8554) + a publisher pipeline that
# reads raw rgb24 frames from the D435 via pyrealsense2 (d435_publish.py) and
# pipes them into ffmpeg, which H.264-encodes (libx264, zerolatency/ultrafast
# -- this Jetson's ffmpeg build has no working hardware encoder path, see
# notes below) and pushes to mediamtx over local RTSP.
#
# Binds to the D435 by serial (222222222222) so it never touches the D405
# workspace camera (111111111111) already in use by other tooling.
#
# Usage:
#   d435_rtsp.sh start   # start mediamtx + publisher in the background (nohup)
#   d435_rtsp.sh stop    # stop both cleanly
#   d435_rtsp.sh status  # report whether running, PIDs, and stream URL
#   d435_rtsp.sh restart
#
# Stream URL:      rtsp://rig-host:8554/d435
# View it with:    ffplay -rtsp_transport tcp rtsp://rig-host:8554/d435
#              or: vlc rtsp://rig-host:8554/d435
#
# Logs:   ~/d435_rtsp.log
# PIDs:   ~/.d435_rtsp/mediamtx.pid, ~/.d435_rtsp/pipeline.pid

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
while [[ -L "$SELF" ]]; do
    target="$(readlink "$SELF")"
    if [[ "$target" != /* ]]; then
        target="$(dirname "$SELF")/$target"
    fi
    SELF="$target"
done
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
STATE_DIR="$HOME/.d435_rtsp"
LOG_FILE="$HOME/d435_rtsp.log"
MEDIAMTX_BIN="$HOME/bin/mediamtx"
MEDIAMTX_CONF="$SCRIPT_DIR/mediamtx_d435.yml"
PUBLISH_PY="$SCRIPT_DIR/d435_publish.py"
PYTHON_BIN="$HOME/.venv/bin/python3"
LIBREALSENSE_PYTHON_DIR="${LIBREALSENSE_PYTHON_DIR:-$HOME/code/librealsense/build_native/release}"

SERIAL="${D435_SERIAL:-222222222222}"
WIDTH="${D435_WIDTH:-640}"
HEIGHT="${D435_HEIGHT:-480}"
FPS="${D435_FPS:-30}"
RTSP_URL="rtsp://127.0.0.1:8554/d435"

MEDIAMTX_PID_FILE="$STATE_DIR/mediamtx.pid"
PIPELINE_PID_FILE="$STATE_DIR/pipeline.pid"

mkdir -p "$STATE_DIR"

is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

start() {
    if is_running "$MEDIAMTX_PID_FILE" && is_running "$PIPELINE_PID_FILE"; then
        echo "already running (mediamtx pid $(cat "$MEDIAMTX_PID_FILE"), pipeline pid $(cat "$PIPELINE_PID_FILE"))"
        exit 0
    fi

    echo "=== d435_rtsp start $(date -Is) ===" >>"$LOG_FILE"

    # mediamtx: RTSP server
    nohup "$MEDIAMTX_BIN" "$MEDIAMTX_CONF" >>"$LOG_FILE" 2>&1 &
    echo $! >"$MEDIAMTX_PID_FILE"
    disown

    # Give mediamtx a moment to bind its port before the publisher connects.
    sleep 1

    # publisher: pyrealsense2 -> rawvideo -> ffmpeg (libx264, zerolatency) -> RTSP
    nohup bash -c "
        LIBREALSENSE_PYTHON_DIR='$LIBREALSENSE_PYTHON_DIR' '$PYTHON_BIN' '$PUBLISH_PY' \
            --serial '$SERIAL' --width '$WIDTH' --height '$HEIGHT' --fps '$FPS' | \
        ffmpeg -hide_banner -loglevel warning \
            -f rawvideo -pix_fmt rgb24 -s '${WIDTH}x${HEIGHT}' -r '$FPS' -i - \
            -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
            -g $((FPS * 2)) -bf 0 \
            -f rtsp -rtsp_transport tcp '$RTSP_URL'
    " >>"$LOG_FILE" 2>&1 &
    echo $! >"$PIPELINE_PID_FILE"
    disown

    sleep 2
    if is_running "$MEDIAMTX_PID_FILE" && is_running "$PIPELINE_PID_FILE"; then
        echo "started."
        echo "  mediamtx pid: $(cat "$MEDIAMTX_PID_FILE")"
        echo "  pipeline pid: $(cat "$PIPELINE_PID_FILE")"
        echo "  stream url:   rtsp://rig-host:8554/d435"
        echo "  log:          $LOG_FILE"
    else
        echo "FAILED to start -- check $LOG_FILE"
        tail -30 "$LOG_FILE"
        exit 1
    fi
}

stop() {
    local stopped=0
    if is_running "$PIPELINE_PID_FILE"; then
        local pgid
        pgid="$(cat "$PIPELINE_PID_FILE")"
        # pipeline pid is the `bash -c "... | ffmpeg ..."` wrapper; kill its
        # process group so the publisher and ffmpeg both die.
        pkill -TERM -P "$pgid" 2>/dev/null || true
        kill -TERM "$pgid" 2>/dev/null || true
        stopped=1
    fi
    rm -f "$PIPELINE_PID_FILE"

    if is_running "$MEDIAMTX_PID_FILE"; then
        kill -TERM "$(cat "$MEDIAMTX_PID_FILE")" 2>/dev/null || true
        stopped=1
    fi
    rm -f "$MEDIAMTX_PID_FILE"

    # Best-effort cleanup of any stray publisher/ffmpeg processes from this script.
    pkill -f "$PUBLISH_PY" 2>/dev/null || true
    pkill -f "rtsp_transport tcp $RTSP_URL" 2>/dev/null || true

    if [[ "$stopped" -eq 1 ]]; then
        echo "stopped."
    else
        echo "was not running."
    fi
}

status() {
    local mm_up=0 pipe_up=0
    if is_running "$MEDIAMTX_PID_FILE"; then
        mm_up=1
        echo "mediamtx:  running (pid $(cat "$MEDIAMTX_PID_FILE"))"
    else
        echo "mediamtx:  not running"
    fi
    if is_running "$PIPELINE_PID_FILE"; then
        pipe_up=1
        echo "pipeline:  running (pid $(cat "$PIPELINE_PID_FILE"))"
    else
        echo "pipeline:  not running"
    fi
    if [[ "$mm_up" -eq 1 && "$pipe_up" -eq 1 ]]; then
        echo "stream:    rtsp://rig-host:8554/d435"
    fi
    echo "log:       $LOG_FILE"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) stop; sleep 1; start ;;
    *)
        echo "usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
