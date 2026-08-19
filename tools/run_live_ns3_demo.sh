#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
VENV="$NS3_DIR/sionna-env"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src"

export SDT_LIVE_EVENTS="${SDT_LIVE_EVENTS:-$NS3_DIR/sdt-live-events.jsonl}"
export SDT_LIVE_DEMO=1
export SDT_LIVE_SPEED="${SDT_LIVE_SPEED:-1.0}"

rm -f "$SDT_LIVE_EVENTS"

python3 "$PROJECT_DIR/tools/live_ns3_visualizer.py" &
GUI_PID=$!

cleanup() {
    kill "$GUI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$NS3_DIR"
./ns3 run "cttc-nr-demo-sionna-rt"

wait "$GUI_PID" 2>/dev/null || true
