#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
TARGET="$NS3_DIR/contrib/nr/examples/cttc-nr-demo-sionna-rt.cc"
SOURCE="$PROJECT_DIR/ns3-patch/cttc-nr-demo-sionna-rt.cc"

if [[ ! -d "$NS3_DIR" ]]; then
    echo "ERROR: ns-3 not found at $NS3_DIR"
    exit 1
fi

if [[ ! -f "$SOURCE" ]]; then
    echo "ERROR: saved ns-3 integration not found:"
    echo "$SOURCE"
    exit 1
fi

if [[ -f "$TARGET" ]]; then
    cp "$TARGET" "${TARGET}.before-sdt-llm"
fi

cp "$SOURCE" "$TARGET"

echo "ns-3 SDT integration installed:"
echo "  $TARGET"

cd "$NS3_DIR"
./ns3 build

echo "ns-3 + Sionna RT integration ready."
