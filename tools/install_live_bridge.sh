#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
EXAMPLE="$NS3_DIR/contrib/nr/examples/cttc-nr-demo-sionna-rt.cc"

HEADER_SRC="$PROJECT_DIR/tools/sdt_live_anim_bridge.h"
HEADER_DST="$NS3_DIR/contrib/nr/examples/sdt_live_anim_bridge.h"

if [[ ! -f "$EXAMPLE" ]]; then
    echo "ERROR: ns-3 example not found:"
    echo "  $EXAMPLE"
    exit 1
fi

cp "$HEADER_SRC" "$HEADER_DST"

python3 - "$EXAMPLE" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
t = p.read_text()

header = '#include "sdt_live_anim_bridge.h"'

# Add the header before the existing includes end.
if header not in t:
    marker = '#include "ns3/'
    idx = t.find(marker)

    if idx == -1:
        raise SystemExit("Could not find ns-3 include section.")

    # Insert immediately before the first ns-3 include.
    t = t[:idx] + header + "\n" + t[idx:]

needle = """    AnimationInterface anim(
        outputDir + "/sdt-topology.xml"
    );
"""

replacement = """    AnimationInterface anim(
        outputDir + "/sdt-topology.xml"
    );

    SdtLiveAnimBridge::Configure(
        anim,
        outputDir + "/sdt-live-events.jsonl"
    );
"""

if "SdtLiveAnimBridge::Configure" not in t:
    if needle not in t:
        raise SystemExit(
            "Could not find the existing AnimationInterface block."
        )
    t = t.replace(needle, replacement, 1)

p.write_text(t)
print("Live AnimationInterface bridge installed.")
PY

echo
echo "Live bridge installation complete."
