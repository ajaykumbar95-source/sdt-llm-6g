# Live ns-3 / 5G-LENA / Sionna RT visualizer

This adds a live Cisco-like packet/topology view while preserving the normal
NetAnim XML trace.

The bridge uses ns-3 AnimationInterface's write callback. Each animation XML
fragment is mirrored into a JSONL event stream, which the PySide6 GUI tails.

## Install

cd ~/Downloads/sdt-llm-6g
source ~/ns-3-dev/sionna-env/bin/activate
python3 -m pip install -r tools/live_visualizer_requirements.txt

## Integrate with ns-3

./tools/install_live_bridge.sh

## Run

./tools/run_live_ns3_demo.sh

Default pacing:
SDT_LIVE_SPEED=1.0 ./tools/run_live_ns3_demo.sh

2x presentation speed:
SDT_LIVE_SPEED=2.0 ./tools/run_live_ns3_demo.sh

The GUI reads:
~/ns-3-dev/sdt-live-events.jsonl

The ordinary:
~/ns-3-dev/sdt-topology.xml
is still generated.

The first version targets the current 1-gNB / 2-UE / EPC scenario and displays
the actual AnimationInterface packet events as moving markers.
