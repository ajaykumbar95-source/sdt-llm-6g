#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
VENV="$NS3_DIR/sionna-env"

source "$VENV/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src"

echo "=============================================="
echo "1. Running ns-3 + 5G-LENA + Sionna RT"
echo "=============================================="

cd "$NS3_DIR"
./ns3 run "cttc-nr-demo-sionna-rt"

echo
echo "=============================================="
echo "1B. Preparing NetAnim visualization"
echo "=============================================="

cd "$PROJECT_DIR"
PYTHONPATH=src python scripts/prepare_netanim_visual.py


echo
echo "=============================================="
echo "2. Running temporal SDT"
echo "=============================================="

cd "$PROJECT_DIR"
PYTHONPATH=src python scripts/test_ns3_radio_network.py

echo
echo "=============================================="
echo "3. Running SDT -> Qwen"
echo "=============================================="

PYTHONPATH=src python scripts/test_ns3_llm.py

echo
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="

echo
echo "=============================================="
echo "4. Generating research visualizations"
echo "=============================================="

PYTHONPATH=src python scripts/visualize_sdt_results.py

PYTHONPATH=src python scripts/visualize_research_topology.py
PYTHONPATH=src python scripts/visualize_sdt_heatmap.py
