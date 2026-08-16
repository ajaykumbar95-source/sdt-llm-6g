#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
VENV="$NS3_DIR/sionna-env"

echo "=============================================="
echo "SDT-LLM + ns-3/Sionna RT setup"
echo "=============================================="

if [[ ! -d "$NS3_DIR" ]]; then
    echo "ERROR: ns-3 not found at $NS3_DIR"
    exit 1
fi

if [[ ! -f "$NS3_DIR/ns3" ]]; then
    echo "ERROR: ns-3 launcher not found"
    exit 1
fi

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: sionna-env not found at $VENV"
    exit 1
fi

source "$VENV/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src"

echo
echo "[1/4] Checking Python environment..."
python -c "import numpy; print('numpy OK')"
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"

echo
echo "[2/4] Installing ns-3 SDT integration..."
cd "$PROJECT_DIR"
./setup_ns3_integration.sh

echo
echo "[3/4] Checking ns-3 Sionna RT example..."
if [[ ! -f "$NS3_DIR/contrib/nr/examples/cttc-nr-demo-sionna-rt.cc" ]]; then
    echo "ERROR: Sionna RT ns-3 example not found"
    exit 1
fi

echo
echo "[4/4] Running Python regression tests..."
cd "$PROJECT_DIR"
PYTHONPATH=src python -m pytest tests/ -q

echo
echo "=============================================="
echo "SETUP OK"
echo "=============================================="
echo "Environment: $VENV"
echo "ns-3:        $NS3_DIR"
echo "Project:     $PROJECT_DIR"
echo
echo "Run the full pipeline with:"
echo "  ./run_pipeline.sh"
echo "=============================================="
