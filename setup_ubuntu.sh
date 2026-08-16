#!/usr/bin/env bash
# Sets up this project on Ubuntu (20.04/22.04/24.04). Safe to re-run.
#
# Usage:
#   chmod +x setup_ubuntu.sh
#   ./setup_ubuntu.sh            # core deps only (mock backends, fully offline)
#   ./setup_ubuntu.sh --full     # also installs torch+transformers for real CLIP/LLM backends
set -euo pipefail

echo "== sdt-llm-6g setup =="

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Installing..."
    sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
fi
echo "python3: $(python3 --version)"

# A venv is optional but recommended; system-wide install (--break-system-packages) also works.
if [ "${USE_VENV:-1}" = "1" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtualenv at .venv ..."
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PIP="pip"
else
    PIP="pip3 install --break-system-packages"
fi

echo "Installing core requirements..."
if [ "${USE_VENV:-1}" = "1" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    pip3 install --break-system-packages --upgrade pip
    pip3 install --break-system-packages -r requirements.txt
fi

if [ "${1:-}" = "--full" ]; then
    echo "Installing optional heavy deps (torch + transformers) for real CLIP/LLM backends..."
    echo "(On a GPU-less machine, consider the CPU-only torch index — see requirements-full.txt)"
    if [ "${USE_VENV:-1}" = "1" ]; then
        pip install -r requirements-full.txt
    else
        pip3 install --break-system-packages -r requirements-full.txt
    fi
fi

echo ""
echo "== Running the test suite =="
PYTHONPATH=src python3 -m pytest tests/ -v

echo ""
echo "== Setup complete =="
echo "Try:"
echo "  PYTHONPATH=src python3 scripts/run_vision_sdt_llm_demo.py"
echo "  PYTHONPATH=src python3 scripts/run_radio_sdt_llm_demo.py     # <- your 6G idea"
echo "  PYTHONPATH=src python3 scripts/run_fused_sdt_llm_demo.py"
