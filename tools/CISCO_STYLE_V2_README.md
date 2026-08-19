# Cisco-style live ns-3 visualizer V2

This replaces the earlier prototype viewer with a topology-first desktop UI.

Features:
- Cisco-style network canvas
- gNB, UE, SGW, PGW, MME, Remote Host
- NR Uu / S1-U / S5 / S11 / SGi labels
- live wired/core packet animation
- live real 5G-LENA NR DL reception animation
- packet trails
- node selection
- packet inspector
- NR SINR converted to dB
- live event table
- Run / Re-run / Stop / Pause
- speed control
- no change to ns-3 simulation semantics

Run:
  cd ~/Downloads/sdt-llm-6g
  source ~/ns-3-dev/sionna-env/bin/activate
  python3 tools/live_ns3_visualizer_v2.py

The GUI can start the existing:
  ./ns3 run cttc-nr-demo-sionna-rt

Default event stream:
  ~/ns-3-dev/sdt-live-events.jsonl
