#!/usr/bin/env python3
"""
Cisco-style live visualizer for the current ns-3 + 5G-LENA + Sionna RT demo.

Data sources:
  1. AnimationInterface JSONL events -> wired/core packet events
  2. 5G-LENA RxPacketTraceUe JSONL events -> real NR DL reception events
  3. Existing sdt_network_trace.csv -> live network metrics when available

The GUI is a visualization layer. ns-3 remains authoritative.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
NS3_DIR = Path(os.environ.get("NS3_DIR", str(Path.home() / "ns-3-dev")))
EVENT_FILE = Path(
    os.environ.get("SDT_LIVE_EVENTS", str(NS3_DIR / "sdt-live-events.jsonl"))
)
NETWORK_TRACE = Path(
    os.environ.get("SDT_NETWORK_TRACE", str(NS3_DIR / "sdt_network_trace.csv"))
)

NS3_PROGRAM = os.environ.get("SDT_NS3_PROGRAM", "cttc-nr-demo-sionna-rt")

NODE_NAMES = {
    0: "gNB-1",
    1: "UE-1",
    2: "UE-2",
    3: "PGW",
    4: "SGW",
    5: "MME",
    6: "Remote Host",
}

NODE_LAYOUT = {
    0: (0.50, 0.78, "gNB-1", "5G-LENA + Sionna RT"),
    1: (0.23, 0.57, "UE-1", "RNTI=1"),
    2: (0.77, 0.57, "UE-2", "RNTI=2"),
    4: (0.27, 0.30, "SGW", "Serving Gateway"),
    3: (0.50, 0.30, "PGW", "Packet Gateway"),
    5: (0.73, 0.30, "MME", "Control Plane"),
    6: (0.50, 0.10, "Remote Host", "External data network"),
}

LINKS = [
    (0, 1, "NR Uu"),
    (0, 2, "NR Uu"),
    (0, 4, "S1-U"),
    (4, 3, "S5"),
    (4, 5, "S11"),
    (3, 6, "SGi"),
]

# Main user-plane forwarding path.
# The MME/S11 remains visible as control-plane context but is
# not treated as a UDP user-data forwarding hop.
USER_PLANE_LINKS = {
    frozenset((6, 3)),  # Remote Host <-> PGW
    frozenset((3, 4)),  # PGW <-> SGW
    frozenset((4, 0)),  # SGW <-> gNB
    frozenset((0, 1)),  # gNB <-> UE-1
    frozenset((0, 2)),  # gNB <-> UE-2
}

BG = QColor("#0f172a")
PANEL = QColor("#111827")
PANEL_2 = QColor("#172033")
TEXT = QColor("#e5e7eb")
MUTED = QColor("#94a3b8")
LINK = QColor("#475569")
LINK_HI = QColor("#64748b")
WIRE_PACKET = QColor("#f59e0b")
NR_PACKET = QColor("#22d3ee")
GOOD = QColor("#22c55e")
BAD = QColor("#ef4444")
GNBC = QColor("#3b82f6")
UEC = QColor("#16a34a")
COREC = QColor("#64748b")
HOSTC = QColor("#8b5cf6")
MMEC = QColor("#7c3aed")


@dataclass
class Packet:
    from_id: int
    to_id: int
    kind: str
    label: str
    sim_time: float
    start_wall: float
    duration_wall: float
    details: dict


class TopologyCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 620)
        self.sim_time = 0.0
        self.packets: list[Packet] = []
        self.selected_node: Optional[int] = None
        self.selected_packet: Optional[Packet] = None
        self.last_node_state: dict[int, dict] = {}
        self.current_route_text = (
            "USER PLANE: Remote Host → PGW → SGW → gNB-1 → UE"
        )
        self.event_count = 0
        self.nr_event_count = 0
        self.wired_event_count = 0
        self.paused_packets = False
        self.speed_display = "0.05x"

        # link -> wall-clock time until which the link is highlighted
        self.active_links: dict[frozenset[int], float] = {}
        self.current_hop: frozenset[int] | None = None
        self.current_hop_until = 0.0
        self.focus_flow_dest: int | None = None

        self.tick = QTimer(self)
        self.tick.timeout.connect(self._tick)
        self.tick.start(30)

    def _tick(self) -> None:
        now = time.monotonic()
        keep = []
        for packet in self.packets:
            if self.paused_packets:
                keep.append(packet)
                continue
            if (now - packet.start_wall) <= packet.duration_wall + 0.3:
                keep.append(packet)
        self.packets = keep[-150:]

        if self.current_hop is not None and now >= self.current_hop_until:
            self.current_hop = None

        cutoff = now
        self.active_links = {
            key: until
            for key, until in self.active_links.items()
            if until > cutoff
        }

        self.update()

    def node_point(self, node_id: int) -> tuple[float, float]:
        x, y, _, _ = NODE_LAYOUT[node_id]
        return x * self.width(), y * self.height()

    def _node_color(self, node_id: int) -> QColor:
        if node_id == 0:
            return GNBC
        if node_id == 1:
            return UEC
        if node_id == 2:
            return BAD if self.last_node_state.get(2, {}).get("degraded") else UEC
        if node_id == 6:
            return HOSTC
        if node_id == 5:
            return MMEC
        return COREC

    def set_current_hop(
        self,
        from_id: int,
        to_id: int,
        seconds: float = 0.9,
    ) -> None:
        key = frozenset((from_id, to_id))
        if key not in USER_PLANE_LINKS:
            return

        self.current_hop = key
        self.current_hop_until = time.monotonic() + seconds
        self.active_links[key] = self.current_hop_until

    def highlight_link(self, from_id: int, to_id: int, seconds: float = 0.65) -> None:
        key = frozenset((from_id, to_id))
        if key not in USER_PLANE_LINKS:
            return
        self.active_links[key] = time.monotonic() + seconds

    def add_packet(
        self,
        from_id: int,
        to_id: int,
        kind: str,
        label: str,
        sim_time: float,
        details: dict,
    ) -> None:
        if from_id not in NODE_LAYOUT or to_id not in NODE_LAYOUT:
            return

        self.packets.append(
            Packet(
                from_id=from_id,
                to_id=to_id,
                kind=kind,
                label=label,
                sim_time=sim_time,
                start_wall=time.monotonic(),
                duration_wall=0.9 if kind == "NR" else 0.75,
                details=details,
            )
        )

    def _link_for(self, a: int, b: int) -> str:
        for x, y, label in LINKS:
            if (x == a and y == b) or (x == b and y == a):
                return label
        return "LINK"

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BG)

        title = QFont()
        title.setBold(True)
        title.setPointSize(14)
        painter.setPen(QPen(TEXT))
        painter.setFont(title)
        painter.drawText(
            QRectF(0, 12, self.width(), 28),
            Qt.AlignCenter,
            "LIVE 5G-LENA + Sionna RT NETWORK",
        )

        sub = QFont()
        sub.setPointSize(8)
        painter.setFont(sub)
        painter.setPen(QPen(MUTED))
        painter.drawText(
            QRectF(0, 36, self.width(), 20),
            Qt.AlignCenter,
            "Actual ns-3 AnimationInterface + 5G-LENA NR events",
        )

        # Links
        for a, b, label in LINKS:
            ax, ay = self.node_point(a)
            bx, by = self.node_point(b)
            key = frozenset((a, b))
            active_event = key in self.active_links

            flow_focused = self.focus_flow_dest in (1, 2)

            active_packet = any(
                (
                    p.from_id == a and p.to_id == b
                )
                or
                (
                    p.from_id == b and p.to_id == a
                )
                for p in self.packets
            )

            active = active_event or active_packet

            if key == self.current_hop:
                pen_color = (
                    NR_PACKET
                    if key in {
                        frozenset((0, 1)),
                        frozenset((0, 2)),
                    }
                    else WIRE_PACKET
                )
                pen = QPen(pen_color, 8)
            elif key in USER_PLANE_LINKS and active_event:
                pen_color = (
                    NR_PACKET
                    if key in {
                        frozenset((0, 1)),
                        frozenset((0, 2)),
                    }
                    else WIRE_PACKET
                )
                pen = QPen(pen_color, 5)
            else:
                if flow_focused and key not in USER_PLANE_LINKS:
                    pen = QPen(QColor("#1f2937"), 1)
                elif flow_focused and not active:
                    pen = QPen(QColor("#334155"), 2)
                else:
                    pen = QPen(LINK_HI if active else LINK, 3 if active else 2)
            painter.setPen(pen)
            painter.drawLine(ax, ay, bx, by)

            mx, my = (ax + bx) / 2, (ay + by) / 2
            label_bg = QColor(PANEL_2)
            painter.setBrush(QBrush(label_bg))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(mx - 35, my - 12, 70, 22), 6, 6)

            painter.setPen(QPen(TEXT if label != "S11" else QColor("#c4b5fd")))
            painter.setFont(sub)
            painter.drawText(
                QRectF(mx - 35, my - 10, 70, 18),
                Qt.AlignCenter,
                label,
            )

        # Packet trails
        now = time.monotonic()
        for packet in self.packets:
            ax, ay = self.node_point(packet.from_id)
            bx, by = self.node_point(packet.to_id)
            progress = 1.0 if self.paused_packets else min(
                1.0,
                max(0.0, (now - packet.start_wall) / packet.duration_wall),
            )

            color = NR_PACKET if packet.kind == "NR" else WIRE_PACKET

            for i in range(6, 0, -1):
                trail_p = max(0.0, progress - i * 0.028)
                tx = ax + (bx - ax) * trail_p
                ty = ay + (by - ay) * trail_p
                radius = max(2.0, 8.0 - i)
                alpha = max(25, 160 - i * 22)
                tc = QColor(color.red(), color.green(), color.blue(), alpha)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(tc))
                painter.drawEllipse(QRectF(tx-radius, ty-radius, radius*2, radius*2))

            x = ax + (bx - ax) * progress
            y = ay + (by - ay) * progress
            painter.setPen(QPen(QColor("#111827"), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(x-10, y-10, 20, 20))

            packet_font = QFont()
            packet_font.setBold(True)
            packet_font.setPointSize(7)
            painter.setFont(packet_font)
            painter.setPen(QPen(color))
            painter.drawText(QRectF(x + 13, y - 11, 110, 18), Qt.AlignLeft, packet.label)

        # Nodes
        for node_id, (nx, ny, title_text, subtitle_text) in NODE_LAYOUT.items():
            x, y = self.node_point(node_id)
            w = 180 if node_id == 0 else 160
            h = 66
            rect = QRectF(x - w/2, y - h/2, w, h)

            selected = node_id == self.selected_node
            pen = QPen(QColor("#f8fafc") if selected else QColor("#334155"), 3 if selected else 1.5)
            painter.setPen(pen)
            painter.setBrush(QBrush(self._node_color(node_id)))
            painter.drawRoundedRect(rect, 12, 12)

            tf = QFont()
            tf.setBold(True)
            tf.setPointSize(10)
            painter.setFont(tf)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(
                QRectF(rect.left(), rect.top()+9, rect.width(), 20),
                Qt.AlignCenter,
                title_text,
            )

            sf = QFont()
            sf.setPointSize(8)
            painter.setFont(sf)
            painter.drawText(
                QRectF(rect.left(), rect.top()+33, rect.width(), 18),
                Qt.AlignCenter,
                subtitle_text,
            )

            # compact status badge for UEs
            if node_id in (1, 2) and node_id in self.last_node_state:
                state = self.last_node_state[node_id]
                text = (
                    f"SINR {state.get('sinr_db', float('nan')):.2f} dB"
                    if math.isfinite(state.get("sinr_db", float("nan")))
                    else "SINR N/A"
                )
                bad = state.get("degraded", False)
                painter.setBrush(QBrush(QColor("#7f1d1d") if bad else QColor("#14532d")))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(
                    QRectF(rect.left()+10, rect.bottom()-19, rect.width()-20, 16),
                    5, 5,
                )
                painter.setPen(QPen(TEXT))
                painter.drawText(
                    QRectF(rect.left()+10, rect.bottom()-18, rect.width()-20, 14),
                    Qt.AlignCenter,
                    text,
                )

        # Current user-plane route.
        route_font = QFont()
        route_font.setBold(True)
        route_font.setPointSize(9)
        painter.setFont(route_font)
        painter.setPen(QPen(QColor("#cbd5e1")))

        route_text = getattr(
            self,
            "current_route_text",
            "USER PLANE: Remote Host → PGW → SGW → gNB-1 → UE",
        )

        painter.drawText(
            QRectF(18, 60, self.width() - 36, 24),
            Qt.AlignLeft,
            route_text,
        )

        # footer legend
        painter.setPen(QPen(MUTED))
        painter.setFont(sf)
        painter.drawText(
            QRectF(18, self.height()-28, 160, 18),
            Qt.AlignLeft,
            "● Core packet",
        )
        painter.setPen(QPen(NR_PACKET))
        painter.drawText(
            QRectF(155, self.height()-28, 180, 18),
            Qt.AlignLeft,
            "● NR-Uu packet",
        )

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        closest = None
        best = 1e9

        for node_id in NODE_LAYOUT:
            x, y = self.node_point(node_id)
            d = (pos.x()-x)**2 + (pos.y()-y)**2
            if d < best and d < 110**2:
                best = d
                closest = node_id

        if closest is not None:
            self.selected_node = closest
            self.selected_packet = None
            self.parent().node_selected.emit(closest) if hasattr(self.parent(), "node_selected") else None
            self.update()
            return

        # select latest packet if click is near a marker
        now = time.monotonic()
        for packet in reversed(self.packets):
            ax, ay = self.node_point(packet.from_id)
            bx, by = self.node_point(packet.to_id)
            progress = min(1.0, max(0.0, (now - packet.start_wall) / packet.duration_wall))
            x = ax + (bx - ax) * progress
            y = ay + (by - ay) * progress
            if (pos.x()-x)**2 + (pos.y()-y)**2 < 14**2:
                self.selected_packet = packet
                self.selected_node = None
                if hasattr(self.parent(), "packet_selected"):
                    self.parent().packet_selected.emit(packet)
                self.update()
                return


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SDT Live — ns-3 + 5G-LENA + Sionna RT")
        self.resize(1560, 960)

        self.event_offset = 0
        self.event_buffer = ""
        self.trace_offset = 0
        self.ns3_process: Optional[subprocess.Popen] = None

        self.total_events = 0
        self.nr_events = 0
        self.wired_events = 0
        self.rx_packets = 0
        self.tx_packets = 0
        self.last_metrics: dict[str, str] = {}
        self.selected_node = None
        self.selected_flow_dest = None
        self.selected_flow_name = "All traffic"

        self.replay_events = []
        self.replay_index = 0
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self.replay_next_hop)

        # Latest observed user-plane forwarding events.
        # This is an observation summary, not a claim that the simulator
        # exposes an end-to-end packet ID across all layers.
        self.recent_user_plane_hops: dict[frozenset[int], float] = {}

        self.flow_hop_sequence = {
            1: [
                (6, 3, "SGi"),
                (3, 4, "S5"),
                (4, 0, "S1-U"),
                (0, 1, "NR Uu"),
            ],
            2: [
                (6, 3, "SGi"),
                (3, 4, "S5"),
                (4, 0, "S1-U"),
                (0, 2, "NR Uu"),
            ],
        }

        self.last_hop_index = {
            1: -1,
            2: -1,
        }

        self.canvas = TopologyCanvas()
        self.canvas.parent = lambda: self
        self.canvas.mousePressEvent = self._canvas_mouse_event

        # Header
        self.sim_label = QLabel("SIM 0.000000 s")
        self.sim_label.setObjectName("metric")
        self.status_label = QLabel("IDLE")
        self.status_label.setObjectName("status")

        self.run_btn = QPushButton("▶ Run")
        self.rerun_btn = QPushButton("↻ Re-run")
        self.stop_btn = QPushButton("■ Stop")
        self.replay_btn = QPushButton("▶ Replay trace")
        self.pause_btn = QPushButton("⏸ Pause packets")
        self.clear_btn = QPushButton("Clear")

        self.speed_box = QComboBox()
        self.speed_box.addItems(["0.02x", "0.05x", "0.1x", "0.25x", "0.5x", "1.0x"])
        self.speed_box.setCurrentText(os.environ.get("SDT_LIVE_SPEED", "0.05x"))

        self.flow_box = QComboBox()
        self.flow_box.addItems([
            "All traffic",
            "Flow 1 → UE-1",
            "Flow 2 → UE-2",
        ])
        self.flow_box.currentTextChanged.connect(self.change_flow)

        self.run_btn.clicked.connect(self.start_simulation)
        self.rerun_btn.clicked.connect(self.rerun_simulation)
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.replay_btn.clicked.connect(self.start_replay)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.clear_btn.clicked.connect(self.clear_view)

        header = QHBoxLayout()
        header.addWidget(QLabel("SDT LIVE 5G/6G"))
        header.addSpacing(12)
        header.addWidget(self.sim_label)
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(QLabel("Flow"))
        header.addWidget(self.flow_box)
        header.addWidget(QLabel("Speed"))
        header.addWidget(self.speed_box)
        header.addWidget(self.run_btn)
        header.addWidget(self.rerun_btn)
        header.addWidget(self.replay_btn)
        header.addWidget(self.pause_btn)
        header.addWidget(self.stop_btn)
        header.addWidget(self.clear_btn)

        # Right-side detail cards
        self.selected_title = QLabel("Nothing selected")
        self.selected_title.setObjectName("detailTitle")
        self.selected_details = QPlainTextEdit()
        self.selected_details.setReadOnly(True)

        self.packet_title = QLabel("Packet inspector")
        self.packet_details = QPlainTextEdit()
        self.packet_details.setReadOnly(True)

        self.metrics = QPlainTextEdit()
        self.metrics.setReadOnly(True)

        self.flow_trace = QPlainTextEdit()
        self.flow_trace.setReadOnly(True)

        self.events = QTableWidget(0, 5)
        self.events.setHorizontalHeaderLabels(
            ["Sim time", "From", "To", "Type", "Info"]
        )
        self.events.horizontalHeader().setStretchLastSection(True)

        # Right panel
        right = QVBoxLayout()
        right.addWidget(self.selected_title)
        right.addWidget(self.selected_details, 2)
        right.addWidget(self.packet_title)
        right.addWidget(self.packet_details, 2)
        right.addWidget(QLabel("Live metrics"))
        right.addWidget(self.metrics, 2)
        right.addWidget(QLabel("Selected flow trace"))
        right.addWidget(self.flow_trace, 2)
        right.addWidget(QLabel("Recent packet events"))
        right.addWidget(self.events, 3)

        right_widget = QWidget()
        right_widget.setLayout(right)

        # Main splitter
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addLayout(header)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        self.poll = QTimer(self)
        self.poll.timeout.connect(self.poll_streams)
        self.poll.start(40)

        self.network_poll = QTimer(self)
        self.network_poll.timeout.connect(self.poll_network_trace)
        self.network_poll.start(200)

        self.apply_style()
        self.update_metrics_panel()

    def apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { background: #0f172a; color: #e5e7eb; }
            QLabel { color: #e5e7eb; }
            QPushButton {
                background: #1e293b;
                border: 1px solid #334155;
                padding: 7px 10px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #334155; }
            QComboBox, QTableWidget, QPlainTextEdit {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QHeaderView::section {
                background: #172033;
                color: #cbd5e1;
                border: 0px;
                padding: 5px;
            }
            QLabel#status {
                color: #22c55e;
                font-weight: bold;
                padding: 5px 10px;
            }
            QLabel#metric {
                color: #67e8f9;
                font-weight: bold;
            }
            QLabel#detailTitle {
                color: #93c5fd;
                font-size: 12pt;
                font-weight: bold;
            }
        """)

    def ns3_pids(self) -> list[int]:
        import subprocess

        result = subprocess.run(
            ["pgrep", "-f", NS3_PROGRAM],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return []

        pids = []
        for line in result.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue

            # Never count this GUI process itself.
            if pid != os.getpid():
                pids.append(pid)

        return sorted(set(pids))

    def change_flow(self, text: str) -> None:
        self.selected_flow_name = text

        if text == "Flow 1 → UE-1":
            self.selected_flow_dest = 1
        elif text == "Flow 2 → UE-2":
            self.selected_flow_dest = 2
        else:
            self.selected_flow_dest = None

        self.canvas.focus_flow_dest = self.selected_flow_dest
        self.canvas.current_route_text = self.route_text()
        self.update_flow_trace()


    def route_text(self) -> str:
        if self.selected_flow_dest == 1:
            return "FLOW 1: Remote Host → PGW → SGW → gNB-1 → UE-1"
        if self.selected_flow_dest == 2:
            return "FLOW 2: Remote Host → PGW → SGW → gNB-1 → UE-2"
        return "USER PLANE: Remote Host → PGW → SGW → gNB-1 → UE"


    def flow_hop_allowed(self, from_id: int, to_id: int) -> bool:
        if self.selected_flow_dest is None:
            return True

        shared = {
            frozenset((6, 3)),  # Remote Host ↔ PGW
            frozenset((3, 4)),  # PGW ↔ SGW
            frozenset((4, 0)),  # SGW ↔ gNB
        }

        final = {
            frozenset((0, self.selected_flow_dest))
        }

        return (
            frozenset((from_id, to_id)) in shared
            or frozenset((from_id, to_id)) in final
        )


    def update_flow_trace(self) -> None:
        if self.selected_flow_dest == 1:
            ue = "UE-1"
            flow = "Flow 1"
        elif self.selected_flow_dest == 2:
            ue = "UE-2"
            flow = "Flow 2"
        else:
            self.flow_trace.setPlainText(
                "All traffic selected.\n\n"
                "Select Flow 1 or Flow 2 to focus the user-plane route."
            )
            return

        lines = [
            f"{flow} selected",
            "",
            "Observed route:",
            "Remote Host",
            "    ↓ SGi",
            "   PGW",
            "    ↓ S5",
            "   SGW",
            "    ↓ S1-U",
            "   gNB-1",
            "    ↓ NR Uu",
            f"   {ue}",
            "",
            "Important:",
            "The core events are shared by both flows and do not carry",
            "a persistent end-to-end packet ID in the current trace.",
            "The final NR hop is identified by RNTI/UE.",
        ]

        self.flow_trace.setPlainText("\n".join(lines))


    def build_replay_hops(self):
        """
        Build a human-readable replay of the observed user-plane route.

        This is a route playback, not end-to-end packet identity tracking.
        The current backend does not expose one persistent packet ID across
        core and NR events.
        """
        if self.selected_flow_dest not in (1, 2):
            return []

        ue = self.selected_flow_dest

        return [
            {
                "from": 6,
                "to": 3,
                "label": "SGi",
                "kind": "CORE",
                "title": "Remote Host → PGW",
            },
            {
                "from": 3,
                "to": 4,
                "label": "S5",
                "kind": "CORE",
                "title": "PGW → SGW",
            },
            {
                "from": 4,
                "to": 0,
                "label": "S1-U",
                "kind": "CORE",
                "title": "SGW → gNB-1",
            },
            {
                "from": 0,
                "to": ue,
                "label": "NR Uu",
                "kind": "NR",
                "title": f"gNB-1 → UE-{ue}",
            },
        ]


    def start_replay(self) -> None:
        if self.selected_flow_dest not in (1, 2):
            QMessageBox.information(
                self,
                "Select a flow",
                "Choose Flow 1 → UE-1 or Flow 2 → UE-2 first.",
            )
            return

        self.replay_events = self.build_replay_hops()

        if not self.replay_events:
            return

        self.replay_index = 0
        self.canvas.packets.clear()
        self.canvas.current_hop = None
        self.canvas.current_hop_until = 0.0

        self.status_label.setText(
            f"REPLAY: {self.selected_flow_name}"
        )

        self.flow_trace.setPlainText(
            f"{self.selected_flow_name}\n\n"
            "Replay mode: observed user-plane route\n\n"
            "The highlighted hop is the current route segment."
        )

        self.replay_timer.start(1100)
        self.replay_next_hop()


    def replay_next_hop(self) -> None:
        if self.replay_index >= len(self.replay_events):
            self.replay_timer.stop()
            self.status_label.setText("REPLAY COMPLETE")
            self.flow_trace.appendPlainText(
                "\n\n✓ Route replay complete"
            )
            return

        hop = self.replay_events[self.replay_index]
        sim_time = (
            self.canvas.sim_time
            if self.canvas.sim_time > 0
            else 0.5
        )

        self.canvas.set_current_hop(
            hop["from"],
            hop["to"],
            seconds=1.15,
        )

        self.canvas.add_packet(
            hop["from"],
            hop["to"],
            hop["kind"],
            hop["label"],
            sim_time,
            {
                "Replay": "Observed route playback",
                "Hop": hop["title"],
                "Interface": hop["label"],
                "Flow": self.selected_flow_name,
            },
        )

        self.flow_trace.setPlainText(
            f"{self.selected_flow_name}\n\n"
            f"ACTIVE HOP\n"
            f"{hop['title']}\n\n"
            f"Interface: {hop['label']}\n"
            f"Type: {hop['kind']}\n\n"
            "Full observed route:\n"
            "Remote Host\n"
            "    ↓ SGi\n"
            "   PGW\n"
            "    ↓ S5\n"
            "   SGW\n"
            "    ↓ S1-U\n"
            "   gNB-1\n"
            f"    ↓ NR Uu\n"
            f"   UE-{self.selected_flow_dest}"
        )

        self.replay_index += 1

    def start_simulation(self) -> None:
        self.replay_timer.stop()

        if self.ns3_process and self.ns3_process.poll() is None:
            self.status_label.setText("RUNNING")
            return

        existing = self.ns3_pids()
        if existing:
            self.status_label.setText("ALREADY RUNNING")
            QMessageBox.warning(
                self,
                "ns-3 already running",
                "A cttc-nr-demo-sionna-rt process is already running. "
                "Stop that simulation before starting another one.\n\n"
                f"PIDs: {existing}"
            )
            return

        try:
            self.prepare_run()

            env = os.environ.copy()
            env["SDT_LIVE_DEMO"] = "1"
            env["SDT_LIVE_SPEED"] = self.speed_box.currentText().replace("x", "")
            env["SDT_LIVE_EVENTS"] = str(EVENT_FILE)
            env["NS3_DIR"] = str(NS3_DIR)

            self.ns3_process = subprocess.Popen(
                ["./ns3", "run", NS3_PROGRAM],
                cwd=str(NS3_DIR),
                env=env,
            )

            self.status_label.setText("STARTING")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Unable to start ns-3",
                str(exc),
            )

    def rerun_simulation(self) -> None:
        self.stop_simulation()

        # Wait briefly for the OS process to disappear.
        deadline = time.monotonic() + 3.0
        while self.ns3_pids() and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.05)

        existing = self.ns3_pids()
        if existing:
            QMessageBox.critical(
                self,
                "Could not re-run safely",
                "An old ns-3 process is still running. "
                "No new simulation was started.\n\n"
                f"PIDs: {existing}"
            )
            self.status_label.setText("RE-RUN BLOCKED")
            return

        self.prepare_run()
        self.start_simulation()

    def prepare_run(self) -> None:
        try:
            EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
            EVENT_FILE.write_text("")
        except OSError as exc:
            raise RuntimeError(f"Cannot reset {EVENT_FILE}: {exc}") from exc

        self.event_offset = 0
        self.event_buffer = ""
        self.trace_offset = 0
        self.clear_view()

    def stop_simulation(self) -> None:
        self.replay_timer.stop()

        if self.ns3_process and self.ns3_process.poll() is None:
            try:
                self.ns3_process.terminate()
                self.ns3_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ns3_process.kill()
            except Exception:
                pass

        self.ns3_process = None
        self.status_label.setText("STOPPED")

    def toggle_pause(self) -> None:
        self.canvas.paused_packets = not self.canvas.paused_packets
        self.pause_btn.setText(
            "▶ Resume packets"
            if self.canvas.paused_packets
            else "⏸ Pause packets"
        )

    def clear_view(self) -> None:
        self.canvas.packets.clear()
        self.canvas.active_links.clear()
        self.canvas.current_route_text = self.route_text()
        self.recent_user_plane_hops.clear()
        self.last_hop_index = {1: -1, 2: -1}
        self.canvas.current_hop = None
        self.canvas.current_hop_until = 0.0
        self.replay_timer.stop()
        self.replay_events = []
        self.replay_index = 0
        self.update_flow_trace()
        self.canvas.selected_node = None
        self.canvas.selected_packet = None
        self.selected_details.setPlainText("")
        self.packet_details.setPlainText("")
        self.events.setRowCount(0)
        self.total_events = 0
        self.nr_events = 0
        self.wired_events = 0
        self.canvas.event_count = 0
        self.canvas.nr_event_count = 0
        self.canvas.wired_event_count = 0
        self.update_metrics_panel()

    def _canvas_mouse_event(self, event) -> None:
        pos = event.position()
        chosen = None
        best = 1e12

        for node_id in NODE_LAYOUT:
            x, y = self.canvas.node_point(node_id)
            d = (pos.x()-x)**2 + (pos.y()-y)**2
            if d < best and d < 110**2:
                best = d
                chosen = node_id

        if chosen is not None:
            self.canvas.selected_node = chosen
            self.canvas.selected_packet = None
            self.show_node(chosen)
            self.packet_details.clear()
            self.canvas.update()
            return

        now = time.monotonic()
        for packet in reversed(self.canvas.packets):
            ax, ay = self.canvas.node_point(packet.from_id)
            bx, by = self.canvas.node_point(packet.to_id)
            progress = min(
                1.0,
                max(0.0, (now - packet.start_wall) / packet.duration_wall),
            )
            x = ax + (bx-ax)*progress
            y = ay + (by-ay)*progress
            if (pos.x()-x)**2 + (pos.y()-y)**2 < 16**2:
                self.canvas.selected_packet = packet
                self.canvas.selected_node = None
                self.show_packet(packet)
                self.selected_details.clear()
                self.canvas.update()
                return

    def show_node(self, node_id: int) -> None:
        name = NODE_NAMES[node_id]
        self.selected_title.setText(name)

        state = self.canvas.last_node_state.get(node_id, {})
        lines = [f"Node ID: {node_id}", f"Role: {NODE_LAYOUT[node_id][3]}"]

        if node_id in (1, 2):
            lines += [
                f"RNTI: {state.get('rnti', 'N/A')}",
                (
                    f"SINR: {state.get('sinr_db', float('nan')):.2f} dB"
                    if math.isfinite(state.get("sinr_db", float("nan")))
                    else "SINR: N/A"
                ),
                f"Last TB: {state.get('tb_size', 'N/A')} B",
                f"Cell: {state.get('cell_id', 'N/A')}",
                f"BWP: {state.get('bwp_id', 'N/A')}",
                f"Corrupt: {state.get('corrupt', 'N/A')}",
                f"State: {'DEGRADED' if state.get('degraded') else 'NORMAL'}",
            ]

        self.selected_details.setPlainText("\n".join(lines))

    def show_packet(self, packet: Packet) -> None:
        self.packet_title.setText("Packet inspector")
        info = packet.details.copy()
        info.update({
            "from": NODE_NAMES.get(packet.from_id, str(packet.from_id)),
            "to": NODE_NAMES.get(packet.to_id, str(packet.to_id)),
            "type": packet.kind,
            "sim_time": f"{packet.sim_time:.9f} s",
        })
        self.packet_details.setPlainText(
            "\n".join(f"{k}: {v}" for k, v in info.items())
        )

    def add_event_row(self, sim_time: float, from_name: str, to_name: str, kind: str, info: str) -> None:
        self.events.insertRow(0)
        vals = [f"{sim_time:.6f}", from_name, to_name, kind, info]
        for col, value in enumerate(vals):
            self.events.setItem(0, col, QTableWidgetItem(value))
        while self.events.rowCount() > 100:
            self.events.removeRow(self.events.rowCount() - 1)

    def update_user_plane_route(self, sim_time: float, destination: int | None = None) -> None:
        now = time.monotonic()
        recent = {
            key: stamp
            for key, stamp in self.recent_user_plane_hops.items()
            if sim_time - stamp <= 0.02
        }
        self.recent_user_plane_hops = recent

        if self.selected_flow_dest == 1:
            route = "FLOW 1: Remote Host → PGW → SGW → gNB-1 → UE-1"
        elif self.selected_flow_dest == 2:
            route = "FLOW 2: Remote Host → PGW → SGW → gNB-1 → UE-2"
        elif destination == 1:
            route = "USER PLANE: Remote Host → PGW → SGW → gNB-1 → UE-1"
        elif destination == 2:
            route = "USER PLANE: Remote Host → PGW → SGW → gNB-1 → UE-2"
        else:
            route = "USER PLANE: Remote Host → PGW → SGW → gNB-1 → UE"

        self.canvas.current_route_text = route

    def advance_flow_hop(
        self,
        from_id: int,
        to_id: int,
        sim_time: float,
    ) -> None:
        destination = None

        if to_id == 1:
            destination = 1
        elif to_id == 2:
            destination = 2
        elif frozenset((from_id, to_id)) == frozenset((6, 3)):
            destination = self.selected_flow_dest
        elif frozenset((from_id, to_id)) == frozenset((3, 4)):
            destination = self.selected_flow_dest
        elif frozenset((from_id, to_id)) == frozenset((4, 0)):
            destination = self.selected_flow_dest

        if destination not in (1, 2):
            return

        if not self.flow_hop_allowed(from_id, to_id):
            return

        sequence = self.flow_hop_sequence[destination]
        key = frozenset((from_id, to_id))

        for index, (a, b, label) in enumerate(sequence):
            if frozenset((a, b)) == key:
                # Only move forward through the path.
                if index >= self.last_hop_index[destination]:
                    self.last_hop_index[destination] = index
                    self.canvas.set_current_hop(from_id, to_id)
                    self.flow_trace.setPlainText(
                        f"Flow {destination} → UE-{destination}\n\n"
                        f"ACTIVE HOP: {label}\n\n"
                        f"{NODE_NAMES[a]}  →  {NODE_NAMES[b]}\n\n"
                        f"Observed simulation time: {sim_time:.6f} s\n\n"
                        f"Path:\n"
                        + "\n".join(
                            [
                                "Remote Host",
                                "    ↓ SGi",
                                "   PGW",
                                "    ↓ S5",
                                "   SGW",
                                "    ↓ S1-U",
                                "   gNB-1",
                                "    ↓ NR Uu",
                                f"   UE-{destination}",
                            ]
                        )
                    )
                return

    def record_user_plane_hop(
        self,
        from_id: int,
        to_id: int,
        sim_time: float,
    ) -> None:
        key = frozenset((from_id, to_id))
        if key not in USER_PLANE_LINKS:
            return

        self.recent_user_plane_hops[key] = sim_time
        self.canvas.highlight_link(from_id, to_id)

    def handle_event(self, event: dict) -> None:
        typ = event.get("type")
        sim_time = float(event.get("sim_time", 0.0))
        self.canvas.sim_time = sim_time
        self.sim_label.setText(f"SIM {sim_time:.6f} s")

        if typ == "live_end":
            self.status_label.setText("COMPLETE")
            return

        if typ == "heartbeat":
            if self.ns3_process and self.ns3_process.poll() is None:
                self.status_label.setText("LIVE")
            return

        if typ == "nr_packet":
            self.nr_events += 1
            frm = int(event.get("from", 0))
            to = int(event.get("to", -1))
            rnti = int(event.get("rnti", 0))
            sinr_linear = float(event.get("sinr", -1))
            sinr_db = 10.0 * math.log10(sinr_linear) if sinr_linear > 0 else float("-inf")
            tb = int(event.get("tb_size", 0))
            corrupt = bool(event.get("corrupt", False))

            # Real 5G-LENA NR reception.
            self.record_user_plane_hop(frm, to, sim_time)
            self.update_user_plane_route(sim_time, to)

            if not self.flow_hop_allowed(frm, to):
                return

            state = self.canvas.last_node_state.setdefault(to, {})
            state.update({
                "rnti": rnti,
                "sinr_db": sinr_db,
                "tb_size": tb,
                "cell_id": event.get("cell_id"),
                "bwp_id": event.get("bwp_id"),
                "corrupt": corrupt,
                "degraded": corrupt or (math.isfinite(sinr_db) and sinr_db < 10),
            })

            self.canvas.add_packet(
                frm, to, "NR", f"NR DL RNTI={rnti}", sim_time,
                {
                    "RNTI": rnti,
                    "TB size": f"{tb} B",
                    "SINR": f"{sinr_db:.2f} dB" if math.isfinite(sinr_db) else "N/A",
                    "Cell": event.get("cell_id"),
                    "BWP": event.get("bwp_id"),
                    "Corrupt": corrupt,
                    "Direction": "DL",
                }
            )
            self.add_event_row(
                sim_time,
                NODE_NAMES.get(frm, str(frm)),
                NODE_NAMES.get(to, str(to)),
                "NR DL",
                f"RNTI={rnti}  TB={tb}B  SINR={sinr_db:.2f} dB",
            )
            self.total_events += 1
            self.canvas.nr_event_count = self.nr_events
            return

        if typ == "anim":
            xml_text = event.get("xml", "")
            if not xml_text.startswith("<"):
                return
            # AnimationInterface fragments have fId/tId plus fbTx/fbRx.
            import xml.etree.ElementTree as ET
            try:
                elem = ET.fromstring(xml_text)
            except ET.ParseError:
                return
            if elem.tag not in ("p", "wp"):
                return

            frm = int(elem.attrib.get("fId", -1))
            to = int(elem.attrib.get("tId", -1))
            if frm < 0 or to < 0 or frm == to:
                return

            fb_tx = float(elem.attrib.get("fbTx", sim_time))
            fb_rx = float(elem.attrib.get("fbRx", fb_tx))
            key = frozenset((frm, to))

            # Keep all real events in the event file, but make the main
            # Cisco-style topology emphasize only user-plane forwarding.
            if elem.tag == "p" and key not in USER_PLANE_LINKS:
                return

            if elem.tag == "p" and not self.flow_hop_allowed(frm, to):
                return

            kind = "NR/XML" if elem.tag == "wp" else "Core"

            self.record_user_plane_hop(frm, to, sim_time)
            self.advance_flow_hop(frm, to, sim_time)
            self.update_flow_trace()

            self.canvas.add_packet(
                frm, to, "NR" if elem.tag == "wp" else "CORE",
                "NR packet" if elem.tag == "wp" else "PACKET",
                sim_time,
                {
                    "Source": NODE_NAMES.get(frm, str(frm)),
                    "Destination": NODE_NAMES.get(to, str(to)),
                    "TX": f"{fb_tx:.9f} s",
                    "RX": f"{fb_rx:.9f} s",
                }
            )
            self.add_event_row(
                sim_time,
                NODE_NAMES.get(frm, str(frm)),
                NODE_NAMES.get(to, str(to)),
                kind,
                f"TX={fb_tx:.6f}  RX={fb_rx:.6f}",
            )
            self.wired_events += 1
            self.total_events += 1
            self.canvas.wired_event_count = self.wired_events

    def poll_streams(self) -> None:
        if not EVENT_FILE.exists():
            self.status_label.setText("WAITING FOR ns-3")
            return

        try:
            with EVENT_FILE.open("r", encoding="utf-8") as handle:
                handle.seek(self.event_offset)
                chunk = handle.read()
                self.event_offset = handle.tell()
        except OSError:
            return

        if chunk:
            self.event_buffer += chunk

            while "\n" in self.event_buffer:
                line, self.event_buffer = self.event_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.handle_event(event)

        self.update_metrics_panel()

    def poll_network_trace(self) -> None:
        if not NETWORK_TRACE.exists():
            return

        try:
            with NETWORK_TRACE.open("r", encoding="utf-8", newline="") as handle:
                handle.seek(self.trace_offset)
                new_data = handle.read()
                self.trace_offset = handle.tell()
        except OSError:
            return

        if not new_data.strip():
            return

        # Network trace may be recreated between runs. Parse only complete lines.
        rows = [r for r in new_data.splitlines() if r.strip()]
        if not rows:
            return

        for row in rows[-20:]:
            parts = next(csv.reader([row]), [])
            if len(parts) >= 2:
                self.last_metrics["last_trace"] = " | ".join(parts[:8])

        self.update_metrics_panel()

    def update_metrics_panel(self) -> None:
        proc = "running" if self.ns3_process and self.ns3_process.poll() is None else "stopped"
        lines = [
            f"Simulation: {proc}",
            f"Total live events: {self.total_events}",
            f"NR events: {self.nr_events}",
            f"Animation events: {self.wired_events}",
            f"Visible packets: {len(self.canvas.packets)}",
            f"Event stream: {EVENT_FILE}",
        ]
        if self.last_metrics:
            lines.append("")
            lines.append("Latest network trace row:")
            lines.append(self.last_metrics.get("last_trace", "N/A"))

        self.metrics.setPlainText("\n".join(lines))

    def closeEvent(self, event) -> None:
        self.stop_simulation()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
