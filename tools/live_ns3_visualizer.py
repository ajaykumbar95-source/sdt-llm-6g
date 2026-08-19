#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


NS3_DIR = Path(os.environ.get("NS3_DIR", str(Path.home() / "ns-3-dev")))
EVENT_FILE = Path(
    os.environ.get(
        "SDT_LIVE_EVENTS",
        str(NS3_DIR / "sdt-live-events.jsonl"),
    )
)

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
    0: (0.50, 0.84, "gNB-1", "5G-LENA + Sionna RT"),
    1: (0.22, 0.67, "UE-1", "RNTI=1"),
    2: (0.78, 0.67, "UE-2", "RNTI=2"),
    4: (0.28, 0.39, "SGW", "Serving Gateway"),
    3: (0.50, 0.39, "PGW", "Packet Gateway"),
    5: (0.72, 0.39, "MME", "Mobility Management"),
    6: (0.50, 0.13, "Remote Host", "External data network"),
}

LINKS = [
    (0, 1, "NR Uu"),
    (0, 2, "NR Uu"),
    (0, 4, "S1-U"),
    (4, 3, "S5"),
    (4, 5, "S11"),
    (3, 6, "SGi"),
]

BG = QColor(248, 249, 251)
LINE = QColor(145, 150, 160)
PACKET = QColor(245, 160, 55)
GNBC = QColor(55, 120, 215)
UEC = QColor(55, 165, 95)
COREC = QColor(125, 130, 145)
HOSTC = QColor(145, 95, 190)


@dataclass
class Packet:
    from_id: int
    to_id: int
    fb_tx: float
    fb_rx: float
    created_wall: float

    @property
    def sim_duration(self) -> float:
        return max(self.fb_rx - self.fb_tx, 0.01)


class TopologyWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(760, 540)
        self.packets: list[Packet] = []
        self.node_descriptions = {
            node_id: (title, subtitle)
            for node_id, (_, _, title, subtitle) in NODE_LAYOUT.items()
        }
        self.packet_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(30)

    def point(self, node_id: int) -> tuple[float, float]:
        x, y, _, _ = NODE_LAYOUT[node_id]
        return x * self.width(), y * self.height()

    def add_packet(self, from_id: int, to_id: int, fb_tx: float, fb_rx: float) -> None:
        if from_id not in NODE_LAYOUT or to_id not in NODE_LAYOUT:
            return
        self.packets.append(
            Packet(
                from_id=from_id,
                to_id=to_id,
                fb_tx=fb_tx,
                fb_rx=fb_rx,
                created_wall=time.monotonic(),
            )
        )
        self.packet_count += 1

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BG)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(13)
        painter.setPen(QPen(QColor(25, 30, 40)))
        painter.setFont(title_font)
        painter.drawText(
            QRectF(0, 12, self.width(), 28),
            Qt.AlignCenter,
            "LIVE ns-3 + 5G-LENA + Sionna RT",
        )

        for a, b, label in LINKS:
            self._draw_link(painter, a, b, label)

        for node_id in NODE_LAYOUT:
            self._draw_node(painter, node_id)

        now = time.monotonic()
        keep = []
        for p in self.packets:
            # Presentation pacing: keep real ns-3 packet events visible
            # long enough for a human observer to follow the path.
            duration = max(1.0, p.sim_duration * 20.0)
            progress = min(
                1.0,
                (now - p.created_wall) / duration
            )
            if progress <= 1.15:
                keep.append(p)
            if progress < 1.0:
                self._draw_packet(painter, p, progress)
        # Keep a recent backlog so multiple packets are visible at once.
        self.packets = keep[-250:]

    def _draw_link(self, painter: QPainter, a: int, b: int, label: str) -> None:
        ax, ay = self.point(a)
        bx, by = self.point(b)
        painter.setPen(QPen(LINE, 2))
        painter.drawLine(ax, ay, bx, by)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(60, 65, 75)))
        painter.drawText(
            QRectF(mx - 40, my - 14, 80, 20),
            Qt.AlignCenter,
            label,
        )

    def _node_color(self, node_id: int) -> QColor:
        if node_id == 0:
            return GNBC
        if node_id in (1, 2):
            return UEC
        if node_id == 6:
            return HOSTC
        return COREC

    def _draw_node(self, painter: QPainter, node_id: int) -> None:
        x, y = self.point(node_id)
        title, subtitle = self.node_descriptions.get(
            node_id,
            (f"Node {node_id}", ""),
        )
        w, h = (180, 64) if node_id == 0 else (158, 60)
        rect = QRectF(x - w / 2, y - h / 2, w, h)

        painter.setPen(QPen(QColor(35, 40, 50), 1.5))
        painter.setBrush(QBrush(self._node_color(node_id)))
        painter.drawRoundedRect(rect, 10, 10)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        painter.setFont(title_font)
        painter.setPen(QPen(Qt.white))
        painter.drawText(
            QRectF(rect.left(), rect.top() + 8, rect.width(), 20),
            Qt.AlignCenter,
            title,
        )

        sub_font = QFont()
        sub_font.setPointSize(8)
        painter.setFont(sub_font)
        painter.drawText(
            QRectF(rect.left(), rect.top() + 31, rect.width(), 18),
            Qt.AlignCenter,
            subtitle,
        )

    def _draw_packet(self, painter: QPainter, packet: Packet, progress: float) -> None:
        ax, ay = self.point(packet.from_id)
        bx, by = self.point(packet.to_id)
        x = ax + (bx - ax) * progress
        y = ay + (by - ay) * progress

        # Visible packet trail.
        trail_steps = 5
        for i in range(trail_steps, 0, -1):
            trail_progress = max(0.0, progress - i * 0.025)
            tx = ax + (bx - ax) * trail_progress
            ty = ay + (by - ay) * trail_progress
            radius = max(2.0, 7.0 - i)
            alpha = max(35, 150 - i * 22)

            trail_color = QColor(
                PACKET.red(),
                PACKET.green(),
                PACKET.blue(),
                alpha,
            )

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(trail_color))
            painter.drawEllipse(
                QRectF(
                    tx - radius,
                    ty - radius,
                    radius * 2,
                    radius * 2,
                )
            )

        # Main packet marker.
        painter.setPen(QPen(QColor(115, 75, 20), 1))
        painter.setBrush(QBrush(PACKET))
        painter.drawEllipse(QRectF(x - 11, y - 11, 22, 22))

        packet_font = QFont()
        packet_font.setBold(True)
        packet_font.setPointSize(8)
        painter.setFont(packet_font)
        painter.setPen(QPen(QColor(80, 55, 20)))

        painter.drawText(
            QRectF(x + 14, y - 14, 120, 22),
            Qt.AlignLeft,
            "PACKET",
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ns-3 Live Packet Visualizer")
        self.resize(1380, 900)

        self.offset = 0
        self.buffer = ""
        self.total_events = 0

        self.topology = TopologyWidget()

        self.sim_label = QLabel("Simulation time: 0.000000 s")
        self.packet_label = QLabel("Packet events: 0")
        self.status_label = QLabel("Waiting for ns-3...")
        self.status_label.setStyleSheet("font-weight: bold;")

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Sim time", "From", "To", "Link", "TX", "RX"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        pause = QPushButton("Pause packets")
        pause.clicked.connect(self.toggle_packets)
        self.packet_paused = False

        clear = QPushButton("Clear log")
        clear.clicked.connect(self.log.clear)

        header = QHBoxLayout()
        header.addWidget(self.sim_label)
        header.addWidget(self.packet_label)
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(pause)
        header.addWidget(clear)

        left = QVBoxLayout()
        left.addLayout(header)
        left.addWidget(self.topology)

        left_widget = QWidget()
        left_widget.setLayout(left)

        right = QVBoxLayout()
        right.addWidget(QLabel("Live packet events"))
        right.addWidget(self.table, 3)
        right.addWidget(QLabel("Raw live events"))
        right.addWidget(self.log, 2)

        right_widget = QWidget()
        right_widget.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)

    def toggle_packets(self) -> None:
        self.packet_paused = not self.packet_paused

    def poll(self) -> None:
        if not EVENT_FILE.exists():
            self.status_label.setText(f"Waiting: {EVENT_FILE}")
            return

        self.status_label.setText("LIVE: receiving ns-3 events")

        try:
            with EVENT_FILE.open("r", encoding="utf-8") as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset = f.tell()
        except OSError:
            return

        if not chunk:
            return

        self.buffer += chunk

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.handle_event(event)

    def handle_event(self, event: dict) -> None:
        sim_time = float(event.get("sim_time", 0.0))
        self.sim_label.setText(
            f"Simulation time: {sim_time:.6f} s"
        )

        if event.get("type") == "nr_packet":
            self.handle_nr_packet(event)
            return

        if event.get("type") != "anim":
            if event.get("type") == "live_end":
                self.status_label.setText(
                    f"SIMULATION FINISHED @ {sim_time:.3f} s"
                )
            return

        xml_text = event.get("xml", "").strip()
        if not xml_text.startswith("<"):
            return

        try:
            element = ET.fromstring(xml_text)
        except ET.ParseError:
            return

        if element.tag not in ("p", "wp"):
            if element.tag == "nu":
                self.handle_node_update(element)
            return

        from_id = int(element.attrib.get("fId", -1))
        to_id = int(element.attrib.get("tId", -1))
        if from_id == to_id or from_id < 0 or to_id < 0:
            return

        fb_tx = float(element.attrib.get("fbTx", sim_time))
        fb_rx = float(element.attrib.get("fbRx", fb_tx))

        if not self.packet_paused:
            self.topology.add_packet(from_id, to_id, fb_tx, fb_rx)

        self.total_events += 1
        self.packet_label.setText(
            f"Packet events: {self.total_events}"
        )

        kind = "NR/wireless" if element.tag == "wp" else "wired"
        row = 0
        self.table.insertRow(row)

        vals = [
            f"{sim_time:.6f}",
            NODE_NAMES.get(from_id, f"Node {from_id}"),
            NODE_NAMES.get(to_id, f"Node {to_id}"),
            kind,
            f"{fb_tx:.6f}",
            f"{fb_rx:.6f}",
        ]
        for c, value in enumerate(vals):
            self.table.setItem(row, c, QTableWidgetItem(value))

        while self.table.rowCount() > 100:
            self.table.removeRow(self.table.rowCount() - 1)

        self.log.appendPlainText(
            f"t={sim_time:.6f}  "
            f"{NODE_NAMES.get(from_id, f'Node {from_id}')} -> "
            f"{NODE_NAMES.get(to_id, f'Node {to_id}')}  "
            f"{kind}"
        )

    def handle_node_update(self, element: ET.Element) -> None:
        node_id = element.attrib.get("id")
        if node_id is None:
            return
        node_id = int(node_id)

        if element.attrib.get("p") != "d":
            return

        descr = element.attrib.get("descr", "")
        parts = [part.strip() for part in descr.split("|", 1)]
        title = parts[0]
        subtitle = parts[1] if len(parts) > 1 else ""
        self.topology.node_descriptions[node_id] = (title, subtitle)


    def handle_nr_packet(self, event: dict) -> None:
        from_id = int(event.get("from", 0))
        to_id = int(event.get("to", -1))
        sim_time = float(event.get("sim_time", 0.0))
        rnti = int(event.get("rnti", 0))
        tb_size = int(event.get("tb_size", 0))
        sinr_linear = float(event.get("sinr", -1.0))

        if sinr_linear > 0.0:
            sinr_db = 10.0 * math.log10(sinr_linear)
        else:
            sinr_db = float("-inf")

        corrupt = bool(event.get("corrupt", False))

        self.sim_label.setText(
            f"Simulation time: {sim_time:.6f} s"
        )

        if not self.packet_paused:
            # The NR event is a real UE reception event. The displayed
            # travel animation is presentation pacing around that event.
            self.topology.add_packet(
                from_id,
                to_id,
                sim_time,
                sim_time + 0.05,
            )

        self.total_events += 1
        self.packet_label.setText(
            f"Packet events: {self.total_events}"
        )

        row = 0
        self.table.insertRow(row)

        vals = [
            f"{sim_time:.6f}",
            NODE_NAMES.get(from_id, f"Node {from_id}"),
            NODE_NAMES.get(to_id, f"Node {to_id}"),
            f"NR DL RNTI={rnti}",
            f"{tb_size} B",
            (
                f"SINR={sinr_db:.2f} dB"
                if math.isfinite(sinr_db)
                else "SINR=N/A"
            ),
        ]

        for c, value in enumerate(vals):
            self.table.setItem(
                row,
                c,
                QTableWidgetItem(value),
            )

        while self.table.rowCount() > 100:
            self.table.removeRow(self.table.rowCount() - 1)

        state = "CORRUPT" if corrupt else "OK"

        self.log.appendPlainText(
            f"t={sim_time:.6f}  "
            f"{NODE_NAMES.get(from_id, f'Node {from_id}')} -> "
            f"{NODE_NAMES.get(to_id, f'Node {to_id}')}  "
            f"NR DL  RNTI={rnti}  "
            f"TB={tb_size}B  "
            f"SINR={sinr_db:.2f} dB  "
            f"{state}"
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
