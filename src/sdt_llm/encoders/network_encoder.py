"""
Semantic encoder for time-aligned ns-3 network measurements.

Each input row represents one FlowMonitor interval for one UE flow.

Expected fields:

    time_s
    flow_id
    rnti
    ue_ip
    ue_x
    ue_y
    ue_z
    protocol
    source
    destination
    interval_tx_packets
    interval_rx_packets
    interval_lost_packets
    interval_packet_loss_pct
    interval_throughput_mbps
    interval_mean_delay_ms
    interval_mean_jitter_ms

The important change is that network state now carries the SAME
UE identity and physical position used by the NR radio encoder:

    rnti
    ue_x
    ue_y
    ue_z

This allows the Semantic Digital Twin to associate radio and
network observations for the same UE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from sdt_llm.encoders.base import (
    BaseSensorEncoder,
    SeededLinearProjection,
)

from sdt_llm.tokens import SemanticToken


@dataclass
class NetworkEncoderConfig:
    embed_dim: int = 256

    excellent_throughput_mbps: float = 50.0
    good_throughput_mbps: float = 20.0

    low_delay_ms: float = 5.0
    high_delay_ms: float = 20.0

    low_jitter_ms: float = 2.0
    high_jitter_ms: float = 10.0

    high_packet_loss_pct: float = 5.0

    seed: int = 73


class NetworkStateEncoder(BaseSensorEncoder):
    """
    Converts interval-based ns-3 FlowMonitor measurements
    into UE-associated semantic network-state tokens.
    """

    modality = "network"

    def __init__(
        self,
        config: NetworkEncoderConfig | None = None,
    ):
        self.cfg = (
            config
            if config is not None
            else NetworkEncoderConfig()
        )

        # 12 features:
        #
        #  1 throughput
        #  2 delay
        #  3 jitter
        #  4 packet loss
        #  5 interval TX packets
        #  6 interval RX packets
        #  7 delivery ratio
        #  8 flow ID
        #  9 RNTI
        # 10 UE x
        # 11 UE y
        # 12 UE z
        #
        self._proj = SeededLinearProjection(
            in_dim=12,
            out_dim=self.cfg.embed_dim,
            seed=self.cfg.seed,
        )

    def _label(
        self,
        throughput_mbps: float,
        delay_ms: float,
        jitter_ms: float,
        packet_loss_pct: float,
    ) -> str:
        """
        Produce a semantic description of network quality.
        """

        if packet_loss_pct >= self.cfg.high_packet_loss_pct:
            return "high_packet_loss"

        if delay_ms >= self.cfg.high_delay_ms:
            return "high_latency"

        if jitter_ms >= self.cfg.high_jitter_ms:
            return "high_jitter"

        if (
            throughput_mbps >= self.cfg.excellent_throughput_mbps
            and delay_ms <= self.cfg.low_delay_ms
            and jitter_ms <= self.cfg.low_jitter_ms
        ):
            return "excellent_network_quality"

        if throughput_mbps >= self.cfg.good_throughput_mbps:
            return "good_network_quality"

        return "degraded_network_quality"

    def encode_measurement(
        self,
        timestamp: float,
        flow_id: int,
        rnti: int,
        ue_ip: str,
        ue_x: float,
        ue_y: float,
        ue_z: float,
        protocol: str,
        source: str,
        destination: str,
        tx_packets: int,
        rx_packets: int,
        throughput_mbps: float,
        mean_delay_ms: float,
        mean_jitter_ms: float,
        packet_loss_pct: float | None = None,
    ) -> SemanticToken:
        """
        Convert one interval network measurement into
        one UE-associated SemanticToken.
        """

        # --------------------------------------------------------
        # Packet loss
        #
        # Prefer the value exported by ns-3, but calculate it
        # from TX/RX if it was not supplied.
        # --------------------------------------------------------

        if packet_loss_pct is None:

            if tx_packets > 0:
                packet_loss_pct = (
                    max(
                        tx_packets - rx_packets,
                        0,
                    )
                    / tx_packets
                    * 100.0
                )
            else:
                packet_loss_pct = 0.0

        packet_loss_pct = float(
            packet_loss_pct
        )

        # --------------------------------------------------------
        # Semantic network label
        # --------------------------------------------------------

        label = self._label(
            throughput_mbps=throughput_mbps,
            delay_ms=mean_delay_ms,
            jitter_ms=mean_jitter_ms,
            packet_loss_pct=packet_loss_pct,
        )

        # --------------------------------------------------------
        # Delivery ratio
        # --------------------------------------------------------

        delivery_ratio = (
            rx_packets / tx_packets
            if tx_packets > 0
            else 0.0
        )

        # --------------------------------------------------------
        # Protocol encoding
        # --------------------------------------------------------

        protocol_value = (
            1.0
            if protocol.upper() == "UDP"
            else 0.0
        )

        # --------------------------------------------------------
        # 12-dimensional semantic feature vector
        # --------------------------------------------------------

        features = np.array(
            [
                np.tanh(
                    throughput_mbps / 100.0
                ),

                np.tanh(
                    mean_delay_ms / 50.0
                ),

                np.tanh(
                    mean_jitter_ms / 20.0
                ),

                np.tanh(
                    packet_loss_pct / 100.0
                ),

                np.tanh(
                    tx_packets / 100.0
                ),

                np.tanh(
                    rx_packets / 100.0
                ),

                np.tanh(
                    delivery_ratio
                ),

                np.tanh(
                    flow_id / 10.0
                ),

                np.tanh(
                    rnti / 100.0
                ),

                np.tanh(
                    ue_x / 100.0
                ),

                np.tanh(
                    ue_y / 100.0
                ),

                np.tanh(
                    ue_z / 20.0
                ),
            ],
            dtype=np.float32,
        )

        embedding = self._proj(features)

        # --------------------------------------------------------
        # Network state now uses the ACTUAL UE position.
        #
        # This is what lets the SDT align the network token
        # spatially with the radio token for the same UE.
        # --------------------------------------------------------

        location = (
            float(ue_x),
            float(ue_y),
            float(ue_z),
        )

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        confidence = float(
            np.clip(
                1.0
                - (
                    packet_loss_pct / 100.0
                ),
                0.05,
                0.99,
            )
        )

        # --------------------------------------------------------
        # Semantic token
        # --------------------------------------------------------

        return SemanticToken(
            embedding=embedding,
            label=label,
            modality="network",
            timestamp=float(timestamp),
            location=location,
            attributes={
                "flow_id": int(flow_id),

                "rnti": int(rnti),

                "ue_ip": str(ue_ip),

                "ue_x": round(
                    float(ue_x),
                    3,
                ),

                "ue_y": round(
                    float(ue_y),
                    3,
                ),

                "ue_z": round(
                    float(ue_z),
                    3,
                ),

                "protocol": str(protocol),

                "source": str(source),

                "destination": str(destination),

                "tx_packets": int(tx_packets),

                "rx_packets": int(rx_packets),

                "packet_loss_pct": round(
                    float(packet_loss_pct),
                    3,
                ),

                "throughput_mbps": round(
                    float(throughput_mbps),
                    4,
                ),

                "mean_delay_ms": round(
                    float(mean_delay_ms),
                    4,
                ),

                "mean_jitter_ms": round(
                    float(mean_jitter_ms),
                    4,
                ),

                "delivery_ratio": round(
                    float(delivery_ratio),
                    4,
                ),

                "measurement_type": (
                    "ns3_cross_layer_ue_network_state"
                ),

                "source_system": (
                    "ns3_5g_lena_sionna_rt"
                ),

                "location_status": (
                    "measured"
                ),
            },
            confidence=confidence,
        )

    def encode(
        self,
        measurements,
    ) -> List[SemanticToken]:
        """
        Encode multiple interval network measurements.
        """

        tokens: List[SemanticToken] = []

        for row in measurements:

            token = self.encode_measurement(
                timestamp=float(
                    row["time_s"]
                ),

                flow_id=int(
                    row["flow_id"]
                ),

                rnti=int(
                    row["rnti"]
                ),

                ue_ip=str(
                    row["ue_ip"]
                ),

                ue_x=float(
                    row["ue_x"]
                ),

                ue_y=float(
                    row["ue_y"]
                ),

                ue_z=float(
                    row["ue_z"]
                ),

                protocol=str(
                    row["protocol"]
                ),

                source=str(
                    row["source"]
                ),

                destination=str(
                    row["destination"]
                ),

                tx_packets=int(
                    row[
                        "interval_tx_packets"
                    ]
                ),

                rx_packets=int(
                    row[
                        "interval_rx_packets"
                    ]
                ),

                throughput_mbps=float(
                    row[
                        "interval_throughput_mbps"
                    ]
                ),

                mean_delay_ms=float(
                    row[
                        "interval_mean_delay_ms"
                    ]
                ),

                mean_jitter_ms=float(
                    row[
                        "interval_mean_jitter_ms"
                    ]
                ),

                packet_loss_pct=float(
                    row[
                        "interval_packet_loss_pct"
                    ]
                ),
            )

            tokens.append(token)

        return tokens
