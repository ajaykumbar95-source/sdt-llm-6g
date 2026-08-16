"""
Semantic encoder for end-to-end network measurements.

Input:
    FlowMonitor statistics exported by ns-3.

The encoder converts network-level measurements into
SemanticToken objects for the Semantic Digital Twin.

Important:
    Network measurements are not assigned a physical location.
    Their location_status is explicitly "unknown".
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
    Converts ns-3 FlowMonitor network statistics
    into semantic network-state tokens.
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

        # Features:
        #
        # throughput
        # delay
        # jitter
        # packet loss
        # tx packets
        # rx packets
        # flow id
        # protocol
        #
        # plus reserved fields for future network metrics.
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
        protocol: str,
        source: str,
        destination: str,
        tx_packets: int,
        rx_packets: int,
        throughput_mbps: float,
        mean_delay_ms: float,
        mean_jitter_ms: float,
    ) -> SemanticToken:
        """
        Convert one FlowMonitor measurement
        into one SemanticToken.
        """

        # --------------------------------------------------------
        # Calculate observed delivery deficit.
        #
        # Do NOT rely solely on FlowMonitor lostPackets.
        # --------------------------------------------------------

        if tx_packets > 0:
            packet_loss_pct = (
                max(tx_packets - rx_packets, 0)
                / tx_packets
                * 100.0
            )
        else:
            packet_loss_pct = 0.0

        label = self._label(
            throughput_mbps=throughput_mbps,
            delay_ms=mean_delay_ms,
            jitter_ms=mean_jitter_ms,
            packet_loss_pct=packet_loss_pct,
        )

        protocol_value = (
            1.0
            if protocol.upper() == "UDP"
            else 0.0
        )

        delivery_ratio = (
            rx_packets / tx_packets
            if tx_packets > 0
            else 0.0
        )

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
                    packet_loss_pct / 20.0
                ),

                np.tanh(
                    tx_packets / 1000.0
                ),

                np.tanh(
                    rx_packets / 1000.0
                ),

                np.tanh(
                    delivery_ratio
                ),

                np.tanh(
                    flow_id / 10.0
                ),

                protocol_value,

                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        embedding = self._proj(features)

        # --------------------------------------------------------
        # Network flows do not currently have a physical
        # location in the ns-3 export.
        #
        # Therefore this is deliberately NOT interpreted
        # as a physical coordinate.
        # --------------------------------------------------------

        location = (
            0.0,
            0.0,
            0.0,
        )

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

        return SemanticToken(
            embedding=embedding,
            label=label,
            modality="network",
            timestamp=float(timestamp),
            location=location,
            attributes={
                "flow_id": int(flow_id),
                "protocol": protocol,
                "source": source,
                "destination": destination,
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
                    "ns3_flowmonitor_network_state"
                ),
                "source_system": (
                    "ns3_5g_lena_sionna_rt"
                ),
                "location_status": "unknown",
            },
            confidence=confidence,
        )

    def encode(
        self,
        measurements,
        timestamp: float,
    ) -> List[SemanticToken]:
        """
        Encode multiple network-flow measurements.
        """

        tokens = []

        for row in measurements:

            token = self.encode_measurement(
                timestamp=timestamp,
                flow_id=int(row["flow_id"]),
                protocol=str(row["protocol"]),
                source=str(row["source"]),
                destination=str(row["destination"]),
                tx_packets=int(row["tx_packets"]),
                rx_packets=int(row["rx_packets"]),
                throughput_mbps=float(
                    row["throughput_mbps"]
                ),
                mean_delay_ms=float(
                    row["mean_delay_ms"]
                ),
                mean_jitter_ms=float(
                    row["mean_jitter_ms"]
                ),
            )

            tokens.append(token)

        return tokens
