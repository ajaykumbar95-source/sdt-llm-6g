"""
Semantic encoder for real ns-3 / 5G-LENA NR PHY measurements.

Input:
    time_s
    cell_id
    rnti
    bwp_id
    sinr_linear
    sinr_db
    ue_x
    ue_y
    ue_z
    gnb_x
    gnb_y
    gnb_z

These measurements come from NrUePhy::DlDataSinr while
5G-LENA uses the Sionna RT propagation/channel model.

The encoder converts each NR measurement into a SemanticToken
for the Semantic Digital Twin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from sdt_llm.encoders.base import BaseSensorEncoder, SeededLinearProjection
from sdt_llm.tokens import SemanticToken


@dataclass
class NrSinrEncoderConfig:
    embed_dim: int = 256

    excellent_sinr_db: float = 25.0
    good_sinr_db: float = 15.0
    poor_sinr_db: float = 5.0

    seed: int = 37


class NrSinrEncoder(BaseSensorEncoder):
    """
    Converts 5G NR UE SINR measurements into semantic radio-state tokens.
    """

    modality = "radio"

    def __init__(
        self,
        config: NrSinrEncoderConfig | None = None,
    ):
        self.cfg = config or NrSinrEncoderConfig()

        # Features:
        # SINR linear
        # SINR dB
        # cell ID
        # RNTI
        # BWP ID
        # UE x/y/z
        # gNB x/y/z
        # UE-gNB distance
        #
        # Total = 12 features.
        self._proj = SeededLinearProjection(
            in_dim=12,
            out_dim=self.cfg.embed_dim,
            seed=self.cfg.seed,
        )

    def _label(self, sinr_db: float) -> str:
        if sinr_db >= self.cfg.excellent_sinr_db:
            return "strong_radio_link"

        if sinr_db >= self.cfg.good_sinr_db:
            return "good_radio_link"

        if sinr_db >= self.cfg.poor_sinr_db:
            return "degraded_radio_link"

        return "poor_radio_link"

    @staticmethod
    def _distance(
        ue_x: float,
        ue_y: float,
        ue_z: float,
        gnb_x: float,
        gnb_y: float,
        gnb_z: float,
    ) -> float:
        return float(
            np.sqrt(
                (ue_x - gnb_x) ** 2
                + (ue_y - gnb_y) ** 2
                + (ue_z - gnb_z) ** 2
            )
        )

    def encode_measurement(
        self,
        timestamp: float,
        cell_id: int,
        rnti: int,
        bwp_id: int,
        sinr_linear: float,
        sinr_db: float,
        ue_x: float = 0.0,
        ue_y: float = 0.0,
        ue_z: float = 0.0,
        gnb_x: float = 0.0,
        gnb_y: float = 0.0,
        gnb_z: float = 0.0,
    ) -> SemanticToken:
        """
        Convert one NR SINR measurement into one SemanticToken.
        """

        label = self._label(sinr_db)

        distance_m = self._distance(
            ue_x,
            ue_y,
            ue_z,
            gnb_x,
            gnb_y,
            gnb_z,
        )

        features = np.array(
            [
                np.tanh(sinr_linear / 100.0),
                np.tanh(sinr_db / 30.0),
                np.tanh(cell_id / 10.0),
                np.tanh(rnti / 100.0),
                np.tanh(bwp_id / 10.0),

                np.tanh(ue_x / 100.0),
                np.tanh(ue_y / 100.0),
                np.tanh(ue_z / 20.0),

                np.tanh(gnb_x / 100.0),
                np.tanh(gnb_y / 100.0),
                np.tanh(gnb_z / 20.0),

                np.tanh(distance_m / 100.0),
            ],
            dtype=np.float32,
        )

        embedding = self._proj(features)

        # The UE position is the physical location stamp
        # associated with this radio measurement.
        location = (
            float(ue_x),
            float(ue_y),
            float(ue_z),
        )

        confidence = float(
            np.clip(
                (sinr_db + 10.0) / 40.0,
                0.05,
                0.99,
            )
        )

        return SemanticToken(
            embedding=embedding,
            label=label,
            modality="radio",
            timestamp=float(timestamp),
            location=location,
            attributes={
                "cell_id": int(cell_id),
                "rnti": int(rnti),
                "bwp_id": int(bwp_id),

                "sinr_linear": round(
                    float(sinr_linear),
                    4,
                ),

                "sinr_db": round(
                    float(sinr_db),
                    3,
                ),

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

                "gnb_x": round(
                    float(gnb_x),
                    3,
                ),

                "gnb_y": round(
                    float(gnb_y),
                    3,
                ),

                "gnb_z": round(
                    float(gnb_z),
                    3,
                ),

                "ue_gnb_distance_m": round(
                    distance_m,
                    3,
                ),

                "measurement_type": "5g_nr_dl_sinr",

                "source": (
                    "ns3_5g_lena_sionna_rt"
                ),

                "location_status": "measured",
            },
            confidence=confidence,
        )

    def encode(
        self,
        measurements,
    ) -> List[SemanticToken]:
        """
        Encode multiple measurement dictionaries.
        """

        tokens = []

        for row in measurements:
            token = self.encode_measurement(
                timestamp=float(row["time_s"]),
                cell_id=int(row["cell_id"]),
                rnti=int(row["rnti"]),
                bwp_id=int(row["bwp_id"]),
                sinr_linear=float(row["sinr_linear"]),
                sinr_db=float(row["sinr_db"]),

                ue_x=float(row.get("ue_x", 0.0)),
                ue_y=float(row.get("ue_y", 0.0)),
                ue_z=float(row.get("ue_z", 0.0)),

                gnb_x=float(row.get("gnb_x", 0.0)),
                gnb_y=float(row.get("gnb_y", 0.0)),
                gnb_z=float(row.get("gnb_z", 0.0)),
            )

            tokens.append(token)

        return tokens
