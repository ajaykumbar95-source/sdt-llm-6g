"""
SDTLLMPipeline — the single high-level entry point tying together:

    vision frame  --\
                      >--  SDT (cluster + fuse + track)  -->  prompt  -->  LLM  --> answer
    radio scene   --/

Real ns-3 / 5G-LENA / Sionna RT NR measurements are also supported.

The NR path is:

    ns-3 + 5G-LENA + Sionna RT
                |
                v
        sdt_radio_trace.csv
                |
                v
          NrSinrEncoder
                |
                v
        Semantic Digital Twin
                |
                v
              LLM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sdt_llm.data.synthetic_radio import RadioScene
from sdt_llm.encoders.radio_encoder import (
    RadioChannelEncoder,
    RadioEncoderConfig,
)
from sdt_llm.encoders.vision_encoder import VisionSensorEncoder
from sdt_llm.encoders.nr_sinr_encoder import (
    NrSinrEncoder,
    NrSinrEncoderConfig,
)
from sdt_llm.encoders.network_encoder import (
    NetworkStateEncoder,
    NetworkEncoderConfig,
)
from sdt_llm.llm.factory import build_llm
from sdt_llm.llm.prompt_builder import build_prompt
from sdt_llm.sdt.digital_twin import (
    SDTConfig,
    SemanticDigitalTwin,
    TimestampRecord,
)


@dataclass
class PipelineConfig:
    sdt: SDTConfig = field(
        default_factory=SDTConfig
    )

    vision_backend: str = "mock"

    llm_backend: str = "mock"

    llm_kwargs: Dict[str, Any] = field(
        default_factory=dict
    )

    k_history: int = 4

    nr_sinr_embed_dim: int = 256


class SDTLLMPipeline:
    """
    High-level SDT -> LLM pipeline.

    Supports:

    1. Vision measurements
    2. Synthetic radio scenes
    3. Real ns-3 / 5G-LENA / Sionna RT NR PHY measurements
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        self.cfg = config or PipelineConfig()

        # --------------------------------------------------------
        # Semantic Digital Twin
        # --------------------------------------------------------

        self.twin = SemanticDigitalTwin(
            self.cfg.sdt
        )

        # --------------------------------------------------------
        # LLM
        # --------------------------------------------------------

        self.llm = build_llm(
            self.cfg.llm_backend,
            **self.cfg.llm_kwargs,
        )

        # --------------------------------------------------------
        # Vision encoder
        # --------------------------------------------------------

        self.vision_encoder = VisionSensorEncoder(
            embed_dim=self.cfg.sdt.embed_dim,
            backend=self.cfg.vision_backend,
        )

        # --------------------------------------------------------
        # Synthetic radio encoder
        # --------------------------------------------------------

        self.radio_encoder = RadioChannelEncoder(
            RadioEncoderConfig(
                embed_dim=self.cfg.sdt.embed_dim
            )
        )

        # --------------------------------------------------------
        # Real ns-3 / 5G-LENA / Sionna RT encoder
        # --------------------------------------------------------

        self.nr_sinr_encoder = NrSinrEncoder(
            NrSinrEncoderConfig(
                embed_dim=self.cfg.nr_sinr_embed_dim
            )
        )

        self.network_encoder = NetworkStateEncoder(
            NetworkEncoderConfig(
                embed_dim=self.cfg.nr_sinr_embed_dim
            )
        )

    # ============================================================
    # Vision ingestion
    # ============================================================

    def ingest_vision(
        self,
        image_path: Union[str, Path],
        timestamp: float,
    ) -> TimestampRecord:
        """
        Encode a vision frame and ingest it into the SDT.
        """

        tokens = self.vision_encoder.encode(
            image_path,
            timestamp,
        )

        return self.twin.ingest(
            tokens,
            timestamp,
        )

    # ============================================================
    # Synthetic radio ingestion
    # ============================================================

    def ingest_radio(
        self,
        radio_scene: RadioScene,
        timestamp: Optional[float] = None,
    ) -> TimestampRecord:
        """
        Encode a synthetic RadioScene and ingest it
        into the SDT.
        """

        ts = (
            timestamp
            if timestamp is not None
            else radio_scene.timestamp
        )

        tokens = self.radio_encoder.encode(
            radio_scene,
            ts,
        )

        return self.twin.ingest(
            tokens,
            ts,
        )

    # ============================================================
    # Real NR SINR ingestion
    # ============================================================

    def ingest_nr_sinr(
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
    ) -> TimestampRecord:
        """
        Ingest one real 5G NR PHY measurement.

        Source:
            ns-3 + 5G-LENA + Sionna RT

        Measurement:
            NrUePhy::DlDataSinr

        Radio information:
            - cell ID
            - RNTI
            - BWP ID
            - linear SINR
            - SINR in dB

        Spatial information:
            - UE x/y/z
            - gNB x/y/z

        The NR SINR encoder converts the measurement into
        a SemanticToken before it enters the SDT.
        """

        token = self.nr_sinr_encoder.encode_measurement(
            timestamp=timestamp,
            cell_id=cell_id,
            rnti=rnti,
            bwp_id=bwp_id,
            sinr_linear=sinr_linear,
            sinr_db=sinr_db,
            ue_x=ue_x,
            ue_y=ue_y,
            ue_z=ue_z,
            gnb_x=gnb_x,
            gnb_y=gnb_y,
            gnb_z=gnb_z,
        )

        return self.twin.ingest(
            [token],
            timestamp,
        )

    # ============================================================
    # Network-state ingestion
    # ============================================================

    def ingest_network(
        self,
        timestamp: float,
        measurements,
    ) -> TimestampRecord:
        """
        Ingest end-to-end network measurements exported
        from ns-3 FlowMonitor.

        Each flow becomes a semantic network token.

        measurements must contain:

            flow_id
            protocol
            source
            destination
            tx_packets
            rx_packets
            throughput_mbps
            mean_delay_ms
            mean_jitter_ms
        """

        tokens = self.network_encoder.encode(
            measurements=measurements,
            timestamp=timestamp,
        )

        return self.twin.ingest(
            tokens,
            timestamp,
        )

    # ============================================================
    # Prompt / LLM
    # ============================================================

    def ingest_multimodal(
        self,
        timestamp: float,
        image_path: Union[str, Path],
        radio_scene: RadioScene,
    ) -> TimestampRecord:
        """
        Ingest vision and radio observations together at the
        same timestamp so the SDT can perform cross-modal
        fusion in one ingest() call.
        """

        vision_tokens = self.vision_encoder.encode(
            image_path,
            timestamp,
        )

        radio_tokens = self.radio_encoder.encode(
            radio_scene,
            timestamp,
        )

        tokens = (
            vision_tokens
            + radio_tokens
        )

        return self.twin.ingest(
            tokens,
            timestamp,
        )

    def answer(
        self,
        question: str,
        k_history: Optional[int] = None,
    ) -> str:
        """
        Build a prompt from the current SDT state
        and obtain an answer from the configured LLM.
        """

        prompt = build_prompt(
            query=question,
            twin=self.twin,
            k_history=(
                self.cfg.k_history
                if k_history is None
                else k_history
            ),
        )

        return self.llm.generate(
            prompt
        )

    def ask(
        self,
        question: str,
        k_history: Optional[int] = None,
    ) -> dict:
        """
        Backwards-compatible API used by the existing tests.

        Returns both the exact prompt sent to the LLM and the
        generated answer.
        """

        prompt = build_prompt(
            query=question,
            twin=self.twin,
            k_history=(
                self.cfg.k_history
                if k_history is None
                else k_history
            ),
        )

        answer = self.llm.generate(
            prompt
        )

        return {
            "prompt": prompt,
            "answer": answer,
        }
