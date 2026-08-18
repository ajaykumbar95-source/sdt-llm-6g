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
from sdt_llm.llm.grounding import (
    UEGroundTruth,
    validate_answer,
)
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

    def _build_ue_ground_truth(self) -> Dict[str, UEGroundTruth]:
        """
        Build simulator-grounded UE evidence from the latest SDT tokens.

        The SDT is authoritative for RNTI identity and measured values.
        """

        latest_by_rnti: Dict[int, Any] = {}

        for token in self.twin.query_context(
            k_history=self.cfg.k_history
        ):
            rnti = token.attributes.get("rnti")

            if rnti is None:
                continue

            try:
                rnti_int = int(rnti)
            except (TypeError, ValueError):
                continue

            previous = latest_by_rnti.get(rnti_int)

            if (
                previous is None
                or token.timestamp >= previous.timestamp
            ):
                latest_by_rnti[rnti_int] = token

        ground_truth: Dict[str, UEGroundTruth] = {}

        for rnti, token in latest_by_rnti.items():
            attrs = token.attributes

            ue_ip = str(
                attrs.get(
                    "ue_ip",
                    attrs.get(
                        "source",
                        f"RNTI={rnti}",
                    ),
                )
            )

            label_parts = token.label.split("+")

            radio_label = next(
                (
                    label
                    for label in label_parts
                    if "radio" in label
                ),
                token.label,
            )

            network_label = next(
                (
                    label
                    for label in label_parts
                    if (
                        "network" in label
                        or "packet_loss" in label
                    )
                ),
                token.label,
            )

            ground_truth[ue_ip] = UEGroundTruth(
                ue_ip=ue_ip,
                rnti=rnti,
                radio_label=radio_label,
                network_label=network_label,
                sinr_db=float(attrs.get("sinr_db", 0.0)),
                packet_loss_pct=float(
                    attrs.get("packet_loss_pct", 0.0)
                ),
                throughput_mbps=float(
                    attrs.get("throughput_mbps", 0.0)
                ),
                delay_ms=float(
                    attrs.get("mean_delay_ms", 0.0)
                ),
            )

        return ground_truth

    @staticmethod
    def _deterministic_fallback(
        question: str,
        ground_truth: Dict[str, UEGroundTruth],
    ) -> str:
        """
        Produce a simulator-grounded answer when the LLM cannot
        satisfy the grounding contract.
        """

        if not ground_truth:
            return (
                "The SDT does not contain sufficient RNTI-grounded "
                "measurements to answer this query."
            )

        worst = max(
            ground_truth.values(),
            key=lambda item: (
                item.packet_loss_pct,
                item.delay_ms,
            ),
        )

        return (
            f"UE IP {worst.ue_ip} (RNTI {worst.rnti}) has the "
            f"strongest cross-layer degradation. "
            f"Radio condition: {worst.radio_label}. "
            f"Network condition: {worst.network_label}. "
            f"SINR: {worst.sinr_db:.2f} dB; "
            f"packet loss: {worst.packet_loss_pct:.0f}%; "
            f"throughput: {worst.throughput_mbps:.2f} Mbps; "
            f"mean delay: {worst.delay_ms:.2f} ms."
        )

    def _generate_grounded_answer(
        self,
        question: str,
        prompt: str,
    ) -> str:
        """
        Generate, validate, retry once, then fall back deterministically.
        """

        ground_truth = self._build_ue_ground_truth()

        answer = self.llm.generate(prompt)

        valid, _ = validate_answer(
            answer,
            ground_truth,
        )

        if valid:
            return answer

        correction = (
            "\n\n[GROUNDING CORRECTION]\n"
            "Your previous answer contradicted simulator-grounded SDT "
            "evidence. Retry using the canonical UE/RNTI evidence. "
            "Do not swap measurements between UE identities. "
            "Return only claims supported by the supplied evidence.\n"
        )

        retry_prompt = prompt + correction

        retry_answer = self.llm.generate(retry_prompt)

        valid, _ = validate_answer(
            retry_answer,
            ground_truth,
        )

        if valid:
            return retry_answer

        return self._deterministic_fallback(
            question,
            ground_truth,
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

        return self._generate_grounded_answer(
            question,
            prompt,
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

        answer = self._generate_grounded_answer(
            question,
            prompt,
        )

        return {
            "prompt": prompt,
            "answer": answer,
        }
