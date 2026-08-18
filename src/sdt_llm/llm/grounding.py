from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class UEGroundTruth:
    ue_ip: str
    rnti: int
    radio_label: str
    network_label: str
    sinr_db: float
    packet_loss_pct: float
    throughput_mbps: float
    delay_ms: float


def _entity_is_referenced(answer: str, gt: UEGroundTruth) -> bool:
    text = answer.lower()

    return (
        gt.ue_ip.lower() in text
        or f"rnti={gt.rnti}" in text
        or f"rnti {gt.rnti}" in text
        or f"ue / rnti={gt.rnti}" in text
        or f"ue/rnti={gt.rnti}" in text
    )


def validate_answer(
    answer: str,
    ground_truth: Dict[str, UEGroundTruth],
) -> tuple[bool, str]:
    """
    Reject hard contradictions in simulator-grounded UE evidence.

    Entity matching accepts either the UE IP or RNTI because the LLM
    may identify a UE using either simulator-grounded identity.
    """

    text = answer.lower()

    for _, gt in ground_truth.items():

        if not _entity_is_referenced(answer, gt):
            continue

        if "degraded_radio_link" in gt.radio_label:
            forbidden = (
                "good radio link",
                "good_radio_link",
                "strong radio link",
                "strong_radio_link",
            )

            for phrase in forbidden:
                if phrase in text:
                    return (
                        False,
                        f"UE/RNTI={gt.rnti}: answer says '{phrase}' "
                        f"but SDT says '{gt.radio_label}'.",
                    )

        if "high_packet_loss" in gt.network_label:
            if "low packet loss" in text:
                return (
                    False,
                    f"UE/RNTI={gt.rnti}: answer contradicts "
                    f"high packet loss.",
                )

        # If the answer reports SINR, require the value associated
        # with the referenced UE rather than another UE's value.
        if "sinr" in text:
            expected = f"{gt.sinr_db:.2f}"

            # Detect the common explicit cross-UE swap.
            other_values = {
                f"{other.sinr_db:.2f}"
                for other in ground_truth.values()
                if other.rnti != gt.rnti
            }

            for other_value in other_values:
                if other_value in text and expected not in text:
                    return (
                        False,
                        f"UE/RNTI={gt.rnti}: answer appears to "
                        f"use another UE's SINR ({other_value}) "
                        f"instead of {expected}.",
                    )

        if "packet loss" in text:
            expected_loss = f"{gt.packet_loss_pct:.0f}%"

            other_losses = {
                f"{other.packet_loss_pct:.0f}%"
                for other in ground_truth.values()
                if other.rnti != gt.rnti
            }

            for other_loss in other_losses:
                if other_loss in text and expected_loss not in text:
                    return (
                        False,
                        f"UE/RNTI={gt.rnti}: answer appears to "
                        f"use another UE's packet-loss value "
                        f"({other_loss}) instead of {expected_loss}.",
                    )

    return True, "No hard grounding contradiction detected."
