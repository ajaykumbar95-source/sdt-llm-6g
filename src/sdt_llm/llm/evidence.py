from __future__ import annotations

from typing import Dict, List, Any

from sdt_llm.tokens import SemanticToken


def _ue_key(token: SemanticToken) -> Any:
    return token.attributes.get("rnti")


def build_canonical_ue_evidence(tokens: List[SemanticToken]) -> str:
    """
    Build a deterministic, RNTI-grouped evidence block for the LLM.

    Numeric measurements and semantic labels remain exactly as supplied
    by the SDT. The model is not responsible for reconstructing UE identity
    from a mixed chronological stream.
    """

    grouped: Dict[Any, List[SemanticToken]] = {}

    for token in tokens:
        rnti = _ue_key(token)

        if rnti is None:
            continue

        grouped.setdefault(rnti, []).append(token)

    lines = [
        "[CANONICAL UE EVIDENCE]",
        "Each UE block contains only measurements belonging to that RNTI.",
        "Do not move measurements between UE blocks.",
        "",
    ]

    for rnti in sorted(grouped, key=lambda x: str(x)):
        # Prefer the latest token for the UE because the current question
        # concerns the current network state.
        latest = max(
            grouped[rnti],
            key=lambda token: token.timestamp,
        )

        attrs = latest.attributes

        lines.append(f"UE / RNTI={rnti}")

        for key in (
            "ue_ip",
            "cell_id",
            "bwp_id",
            "sinr_db",
            "packet_loss_pct",
            "throughput_mbps",
            "mean_delay_ms",
            "mean_jitter_ms",
            "delivery_ratio",
            "ue_gnb_distance_m",
        ):
            if key in attrs:
                lines.append(f"  {key}={attrs[key]}")

        lines.append(
            f"  semantic_label={latest.label}"
        )

        lines.append(
            f"  timestamp={latest.timestamp:.3f}s"
        )

        lines.append("")

    return "\n".join(lines)
