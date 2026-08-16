"""
Prompt builder — Section 3.2 "Effective Prompt Engineering" / "Context-Aware
Prediction":

  "the SDT can supply additional prompts to the LLM, including historical
  and spatial information, thereby enriching the context available... This
  allows LLMs to leverage both current and historical data for more precise
  and contextually relevant predictions."

`build_prompt()` renders recent (and, optionally, specifically-recalled)
fused SDT tokens into a structured context block, then appends the task/
query. The exact "[SDT context ...]\\n...\\n\\n[Task]\\n..." framing is a
plain, explicit format chosen so it's trivial for *any* backend — mock,
local, or API — to parse reliably; feel free to restyle it once you're
driving a specific model that prefers a different convention (e.g. JSON, or
that model's own system/user role structure).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from sdt_llm.sdt.digital_twin import SemanticDigitalTwin
from sdt_llm.tokens import SemanticToken

SYSTEM_PREAMBLE = (
    "You are the inference engine for a Semantic Digital Twin (SDT) of a "
    "physical environment. The SDT fuses semantic tokens extracted from "
    "sensors (camera vision) and/or 5G NR radio measurements from 5G-LENA/Sionna RT into "
    "a single spatiotemporal token stream. Each context line below is one "
    "fused semantic token: its timestamp, its (x, y, z) location in metres, "
    "the sensing modality that produced it, its semantic label, and any "
    "extra attributes."
)


def _format_lines(tokens: Iterable[SemanticToken]) -> str:
    return "\n".join(f"- {t.describe()}" for t in tokens)


def build_prompt(
    query: str,
    twin: SemanticDigitalTwin,
    k_history: int = 4,
    recall_labels: Optional[List[str]] = None,
    include_system_preamble: bool = True,
) -> str:
    """
    Parameters
    ----------
    query : the task/question for the LLM to answer.
    twin : the SemanticDigitalTwin to pull context from.
    k_history : how many recent timestamps' fused tokens to include.
    recall_labels : optional list of label substrings to explicitly recall
        from the *entire* history even if outside the k_history window
        (Sec. 3.2's "remembers where the biscuits were even though they're
        no longer in view" pattern) — each gets appended as a clearly-marked
        "recalled from earlier" line.
    """
    recent = twin.query_context(k_history=k_history)
    context_block = _format_lines(recent) if recent else "(no recent SDT tokens)"

    recall_lines = []
    if recall_labels:
        recent_ids = {t.token_id for t in recent}
        for lbl in recall_labels:
            fc = twin.last_seen(lbl)
            if fc is not None and fc.fused_token.token_id not in recent_ids:
                recall_lines.append(f"- [recalled from earlier] {fc.fused_token.describe()}")
    if recall_lines:
        context_block += "\n" + "\n".join(recall_lines)

    parts = []
    if include_system_preamble:
        parts.append(SYSTEM_PREAMBLE)
        parts.append("")
    parts.append(f"[SDT context — last {k_history} timestamp(s), fused semantic tokens]")
    parts.append(context_block)
    parts.append("")
    parts.append("[Task]")
    parts.append(query)
    parts.append("")
    parts.append(
        "Using the SDT context above (plus ordinary world knowledge where needed), "
        "answer concisely. If the context is insufficient, say what's missing rather than guessing."
    )
    return "\n".join(parts)
