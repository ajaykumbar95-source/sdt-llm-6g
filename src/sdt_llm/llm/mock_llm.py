"""
A deterministic, template-driven stand-in for a real LLM. It does simple,
real parsing/counting over the SDT context block it's given (no hidden
ground truth, no randomness) so the *pipeline wiring* (SDT -> prompt -> "LLM"
-> answer) can be smoke-tested with zero downloads, zero API keys, and zero
GPU. It is NOT a language model and should never be mistaken for one — every
response is prefixed accordingly.

Use LocalHFLLM (local_hf_llm.py) or an API backend (api_llm.py) for actual
LLM reasoning quality.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from sdt_llm.llm.base import BaseLLM

_TAG = "[MOCK LLM — deterministic offline stand-in, NOT a real language model]"


def _parse_context_lines(prompt: str) -> List[str]:
    m = re.search(r"\[SDT context.*?\]\n(.*?)\n\n\[Task\]", prompt, re.S)
    if not m:
        return []
    return [ln.strip("- ").strip() for ln in m.group(1).splitlines() if ln.strip()]


def _parse_task(prompt: str) -> str:
    m = re.search(r"\[Task\]\n(.*?)(\n\n|\Z)", prompt, re.S)
    return m.group(1).strip() if m else ""


class MockLLM(BaseLLM):
    name = "mock"

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        lines = _parse_context_lines(prompt)
        task = _parse_task(prompt).lower()

        if not lines:
            return f"{_TAG}\nNo SDT context was found in the prompt; nothing to reason over."

        labels = [re.search(r"\]\s*([a-zA-Z0-9_:+]+)", ln) for ln in lines]
        labels = [m.group(1) for m in labels if m]
        label_counts = Counter(labels)
        movement_lines = [ln for ln in lines if "movement" in ln]
        obstacle_lines = [ln for ln in lines if "obstacle" in ln]
        interference_lines = [ln for ln in lines if "interference" in ln]

        parts = [_TAG]

        if any(w in task for w in ("how many", "count", "number of")):
            summary = ", ".join(f"{v}x {k}" for k, v in label_counts.most_common())
            parts.append(f"Counting distinct semantic-token labels in the supplied SDT context: {summary}.")
        elif any(w in task for w in ("where", "location")):
            for ln in lines[-5:]:
                parts.append(f"- {ln}")
        elif any(w in task for w in ("safe", "should", "proceed", "avoid", "reroute", "collision")):
            if movement_lines:
                parts.append(
                    f"{len(movement_lines)} 'movement' token(s) are present in the recent SDT context "
                    f"(e.g. \"{movement_lines[-1]}\"). A cautious recommendation would be to slow down "
                    f"or re-route until the moving entity's trajectory is clearly clear of the planned path."
                )
            else:
                parts.append("No 'movement' tokens found in the recent SDT context, so proceeding is "
                              "plausible from a pure obstacle/motion standpoint — but confirm no other "
                              "safety constraints apply.")
            if obstacle_lines:
                parts.append(f"Static 'obstacle' token(s) noted: {len(obstacle_lines)} "
                              f"(e.g. \"{obstacle_lines[-1]}\").")
        else:
            parts.append("Summary of the current SDT context:")
            for ln in lines[-6:]:
                parts.append(f"- {ln}")
            if interference_lines:
                parts.append(f"Note: {len(interference_lines)} low-confidence 'interference' "
                              f"token(s) were present and should be treated with caution.")

        parts.append(
            "(This is a template-filled response for pipeline smoke-testing — install "
            "requirements-full.txt and set llm_backend: hf_local (or api) in the config for "
            "real language-model reasoning over this same SDT context.)"
        )
        return "\n".join(parts)
