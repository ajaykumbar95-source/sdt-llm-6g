"""Pick an LLM backend by name; keeps demo scripts/config free of import
branching. Only `mock` is guaranteed to work with just requirements.txt."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sdt_llm.llm.base import BaseLLM


def build_llm(backend: str = "mock", **kwargs: Any) -> BaseLLM:
    if backend == "mock":
        from sdt_llm.llm.mock_llm import MockLLM
        return MockLLM()
    if backend == "hf_local":
        from sdt_llm.llm.local_hf_llm import LocalHFLLM
        return LocalHFLLM(**kwargs)
    if backend == "anthropic_api":
        from sdt_llm.llm.api_llm import AnthropicLLM
        return AnthropicLLM(**kwargs)
    if backend == "openai_compatible_api":
        from sdt_llm.llm.api_llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM(**kwargs)
    raise ValueError(
        f"Unknown llm backend '{backend}'. Choose one of: "
        "mock, hf_local, anthropic_api, openai_compatible_api."
    )
