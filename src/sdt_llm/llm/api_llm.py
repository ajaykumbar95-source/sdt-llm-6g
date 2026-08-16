"""
API-based LLM inference — for when you'd rather call a hosted model than
run weights locally. Reads the API key from an environment variable; never
put keys in code or config files you might commit/share.

Requires the `requests` package (lightweight, in requirements.txt already)
and outbound internet access to the provider's API endpoint — not available
in the sandbox this project was built in, but normal on your own machine.

    export ANTHROPIC_API_KEY=...      # for AnthropicLLM
    export OPENAI_API_KEY=...         # for OpenAICompatibleLLM (or pass api_key_env=)
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from sdt_llm.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    name = "anthropic_api"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        system_prompt: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1/messages",
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.timeout_s = timeout_s

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set the {self.api_key_env} environment variable to use AnthropicLLM."
            )
        payload = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt
        resp = requests.post(
            self.base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text").strip()


class OpenAICompatibleLLM(BaseLLM):
    """Works with the OpenAI API itself, or any server exposing the same
    /chat/completions schema (many local-inference servers do, e.g. vLLM,
    text-generation-webui, LM Studio, Ollama's OpenAI-compat endpoint)."""

    name = "openai_compatible_api"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        system_prompt: Optional[str] = None,
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set the {self.api_key_env} environment variable to use OpenAICompatibleLLM."
            )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json={"model": self.model, "messages": messages, "max_tokens": max_new_tokens},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
