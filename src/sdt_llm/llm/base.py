"""Common interface every LLM inference backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        ...
