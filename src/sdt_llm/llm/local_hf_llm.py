"""
Local Hugging Face LLM backend.

Compatible with the installed Transformers 5.x API.

Default model:
    Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

from typing import Optional

from sdt_llm.llm.base import BaseLLM


class LocalHFLLM(BaseLLM):
    name = "hf_local"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        dtype: str = "float32",
        system_prompt: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "LocalHFLLM requires torch and transformers. "
                "Install them with: pip install torch transformers"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt

        if dtype == "float32":
            torch_dtype = torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "auto":
            torch_dtype = "auto"
        else:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. "
                "Use float32, float16, bfloat16, or auto."
            )

        print(
            f"[LocalHFLLM] Loading tokenizer: {model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        print(
            f"[LocalHFLLM] Loading model on {device}: "
            f"{model_name}"
        )

        load_kwargs = {
            "dtype": torch_dtype,
        }

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                **load_kwargs,
            )
            .to(device)
        )

        self.model.eval()

        print("[LocalHFLLM] Model ready.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
    ) -> str:

        messages = []

        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        encoded = (
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        )

        # Transformers 5.x may return a BatchEncoding-like object.
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids

            attention_mask = getattr(
                encoded,
                "attention_mask",
                None,
            )
        else:
            input_ids = encoded
            attention_mask = None

        input_ids = input_ids.to(
            self.device
        )

        if attention_mask is not None:
            attention_mask = (
                attention_mask.to(
                    self.device
                )
            )

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": (
                self.tokenizer.eos_token_id
            ),
        }

        if attention_mask is not None:
            generation_kwargs[
                "attention_mask"
            ] = attention_mask

        with self._torch.no_grad():
            outputs = self.model.generate(
                **generation_kwargs
            )

        prompt_length = input_ids.shape[1]

        generated_tokens = outputs[
            0,
            prompt_length:,
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
