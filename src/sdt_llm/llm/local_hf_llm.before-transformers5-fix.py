"""
Real local LLM inference via Hugging Face `transformers`.

Requires `pip install -r requirements-full.txt` (torch + transformers) AND
internet access to Hugging Face Hub to download weights the first time —
neither is available in the sandbox this project was built/tested in (see
README, "Why some paths are marked 'not run in this sandbox'"), but both
are normal on your own Ubuntu machine.

Any instruction-tuned causal-LM on the Hub that supports
`tokenizer.apply_chat_template` will work — swap `model_name` for whatever
you have room for. A few reasonable, small (CPU-friendly) choices as of
early/mid-2026 are listed below; check the Hub for what's current when you
read this, model rankings move fast.

    "Qwen/Qwen2.5-1.5B-Instruct"      # good quality/size trade-off, ~3GB fp16
    "meta-llama/Llama-3.2-1B-Instruct"  # gated on HF, needs a license click-through
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # small, fully open
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
        dtype: str = "auto",
        system_prompt: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "LocalHFLLM needs torch+transformers. Install with:\n"
                "  pip install -r requirements-full.txt\n"
                f"(original error: {e})"
            ) from e

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype if dtype != "auto" else "auto"
        ).to(device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with self._torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
