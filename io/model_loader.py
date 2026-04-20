"""
HuggingFace model loading and batched generation via ModelManager.
"""

import gc
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

_GENERATION_DEFAULTS = dict(
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)



class ModelManager:
    """Lazy-loading HF model wrapper with single and batched generation.

    Attributes:
        model_path: HF model path or local checkpoint.
        quantization: Optional BnB quantization level ('4bit' or '8bit').
        template_path: Optional path to a custom chat template file.
    """

    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = "4bit",
        template_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.template_path = template_path
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer (no-op if already loaded).

        Returns:
            (model, tokenizer) ready for inference.
        """
        if self.model is not None:
            return self.model, self.tokenizer

        print(f"Loading model: {self.model_path}")
        bnb_config = None
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        self.model.eval()

        tok_kwargs: Dict = {"padding_side": "left"}
        if self.template_path:
            print(f"Using chat template: {self.template_path}")
            with open(self.template_path) as f:
                tok_kwargs["chat_template"] = f.read()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model, self.tokenizer

    def _device(self):
        return next(self.model.parameters()).device

    def _assert_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512
    ) -> str:
        """Generate a single response for a conversation history."""
        self._assert_loaded()
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **_GENERATION_DEFAULTS,
            )

        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0][input_length:]
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        max_tokens: int = 1500,
        forced_token: str = "",
    ) -> List[str]:
        """Generate responses for a batch of conversation histories.

        Args:
            batch_messages: List of conversation histories.
            max_tokens: Max new tokens per response.
            forced_token: Token prepended to each generation prompt (force-start).
        """
        self._assert_loaded()
        formatted = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ) + forced_token
            for msgs in batch_messages
        ]
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **_GENERATION_DEFAULTS,
            )

        input_length = inputs["input_ids"].shape[1]
        responses = [
            self.tokenizer.decode(out[input_length:], skip_special_tokens=True).strip()
            for out in outputs
        ]
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        return responses
