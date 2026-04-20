"""
Model loading and PEFT/LoRA configuration utilities for fine-tuning.
"""

import logging
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_peft_config(config: Dict[str, Any]):
    """Build a LoRA config from a nested config dict.

    Looks for a ``peft`` sub-key with the fields below.  Returns ``None``
    if ``peft.enabled`` is falsy.

    Config keys (under ``peft``):
        enabled, lora_r, lora_alpha, lora_dropout, bias,
        target_modules, modules_to_save.

    Returns:
        ``peft.LoraConfig`` or ``None``.
    """
    peft_cfg = config.get("peft", {})
    if not peft_cfg.get("enabled", False):
        return None

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_cfg.get("lora_r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        lora_dropout=peft_cfg.get("lora_dropout", 0.1),
        bias=peft_cfg.get("bias", "none"),
        target_modules=peft_cfg.get("target_modules", "all-linear"),
        modules_to_save=peft_cfg.get("modules_to_save", None),
    )
    return lora


def build_peft_model(
    model: PreTrainedModel,
    config: Dict[str, Any],
) -> PreTrainedModel:
    """Apply PEFT (LoRA) to *model* if ``config['peft']['enabled']`` is True.

    Handles k-bit training preparation when quantization is also enabled.

    Returns:
        The original *model* (unchanged) or a ``peft.PeftModel``.
    """
    logger = logging.getLogger(__name__)
    lora_config = get_peft_config(config)
    if lora_config is None:
        return model

    if config.get("quantization", {}).get("enabled", False):
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "PEFT applied: %d / %d trainable params (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )
    return peft_model


def load_model_for_training(config: Dict[str, Any]) -> PreTrainedModel:
    """Load a causal LM from config dict and optionally apply PEFT.

    Config keys (top-level):
        name, quantization (dict with enabled/load_in_4bit/…),
        torch_dtype, trust_remote_code, device_map,
        gradient_checkpointing, peft (see :func:`get_peft_config`).

    Returns:
        ``PreTrainedModel`` (possibly wrapped with PEFT).
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading model: %s", config["name"])

    quant_cfg = config.get("quantization", {})
    bnb = None
    if quant_cfg.get("enabled", False):
        bnb = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            load_in_8bit=quant_cfg.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    dtype_str = config.get("torch_dtype")
    kwargs: Dict[str, Any] = {
        "quantization_config": bnb,
        "torch_dtype": getattr(torch, dtype_str) if dtype_str else None,
        "trust_remote_code": config.get("trust_remote_code", False),
        "low_cpu_mem_usage": True,
    }
    if config.get("device_map") is not None:
        kwargs["device_map"] = config["device_map"]

    model = AutoModelForCausalLM.from_pretrained(config["name"], **kwargs)

    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return build_peft_model(model, config)
