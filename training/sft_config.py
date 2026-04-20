"""
SFTConfig builder for TRL supervised fine-tuning.
"""

from pathlib import Path
from typing import Any, Dict

from trl import SFTConfig


def get_sft_config(config: Dict[str, Any], output_dir: Path):
    """Build a ``trl.SFTConfig`` from a nested config dict.

    Reads the ``training`` sub-key for hyperparameters and optionally reads
    ``wandb.enabled`` for reporting.  All values have sensible defaults so
    only the fields you care about need to be present.

    Config keys (under ``training``):
        num_epochs, batch_size, eval_batch_size, gradient_accumulation_steps,
        learning_rate, weight_decay, warmup_ratio, lr_scheduler_type,
        logging_steps, eval_steps, save_steps, save_total_limit,
        fp16, bf16, num_workers, packing, dataset_num_proc,
        neftune_noise_alpha.

    Top-level keys:
        max_length, wandb.enabled.

    Returns:
        ``trl.SFTConfig`` ready to pass to ``trl.SFTTrainer``.
    """
    t = config["training"]
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t.get("eval_batch_size", t["batch_size"]),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=t["learning_rate"],
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.1),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        assistant_only_loss=True,
        logging_steps=t.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=t.get("fp16", False),
        bf16=t.get("bf16", False),
        dataloader_num_workers=t.get("num_workers", 4),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        max_length=config.get("max_length", 512),
        packing=t.get("packing", False),
        dataset_num_proc=t.get("dataset_num_proc", 4),
        dataset_text_field=None,
        neftune_noise_alpha=t.get("neftune_noise_alpha"),
        report_to=["wandb"] if config.get("wandb", {}).get("enabled", False) else [],
        push_to_hub=False,
    )
