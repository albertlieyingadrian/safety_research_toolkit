"""
Hook lifecycle utilities: cleanup, memory management, and position-based
hidden-state extraction helpers.

Source: agentic-misalignment/utils/model_utils.py
"""

import gc
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer


def remove_all_hooks(model: AutoModelForCausalLM) -> None:
    """Remove all forward and backward hooks from a model and its submodules."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
        module._backward_pre_hooks.clear()


def clean_memory() -> None:
    """Run garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_hidden_states_for_each_sentence_end_position(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    example: Dict[str, str],
    layers_to_collect: Optional[List[int]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Extract hidden states at sentence-ending token positions in the reasoning.

    Performs a single forward pass over the full prompt + reasoning text.
    Identifies positions where a token decodes to text ending with ``.`` and
    collects the hidden state at each such position.

    Args:
        model: HF causal LM (already loaded and eval'd).
        tokenizer: Corresponding tokenizer.
        example: Dict with keys ``'system_prompt'``, ``'user_prompt'``,
            ``'reasoning'``.
        layers_to_collect: Layer indices to collect from (None = all layers).

    Returns:
        Dict mapping ``'layer_N'`` → list of (hidden_dim,) tensors, one per
        sentence end found in the reasoning.
    """
    model.eval()
    n_layers = model.config.num_hidden_layers

    applied_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(applied_prompt)
    reasoning = example["reasoning"]
    full_reasoning_tokens = tokenizer.encode(reasoning, add_special_tokens=False)

    sentence_end_positions: List[int] = []
    current_position = len(prompt_tokens) - 1
    for token_id in full_reasoning_tokens:
        token_text = tokenizer.decode([token_id])
        if token_text.rstrip().endswith("."):
            sentence_end_positions.append(current_position)
        current_position += 1

    inputs = tokenizer(
        applied_prompt + reasoning, return_tensors="pt", padding=True
    ).to(model.device)

    activation_dict: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, input, output):
            hidden = output[0].detach()  # (batch, seq_len, hidden_dim)
            activation_dict[name] = [hidden[0, pos, :] for pos in sentence_end_positions]
        return hook

    targets = layers_to_collect if layers_to_collect is not None else range(n_layers)
    for idx in targets:
        handles.append(
            model.model.layers[idx].register_forward_hook(_make_hook(f"layer_{idx}"))
        )

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return activation_dict


def get_hidden_states_for_every_nth_token(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    example: Dict[str, str],
    layers_to_collect: Optional[List[int]] = None,
    n: int = 100,
) -> Dict[str, List[torch.Tensor]]:
    """Extract hidden states at every *n*-th reasoning token position.

    Performs a single forward pass over the full prompt + reasoning text and
    samples positions ``[prompt_len, prompt_len + n, prompt_len + 2n, ...]``.

    Args:
        model: HF causal LM (already loaded and eval'd).
        tokenizer: Corresponding tokenizer.
        example: Dict with keys ``'system_prompt'``, ``'user_prompt'``,
            ``'reasoning'``.
        layers_to_collect: Layer indices to collect from (None = all layers).
        n: Stride — collect every n-th token in the reasoning.

    Returns:
        Dict mapping ``'layer_N'`` → list of (hidden_dim,) tensors, one per
        sampled position.
    """
    model.eval()
    n_layers = model.config.num_hidden_layers

    applied_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(applied_prompt)
    prompt_len = len(prompt_tokens)
    reasoning = example["reasoning"]
    reasoning_tokens = tokenizer.encode(reasoning, add_special_tokens=False)

    token_positions = list(range(prompt_len, prompt_len + len(reasoning_tokens)))[::n]

    inputs = tokenizer(
        applied_prompt + reasoning, return_tensors="pt", padding=True
    ).to(model.device)

    activation_dict: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, input, output):
            hidden = output[0].detach()
            activation_dict[name] = [hidden[0, pos, :] for pos in token_positions]
        return hook

    targets = layers_to_collect if layers_to_collect is not None else range(n_layers)
    for idx in targets:
        handles.append(
            model.model.layers[idx].register_forward_hook(_make_hook(f"layer_{idx}"))
        )

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return activation_dict
