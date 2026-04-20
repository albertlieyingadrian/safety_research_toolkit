"""
PyTorch forward-hook-based activation extraction and steering.
"""

from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from tqdm import tqdm


def steering_hook(module, input, output, steering_vector: torch.Tensor, scale: float = 1.0):
    """Forward hook that adds a projection-based steering update.

    Computes ``scale * (output @ r_hat) * r_hat`` and adds it to the hidden
    states, where ``r_hat`` is the unit-normalised steering vector.
    """
    r_hat = (steering_vector / steering_vector.norm()).to(output[0].device)
    projection = output[0] @ r_hat
    output[0][:, :] += scale * projection.unsqueeze(-1) * r_hat


def get_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    steering_vector: Optional[torch.Tensor] = None,
    steering_layer: Optional[int] = None,
    layers_to_collect: Optional[List[int]] = None,
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Collect hidden states from model layers for a list of prompts.

    Performs a single forward pass (no generation). Optionally applies
    a steering vector at one layer during the pass.

    Args:
        model: HF causal LM (already loaded and eval'd).
        tokenizer: Corresponding tokenizer.
        prompts: Text prompts.
        steering_vector: Optional steering tensor to apply.
        steering_layer: Layer index to apply steering at.
        layers_to_collect: Layer indices to collect from (None = all).
        scale: Steering scale factor.

    Returns:
        Dict mapping ``'layer_N'`` → hidden-states tensor (seq_len, hidden_dim).
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    activation_dict: Dict[str, torch.Tensor] = {}
    handles = []

    def _hook_fn(name):
        def hook(module, input, output):
            activation_dict[name] = output[0].detach()
        return hook

    targets = layers_to_collect if layers_to_collect is not None else range(n_layers)
    for idx in targets:
        handles.append(
            model.model.layers[idx].register_forward_hook(_hook_fn(f"layer_{idx}"))
        )

    if steering_vector is not None and steering_layer is not None:
        handles.append(
            model.model.layers[steering_layer].register_forward_hook(
                partial(steering_hook, steering_vector=steering_vector, scale=scale)
            )
        )

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return activation_dict


def collect_qa_hidden_states(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 20,
    steering_vector: Optional[torch.Tensor] = None,
    steering_layer: Optional[int] = None,
    layers_to_collect: Optional[List[int]] = None,
    scale: float = 1.0,
    average_per_question_first: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect hidden states split by question tokens vs answer tokens.

    Args:
        df: DataFrame with 'question' and 'answer' columns.
        model: HF causal LM.
        tokenizer: Corresponding tokenizer.
        batch_size: Rows processed per forward pass.
        steering_vector: Optional steering tensor.
        steering_layer: Layer to steer at.
        layers_to_collect: Layers to collect (None = all).
        scale: Steering scale.
        average_per_question_first: If True, average per unique question before
            final mean (useful when multiple answers per question exist).

    Returns:
        Dict with keys 'question' and 'answer', each mapping layer name →
        averaged hidden-state tensor.
    """
    assert "question" in df.columns and "answer" in df.columns

    model.eval()

    if average_per_question_first:
        hs_q: Dict = defaultdict(lambda: defaultdict(list))
        hs_a: Dict = defaultdict(lambda: defaultdict(list))
        cnt_q: Dict = defaultdict(lambda: defaultdict(float))
        cnt_a: Dict = defaultdict(lambda: defaultdict(float))
    else:
        hs_q: Dict = defaultdict(list)
        hs_a: Dict = defaultdict(list)
        cnt_q: Dict = defaultdict(float)
        cnt_a: Dict = defaultdict(float)

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        questions = batch["question"].tolist()
        answers = batch["answer"].tolist()

        qa_prompts = [
            str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            for q, a in zip(questions, answers)
        ]

        inputs = tokenizer(qa_prompts, return_tensors="pt", padding=True).to(model.device)
        hidden_states = get_hidden_states(
            model, tokenizer, qa_prompts, steering_vector, steering_layer,
            layers_to_collect, scale,
        )

        q_lengths = [
            len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            for q in questions
        ]

        for layer, h in hidden_states.items():
            for idx, q_len in enumerate(q_lengths):
                mask = inputs["attention_mask"][idx].to(h.device)
                real = torch.where(mask == 1)[0]
                q_idx = real[:q_len]
                a_idx = real[q_len:]

                q_mask = torch.zeros_like(mask, dtype=torch.bool)
                q_mask[q_idx] = True
                a_mask = torch.zeros_like(mask, dtype=torch.bool)
                a_mask[a_idx] = True

                if average_per_question_first:
                    qt = questions[idx]
                    hs_q[layer][qt].append(h[q_mask].sum(0))
                    cnt_q[layer][qt] += float(q_mask.sum())
                    hs_a[layer][qt].append(h[a_mask].sum(0))
                    cnt_a[layer][qt] += float(a_mask.sum())
                else:
                    hs_q[layer].append(h[q_mask].sum(0))
                    cnt_q[layer] += float(q_mask.sum())
                    hs_a[layer].append(h[a_mask].sum(0))
                    cnt_a[layer] += float(a_mask.sum())

    def _reduce(hs, cnt, per_q: bool):
        out = {}
        for layer, data in hs.items():
            if per_q:
                avgs = []
                for qt, acts in data.items():
                    avgs.append(torch.stack(acts).sum(0) / cnt[layer][qt])
                out[layer] = torch.stack(avgs).mean(0)
            else:
                out[layer] = torch.stack(data).sum(0) / cnt[layer]
        return out

    return {
        "question": {k: v.detach().cpu() for k, v in _reduce(hs_q, cnt_q, average_per_question_first).items()},
        "answer": {k: v.detach().cpu() for k, v in _reduce(hs_a, cnt_a, average_per_question_first).items()},
    }


def get_steering_vector(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    safe_df: pd.DataFrame,
    unsafe_df: pd.DataFrame,
    layers: List[int],
    average_per_question_first: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute contrastive mean-diff steering vectors via PyTorch hooks.

    Args:
        model: HF causal LM.
        tokenizer: Corresponding tokenizer.
        safe_df: DataFrame with 'question'/'answer' for safe (negative) examples.
        unsafe_df: DataFrame with 'question'/'answer' for unsafe (positive) examples.
        layers: Layer indices to generate vectors for.
        average_per_question_first: Passed through to collect_qa_hidden_states.

    Returns:
        Dict mapping ``'layer_N'`` → steering tensor (unsafe - safe).
    """
    safe_hs = collect_qa_hidden_states(
        safe_df, model, tokenizer, layers_to_collect=layers,
        average_per_question_first=average_per_question_first,
    )
    unsafe_hs = collect_qa_hidden_states(
        unsafe_df, model, tokenizer, layers_to_collect=layers,
        average_per_question_first=average_per_question_first,
    )
    return {
        f"layer_{layer}": unsafe_hs["answer"][f"layer_{layer}"] - safe_hs["answer"][f"layer_{layer}"]
        for layer in layers
    }


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    steering_vector: Dict[str, torch.Tensor],
    prompts: List[str],
    layers: List[int],
    scale: float,
    max_new_tokens: int = 200,
) -> List[tuple]:
    """Generate text with a steering vector applied at specified layers.

    Args:
        model: HF causal LM.
        tokenizer: Corresponding tokenizer.
        steering_vector: Dict from get_steering_vector / layer_N → tensor.
        prompts: Input prompts.
        layers: Layer indices to steer at.
        scale: Steering magnitude.
        max_new_tokens: Max generation length.

    Returns:
        List of (prompt, generated_text) tuples.
    """
    model.eval()
    handles = []
    for layer in layers:
        handles.append(
            model.model.layers[layer].register_forward_hook(
                partial(
                    steering_hook,
                    steering_vector=steering_vector[f"layer_{layer}"],
                    scale=scale,
                )
            )
        )
    try:
        messages = [[{"role": "user", "content": p}] for p in prompts]
        formatted = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                use_cache=False,
                top_p=1.0,
            )

        results = []
        for i, out in enumerate(outputs):
            prompt_len = int(inputs["attention_mask"][i].sum().item())
            text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            results.append((prompts[i], text))
        return results
    finally:
        for h in handles:
            h.remove()
