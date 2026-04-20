"""
Linear probe training and steering-vector probe scoring utilities.

``collect_last_token_activations`` and the ``analyze_rollout_with_probe*``
functions require nnsight (pip install safety-research-toolkit[nnsight]).
``train_linear_probe`` and ``save_npz`` only need numpy + sklearn.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------------
# Activation collection (nnsight)
# ---------------------------------------------------------------------------

def collect_last_token_activations(
    llm,
    prompts: List[str],
    layer_index: int,
) -> np.ndarray:
    """Collect the last-token hidden state at *layer_index* for each prompt.

    Requires nnsight (``pip install maydixit-toolkit[nnsight]``).

    Args:
        llm: Loaded ``nnsight.LanguageModel``.
        prompts: List of raw text strings.
        layer_index: Transformer layer to extract from.

    Returns:
        Array of shape ``(len(prompts), hidden_dim)``.
    """
    layers = llm.model.model.layers
    activations: List[np.ndarray] = []
    for text in prompts:
        token_ids = llm.tokenizer(text)["input_ids"]
        last_index = len(token_ids) - 1
        with llm.trace(text):
            saved = layers[layer_index].output[0].save()
        vec = saved[0, last_index, :].detach().cpu().numpy()
        activations.append(vec)
    return np.stack(activations, axis=0)


# ---------------------------------------------------------------------------
# Activation persistence
# ---------------------------------------------------------------------------

def save_npz(npz_path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Save activations and labels to a compressed ``.npz`` file."""
    np.savez_compressed(npz_path, X=X, y=y)


# ---------------------------------------------------------------------------
# Probe training (sklearn)
# ---------------------------------------------------------------------------

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 200,
):
    """Train a logistic-regression linear probe and return metrics.

    Args:
        X_train: Training activations, shape ``(n, hidden_dim)``.
        y_train: Binary training labels.
        X_val: Validation activations.
        y_val: Binary validation labels.
        max_iter: ``LogisticRegression`` max_iter.

    Returns:
        Tuple of ``(fitted_clf, metrics_dict)``.
        ``metrics_dict`` keys: ``train_accuracy``, ``train_roc_auc``,
        ``val_accuracy``, ``val_roc_auc`` (``None`` when only one class present).
    """
    clf = LogisticRegression(max_iter=max_iter, n_jobs=1)
    clf.fit(X_train, y_train)

    def _metrics(X, y, split):
        pred = clf.predict(X)
        proba = clf.predict_proba(X)[:, 1]
        auc = (
            float(roc_auc_score(y, proba))
            if len(np.unique(y)) == 2
            else None
        )
        return {
            f"{split}_accuracy": float(accuracy_score(y, pred)),
            f"{split}_roc_auc": auc,
        }

    metrics = {**_metrics(X_train, y_train, "train"), **_metrics(X_val, y_val, "val")}
    return clf, metrics


# ---------------------------------------------------------------------------
# Rollout probe scoring (nnsight)
# ---------------------------------------------------------------------------

def analyze_rollout_with_probe(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    layer: int,
    prompt: str,
    response: str,
    tokenizer=None,
) -> Dict:
    """Score every token in a prompt+response against a steering probe.

    Uses teacher-forcing: the full ``prompt + response`` is passed through
    the model in one shot.  Causal masking ensures each position only attends
    to prior tokens, making this equivalent to per-step scoring.

    Expects Qwen-style thinking: ``response`` already contains ``</think>``;
    a ``<think>`` prefix is inserted before it.

    Args:
        model: Loaded ``nnsight.LanguageModel``.
        steering_vectors: Dict mapping layer index → steering tensor.
        layer: Layer to probe.
        prompt: Raw user prompt (no chat formatting).
        response: Model response text.
        tokenizer: Defaults to ``model.tokenizer``.

    Returns:
        Dict with keys ``prompt_scores``, ``response_scores``,
        ``prompt_mean``, ``response_mean``, ``response_max``,
        ``response_min``, ``full_tokens``, ``prompt_token_count``.
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    if layer not in steering_vectors:
        raise ValueError(f"No steering vector for layer {layer}.")

    sv = steering_vectors[layer].to(torch.bfloat16)

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = formatted_prompt + "<think>" + response
    prompt_len = len(tokenizer(formatted_prompt + "<think>")["input_ids"])
    full_tokens = tokenizer(full_text)["input_ids"]

    with torch.no_grad():
        with model.trace(full_text):
            layer_output = model.model.layers[layer].output.save()

    hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
    all_scores = [
        (torch.dot(hidden[0, pos, :], sv) / sv.norm()).item()
        for pos in range(hidden.shape[1])
    ]
    del hidden, layer_output
    torch.cuda.empty_cache()

    prompt_scores = all_scores[:prompt_len]
    response_scores = all_scores[prompt_len:]

    return {
        "prompt_scores": prompt_scores,
        "response_scores": response_scores,
        "prompt_mean": float(np.mean(prompt_scores)) if prompt_scores else 0.0,
        "response_mean": float(np.mean(response_scores)) if response_scores else 0.0,
        "response_max": float(max(response_scores)) if response_scores else 0.0,
        "response_min": float(min(response_scores)) if response_scores else 0.0,
        "full_tokens": [tokenizer.decode([t]) for t in full_tokens],
        "prompt_token_count": prompt_len,
    }


def analyze_rollout_with_probe_multi_layer(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    layers: List[int],
    prompt: str,
    response: str,
    tokenizer=None,
) -> Dict[int, Dict]:
    """Score a rollout against steering probes at multiple layers in one pass.

    More efficient than calling :func:`analyze_rollout_with_probe` in a loop.

    Args:
        model: Loaded ``nnsight.LanguageModel``.
        steering_vectors: Dict mapping layer index → steering tensor.
        layers: Layer indices to score.
        prompt: Raw user prompt.
        response: Model response text.
        tokenizer: Defaults to ``model.tokenizer``.

    Returns:
        Dict mapping layer index → result dict (same schema as
        :func:`analyze_rollout_with_probe`).
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = formatted_prompt + "<think>" + response
    prompt_len = len(tokenizer(formatted_prompt + "<think>")["input_ids"])

    layer_outputs: Dict[int, object] = {}
    with torch.no_grad():
        with model.trace(full_text):
            for layer in layers:
                if layer in steering_vectors:
                    layer_outputs[layer] = model.model.layers[layer].output.save()

    results: Dict[int, Dict] = {}
    for layer in layers:
        if layer not in steering_vectors or layer not in layer_outputs:
            continue

        sv = steering_vectors[layer].to(torch.bfloat16)
        raw = layer_outputs[layer]
        hidden = raw[0] if isinstance(raw, tuple) else raw

        all_scores = [
            (torch.dot(hidden[0, pos, :], sv) / sv.norm()).item()
            for pos in range(hidden.shape[1])
        ]

        prompt_scores = all_scores[:prompt_len]
        response_scores = all_scores[prompt_len:]

        results[layer] = {
            "prompt_scores": prompt_scores,
            "response_scores": response_scores,
            "prompt_mean": float(np.mean(prompt_scores)) if prompt_scores else 0.0,
            "response_mean": float(np.mean(response_scores)) if response_scores else 0.0,
            "response_max": float(max(response_scores)) if response_scores else 0.0,
            "response_min": float(min(response_scores)) if response_scores else 0.0,
            "prompt_token_count": prompt_len,
        }

    del layer_outputs
    torch.cuda.empty_cache()
    return results
