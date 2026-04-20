"""
nnsight-based activation extraction with precise token-position targeting.

Requires the optional nnsight dependency: pip install safety-research-toolkit[nnsight]
"""

import gc
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoTokenizer

try:
    from nnsight import LanguageModel as NNsightLanguageModel  # type: ignore
    _NNSIGHT_AVAILABLE = True
except ImportError:
    _NNSIGHT_AVAILABLE = False


def _require_nnsight() -> None:
    if not _NNSIGHT_AVAILABLE:
        raise ImportError(
            "nnsight is required for this function.\n"
            "Install it with:  pip install maydixit-toolkit[nnsight]"
        )


def extract_activations_at_token(
    model,
    prompt: str,
    token_type: str,
    layers: Iterable[int],
) -> Dict[int, torch.Tensor]:
    """Extract hidden-state activations at the last occurrence of a specific token.

    Uses nnsight's trace context manager for exact token-position access.

    Args:
        model: A loaded ``nnsight.LanguageModel``.
        prompt: Input text.
        token_type: Token string to locate (e.g. ``'A'`` to find the ``(A)`` token).
        layers: Layer indices to extract from.

    Returns:
        Dict mapping layer index → 1-D activation tensor (hidden_dim,).

    Raises:
        ValueError: If ``token_type`` is not found in the tokenised prompt.
        ImportError: If nnsight is not installed.
    """
    _require_nnsight()
    layers = list(layers)

    token_ids = model.tokenizer(prompt)["input_ids"]
    target_id = model.tokenizer.convert_tokens_to_ids(token_type)
    positions = [i for i, tid in enumerate(token_ids) if tid == target_id]

    if not positions:
        raise ValueError(
            f"Token '{token_type}' (id={target_id}) not found in prompt.\n"
            f"Prompt (first 120 chars): {prompt[:120]!r}"
        )

    last_pos = positions[-1]
    layer_modules = model.model.layers
    saved: Dict[int, object] = {}

    with model.trace(prompt):
        for layer in layers:
            saved[layer] = layer_modules[layer].output.save()

    result: Dict[int, torch.Tensor] = {}
    for layer, raw in saved.items():
        hidden = raw[0] if isinstance(raw, tuple) else raw
        result[layer] = hidden[0, last_pos, :].detach().clone()

    return result


def extract_activations_all_layers(
    model,
    prompt: str,
    token_type: str,
    layers: Iterable[int],
) -> Dict[int, torch.Tensor]:
    """Alias for :func:`extract_activations_at_token` with all-layers framing.

    Kept for API compatibility with eval-awareness-probes scripts.
    """
    return extract_activations_at_token(model, prompt, token_type, layers)


def generate_contrastive_vectors(
    model,
    dataset: List[Dict],
    layers: Iterable[int],
) -> Dict[int, torch.Tensor]:
    """Compute mean-diff steering vectors from a contrastive dataset.

    Dataset items must have keys ``'question'``, ``'test'``, ``'deploy'``.
    ``'test'`` and ``'deploy'`` answers are expected to start with a token like
    ``'(A)'`` — the second character (index 1) is used as the extraction token.

    Args:
        model: Loaded nnsight LanguageModel.
        dataset: List of contrastive examples.
        layers: Layer indices to generate vectors for.

    Returns:
        Dict mapping layer index → steering vector (test_mean - deploy_mean).
    """
    _require_nnsight()
    layers = list(layers)
    test_acts: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    deploy_acts: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

    for idx, item in enumerate(dataset):
        q = item["question"]
        test_ans = item["test"]
        deploy_ans = item["deploy"]

        test_prompt = f"{q}\n\n {test_ans}"
        deploy_prompt = f"{q}\n\n {deploy_ans}"

        test_tok = test_ans[1]    # e.g. 'A' from '(A) ...'
        deploy_tok = deploy_ans[1]

        t_acts = extract_activations_at_token(model, test_prompt, test_tok, layers)
        d_acts = extract_activations_at_token(model, deploy_prompt, deploy_tok, layers)

        for layer in layers:
            test_acts[layer].append(t_acts[layer])
            deploy_acts[layer].append(d_acts[layer])

        del t_acts, d_acts
        gc.collect()

    vectors: Dict[int, torch.Tensor] = {}
    for layer in layers:
        t_mean = torch.stack(test_acts[layer]).mean(dim=0)
        d_mean = torch.stack(deploy_acts[layer]).mean(dim=0)
        vectors[layer] = t_mean - d_mean

    return vectors


def load_nnsight_model(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    **kwargs,
):
    """Convenience wrapper to load a nnsight LanguageModel.

    Args:
        model_name: HuggingFace model identifier.
        tokenizer_name: If None, uses model_name for the tokenizer.
        **kwargs: Passed through to nnsight.LanguageModel (e.g. device_map, torch_dtype).

    Returns:
        nnsight.LanguageModel instance.
    """
    _require_nnsight()
    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = NNsightLanguageModel(model_name, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model


