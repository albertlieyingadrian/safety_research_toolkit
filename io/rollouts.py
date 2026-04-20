"""
Rollout generation utilities using the RolloutsClient (OpenRouter).
"""

import asyncio
import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

from rollouts import RolloutsClient  # type: ignore

DEFAULT_NUM_ROLLOUTS = 100
DEFAULT_MAX_TOKENS = 30_000
DEFAULT_TEMPERATURE = 0.8
DEFAULT_THINKING_BUDGET = 300

DEFAULT_PROVIDER_ROUTING = {
    "order": ["Fireworks", "Lepton", "Hyperbolic", "DeepInfra", "Together"],
    "allow_fallbacks": True,
}


async def agenerate_rollout(
    model_name: str,
    prompt: str,
    api_key: str,
    n_samples: int = DEFAULT_NUM_ROLLOUTS,
    has_reasoning: bool = True,
    reasoning: Optional[dict] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    provider_routing: Optional[dict] = None,
    use_cache: bool = False,
) -> Any:
    """Generate rollouts for a single prompt via OpenRouter.

    Args:
        model_name: OpenRouter model identifier.
        prompt: Prompt string.
        api_key: OpenRouter API key.
        n_samples: Number of rollouts to generate.
        has_reasoning: Whether the model supports extended thinking.
        reasoning: Reasoning config, e.g. ``{"effort": "medium"}``.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        thinking_budget: Budget for thinking tokens.
        provider_routing: Override default provider order.
        use_cache: Enable OpenRouter caching.

    Returns:
        Rollouts object from RolloutsClient.
    """
    routing = provider_routing or DEFAULT_PROVIDER_ROUTING
    client_kwargs = dict(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        api_key=api_key,
        provider={"order": routing["order"], "allow_fallbacks": routing["allow_fallbacks"]},
        use_cache=use_cache,
    )
    if has_reasoning:
        client_kwargs["reasoning"] = reasoning or {"effort": "medium"}
        print(f"  Using model: {model_name} (reasoning enabled)")
    else:
        print(f"  Using model: {model_name}")

    client = RolloutsClient(**client_kwargs)
    rollouts = await client.agenerate(prompt, n_samples=n_samples, progress_bar=True)

    if not rollouts or len(rollouts) == 0:
        raise ValueError(f"No rollouts generated for {model_name}")

    return rollouts


async def agenerate_rollouts(
    model_name: str,
    prompts: List[str],
    api_key: str,
    n_samples: int = DEFAULT_NUM_ROLLOUTS,
    has_reasoning: bool = True,
    reasoning: Optional[dict] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    provider_routing: Optional[dict] = None,
    use_cache: bool = False,
    max_concurrent: int = 10,
) -> List[Any]:
    """Generate rollouts for multiple prompts in parallel with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _with_semaphore(p: str) -> Any:
        async with semaphore:
            return await agenerate_rollout(
                model_name=model_name,
                prompt=p,
                api_key=api_key,
                n_samples=n_samples,
                has_reasoning=has_reasoning,
                reasoning=reasoning,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_budget=thinking_budget,
                provider_routing=provider_routing,
                use_cache=use_cache,
            )

    print(f"Generating rollouts for {len(prompts)} prompts (max {max_concurrent} concurrent)...")
    return list(await asyncio.gather(*[_with_semaphore(p) for p in prompts]))


def save_rollouts(rollouts: Union[List, Any], filepath: str) -> None:
    """Pickle rollouts to disk, creating parent dirs as needed."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rollouts, f)
    print(f"Saved rollouts → {path}")


def load_rollouts(filepath: str) -> Any:
    """Load pickled rollouts from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
