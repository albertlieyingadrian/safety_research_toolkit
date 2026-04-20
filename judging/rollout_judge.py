"""
Async LLM judging for rollouts and plain text responses.
"""

import asyncio
import logging
from typing import Any, List, Optional, Tuple

from safety_research_toolkit.io.api_client import generate_content_async, parse_json_response

logger = logging.getLogger(__name__)


def _make_judge_fn(
    client: Any,
    api_provider: str,
    model_name: str,
    judge_prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int,
):
    """Return an async function that judges a single text string."""
    async def _judge_one(text: str) -> Optional[str]:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    return await generate_content_async(
                        client=client,
                        api_provider=api_provider,
                        model_name=model_name,
                        prompt=judge_prompt.format(model_response=text),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as exc:
                    if attempt == max_retries - 1:
                        logger.warning("Judge call failed after %d retries: %s", max_retries, exc)
                        return None
                    await asyncio.sleep(2 ** attempt)
            return None
    return _judge_one


async def ajudge_rollouts(
    judge_prompt: str,
    rollouts: List[Any],
    client: Any,
    model_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_concurrent: int = 10,
    max_retries: int = 3,
    api_provider: str = "anthropic",
) -> List[Tuple[Any, List[Tuple[Any, Optional[dict]]]]]:
    """Judge every response inside each rollout asynchronously.

    ``rollout.responses`` is expected to be an iterable of objects with a
    ``.full`` attribute containing the response text.

    Args:
        judge_prompt: Prompt template with a ``{model_response}`` placeholder.
        rollouts: Rollout objects (as returned by
            :func:`~safety_research_toolkit.io.rollouts.agenerate_rollout`).
        client: Async API client (e.g. ``anthropic.AsyncAnthropic``).
        model_name: Judge model identifier.
        max_tokens: Max response tokens for the judge.
        temperature: Sampling temperature.
        max_concurrent: Maximum simultaneous judge requests.
        max_retries: Retries per request on failure (exponential backoff).
        api_provider: One of ``'anthropic'``, ``'openai'``, ``'together'``.

    Returns:
        List of ``(rollout, [(response, judge_result), ...])`` tuples.
        ``judge_result`` is a dict if the response was valid JSON, else ``None``.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    judge_one = _make_judge_fn(
        client, api_provider, model_name, judge_prompt,
        max_tokens, temperature, semaphore, max_retries,
    )

    async def _judge_rollout(rollout):
        raw_results = await asyncio.gather(*[judge_one(r.full) for r in rollout.responses])
        return rollout, [
            (resp, parse_json_response(raw) if raw is not None else None)
            for resp, raw in zip(rollout.responses, raw_results)
        ]

    logger.info(
        "Judging %d rollouts (max %d concurrent, provider=%s)...",
        len(rollouts), max_concurrent, api_provider,
    )
    return list(await asyncio.gather(*[_judge_rollout(r) for r in rollouts]))


async def ajudge_responses(
    judge_prompt: str,
    responses: List[str],
    client: Any,
    model_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_concurrent: int = 10,
    max_retries: int = 3,
    api_provider: str = "anthropic",
) -> List[Tuple[str, Optional[dict]]]:
    """Judge a flat list of response strings asynchronously.

    Simpler variant of :func:`ajudge_rollouts` when you have plain strings
    rather than rollout objects.

    Args:
        judge_prompt: Prompt template with a ``{model_response}`` placeholder.
        responses: List of response strings to judge.
        client: Async API client.
        model_name: Judge model identifier.
        max_tokens: Max response tokens for the judge.
        temperature: Sampling temperature.
        max_concurrent: Maximum simultaneous judge requests.
        max_retries: Retries per request on failure (exponential backoff).
        api_provider: One of ``'anthropic'``, ``'openai'``, ``'together'``.

    Returns:
        List of ``(response, judge_result)`` tuples.
        ``judge_result`` is a dict if the response was valid JSON, else ``None``.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    judge_one = _make_judge_fn(
        client, api_provider, model_name, judge_prompt,
        max_tokens, temperature, semaphore, max_retries,
    )

    logger.info(
        "Judging %d responses (max %d concurrent, provider=%s)...",
        len(responses), max_concurrent, api_provider,
    )
    raw_results = await asyncio.gather(*[judge_one(r) for r in responses])
    return [
        (resp, parse_json_response(raw) if raw is not None else None)
        for resp, raw in zip(responses, raw_results)
    ]
