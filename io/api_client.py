"""
API client utilities for OpenAI, Anthropic, Gemini, Together, and OpenRouter.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import google.generativeai as genai
import openai
from dotenv import load_dotenv

DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openrouter": "openrouter",
}


def init_api_client(
    api_provider: str,
    async_client: bool = True,
    model_name: Optional[str] = None,
) -> Tuple[Any, str]:
    """Initialise an API client for the specified provider.

    Args:
        api_provider: One of 'openai', 'anthropic', 'gemini', 'together', 'openrouter'.
        async_client: Return an async client where supported (Anthropic, OpenAI).

    Returns:
        (client, model_name) for the provider.

    Raises:
        ValueError: Missing API key or unknown provider.
    """
    load_dotenv()
    provider = api_provider.lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown API provider '{api_provider}'. "
            "Choose from: openai, anthropic, gemini, together, openrouter"
        )
    resolved_model = model_name or DEFAULT_MODELS[provider]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        client = (
            openai.AsyncOpenAI(api_key=api_key, timeout=120.0)
            if async_client
            else openai.OpenAI(api_key=api_key, timeout=120.0)
        )
        return client, resolved_model

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        client = (
            anthropic.AsyncAnthropic(api_key=api_key)
            if async_client
            else anthropic.Anthropic(api_key=api_key)
        )
        return client, resolved_model

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(resolved_model), resolved_model

    if provider == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")
        client = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        return client, resolved_model

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        client = (
            openai.AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            if async_client
            else openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        )
        return client, resolved_model


def generate_content(
    client: Any,
    api_provider: str,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """Generate content synchronously using the provided client.

    Args:
        client: Initialised API client from init_api_client.
        api_provider: Provider name (used for dispatch).
        model_name: Model identifier.
        prompt: User message.
        system_prompt: Optional system message.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.

    Returns:
        Generated text as a string.
    """
    provider = api_provider.lower()
    user_messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

    if provider in ("openai", "together", "openrouter"):
        messages = (
            [{"role": "system", "content": system_prompt}] + user_messages
            if system_prompt
            else user_messages
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        kwargs: Dict[str, Any] = dict(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        return response.content[0].text

    if provider == "gemini":
        response = client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return response.text

    raise ValueError(f"Unknown provider: {api_provider}")


async def generate_content_async(
    client: Any,
    api_provider: str,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """Async version of generate_content. Uses async clients where available."""
    provider = api_provider.lower()
    user_messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

    if provider in ("openai", "together", "openrouter"):
        messages = (
            [{"role": "system", "content": system_prompt}] + user_messages
            if system_prompt
            else user_messages
        )
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        kwargs: Dict[str, Any] = dict(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        response = await client.messages.create(**kwargs)
        return response.content[0].text

    if provider == "gemini":
        # google-generativeai doesn't have a true async path; run sync
        response = client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return response.text

    raise ValueError(f"Unknown provider: {api_provider}")


def parse_json_response(text: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from an API response.

    Returns the parsed dict, or None if parsing fails.
    """
    text = text.strip()

    # Strip markdown code fences
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object by braces
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError:
            pass

    if debug:
        print(f"[parse_json_response] Failed on: {text[:200]!r}")
    return None
