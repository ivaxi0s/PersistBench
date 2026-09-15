"""OpenAI API direct targets (not OpenRouter) + optional prompt caching on target calls."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

# Use api.openai.com with OPENAI_API_KEY. Add more snapshot IDs as needed.
OPENAI_DIRECT_TARGET_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.2-2025-12-11",
    }
)


def extra_openai_target_models_from_env() -> frozenset[str]:
    raw = os.getenv("OPENAI_TARGET_MODELS_EXTRA", "").strip()
    if not raw:
        return frozenset()
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return frozenset(parts)


def uses_openai_direct_target(model: str) -> bool:
    return model in (OPENAI_DIRECT_TARGET_MODELS | extra_openai_target_models_from_env())


def openai_target_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for this target model")
    return OpenAI(api_key=key)


def chat_completion_target(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    prompt_cache_key: str | None,
):
    """
    Target-only chat completion. Optional prompt_cache_key improves cache routing for
    repeated prefixes (same conversation across turns). Retention is API default
    (in-memory; typically ~5–60 min idle), not extended 24h storage.

    Per OpenAI, cached vs uncached *input* tokens are priced differently; extended
    vs in-memory *retention* is the same price—24h is not a surcharge, but it
    changes data-handling / ZDR eligibility, so we do not set it here.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    try:
        return client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("prompt_cache_key", None)
        return client.chat.completions.create(**kwargs)


def usage_cached_tokens(resp) -> int | None:
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    d = getattr(u, "prompt_tokens_details", None)
    if d is None:
        return None
    if isinstance(d, dict):
        return d.get("cached_tokens")
    return getattr(d, "cached_tokens", None)
