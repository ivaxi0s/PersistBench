"""Anthropic Messages API for Claude targets + cached system prompt (ephemeral)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

# Official Anthropic model IDs look like claude-sonnet-4-5-20250929 (no slash).
# OpenRouter uses anthropic/claude-... — those keep using OpenRouter.
def uses_anthropic_direct_target(model: str) -> bool:
    if "/" in model:
        return False
    if not model.startswith("claude-"):
        return False
    block = os.getenv("ANTHROPIC_DIRECT_TARGET_DISABLE", "").strip().lower()
    if block in ("1", "true", "yes"):
        return False
    return True


def anthropic_target_client():
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Claude direct targets")
    return anthropic.Anthropic(api_key=key)


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                parts.append(c)
        else:
            rest.append(m)
    text = "\n\n".join(parts).strip()
    return text, rest


def messages_create_with_cached_system(
    client: Any,
    *,
    model: str,
    openai_style_messages: list[dict[str, Any]],
    temperature: float,
) -> Any:
    """
    Maps OpenAI-style chat (system + user/assistant) to Anthropic Messages API.
    Puts the full system string in one text block with cache_control ephemeral so
    the stable system prefix can be cached across turns for the same conversation.
    """
    max_tokens = int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "16384"))
    sys_text, msgs = _split_system(openai_style_messages)
    if not sys_text:
        raise ValueError("Claude target requires a system message in the conversation")

    system_blocks = [
        {
            "type": "text",
            "text": sys_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    anth_msgs: list[dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content", "")
        if not isinstance(c, str):
            c = str(c)
        anth_msgs.append({"role": role, "content": c})

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": anth_msgs,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    raw = client.messages.create(**kwargs)

    text_parts: list[str] = []
    for block in raw.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
    content = "".join(text_parts).strip()

    u = raw.usage
    usage = SimpleNamespace(
        prompt_tokens=getattr(u, "input_tokens", None),
        completion_tokens=getattr(u, "output_tokens", None),
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", None),
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", None),
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice], usage=usage, _anthropic_message=raw)
