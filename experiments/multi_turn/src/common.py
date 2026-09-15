"""Shared helpers for the multi-turn conversation pipeline.

All scripts in this folder resolve paths relative to the experiment root
(``experiments/multi_turn/``), so they work from any working directory.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

DEFAULT_DATASET = DATA_DIR / "samples_150.json"
DEFAULT_EVAL_PROMPT = PROMPTS_DIR / "eval_prompt.txt"
DEFAULT_GATEKEEPER_PROMPT = PROMPTS_DIR / "gate_llm.txt"
DEFAULT_CHAT_PROMPT = PROMPTS_DIR / "chat_prompt.txt"
DEFAULT_ADAPTER_PROMPT = PROMPTS_DIR / "adapter.txt"
DEFAULT_SYCOPHANCY_CHAT_PROMPT = PROMPTS_DIR / "sycophancy_chat_prompt.txt"
DEFAULT_SYCOPHANCY_ADAPTER_PROMPT = PROMPTS_DIR / "adapter_sycophancy.txt"


def format_memories(memory_list: list) -> str:
    """Convert the JSON list of memories into a bulleted string block."""
    return "\n".join(f"- {m}" for m in memory_list)


def clean_json(raw_text: str) -> str:
    """Strip markdown fences so gatekeeper output parses as JSON."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def completion_text(resp) -> str:
    """Extract text from an OpenAI-style chat completion (content may be None)."""
    msg = resp.choices[0].message
    raw = getattr(msg, "content", None)
    if raw is None:
        return ""
    return raw.strip() if isinstance(raw, str) else str(raw).strip()


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def prompts_for_failure_type(failure_type: str) -> tuple[Path, Path]:
    """Return (chat_prompt_path, adapter_prompt_path) for a failure type."""
    if failure_type == "sycophancy":
        return (DEFAULT_SYCOPHANCY_CHAT_PROMPT, DEFAULT_SYCOPHANCY_ADAPTER_PROMPT)
    if failure_type in ("beneficial_memory_usage", "cross_domain"):
        return (DEFAULT_CHAT_PROMPT, DEFAULT_ADAPTER_PROMPT)
    raise ValueError(
        f"Unsupported failure_type {failure_type!r}; "
        "expected beneficial_memory_usage, cross_domain, or sycophancy"
    )
