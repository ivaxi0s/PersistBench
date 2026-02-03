"""Configuration loading and validation."""

import json
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt


FAILURE_TYPE_CROSS_DOMAIN = "cross_domain"
FAILURE_TYPE_SYCOPHANCY = "sycophancy"
FAILURE_TYPE_BENEFICIAL_MEMORY = "beneficial_memory_usage"

VALID_FAILURE_TYPES = {
    FAILURE_TYPE_CROSS_DOMAIN,
    FAILURE_TYPE_SYCOPHANCY,
    FAILURE_TYPE_BENEFICIAL_MEMORY,
}

# Per-category generation defaults when no global override is set
DEFAULT_GENERATIONS_BY_FAILURE_TYPE = {
    FAILURE_TYPE_CROSS_DOMAIN: 3,
    FAILURE_TYPE_SYCOPHANCY: 3,
    FAILURE_TYPE_BENEFICIAL_MEMORY: 1,
}


def get_generations_for_failure_type(
    failure_type: str,
    generations_override: int | None = None,
) -> int:
    """Resolve generation count for a failure type.

    If generations_override is set, uses that for all types.
    Otherwise uses per-category defaults (3 for cross_domain/sycophancy, 1 for beneficial).
    """
    if generations_override is not None:
        return generations_override
    return DEFAULT_GENERATIONS_BY_FAILURE_TYPE.get(failure_type, 3)

# Backwards compatibility aliases
_LEGACY_ALIASES = {
    "positive_memory_usage": FAILURE_TYPE_BENEFICIAL_MEMORY,
    "leakage_type": "failure_type",
}

JUDGE_MODEL = "moonshotai/kimi-k2-thinking-maas"  # Vertex AI model name
JUDGE_MODEL_OPENROUTER = "moonshotai/kimi-k2-thinking"  # OpenRouter model name
JUDGE_LOCATION = "global"
JUDGE_TEMPERATURE = 0.0

VALID_JUDGE_PROVIDERS = {"vertexai", "openrouter"}


class ModelEntry(BaseModel):
    """Individual model entry with API parameters."""

    name: str
    provider: str = "openrouter"
    mode: str | None = "sequential"
    api_params: dict[str, Any] | None = None
    base_url: str | None = None
    api_key_env: str | None = None


JUDGE_MODEL_ENTRY_OPENROUTER = ModelEntry(
    name=JUDGE_MODEL_OPENROUTER,
    api_params={
        "temperature": JUDGE_TEMPERATURE,
        "provider": {"order": ["google-vertex"], "allow_fallbacks": False},
        "reasoning": {"enabled": True, "effort": "high"},
    },
)


def resolve_entry_configuration(entry: dict[str, Any]) -> str:
    """Resolve failure type with defaults. Accepts legacy 'leakage_type' field."""
    failure_type = entry.get("failure_type") or entry.get(
        "leakage_type", FAILURE_TYPE_CROSS_DOMAIN
    )
    return _LEGACY_ALIASES.get(failure_type, failure_type)


def validate_failure_type(failure_type: str) -> None:
    """Validate that failure_type is supported."""
    if failure_type not in VALID_FAILURE_TYPES:
        raise ValueError(
            f"Invalid failure_type={failure_type}. Valid values: {sorted(VALID_FAILURE_TYPES)}"
        )


class BenchmarkConfig(BaseModel):
    """Benchmark configuration schema."""

    models: list[ModelEntry] = Field(min_length=1)
    judge: ModelEntry | None = None
    judge_provider: str | None = None
    input: Path
    output: Path
    store_raw_api_responses: bool = False
    generations: PositiveInt | None = None
    concurrency: PositiveInt = 1
    limit: PositiveInt | None = None
    batch_poll_timeout_minutes: PositiveInt = 25
    prompt_template: Path | None = None

    # Loaded template content (not part of JSON schema)
    prompt_template_content: str | None = None


def load_benchmark_config_data(
    data: dict[str, Any], config_path: str | Path | None = None
) -> BenchmarkConfig:
    """Load and validate config from a parsed dict."""
    if "no_memory_baseline" in data:
        raise ValueError(
            f"Config field 'no_memory_baseline' is no longer supported ({config_path}). "
            "Single-response evaluation always generates with the provided memories."
        )

    if data.get("judge") is not None:
        warning_msg = f"WARNING: The 'judge' field in configs ({config_path}) is deprecated and will be ignored."
        print(f"\n{warning_msg}\n")
        warnings.warn(
            warning_msg,
            DeprecationWarning,
            stacklevel=2,
        )
        del data["judge"]

    config = BenchmarkConfig(**data)

    # Validate unique model names (results are keyed by name in checkpoint)
    model_names = [m.name for m in config.models]
    seen: set[str] = set()
    for name in model_names:
        if name in seen:
            raise ValueError(
                f"Duplicate model name '{name}' in config. "
                f"Each model must have a unique name since results are keyed by model name."
            )
        seen.add(name)

    # Validate judge_provider if specified
    if (
        config.judge_provider is not None
        and config.judge_provider not in VALID_JUDGE_PROVIDERS
    ):
        raise ValueError(
            f"Invalid judge_provider '{config.judge_provider}' in config ({config_path}). "
            f"Valid values: {sorted(VALID_JUDGE_PROVIDERS)}"
        )

    # Load prompt template content if specified and not already provided (e.g. from checkpoint)
    if config.prompt_template and not config.prompt_template_content:
        template_path = Path(config.prompt_template)
        if not template_path.exists():
            raise ValueError(
                f"Prompt template file {config.prompt_template} does not exist"
            )
        with open(template_path, "r", encoding="utf-8") as f:
            config.prompt_template_content = f.read()

    if (
        config.prompt_template_content
        and "{memories}" not in config.prompt_template_content
    ):
        raise ValueError(
            "Prompt template must contain the {memories} placeholder. "
            "Without it, the model receives no user memories and evaluation is meaningless."
        )

    return config


def load_benchmark_config(config_path: str | Path) -> BenchmarkConfig:
    """Load and validate config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file {config_path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} must contain a JSON object at top-level"
        )
    return load_benchmark_config_data(data, config_path=path)
