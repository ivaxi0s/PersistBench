"""Provider registry and implementations."""

from collections.abc import MutableMapping
from typing import TypedDict

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError
from benchmark.providers.openrouter import openrouter_generate_response
from benchmark.providers import (
    AnthropicBatchProvider,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    anthropic_generate,
    gemini_generate,
    openai_compatible_generate,
    openai_generate,
    vertexai_generate,
)
from benchmark.protocols import (
    BatchGenerateFn,
    GenerateFn,
)


class ProviderConfig(TypedDict):
    """Configuration for a provider.

    Attributes:
        generate_fn: Function for sequential/real-time generation (optional).
        batch_provider_class: Class implementing BatchGenerateFn protocol (optional).
            Must have async methods: submit(work_items) -> BatchSubmitResult
                                     poll(job_info) -> BatchPollResult
    """

    generate_fn: GenerateFn | None
    batch_provider_class: type[BatchGenerateFn] | None


# Cache for lazily initialized provider instances
ProviderInstanceCache = MutableMapping[str, BatchGenerateFn]
_provider_instances: ProviderInstanceCache = {}

VALID_PROVIDER_MODES = {"sequential", "batch"}


def get_provider_config(provider_name: str) -> ProviderConfig:
    """Return provider config or raise a FatalBenchmarkError for unknown providers."""
    provider_config = PROVIDERS.get(provider_name)
    if provider_config is None:
        raise FatalBenchmarkError(f"Unknown provider: {provider_name}")
    return provider_config


def resolve_model_generation_mode(model: ModelEntry) -> str:
    """Resolve generation mode for a model and validate provider support."""
    mode = model.mode or "sequential"
    if mode not in VALID_PROVIDER_MODES:
        raise FatalBenchmarkError(
            f"Invalid mode '{mode}' for model '{model.name}'. Expected one of {sorted(VALID_PROVIDER_MODES)}."
        )

    provider_config = get_provider_config(model.provider)

    if mode == "batch" and provider_config.get("batch_provider_class") is None:
        raise FatalBenchmarkError(
            f"Model '{model.name}' requested batch mode but provider '{model.provider}' has no batch executor."
        )
    if mode == "sequential" and provider_config.get("generate_fn") is None:
        raise FatalBenchmarkError(
            f"Model '{model.name}' requested sequential mode but provider '{model.provider}' has no sequential executor."
        )

    return mode


def get_batch_provider(provider_name: str) -> BatchGenerateFn:
    """Get or create a batch provider instance (lazy initialization).

    Args:
        provider_name: Name of the provider (e.g., 'anthropic', 'openai', 'vertexai_gemini')

    Returns:
        BatchGenerateFn: Provider instance implementing the batch generation protocol.
            The instance has async methods:
            - submit(work_items: list[BatchWorkItem]) -> BatchSubmitResult
            - poll(job_info: BatchJobInfo) -> BatchPollResult

    Raises:
        ValueError: If provider is unknown or doesn't support batch generation.
    """
    if provider_name not in _provider_instances:
        provider_config = PROVIDERS.get(provider_name)
        if not provider_config:
            raise ValueError(f"Unknown provider: {provider_name}")

        provider_class = provider_config.get("batch_provider_class")
        if not provider_class:
            raise ValueError(
                f"Provider {provider_name} does not support batch generation"
            )

        _provider_instances[provider_name] = provider_class()

    return _provider_instances[provider_name]


PROVIDERS: dict[str, ProviderConfig] = {
    # default provider, simple to use and supports all of the benchmark models
    "openrouter": {
        "generate_fn": openrouter_generate_response,
        "batch_provider_class": None,
    },
    # openai provider supporting both sequential and batch generation (50% discount on batch)
    "openai": {
        "generate_fn": openai_generate,
        "batch_provider_class": OpenAIBatchProvider,
    },
    # generic provider for any OpenAI-compatible API endpoint
    # requires base_url in model config, api_key_env defaults to OPENAI_API_KEY
    "openai_compatible": {
        "generate_fn": openai_compatible_generate,
        "batch_provider_class": None,
    },
    # anthropic provider supporting both sequential and batch generation (50% discount on batch)
    "anthropic": {
        "generate_fn": anthropic_generate,
        "batch_provider_class": AnthropicBatchProvider,
    },
    # vertexai_oss provider supporting sequential generation only
    # Uses OpenAI-compatible API for Gemini and Model Garden open source models
    # Requires service account credentials with appropriate permissions
    # Location can be specified in api_params using the "location" key
    "vertexai_oss": {
        "generate_fn": vertexai_generate,
        "batch_provider_class": None,
    },
    # gemini provider using Google AI Studio / Gemini API directly (not Vertex AI)
    # Supports both sequential and batch generation (50% discount on batch)
    # Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable
    # Simpler setup than vertexai - no GCP project or service account needed
    "gemini": {
        "generate_fn": gemini_generate,
        "batch_provider_class": GeminiBatchProvider,
    },
}
