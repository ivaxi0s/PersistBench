"""Provider implementations for batch and sequential generation.

This package contains provider-specific implementations for:
- Anthropic (Claude models)
- OpenAI (GPT models)
- OpenRouter (multi-provider routing)
- Vertex AI (Gemini models)

Each provider module implements the BatchGenerateFn protocol and/or
GenerateFn protocol for integration with the benchmark runner.
"""

from benchmark.config import JUDGE_TEMPERATURE
from benchmark.providers.anthropic import (
    AnthropicBatchProvider,
    anthropic_generate,
)
from benchmark.providers.gemini import GeminiBatchProvider, gemini_generate
from benchmark.types import (
    GenerateResult,
    GenerationEntry,
    JudgeResult,
)
from benchmark.utils import (
    JUDGE_RESPONSE_SCHEMA,
    extract_json_from_response,
)
from benchmark.providers.openai import OpenAIBatchProvider, openai_generate
from benchmark.providers.openai_compatible import openai_compatible_generate
from benchmark.providers.openrouter import openrouter_generate_response
from benchmark.providers.vertexai import vertexai_generate

__all__ = [
    "AnthropicBatchProvider",
    "anthropic_generate",
    "extract_json_from_response",
    "GeminiBatchProvider",
    "gemini_generate",
    "GenerateResult",
    "GenerationEntry",
    "JudgeResult",
    "JUDGE_RESPONSE_SCHEMA",
    "JUDGE_TEMPERATURE",
    "OpenAIBatchProvider",
    "openai_compatible_generate",
    "openai_generate",
    "openrouter_generate_response",
    "vertexai_generate",
]
