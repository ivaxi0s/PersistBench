"""OpenRouter API client using OpenAI SDK."""

import os

from openai import AsyncOpenAI

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError
from benchmark.types import GenerateResult
from benchmark.utils import api_retry, openai_compat_generate


@api_retry()
async def openrouter_generate_response(
    model: ModelEntry, system_prompt: str, user_message: str
) -> GenerateResult:
    """Generate model response via OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise FatalBenchmarkError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please set it before running the benchmark."
        )

    async with AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    ) as client:
        return await openai_compat_generate(client, model, system_prompt, user_message)
