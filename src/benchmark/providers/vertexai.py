"""Vertex AI provider for sequential inference."""

from benchmark.config import ModelEntry
from benchmark.protocols import GenerateResult
from benchmark.utils import (
    api_retry,
    get_vertex_ai_client,
    openai_compat_generate,
    strip_reasoning_tags,
)


@api_retry()
async def vertexai_generate(
    model: ModelEntry,
    system_prompt: str,
    user_message: str,
) -> GenerateResult:
    """Generate response using Vertex AI via OpenAI-compatible API.

    The model location can be specified in api_params using the "location" key.
    """
    location = None
    if model.api_params and "location" in model.api_params:
        location = model.api_params["location"]
        model = model.model_copy(
            update={
                "api_params": {
                    k: v for k, v in model.api_params.items() if k != "location"
                }
            }
        )

    async with get_vertex_ai_client(location) as client:
        result = await openai_compat_generate(
            client, model, system_prompt, user_message
        )

    cleaned, reasoning = strip_reasoning_tags(result["response"])
    if reasoning is not None:
        result["raw_api_response"]["extracted_reasoning_content"] = reasoning
        result["response"] = cleaned

    return result
