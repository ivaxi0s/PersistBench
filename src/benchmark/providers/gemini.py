"""Gemini API provider for batch and sequential inference."""

import atexit
import io
import json
import os
import time
from typing import Any

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError, NonRetryableError
from benchmark.utils import api_retry, parse_jsonl
from benchmark.protocols import (
    BatchCancelResult,
    BatchJobInfo,
    BatchPollResult,
    BatchResult,
    BatchStatus,
    BatchSubmitResult,
    BatchWorkItem,
)
from benchmark.types import GenerateResult
from google import genai
from google.genai import types
from google.genai.types import JobState

GEMINI_BATCH_LOG_PREFIX = "[Gemini Batch]"
GEMINI_SUCCESS_FINISH_REASONS = {"STOP", "MAX_TOKENS"}

# Module-level client for connection reuse, with proper cleanup at exit
_shared_client: genai.Client | None = None


def _get_shared_client() -> genai.Client:
    """Get or create a shared Gemini client with connection reuse."""
    global _shared_client
    if _shared_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise FatalBenchmarkError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        _shared_client = genai.Client(api_key=api_key)
        atexit.register(_cleanup_shared_client)
    return _shared_client


def _cleanup_shared_client() -> None:
    """Close the shared client at process exit."""
    global _shared_client
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None


def _parse_gemini_response(response: Any) -> tuple[str | None, str | None]:
    """Parse Gemini response and extract text or error, filtering out thought content.

    Returns:
        Tuple of (error, text). One will be None.
    """
    if hasattr(response, "prompt_feedback") and response.prompt_feedback:
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
        if block_reason:
            return f"Prompt blocked: {block_reason}", None

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return "No candidates in response", None

    candidate = candidates[0]

    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason:
        reason_str = finish_reason.name if hasattr(finish_reason, "name") else "unknown"
        if reason_str not in GEMINI_SUCCESS_FINISH_REASONS:
            return f"Unsuccessful finish_reason: {reason_str}", None

    content = getattr(candidate, "content", None)
    if not content:
        return "No content in candidate", None

    parts = getattr(content, "parts", None)
    if not parts:
        return "No parts in content", None

    # Filter out thought parts, only keep non-thought text
    non_thought_texts = []
    for part in parts:
        is_thought = getattr(part, "thought", False)
        if not is_thought:
            text = getattr(part, "text", None)
            if text is not None:
                non_thought_texts.append(text)

    if not non_thought_texts:
        return "No non-thought text in response", None

    return None, "".join(non_thought_texts)


class GeminiBatchProvider:
    """Gemini API batch provider using JSONL file-based requests."""

    def __init__(self, api_key: str | None = None):
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = _get_shared_client()

    @staticmethod
    def _build_jsonl_request(item: BatchWorkItem) -> dict[str, Any]:
        """Convert BatchWorkItem to JSONL request format with key tracking."""
        request: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": item["user_message"]}]}],
        }
        if item["system_prompt"]:
            request["system_instruction"] = {"parts": [{"text": item["system_prompt"]}]}

        if item["model"].api_params:
            request["generation_config"] = dict(item["model"].api_params)

        return {"key": item["request_id"], "request": request}

    async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
        """Submit batch job to Gemini API using JSONL file."""
        # Build JSONL content
        jsonl_lines = [
            json.dumps(self._build_jsonl_request(item)) for item in work_items
        ]
        jsonl_content = "\n".join(jsonl_lines)

        # Upload the JSONL file
        jsonl_buffer = io.BytesIO(jsonl_content.encode("utf-8"))
        uploaded_file = self.client.files.upload(
            file=jsonl_buffer,
            config=types.UploadFileConfig(
                display_name=f"benchmark-batch-{int(time.time())}.jsonl",
                mime_type="application/jsonl",
            ),
        )

        assert uploaded_file.name is not None
        print(f"{GEMINI_BATCH_LOG_PREFIX} Uploaded input file: {uploaded_file.name}")

        # Create batch job referencing the uploaded file
        model_name = work_items[0]["model"].name
        batch_job = self.client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={"display_name": f"gemini-benchmark-batch-{int(time.time())}"},
        )

        assert batch_job.name is not None
        print(f"{GEMINI_BATCH_LOG_PREFIX} Created job: {batch_job.name}")

        return {
            "job_info": {
                "job_id": batch_job.name,
                "provider": "gemini",
                "status": "submitted",
                "model_name": model_name,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metadata": {"input_file": uploaded_file.name},
            },
            "submitted_count": len(work_items),
        }

    async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
        """Poll batch job for completion."""
        batch_job = self.client.batches.get(name=job_info["job_id"])
        state = batch_job.state

        if state in (JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED):
            return {
                "status": BatchStatus.FAILED,
                "completed_count": None,
                "results": None,
            }

        if state != JobState.JOB_STATE_SUCCEEDED:
            return {
                "status": BatchStatus.RUNNING,
                "completed_count": None,
                "results": None,
            }

        # Download and parse output file
        output_file_name = batch_job.dest.file_name if batch_job.dest else None
        if not output_file_name:
            return {
                "status": BatchStatus.FAILED,
                "completed_count": None,
                "results": None,
            }

        output_bytes = self.client.files.download(file=output_file_name)
        output_content = output_bytes.decode("utf-8")

        results: list[BatchResult] = []
        for data in parse_jsonl(output_content):
            request_id = data.get("key", "unknown")
            response_data = data.get("response", {})

            # Parse the response
            error, text = _parse_gemini_response_dict(response_data)

            generation: GenerateResult | None = None
            if error is None and text is not None:
                generation = {"response": text, "raw_api_response": response_data}

            results.append(
                {
                    "request_id": request_id,
                    "error": error,
                    "raw_api_response": response_data,
                    "generation": generation,
                    "judge": None,
                }
            )

        return {
            "status": BatchStatus.COMPLETED,
            "completed_count": len(results),
            "results": results,
        }

    async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
        """Cancel an active Gemini batch job.

        Args:
            job_info: Job metadata from submit()

        Returns:
            BatchCancelResult with success status and message
        """
        job_id = job_info["job_id"]
        try:
            self.client.batches.delete(name=job_id)
            return {
                "success": True,
                "message": f"Batch {job_id} deleted/cancelled successfully",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel batch {job_id}: {e}",
            }


def _parse_gemini_response_dict(
    response: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Parse Gemini response dict (from JSONL) and extract text or error, filtering out thought content."""
    prompt_feedback = response.get("promptFeedback")
    if prompt_feedback:
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            return f"Prompt blocked: {block_reason}", None

    candidates = response.get("candidates")
    if not candidates:
        return "No candidates in response", None

    candidate = candidates[0]

    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason not in GEMINI_SUCCESS_FINISH_REASONS:
        return f"Unsuccessful finish_reason: {finish_reason}", None

    content = candidate.get("content")
    if not content:
        return "No content in candidate", None

    parts = content.get("parts")
    if not parts:
        return "No parts in content", None

    # Filter out thought parts, only keep non-thought text
    non_thought_texts = []
    for part in parts:
        is_thought = part.get("thought", False)
        if not is_thought:
            text = part.get("text")
            if text is not None:
                non_thought_texts.append(text)

    if not non_thought_texts:
        return "No non-thought text in response", None

    return None, "".join(non_thought_texts)


@api_retry()
async def gemini_generate(
    model: ModelEntry,
    system_prompt: str,
    user_message: str,
) -> GenerateResult:
    """Generate response using Gemini API."""
    client = _get_shared_client()
    config_kwargs: dict[str, Any] = dict(model.api_params or {})
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    response = await client.aio.models.generate_content(
        model=model.name,
        contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )

    error, text = _parse_gemini_response(response)
    if error and error.startswith("Prompt blocked"):
        raise NonRetryableError(error)
    if error:
        raise RuntimeError(error)

    assert text is not None  # Guaranteed by error handling above
    return {
        "response": text,
        "raw_api_response": response.model_dump(mode="json"),
    }
