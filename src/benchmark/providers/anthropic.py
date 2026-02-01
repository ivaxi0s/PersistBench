"""Anthropic batch provider for cost-effective batch inference."""

import os
import time
from typing import Any

import anthropic

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError, NonRetryableError
from benchmark.types import GenerateResult
from benchmark.utils import api_retry
from benchmark.protocols import (
    BatchCancelResult,
    BatchJobInfo,
    BatchPollResult,
    BatchResult,
    BatchStatus,
    BatchSubmitResult,
    BatchWorkItem,
)

ANTHROPIC_SUCCESS_STOP_REASONS = {"end_turn", "max_tokens", "stop_sequence"}


class AnthropicBatchProvider:
    """Anthropic batch provider implementing BatchGenerateFn protocol.

    Supports batch generation using Anthropic Message Batches API.
    Automatically handles:
    - Conversion to/from Anthropic batch format
    - Job submission and polling
    - Partial result handling
    - Result download and parsing
    """

    def __init__(self, api_key: str | None = None):
        """Initialize Anthropic batch provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens per response
        """
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key is None:
            raise FatalBenchmarkError("ANTHROPIC_API_KEY required")
        self.client = anthropic.AsyncAnthropic(api_key=key)

    @staticmethod
    def _build_request(item: BatchWorkItem) -> dict[str, Any]:
        """Convert a BatchWorkItem to Anthropic batch request format."""
        params = {
            "model": item["model"].name,
            "max_tokens": 4096,
            "system": item["system_prompt"],
            "messages": [{"role": "user", "content": item["user_message"]}],
        }
        if item["model"].api_params:
            for key, value in item["model"].api_params.items():
                params.setdefault(key, value)

        return {"custom_id": item["request_id"], "params": params}

    @staticmethod
    def _extract_text_blocks(message: dict[str, Any]) -> str:
        content = message.get("content") or []
        text_segments = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_segments)

    def _build_result_payload(
        self,
        custom_id: str,
        raw_response: dict[str, Any],
        response_text: str | None,
        error: str | None,
    ) -> BatchResult:
        generation: GenerateResult | None = None
        if error is None and response_text is not None:
            generation = {
                "response": response_text,
                "raw_api_response": raw_response,
            }

        return {
            "request_id": custom_id,
            "raw_api_response": raw_response,
            "error": error,
            "generation": generation,
            "judge": None,
        }

    def _convert_from_anthropic_format(
        self,
        anthropic_results: list[dict[str, Any]],
    ) -> list[BatchResult]:
        """Convert Anthropic batch results to BatchResult format.

        Anthropic result format:
        {
            "custom_id": "request-1",
            "result": {
                "type": "succeeded" or "errored",
                "message": {...} (if succeeded),
                "error": {...} (if errored)
            }
        }
        """
        batch_results: list[BatchResult] = []

        for result in anthropic_results:
            raw_result = result.get("result", {}) or {}
            request_id = result.get("custom_id", "")
            result_type = raw_result.get("type")

            error: str | None = None
            response_text: str | None = None

            if result_type == "succeeded":
                message = raw_result.get("message", {})
                stop_reason = message.get("stop_reason")

                # Check for unsuccessful stop_reason
                if stop_reason and stop_reason not in ANTHROPIC_SUCCESS_STOP_REASONS:
                    error = f"Unsuccessful stop_reason: {stop_reason}"
                else:
                    response_text = self._extract_text_blocks(message)
            elif result_type == "errored":
                error_info = raw_result.get("error", {}) or {}
                error = (
                    f"{error_info.get('type', 'Unknown error type')}: "
                    f"{error_info.get('message', 'No error message')}"
                )
            elif result_type == "canceled":
                error = "Request was canceled before processing completed"
            elif result_type == "expired":
                error = "Request expired before processing could begin"
            else:
                error = f"Unknown result type: {result_type}"

            batch_results.append(
                self._build_result_payload(
                    custom_id=request_id,
                    raw_response=raw_result,
                    response_text=response_text,
                    error=error,
                )
            )

        return batch_results

    async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
        """Submit a batch generation job to Anthropic.

        Args:
            work_items: List of generation requests

        Returns:
            BatchSubmitResult with job info and count
        """
        if not work_items:
            raise ValueError("work_items cannot be empty")

        requests = [self._build_request(item) for item in work_items]
        batch = await self.client.messages.batches.create(requests=requests)  # type: ignore[arg-type]

        job_info: BatchJobInfo = {
            "job_id": batch.id,
            "provider": "anthropic",
            "status": "submitted",
            "model_name": work_items[0]["model"].name,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {
                "processing_status": batch.processing_status,
            },
        }

        return {"job_info": job_info, "submitted_count": len(work_items)}

    async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
        """Poll an Anthropic batch job for completion.

        Args:
            job_info: Job metadata from submit()

        Returns:
            BatchPollResult with status and results (if complete)
        """
        batch = await self.client.messages.batches.retrieve(job_info["job_id"])

        # Possible processing statuses: in_progress, canceling, ended
        if batch.processing_status in {"in_progress", "canceling"}:
            total_processed = 0
            if batch.request_counts:
                total_processed = (
                    batch.request_counts.succeeded
                    + batch.request_counts.errored
                    + batch.request_counts.canceled
                    + batch.request_counts.expired
                )
            return {
                "status": BatchStatus.RUNNING,
                "completed_count": total_processed if total_processed > 0 else None,
                "results": None,
            }

        if batch.processing_status == "ended":
            results = await self._fetch_batch_results(batch.id)

            return {
                "status": BatchStatus.COMPLETED,
                "completed_count": len(results),
                "results": results,
            }

        return {
            "status": BatchStatus.FAILED,
            "completed_count": None,
            "results": None,
        }

    async def _fetch_batch_results(self, batch_id: str) -> list[BatchResult]:
        """Fetch all results from a completed batch."""
        all_results = []

        results_stream = await self.client.messages.batches.results(batch_id)
        async for result in results_stream:
            all_results.append(result.model_dump(mode="json"))

        return self._convert_from_anthropic_format(all_results)

    async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
        """Cancel an active Anthropic batch job.

        Args:
            job_info: Job metadata from submit()

        Returns:
            BatchCancelResult with success status and message
        """
        job_id = job_info["job_id"]
        try:
            batch = await self.client.messages.batches.cancel(job_id)
            return {
                "success": True,
                "message": f"Batch {job_id} cancellation initiated (status: {batch.processing_status})",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel batch {job_id}: {e}",
            }


@api_retry()
async def anthropic_generate(
    model: ModelEntry,
    system_prompt: str,
    user_message: str,
) -> GenerateResult:
    """Generate response using Anthropic Messages API with streaming.

    Args:
        model: Model configuration with name and api_params
        system_prompt: System prompt with context
        user_message: User query

    Returns:
        GenerateResult with response text and raw API response
    """
    async with anthropic.AsyncAnthropic() as client:
        params = {
            "model": model.name,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        if model.api_params:
            for key, value in model.api_params.items():
                params.setdefault(key, value)

        async with client.messages.stream(**params) as stream:  # type: ignore[arg-type]
            final_message = await stream.get_final_message()

        if final_message.stop_reason == "refusal":
            raise NonRetryableError(
                f"Model refused: stop_reason={final_message.stop_reason}"
            )

        if final_message.stop_reason not in ANTHROPIC_SUCCESS_STOP_REASONS:
            raise RuntimeError(f"Unsuccessful stop_reason: {final_message.stop_reason}")

        text_response = "".join(
            getattr(block, "text", "")
            for block in final_message.content
            if block.type == "text"
        )

        return {
            "response": text_response,
            "raw_api_response": final_message.model_dump(mode="json"),
        }
