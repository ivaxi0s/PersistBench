"""OpenAI provider for batch and sequential inference."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError
from benchmark.types import GenerateResult
from benchmark.utils import api_retry, openai_compat_generate, parse_jsonl
from benchmark.protocols import (
    BatchCancelResult,
    BatchJobInfo,
    BatchPollResult,
    BatchResult,
    BatchStatus,
    BatchSubmitResult,
    BatchWorkItem,
)

OPENAI_SUCCESS_FINISH_REASONS = {"stop", "length"}
OPENAI_BATCH_LOG_PREFIX = "[OpenAI Batch]"


class OpenAIBatchProvider:
    """OpenAI batch provider implementing BatchGenerateFn protocol."""

    def __init__(
        self,
        api_key: str | None = None,
    ):
        """Initialize OpenAI batch provider."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise FatalBenchmarkError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=self.api_key)

    @staticmethod
    def _build_request(item: BatchWorkItem) -> dict[str, Any]:
        """Convert a BatchWorkItem to OpenAI batch request format."""
        body: dict[str, Any] = {
            "model": item["model"].name,
            "messages": [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_message"]},
            ],
        }
        if item["model"].api_params:
            for key, value in item["model"].api_params.items():
                body.setdefault(key, value)

        return {
            "custom_id": item["request_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    @staticmethod
    def _decode_json(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value or {}

    @staticmethod
    def _parse_openai_result(
        result: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any], str | None, dict[str, Any]]:
        """Extract error info, raw payload, and response text from OpenAI response."""
        response_data = result.get("response") or {}
        error_data = result.get("error")
        top_level_raw = response_data or error_data or {}

        # Case 1: Top-level error (e.g., batch_expired)
        if error_data:
            code = error_data.get("code", "unknown")
            msg = error_data.get("message", "No error message")
            return f"{code}: {msg}", {}, None, top_level_raw

        # Case 2: HTTP error
        if response_data.get("status_code") != 200:
            body = OpenAIBatchProvider._decode_json(response_data.get("body"))
            err = body.get("error") or {}
            parts = [f"HTTP {response_data.get('status_code')}"]
            if err.get("code"):
                parts.append(f"code={err['code']}")
            if err.get("type"):
                parts.append(f"type={err['type']}")
            parts.append(err.get("message", "Unknown error"))
            return ": ".join(parts), {}, None, top_level_raw

        # Case 3: Success - parse choices
        body = OpenAIBatchProvider._decode_json(response_data.get("body"))
        choices = body.get("choices", [])
        if not choices:
            return "No choices in response", {}, None, top_level_raw

        choice = choices[0]
        message = choice.get("message") or {}

        if message.get("refusal"):
            return f"Model refused: {message['refusal']}", body, None, top_level_raw

        finish_reason = choice.get("finish_reason")
        if finish_reason and finish_reason not in OPENAI_SUCCESS_FINISH_REASONS:
            return (
                f"Unsuccessful finish_reason: {finish_reason}",
                body,
                None,
                top_level_raw,
            )

        return None, body, message.get("content", ""), top_level_raw

    def _convert_from_openai_format(
        self,
        openai_results: list[dict[str, Any]],
    ) -> list[BatchResult]:
        """Convert OpenAI batch results to BatchResult format.

        OpenAI result format:
        {
            "id": "batch_req_123",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "..."}}],
                    ...
                }
            },
            "error": null or {"message": "...", "code": "..."}
        }
        """
        batch_results = []

        for result in openai_results:
            request_id = result.get("custom_id", "")
            if not request_id:
                print(
                    f"{OPENAI_BATCH_LOG_PREFIX} Skipping result with no custom_id: {result}"
                )
                continue
            error, raw_body, response_text, top_level_raw = self._parse_openai_result(
                result
            )

            generation_payload = None

            if error is None:
                generation_payload = {
                    "response": response_text if response_text else "",
                    "raw_api_response": raw_body,
                }

            batch_results.append(
                {
                    "request_id": request_id,
                    "error": error,
                    "raw_api_response": top_level_raw,
                    "generation": generation_payload,
                    "judge": None,
                }
            )

        return batch_results

    async def _download_jsonl(self, file_id: str | None) -> list[dict[str, Any]]:
        if not file_id:
            return []
        content = await self.client.files.content(file_id)
        return parse_jsonl(content.text)

    async def _get_batch_results(self, batch: Any) -> list[BatchResult]:
        raw_results: list[dict[str, Any]] = []
        raw_results.extend(
            await self._download_jsonl(getattr(batch, "output_file_id", None))
        )
        raw_results.extend(
            await self._download_jsonl(getattr(batch, "error_file_id", None))
        )
        return self._convert_from_openai_format(raw_results)

    async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
        """Submit a batch generation job to OpenAI."""
        if not work_items:
            raise ValueError("work_items cannot be empty")

        requests = [self._build_request(item) for item in work_items]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            prefix="openai_batch_input_",
            delete=False,
        ) as tmp_file:
            input_path = Path(tmp_file.name)
            tmp_file.write("\n".join(json.dumps(req) for req in requests))
            tmp_file.write("\n")

        try:
            with open(input_path, "rb") as data:
                file_response = await self.client.files.create(
                    file=data, purpose="batch"
                )
            file_id = file_response.id
        finally:
            input_path.unlink(missing_ok=True)

        batch = await self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        job_info: BatchJobInfo = {
            "job_id": batch.id,
            "provider": "openai",
            "status": batch.status,
            "model_name": work_items[0]["model"].name if work_items else "unknown",
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "created_at": batch.created_at,
                "request_counts": {
                    "total": batch.request_counts.total if batch.request_counts else 0,
                    "completed": batch.request_counts.completed
                    if batch.request_counts
                    else 0,
                    "failed": batch.request_counts.failed
                    if batch.request_counts
                    else 0,
                }
                if batch.request_counts
                else None,
            },
        }

        return {"job_info": job_info, "submitted_count": len(work_items)}

    async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
        """Poll an OpenAI batch job for completion.

        Args:
            job_info: Job metadata from submit()

        Returns:
            BatchPollResult with status and results (if complete)
        """
        batch = await self.client.batches.retrieve(job_info["job_id"])

        # Possible statuses: validating, failed, in_progress, finalizing, completed, expired, cancelling, cancelled
        if batch.status in ["validating", "in_progress", "finalizing", "cancelling"]:
            completed = batch.request_counts.completed if batch.request_counts else None
            return {
                "status": BatchStatus.RUNNING,
                "completed_count": completed,
                "results": None,
            }

        if batch.status == "failed":
            if batch.errors:
                error_data = batch.errors.data if batch.errors.data else []
                if error_data:
                    print(
                        f"{OPENAI_BATCH_LOG_PREFIX} Batch {batch.id} failed: {str(error_data)}"
                    )

            return {
                "status": BatchStatus.FAILED,
                "completed_count": None,
                "results": None,
            }

        if batch.status in ["expired", "cancelled", "completed"]:
            batch_results = await self._get_batch_results(batch)
            if batch_results:
                return {
                    "status": BatchStatus.COMPLETED,
                    "completed_count": len(batch_results),
                    "results": batch_results,
                }
            print(
                f"{OPENAI_BATCH_LOG_PREFIX} Batch {batch.id} {batch.status} with no results"
            )
            return {
                "status": BatchStatus.FAILED,
                "completed_count": None,
                "results": None,
            }

        # Unknown status
        return {
            "status": BatchStatus.FAILED,
            "completed_count": None,
            "results": None,
        }

    async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
        """Cancel an active OpenAI batch job.

        Args:
            job_info: Job metadata from submit()

        Returns:
            BatchCancelResult with success status and message
        """
        job_id = job_info["job_id"]
        try:
            batch = await self.client.batches.cancel(job_id)
            return {
                "success": True,
                "message": f"Batch {job_id} cancellation initiated (status: {batch.status})",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel batch {job_id}: {e}",
            }


@api_retry()
async def openai_generate(
    model: ModelEntry,
    system_prompt: str,
    user_message: str,
) -> GenerateResult:
    """Generate response using OpenAI Chat Completions API."""
    async with AsyncOpenAI() as client:
        return await openai_compat_generate(client, model, system_prompt, user_message)
