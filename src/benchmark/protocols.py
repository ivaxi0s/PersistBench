"""Protocol definitions for custom function contracts."""

from enum import StrEnum
from typing import Any, Protocol, TypedDict

from benchmark.config import ModelEntry
from benchmark.types import GenerateResult, JudgeResult


class GenerateFn(Protocol):
    """Protocol for custom generation functions.

    Custom implementations allow you to:
    - Use different API providers (Anthropic, Google, local models)
    - Implement mock responses for testing
    - Add custom preprocessing/postprocessing
    - Control API parameters dynamically

    Requirements:
    - Must be async (use 'async def')
    - Must return GenerateResult TypedDict with:
        - response: str (the actual text response)
        - raw_api_response: dict (full API response or custom metadata, can be {})

    Example:
        async def my_generate(
            model: ModelEntry,
            system_prompt: str,
            user_message: str
        ) -> GenerateResult:
            # Your custom implementation
            response = await your_api_call(...)
            return {
                "response": response.text,
                "raw_api_response": response.to_dict()
            }
    """

    async def __call__(
        self, model: ModelEntry, system_prompt: str, user_message: str
    ) -> GenerateResult:
        """Generate response from model.

        Args:
            model: Model configuration (model.name and model.api_params)
            system_prompt: System prompt with context (includes memories from config)
            user_message: User query from input file

        Returns:
            GenerateResult TypedDict with 'response' and 'raw_api_response' keys
        """
        ...


class BatchJobInfo(TypedDict):
    """Metadata about a submitted batch job.

    Used to track batch jobs in checkpoint metadata for resumability.
    """

    job_id: str
    provider: str  # e.g., "vertex_ai", "anthropic", "openai"
    status: str  # "submitted", "running", "completed", "failed"
    model_name: str
    submitted_at: str  # ISO timestamp
    metadata: dict[str, Any]  # Provider-specific metadata


class BatchWorkItem(TypedDict):
    """Single item in a batch request.

    Request ID format (delimiter is '__'):
    - "hash_id__gen_idx"
    """

    request_id: str
    model: ModelEntry
    system_prompt: str
    user_message: str


class BatchResult(TypedDict):
    """Single result from a batch job.

    For partial failures, set error and leave judge/generation payload empty.
    Exactly one of judge or generation should be populated when successful.
    """

    request_id: str
    error: str | None
    raw_api_response: dict[str, Any]
    generation: GenerateResult | None
    judge: JudgeResult | None


class BatchSubmitResult(TypedDict):
    """Result from submitting a batch job."""

    job_info: BatchJobInfo
    submitted_count: int


class BatchPollResult(TypedDict):
    """Result from polling a batch job.

    Status values:
    - "running": Job still processing
    - "completed": Job finished successfully (results available)
    - "failed": Job failed (no results)
    """

    status: "BatchStatus"
    completed_count: int | None
    results: list[BatchResult] | None


class BatchCancelResult(TypedDict):
    """Result from cancelling a batch job."""

    success: bool
    message: str


class BatchGenerateFn(Protocol):
    """Protocol for batch generation providers.

    Enables cost-effective batch processing by submitting multiple generation
    requests as a single batch job. Supports submit → poll → import workflow
    for long-running batch jobs.

    Example providers: Vertex AI Batch, Anthropic Message Batches, OpenAI Batch API

    Example:
        class MyBatchProvider:
            async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
                # Submit batch job to provider
                job = await provider.submit_batch(work_items)
                return {
                    "job_info": {
                        "job_id": job.id,
                        "provider": "my_provider",
                        "status": "submitted",
                        "model_name": work_items[0]["model"].name,
                        "submitted_at": datetime.utcnow().isoformat() + "Z",
                        "metadata": {},
                    },
                    "submitted_count": len(work_items),
                }

            async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
                # Check job status
                job = await provider.get_batch(job_info["job_id"])
                if job.status == "running":
                    return {"status": "running", "completed_count": None, "results": None}
                # Download and parse results
                results = await provider.get_results(job.id)
                return {
                    "status": "completed",
                    "completed_count": len(results),
                    "results": results,
                }

            async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
                # Cancel the batch job
                await provider.cancel_batch(job_info["job_id"])
                return {"success": True, "message": "Job cancelled"}
    """

    async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
        """Submit a batch generation job.

        Args:
            work_items: List of generation requests to process as a batch

        Returns:
            BatchSubmitResult with job_info (for polling) and submitted_count

        Raises:
            Exception: If batch submission fails (will stop benchmark execution)
        """
        ...

    async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
        """Poll a batch job for completion.

        Args:
            job_info: Job metadata from submit() (stored in checkpoint)

        Returns:
            BatchPollResult with status and results (if complete).
            Supports partial results - some items may have errors.

        Raises:
            Exception: If polling fails (will stop benchmark execution)
        """
        ...

    async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
        """Cancel a batch job.

        Args:
            job_info: Job metadata from submit()

        Returns:
            BatchCancelResult with success status and message
        """
        ...


class BatchStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
