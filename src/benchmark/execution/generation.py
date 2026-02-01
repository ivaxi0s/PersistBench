"""Generation executors for benchmark workflow."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass
from typing import Any, Sequence

import time
from tqdm.asyncio import tqdm
from rich.console import Console

from benchmark.checkpoint import (
    Checkpoint,
    CheckpointWriter,
    clear_batch_job,
    get_batch_job_info,
    save_batch_job_info,
    save_checkpoint,
)
from benchmark.config import (
    BenchmarkConfig,
    ModelEntry,
)
from benchmark.exceptions import FatalBenchmarkError
from benchmark.types import GenerationEntry
from benchmark.prompts import build_generation_prompt
from benchmark.provider_registry import PROVIDERS, get_batch_provider
from benchmark.protocols import (
    BatchGenerateFn,
    BatchJobInfo,
    BatchResult,
    BatchStatus,
    BatchWorkItem,
    GenerateFn,
)
from benchmark.work_planner import InputEntry, WorkItem

BATCH_REQUEST_DELIMITER = "__"
PHASE_GENERATION = "generation"
ERROR_EMPTY_RESPONSE = "Empty or whitespace-only response"

__all__ = [
    "BATCH_REQUEST_DELIMITER",
    "BatchGenerationExecutor",
    "GenerationResult",
    "GenerationTask",
    "SequentialGenerationExecutor",
    "build_generation_tasks",
    "poll_all_batch_jobs",
]


@dataclass(slots=True)
class GenerationTask:
    """Single generation workload item."""

    entry: InputEntry
    model: ModelEntry
    gen_idx: int

    @property
    def hash_id(self) -> str:
        return self.entry["hash_id"]


@dataclass(slots=True)
class GenerationResult:
    """Result payload ready for checkpoint persistence."""

    hash_id: str
    model_name: str
    gen_idx: int
    payload: GenerationEntry


def build_generation_tasks(pending_work: Sequence[WorkItem]) -> list[GenerationTask]:
    """Convert work items into structured generation tasks."""
    tasks: list[GenerationTask] = []
    for entry, model, gen_idx in pending_work:
        tasks.append(
            GenerationTask(
                entry=entry,
                model=model,
                gen_idx=gen_idx,
            )
        )
    return tasks


class SequentialGenerationExecutor:
    """Runs generation tasks directly against provider APIs."""

    def __init__(self) -> None:
        self._semaphore: asyncio.Semaphore | None = None

    async def run(
        self,
        tasks: Sequence[GenerationTask],
        checkpoint: Checkpoint,
        config: BenchmarkConfig,
    ) -> None:
        if not tasks:
            return

        self._semaphore = asyncio.Semaphore(config.concurrency)
        checkpoint_writer = CheckpointWriter(checkpoint, config.output)
        prompt_template = config.prompt_template_content

        success_count = 0
        error_count = 0
        count_lock = asyncio.Lock()

        async def _run_task(task: GenerationTask, pbar: tqdm) -> None:
            nonlocal success_count, error_count
            assert self._semaphore is not None
            async with self._semaphore:
                result = await _process_generation_task(
                    task,
                    prompt_template,
                    store_raw_api_responses=config.store_raw_api_responses,
                )
                await _save_generation_result(
                    checkpoint_writer=checkpoint_writer,
                    hash_id=result.hash_id,
                    model_name=result.model_name,
                    gen_idx=result.gen_idx,
                    generation_data=result.payload,
                )
                async with count_lock:
                    if result.payload.get("error"):
                        error_count += 1
                    else:
                        success_count += 1
                    pbar.set_postfix_str(f"ok={success_count} err={error_count}")
                    pbar.update(1)

        with tqdm(total=len(tasks), desc="Generating responses") as pbar:
            try:
                await asyncio.gather(*(_run_task(task, pbar) for task in tasks))
            finally:
                await checkpoint_writer.close()


class BatchGenerationExecutor:
    """Runs generation tasks via provider batch APIs."""

    async def run(
        self,
        tasks: Sequence[GenerationTask],
        checkpoint: Checkpoint,
        config: BenchmarkConfig,
    ) -> None:
        """Submit new batch jobs for pending generation tasks.

        Only submits jobs for models without active batch jobs. Models with
        running jobs are skipped - use poll_all_batch_jobs() to check them.
        """
        if not tasks:
            return

        tasks_by_model: dict[str, list[GenerationTask]] = {}
        for task in tasks:
            tasks_by_model.setdefault(task.model.name, []).append(task)

        for model_name, model_tasks in tasks_by_model.items():
            job_info = get_batch_job_info(checkpoint, PHASE_GENERATION, model_name)
            if job_info:
                print(
                    f"  Skipping {model_name}: active batch job {job_info['job_id']} already in checkpoint"
                )
                continue

            model = model_tasks[0].model
            try:
                batch_provider = get_batch_provider(model.provider)
            except ValueError as e:
                raise FatalBenchmarkError(
                    f"Provider '{model.provider}' does not support batch generation (model '{model.name}')."
                ) from e

            batch_items = _prepare_generation_batch_items(
                model_tasks, config.prompt_template_content
            )
            if not batch_items:
                continue

            print(
                f"\nSubmitting batch generation job for {model_name} ({len(batch_items)} requests)"
            )
            submit_result = await batch_provider.submit(batch_items)

            save_batch_job_info(
                checkpoint, PHASE_GENERATION, submit_result["job_info"], model_name
            )
            save_checkpoint(checkpoint, config.output)

            print(
                f"Submitted job {submit_result['job_info']['job_id']} ({submit_result['submitted_count']} requests)"
            )
            print("Re-run to poll for completion")


async def _process_generation_task(
    task: GenerationTask,
    prompt_template: str | None = None,
    store_raw_api_responses: bool = False,
) -> GenerationResult:
    provider_name = task.model.provider
    if provider_name not in PROVIDERS:
        raise FatalBenchmarkError(
            f"Unknown provider '{provider_name}' for model '{task.model.name}'"
        )

    generate_fn = PROVIDERS[provider_name]["generate_fn"]
    if generate_fn is None:
        raise FatalBenchmarkError(
            f"Provider '{provider_name}' lacks sequential generation support"
        )

    try:
        memory_response, memory_raw, error_msg = await _generate_model_response(
            task.model,
            generate_fn,
            task.entry["query"],
            task.entry["memories"],
            prompt_template,
        )

        if not store_raw_api_responses:
            memory_raw = {}

        generation_data: GenerationEntry = {
            "generation_index": task.gen_idx,
            "error": error_msg,
            "memory_response": memory_response,
            "memory_raw_api_response": memory_raw,
            "judge": None,
        }

        return GenerationResult(
            hash_id=task.hash_id,
            model_name=task.model.name,
            gen_idx=task.gen_idx,
            payload=generation_data,
        )

    except (asyncio.CancelledError, FatalBenchmarkError):
        raise
    except Exception as exc:
        error_data: GenerationEntry = {
            "generation_index": task.gen_idx,
            "error": f"Unexpected error: {exc}",
            "memory_response": None,
            "memory_raw_api_response": {},
            "judge": None,
        }

        return GenerationResult(
            hash_id=task.hash_id,
            model_name=task.model.name,
            gen_idx=task.gen_idx,
            payload=error_data,
        )


async def _generate_model_response(
    model: ModelEntry,
    generate_response_fn: GenerateFn,
    query: str,
    memories: list[str],
    prompt_template: str | None = None,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Call generation function and handle errors."""
    try:
        if prompt_template:
            generation_prompt = build_generation_prompt(
                memories, model.name, prompt_template
            )
        else:
            generation_prompt = build_generation_prompt(memories, model.name)
        result = await generate_response_fn(model, generation_prompt, query)

        sanitized_response = _sanitize_response_text(result.get("response"))
        if sanitized_response is None:
            return None, result.get("raw_api_response", {}), ERROR_EMPTY_RESPONSE

        return sanitized_response, result.get("raw_api_response", {}), None
    except FatalBenchmarkError:
        raise
    except Exception:
        return (
            None,
            {},
            f"Generation failed: {traceback.format_exc()}",
        )


def _sanitize_response_text(value: Any) -> str | None:
    """Trim whitespace and treat empty or non-string responses as missing."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


async def _save_generation_result(
    checkpoint_writer: CheckpointWriter,
    hash_id: str,
    model_name: str,
    gen_idx: int,
    generation_data: GenerationEntry,
) -> None:
    """Save a generation result and schedule a buffered checkpoint flush."""
    await checkpoint_writer.update(
        lambda checkpoint: _set_generation_entry(
            checkpoint, hash_id, model_name, gen_idx, generation_data
        )
    )


def _set_generation_entry(
    checkpoint: Checkpoint,
    hash_id: str,
    model_name: str,
    gen_idx: int,
    generation_data: GenerationEntry,
) -> None:
    """Insert or update a generation entry, extending the list if needed."""
    model_results = checkpoint["entries"][hash_id]["results"][model_name]
    generations_list = model_results.setdefault("generations", [])
    while len(generations_list) <= gen_idx:
        generations_list.append({})
    generations_list[gen_idx] = generation_data


@dataclass
class _ActiveJob:
    """Tracks an active batch job for polling."""

    model_name: str
    provider_name: str
    job_info: BatchJobInfo
    batch_provider: BatchGenerateFn


async def poll_all_batch_jobs(
    checkpoint: Checkpoint,
    config: BenchmarkConfig,
) -> None:
    """Poll ALL batch jobs from ALL providers in a single unified loop.

    Uses one shared global timeout across all providers and models.
    """
    poll_interval_seconds = 5
    timeout_seconds = config.batch_poll_timeout_minutes * 60

    # Collect all active jobs across all providers
    active_jobs: dict[str, _ActiveJob] = {}
    generation_jobs = (
        checkpoint.get("metadata", {}).get("batch_jobs", {}).get(PHASE_GENERATION, {})
    )
    if not isinstance(generation_jobs, dict):
        return

    expected_models = {m.name for m in config.models}
    orphaned_jobs = sorted(
        name for name in generation_jobs.keys() if name not in expected_models
    )
    if orphaned_jobs:
        print(
            "Skipping batch job(s) for model(s) not in the current config: "
            + ", ".join(orphaned_jobs)
        )
        print(
            "Cancel active batch jobs with: benchmark <run|generate> <file> --cancel (cancels all jobs), "
            "or remove the orphaned job records from the checkpoint metadata."
        )

    for model_name, job_info in generation_jobs.items():
        if model_name not in expected_models:
            continue
        if not isinstance(job_info, dict):
            print(
                f"Warning: Invalid batch job record for {model_name!r} (expected object), skipping."
            )
            continue

        provider_name = job_info.get("provider")
        if not provider_name:
            print(
                f"Warning: Batch job for {model_name!r} missing provider; cannot poll."
            )
            continue

        try:
            batch_provider = get_batch_provider(provider_name)
        except ValueError as e:
            raise FatalBenchmarkError(
                f"Cannot poll batch job {job_info.get('job_id')!r} for model {model_name!r}: "
                f"unknown/unsupported provider {provider_name!r}."
            ) from e

        active_jobs[model_name] = _ActiveJob(
            model_name=model_name,
            provider_name=provider_name,
            job_info=job_info,
            batch_provider=batch_provider,
        )

    if not active_jobs:
        return

    console = Console()
    console.print(
        f"\nPolling {len(active_jobs)} batch generation job(s) across all providers..."
    )
    for model_name, job in active_jobs.items():
        console.print(
            f"  - {model_name} ({job.provider_name}): {job.job_info['job_id']}"
        )

    start_time = time.time()

    def format_remaining() -> str:
        remaining = max(0, int(timeout_seconds - (time.time() - start_time)))
        mins, secs = divmod(remaining, 60)
        return f"{mins}:{secs:02d}"

    def build_status() -> str:
        models = ", ".join(active_jobs.keys())
        return f"[{format_remaining()} remaining] {len(active_jobs)} job(s): {models}"

    with console.status(build_status()) as status:
        while active_jobs:
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"Timeout reached ({config.batch_poll_timeout_minutes} minutes)"
                )
                console.print("Remaining jobs:")
                for model_name, job in active_jobs.items():
                    console.print(f"  - {model_name}: {job.job_info['job_id']}")
                console.print("Re-run later to check status and import results.")
                break

            status.update(build_status())

            completed_this_round: list[str] = []
            for model_name, job in list(active_jobs.items()):
                poll_result = await job.batch_provider.poll(job.job_info)

                if poll_result["status"] == BatchStatus.FAILED:
                    clear_batch_job(checkpoint, PHASE_GENERATION, model_name)
                    save_checkpoint(checkpoint, config.output)
                    raise RuntimeError(
                        f"Batch job failed for {model_name}: {job.job_info['job_id']}\n"
                        f"Job cleared from checkpoint. Retry by running again."
                    )

                if poll_result["status"] == BatchStatus.COMPLETED:
                    results: list[BatchResult] = poll_result.get("results") or []
                    console.print(
                        f"{model_name}: Completed, importing {len(results)} results..."
                    )
                    stats = await _import_batch_generation_results(
                        results, checkpoint, config, model_name=model_name
                    )
                    if stats.errors > 0 or stats.skipped > 0:
                        console.print(
                            f"  [yellow]{stats.imported} ok, {stats.errors} errors, {stats.skipped} skipped[/yellow]"
                        )
                    clear_batch_job(checkpoint, PHASE_GENERATION, model_name)
                    save_checkpoint(checkpoint, config.output)
                    completed_this_round.append(model_name)

            for model_name in completed_this_round:
                del active_jobs[model_name]

            if active_jobs:
                await asyncio.sleep(poll_interval_seconds)


def _prepare_generation_batch_items(
    tasks: list[GenerationTask],
    prompt_template: str | None = None,
) -> list[BatchWorkItem]:
    batch_items: list[BatchWorkItem] = []
    for task in tasks:
        if prompt_template:
            generation_prompt = build_generation_prompt(
                task.entry["memories"], task.model.name, prompt_template
            )
        else:
            generation_prompt = build_generation_prompt(
                task.entry["memories"], task.model.name
            )

        single_item: BatchWorkItem = {
            "request_id": _make_request_id(task.hash_id, task.gen_idx),
            "model": task.model,
            "system_prompt": generation_prompt,
            "user_message": task.entry["query"],
        }
        batch_items.append(single_item)

    return batch_items


def _parse_batch_request_id(
    request_id: str,
) -> tuple[str, int] | None:
    """Parse a batch request_id into hash_id and gen_idx.

    Current format: {hash_id}__{gen_idx}
    """
    parts = request_id.split(BATCH_REQUEST_DELIMITER)
    if len(parts) != 2:
        print(f"Warning: Invalid request_id format: {request_id!r}, skipping")
        return None

    hash_id, gen_idx_str = parts
    try:
        gen_idx = int(gen_idx_str)
    except ValueError:
        print(f"Warning: Invalid request_id format: {request_id!r}, skipping")
        return None

    return hash_id, gen_idx


def _make_request_id(hash_id: str, gen_idx: int) -> str:
    """Create a batch-API-compatible request ID.

    Format: {hash_id}__{gen_idx}
    """
    return BATCH_REQUEST_DELIMITER.join([hash_id, str(gen_idx)])


def _extract_generation_payload(
    result: BatchResult,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Pull sanitized generation text and metadata from batch result."""
    error = result.get("error")
    payload = result.get("generation")
    if payload:
        raw_response: dict[str, Any] = payload.get("raw_api_response", {})
        response_text = payload.get("response")
    else:
        raw_response = result.get("raw_api_response", {})
        response_text = None

    sanitized_response = None
    if not error:
        sanitized_response = _sanitize_response_text(response_text)
        if sanitized_response is None:
            error = ERROR_EMPTY_RESPONSE

    return sanitized_response, error, raw_response


@dataclass
class BatchImportStats:
    """Statistics from batch result import."""

    imported: int = 0
    errors: int = 0
    skipped: int = 0


async def _import_batch_generation_results(
    results: list[BatchResult],
    checkpoint: Checkpoint,
    config: BenchmarkConfig,
    model_name: str,
) -> BatchImportStats:
    """Import batch generation results into checkpoint."""
    stats = BatchImportStats()
    for result in results:
        if result.get("generation") is None and not result.get("error"):
            print(
                f"Warning: Missing generation payload for {result['request_id']}, skipping"
            )
            stats.skipped += 1
            continue

        try:
            parsed = _parse_batch_request_id(result["request_id"])
            if parsed is None:
                stats.skipped += 1
                continue
            hash_id, gen_idx = parsed

            entry_data = checkpoint.get("entries", {}).get(hash_id)
            if not isinstance(entry_data, dict):
                print(
                    f"Warning: No checkpoint entry for hash_id {hash_id!r} (request_id={result['request_id']!r}), skipping"
                )
                stats.skipped += 1
                continue

            # Ensure destination model bucket exists (the checkpoint may predate the model
            # entry, or the work plan may not have hydrated this row yet).
            entry_results = entry_data.setdefault("results", {})
            entry_results.setdefault(model_name, {"generations": []})

            memory_response, error, raw_api_response = _extract_generation_payload(
                result
            )
            if not config.store_raw_api_responses:
                raw_api_response = {}

            generation_data: GenerationEntry = {
                "generation_index": gen_idx,
                "error": error,
                "memory_response": memory_response,
                "memory_raw_api_response": raw_api_response,
                "judge": None,
            }

            _set_generation_entry(
                checkpoint, hash_id, model_name, gen_idx, generation_data
            )
            if generation_data.get("error"):
                stats.errors += 1
            else:
                stats.imported += 1
        except (ValueError, KeyError, TypeError) as e:
            print(
                f"Warning: Error processing batch result {result.get('request_id')}: {e}, skipping"
            )
            stats.skipped += 1

    save_checkpoint(checkpoint, config.output)
    return stats
