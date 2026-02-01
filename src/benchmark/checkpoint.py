"""Checkpoint management for benchmark tool."""

import asyncio
from contextlib import suppress
import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import orjson

from benchmark.config import (
    BenchmarkConfig,
    JUDGE_MODEL,
    JUDGE_MODEL_OPENROUTER,
)
from benchmark.exceptions import FatalBenchmarkError
from benchmark.protocols import BatchJobInfo

# Type aliases
Checkpoint: TypeAlias = dict[str, Any]

BENCHMARK_NAME = "PersistBench"


class GenerationStatus(Enum):
    """Status of a single generation in the checkpoint."""

    NEEDS_GENERATION = "needs_generation"  # No response; needs (re)generation
    NEEDS_JUDGE = "needs_judge"  # Response exists; needs judge evaluation
    COMPLETED = "completed"  # Has response and judge, no error


def load_checkpoint(output_file: Path) -> Checkpoint:
    """Load checkpoint from file, or return empty checkpoint if file doesn't exist."""
    if not output_file.exists():
        return {"metadata": {}, "entries": {}}

    return orjson.loads(output_file.read_bytes())


def save_checkpoint(data: Checkpoint, output_file: Path) -> None:
    """Save checkpoint to file using atomic write.

    Uses write-to-temp-then-rename pattern to ensure the checkpoint file
    is never left in a partial/corrupted state if the process is interrupted.

    Creates parent directory if it doesn't exist.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory, then atomically rename
    # This ensures we never have a partial checkpoint file
    fd, temp_path = tempfile.mkstemp(
        dir=output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(orjson.dumps(data))
        # Atomic rename - either fully succeeds or file remains unchanged
        os.replace(temp_path, output_file)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


class CheckpointWriter:
    """Buffered checkpoint writer that coalesces updates into periodic flushes."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        output_file: Path,
        flush_interval: float = 1.0,
    ) -> None:
        self._checkpoint = checkpoint
        self._output_file = output_file
        self._dirty = False
        self._lock = asyncio.Lock()
        self._closed = False
        self._task = asyncio.create_task(self._flush_loop(flush_interval))

    @property
    def checkpoint(self) -> Checkpoint:
        return self._checkpoint

    async def update(self, fn: Callable[[Checkpoint], None]) -> None:
        """Apply mutation and mark checkpoint dirty."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("CheckpointWriter is closed")
            fn(self._checkpoint)
            self._dirty = True

    async def _flush_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if self._dirty:
                await asyncio.to_thread(
                    save_checkpoint, self._checkpoint, self._output_file
                )
                self._dirty = False

    async def close(self) -> None:
        """Stop background task and flush any pending writes."""
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._closed = True
        await self._flush()


def get_generation_status(
    checkpoint: Checkpoint, hash_id: str, model: str, gen_index: int
) -> GenerationStatus:
    """Determine current status of a specific generation in the checkpoint."""
    entries = checkpoint.get("entries", {})
    if hash_id not in entries:
        return GenerationStatus.NEEDS_GENERATION

    entry = entries[hash_id]
    model_results = entry.get("results", {})
    if model not in model_results:
        return GenerationStatus.NEEDS_GENERATION

    result_data = model_results[model]
    if not isinstance(result_data, dict) or "generations" not in result_data:
        return GenerationStatus.NEEDS_GENERATION

    generations = result_data["generations"]
    if gen_index >= len(generations):
        return GenerationStatus.NEEDS_GENERATION

    generation = generations[gen_index]
    error = generation.get("error")
    judge = generation.get("judge")
    memory_response = generation.get("memory_response")
    has_response = bool(memory_response)

    # Check if completed (has responses, judge, and no error)
    if (
        has_response
        and not error
        and judge is not None
        and judge.get("score") is not None
        and judge.get("reasoning")
    ):
        return GenerationStatus.COMPLETED

    if not has_response:
        return GenerationStatus.NEEDS_GENERATION

    # Response exists but judge missing or failed
    return GenerationStatus.NEEDS_JUDGE


def _has_any_response(result_data: dict[str, Any]) -> bool:
    """Check if model result has at least one generation with a response."""
    generations = result_data.get("generations", [])
    for gen in generations:
        if gen.get("memory_response"):
            return True
    return False


def _has_completed_generation(result_data: dict[str, Any]) -> bool:
    """Check if model result has at least one fully completed generation (response + judge)."""
    generations = result_data.get("generations", [])
    for gen in generations:
        error = gen.get("error")
        judge = gen.get("judge")
        memory_response = gen.get("memory_response")

        # A generation is completed if it has a response, no error, and valid judge
        if (
            memory_response
            and not error
            and judge is not None
            and judge.get("score") is not None
            and judge.get("reasoning")
        ):
            return True
    return False


def initialize_checkpoint(
    input_entries: list[dict[str, Any]],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool = False,
    judge_provider: str | None = None,
    config_dict: dict[str, Any] | None = None,
    existing_checkpoint: Checkpoint | None = None,
) -> Checkpoint:
    """Load checkpoint, validate judge and model consistency, and update metadata.

    Args:
        config_dict: Serialized config to store in checkpoint for self-contained resume.
            If None, preserves any existing checkpoint["config"].

    Raises:
        FatalBenchmarkError: If model api_params changed and responses already exist
            for that model, or if judge model changed and completed entries exist.
    """
    checkpoint = (
        existing_checkpoint
        if existing_checkpoint is not None
        else load_checkpoint(config.output)
    )

    # Validate model config against metadata (not per-entry)
    previous_metadata = checkpoint.get("metadata", {})
    stored_models = {m["name"]: m for m in previous_metadata.get("models", [])}

    for model in config.models:
        stored = stored_models.get(model.name)
        if stored is None:
            continue

        mismatches: list[str] = []
        config_params = model.api_params or {}
        stored_params = stored.get("api_params", {})
        if config_params != stored_params:
            mismatches.append(
                f"api_params: stored={stored_params}, config={config_params}"
            )
        if model.provider != stored.get("provider"):
            mismatches.append(
                f"provider: stored={stored.get('provider')}, config={model.provider}"
            )
        if model.mode != stored.get("mode"):
            mismatches.append(f"mode: stored={stored.get('mode')}, config={model.mode}")

        if mismatches and not ignore_config_mismatch:
            has_responses = any(
                _has_any_response(entry_data.get("results", {}).get(model.name, {}))
                for entry_data in checkpoint.get("entries", {}).values()
            )
            has_active_batch_job = (
                get_batch_job_info(checkpoint, "generation", model.name) is not None
            )
            if has_responses or has_active_batch_job:
                detail = "; ".join(mismatches)
                raise FatalBenchmarkError(
                    f"Model '{model.name}' config changed between runs ({detail}). "
                    f"Revert to original config, delete the output file to start fresh, "
                    f"or bypass with --ignore-config-mismatch."
                )

    existing_batch_jobs = previous_metadata.get("batch_jobs")

    # Resolve judge model name based on provider
    current_judge_model = (
        JUDGE_MODEL_OPENROUTER if judge_provider == "openrouter" else JUDGE_MODEL
    )
    if previous_metadata:
        existing_judge_model = previous_metadata.get("judge_model")

        if existing_judge_model and existing_judge_model != current_judge_model:
            has_any_completed = any(
                _has_completed_generation(result_data)
                for entry_data in checkpoint.get("entries", {}).values()
                for result_data in entry_data.get("results", {}).values()
            )
            if has_any_completed and not ignore_config_mismatch:
                raise FatalBenchmarkError(
                    f"Judge model changed from '{existing_judge_model}' to '{current_judge_model}'. "
                    "Please revert to original config (stored in the output file) or delete the output file to start fresh."
                )

    checkpoint["metadata"] = {
        "benchmark_name": BENCHMARK_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_entries": len(input_entries),
        "models": [
            {
                "name": model.name,
                "provider": model.provider,
                "mode": model.mode,
                "api_params": model.api_params or {},
            }
            for model in config.models
        ],
        "judge_model": current_judge_model,
        "judge_provider": judge_provider,
        "store_raw_api_responses": config.store_raw_api_responses,
        "generations": config.generations,
        "concurrency": config.concurrency,
        "prompt_template": str(config.prompt_template)
        if config.prompt_template
        else None,
    }

    if existing_batch_jobs:
        checkpoint["metadata"]["batch_jobs"] = {
            "generation": existing_batch_jobs.get("generation", {}),
            "judgment": existing_batch_jobs.get("judgment"),
        }
    else:
        checkpoint["metadata"]["batch_jobs"] = {
            "generation": {},
            "judgment": None,
        }

    # Store config for self-contained checkpoint resume
    if config_dict is not None:
        checkpoint["config"] = config_dict

    return checkpoint


def get_batch_job_info(
    checkpoint: Checkpoint,
    phase: str,
    model_name: str | None = None,
) -> BatchJobInfo | None:
    """Get batch job info for a specific phase and model."""
    batch_jobs = checkpoint.get("metadata", {}).get("batch_jobs", {})

    if phase == "generation":
        job_info = batch_jobs.get("generation", {}).get(model_name)
    else:  # judgment
        job_info = batch_jobs.get("judgment")

    return cast(BatchJobInfo | None, job_info)


def save_batch_job_info(
    checkpoint: Checkpoint,
    phase: str,
    job_info: BatchJobInfo,
    model_name: str | None = None,
) -> None:
    """Save batch job info to checkpoint metadata."""
    if "batch_jobs" not in checkpoint["metadata"]:
        checkpoint["metadata"]["batch_jobs"] = {"generation": {}, "judgment": None}

    if phase == "generation":
        checkpoint["metadata"]["batch_jobs"]["generation"][model_name] = job_info
    else:  # judgment
        checkpoint["metadata"]["batch_jobs"]["judgment"] = job_info


def clear_batch_job(
    checkpoint: Checkpoint,
    phase: str,
    model_name: str | None = None,
) -> None:
    """Clear batch job info after completion."""
    if "batch_jobs" not in checkpoint["metadata"]:
        return

    if phase == "generation":
        checkpoint["metadata"]["batch_jobs"]["generation"].pop(model_name, None)
    else:  # judgment
        checkpoint["metadata"]["batch_jobs"]["judgment"] = None
