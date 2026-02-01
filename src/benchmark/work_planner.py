"""Utilities for loading benchmark inputs and building generation work queues."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from benchmark.checkpoint import (
    Checkpoint,
    GenerationStatus,
    get_generation_status,
    initialize_checkpoint,
    save_checkpoint,
)
from benchmark.config import (
    BenchmarkConfig,
    ModelEntry,
    load_benchmark_config_data,
    resolve_entry_configuration,
    validate_failure_type,
)
from benchmark.exceptions import FatalBenchmarkError
from benchmark.utils import generate_hash_id

# Normalized input row and queued generation triple (entry, model, gen_idx)
InputEntry: TypeAlias = dict[str, Any]
WorkItem: TypeAlias = tuple[InputEntry, ModelEntry, int]


@dataclass(slots=True)
class WorkPlan:
    """Structured view of pending benchmark work."""

    checkpoint: Checkpoint
    pending_work: list[WorkItem]
    completed: int
    total: int


def load_input_file(file_path: Path) -> list[InputEntry]:
    """Load entries from JSON or JSONL file."""
    if not file_path.exists():
        raise ValueError(f"Input file {file_path} does not exist")

    suffix = file_path.suffix.lower()
    if suffix not in (".json", ".jsonl"):
        raise ValueError(f"Input file must be JSON or JSONL (got {suffix})")

    with open(file_path, "r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            entries = [json.loads(line) for line in f if line.strip()]
        else:
            entries = json.load(f)

    print(f"Loaded {len(entries)} rows from {file_path}")
    return entries


def load_and_validate_entries(input_file: Path) -> list[InputEntry]:
    """Load, validate, and deduplicate entries from input file."""
    raw_entries = load_input_file(input_file)
    input_entries: list[dict[str, Any]] = []
    seen_failure_types: dict[str, str] = {}

    for i, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise FatalBenchmarkError("Entry must be a dict")
        if "memories" not in raw_entry or "query" not in raw_entry:
            raise FatalBenchmarkError("Entry must have 'memories' and 'query' fields")

        memories = raw_entry["memories"]
        query = raw_entry["query"]

        if not isinstance(memories, list) or not memories:
            raise FatalBenchmarkError("'memories' must be a non-empty list")
        if not isinstance(query, str) or not query.strip():
            raise FatalBenchmarkError("'query' must be a non-empty string")

        hash_id = generate_hash_id(memories, query)
        failure_type = resolve_entry_configuration(raw_entry)
        validate_failure_type(failure_type)

        if hash_id in seen_failure_types:
            prev_leak = seen_failure_types[hash_id]
            if prev_leak != failure_type:
                raise FatalBenchmarkError(
                    f"Duplicate memories+query at index {i} with conflicting evaluation configuration.\n"
                    f"Previous: failure_type={prev_leak}\n"
                    f"Current: failure_type={failure_type}\n"
                    f"The same memories+query cannot be evaluated with different configurations in one run.\n"
                    f"Either remove the duplicate or use separate input files."
                )
            continue

        seen_failure_types[hash_id] = failure_type
        entry_data = {
            "memories": memories,
            "query": query,
            "hash_id": hash_id,
            "original_index": i,
            "failure_type": failure_type,
        }
        input_entries.append(entry_data)

    if not input_entries:
        raise ValueError("No valid entries found")

    return input_entries


def ensure_entry_configuration(entry: dict[str, Any]) -> str:
    failure_type = entry.get("failure_type")

    if failure_type is None:
        failure_type = resolve_entry_configuration(entry)

    validate_failure_type(failure_type)
    entry["failure_type"] = failure_type

    return failure_type


def _hydrate_checkpoint_entry(
    checkpoint: Checkpoint,
    entry: InputEntry,
    ignore_config_mismatch: bool,
    output_file: Path,
) -> None:
    hash_id = entry["hash_id"]

    if hash_id not in checkpoint["entries"]:
        resolved_leak = ensure_entry_configuration(entry)
        checkpoint["entries"][hash_id] = {
            "memories": entry["memories"],
            "query": entry["query"],
            "results": {},
            "failure_type": resolved_leak,
        }
        return

    existing_entry = checkpoint["entries"][hash_id]
    existing_leak = ensure_entry_configuration(existing_entry)
    new_leak = ensure_entry_configuration(entry)

    if existing_leak != new_leak:
        if not ignore_config_mismatch:
            raise FatalBenchmarkError(
                f"Evaluation configuration changed for entry {hash_id}.\n"
                f"Checkpoint has: failure_type={existing_leak}\n"
                f"Input file has: failure_type={new_leak}\n"
                f"Cannot change evaluation config on resume. Either:\n"
                f"  1. Revert input file to original config, or\n"
                f"  2. Delete the checkpoint file at {output_file} to start fresh"
            )
        checkpoint["entries"][hash_id]["failure_type"] = new_leak


def _queue_generations_for_entry(
    checkpoint: Checkpoint,
    entry: InputEntry,
    models: list[ModelEntry],
    generations: int,
) -> tuple[list[WorkItem], int]:
    hash_id = entry["hash_id"]
    pending_work: list[WorkItem] = []
    completed_count = 0

    for model in models:
        model_name = model.name

        if model_name not in checkpoint["entries"][hash_id]["results"]:
            checkpoint["entries"][hash_id]["results"][model_name] = {
                "generations": [],
            }

        for gen_idx in range(generations):
            status = get_generation_status(checkpoint, hash_id, model_name, gen_idx)
            if status == GenerationStatus.COMPLETED:
                completed_count += 1
            else:
                pending_work.append((entry, model, gen_idx))

    return pending_work, completed_count


def _build_work_queue(
    checkpoint: Checkpoint,
    input_entries: list[InputEntry],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool,
) -> tuple[list[WorkItem], int]:
    pending_work: list[WorkItem] = []
    completed_count = 0

    for entry in input_entries:
        _hydrate_checkpoint_entry(
            checkpoint, entry, ignore_config_mismatch, config.output
        )
        entry_work, entry_completed = _queue_generations_for_entry(
            checkpoint, entry, config.models, config.generations
        )
        pending_work.extend(entry_work)
        completed_count += entry_completed

    return pending_work, completed_count


def extract_entries_from_checkpoint(checkpoint: Checkpoint) -> list[InputEntry]:
    """Build InputEntry dicts from checkpoint entries for resume without original input file."""
    entries: list[InputEntry] = []
    for hash_id, entry_data in checkpoint.get("entries", {}).items():
        entries.append(
            {
                "memories": entry_data["memories"],
                "query": entry_data["query"],
                "hash_id": hash_id,
                "failure_type": entry_data.get("failure_type"),
            }
        )
    return entries


def reconstruct_config(
    checkpoint: Checkpoint, checkpoint_path: Path
) -> BenchmarkConfig:
    """Rebuild BenchmarkConfig from checkpoint's stored config.

    Uses load_benchmark_config_data to ensure prompt templates are loaded
    and model names are validated, same as a fresh config load.
    Overrides output to point to the checkpoint file path (the file may have been moved).
    """
    stored_config = checkpoint.get("config")
    if stored_config is None:
        raise FatalBenchmarkError(
            f"Checkpoint {checkpoint_path} has no stored config. "
            f"This checkpoint was created before config-in-checkpoint support. "
            f"Please provide the original config file instead."
        )

    config = load_benchmark_config_data(
        dict(stored_config), config_path=checkpoint_path
    )
    config.output = checkpoint_path
    return config


def prepare_work_plan(
    input_entries: list[InputEntry],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool = False,
    judge_provider: str | None = None,
    config_dict: dict[str, Any] | None = None,
    existing_checkpoint: Checkpoint | None = None,
) -> WorkPlan:
    """Initialize checkpoint, build work queue, and return planning summary."""
    checkpoint = initialize_checkpoint(
        input_entries,
        config,
        ignore_config_mismatch,
        judge_provider=judge_provider,
        config_dict=config_dict,
        existing_checkpoint=existing_checkpoint,
    )

    total_count = len(input_entries) * len(config.models) * config.generations

    pending_work, completed_count = _build_work_queue(
        checkpoint, input_entries, config, ignore_config_mismatch
    )

    save_checkpoint(checkpoint, config.output)

    return WorkPlan(
        checkpoint=checkpoint,
        pending_work=pending_work,
        completed=completed_count,
        total=total_count,
    )
