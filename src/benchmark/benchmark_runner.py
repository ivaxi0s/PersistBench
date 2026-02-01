"""Batch LLM generation and judge evaluation tool."""

import asyncio
from pathlib import Path
from typing import Any, TypeAlias

import orjson

from benchmark.checkpoint import (
    GenerationStatus,
    clear_batch_job,
    get_generation_status,
    load_checkpoint,
    save_checkpoint,
)
from benchmark.config import (
    BenchmarkConfig,
    load_benchmark_config_data,
)
from benchmark.dry_run import run_dry_run
from benchmark.exceptions import FatalBenchmarkError
from benchmark.provider_registry import (
    get_batch_provider,
    resolve_model_generation_mode,
)
from benchmark.utils import BenchmarkStats, print_benchmark_summary
from benchmark.execution.generation import (
    BatchGenerationExecutor,
    GenerationTask,
    SequentialGenerationExecutor,
    build_generation_tasks,
    poll_all_batch_jobs,
)
from benchmark.execution.judgment import (
    SequentialJudgmentExecutor,
    build_judgment_tasks,
    get_judge_provider,
    set_judge_provider,
)
from benchmark.work_planner import (
    InputEntry,
    WorkItem,
    extract_entries_from_checkpoint,
    load_and_validate_entries,
    prepare_work_plan,
    reconstruct_config,
)

Checkpoint: TypeAlias = dict[str, Any]

# Run-level retry defaults
RUN_RETRY_MAX_ATTEMPTS = 3
RUN_RETRY_DELAY_SECONDS = 30
RUN_RETRY_CONCURRENCY_FACTOR = 0.7


def _load_json_file(file_path: Path) -> tuple[dict[str, Any], bool]:
    """Read JSON file and detect whether it's a config or checkpoint.

    Returns:
        (data, is_checkpoint) where is_checkpoint is True if the file
        has an 'entries' key (checkpoint format).
    """
    if not file_path.exists():
        raise FatalBenchmarkError(f"File not found: {file_path}")

    try:
        data = orjson.loads(file_path.read_bytes())
    except orjson.JSONDecodeError as e:
        raise FatalBenchmarkError(f"Invalid JSON in file {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise FatalBenchmarkError(
            f"File {file_path} must contain a JSON object at top-level"
        )

    is_checkpoint = "entries" in data
    if is_checkpoint and "metadata" not in data:
        raise FatalBenchmarkError(
            f"File {file_path} looks like a checkpoint (has 'entries') but is missing 'metadata'."
        )

    return data, is_checkpoint


async def cancel_batch_jobs(file_path: str | Path) -> None:
    """Cancel all active batch jobs for a benchmark.

    Accepts either a config file or checkpoint file. Locates the checkpoint,
    finds all active batch jobs, cancels them via provider APIs, and clears
    them from the checkpoint.
    """
    path = Path(file_path)
    data, is_checkpoint = _load_json_file(path)

    if is_checkpoint:
        checkpoint = data
        output_file = path
    else:
        config = load_benchmark_config_data(data, config_path=path)
        checkpoint = load_checkpoint(config.output)
        output_file = config.output

    batch_jobs = checkpoint.get("metadata", {}).get("batch_jobs", {})
    generation_jobs = batch_jobs.get("generation", {})

    if not generation_jobs:
        print("No active batch jobs found in checkpoint.")
        return

    print(f"Found {len(generation_jobs)} active batch job(s) to cancel...")

    cancelled_count = 0
    failed_count = 0

    for model_name, job_info in list(generation_jobs.items()):
        provider_name = job_info.get("provider")
        job_id = job_info.get("job_id")

        print(
            f"  Cancelling {provider_name} batch job {job_id} for model {model_name}..."
        )

        try:
            batch_provider = get_batch_provider(provider_name)
        except (ValueError, FatalBenchmarkError):
            print(
                f"    Warning: Provider '{provider_name}' not found, clearing job from checkpoint"
            )
            clear_batch_job(checkpoint, "generation", model_name)
            failed_count += 1
            continue

        try:
            result = await batch_provider.cancel(job_info)

            if result["success"]:
                print(f"    {result['message']}")
                clear_batch_job(checkpoint, "generation", model_name)
                cancelled_count += 1
            else:
                print(f"    {result['message']}")
                failed_count += 1

        except Exception as e:
            print(f"    Error cancelling job: {e}")
            failed_count += 1

    save_checkpoint(checkpoint, output_file)

    print(
        f"\nCancellation complete: {cancelled_count} succeeded, {failed_count} failed"
    )
    if cancelled_count > 0:
        print(f"Checkpoint updated: {output_file}")


def _prepare_benchmark_execution(
    input_entries: list[InputEntry],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool = False,
    concurrency_override: int | None = None,
    judge_provider: str | None = None,
    config_dict: dict[str, Any] | None = None,
    existing_checkpoint: Checkpoint | None = None,
) -> tuple[Checkpoint, list[WorkItem]]:
    """Initialize checkpoint, build work queue, and print progress summary."""
    work_plan = prepare_work_plan(
        input_entries=input_entries,
        config=config,
        ignore_config_mismatch=ignore_config_mismatch,
        judge_provider=judge_provider,
        config_dict=config_dict,
        existing_checkpoint=existing_checkpoint,
    )
    completed_count = work_plan.completed
    total_count = work_plan.total
    effective_concurrency = (
        concurrency_override if concurrency_override is not None else config.concurrency
    )
    print(
        f"Starting benchmark: {len(input_entries)} entries, {total_count} generations "
        f"({completed_count} completed, {total_count - completed_count} remaining, "
        f"concurrency: {effective_concurrency})"
    )

    return work_plan.checkpoint, work_plan.pending_work


def _load_from_file(
    file_path: Path,
    limit: int | None = None,
) -> tuple[list[InputEntry], BenchmarkConfig, bool, dict[str, Any] | None]:
    """Load entries and config from either a config file or a checkpoint file.

    Returns:
        (entries, config, is_fresh_config, checkpoint) where is_fresh_config
        is True when loading from a config file (config_dict should be captured
        after CLI overrides), and checkpoint is the loaded dict when resuming.
    """
    data, is_checkpoint = _load_json_file(file_path)

    checkpoint: dict[str, Any] | None = None

    if is_checkpoint:
        checkpoint = data
        config = reconstruct_config(checkpoint, file_path)
        entries = extract_entries_from_checkpoint(checkpoint)
        is_fresh_config = False
        print(f"Resuming from checkpoint: {file_path} ({len(entries)} entries)")
    else:
        config = load_benchmark_config_data(data, config_path=file_path)
        entries = load_and_validate_entries(config.input)
        is_fresh_config = True
        # Config-level limit applies only when loading fresh from config
        if limit is None:
            limit = config.limit

    if limit is not None and limit > 0 and limit < len(entries):
        print(f"Limiting to {limit} entries (from {len(entries)})")
        entries = entries[:limit]

    return entries, config, is_fresh_config, checkpoint


async def run_benchmark(
    file_path: str | Path,
    dry_run: bool = False,
    ignore_config_mismatch: bool = False,
    limit: int | None = None,
    skip_judge: bool = False,
    skip_generation: bool = False,
    batch_poll_timeout_minutes: int | None = None,
    concurrency_override: int | None = None,
    judge_provider: str | None = None,
    store_raw_api_responses: bool | None = None,
) -> BenchmarkStats:
    """Run benchmark workflow from a config file or checkpoint file.

    Auto-detects whether file_path is a config or checkpoint.

    Args:
        file_path: Path to config file or checkpoint file.
        skip_generation: Skip generation phase (judge-only mode).
        skip_judge: Skip judgment phase (generation-only mode).
        judge_provider: Override judge provider (CLI > config > env > default).

    Returns:
        Benchmark stats for this run/checkpoint.
    """
    path = Path(file_path)
    entries, config, is_fresh_config, existing_checkpoint = _load_from_file(
        path, limit=limit
    )

    # Apply CLI overrides before capturing config_dict
    if store_raw_api_responses is not None:
        config.store_raw_api_responses = store_raw_api_responses

    # Resolve judge provider: CLI > config > env > default (openrouter)
    set_judge_provider(judge_provider or config.judge_provider)
    resolved_provider = get_judge_provider()
    config.judge_provider = resolved_provider

    if batch_poll_timeout_minutes is not None:
        config.batch_poll_timeout_minutes = batch_poll_timeout_minutes

    if concurrency_override is not None:
        config.concurrency = concurrency_override

    # Capture config_dict AFTER CLI overrides so checkpoint stores actual runtime config
    config_dict: dict[str, Any] | None = (
        config.model_dump(mode="json") if is_fresh_config else None
    )

    if dry_run:
        run_dry_run(entries, config, ignore_config_mismatch=ignore_config_mismatch)
        return BenchmarkStats()

    checkpoint, pending_work = _prepare_benchmark_execution(
        entries,
        config,
        ignore_config_mismatch,
        judge_provider=resolved_provider,
        config_dict=config_dict,
        existing_checkpoint=existing_checkpoint,
    )

    # Phase 1: Generation
    if skip_generation:
        print("Skipping generation phase (judge-only mode)")

        # Verify in-scope generations have responses before judging (respects --limit)
        missing_generations: list[tuple[str, str, int]] = []
        expected_generations = checkpoint.get("metadata", {}).get("generations", 1)
        expected_models = [m.name for m in config.models]
        entry_hash_ids = {e["hash_id"] for e in entries}
        for hash_id in entry_hash_ids:
            if hash_id not in checkpoint.get("entries", {}):
                continue
            for model_name in expected_models:
                for gen_idx in range(expected_generations):
                    status = get_generation_status(
                        checkpoint, hash_id, model_name, gen_idx
                    )
                    if status == GenerationStatus.NEEDS_GENERATION:
                        missing_generations.append((hash_id, model_name, gen_idx))

        if missing_generations:
            missing_models = sorted({m for _, m, _ in missing_generations})
            raise FatalBenchmarkError(
                f"{len(missing_generations)} generation(s) have no model response and cannot be judged.\n"
                f"Models with missing responses: {', '.join(missing_models)}\n"
                f"Run `benchmark run <file>` or `benchmark generate <file>` to generate responses first."
            )
    else:
        generation_tasks = build_generation_tasks(pending_work)
        sequential_tasks: list[GenerationTask] = []
        batch_tasks: list[GenerationTask] = []

        mode_by_model_name = {
            model.name: resolve_model_generation_mode(model) for model in config.models
        }

        for task in generation_tasks:
            model = task.model
            status = get_generation_status(
                checkpoint, task.hash_id, model.name, task.gen_idx
            )
            if status != GenerationStatus.NEEDS_GENERATION:
                continue

            use_batch = mode_by_model_name[model.name] == "batch"

            (batch_tasks if use_batch else sequential_tasks).append(task)

        batch_executor = BatchGenerationExecutor()
        if batch_tasks:
            print(f"Processing {len(batch_tasks)} items via batch generation...")
            await batch_executor.run(batch_tasks, checkpoint, config)

        if sequential_tasks:
            print(
                f"Processing {len(sequential_tasks)} items via sequential generation..."
            )
            sequential_executor = SequentialGenerationExecutor()
            await sequential_executor.run(sequential_tasks, checkpoint, config)

        if batch_tasks:
            print("Polling outstanding batch generation jobs...")
            await poll_all_batch_jobs(checkpoint, config)

    # Phase 2: Judgment (sequential only, batch judge not supported)
    if skip_judge:
        print("Skipping judge evaluation phase (generation-only mode)")
    else:
        judgment_tasks = build_judgment_tasks(checkpoint, pending_work)
        if judgment_tasks:
            print(
                f"Evaluating {len(judgment_tasks)} responses via sequential judgment..."
            )
            judgment_executor = SequentialJudgmentExecutor()
            await judgment_executor.run(judgment_tasks, checkpoint, config)

    stats = print_benchmark_summary(
        checkpoint=checkpoint,
        output_file=config.output,
        skip_generation=skip_generation,
        skip_judge=skip_judge,
    )
    return stats


async def run_benchmark_with_retry(
    file_path: str | Path,
    dry_run: bool = False,
    ignore_config_mismatch: bool = False,
    limit: int | None = None,
    skip_judge: bool = False,
    skip_generation: bool = False,
    batch_poll_timeout_minutes: int | None = None,
    retry_enabled: bool = False,
    judge_provider: str | None = None,
    concurrency_override: int | None = None,
    store_raw_api_responses: bool | None = None,
) -> BenchmarkStats:
    """Run benchmark with optional run-level retry on failures.

    When retry_enabled is True, retries the entire benchmark run up to
    RUN_RETRY_MAX_ATTEMPTS times if failures occur, reducing concurrency
    by RUN_RETRY_CONCURRENCY_FACTOR each retry.

    Args:
        file_path: Path to config file or checkpoint file.
        skip_generation: Skip generation phase (judge-only mode).
        skip_judge: Skip judgment phase (generation-only mode).

    Returns:
        Benchmark stats for this run/checkpoint.
    """
    if not retry_enabled:
        return await run_benchmark(
            file_path=file_path,
            dry_run=dry_run,
            ignore_config_mismatch=ignore_config_mismatch,
            limit=limit,
            skip_judge=skip_judge,
            skip_generation=skip_generation,
            batch_poll_timeout_minutes=batch_poll_timeout_minutes,
            judge_provider=judge_provider,
            concurrency_override=concurrency_override,
            store_raw_api_responses=store_raw_api_responses,
        )

    path = Path(file_path)
    data, is_checkpoint = _load_json_file(path)

    if is_checkpoint:
        config = reconstruct_config(data, path)
    else:
        config = load_benchmark_config_data(data, config_path=path)

    current_concurrency = (
        concurrency_override if concurrency_override is not None else config.concurrency
    )
    current_model_names = {m.name for m in config.models}
    last_stats = BenchmarkStats()

    for attempt in range(1 + RUN_RETRY_MAX_ATTEMPTS):
        try:
            last_stats = await run_benchmark(
                file_path=file_path,
                dry_run=dry_run,
                ignore_config_mismatch=ignore_config_mismatch,
                limit=limit,
                skip_judge=skip_judge,
                skip_generation=skip_generation,
                batch_poll_timeout_minutes=batch_poll_timeout_minutes,
                concurrency_override=current_concurrency,
                judge_provider=judge_provider,
                store_raw_api_responses=store_raw_api_responses,
            )

            if last_stats.failed == 0:
                return last_stats

            # Only retry if failures are for models in current config
            current_model_failures = sum(
                ms.failed
                for model_name, ms in last_stats.model_stats.items()
                if model_name in current_model_names
            )

            if current_model_failures == 0:
                return last_stats

            if attempt < RUN_RETRY_MAX_ATTEMPTS:
                current_concurrency = max(
                    1, int(current_concurrency * RUN_RETRY_CONCURRENCY_FACTOR)
                )
                print(
                    f"\nRun had {current_model_failures} failure(s) for current models. "
                    f"Retrying in {RUN_RETRY_DELAY_SECONDS}s with concurrency={current_concurrency}..."
                )
                await asyncio.sleep(RUN_RETRY_DELAY_SECONDS)
            else:
                return last_stats

        except FatalBenchmarkError:
            raise
        except Exception as e:
            if attempt < RUN_RETRY_MAX_ATTEMPTS:
                current_concurrency = max(
                    1, int(current_concurrency * RUN_RETRY_CONCURRENCY_FACTOR)
                )
                print(
                    f"\nRun failed with error: {e}. "
                    f"Retrying in {RUN_RETRY_DELAY_SECONDS}s with concurrency={current_concurrency}..."
                )
                await asyncio.sleep(RUN_RETRY_DELAY_SECONDS)
            else:
                raise

    return last_stats
