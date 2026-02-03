"""Dry run preview functionality with Rich table formatting."""

import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from benchmark.checkpoint import initialize_checkpoint
from benchmark.config import (
    JUDGE_MODEL,
    JUDGE_MODEL_OPENROUTER,
    BenchmarkConfig,
    get_generations_for_failure_type,
    resolve_entry_configuration,
)
from benchmark.execution.judgment import get_judge_provider
from benchmark.exceptions import FatalBenchmarkError
from benchmark.provider_registry import resolve_model_generation_mode
from benchmark.prompts import (
    build_generation_prompt,
    build_judge_prompt,
    get_judge_system_prompt,
)
from benchmark.utils import truncate_middle
from benchmark.work_planner import (
    _build_work_queue,
)

console = Console()


def run_dry_run(
    valid_entries: list[dict[str, Any]],
    config: BenchmarkConfig,
    ignore_config_mismatch: bool = False,
) -> None:
    """Preview configuration and save prompts without API calls."""

    models = config.models
    output_file = config.output
    judge_provider = get_judge_provider()
    judge_model = (
        JUDGE_MODEL_OPENROUTER if judge_provider == "openrouter" else JUDGE_MODEL
    )
    concurrency = config.concurrency

    total_generations = sum(
        get_generations_for_failure_type(
            resolve_entry_configuration(e), config.generations
        )
        for e in valid_entries
    ) * len(models)

    console.print(
        "[bold cyan]Dry run mode: no network requests will be sent.[/bold cyan]"
    )
    if config.generations is not None:
        gen_desc = f"{config.generations} generation(s) per entry (global override)"
    else:
        gen_desc = "per-category defaults (cross_domain: 3, sycophancy: 3, beneficial: 1)"
    console.print(
        f"Found {len(valid_entries)} unique entries, {len(models)} model(s), "
        f"{gen_desc} = {total_generations} planned generations."
    )

    routing_rows = _summarize_model_routes(models)
    checkpoint, pending_work, completed_count = _preview_work_queue(
        valid_entries=valid_entries,
        config=config,
        ignore_config_mismatch=ignore_config_mismatch,
        judge_provider=judge_provider,
    )

    # Derive per-model statistics from pending_work
    total_gens_all_entries = sum(
        get_generations_for_failure_type(
            resolve_entry_configuration(e), config.generations
        )
        for e in valid_entries
    )
    total_by_model = {model.name: total_gens_all_entries for model in models}
    completed_by_model = {model.name: 0 for model in models}
    remaining_by_model = {model.name: 0 for model in models}

    for entry, model, _gen_idx in pending_work:
        remaining_by_model[model.name] += 1

    for model in models:
        model_name = model.name
        completed_by_model[model_name] = (
            total_by_model[model_name] - remaining_by_model[model_name]
        )

    total_completed = completed_count
    total_remaining = len(pending_work)

    if total_completed > 0:
        console.print(
            f"\n[yellow]Checkpoint found: {total_completed} generations already completed, {total_remaining} remaining[/yellow]"
        )

    preview_entries = 1
    entries_table = Table(
        title=f"Sample Entry (first of {len(valid_entries)})",
        show_lines=True,
        title_style="bold",
    )
    entries_table.add_column("Entry #", justify="right", style="cyan")
    entries_table.add_column("Hash", style="magenta")
    entries_table.add_column("Failure Type", style="red")
    entries_table.add_column("Query Preview", style="white")
    entries_table.add_column("Memories Preview", style="green")

    # Collect all unique failure_type values and find representative entries
    failure_types: set[str] = set()
    type_to_entry: dict[str, dict[str, Any]] = {}
    failure_counts: dict[str, int] = {}
    for entry in valid_entries:
        leak_type = resolve_entry_configuration(entry)
        failure_types.add(leak_type)
        if leak_type not in type_to_entry:
            type_to_entry[leak_type] = entry
        failure_counts[leak_type] = failure_counts.get(leak_type, 0) + 1

    for entry_idx, entry in enumerate(valid_entries[:preview_entries]):
        failure_type = resolve_entry_configuration(entry)

        mem_lines = [
            f"- {truncate_middle(memory, 160)}" for memory in entry["memories"][:5]
        ]
        if len(entry["memories"]) > 5:
            remaining = len(entry["memories"]) - 5
            mem_lines.append(f"... ({remaining} more)")
        mem_preview = "\n".join(mem_lines) if mem_lines else "—"

        entries_table.add_row(
            str(entry_idx + 1),
            entry["hash_id"],
            failure_type,
            truncate_middle(entry["query"], 200),
            mem_preview,
        )

    config_table = Table(
        title="Model Routing & Status", show_lines=False, title_style="bold"
    )
    config_table.add_column("Model", style="cyan")
    config_table.add_column("Provider", style="yellow")
    config_table.add_column("Mode", style="magenta")
    config_table.add_column("Total", justify="right", style="magenta")
    config_table.add_column("Completed", justify="right", style="green")
    config_table.add_column("Remaining", justify="right", style="blue")

    for model, provider_label, resolved_mode in routing_rows:
        model_name = model.name
        total = total_by_model[model_name]
        completed = completed_by_model[model_name]
        remaining = remaining_by_model[model_name]
        config_table.add_row(
            model_name,
            provider_label,
            resolved_mode,
            str(total),
            f"{completed} ({completed / total * 100:.0f}%)" if total > 0 else "0",
            str(remaining),
        )

    if judge_model:
        # Provider details are internal, not displayed
        config_table.add_row(f"{judge_model} (judge)", "—", "—", "—", "—")

    # Show model routing and sample entry
    console.print(config_table)
    console.print(entries_table)
    console.print("\n[bold]Entry Distribution by Failure Type:[/bold]")
    for leak_type in sorted(failure_types):
        count = failure_counts.get(leak_type, 0)
        console.print(f"  {leak_type}: {count}")
    console.print(f"[green]Concurrency:[/green] {concurrency}")
    if config.prompt_template:
        console.print(f"[green]Prompt Template:[/green] {config.prompt_template}")

    example_memory_response = "This is an example memory response (with memories)"

    # Save full prompts to markdown file
    dry_run_prompts_file = f"{output_file}.dry_run_prompts.md"
    try:
        Path(dry_run_prompts_file).parent.mkdir(parents=True, exist_ok=True)
        with open(dry_run_prompts_file, "w", encoding="utf-8") as f:
            f.write("# Dry Run Prompt Preview\n\n")
            f.write(
                f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n"
            )
            f.write("---\n\n")

            f.write("## Failure Type Configuration\n\n")
            f.write("**All failure_type values in input:**\n\n")
            for leak_type in sorted(failure_types):
                f.write(f"- `failure_type={leak_type}`\n")
            f.write("\n")
            f.write("---\n\n")

            # Generate prompts for each unique failure_type
            for leak_type in sorted(failure_types):
                entry = type_to_entry[leak_type]

                f.write(f"## Prompts for `failure_type={leak_type}`\n\n")

                # Build generator prompts
                prompt_template = config.prompt_template_content
                if models:
                    if prompt_template:
                        generator_system_prompt = build_generation_prompt(
                            entry["memories"], models[0].name, prompt_template
                        )
                    else:
                        generator_system_prompt = build_generation_prompt(
                            entry["memories"], models[0].name
                        )
                else:
                    generator_system_prompt = ""

                # Build judge prompts
                judge_system_prompt = get_judge_system_prompt(leak_type)
                judge_user_prompt = build_judge_prompt(
                    memories=entry["memories"],
                    query=entry["query"],
                    memory_response=example_memory_response,
                )

                if models:
                    f.write(
                        "<details>\n<summary><strong>Generator System Prompt</strong></summary>\n\n"
                    )
                    f.write(f"**Model:** `{models[0]}`\n\n")
                    f.write("```\n")
                    f.write(generator_system_prompt)
                    f.write("\n```\n\n")
                    f.write("</details>\n\n")
                    f.write("---\n\n")

                f.write(
                    "<details>\n<summary><strong>Generator User Prompt</strong></summary>\n\n"
                )
                f.write(f"*Using example entry (hash_id: `{entry['hash_id']}`)*\n\n")
                f.write("```\n")
                f.write(entry["query"])
                f.write("\n```\n\n")
                f.write("</details>\n\n")
                f.write("---\n\n")

                if judge_model:
                    f.write(
                        "<details>\n<summary><strong>Judge System Prompt</strong></summary>\n\n"
                    )
                    f.write(f"**Model:** `{judge_model}`\n\n")
                    f.write(f"**Failure Type:** `{leak_type}`\n\n")
                    f.write("```\n")
                    f.write(judge_system_prompt)
                    f.write("\n```\n\n")
                    f.write("</details>\n\n")
                    f.write("---\n\n")

                    f.write(
                        "<details>\n<summary><strong>Judge User Prompt</strong></summary>\n\n"
                    )
                    f.write(
                        f"*Using example entry (hash_id: `{entry['hash_id']}`) with example response*\n\n"
                    )
                    f.write("```\n")
                    f.write(judge_user_prompt)
                    f.write("\n```\n\n")
                    f.write("</details>\n\n")

                f.write("---\n\n")

            f.write(
                "**Note:** All judge user prompts include example responses generated during dry run, as no actual model generation occurs in dry run mode.\n"
            )

        console.print(
            f"\n[bold green]📄 Full prompts saved to:[/bold green] [cyan]{dry_run_prompts_file}[/cyan]"
        )
        console.print("[dim]   View this file to see all prompt sections[/dim]\n")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save prompts to file: {e}[/yellow]")

    console.print(
        f"[green]{sum(remaining_by_model.values())} generation(s) would be attempted during a live run.[/green]"
    )


def _format_provider_label(model) -> str:
    provider_override = model.api_params.get("provider") if model.api_params else None
    if (
        provider_override
        and isinstance(provider_override, dict)
        and "order" in provider_override
    ):
        override = ", ".join(provider_override["order"])
        return f"{model.provider} (override: {override})"
    return model.provider


def _summarize_model_routes(models):
    rows = []
    errors = []
    for model in models:
        try:
            resolved_mode = resolve_model_generation_mode(model)
        except FatalBenchmarkError as exc:
            errors.append(str(exc))
            resolved_mode = "invalid"
        provider_label = _format_provider_label(model)
        rows.append((model, provider_label, resolved_mode))

    if errors:
        console.print("")
        for message in errors:
            console.print(f"[red]Configuration error:[/red] {message}")
        raise FatalBenchmarkError(
            "Dry run aborted due to invalid model/provider configuration."
        )

    return rows


def _preview_work_queue(
    valid_entries,
    config: BenchmarkConfig,
    ignore_config_mismatch: bool = False,
    judge_provider: str | None = None,
):
    """Build pending work queue without writing checkpoint to disk."""

    checkpoint = initialize_checkpoint(
        valid_entries,
        config,
        ignore_config_mismatch=ignore_config_mismatch,
        judge_provider=judge_provider,
    )
    pending_work, completed_count = _build_work_queue(
        checkpoint, valid_entries, config, ignore_config_mismatch
    )

    return checkpoint, pending_work, completed_count
