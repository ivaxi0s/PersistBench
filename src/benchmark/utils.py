"""Utility functions for benchmark tool."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from benchmark.config import ModelEntry
    from benchmark.types import GenerateResult

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from openai import AsyncOpenAI

from benchmark.checkpoint import Checkpoint, GenerationStatus, get_generation_status
from benchmark.config import get_generations_for_failure_type
from benchmark.exceptions import FatalBenchmarkError, NonRetryableError


def get_max_retries() -> int:
    """Get maximum retry attempts from MAX_RETRIES environment variable.

    Returns:
        Maximum number of retry attempts (defaults to 3 if not set).
    """
    return int(os.getenv("MAX_RETRIES", "3"))


def api_retry() -> Any:
    """Tenacity retry wrapper for API calls (shared across providers)."""
    return retry(
        stop=stop_after_attempt(get_max_retries()),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type((FatalBenchmarkError, NonRetryableError)),
        reraise=True,
    )


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse JSONL content into a list of objects."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


JUDGE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Detailed explanation of your evaluation and why you chose this score",
                },
                "score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Score from 1-5 as specified by the scoring rubric",
                },
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        },
    },
}


def extract_json_from_response(content: str) -> dict[str, Any]:
    """Extract JSON from response content with fallback strategies for malformed responses.

    Tries multiple strategies: parse as-is, regex extraction, and code block extraction.

    Raises:
        ValueError: If no valid JSON can be extracted from content.
    """
    content = content.strip()

    # Try parsing as-is first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to extract from code block
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # If all else fails, raise with the original content for debugging
    raise ValueError(
        f"Could not extract valid JSON from response. Content: {content[:500]}"
    )


def truncate_middle(text: str, max_chars: int = 200) -> str:
    """Truncate text preserving start and end."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    ellipsis = " … "
    half = (max_chars - len(ellipsis)) // 2
    if half <= 0:
        return text[:max_chars]
    return f"{text[:half]}{ellipsis}{text[-half:]}"


def generate_hash_id(memories: list[str], query: str) -> str:
    """Generate MD5 hash from memories and query.

    The hash uniquely identifies the data entry (memories + query combination),
    not the evaluation strategy. This design allows the same data to exist in
    checkpoints from different benchmark runs with different evaluation configs.

    Note: The validation layer prevents mixing evaluation configs for the same
    data within a single run - see load_and_validate_entries() for details.
    """
    content = json.dumps({"memories": sorted(memories), "query": query}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


class ModelStats:
    """Per-model generation statistics."""

    __slots__ = (
        "successful",
        "failed_generation",
        "failed_judge",
        "pending_judge",
        "pending_generation",
    )

    def __init__(self) -> None:
        self.successful: int = 0
        self.failed_generation: int = 0
        self.failed_judge: int = 0
        self.pending_judge: int = 0
        self.pending_generation: int = 0

    @property
    def failed(self) -> int:
        return self.failed_generation + self.failed_judge

    @property
    def pending(self) -> int:
        return self.pending_judge + self.pending_generation


class BenchmarkStats:
    """Aggregated benchmark statistics."""

    __slots__ = (
        "successful",
        "failed_generation",
        "failed_judge",
        "pending_judge",
        "pending_generation",
        "model_stats",
    )

    def __init__(self) -> None:
        self.successful: int = 0
        self.failed_generation: int = 0
        self.failed_judge: int = 0
        self.pending_judge: int = 0
        self.pending_generation: int = 0
        self.model_stats: dict[str, ModelStats] = {}

    @property
    def failed(self) -> int:
        return self.failed_generation + self.failed_judge

    @property
    def pending(self) -> int:
        return self.pending_judge + self.pending_generation

    @property
    def processed(self) -> int:
        return self.successful + self.failed


def get_benchmark_stats(checkpoint: Checkpoint) -> BenchmarkStats:
    """Compute benchmark statistics from checkpoint."""
    stats = BenchmarkStats()

    generations_override = checkpoint.get("metadata", {}).get("generations")
    expected_models: list[str] | None = None

    metadata_models = checkpoint.get("metadata", {}).get("models")
    if isinstance(metadata_models, list):
        model_names: list[str] = []
        for model_entry in metadata_models:
            if isinstance(model_entry, dict):
                name = model_entry.get("name")
                if isinstance(name, str):
                    model_names.append(name)
        if model_names:
            expected_models = model_names

    for hash_id, entry_data in checkpoint["entries"].items():
        models_to_check = expected_models or list(entry_data["results"].keys())
        entry_generations = get_generations_for_failure_type(
            entry_data.get("failure_type", "cross_domain"),
            generations_override,
        )
        for model in models_to_check:
            result_data = entry_data["results"].get(model, {})
            if model not in stats.model_stats:
                stats.model_stats[model] = ModelStats()

            ms = stats.model_stats[model]
            for gen_idx in range(entry_generations):
                status = get_generation_status(checkpoint, hash_id, model, gen_idx)
                if status == GenerationStatus.COMPLETED:
                    stats.successful += 1
                    ms.successful += 1
                elif status == GenerationStatus.NEEDS_JUDGE:
                    generations_list = result_data.get("generations", [])
                    entry_error = None
                    if gen_idx < len(generations_list):
                        entry_error = generations_list[gen_idx].get("error")

                    if entry_error:
                        stats.failed_judge += 1
                        ms.failed_judge += 1
                    else:
                        stats.pending_judge += 1
                        ms.pending_judge += 1
                elif status == GenerationStatus.NEEDS_GENERATION:
                    generations_list = result_data.get("generations", [])
                    has_error = False
                    if gen_idx < len(generations_list):
                        has_error = generations_list[gen_idx].get("error") is not None

                    if has_error:
                        stats.failed_generation += 1
                        ms.failed_generation += 1
                    else:
                        stats.pending_generation += 1
                        ms.pending_generation += 1

    return stats


def print_benchmark_summary(
    checkpoint: Checkpoint,
    output_file: Path,
    skip_generation: bool = False,
    skip_judge: bool = False,
) -> BenchmarkStats:
    """Print summary statistics for completed benchmark run.

    Args:
        skip_generation: True when running judge-only (benchmark judge).
        skip_judge: True when running generation-only (benchmark generate).

    Returns:
        Computed benchmark stats for this checkpoint.
    """
    if skip_generation:
        rerun_cmd = "benchmark judge"
    elif skip_judge:
        rerun_cmd = "benchmark generate"
    else:
        rerun_cmd = "benchmark run"

    stats = get_benchmark_stats(checkpoint)

    if skip_judge and not skip_generation:
        summary_title = "Generation Summary"
    elif skip_generation and not skip_judge:
        summary_title = "Judge Summary"
    else:
        summary_title = "Benchmark Summary"

    print(f"\n{summary_title}")
    print(f"Total generations processed: {stats.processed}")
    print(f"Successful: {stats.successful}")
    print(f"Failed: {stats.failed}")
    if stats.pending_generation > 0 and stats.pending_judge > 0:
        print(
            f"Pending: {stats.pending}  ({stats.pending_generation} missing response, {stats.pending_judge} awaiting judge)"
        )
    elif stats.pending > 0:
        print(f"Pending: {stats.pending}")
    else:
        print("Pending: 0")

    if stats.processed > 0:
        success_rate = (stats.successful / stats.processed) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print("\nPer-model breakdown:")
    for model, ms in stats.model_stats.items():
        print(
            f"  {model}: {ms.successful} successful, {ms.failed} failed, {ms.pending} pending"
        )

    if stats.failed > 0:
        print("\nFailed Entries:")
        print(f"  {stats.failed} generation(s) failed during this run")
        print("  Failed entries are stored in the checkpoint with an 'error' field")
        print(f"  To identify failures, search for '\"error\":' in {output_file}")
        print(
            f"  Or use: jq '.entries[].results[].generations[] | select(.error != null)' {output_file}"
        )
        print(f"  Rerun with: {rerun_cmd} {output_file}")

    if stats.pending_generation > 0:
        active_batch_jobs = (
            checkpoint.get("metadata", {}).get("batch_jobs", {}).get("generation", {})
        )
        metadata_models = checkpoint.get("metadata", {}).get("models")
        expected_models = (
            {
                m.get("name")
                for m in metadata_models
                if isinstance(m, dict) and isinstance(m.get("name"), str)
            }
            if isinstance(metadata_models, list)
            else set()
        )
        relevant_batch_jobs = (
            {k: v for k, v in active_batch_jobs.items() if k in expected_models}
            if isinstance(active_batch_jobs, dict) and expected_models
            else (active_batch_jobs if isinstance(active_batch_jobs, dict) else {})
        )
        if relevant_batch_jobs:
            job_models = ", ".join(sorted(relevant_batch_jobs.keys()))
            print("\nBatch Jobs In Progress:")
            print(
                f"  {stats.pending_generation} generation(s) awaiting batch job completion ({job_models})"
            )
            print(f"  Rerun with: {rerun_cmd} {output_file}")
        else:
            print("\nMissing Responses:")
            print(
                f"  {stats.pending_generation} generation(s) have no model response yet"
            )
            if skip_generation:
                print(
                    f"  These cannot be judged — run `benchmark generate {output_file}` first"
                )
            else:
                print(f"  Rerun with: {rerun_cmd} {output_file}")

    if stats.pending_judge > 0:
        print("\nAwaiting Judge:")
        print(
            f"  {stats.pending_judge} generation(s) have responses but no judge score"
        )
        print(f"  Rerun with: benchmark judge {output_file}")

    # Final status banner
    if skip_judge and not skip_generation:
        # Generation-only: only generation failures matter (judge errors are irrelevant).
        generation_complete = (
            stats.failed_generation == 0 and stats.pending_generation == 0
        )
        if generation_complete:
            print("\n" + "=" * 50)
            print("  GENERATION COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print("  All generations have model responses (ready for judging).")
            print(f"  Results saved to {output_file}")
            print("=" * 50)
        elif stats.failed_generation > 0:
            print("\n" + "-" * 50)
            print(f"  GENERATION COMPLETED WITH {stats.failed_generation} FAILURE(S)")
            print("-" * 50)
            print(f"  Results saved to {output_file}")
            print("-" * 50)
        else:
            print(f"\nGeneration incomplete. Results saved to {output_file}")
    elif skip_generation and not skip_judge:
        # Judge-only: should have no pending generation and no pending judge on success.
        if stats.failed == 0 and stats.pending == 0:
            print("\n" + "=" * 50)
            print("  JUDGING COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print(f"  All {stats.successful} generations were judged without errors")
            print(f"  Results saved to {output_file}")
            print("=" * 50)
        elif stats.failed > 0:
            print("\n" + "-" * 50)
            print(f"  JUDGING COMPLETED WITH {stats.failed} FAILURE(S)")
            print("-" * 50)
            print(f"  Results saved to {output_file}")
            print("-" * 50)
        elif stats.pending > 0:
            print(f"\nJudging incomplete. Results saved to {output_file}")
    else:
        if stats.failed == 0 and stats.pending == 0:
            print("\n" + "=" * 50)
            print("  BENCHMARK COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print(f"  All {stats.successful} generations completed without errors")
            print(f"  Results saved to {output_file}")
            print("=" * 50)
        elif stats.failed > 0:
            print("\n" + "-" * 50)
            print(f"  BENCHMARK COMPLETED WITH {stats.failed} FAILURE(S)")
            print("-" * 50)
            print(f"  Results saved to {output_file}")
            print("-" * 50)
        elif stats.pending > 0:
            print(f"\nBenchmark incomplete. Results saved to {output_file}")

    return stats


VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", "global")
DEFAULT_SERVICE_ACCOUNT_PATH = "service_account.json"


def get_vertex_project_id() -> str:
    """Get Vertex AI project ID from service account or env var."""
    if project_id := os.getenv("VERTEXAI_PROJECT"):
        return project_id

    # Extract from service account JSON
    service_account_path = os.getenv(
        "VERTEXAI_SERVICE_ACCOUNT_PATH", DEFAULT_SERVICE_ACCOUNT_PATH
    )
    try:
        with open(service_account_path, "r") as f:
            service_account_data = json.load(f)
            if project_id := service_account_data.get("project_id"):
                return project_id
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    raise ValueError(
        "Could not determine Vertex AI project ID. "
        "Set VERTEXAI_PROJECT env var or ensure service_account.json contains project_id"
    )


def get_vertex_ai_base_url(location: str | None = None) -> str:
    """Get Vertex AI base URL for a specific location.

    Args:
        location: GCP location (e.g., "us-central1", "global"). Falls back to env var or default.
    """
    project_id = get_vertex_project_id()
    loc = location or VERTEX_LOCATION

    # Global endpoint has no location prefix in hostname
    if loc == "global":
        hostname = "aiplatform.googleapis.com"
    else:
        hostname = f"{loc}-aiplatform.googleapis.com"

    return (
        f"https://{hostname}/v1/projects/{project_id}/locations/{loc}/endpoints/openapi"
    )


def get_vertex_ai_client(location: str | None = None) -> AsyncOpenAI:
    """Get authenticated Vertex AI client using service account.

    Args:
        location: GCP location (e.g., "us-central1"). Falls back to env var or default.
    """
    if not os.getenv("VERTEXAI_SERVICE_ACCOUNT_PATH"):
        raise FatalBenchmarkError(
            "VERTEXAI_SERVICE_ACCOUNT_PATH is not set, point it to the Vertex AI service account credentials fil path"
        )
    service_account_path = os.getenv("VERTEXAI_SERVICE_ACCOUNT_PATH")
    credentials = Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    auth_request = Request()
    credentials.refresh(auth_request)
    client = AsyncOpenAI(
        base_url=get_vertex_ai_base_url(location),
        api_key=credentials.token,
    )
    return client


SUCCESS_FINISH_REASONS = {"stop", "length"}

OPENAI_SDK_PARAMS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
}


_REASONING_TAG_PATTERN = re.compile(
    r"<(think|thinking|reasoning|thought|reflection)>"
    r".*?"
    r"</\1>",
    re.DOTALL,
)


def strip_reasoning_tags(content: str) -> tuple[str, str | None]:
    """Strip common reasoning/thinking XML tags from model output.

    Returns (cleaned_content, extracted_reasoning). If no tags are found,
    returns the original content unchanged and None for reasoning.
    """
    matches = _REASONING_TAG_PATTERN.findall(content)
    if not matches:
        return content, None

    # Extract all reasoning text before stripping
    reasoning_parts = []
    for m in _REASONING_TAG_PATTERN.finditer(content):
        reasoning_parts.append(m.group(0))

    cleaned = _REASONING_TAG_PATTERN.sub("", content).strip()
    return cleaned, "\n".join(reasoning_parts)


async def openai_compat_generate(
    client: AsyncOpenAI,
    model: "ModelEntry",
    system_prompt: str,
    user_message: str,
) -> "GenerateResult":
    """Shared generation logic for OpenAI-compatible APIs.

    Routes api_params to either direct kwargs or extra_body based on whether
    they are recognized by the OpenAI SDK.
    """
    params: dict[str, Any] = {
        "model": model.name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    extra_body: dict[str, Any] = {}
    if model.api_params:
        for k, v in model.api_params.items():
            if k in OPENAI_SDK_PARAMS:
                params.setdefault(k, v)
            else:
                extra_body[k] = v
    if extra_body:
        params["extra_body"] = extra_body

    response = await client.chat.completions.create(**params)

    if not response.choices:
        return {
            "response": "",
            "raw_api_response": response.model_dump(mode="json"),
        }

    choice = response.choices[0]

    if hasattr(choice.message, "refusal") and choice.message.refusal:
        raise NonRetryableError(f"Model refused: {choice.message.refusal}")

    if choice.finish_reason and choice.finish_reason not in SUCCESS_FINISH_REASONS:
        raise RuntimeError(f"Unsuccessful finish_reason: {choice.finish_reason}")

    return {
        "response": choice.message.content or "",
        "raw_api_response": response.model_dump(mode="json"),
    }
