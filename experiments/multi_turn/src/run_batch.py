"""
Batch-run multi-turn conversations for all samples in 150_samples.json.

Prompt routing:
- beneficial_memory_usage, cross_domain -> chat_prompt.txt + adapter.txt
- sycophancy -> sycophancy_chat_prompt.txt + adapter_sycophancy.txt

Usage:
  pip install -r requirements.txt
  cp .env.example .env   # then fill in OPENROUTER_API_KEY
  cd experiments/multi_turn
  python src/run_batch.py google/gemini-2.5-flash-lite
  python src/run_batch.py google/gemini-2.5-flash-lite --workers 8 --seed 42
  python src/run_batch.py MODEL --checkpoint-every 10   # default; use 0 for every sample
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from anthropic_target import (
    anthropic_target_client,
    messages_create_with_cached_system,
    uses_anthropic_direct_target,
)
from openai_target import (
    chat_completion_target,
    openai_target_client,
    usage_cached_tokens,
    uses_openai_direct_target,
)

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
PROMPTS_DIR = ROOT / "prompts"

DATASET_DEFAULT = str(ROOT / "data" / "samples_150.json")
EVAL_PROMPT = "eval_prompt.txt"
GATEKEEPER_PROMPT = "gate_llm.txt"
CHAT_DEFAULT = "chat_prompt.txt"
ADAPTER_DEFAULT = "adapter.txt"
CHAT_SYCOPHANCY = "sycophancy_chat_prompt.txt"
ADAPTER_SYCOPHANCY = "adapter_sycophancy.txt"

CHAT_MODEL = "moonshotai/kimi-k2-0905"
GATEKEEPER_MODEL = "openai/gpt-oss-20b"
MAX_RETRIES = 3
SIMULATOR_ACTIVATION_PROMPT = ""
SIMULATOR_TARGET_FROM_START = True


def clean_json(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def format_memories(memory_list: list) -> str:
    return "\n".join(f"- {m}" for m in memory_list)


def _completion_text(resp) -> str:
    """OpenAI-style completion body; API may return content=None."""
    msg = resp.choices[0].message
    raw = getattr(msg, "content", None)
    if raw is None:
        return ""
    return raw.strip() if isinstance(raw, str) else str(raw).strip()


def sanitize_model_for_filename(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_") or "model"


def atomic_write_json(path: Path, payload: dict, *, compact: bool) -> None:
    """Write JSON atomically (temp + os.replace) so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if compact:
                json.dump(payload, f, ensure_ascii=False)
            else:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def build_batch_payload(
    *,
    results: list,
    args: argparse.Namespace,
    dataset_path: Path,
    n: int,
    workers: int,
    sim_from_start: bool,
    retry_indices: set[int] | None,
    checkpoint_every: int,
) -> dict:
    errors: list[tuple[int, str]] = [
        (i, str(r.get("error", "")))
        for i, r in enumerate(results)
        if isinstance(r, dict) and r.get("error")
    ]
    meta: dict = {
        "target_model": args.target_model,
        "target_provider": (
            "anthropic"
            if uses_anthropic_direct_target(args.target_model)
            else (
                "openai"
                if uses_openai_direct_target(args.target_model)
                else "openrouter"
            )
        ),
        "openai_prompt_caching": uses_openai_direct_target(args.target_model),
        "anthropic_prompt_caching": uses_anthropic_direct_target(args.target_model),
        "dataset": str(dataset_path.resolve()),
        "num_samples": n,
        "workers": workers,
        "seed": args.seed,
        "sim_target_from_start": sim_from_start,
        "chat_model": CHAT_MODEL,
        "gatekeeper_model": GATEKEEPER_MODEL,
        "traces_completed": sum(1 for r in results if r is not None),
        "traces_pending": sum(1 for r in results if r is None),
        "checkpoint_every": checkpoint_every,
    }
    if retry_indices is not None:
        meta["retried_indices"] = sorted(retry_indices)
    return {
        "meta": meta,
        "errors": [{"sample_index": i, "detail": d} for i, d in errors],
        "traces": results,
    }


def prompts_for_failure_type(
    failure_type: str,
    loaded: dict[str, str],
) -> tuple[str, str, str, str]:
    """Returns (chat_raw, adapter_raw, chat_path, adapter_path)."""
    if failure_type == "sycophancy":
        return (
            loaded[CHAT_SYCOPHANCY],
            loaded[ADAPTER_SYCOPHANCY],
            CHAT_SYCOPHANCY,
            ADAPTER_SYCOPHANCY,
        )
    if failure_type in ("beneficial_memory_usage", "cross_domain"):
        return (
            loaded[CHAT_DEFAULT],
            loaded[ADAPTER_DEFAULT],
            CHAT_DEFAULT,
            ADAPTER_DEFAULT,
        )
    raise ValueError(
        f"Unsupported failure_type {failure_type!r}; "
        "expected beneficial_memory_usage, cross_domain, or sycophancy"
    )


def run_conversation_trace(
    sample_index: int,
    sample: dict,
    target_model: str,
    chat_prompt_raw: str,
    adapter_prompt_raw: str,
    chat_prompt_file: str,
    adapter_prompt_file: str,
    eval_prompt_raw: str,
    gatekeeper_prompt_raw: str,
    router_client: OpenAI,
    target_client: object,
    use_openai_target: bool,
    use_anthropic_target: bool,
    rng: random.Random,
    sim_target_from_start: bool,
) -> dict:
    broad_topic = sample["broad_topic"]
    target_query = sample["query"]
    memories_str = format_memories(sample["memories"])
    failure_type = sample.get("failure_type", "")

    target_sys_prompt = eval_prompt_raw.replace("{memories}", memories_str).replace(
        "{model_name}", target_model
    )
    target_history: list[dict] = [{"role": "system", "content": target_sys_prompt}]
    simulator_history: list[dict] = []
    total_turns = rng.randint(3, 5)

    turns_out: list[dict] = []

    for current_turn in range(1, total_turns + 1):
        turn_record: dict = {"turn": current_turn, "total_turns": total_turns}

        if current_turn < total_turns:
            if sim_target_from_start or current_turn == total_turns - 1:
                query_for_simulator = target_query
            else:
                query_for_simulator = (
                    "[Withheld — the exact final question is not shown yet. "
                    "Discuss and narrow the Broad Topic only; stay general and exploratory.]"
                )
            current_chat_prompt = (
                chat_prompt_raw.replace("{broad_topic}", broad_topic)
                .replace("{target_query}", query_for_simulator)
                .replace("{total_turns}", str(total_turns))
                .replace("{current_turn}", str(current_turn))
            )
            if not simulator_history:
                simulator_history.append({"role": "system", "content": current_chat_prompt})
            else:
                simulator_history[0]["content"] = current_chat_prompt

            passed_gatekeeper = False
            saved_fixes: list[str] = []
            final_user_message = ""
            attempts_log: list[dict] = []

            sim_turn_messages = [
                *simulator_history,
                {"role": "user", "content": SIMULATOR_ACTIVATION_PROMPT},
            ]
            for attempt in range(MAX_RETRIES):
                sim_resp = router_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=sim_turn_messages,
                    temperature=0.7,
                )
                candidate_message = _completion_text(sim_resp)

                gatekeeper_sys_prompt = (
                    gatekeeper_prompt_raw.replace("{target_query}", target_query)
                    .replace("{memories}", memories_str)
                    .replace("{simulated_message}", candidate_message)
                )
                gk_resp = router_client.chat.completions.create(
                    model=GATEKEEPER_MODEL,
                    messages=[{"role": "system", "content": gatekeeper_sys_prompt}],
                    temperature=0.1,
                )
                raw_gate_response = _completion_text(gk_resp)

                attempt_entry: dict = {
                    "attempt": attempt + 1,
                    "simulator_model": CHAT_MODEL,
                    "simulator_request_messages": [
                        dict(m) for m in sim_turn_messages
                    ],
                    "candidate_message": candidate_message,
                    "gatekeeper_model": GATEKEEPER_MODEL,
                    "gatekeeper_system_prompt": gatekeeper_sys_prompt,
                    "gatekeeper_raw_response": raw_gate_response,
                    "passed": False,
                    "parse_error": False,
                    "evaluation": None,
                }

                try:
                    evaluation = json.loads(clean_json(raw_gate_response))
                    attempt_entry["evaluation"] = evaluation
                    is_bad = evaluation.get("is_too_similar", False) or evaluation.get(
                        "contradicts_memory", False
                    )
                    if not is_bad:
                        passed_gatekeeper = True
                        final_user_message = candidate_message
                        attempt_entry["passed"] = True
                        attempts_log.append(attempt_entry)
                        break
                    reason = evaluation.get("reason", "Failed safety check.")
                    suggested_fix = (evaluation.get("suggested_fix", "") or "").strip()
                    if suggested_fix:
                        saved_fixes.append(suggested_fix)
                    attempt_entry["rejection_reason"] = reason
                    attempt_entry["suggested_fix"] = suggested_fix or None
                    fix_lines = [
                        "A gatekeeper rejected your last user message.",
                        f"Rejection reason: {reason}",
                    ]
                    if suggested_fix:
                        fix_lines.append(f"Suggested fix: {suggested_fix}")
                    fix_lines.append(
                        "Write a new user message that fixes this issue. "
                        "Follow the system instructions and output only the new message text."
                    )
                    sim_turn_messages.append(
                        {"role": "assistant", "content": candidate_message}
                    )
                    sim_turn_messages.append({"role": "user", "content": "\n".join(fix_lines)})
                except json.JSONDecodeError:
                    attempt_entry["parse_error"] = True
                    sim_turn_messages.append(
                        {"role": "assistant", "content": candidate_message}
                    )
                    sim_turn_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The gatekeeper returned invalid JSON, so your message could not be scored. "
                                "Produce a different user message that still follows the system instructions. "
                                "Output only the new message text."
                            ),
                        }
                    )
                attempts_log.append(attempt_entry)

            fallback_used = False
            fallback_kind = None
            if not passed_gatekeeper:
                fallback_used = True
                if saved_fixes:
                    final_user_message = rng.choice(saved_fixes)
                    fallback_kind = "saved_fixes"
                else:
                    final_user_message = (
                        f"That is interesting. Can you tell me more general facts about {broad_topic}?"
                    )
                    fallback_kind = "broad_topic_template"

            simulator_history.append({"role": "assistant", "content": final_user_message})
            user_message_to_send = final_user_message

            turn_record["precursor"] = {
                "simulator_system_prompt": current_chat_prompt,
                "query_shown_to_simulator": query_for_simulator,
                "gatekeeper_attempts": attempts_log,
                "final_user_message": final_user_message,
                "fallback_used": fallback_used,
                "fallback_kind": fallback_kind,
            }
        else:
            last_target_reply = ""
            for msg in reversed(target_history):
                if msg.get("role") == "assistant":
                    last_target_reply = msg.get("content", "") or ""
                    break
            adapter_sys = adapter_prompt_raw.replace("{target_query}", target_query).replace(
                "{target_reply}", last_target_reply or "(none)"
            )
            adapter_messages = [
                {"role": "system", "content": adapter_sys},
                {
                    "role": "user",
                    "content": "Output only the rewritten query, per your instructions.",
                },
            ]
            ad_resp = router_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=adapter_messages,
                temperature=0.5,
            )
            user_message_to_send = _completion_text(ad_resp)
            adapter_fallback = False
            if not user_message_to_send:
                user_message_to_send = target_query
                adapter_fallback = True

            turn_record["adapter"] = {
                "adapter_model": CHAT_MODEL,
                "adapter_prompt_file": adapter_prompt_file,
                "adapter_system_prompt": adapter_sys,
                "adapter_request_messages": adapter_messages,
                "original_target_query": target_query,
                "last_target_reply_before_adapter": last_target_reply,
                "paraphrased_user_message": user_message_to_send,
                "fallback_to_original_target_query": adapter_fallback,
            }

        tgt_messages = target_history + [
            {"role": "user", "content": user_message_to_send}
        ]
        if use_anthropic_target:
            tgt_resp = messages_create_with_cached_system(
                target_client,
                model=target_model,
                openai_style_messages=tgt_messages,
                temperature=0.7,
            )
        elif use_openai_target:
            tgt_resp = chat_completion_target(
                target_client,
                model=target_model,
                messages=tgt_messages,
                temperature=0.7,
                prompt_cache_key=f"multiturn-sample-{sample_index}",
            )
        else:
            tgt_resp = router_client.chat.completions.create(
                model=target_model,
                messages=tgt_messages,
                temperature=0.7,
            )
        target_reply = _completion_text(tgt_resp)

        tu = getattr(tgt_resp, "usage", None)
        prov = (
            "anthropic"
            if use_anthropic_target
            else ("openai" if use_openai_target else "openrouter")
        )
        turn_record["target"] = {
            "target_model": target_model,
            "target_provider": prov,
            "user_message": user_message_to_send,
            "assistant_reply": target_reply,
        }
        if tu is not None:
            usage_d: dict = {
                "prompt_tokens": getattr(tu, "prompt_tokens", None),
                "completion_tokens": getattr(tu, "completion_tokens", None),
            }
            if use_anthropic_target:
                usage_d["anthropic_cache_read_input_tokens"] = getattr(
                    tu, "cache_read_input_tokens", None
                )
                usage_d["anthropic_cache_creation_input_tokens"] = getattr(
                    tu, "cache_creation_input_tokens", None
                )
            elif use_openai_target:
                usage_d["cached_input_tokens"] = usage_cached_tokens(tgt_resp)
            turn_record["target"]["usage"] = usage_d

        target_history.append({"role": "user", "content": user_message_to_send})
        target_history.append({"role": "assistant", "content": target_reply})

        if current_turn < total_turns:
            simulator_history.append({"role": "user", "content": target_reply})

        turns_out.append(turn_record)

    return {
        "sample_index": sample_index,
        "failure_type": failure_type,
        "chat_prompt_file": chat_prompt_file,
        "adapter_prompt_file": adapter_prompt_file,
        "broad_topic": broad_topic,
        "target_query": target_query,
        "memories": sample.get("memories", []),
        "sim_threshold": sample.get("sim_threshold"),
        "models": {
            "target": target_model,
            "target_provider": (
                "anthropic"
                if use_anthropic_target
                else ("openai" if use_openai_target else "openrouter")
            ),
            "simulator_and_adapter": CHAT_MODEL,
            "gatekeeper": GATEKEEPER_MODEL,
        },
        "run_settings": {
            "total_turns": total_turns,
            "sim_target_from_start": sim_target_from_start,
            "max_gatekeeper_retries": MAX_RETRIES,
            "simulator_activation_prompt": SIMULATOR_ACTIVATION_PROMPT,
        },
        "turns": turns_out,
        "target_conversation_messages": [dict(m) for m in target_history],
        "simulator_conversation_messages": [dict(m) for m in simulator_history],
    }


def process_one(
    idx: int,
    sample: dict,
    target_model: str,
    loaded: dict[str, str],
    eval_raw: str,
    gk_raw: str,
    seed: int | None,
    sim_target_from_start: bool,
) -> dict:
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    router_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)
    use_openai = uses_openai_direct_target(target_model)
    use_anthropic = uses_anthropic_direct_target(target_model)
    if use_openai and use_anthropic:
        raise RuntimeError(
            "Target model matched both OpenAI-direct and Anthropic-direct routing"
        )
    if use_anthropic:
        target_client = anthropic_target_client()
    elif use_openai:
        target_client = openai_target_client()
    else:
        target_client = router_client
    rng = random.Random((seed if seed is not None else 0) * 1_000_003 + idx)
    ft = sample.get("failure_type", "")
    chat_raw, adapter_raw, chat_fn, adapter_fn = prompts_for_failure_type(ft, loaded)
    return run_conversation_trace(
        idx,
        sample,
        target_model,
        chat_raw,
        adapter_raw,
        chat_fn,
        adapter_fn,
        eval_raw,
        gk_raw,
        router_client,
        target_client,
        use_openai,
        use_anthropic,
        rng,
        sim_target_from_start,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Batch multi-turn runs for 150 samples.")
    p.add_argument(
        "target_model",
        help=(
            "Target: OpenRouter id (e.g. google/gemini-2.5-flash-lite), "
            "OpenAI id gpt-5.2-2025-12-11 (OPENAI_API_KEY + cache key), "
            "or Claude id claude-sonnet-4-5-20250929 (ANTHROPIC_API_KEY + cached system)"
        ),
    )
    p.add_argument(
        "--dataset",
        default=DATASET_DEFAULT,
        help="Path to JSON array of samples",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: batch_traces_<model>.json in script dir)",
    )
    p.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed (per-sample seed derived for reproducibility)",
    )
    p.add_argument(
        "--sim-target-from-start",
        action="store_true",
        help="Simulator sees real target query from turn 1 (same as main.py flag)",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON (smaller file)",
    )
    p.add_argument(
        "--merge-into",
        default="",
        help="Existing batch_traces JSON to patch (use with --retry-indices)",
    )
    p.add_argument(
        "--retry-indices",
        default="",
        help="Comma-separated 0-based indices to re-run only (requires --merge-into)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=(
            "Atomic save after this many samples finish in the current run (default 10). "
            "Use 0 or a negative value to save after every sample."
        ),
    )
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise SystemExit("Dataset must be a JSON array")

    eval_raw = (PROMPTS_DIR / EVAL_PROMPT).read_text(encoding="utf-8")
    gk_raw = (PROMPTS_DIR / GATEKEEPER_PROMPT).read_text(encoding="utf-8")
    loaded = {
        CHAT_DEFAULT: (PROMPTS_DIR / CHAT_DEFAULT).read_text(encoding="utf-8"),
        ADAPTER_DEFAULT: (PROMPTS_DIR / ADAPTER_DEFAULT).read_text(encoding="utf-8"),
        CHAT_SYCOPHANCY: (PROMPTS_DIR / CHAT_SYCOPHANCY).read_text(encoding="utf-8"),
        ADAPTER_SYCOPHANCY: (PROMPTS_DIR / ADAPTER_SYCOPHANCY).read_text(encoding="utf-8"),
    }

    retry_raw = (args.retry_indices or "").strip()
    merge_into_s = (args.merge_into or "").strip()
    merge_path = Path(merge_into_s) if merge_into_s else None
    if retry_raw and not merge_path:
        raise SystemExit("--retry-indices requires --merge-into <existing batch JSON>")
    retry_indices: set[int] | None = None
    if retry_raw:
        retry_indices = set()
        for part in retry_raw.split(","):
            part = part.strip()
            if not part:
                continue
            retry_indices.add(int(part))
    if merge_path and not merge_path.is_file():
        raise SystemExit(f"--merge-into file not found: {merge_path}")

    out_path = Path(args.out) if args.out else (
        merge_path
        if merge_path and retry_indices is not None
        else ROOT / "outputs" / f"batch_traces_{sanitize_model_for_filename(args.target_model)}.json"
    )

    n = len(dataset)
    workers = max(1, args.workers)
    sim_from_start = args.sim_target_from_start or SIMULATOR_TARGET_FROM_START

    results: list[dict | None]

    if merge_path and retry_indices is not None:
        prev = json.loads(merge_path.read_text(encoding="utf-8"))
        results = list(prev.get("traces") or [])
        if len(results) != n:
            raise SystemExit(
                f"merge-into traces length {len(results)} != dataset length {n}"
            )
        for j in retry_indices:
            if j < 0 or j >= n:
                raise SystemExit(f"retry index out of range: {j}")
        print(
            f"Retry-only: indices {sorted(retry_indices)}, merge into {merge_path}, out {out_path}"
        )
    else:
        results = [None] * n

    run_all = retry_indices is None
    indices_to_run: list[int]
    if run_all:
        indices_to_run = list(range(n))
    else:
        indices_to_run = sorted(retry_indices)

    print(f"Samples: {n}, target_model={args.target_model}, workers={workers}, out={out_path}")
    ce = args.checkpoint_every
    if ce <= 0:
        print("Checkpoint: atomic save after every sample (thread-safe).")
    else:
        print(
            f"Checkpoint: atomic save every {ce} completed sample(s), plus final (thread-safe)."
        )

    save_lock = threading.Lock()
    run_total = len(indices_to_run)
    completed_in_run = [0]

    def persist_locked() -> None:
        payload = build_batch_payload(
            results=results,
            args=args,
            dataset_path=dataset_path,
            n=n,
            workers=workers,
            sim_from_start=sim_from_start,
            retry_indices=retry_indices,
            checkpoint_every=ce,
        )
        atomic_write_json(out_path, payload, compact=args.compact)

    def save_checkpoint() -> None:
        with save_lock:
            persist_locked()

    save_checkpoint()

    def run_and_checkpoint(i: int) -> None:
        sample = dataset[i]
        if not isinstance(sample, dict):
            rec = {
                "sample_index": i,
                "error": "not an object",
                "failure_type": None,
            }
        else:
            try:
                rec = process_one(
                    i,
                    sample,
                    args.target_model,
                    loaded,
                    eval_raw,
                    gk_raw,
                    args.seed,
                    sim_from_start,
                )
            except Exception as e:
                rec = {
                    "sample_index": i,
                    "error": repr(e),
                    "failure_type": sample.get("failure_type"),
                }
        with save_lock:
            results[i] = rec
            completed_in_run[0] += 1
            c = completed_in_run[0]
            if ce <= 0 or c % ce == 0 or c == run_total:
                persist_locked()

    if workers == 1:
        for k, i in enumerate(indices_to_run):
            run_and_checkpoint(i)
            print(f"  done {k + 1}/{len(indices_to_run)} (index {i})", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_and_checkpoint, i): i for i in indices_to_run}
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                print(f"  done {done}/{len(indices_to_run)}", flush=True)

    if run_all:
        changed = False
        for i, r in enumerate(results):
            if r is None and i < len(dataset):
                results[i] = {
                    "sample_index": i,
                    "error": "skipped_invalid_sample",
                    "failure_type": None,
                }
                changed = True
        if changed:
            save_checkpoint()

    errors: list[tuple[int, str]] = [
        (i, str(r.get("error", "")))
        for i, r in enumerate(results)
        if isinstance(r, dict) and r.get("error")
    ]

    print(f"Wrote {out_path}")
    if errors:
        print(f"Completed with {len(errors)} errors (see meta.errors in output)")


if __name__ == "__main__":
    main()
