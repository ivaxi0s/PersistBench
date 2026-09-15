"""
Flatten multi_turn batch_traces_*.json files into one judge-ready JSON.

Each target turn becomes one entry: (memories, query=user_message, response=assistant_reply).
PersistBench expects every config model and the right number of generation slots per failure_type
(1 for beneficial_memory_usage, 3 for sycophancy/cross_domain); entries are padded with error-marked
placeholders for other models and extra slots (multi-turn export only has generation 0).

Usage:
  cd experiments/multi_turn
  python src/to_judge_format.py
  python src/to_judge_format.py --inputs outputs/batch_traces_gpt*.json --out outputs/my_judge.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
DEFAULT_OUT = ROOT / "outputs" / "batch_traces_all_models_for_judging.json"

# Match legacy judge filenames where OpenAI snapshot IDs are shortened.
DEFAULT_RESULT_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.2-2025-12-11": "gpt-5.2",
}


def _mode_for_provider(provider: str) -> str:
    if provider in ("openai", "anthropic"):
        return "batch"
    return "sequential"


def _infer_provider(meta: dict, trace: dict) -> str:
    p = (meta.get("target_provider") or "").strip()
    if p:
        return p
    p = (trace.get("models") or {}).get("target_provider") or ""
    if p:
        return str(p).strip()
    for turn in trace.get("turns") or []:
        t = turn.get("target") or {}
        p = (t.get("target_provider") or "").strip()
        if p:
            return p
    return "openrouter"


def _entry_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_persistbench_config(
    models: list[dict[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Fields expected by PersistBench BenchmarkConfig (Pydantic)."""
    return {
        "models": models,
        # Omit "judge": loader deprecates non-null judge and mutates data.
        "judge_provider": "openrouter",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "store_raw_api_responses": False,
        "generations": None,
        "concurrency": 20,
        "limit": None,
        "batch_poll_timeout_minutes": 25,
        "prompt_template": None,
        "prompt_template_content": None,
    }


def expected_generation_count(failure_type: str) -> int:
    """Align with PersistBench / expand_main_output_with_thresholds."""
    if failure_type == "beneficial_memory_usage":
        return 1
    return 3


def _generation_dict(
    *,
    generation_index: int,
    response: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "generation_index": generation_index,
        "error": error,
        "memory_response": response,
        "memory_raw_api_response": {},
        "judge": None,
    }


def _one_generation(*, response: str, error: str | None) -> dict[str, Any]:
    return _generation_dict(generation_index=0, response=response, error=error)


# PersistBench validates every config model × every generation slot; combined file only
# has one real model per entry (different conversations per model run).
_ERR_OTHER_MODEL = "multi_turn_flat_not_applicable_other_model"
_ERR_EXTRA_SLOT = "multi_turn_flat_only_generation_0_collected"


def pad_entries_for_persistbench(
    entries: dict[str, Any],
    model_names: list[str],
) -> None:
    """Mutate entries so each has all models and full generations[] per failure_type."""
    for ent in entries.values():
        res = ent.get("results") or {}
        if len(res) != 1:
            raise SystemExit(
                "Expected one model key per entry before padding; got "
                f"{list(res.keys())}"
            )
        (only_model,) = res.keys()
        if only_model not in model_names:
            raise SystemExit(f"Entry references unknown model {only_model!r}")

        n = expected_generation_count(ent.get("failure_type") or "")
        raw_gens = (res[only_model].get("generations") or [])[:]
        by_idx: dict[int, dict[str, Any]] = {}
        for g in raw_gens:
            idx = int(g.get("generation_index", 0))
            by_idx[idx] = g

        primary: list[dict[str, Any]] = []
        for i in range(n):
            if i in by_idx:
                g = dict(by_idx[i])
                g["generation_index"] = i
                primary.append(g)
            elif i == 0 and raw_gens:
                g = dict(raw_gens[0])
                g["generation_index"] = 0
                primary.append(g)
            else:
                primary.append(
                    _generation_dict(
                        generation_index=i,
                        response="",
                        error=_ERR_EXTRA_SLOT,
                    )
                )

        new_results: dict[str, Any] = {}
        for m in model_names:
            if m == only_model:
                new_results[m] = {"generations": primary}
            else:
                new_results[m] = {
                    "generations": [
                        _generation_dict(
                            generation_index=i,
                            response="",
                            error=_ERR_OTHER_MODEL,
                        )
                        for i in range(n)
                    ]
                }
        ent["results"] = new_results


def traces_to_entries(
    batch_path: Path,
    *,
    result_model_aliases: dict[str, str],
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    data = json.loads(batch_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    raw_model = meta.get("target_model") or ""
    if not raw_model and data.get("traces"):
        raw_model = (data["traces"][0].get("models") or {}).get("target") or ""
    result_model = result_model_aliases.get(raw_model, raw_model)
    provider = _infer_provider(meta, data.get("traces", [{}])[0] if data.get("traces") else {})

    out: list[tuple[str, dict[str, Any]]] = []
    batch_stem = batch_path.stem

    for trace in data.get("traces") or []:
        memories = trace.get("memories") or []
        failure_type = trace.get("failure_type") or ""
        sample_index = trace.get("sample_index")
        for turn in trace.get("turns") or []:
            tgt = turn.get("target") or {}
            query = (tgt.get("user_message") or "").strip()
            reply = tgt.get("assistant_reply")
            if reply is None:
                reply = ""
            else:
                reply = str(reply).strip()
            err: str | None = None
            if not tgt:
                err = "missing_target_block"
            elif not reply:
                err = "empty_assistant_reply"

            eid = _entry_id(
                {
                    "batch_stem": batch_stem,
                    "failure_type": failure_type,
                    "memories": memories,
                    "model": result_model,
                    "query": query,
                    "sample_index": sample_index,
                    "turn": turn.get("turn"),
                }
            )
            entry = {
                "failure_type": failure_type,
                "memories": memories,
                "query": query,
                "results": {
                    result_model: {
                        "generations": [_one_generation(response=reply, error=err)],
                    }
                },
            }
            out.append((eid, entry))

    model_info = {
        "name": result_model,
        "provider": provider,
        "mode": _mode_for_provider(provider),
        "api_params": None,
        "base_url": None,
        "api_key_env": None,
    }
    return out, model_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="batch_traces JSON paths (default: all batch_traces_*.json in this directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path",
    )
    parser.add_argument(
        "--no-model-aliases",
        action="store_true",
        help="Use raw target_model strings as results keys (no gpt-5.2 shortening)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Minified JSON (smaller file)",
    )
    parser.add_argument(
        "--config-input",
        type=Path,
        default=None,
        help=(
            "PersistBench config.input path string (default: same as --out, "
            "the merged entries file)"
        ),
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help=(
            "PersistBench config.output path string (default: <out_stem>_judged.json "
            "next to --out)"
        ),
    )
    args = parser.parse_args()

    aliases: dict[str, str] = {} if args.no_model_aliases else dict(DEFAULT_RESULT_MODEL_ALIASES)

    if args.inputs:
        paths = [Path(p).resolve() for p in args.inputs]
    else:
        search_dirs = [Path.cwd() / "outputs", ROOT / "outputs", DIR]
        paths = []
        for d in search_dirs:
            if d.is_dir():
                paths.extend(
                    p for p in d.glob("batch_traces_*.json")
                    if "for_judging" not in p.stem
                )
        paths = sorted(set(paths))

    if not paths:
        raise SystemExit("No input files matched.")

    entries: dict[str, Any] = {}
    models_by_name: dict[str, dict[str, Any]] = {}

    for p in paths:
        pairs, model_info = traces_to_entries(p, result_model_aliases=aliases)
        name = model_info["name"]
        if name in models_by_name and models_by_name[name] != model_info:
            raise SystemExit(f"Conflicting model_info for {name!r} from different batches")
        models_by_name[name] = model_info
        for eid, ent in pairs:
            if eid in entries:
                raise SystemExit(f"Duplicate entry id {eid} (from {p.name})")
            entries[eid] = ent

    model_names = list(models_by_name.keys())
    pad_entries_for_persistbench(entries, model_names)

    out_path = args.out.resolve()
    config_input = (args.config_input or out_path).resolve()
    config_output = (args.config_output or out_path.with_name(f"{out_path.stem}_judged.json")).resolve()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    models_list = list(models_by_name.values())
    payload = {
        "metadata": {
            "benchmark_name": "PersistBench_multi_turn_flat",
            "timestamp": ts,
            "total_entries": len(entries),
            "models": models_list,
            "judge_model": "moonshotai/kimi-k2-thinking",
            "judge_provider": "openrouter",
            "store_raw_api_responses": False,
            "generations": None,
            "concurrency": 20,
            "prompt_template": None,
            "batch_jobs": {"generation": {}, "judgment": None},
        },
        "entries": entries,
        "config": build_persistbench_config(
            models_list,
            input_path=config_input,
            output_path=config_output,
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if args.compact:
            json.dump(payload, f, ensure_ascii=False)
        else:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {out_path} ({len(entries)} entries, {len(models_by_name)} models)")


if __name__ == "__main__":
    main()
