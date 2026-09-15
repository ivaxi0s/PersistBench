"""
Analyze PersistBench-style judged JSON for multi-turn memory traces.

- Aligns each judged entry to a batch trace by (memories, user query) so turn order is correct.
- Treats the last pipeline turn (adapter / target query) as the *final* turn for reporting.
- **Marginal table (printed first):** for each model, every judged row where that model's slot has
  no `error` and a `judge.score`. Those rows differ per model (each JSON entry has one “live”
  model and four placeholders), so ~588 successes each in PersistBench is **not** 588 identical
  queries per model.
- **Trace-aligned table:** joins to `--traces` turn order; on each `(memories, query)` you usually
  only see a score for the model that was “live” on that judged row (often the same as the trace’s
  target model), so use this for precursor vs **final-turn** analysis for that reference model.

Duplicate `(memories, query)` rows (rare in this export) are merged so each model’s non-error
block is kept when present.

PersistBench-style runners often report per-model "successful" vs "failed" over *every* generation
cell in the JSON (including padded slots, error-marked placeholders, and skipped paths after custom
benchmark changes). A large "failed" count alongside ~one successful judgment per real conversation
turn is expected and does not mean the run broke; use alignment + non-null `judge` (as this script
does) for analysis.

Usage:
  cd experiments/multi_turn
  python src/analyze_judged.py \
    --judged outputs/batch_traces_all_models_judged.json \
    --traces outputs/batch_traces_gpt-5.2-2025-12-11.json

  python src/analyze_judged.py --judged ... --traces ... --out-csv summary.csv
  python src/analyze_judged.py --judged ... --traces ... --out-html report.html
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def memories_key(memories: list[str]) -> tuple[str, ...]:
    return tuple(memories)


def extract_score(gen: dict[str, Any]) -> float | None:
    if gen.get("error"):
        return None
    j = gen.get("judge")
    if not isinstance(j, dict):
        return None
    s = j.get("score")
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _fmt_mean(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


def coalesce_duplicate_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge duplicate (memories, query) rows.

    Multi-model flat judging stores **several** JSON entries with the same (memories, query):
    each copy has exactly one model with a real generation (no `error`) and the other models
    are `multi_turn_flat_not_applicable_other_model` placeholders. PersistBench reports ~588
    successes per model across those copies; a naive merge that keeps only the first copy
    would show 0 scores for every other model. We take each model's `results[model]` block
    from whichever duplicate has that model without `error` (prefer one with a judge score).
    """
    if len(entries) == 1:
        return entries[0]
    base = json.loads(json.dumps(entries[0]))
    all_models: set[str] = set()
    for ent in entries:
        all_models.update((ent.get("results") or {}).keys())

    merged: dict[str, Any] = {}
    for ent in entries:
        for model, block in (ent.get("results") or {}).items():
            g = (block.get("generations") or [{}])[0]
            if g.get("error"):
                continue
            prev = merged.get(model)
            if prev is None:
                merged[model] = json.loads(json.dumps(block))
                continue
            g_prev = (prev.get("generations") or [{}])[0]
            if extract_score(g_prev) is None and extract_score(g) is not None:
                merged[model] = json.loads(json.dumps(block))

    for model in all_models:
        if model not in merged:
            for ent in entries:
                if model in (ent.get("results") or {}):
                    merged[model] = json.loads(json.dumps(ent["results"][model]))
                    break

    base["results"] = merged
    return base


def build_judged_lookup(judged: dict[str, Any]) -> dict[tuple[tuple[str, ...], str], dict[str, Any]]:
    raw: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for _eid, ent in judged["entries"].items():
        mem_k = memories_key(ent.get("memories") or [])
        q = (ent.get("query") or "").strip()
        raw[(mem_k, q)].append(ent)
    return {k: coalesce_duplicate_entries(v) for k, v in raw.items()}


def load_traces(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces") or []
    if not isinstance(traces, list):
        raise SystemExit("traces file missing 'traces' array")
    return traces


def align_trace_to_judged(
    trace: dict[str, Any],
    lookup: dict[tuple[tuple[str, ...], str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ordered rows: one per turn with models -> scores, is_final, target_query."""
    mem_k = memories_key(trace.get("memories") or [])
    target_query = (trace.get("target_query") or "").strip()
    rows: list[dict[str, Any]] = []
    for turn in trace.get("turns") or []:
        tgt = turn.get("target") or {}
        q = (tgt.get("user_message") or "").strip()
        tnum = turn.get("turn")
        total = turn.get("total_turns")
        is_final = tnum == total and total is not None
        key = (mem_k, q)
        ent = lookup.get(key)
        scores: dict[str, float | None] = {}
        models: list[str] = []
        if ent:
            for model, block in ent.get("results", {}).items():
                models.append(model)
                gens = block.get("generations") or []
                g0 = gens[0] if gens else {}
                scores[model] = extract_score(g0)
        rows.append(
            {
                "sample_index": trace.get("sample_index"),
                "failure_type": trace.get("failure_type"),
                "turn": tnum,
                "total_turns": total,
                "is_final": is_final,
                "is_precursor": not is_final,
                "query": q,
                "target_query": target_query,
                "matched_judged": ent is not None,
                "scores_by_model": scores,
            }
        )
    return rows


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def summarize_by_failure_type(
    aligned_all: list[list[dict[str, Any]]],
    model_names: list[str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for conv in aligned_all:
        if not conv:
            continue
        ft = conv[0].get("failure_type") or "unknown"
        groups[str(ft)].append(conv)
    return {ft: summarize_models(convs, model_names) for ft, convs in groups.items()}


def summarize_models(
    aligned_all: list[list[dict[str, Any]]],
    model_names: list[str],
) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = {
        m: {"all": [], "final": [], "precursor": []} for m in model_names
    }
    n_matched_turns = 0
    n_unmatched = 0
    for conv in aligned_all:
        for row in conv:
            if not row["matched_judged"]:
                n_unmatched += 1
                continue
            n_matched_turns += 1
            for m in model_names:
                s = row["scores_by_model"].get(m)
                if s is None:
                    continue
                by_model[m]["all"].append(s)
                if row["is_final"]:
                    by_model[m]["final"].append(s)
                else:
                    by_model[m]["precursor"].append(s)

    per_model = {}
    for m in model_names:
        buckets = by_model[m]
        per_model[m] = {
            "n_scored_all": len(buckets["all"]),
            "mean_all": mean(buckets["all"]),
            "n_scored_final": len(buckets["final"]),
            "mean_final": mean(buckets["final"]),
            "n_scored_precursor": len(buckets["precursor"]),
            "mean_precursor": mean(buckets["precursor"]),
        }
    return {
        "per_model": per_model,
        "n_conversations": len(aligned_all),
        "n_matched_turns": n_matched_turns,
        "n_unmatched_turns": n_unmatched,
    }


def marginal_stats_judged_file(
    judged: dict[str, Any], model_names: list[str]
) -> dict[str, dict[str, Any]]:
    """
    For each model, collect scores on every judged entry where that model's generation
    has no `error` and has a judge score. These rows are **not** the same 588 rows for
    each model: each entry has one 'live' model and four placeholders, so models accrue
    scores on different (memories, query) rows. Matches PersistBench per-model success counts.
    """
    buckets: dict[str, list[float]] = {m: [] for m in model_names}
    for ent in judged["entries"].values():
        res = ent.get("results") or {}
        for m in model_names:
            block = res.get(m)
            if not block:
                continue
            g = (block.get("generations") or [{}])[0]
            s = extract_score(g)
            if s is not None:
                buckets[m].append(s)
    return {
        m: {"n": len(buckets[m]), "mean": mean(buckets[m])} for m in model_names
    }


def print_marginal_table(marginal: dict[str, dict[str, Any]], model_names: list[str]) -> None:
    print("\n## Marginal: all judged rows where that model is non-error (full file)\n")
    print("| model | n scored | mean score |")
    print("| --- | ---: | ---: |")
    for m in sorted(model_names):
        r = marginal[m]
        mf = _fmt_mean(r["mean"]) if r["mean"] is not None else ""
        print(f"| {m} | {r['n']} | {mf} |")
    print(
        "\n*These counts should line up with PersistBench ~588 successes per model when judging "
        "completed. They are **not** comparable on the same (query) row unless your judged JSON "
        "has merged duplicates per (memories, query).*\n"
    )


def print_markdown_table(summary: dict[str, Any], model_names: list[str]) -> None:
    print("\n## Per-model aggregates (matched turns only)\n")
    print(
        "| model | n (all) | mean all | n (precursor) | mean precursor | "
        "n (final) | mean final |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in sorted(model_names):
        r = summary["per_model"][m]
        mf = _fmt_mean(r["mean_final"])
        mf_cell = f"**{mf}**" if mf else ""
        print(
            f"| {m} | {r['n_scored_all']} | {_fmt_mean(r['mean_all'])} | "
            f"{r['n_scored_precursor']} | {_fmt_mean(r['mean_precursor'])} | "
            f"{r['n_scored_final']} | {mf_cell} |"
        )
    print(
        f"\nConversations: {summary['n_conversations']}, "
        f"matched turns: {summary['n_matched_turns']}, "
        f"unmatched turns: {summary['n_unmatched_turns']}\n"
    )


def write_csv(path: Path, summary: dict[str, Any], model_names: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "n_all",
                "mean_all",
                "n_precursor",
                "mean_precursor",
                "n_final",
                "mean_final",
            ]
        )
        for m in sorted(model_names):
            r = summary["per_model"][m]
            w.writerow(
                [
                    m,
                    r["n_scored_all"],
                    r["mean_all"] if r["mean_all"] is not None else "",
                    r["n_scored_precursor"],
                    r["mean_precursor"] if r["mean_precursor"] is not None else "",
                    r["n_scored_final"],
                    r["mean_final"] if r["mean_final"] is not None else "",
                ]
            )


def write_html_report(
    path: Path,
    summary: dict[str, Any],
    model_names: list[str],
    judged_meta: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for m in sorted(model_names):
        r = summary["per_model"][m]
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(m)}</td>"
            f"<td>{r['n_scored_all']}</td>"
            f"<td>{r['mean_all'] if r['mean_all'] is not None else ''}</td>"
            f"<td>{r['n_scored_precursor']}</td>"
            f"<td>{r['mean_precursor'] if r['mean_precursor'] is not None else ''}</td>"
            f"<td>{r['n_scored_final']}</td>"
            f"<td><strong>{r['mean_final'] if r['mean_final'] is not None else ''}</strong></td>"
            "</tr>"
        )
    meta_b = ""
    if judged_meta:
        meta_b = f"<p><small>{html.escape(json.dumps(judged_meta, indent=2)[:2000])}</small></p>"
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Multi-turn judged summary</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1200px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f4f4; }}
    caption {{ font-weight: bold; margin-bottom: 0.5rem; text-align: left; }}
    .note {{ color: #444; margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Multi-turn judged analysis</h1>
  <p>Final-turn column uses the last turn of each conversation (target query / adapter turn).</p>
  {meta_b}
  <table>
    <caption>Per-model means</caption>
    <thead>
      <tr>
        <th>model</th>
        <th>n all</th>
        <th>mean all</th>
        <th>n precursor</th>
        <th>mean precursor</th>
        <th>n final</th>
        <th>mean final</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <p class="note">
    Conversations: {summary["n_conversations"]} —
    matched turns: {summary["n_matched_turns"]} —
    unmatched: {summary["n_unmatched_turns"]}
  </p>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judged", type=Path, required=True)
    p.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Any batch_traces_*.json from the same run (for turn order + final-turn flags)",
    )
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-html", type=Path, default=None)
    p.add_argument(
        "--per-conversation-jsonl",
        type=Path,
        default=None,
        help="Optional: one JSON line per sample with turn-by-turn scores",
    )
    p.add_argument(
        "--by-failure-type",
        action="store_true",
        help="Print extra tables split by failure_type",
    )
    args = p.parse_args()

    judged = json.loads(Path(args.judged).read_text(encoding="utf-8"))
    traces = load_traces(Path(args.traces))

    lookup = build_judged_lookup(judged)
    model_names: list[str] = []
    for ent in judged["entries"].values():
        model_names = sorted(ent.get("results", {}).keys())
        break

    aligned_all: list[list[dict[str, Any]]] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        aligned_all.append(align_trace_to_judged(trace, lookup))

    marginal = marginal_stats_judged_file(judged, model_names)
    print_marginal_table(marginal, model_names)

    print(
        "\n## Trace-aligned (same user message as --traces)\n\n"
        "Each judged entry has **one** non-error model (the rest are padding). "
        "Rows are keyed by `(memories, query)`; the reference trace usually matches entries "
        "where that run's target model is the live column, so **precursor/final splits here are "
        "mostly for that model**. Use the marginal table above for comparable N per model.\n"
    )
    summary = summarize_models(aligned_all, model_names)
    print_markdown_table(summary, model_names)

    if args.by_failure_type:
        by_ft = summarize_by_failure_type(aligned_all, model_names)
        for ft in sorted(by_ft.keys()):
            print(f"\n### failure_type = {ft!r}\n")
            print_markdown_table(by_ft[ft], model_names)

    if args.out_csv:
        write_csv(Path(args.out_csv), summary, model_names)
        print(f"Wrote CSV: {args.out_csv}")
    if args.out_html:
        meta = judged.get("metadata")
        write_html_report(Path(args.out_html), summary, model_names, meta if isinstance(meta, dict) else None)
        print(f"Wrote HTML: {args.out_html}")
    if args.per_conversation_jsonl:
        outp = Path(args.per_conversation_jsonl)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for conv in aligned_all:
                if not conv:
                    continue
                rec = {
                    "sample_index": conv[0].get("sample_index"),
                    "failure_type": conv[0].get("failure_type"),
                    "target_query": conv[0].get("target_query"),
                    "turns": [
                        {
                            "turn": r["turn"],
                            "is_final": r["is_final"],
                            "matched": r["matched_judged"],
                            "scores_by_model": r["scores_by_model"],
                        }
                        for r in conv
                    ],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL: {args.per_conversation_jsonl}")


if __name__ == "__main__":
    main()
