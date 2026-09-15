# Multi-turn Memory Stress Test (`experiments/multi_turn/`)

Dynamic multi-turn extension of [PersistBench](https://github.com/ivaxi0s/PersistBench):
instead of asking the benchmark query in one shot, a **simulator model** holds a short
natural conversation (3–5 turns) that funnels toward the query, then an **adapter model**
paraphrases the query so it lands as an organic follow-up. A **gatekeeper model** rejects
precursor turns that leak the target early or contradict the user's memories.

Nothing about the conversation is pre-written: every run generates fresh precursor turns.
The only static inputs are the 150 seed samples in `data/` (query + memories + broad topic).

## How it works

```
data/samples_150.json ──► simulator (kimi-k2-0905 + prompts/chat_prompt.txt)
      (broad_topic,            │  ▲ target reply fed back each turn
       query, memories)        ▼  │
  target (eval_prompt.txt + memories as system) ──► gatekeeper (gpt-oss-20b + gate_llm.txt)
         │         pass/fail, ≤3 retries                    │
         │◄─────────────────────────────────────────────────┘
         ▼ final turn: adapter paraphrases target query (prompts/adapter*.txt)
outputs/batch_traces_<model>.json ──► to_judge_format.py ──► PersistBench `benchmark judge`
```

- **Turns 1..N-1 (precursors):** simulator plays a casual human. It only sees the real
  target query on the penultimate turn by default (`--sim-target-from-start` disables
  the withholding). Each candidate message is checked by the gatekeeper
  (`is_too_similar` / `contradicts_memory`); rejected messages are rewritten, up to
  `MAX_RETRIES = 3`, with a generic fallback after that.
- **Turn N (target):** the adapter rewrites the original benchmark query with a
  conversational bridge ("That makes sense. So if we apply that...") while preserving
  every entity and constraint. The target model always has the memories injected via
  `prompts/eval_prompt.txt` from turn 1.
- **Prompt routing:** `beneficial_memory_usage` / `cross_domain` use `chat_prompt.txt` +
  `adapter.txt`; `sycophancy` uses `sycophancy_chat_prompt.txt` + `adapter_sycophancy.txt`.
- **`context_switch/` variant:** same harness, but precursors are decoy UltraChat topics
  (unrelated to the query) instead of a funnel — tests memory use after a topic switch.

## Layout

```
experiments/multi_turn/
├── README.md                  # this file
├── requirements.txt           # install (PersistBench deps already cover openai/anthropic/dotenv)
├── .env.example               # copy to .env and fill in keys
├── .gitignore                 # keeps outputs/*.json, viewers, .env out of git
├── data/
│   └── samples_150.json       # 150 static seeds: {query, memories, broad_topic, failure_type}
├── prompts/                   # all LLM prompts (simulator, gatekeeper, adapter, target, extractor)
├── src/
│   ├── common.py              # shared paths + helpers (format_memories, clean_json, ...)
│   ├── run_single.py          # one demo conversation (refactored main.py)
│   ├── run_batch.py           # full 150-sample run, checkpoint/resume, parallel workers
│   ├── to_judge_format.py     # flatten traces → PersistBench judge-ready JSON
│   ├── build_viewer.py        # standalone HTML conversation viewer
│   ├── analyze_judged.py      # precursor-vs-final-turn metrics
│   ├── plot_metrics.py        # decay / survival plots
│   ├── extract_topics.py      # regenerate broad_topic values (how data/samples_150.json was made)
│   ├── stream_ultrachat_prompts.py  # fetch decoy prompts for context_switch/
│   ├── openai_target.py / anthropic_target.py  # direct-API routing + prompt caching
├── context_switch/            # decoy-topic variant (own prompts + runner, shares src/ targets)
│   ├── main.py
│   ├── chat_prompt.txt / adapter*.txt / gate_llm.txt
│   └── ultrachat_200k_prompts_first200.jsonl  # 200 decoy prompts (regenerable, see below)
├── examples/
│   └── sample_trace.json      # 2 sample traces, format reference (239 KB)
├── paper_traces/              # paper result traces (committed): 5×150 natural-setting
│                              # runs + HTML viewers + seed dataset + single-turn baseline
└── outputs/                   # created on run; git-ignored
```

## Setup

```bash
# 1. Install (from the PersistBench repo root, or standalone):
pip install -r experiments/multi_turn/requirements.txt
# PersistBench itself uses: uv sync && uv pip install -e .

# 2. Keys:
cp experiments/multi_turn/.env.example experiments/multi_turn/.env
# then edit .env — OPENROUTER_API_KEY is required;
# OPENAI_API_KEY / ANTHROPIC_API_KEY only for direct-routed targets.
```

Model defaults: simulator + adapter `moonshotai/kimi-k2-0905`,
gatekeeper `openai/gpt-oss-20b` (both via OpenRouter). Targets can be any OpenRouter
ID, a direct OpenAI snapshot (e.g. `gpt-5.2-2025-12-11`), or a direct Claude ID
(e.g. `claude-sonnet-4-5-20250929`).

## Reproduce

```bash
cd experiments/multi_turn

# Smoke test — one conversation, free/cheap target:
python src/run_single.py --target-model google/gemini-2.5-flash-lite --seed 0
# deterministic variant:
python src/run_single.py --target-model google/gemini-2.5-flash-lite \
  --sample-index 0 --total-turns 4 --seed 42

# Full run — all 150 samples, resumable checkpoint in outputs/:
python src/run_batch.py google/gemini-2.5-flash-lite --workers 4 --seed 42
# retry failed indices only:
python src/run_batch.py google/gemini-2.5-flash-lite \
  --merge-into outputs/batch_traces_google_gemini-2.5-flash-lite.json --retry-indices 3,7

# Judge with PersistBench (from the PersistBench repo root):
python src/to_judge_format.py --inputs "outputs/batch_traces_*.json" \
  --out outputs/all_models_for_judging.json
uv run benchmark judge outputs/all_models_for_judging.json

# Inspect:
python src/build_viewer.py outputs/batch_traces_google_gemini-2.5-flash-lite.json
python src/analyze_judged.py --judged outputs/all_models_judged.json \
  --traces outputs/batch_traces_google_gemini-2.5-flash-lite.json --out-csv summary.csv

# Context-switch variant:
python context_switch/main.py --batch --target-model google/gemini-2.5-flash-lite --workers 4
# regenerate decoy prompts (needs `datasets`):
python src/stream_ultrachat_prompts.py --n 200 \
  --out context_switch/ultrachat_200k_prompts_first200.jsonl
```

## Reproducibility notes

- `run_batch.py --seed <int>` derives a per-sample RNG (`seed * 1_000_003 + index`),
  so reruns with the same seed reproduce `total_turns` (3–5) and fallbacks exactly.
  Simulator/adapter/gatekeeper sampling temperatures are fixed in code
  (0.7 / 0.5 / 0.1). `run_single.py` accepts `--seed`, `--sample-index`, `--total-turns`.
- Pin an OpenRouter backend via `api_params` provider routing if you need
  bit-stable targets (see PersistBench README → Providers).
- `data/samples_150.json` is the refined dataset (`broad_topic` values produced by
  `src/extract_topics.py` + `prompts/extractor.txt` over the original seed queries).
  To regenerate topics: `python src/extract_topics.py --input data/samples_150.json`.

## What is NOT in this upload (and why)

- Merged judging files (`*_for_judging.json`, `*_judged.json`) — regenerable with
  `src/to_judge_format.py` + `uv run benchmark judge ...`; `examples/sample_trace.json`
  shows the exact trace schema (2 traces).
- `.env` / API keys, `__pycache__`, plots (`*.png`) — all git-ignored.
- The single-turn PersistBench harness itself — this folder assumes a checkout of
  [PersistBench](https://github.com/ivaxi0s/PersistBench) as the parent repo
  (`uv run benchmark judge ...`).

Paper result traces live in `paper_traces/` (see its README): all five 150-sample
natural-setting runs with HTML viewers, the exact seed dataset, and the single-turn
baseline.

## Cite

```bibtex
@misc{pulipaka2026persistbenchlongtermmemoriesforgotten,
  title={PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?},
  author={Sidharth Pulipaka and Oliver Chen and Manas Sharma and Taaha S Bajwa and Vyas Raina and Ivaxi Sheth},
  year={2026}, eprint={2602.01146}, archivePrefix={arXiv}, primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.01146},
}
```
