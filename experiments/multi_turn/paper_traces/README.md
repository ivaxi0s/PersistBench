# Paper traces — natural-setting multi-turn runs

Full 150-sample traces behind the paper's natural-conversation results, one file per
target model. Each `batch_traces_<model>.json` has `{meta, errors, traces}` where every
trace holds the per-turn `{user_message, assistant_reply}` pairs plus the simulator /
gatekeeper / adapter internals. Open the matching `*_viewer.html` in a browser for a
readable rendering (final accepted conversation only, grouped by failure type).

| File | Target model |
|---|---|
| `batch_traces_gpt-5.2-2025-12-11.json` | `gpt-5.2-2025-12-11` (direct OpenAI, prompt caching on) |
| `batch_traces_claude-sonnet-4-5-20250929.json` | `claude-sonnet-4-5-20250929` (direct Anthropic, cached system) |
| `batch_traces_google_gemini-3.1-pro-preview.json` | `google/gemini-3.1-pro-preview` (via OpenRouter) |
| `batch_traces_x-ai_grok-4.1-fast.json` | `x-ai/grok-4.1-fast` (via OpenRouter) |
| `batch_traces_meta-llama_llama-4-maverick.json` | `meta-llama/llama-4-maverick` (via OpenRouter) |

- All runs: simulator + adapter `moonshotai/kimi-k2-0905`, gatekeeper
  `openai/gpt-oss-20b`, `sim_target_from_start: true`, unseeded (`seed: null`), except
  gpt-5.2 which retried sample index 19.
- `150_samples_orig.json` is the exact seed dataset these runs used (kept for
  provenance; `../data/samples_150.json` is the refined-topic equivalent).
- `baseline_all_models.json` is the single-turn PersistBench baseline
  (`{metadata, entries, config}`, all memories unfiltered) used as the comparison
  point. Merged-source filenames only — absolute local paths scrubbed.
- Judged / `_for_judging` derivatives are intentionally excluded; regenerate them:
  `python src/to_judge_format.py --inputs "paper_traces/batch_traces_*.json" --out outputs/paper_for_judging.json`
