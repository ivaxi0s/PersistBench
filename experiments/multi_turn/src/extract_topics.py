"""Extract broad topics for each query via OpenRouter (parallel workers)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
SAMPLES_DEFAULT = ROOT / "data" / "samples_150.json"
EXTRACTOR_DEFAULT = ROOT / "prompts" / "extractor.txt"
MODEL = "google/gemma-4-31b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_WORKERS = 16
MAX_RETRIES = 4
RETRY_BACKOFF = 1.5

_load_lock = threading.Lock()
_system_prompt_cache: str | None = None


def build_system_prompt(extractor_path: Path) -> str:
    global _system_prompt_cache
    with _load_lock:
        if _system_prompt_cache is not None:
            return _system_prompt_cache
        raw = extractor_path.read_text(encoding="utf-8")
        marker = "Input: [INSERT_TARGET_QUERY_HERE]"
        if marker in raw:
            _system_prompt_cache = raw.split(marker)[0].rstrip()
        else:
            _system_prompt_cache = raw.rstrip()
        return _system_prompt_cache


def normalize_topic(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^["\']|["\']$', "", text.strip())
    text = re.sub(r"^Output:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def call_openrouter(api_key: str, query: str, extractor_path: Path) -> str:
    system = build_system_prompt(extractor_path)
    user = f"Input: {query}\nOutput:"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/extract-topics",
        "X-Title": "Topic extraction",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return normalize_topic(content or "")
        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"OpenRouter failed after retries: {last_err}") from last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract broad_topic for each query via OpenRouter.")
    ap.add_argument("--input", type=Path, default=SAMPLES_DEFAULT)
    ap.add_argument("--prompt", type=Path, default=EXTRACTOR_DEFAULT)
    ap.add_argument("--force", action="store_true", help="Re-extract even if broad_topic exists.")
    cli = ap.parse_args()
    SAMPLES_PATH: Path = cli.input
    EXTRACTOR_PATH: Path = cli.prompt

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing (copy .env.example to .env first)")

    data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Expected top-level JSON array")

    indices_needing: list[int] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        if "broad_topic" not in row or not str(row.get("broad_topic", "")).strip():
            indices_needing.append(i)

    if not indices_needing:
        force_env = os.environ.get("FORCE_OVERRIDE_BROAD_TOPIC", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        force_arg = "--force" in sys.argv or cli.force
        override = bool(force_env or force_arg)
        if not override and sys.stdin.isatty():
            ans = input(
                "All rows already have broad_topic. Re-extract and override them? [y/N]: "
            ).strip().lower()
            override = ans in ("y", "yes")
        if not override:
            if not (force_env or force_arg) and not sys.stdin.isatty():
                print(
                    "All rows already have broad_topic. Use --force or set "
                    "FORCE_OVERRIDE_BROAD_TOPIC=1 to override without a prompt, "
                    "or run interactively to choose."
                )
            else:
                print("Exiting without changes.")
            return
        indices_needing = [
            i
            for i, row in enumerate(data)
            if isinstance(row, dict)
            and isinstance(row.get("query"), str)
            and row["query"].strip()
        ]
        if not indices_needing:
            raise SystemExit("No rows with a non-empty query to process.")

    print(f"Extracting topics for {len(indices_needing)} / {len(data)} rows ({MAX_WORKERS} workers)…")

    def work(idx: int) -> tuple[int, str]:
        q = data[idx].get("query", "")
        if not isinstance(q, str) or not q.strip():
            return idx, ""
        topic = call_openrouter(api_key, q, EXTRACTOR_PATH)
        return idx, topic

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(work, i): i for i in indices_needing}
        done = 0
        for fut in as_completed(futures):
            idx, topic = fut.result()
            data[idx]["broad_topic"] = topic
            done += 1
            if done % 10 == 0 or done == len(indices_needing):
                print(f"  {done}/{len(indices_needing)}")

    SAMPLES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
