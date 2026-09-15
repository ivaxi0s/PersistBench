"""Stream HuggingFaceH4/ultrachat_200k and write the first N rows' prompt field to JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="train_sft",
        help="Dataset split (e.g. train_sft, test_sft, train_gen, test_gen)",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of rows to save")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ultrachat_200k_prompts_first200.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=args.split,
        streaming=True,
    )

    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            if written >= args.n:
                break
            f.write(
                json.dumps({"prompt": row["prompt"]}, ensure_ascii=False) + "\n"
            )
            written += 1

    print(f"Wrote {written} lines to {args.out.resolve()}")


if __name__ == "__main__":
    main()
