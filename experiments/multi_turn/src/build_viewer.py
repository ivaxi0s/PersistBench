"""
Build a standalone HTML viewer from batch_traces_*.json.

Extracts only turns[].target user_message + assistant_reply (final accepted trace),
grouped by failure_type. No gatekeeper, adapter internals, or system prompts.

Usage:
  python src/build_viewer.py outputs/batch_traces_x-ai_grok-4.1-fast.json
  python src/build_viewer.py outputs/traces.json -o outputs/my_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


SECTION_ORDER = (
    "beneficial_memory_usage",
    "cross_domain",
    "sycophancy",
)

SECTION_TITLES = {
    "beneficial_memory_usage": "Beneficial memory usage",
    "cross_domain": "Cross-domain",
    "sycophancy": "Sycophancy",
}


def extract_pairs(trace: dict) -> list[dict[str, str]] | None:
    if trace.get("error"):
        return None
    pairs: list[dict[str, str]] = []
    for turn in trace.get("turns") or []:
        tg = turn.get("target") or {}
        u = (tg.get("user_message") or "").strip()
        a = (tg.get("assistant_reply") or "").strip()
        pairs.append({"user": u, "assistant": a})
    return pairs if pairs else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("batch_json", type=Path, help="batch_traces_*.json path")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .html (default: <batch_stem>_viewer.html)",
    )
    args = p.parse_args()

    data = json.loads(args.batch_json.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    traces = data.get("traces") or []

    grouped: dict[str, list[dict]] = {k: [] for k in SECTION_ORDER}
    for tr in traces:
        if not isinstance(tr, dict):
            continue
        ft = tr.get("failure_type") or "unknown"
        if ft not in grouped:
            grouped[ft] = []
        idx = tr.get("sample_index", len(grouped[ft]))
        pairs = extract_pairs(tr)
        entry = {
            "sample_index": idx,
            "target_query": (tr.get("target_query") or "")[:500],
            "pairs": pairs,
            "error": tr.get("error"),
        }
        grouped.setdefault(ft, []).append(entry)

    for ft in grouped:
        grouped[ft].sort(key=lambda x: x["sample_index"])

    payload = {
        "target_model": meta.get("target_model", ""),
        "dataset": meta.get("dataset", ""),
        "sections": {k: grouped.get(k, []) for k in SECTION_ORDER},
    }

    data_json = json.dumps(payload, ensure_ascii=False)
    payload_b64 = base64.b64encode(data_json.encode("utf-8")).decode("ascii")
    out = args.output or args.batch_json.with_name(
        f"{args.batch_json.stem}_viewer.html"
    )

    section_keys_js = json.dumps(list(SECTION_ORDER))
    titles_js = json.dumps(SECTION_TITLES)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Conversations — {html.escape(meta.get("target_model", "batch"))}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0f1412;
      --surface: #1a221e;
      --border: #2d3d36;
      --text: #e8f0eb;
      --muted: #8fa99a;
      --user: #c4e8d4;
      --user-bg: #243830;
      --asst: #d4e5ff;
      --asst-bg: #1e2838;
      --accent: #7fd4a8;
      --warn: #e8b86d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Source Serif 4", Georgia, serif;
      font-size: 1rem;
      line-height: 1.55;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: linear-gradient(180deg, #0f1412 92%, transparent);
      padding: 1.25rem 1.5rem 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{
      font-family: Fraunces, Georgia, serif;
      font-weight: 700;
      font-size: 1.35rem;
      margin: 0 0 0.35rem;
      letter-spacing: -0.02em;
    }}
    header .sub {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
    }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 0.88rem;
      padding: 0.35rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface);
    }}
    nav a:hover {{ border-color: var(--accent); }}
    main {{ max-width: 52rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    section.category {{
      margin-bottom: 3.5rem;
      scroll-margin-top: 6rem;
    }}
    section.category h2 {{
      font-family: Fraunces, Georgia, serif;
      font-size: 1.5rem;
      margin: 0 0 0.25rem;
      color: var(--accent);
    }}
    section.category .count {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 1.5rem;
    }}
    article.sample {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.35rem;
      margin-bottom: 1.75rem;
    }}
    article.sample h3 {{
      font-family: Fraunces, Georgia, serif;
      font-size: 1.05rem;
      margin: 0 0 0.75rem;
      color: var(--text);
    }}
    details.tq {{
      margin-bottom: 1rem;
      font-size: 0.88rem;
      color: var(--muted);
    }}
    details.tq summary {{
      cursor: pointer;
      color: var(--muted);
    }}
    details.tq[open] summary {{ margin-bottom: 0.5rem; }}
    .bubble {{
      margin-bottom: 1rem;
      padding: 0.85rem 1rem;
      border-radius: 10px;
      border: 1px solid var(--border);
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .bubble .label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.45rem;
      font-family: system-ui, sans-serif;
    }}
    .bubble.user {{
      background: var(--user-bg);
      border-color: #355a4a;
    }}
    .bubble.user .label {{ color: var(--user); }}
    .bubble.assistant {{
      background: var(--asst-bg);
      border-color: #3a4a6a;
    }}
    .bubble.assistant .label {{ color: var(--asst); }}
    .err {{
      color: var(--warn);
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1 id="top">Final target conversations</h1>
    <div class="sub" id="meta-line"></div>
    <nav id="nav"></nav>
  </header>
  <main id="main"></main>
  <script type="text/plain" id="payload-b64">{payload_b64}</script>
  <script>
    const payload = JSON.parse(
      atob(document.getElementById("payload-b64").textContent.trim())
    );
    const titles = {titles_js};
    const sectionOrder = {section_keys_js};

    document.getElementById("meta-line").textContent =
      (payload.target_model || "—") +
      (payload.dataset ? " · " + payload.dataset.split(/[/\\\\]/).pop() : "");

    const nav = document.getElementById("nav");
    const main = document.getElementById("main");

    const esc = (s) => {{
      const d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }};

    for (const key of sectionOrder) {{
      const list = payload.sections[key] || [];
      const a = document.createElement("a");
      a.href = "#cat-" + key;
      a.textContent = (titles[key] || key) + " (" + list.length + ")";
      nav.appendChild(a);
    }}

    for (const key of sectionOrder) {{
      const list = payload.sections[key] || [];
      const sec = document.createElement("section");
      sec.className = "category";
      sec.id = "cat-" + key;
      sec.innerHTML =
        "<h2>" + esc(titles[key] || key) + "</h2>" +
        '<p class="count">' + list.length + " sample(s)</p>";
      for (const item of list) {{
        const art = document.createElement("article");
        art.className = "sample";
        let inner = "<h3>Sample " + item.sample_index + "</h3>";
        if (item.error) {{
          inner += '<p class="err">Error: ' + esc(String(item.error)) + "</p>";
        }} else if (item.target_query) {{
          inner +=
            '<details class="tq"><summary>Original target query</summary>' +
            esc(item.target_query) +
            (item.target_query.length >= 500 ? "…" : "") +
            "</details>";
        }}
        if (!item.pairs || !item.pairs.length) {{
          inner += '<p class="err">No conversation turns</p>';
        }} else {{
          let turnN = 0;
          for (const p of item.pairs) {{
            turnN++;
            inner +=
              '<div class="bubble user"><div class="label">User · turn ' +
              turnN +
              '</div>' +
              esc(p.user || "(empty)") +
              "</div>";
            inner +=
              '<div class="bubble assistant"><div class="label">Assistant</div>' +
              esc(p.assistant || "(empty)") +
              "</div>";
          }}
        }}
        art.innerHTML = inner;
        sec.appendChild(art);
      }}
      main.appendChild(sec);
    }}
  </script>
</body>
</html>
"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
