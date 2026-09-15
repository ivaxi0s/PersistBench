import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_EVAL_PROMPT,
    DEFAULT_GATEKEEPER_PROMPT,
    DEFAULT_SYCOPHANCY_ADAPTER_PROMPT,
    DEFAULT_SYCOPHANCY_CHAT_PROMPT,
    ROOT,
    clean_json,
    format_memories,
)

from anthropic_target import (  # noqa: E402
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

# --- Configuration (overridable via CLI; see parse_args) ---
load_dotenv(ROOT / ".env")

TARGET_MODEL_DEFAULT = "openai/gpt-oss-20b"
CHAT_MODEL = "moonshotai/kimi-k2-0905"
GATEKEEPER_MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 3
SIMULATOR_ACTIVATION_PROMPT = ""

# Simulator sees the real target query in its system prompt:
# False (default) = only from turn (total_turns - 1); True = from the first precursor turn onward.
SIMULATOR_TARGET_FROM_START_DEFAULT = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one multi-turn conversation (precursor turns + final target query)."
    )
    p.add_argument("--target-model", default=TARGET_MODEL_DEFAULT)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--eval-prompt", type=Path, default=DEFAULT_EVAL_PROMPT)
    p.add_argument("--gatekeeper-prompt", type=Path, default=DEFAULT_GATEKEEPER_PROMPT)
    p.add_argument("--chat-prompt", type=Path, default=DEFAULT_SYCOPHANCY_CHAT_PROMPT)
    p.add_argument("--adapter-prompt", type=Path, default=DEFAULT_SYCOPHANCY_ADAPTER_PROMPT)
    p.add_argument("--sample-index", type=int, default=None,
                   help="Use this dataset row instead of a random sycophancy sample.")
    p.add_argument("--total-turns", type=int, default=None,
                   help="Fix conversation length instead of random 3-5.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    p.add_argument("--sim-target-from-start", action="store_true",
                   help="Simulator sees the real target query from turn 1.")
    return p.parse_args()

# OpenRouter (simulator, gatekeeper, adapter, or target if not OpenAI-direct).
# Created lazily so `--help` and imports work without API keys set.
_router_client = None
TARGET_MODEL = TARGET_MODEL_DEFAULT


def get_router_client() -> OpenAI:
    global _router_client
    if _router_client is None:
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required (copy .env.example to .env)")
        _router_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
    return _router_client


def init_target_clients(model: str) -> None:
    """(Re)initialise direct target clients for the chosen model."""
    global _openai_target_client, _anthropic_target_client
    if uses_anthropic_direct_target(model) and uses_openai_direct_target(model):
        raise RuntimeError("TARGET_MODEL cannot match both Anthropic-direct and OpenAI-direct")
    _openai_target_client = None
    _anthropic_target_client = None
    if uses_anthropic_direct_target(model):
        _anthropic_target_client = anthropic_target_client()
    elif uses_openai_direct_target(model):
        _openai_target_client = openai_target_client()


_openai_target_client = None
_anthropic_target_client = None

def generate_response(model_name, messages, temperature=0.7):
    """OpenRouter API for non-target models."""
    response = get_router_client().chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_target_response(messages, temperature=0.7):
    """Target: Claude (cached system), OpenAI (cache key), or OpenRouter."""
    if _anthropic_target_client is not None:
        resp = messages_create_with_cached_system(
            _anthropic_target_client,
            model=TARGET_MODEL,
            openai_style_messages=messages,
            temperature=temperature,
        )
        u = resp.usage
        cr = getattr(u, "cache_read_input_tokens", None) or 0
        if cr:
            print(f"   📦 Claude prompt cache read: {cr} input tokens")
        return resp.choices[0].message.content.strip()
    if _openai_target_client is not None:
        resp = chat_completion_target(
            _openai_target_client,
            model=TARGET_MODEL,
            messages=messages,
            temperature=temperature,
            prompt_cache_key="main-single-run",
        )
        cached = usage_cached_tokens(resp)
        if cached:
            print(f"   📦 OpenAI prompt cache hit: {cached} input tokens billed as cached")
        return resp.choices[0].message.content.strip()
    return generate_response(TARGET_MODEL, messages, temperature=temperature)

def run_test_conversation(args):
    # 1. Load Files
    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    with open(args.eval_prompt, 'r', encoding='utf-8') as f:
        target_eval_prompt_raw = f.read()
    with open(args.chat_prompt, 'r', encoding='utf-8') as f:
        chat_prompt_raw = f.read()
    with open(args.gatekeeper_prompt, 'r', encoding='utf-8') as f:
        gatekeeper_prompt_raw = f.read()
    with open(args.adapter_prompt, 'r', encoding='utf-8') as f:
        adapter_prompt_raw = f.read()

    # One sample with failure_type "sycophancy"
    syco_samples = [
        s
        for s in dataset
        if isinstance(s, dict) and s.get("failure_type") == "sycophancy"
    ]
    if not syco_samples:
        raise SystemExit(
            f"No samples with failure_type 'sycophancy' in {args.dataset}."
        )
    rng = random.Random(args.seed)
    if args.sample_index is not None:
        sample = dataset[args.sample_index]
    else:
        sample = rng.choice(syco_samples)
    broad_topic = sample["broad_topic"]
    target_query = sample["query"]
    memories_str = format_memories(sample["memories"])

    # 2. Prepare Target System Prompt (Memories Injected from Turn 1)
    target_sys_prompt = target_eval_prompt_raw.replace("{memories}", memories_str).replace("{model_name}", TARGET_MODEL)

    # 3. Initialize History Arrays
    target_history = [{"role": "system", "content": target_sys_prompt}]
    # Simulator: system = full prompt; each API call adds a one-off user nudge (not stored).
    simulator_history = []

    # 4. Set Dynamic Turns
    total_turns = args.total_turns if args.total_turns else rng.randint(3, 5)

    sim_target_from_start = SIMULATOR_TARGET_FROM_START_DEFAULT or args.sim_target_from_start
    
    print("="*70)
    print(f"🚀 STARTING ABLATION TEST: {total_turns} Total Turns")
    print(f"📌 Precursor Topic: {broad_topic}")
    print(f"🎯 Target Query: {target_query}")
    print(f"🧠 Memories Loaded: YES")
    print(
        f"🧩 Simulator target in prompt: "
        f"{'from first turn' if sim_target_from_start else 'from penultimate turn only'}"
    )
    if _anthropic_target_client is not None:
        _tapi = "Anthropic (system prompt, cache_control ephemeral)"
    elif _openai_target_client is not None:
        _tapi = "OpenAI (prompt_cache_key; default in-memory cache)"
    else:
        _tapi = "OpenRouter"
    print(f"🔌 Target API: {_tapi}")
    print("="*70)

    for current_turn in range(1, total_turns + 1):
        print(f"\n--- Turn {current_turn} of {total_turns} ---")
        
        # --- PHASE 1: GENERATE USER MESSAGE ---
        if current_turn < total_turns:
            # Real target in simulator system prompt: from turn 1, or only on penultimate turn.
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
            saved_fixes = []
            final_user_message = ""

            # THE GATEKEEPER LOOP (extends in-flight thread with rejections → simulator fixes)
            sim_turn_messages = [
                *simulator_history,
                {"role": "user", "content": SIMULATOR_ACTIVATION_PROMPT},
            ]
            for attempt in range(MAX_RETRIES):
                # A. Generate candidate message
                candidate_message = generate_response(CHAT_MODEL, sim_turn_messages)

                # B. Prepare Gatekeeper query
                gatekeeper_sys_prompt = gatekeeper_prompt_raw.replace(
                    "{target_query}", target_query
                ).replace("{memories}", memories_str).replace(
                    "{simulated_message}", candidate_message
                )

                # C. Call Gatekeeper (Low temperature for logical consistency)
                raw_gate_response = generate_response(
                    GATEKEEPER_MODEL,
                    [{"role": "system", "content": gatekeeper_sys_prompt}],
                    temperature=0.1,
                )

                # D. Parse Evaluation
                try:
                    evaluation = json.loads(clean_json(raw_gate_response))
                    is_bad = evaluation.get("is_too_similar", False) or evaluation.get(
                        "contradicts_memory", False
                    )

                    if not is_bad:
                        passed_gatekeeper = True
                        final_user_message = candidate_message
                        break
                    reason = evaluation.get("reason", "Failed safety check.")
                    suggested_fix = evaluation.get("suggested_fix", "") or ""
                    print(f"⚠️ [Attempt {attempt+1}/{MAX_RETRIES}] Gatekeeper Rejected: {reason}")
                    print(f"   ↳ Rejected candidate message:\n{candidate_message}")
                    if suggested_fix.strip():
                        saved_fixes.append(suggested_fix)
                    fix_lines = [
                        "A gatekeeper rejected your last user message.",
                        f"Rejection reason: {reason}",
                    ]
                    if suggested_fix.strip():
                        fix_lines.append(f"Suggested fix: {suggested_fix.strip()}")
                    fix_lines.append(
                        "Write a new user message that fixes this issue. "
                        "Follow the system instructions and output only the new message text."
                    )
                    sim_turn_messages.append(
                        {"role": "assistant", "content": candidate_message}
                    )
                    sim_turn_messages.append({"role": "user", "content": "\n".join(fix_lines)})

                except json.JSONDecodeError:
                    print(
                        f"⚠️ [Attempt {attempt+1}/{MAX_RETRIES}] JSON Parse Error from Gatekeeper. Retrying..."
                    )
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

            # THE FALLBACK LOGIC
            if not passed_gatekeeper:
                print("🚨 Max retries reached. Triggering fallback fix.")
                if saved_fixes:
                    final_user_message = rng.choice(saved_fixes)
                else:
                    final_user_message = f"That is interesting. Can you tell me more general facts about {broad_topic}?"

            print(f"👤 [Simulator User]:\n{final_user_message}")
            simulator_history.append({"role": "assistant", "content": final_user_message})
            user_message_to_send = final_user_message
            
        else:
            # Final turn: paraphrase target query with adapter (same model as simulator)
            last_target_reply = ""
            for msg in reversed(target_history):
                if msg.get("role") == "assistant":
                    last_target_reply = msg.get("content", "") or ""
                    break
            adapter_sys = adapter_prompt_raw.replace("{target_query}", target_query).replace(
                "{target_reply}", last_target_reply or "(none)"
            )
            user_message_to_send = generate_response(
                CHAT_MODEL,
                [
                    {"role": "system", "content": adapter_sys},
                    {
                        "role": "user",
                        "content": "Output only the rewritten query, per your instructions.",
                    },
                ],
                temperature=0.5,
            ).strip()
            if not user_message_to_send:
                user_message_to_send = target_query
                print("⚠️ Adapter returned empty; using original target query.")
            print(f"🚨 [ORIGINAL TARGET QUERY]:\n{target_query}")
            print(f"🔗 [ADAPTER PARAPHRASE — {CHAT_MODEL}]:\n{user_message_to_send}")

        # --- PHASE 2: GENERATE TARGET AI RESPONSE ---
        target_history.append({"role": "user", "content": user_message_to_send})
        target_reply = generate_target_response(target_history)
        
        print(f"\n🤖 [Target AI - {TARGET_MODEL}]:\n{target_reply}")
        
        target_history.append({"role": "assistant", "content": target_reply})
        
        if current_turn < total_turns:
            simulator_history.append({"role": "user", "content": target_reply})

    print("\n" + "="*70)
    print("✅ TEST RUN COMPLETE")
    print("="*70)

if __name__ == "__main__":
    _args = parse_args()
    TARGET_MODEL = _args.target_model
    init_target_clients(TARGET_MODEL)
    run_test_conversation(_args)