#
# EXTENSION — Zero-shot NER via Google Gemini 2.5 Flash-Lite (OpenAI-compatible endpoint).
#
# Outside-assignment-scope bonus experiment. Same protocol as Q9 / Q10
# (3 prompts × 200 sentences, JSONL per sentence, JSON aggregate per prompt).
# Pacing set to ~10 RPM to respect Gemini's free-tier hidden cap.
#
# Auth: GEMINI_API_KEY env var (in .env at project root).
# Base URL: Gemini's official OpenAI-compatible endpoint
#   https://generativelanguage.googleapis.com/v1beta/openai/
#
# Results land in their own namespace so they don't collide with Q9/Q10:
#   results/ext_gemini_flash_lite/<prompt>/sentences.jsonl
#
# Usage:
#   .venv/bin/python src/decoder_zero_shot/ext_gemini_flash_lite.py
#   .venv/bin/python src/decoder_zero_shot/ext_gemini_flash_lite.py --prompt v1_minimal --limit 5
#

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import _common  # noqa: E402  # pyright: ignore[reportMissingImports]

from dotenv import load_dotenv  # noqa: E402
from openai import APIError, OpenAI, RateLimitError  # noqa: E402

MODEL = _common.MODEL_GEMINI_FLASH_LITE
QUESTION = "ext_gemini_flash_lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini Flash-Lite free tier is the strictest of the three providers; if we
# get a 429 the cooldown can be long. Wider, slower backoff series.
GEMINI_RETRY_BACKOFF_SECONDS = (60, 120, 240, 480)


def make_gemini_call_fn() -> _common.CallFn:
    """Build a Gemini call_fn using the OpenAI SDK + Gemini's compat base URL."""
    load_dotenv(_common.REPO_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env at project root.")
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    def call(model: str, system_prompt: str, user_msg: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(len(GEMINI_RETRY_BACKOFF_SECONDS) + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=_common.TEMPERATURE,
                    max_tokens=_common.MAX_TOKENS,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                last_exc = e
                if attempt < len(GEMINI_RETRY_BACKOFF_SECONDS):
                    wait = GEMINI_RETRY_BACKOFF_SECONDS[attempt]
                    print(f"  gemini rate limit; sleeping {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise
            except APIError as e:
                raise e
        raise last_exc if last_exc else RuntimeError("gemini retries exhausted")

    return call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", choices=_common.PROMPT_NAMES, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Override the Gemini model identifier (default: {MODEL}).",
    )
    args = parser.parse_args()

    prompts_to_run = (args.prompt,) if args.prompt else _common.PROMPT_NAMES
    call_fn = make_gemini_call_fn()

    for prompt_name in prompts_to_run:
        _common.run_zero_shot(
            call_fn=call_fn,
            sleep_seconds=_common.SLEEP_GEMINI,
            model=args.model,
            prompt_name=prompt_name,
            question=QUESTION,
            limit=args.limit,
        )
        if not args.skip_aggregate:
            metrics = _common.aggregate(
                question=QUESTION, prompt_name=prompt_name, model=args.model
            )
            print(f"\n--- aggregated: {QUESTION} / {prompt_name} ---")
            for key in ("completed_count", "parse_failed_count", "unknown_tag_total"):
                print(f"  {key}: {metrics.get(key)}")
            test = metrics.get("test") or {}
            for key in (
                "token_micro_accuracy",
                "token_macro_accuracy",
                "entity_micro_f1",
                "entity_macro_f1",
            ):
                v = test.get(key)
                if v is not None:
                    print(f"  {key}: {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
