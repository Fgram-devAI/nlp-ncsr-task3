#
# Q9 — Zero-shot NER on the first 200 test sentences via Groq Llama-3.1-8B.
#
# Pure HTTP — no GPU, no MPS. Runs locally.
#
# Usage:
#   .venv/bin/python src/decoder_zero_shot/q9_llama_8b.py                # all 3 prompts × 200 sentences
#   .venv/bin/python src/decoder_zero_shot/q9_llama_8b.py --prompt v1_minimal     # one prompt only
#   .venv/bin/python src/decoder_zero_shot/q9_llama_8b.py --limit 5               # smoke test (5 sentences)
#
# Resumable: any sentence whose JSONL line already exists in
# results/q9/<prompt>/sentences.jsonl is skipped on subsequent runs.
#
# Architecture note: this entrypoint constructs a Groq-specific `call_fn`
# and hands it to the provider-agnostic driver in `_common.run_zero_shot`.
# Provider-specific retry on rate limits / API errors is handled here, not
# in _common.
#

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow `import _common` when running this file directly as a script.
sys.path.insert(0, str(Path(__file__).parent))

import _common  # noqa: E402  # pyright: ignore[reportMissingImports]

from dotenv import load_dotenv  # noqa: E402
from groq import APIError, Groq, RateLimitError  # noqa: E402

MODEL = _common.MODEL_8B
QUESTION = "q9"

# Per-provider retry policy for transient errors (rate limits in particular).
GROQ_RETRY_BACKOFF_SECONDS = (30, 60, 120)


def make_groq_call_fn() -> _common.CallFn:
    """Build a `call_fn` that hits Groq with retry-on-rate-limit."""
    load_dotenv(_common.REPO_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env at project root.")
    client = Groq(api_key=api_key)

    def call(model: str, system_prompt: str, user_msg: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(len(GROQ_RETRY_BACKOFF_SECONDS) + 1):
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
                if attempt < len(GROQ_RETRY_BACKOFF_SECONDS):
                    wait = GROQ_RETRY_BACKOFF_SECONDS[attempt]
                    print(f"  groq rate limit; sleeping {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise
            except APIError as e:
                # Non-retryable API failure (bad request, server error, etc.)
                raise e
        raise last_exc if last_exc else RuntimeError("groq retries exhausted")

    return call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        choices=_common.PROMPT_NAMES,
        default=None,
        help="Run only this prompt variant. Default: run all three.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap to first N sentences.")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    prompts_to_run = (args.prompt,) if args.prompt else _common.PROMPT_NAMES
    call_fn = make_groq_call_fn()

    for prompt_name in prompts_to_run:
        _common.run_zero_shot(
            call_fn=call_fn,
            sleep_seconds=_common.SLEEP_GROQ,
            model=MODEL,
            prompt_name=prompt_name,
            question=QUESTION,
            limit=args.limit,
        )
        if not args.skip_aggregate:
            metrics = _common.aggregate(
                question=QUESTION, prompt_name=prompt_name, model=MODEL
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
