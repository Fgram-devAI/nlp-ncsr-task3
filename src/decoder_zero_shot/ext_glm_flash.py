#
# EXTENSION — Zero-shot NER via Z.ai GLM-4.7-Flash.
#
# Outside-assignment-scope bonus experiment. Same protocol as Q9 / Q10.
# Pacing set to ~20 RPM (3 s sleep) per user spec.
#
# Auth: ZAI_API_KEY env var (in .env at project root).
# Uses the official `zai` Python SDK (Z.ai's open-platform client).
#
# Results land in their own namespace so they don't collide with Q9/Q10:
#   results/ext_glm_flash/<prompt>/sentences.jsonl
#
# If `glm-4.7-flash` is not available in your Z.ai account, override the
# model identifier with --model, e.g.:
#   .venv/bin/python src/decoder_zero_shot/ext_glm_flash.py --model glm-4.5-flash
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
from zai import ZaiClient  # noqa: E402  # pyright: ignore[reportMissingImports]

MODEL = _common.MODEL_GLM_FLASH
QUESTION = "ext_glm_flash"

GLM_RETRY_BACKOFF_SECONDS = (30, 60, 120)


def make_glm_call_fn() -> _common.CallFn:
    """Build a GLM call_fn using Z.ai's native SDK."""
    load_dotenv(_common.REPO_ROOT / ".env")
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY not set in .env at project root.")
    client = ZaiClient(api_key=api_key)

    def call(model: str, system_prompt: str, user_msg: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(len(GLM_RETRY_BACKOFF_SECONDS) + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=_common.TEMPERATURE,
                    max_tokens=_common.MAX_TOKENS,
                    # Disable extended-reasoning mode for Flash variants.
                    # GLM-4.5+ default to thinking-enabled which prepends a
                    # `<think>...</think>` block to the response and can add
                    # 5-30 s of latency per call — useless for our NER task
                    # where we want only the tagged-token JSON.
                    thinking={"type": "disabled"},  # pyright: ignore[reportCallIssue]
                )
                # Z.ai SDK types return as a union with streaming; we always
                # use stream=False (default), so a non-streaming response is
                # guaranteed at runtime.
                return resp.choices[0].message.content or ""  # pyright: ignore[reportAttributeAccessIssue]
            except Exception as e:
                last_exc = e
                # Z.ai's SDK doesn't expose a typed RateLimitError publicly the
                # same way OpenAI/Groq do; detect rate-limit via message/status.
                msg = str(e).lower()
                is_rate_limit = (
                    "rate limit" in msg or "429" in msg or "too many requests" in msg
                )
                if is_rate_limit and attempt < len(GLM_RETRY_BACKOFF_SECONDS):
                    wait = GLM_RETRY_BACKOFF_SECONDS[attempt]
                    print(f"  glm rate limit; sleeping {wait}s ({e})")
                    time.sleep(wait)
                    continue
                raise
        raise last_exc if last_exc else RuntimeError("glm retries exhausted")

    return call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", choices=_common.PROMPT_NAMES, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Override the GLM model identifier (default: {MODEL}).",
    )
    args = parser.parse_args()

    prompts_to_run = (args.prompt,) if args.prompt else _common.PROMPT_NAMES
    call_fn = make_glm_call_fn()

    for prompt_name in prompts_to_run:
        _common.run_zero_shot(
            call_fn=call_fn,
            sleep_seconds=_common.SLEEP_GLM,
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
