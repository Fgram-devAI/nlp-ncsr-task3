#
# Q10 — Zero-shot NER on the first 200 test sentences via Llama-3.3-70B.
#
# Primary provider: Groq  (model: llama-3.3-70b-versatile).
# Fallback provider: OpenRouter (model: meta-llama/llama-3.3-70b-instruct:free),
# engaged automatically when Groq's daily TPD limit is exhausted, then stays
# engaged for the rest of the run.
#
# Both providers serve the same Meta Llama-3.3-70B checkpoint — only the host
# infrastructure changes when fallback activates. Each JSONL line records a
# `provider` field ("groq:llama-3.3-70b-versatile" or
# "openrouter:meta-llama/llama-3.3-70b-instruct:free") so the report can
# attribute per-call results honestly.
#
# Why OpenRouter as fallback: their free tier exposes meta-llama/llama-3.3-70b
# at ~50 RPD per :free model variant, which gives us extra headroom on top
# of Groq's 100K-TPD ceiling (~150 successful Q10 calls/day on v3).
#
# Fallback is only enabled if OPENROUTER_API_KEY is set in .env. Without it,
# behavior reverts to Groq-only; Q10 will need multiple days to complete
# across all 3 prompts × 200 sentences.
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
from groq import APIError, Groq, RateLimitError  # noqa: E402
from openai import APIError as OpenAIAPIError  # noqa: E402
from openai import OpenAI, RateLimitError as OpenAIRateLimitError  # noqa: E402

QUESTION = "q10"
PRIMARY_MODEL = _common.MODEL_70B                       # reported in aggregated.json
FALLBACK_MODEL = _common.MODEL_OPENROUTER_70B           # used internally when Groq dies

GROQ_RETRY_BACKOFF_SECONDS = (30, 60, 120)
OPENROUTER_RETRY_BACKOFF_SECONDS = (15, 30, 60, 120)
Q10_MAX_TOKENS = 1024


class GroqWithOpenRouterFallback:
    """Callable that tries Groq first, falls back to OpenRouter on Groq TPD exhaustion.

    Designed to plug into `_common.run_zero_shot`. Exposes a `last_provider`
    attribute (string) so `tag_sentence` records which backend served each
    call in the per-sentence JSONL.
    """

    def __init__(self) -> None:
        load_dotenv(_common.REPO_ROOT / ".env")

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not set in .env at project root.")
        self.groq_client = Groq(api_key=groq_key)

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            self.openrouter_client: OpenAI | None = OpenAI(
                api_key=openrouter_key,
                base_url=_common.OPENROUTER_BASE_URL,
            )
            print("OpenRouter fallback enabled (engages on Groq TPD exhaustion).")
        else:
            self.openrouter_client = None
            print("(no OPENROUTER_API_KEY in .env; Groq-only mode — will fail hard on TPD.)")

        self.groq_dead = False
        self.last_provider: str | None = None
        self.sleep_seconds = _common.SLEEP_GROQ

    # ── public callable ─────────────────────────────────────────────────────
    def __call__(self, model: str, system: str, user: str) -> str:
        # Primary path: Groq, with RPM retry.
        if not self.groq_dead:
            try:
                resp = self._call_groq(model, system, user)
                self.last_provider = f"groq:{model}"
                return resp
            except RateLimitError as e:
                if self._is_tpd_exhausted(e):
                    if self.openrouter_client is None:
                        print("\n  ⚠ Groq TPD exhausted; no OPENROUTER_API_KEY for fallback.")
                        raise
                    print(
                        f"\n  ⚠ Groq TPD exhausted ({self._extract_tpd_info(e)}); "
                        f"switching permanently to OpenRouter {FALLBACK_MODEL}."
                    )
                    self.groq_dead = True
                    self.sleep_seconds = _common.SLEEP_OPENROUTER
                    # Fall through to the OpenRouter path below.
                else:
                    # RPM or other transient — already handled by _call_groq's retry.
                    raise

        # Fallback path: OpenRouter.
        assert self.openrouter_client is not None  # mode guarded above
        resp = self._call_openrouter(system, user)
        self.last_provider = f"openrouter:{FALLBACK_MODEL}"
        return resp

    # ── provider implementations ────────────────────────────────────────────
    def _call_groq(self, model: str, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(len(GROQ_RETRY_BACKOFF_SECONDS) + 1):
            try:
                resp = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=_common.TEMPERATURE,
                    max_tokens=Q10_MAX_TOKENS,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                last_exc = e
                # If it's a TPD exhaustion, surface it to the outer __call__
                # immediately so the fallback can engage — don't waste retries.
                if self._is_tpd_exhausted(e):
                    raise
                if attempt < len(GROQ_RETRY_BACKOFF_SECONDS):
                    wait = GROQ_RETRY_BACKOFF_SECONDS[attempt]
                    print(f"  groq RPM rate limit; sleeping {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise
            except APIError as e:
                raise e
        raise last_exc if last_exc else RuntimeError("groq retries exhausted")

    def _call_openrouter(self, system: str, user: str) -> str:
        assert self.openrouter_client is not None
        last_exc: Exception | None = None
        ignored_providers: set[str] = set()
        for attempt in range(len(OPENROUTER_RETRY_BACKOFF_SECONDS) + 1):
            try:
                resp = self.openrouter_client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=_common.TEMPERATURE,
                    max_completion_tokens=Q10_MAX_TOKENS,
                    extra_body=self._openrouter_extra_body(ignored_providers),
                )
                return resp.choices[0].message.content or ""
            except OpenAIRateLimitError as e:
                last_exc = e
                provider_slug = self._openrouter_provider_slug(e)
                if provider_slug and provider_slug not in ignored_providers:
                    ignored_providers.add(provider_slug)
                    print(
                        f"  openrouter upstream provider {provider_slug!r} rate-limited; "
                        "asking router to avoid it on retry"
                    )
                if attempt < len(OPENROUTER_RETRY_BACKOFF_SECONDS):
                    wait = self._openrouter_retry_after(e) or OPENROUTER_RETRY_BACKOFF_SECONDS[attempt]
                    print(f"  openrouter rate limit; sleeping {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise
            except OpenAIAPIError as e:
                raise e
        raise last_exc if last_exc else RuntimeError("openrouter retries exhausted")

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _is_tpd_exhausted(e: Exception) -> bool:
        """Detect Groq's tokens-per-day exhaustion vs a recoverable RPM hit."""
        msg = str(e).lower()
        return "tokens per day" in msg or "tpd" in msg

    @staticmethod
    def _extract_tpd_info(e: Exception) -> str:
        """Pull the 'Limit X, Used Y' fragment from Groq's TPD error for the log."""
        msg = str(e)
        if "Limit" in msg:
            start = msg.find("Limit")
            return msg[start : start + 60]
        return "no TPD detail"

    @staticmethod
    def _openrouter_retry_after(e: Exception) -> float | None:
        """Return OpenRouter/upstream Retry-After seconds when the SDK exposes it."""
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        body = getattr(e, "body", None)
        if isinstance(body, dict):
            metadata = body.get("metadata")
            if isinstance(metadata, dict):
                retry_after = metadata.get("retry_after_seconds")
                if isinstance(retry_after, int | float):
                    return float(retry_after)

                raw_retry_after = metadata.get("retry_after_seconds_raw")
                if isinstance(raw_retry_after, int | float):
                    return float(raw_retry_after)
        return None

    @staticmethod
    def _openrouter_provider_slug(e: Exception) -> str | None:
        """Extract a provider slug from OpenRouter provider-error metadata."""
        body = getattr(e, "body", None)
        if not isinstance(body, dict):
            return None
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            return None
        provider_name = metadata.get("provider_name")
        if not isinstance(provider_name, str):
            return None
        slug = provider_name.strip().lower().replace(" ", "-")
        return slug or None

    @staticmethod
    def _openrouter_extra_body(ignored_providers: set[str]) -> dict[str, object]:
        """Ask OpenRouter to prefer throughput and optionally skip bad upstreams."""
        provider: dict[str, object] = {
            "sort": "throughput",
            "allow_fallbacks": True,
        }
        if ignored_providers:
            provider["ignore"] = sorted(ignored_providers)
        return {"provider": provider}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", choices=_common.PROMPT_NAMES, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    prompts_to_run = (args.prompt,) if args.prompt else _common.PROMPT_NAMES
    call_fn = GroqWithOpenRouterFallback()

    for prompt_name in prompts_to_run:
        _common.run_zero_shot(
            call_fn=call_fn,
            sleep_seconds=_common.SLEEP_GROQ,  # 2.5 s is safe for both providers (24 RPM)
            model=PRIMARY_MODEL,
            prompt_name=prompt_name,
            question=QUESTION,
            limit=args.limit,
        )
        if not args.skip_aggregate:
            metrics = _common.aggregate(
                question=QUESTION, prompt_name=prompt_name, model=PRIMARY_MODEL
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
