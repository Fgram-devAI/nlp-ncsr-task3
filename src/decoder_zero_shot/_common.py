"""Shared logic for Q9, Q10, and the bonus Gemini / GLM extensions — zero-shot
NER via any chat-completion LLM endpoint.

Provider-agnostic: the per-question entrypoints construct a `call_fn` callable
that takes `(model, system_prompt, user_msg)` and returns the raw response
string. Each entrypoint also picks the sleep interval that matches its
provider's free-tier RPM ceiling.

JSONL persistence: one line per sentence in
    results/<question>/<prompt_name>/sentences.jsonl
plus a single
    results/<question>/<prompt_name>/aggregated.json
at the end. JSONL is append-only and line-atomic; a crash loses at most the
in-flight request.

Q9 / Q10 use Groq's SDK (in their own entrypoint files); the Gemini extension
uses the OpenAI SDK with Gemini's OpenAI-compat base URL; the GLM extension
uses Z.ai's own SDK. All three plug into `run_zero_shot` via the `call_fn`
contract — no provider-specific code lives in this module.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import kagglehub
from dotenv import load_dotenv
from seqeval.metrics import f1_score as seqeval_f1
from sklearn.metrics import accuracy_score, balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).parent / "prompts"
RESULTS_ROOT = REPO_ROOT / "results"

# Load `.env` from the project root if present (GROQ_API_KEY lives there).
load_dotenv(REPO_ROOT / ".env")

# ─── Configuration ──────────────────────────────────────────────────────────
# Model identifiers exposed for entrypoints that want to import them from here.
MODEL_8B = "llama-3.1-8b-instant"
MODEL_70B = "llama-3.3-70b-versatile"               # Groq's 70B identifier
MODEL_OPENROUTER_70B = "meta-llama/llama-3.3-70b-instruct:free"  # OpenRouter free variant
MODEL_GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
MODEL_GLM_FLASH = "glm-4.7-flash"                   # change here if your Z.ai account has a different alias

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"        # OpenAI-compatible endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"    # OpenAI-compatible endpoint

# Default sleep windows per provider — match free-tier RPM ceilings with margin.
SLEEP_GROQ = 2.5         # 24 RPM, under Groq's 30 RPM cap
SLEEP_CEREBRAS = 2.5     # 24 RPM, under Cerebras's 30 RPM cap
SLEEP_OPENROUTER = 4.0   # 15 RPM, under OpenRouter free-tier 20 RPM cap
SLEEP_GEMINI = 6.0       # 10 RPM, matches Gemini Flash-Lite free-tier (per user spec)
SLEEP_GLM = 3.0          # 20 RPM, matches GLM-Flash free-tier (per user spec)

MAX_TOKENS = 4096        # generous; longest sentence × ~4 chars × 2 ≈ 1.5k tokens
TEMPERATURE = 0          # deterministic output for reproducibility

VALID_TAGS = {
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
}

PROMPT_NAMES = ("v1_minimal", "v2_strict_json", "v3_with_glossary")


# ─── Type contract for provider call_fn ─────────────────────────────────────
# A `call_fn` is a Callable[(model: str, system_prompt: str, user_msg: str), str]
# It MUST return the raw response text (the model's content), or raise on
# unrecoverable error. Provider-specific retry / backoff on rate-limit errors
# is the call_fn's responsibility (each provider's SDK has its own exception
# types and rate-limit semantics).
CallFn = Callable[[str, str, str], str]


def load_prompt(name: str) -> str:
    """Read prompt body from prompts/<name>.md."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_first_200_test_sentences() -> list[dict[str, list[str]]]:
    """Read the first 200 sentences from CoNLL-2003 test split."""
    dataset_path = Path(kagglehub.dataset_download("alaakhaled/conll003-englishversion"))
    test_file = next(dataset_path.rglob("test.txt"))

    sentences: list[dict[str, list[str]]] = []
    tokens: list[str] = []
    ner: list[str] = []

    with open(test_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if tokens:
                    sentences.append({"tokens": list(tokens), "ner_tags": list(ner)})
                    tokens.clear()
                    ner.clear()
                    if len(sentences) >= 200:
                        break
            else:
                parts = line.strip().split(" ")
                if len(parts) >= 4:
                    tokens.append(parts[0])
                    ner.append(parts[3])
    if tokens and len(sentences) < 200:
        sentences.append({"tokens": tokens, "ner_tags": ner})
    return sentences[:200]


# ─── Per-sentence call + parse ──────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json_array(raw: str) -> str:
    """Strip markdown fences and slice to the outer [...]."""
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no JSON array brackets found")
    return cleaned[start : end + 1]


def parse_response(input_tokens: list[str], raw: str) -> dict[str, Any]:
    """Parse the LLM response into a {tokens, tags, parse_failed, raw} dict."""
    result: dict[str, Any] = {
        "tokens": input_tokens,
        "tags": None,
        "parse_failed": False,
        "raw": raw,
    }
    try:
        array_text = _extract_json_array(raw)
        parsed: Any = json.loads(array_text)
    except (json.JSONDecodeError, ValueError) as e:
        result["parse_failed"] = True
        result["error"] = f"parse: {e}"
        return result

    if not isinstance(parsed, list):
        result["parse_failed"] = True
        result["error"] = f"not_a_list: {type(parsed).__name__}"
        return result

    if len(parsed) != len(input_tokens):
        result["parse_failed"] = True
        result["error"] = f"len_mismatch: got {len(parsed)}, expected {len(input_tokens)}"
        return result

    tags: list[str] = []
    unknown_count = 0
    for item in parsed:
        if not isinstance(item, dict) or "tag" not in item:
            result["parse_failed"] = True
            result["error"] = "item_missing_tag"
            return result
        tag = str(item["tag"]).strip()
        if tag not in VALID_TAGS:
            unknown_count += 1
            tag = "O"  # coerce unknown tags to O; count separately
        tags.append(tag)

    result["tags"] = tags
    result["unknown_tag_count"] = unknown_count
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file as a list of dicts. Empty list if file missing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: {path}:{line_no} is malformed JSON ({e}); skipping")
    return rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a single line to `path`. Atomic per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_indices(jsonl_path: Path) -> set[int]:
    """Return the set of `sentence_idx` values already present in `jsonl_path`."""
    completed: set[int] = set()
    for row in _read_jsonl(jsonl_path):
        idx = row.get("sentence_idx")
        if isinstance(idx, int):
            completed.add(idx)
    return completed


def tag_sentence(
    call_fn: CallFn,
    model: str,
    system_prompt: str,
    tokens: list[str],
    sentence_idx: int,
) -> dict[str, Any]:
    """Tag one sentence using a provider-agnostic call_fn. No file I/O.

    Exceptions from `call_fn` are caught and recorded as `parse_failed: True`
    with the error string preserved. Provider-specific retry on transient
    errors (rate limit, server overload, transient network) is the call_fn's
    responsibility.
    """
    user_msg = f"Tokens: {json.dumps(tokens, ensure_ascii=False)}"

    try:
        raw = call_fn(model, system_prompt, user_msg)
    except Exception as e:
        return {
            "sentence_idx": sentence_idx,
            "tokens": tokens,
            "tags": None,
            "parse_failed": True,
            "raw": "",
            "error": f"call_error: {type(e).__name__}: {e}",
            "provider": getattr(call_fn, "last_provider", None),
        }

    result = parse_response(tokens, raw)
    result["sentence_idx"] = sentence_idx
    # If the call_fn exposes a last_provider attribute (e.g. multi-provider
    # fallback wrapper), record it so the JSONL captures which backend served
    # each individual call.
    if hasattr(call_fn, "last_provider"):
        result["provider"] = call_fn.last_provider
    return result


# ─── Driver ────────────────────────────────────────────────────────────────

def run_zero_shot(
    *,
    call_fn: CallFn,
    sleep_seconds: float,
    model: str,
    prompt_name: str,
    question: str,
    limit: int | None = None,
) -> None:
    """Run the 200-sentence sweep for one (provider, model, prompt) combination.

    `call_fn` provides the HTTP call (provider-specific); `sleep_seconds`
    paces it to stay under the provider's RPM ceiling.

    Appends each completed sentence as a JSONL line to
    `results/<question>/<prompt_name>/sentences.jsonl`. Resume-safe:
    sentences whose `sentence_idx` is already in the JSONL are skipped.
    """
    output_dir = RESULTS_ROOT / question / prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "sentences.jsonl"

    print(f"\n=== {question} — model={model} — prompt={prompt_name} ===")
    print(f"output: {jsonl_path}")
    print(f"pacing: sleep {sleep_seconds:.1f}s between calls (≈{60 / sleep_seconds:.0f} RPM)")
    if limit is not None:
        print(f"LIMIT: only running on first {limit} sentences (smoke mode)")

    system_prompt = load_prompt(prompt_name)
    sentences = load_first_200_test_sentences()
    if limit is not None:
        sentences = sentences[:limit]

    completed_indices = load_completed_indices(jsonl_path)
    if completed_indices:
        print(f"resume: {len(completed_indices)} sentence(s) already in JSONL; skipping those")

    completed = 0
    skipped = 0
    failed = 0

    for i, sent in enumerate(sentences):
        if i in completed_indices:
            skipped += 1
            continue

        result = tag_sentence(
            call_fn=call_fn,
            model=model,
            system_prompt=system_prompt,
            tokens=sent["tokens"],
            sentence_idx=i,
        )
        # Persist immediately so a crash next iteration loses at most the
        # in-flight request — never a completed one.
        _append_jsonl(jsonl_path, result)

        if result.get("parse_failed"):
            failed += 1
            err = result.get("error", "unknown")
            print(f"  [{i:03d}] FAILED: {err}")
        else:
            completed += 1
            n_tags = len(result["tags"]) if result.get("tags") else 0
            print(f"  [{i:03d}] OK ({n_tags} tags)")

        current_sleep = getattr(call_fn, "sleep_seconds", sleep_seconds)
        time.sleep(current_sleep)

    print(f"\n=== done. new_completed={completed}, skipped={skipped}, new_failed={failed} ===")
    print(f"total lines in {jsonl_path.name}: {sum(1 for _ in open(jsonl_path))}")


# ─── Aggregation ────────────────────────────────────────────────────────────

def aggregate(question: str, prompt_name: str, *, model: str) -> dict[str, Any]:
    """Read sentences.jsonl and compute summary metrics."""
    output_dir = RESULTS_ROOT / question / prompt_name
    jsonl_path = output_dir / "sentences.jsonl"
    sentences = load_first_200_test_sentences()

    label2id = {t: i for i, t in enumerate(sorted(VALID_TAGS))}

    # Index rows by sentence_idx so we can pair them with the gold tags.
    rows_by_idx: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(jsonl_path):
        idx = row.get("sentence_idx")
        if isinstance(idx, int):
            rows_by_idx[idx] = row

    y_true_flat: list[int] = []
    y_pred_flat: list[int] = []
    y_true_tags: list[list[str]] = []
    y_pred_tags: list[list[str]] = []
    parse_failed = 0
    unknown_tag_total = 0
    completed_count = 0

    for i, sent in enumerate(sentences):
        data = rows_by_idx.get(i)
        if data is None:
            continue
        if data.get("parse_failed"):
            parse_failed += 1
            continue
        tags_pred = data.get("tags")
        if not isinstance(tags_pred, list):
            parse_failed += 1
            continue
        tags_true = sent["ner_tags"]
        if len(tags_pred) != len(tags_true):
            parse_failed += 1
            continue
        completed_count += 1
        unknown_tag_total += int(data.get("unknown_tag_count", 0))
        y_true_flat.extend(label2id[t] for t in tags_true)
        y_pred_flat.extend(label2id[t] for t in tags_pred)
        y_true_tags.append(tags_true)
        y_pred_tags.append(tags_pred)

    metrics: dict[str, Any] = {
        "question": question,
        "model": model,
        "prompt": prompt_name,
        "test_subset": "first_200_test_sentences",
        "completed_count": completed_count,
        "parse_failed_count": parse_failed,
        "unknown_tag_total": unknown_tag_total,
    }

    if completed_count > 0:
        metrics["test"] = {
            "token_micro_accuracy": float(accuracy_score(y_true_flat, y_pred_flat)),  # pyright: ignore[reportArgumentType]
            "token_macro_accuracy": float(balanced_accuracy_score(y_true_flat, y_pred_flat)),  # pyright: ignore[reportArgumentType]
            "entity_micro_f1": float(seqeval_f1(y_true_tags, y_pred_tags)),  # pyright: ignore[reportArgumentType]
            "entity_macro_f1": float(seqeval_f1(y_true_tags, y_pred_tags, average="macro")),  # pyright: ignore[reportArgumentType]
        }
    else:
        metrics["test"] = None
        metrics["note"] = "no sentences successfully parsed; metrics unavailable"

    out_path = output_dir / "aggregated.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
