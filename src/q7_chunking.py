#
# Q7 — NER-BERT modified for text chunking, 3 seeded runs.
#
# Modification of NER-BERT.py: same model architecture, same hyperparameters,
# same 3-seed sweep as Q1 — but the classifier head is trained on chunk tags
# (the `chunk_tags` column of CoNLL-2003) instead of NER tags. The output tag
# space jumps from 9 BIO NER classes to ~22 BIO chunk classes (B-NP / I-NP /
# B-VP / I-VP / B-PP / B-ADJP / …).
#
# Chunk tags ARE BIO-prefixed, so seqeval works as-is — the F1 values that
# `entity_*` keys reported for Q1 now measure chunk-span F1 (NP, VP, PP, …)
# rather than entity-span F1 (PER, ORG, LOC, MISC). We keep the SAME JSON key
# names (`entity_micro_f1`, `entity_macro_f1`) for aggregator parity with Q1's
# results; the report explains the semantic shift in one sentence (see
# brainstorming spec D3 / risk R3).
#
# Persists per-seed JSON to results/q7/. Runs locally on MPS (Apple Silicon)
# by default; CUDA on Colab; CPU as last-resort fallback. See CLAUDE.md §3.
#
# Q3-style error analysis (PART OF Q7 per assignment §7): when the running
# seed equals Q3_ANALYSIS_SEED (42 by default) the script additionally dumps
# results/q7/q3_analysis_seed{seed}.json with the worst-tagged test sentence
# (>=10 tokens, >=1 true chunk, >=1 error, lowest per-sentence seqeval F1)
# and the model's predictions on the wild sentence shared with Q3 NER and Q6
# POS. The 3-seed sweep is otherwise unchanged.
#
# Diff anchors for the report:
#   git diff src/NER-BERT.py        src/q7_chunking.py  → full Q7 modification
#   git diff src/q1_baseline_3runs.py src/q7_chunking.py → concentrated Q7 delta
#

# dependencies
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.optim as optim
from transformers import AutoTokenizer, BertForTokenClassification
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from tqdm.auto import tqdm

# hyper-parameters (identical to Q1 for a fair comparison)
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-5
SEEDS = [42, 43, 44]   # 3-run sweep; set to [42] for a fast local Q3-only run

# results location — separate q7 namespace
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "q7"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Q3-style error analysis constants. Triggered only when the running seed
# equals Q3_ANALYSIS_SEED. Wild-sentence tokens are shared with Q3 NER
# (src/q3_error_analysis.py) and Q6 POS (src/q6_pos_tagging.py) so the three
# error analyses align like-for-like across the report.
Q3_ANALYSIS_SEED = 42
WILD_SENTENCE_TOKENS = [
    "Greek", "startup", "Helios", "bought", "Bavarian", "rival",
    "KronosAI", "in", "Berlin", "yesterday", "for", "2",
    "billion", "euros", ".",
]
WILD_SENTENCE_CHUNK_TAGS = [
    "B-NP", "I-NP", "I-NP", "B-VP", "B-NP", "I-NP",
    "I-NP", "B-PP", "B-NP", "B-NP", "B-PP", "B-NP",
    "I-NP", "I-NP", "O",
]

# fetch the data via kagglehub (cached at ~/.cache/kagglehub/...)
print("downloading dataset (cached on second run)")
dataset_path = Path(kagglehub.dataset_download("alaakhaled/conll003-englishversion"))
train_file = next(dataset_path.rglob("train.txt"))
valid_file = next(dataset_path.rglob("valid.txt"))
test_file = next(dataset_path.rglob("test.txt"))

# device selection: MPS on M4, CUDA on Colab, CPU as fallback
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)


# read the data files (unchanged from NER-BERT.py — still parses all 4 tag columns)
def load_sentences(filepath):

    sentences = []
    tokens = []
    pos_tags = []
    chunk_tags = []
    ner_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # sentence boundary
            if (line.startswith('-DOCSTART-') or line.strip() == ''):
                if len(tokens) > 0:
                    sentences.append({
                        'tokens': tokens,
                        'pos_tags': pos_tags,
                        'chunk_tags': chunk_tags,
                        'ner_tags': ner_tags
                    })
                    tokens = []
                    pos_tags = []
                    chunk_tags = []
                    ner_tags = []
            else:
                l = line.strip().split(' ')
                if len(l) >= 4:
                    tokens.append(l[0])
                    pos_tags.append(l[1])
                    chunk_tags.append(l[2])
                    ner_tags.append(l[3])
    # last sentence if file doesn't end with blank line
    if len(tokens) > 0:
        sentences.append({
            'tokens': tokens,
            'pos_tags': pos_tags,
            'chunk_tags': chunk_tags,
            'ner_tags': ner_tags
        })
    return sentences


print('loading data')
train_sentences = load_sentences(train_file)
test_sentences = load_sentences(test_file)
valid_sentences = load_sentences(valid_file)
print(f"train={len(train_sentences)}, valid={len(valid_sentences)}, test={len(test_sentences)}")

# ----- Q7 CHANGE: build tag set from CHUNK column instead of NER column -------
all_tags = sorted({tag for s in train_sentences for tag in s['chunk_tags']})
label2id = {tag: i for i, tag in enumerate(all_tags)}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(all_tags)
print('Tagset size:', num_labels)
print('Tags:', all_tags)
# ------------------------------------------------------------------------------

# load BERT tokenizer (AutoTokenizer dispatches to BertTokenizerFast for bert-base-uncased)
bert_version = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_version)


# unchanged from NER-BERT.py
def align_label(tokens, labels):
    word_ids = tokens.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id.get(labels[word_idx], -100))
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids


# ----- Q7 CHANGE: encode reads from `chunk_tags` instead of `ner_tags` --------
def encode(sentence):
    encodings = tokenizer(
        sentence['tokens'],
        truncation=True,
        padding='max_length',
        is_split_into_words=True,
        return_tensors='pt'
    )
    labels = align_label(encodings, sentence['chunk_tags'])
    return {
        'input_ids': encodings['input_ids'].squeeze(0),
        'attention_mask': encodings['attention_mask'].squeeze(0),
        'labels': torch.tensor(labels, dtype=torch.long)
    }
# ------------------------------------------------------------------------------


print('encoding data')
train_dataset = [encode(sentence) for sentence in train_sentences]
valid_dataset = [encode(sentence) for sentence in valid_sentences]
test_dataset = [encode(sentence) for sentence in test_sentences]


# unchanged from NER-BERT.py
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = InputDataset(train_dataset)
valid_dataset = InputDataset(valid_dataset)
test_dataset = InputDataset(test_dataset)


# unchanged from NER-BERT.py — chunk tags are BIO so seqeval works as-is
def EvaluateModel(model, data_loader):
    model.eval()
    Y_actual_flat, Y_preds_flat = [], []
    y_true_tags, y_pred_tags = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            for idx in range(batch['labels'].size(0)):
                true_values_all = batch['labels'][idx]
                mask = (true_values_all != -100)

                true_values = true_values_all[mask]
                pred_values = preds[idx][mask]

                Y_actual_flat.append(true_values)
                Y_preds_flat.append(pred_values)

                true_tags_sent = [id2label[i] for i in true_values.tolist()]
                pred_tags_sent = [id2label[i] for i in pred_values.tolist()]
                y_true_tags.append(true_tags_sent)
                y_pred_tags.append(pred_tags_sent)

    Y_actual_flat = torch.cat(Y_actual_flat).detach().cpu().numpy()
    Y_preds_flat = torch.cat(Y_preds_flat).detach().cpu().numpy()
    return Y_actual_flat, Y_preds_flat, y_true_tags, y_pred_tags


# unchanged from NER-BERT.py — for chunking, the "entity" lines measure CHUNK
# spans (NP, VP, PP, ADJP, …) rather than NE spans. The report makes this
# explicit when comparing to Q1.
def report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags, split_name):
    print(f"\n=== {split_name} — Token-level metrics ===")
    print("Accuracy          : {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
    print("Balanced accuracy : {:.3f}".format(balanced_accuracy_score(Y_actual, Y_preds)))
    print(f"\n=== {split_name} — Chunk-span metrics (seqeval) ===")
    print("Precision : {:.3f}".format(seqeval_precision(y_true_tags, y_pred_tags)))
    print("Recall    : {:.3f}".format(seqeval_recall(y_true_tags, y_pred_tags)))
    print("F1        : {:.3f}".format(seqeval_f1(y_true_tags, y_pred_tags)))


# deterministic seeding (CLAUDE.md §3 / spec §D2)
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# capture git sha for traceability
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


GIT_SHA = _git_sha()
PER_SEED_RESULTS: list[dict] = []

# 3-seed sweep — same structure as Q1, with chunk tags as the target
for run_index, seed in enumerate(SEEDS):
    print(f"\n{'#' * 60}\n# Q7 run {run_index + 1}/{len(SEEDS)} — seed={seed}\n{'#' * 60}")

    set_seeds(seed)

    # initialize the model afresh per seed (classifier head re-inits stochastically).
    # num_labels here is ~22 (chunk BIO classes); larger than Q1's 9 but smaller
    # than Q6's ~45 POS classes. Encoder cost still dominates wall-time.
    print('initializing the model')
    model = BertForTokenClassification.from_pretrained(
        bert_version,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    # `from_pretrained` returns a union of model classes; pyright loses narrowing on .to(device).
    model = model.to(device)  # pyright: ignore[reportArgumentType]
    optimizer = optim.AdamW(params=model.parameters(), lr=LR)

    # seed the train-loader's shuffle generator so batch order is reproducible
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator,
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print('training the model')
    train_start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        print(f"epoch {epoch + 1}/{EPOCHS}")
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Y_actual, Y_preds, y_true_tags, y_pred_tags = EvaluateModel(model, valid_loader)
        report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags,
                       split_name=f"Validation (epoch {epoch + 1})")
    training_seconds = time.perf_counter() - train_start

    print(f"\napplying the model to the test set (seed={seed})")
    Y_actual, Y_preds, y_true_tags, y_pred_tags = EvaluateModel(model, test_loader)
    report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags, split_name=f"Test (seed={seed})")

    # detailed token-level report
    label_ids_sorted = list(range(num_labels))
    target_names = [id2label[i] for i in label_ids_sorted]
    print(f"\n=== Test (seed={seed}) — Token-level classification report (sklearn) ===")
    # sklearn stubs claim zero_division must be str; runtime accepts int 0 / 1 / np.nan / "warn".
    print(classification_report(
        Y_actual, Y_preds, labels=label_ids_sorted,
        target_names=target_names,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    ))
    print(f"=== Test (seed={seed}) — Chunk-span classification report (seqeval) ===")
    print(seqeval_report(
        y_true_tags, y_pred_tags, digits=3,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    ))

    # persist this run's metrics JSON — Q7 shape reuses Q1's `entity_*` keys
    # for aggregator parity; the values measure chunk-span F1 here.
    metrics = {
        "question": "Q7",
        "script": "src/q7_chunking.py",
        "model": bert_version,
        "task": "chunking",
        "tag_column": "chunk_tags",
        "seed": seed,
        "run_index": run_index,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "training_seconds": training_seconds,
        "num_labels": num_labels,
        # sklearn + seqeval stubs annotate scalar metrics as list[float] | float — at runtime
        # they return scalars. Per-line pyright ignores below are stub-quirk silencers, not bugs.
        "test": {
            "token_micro_accuracy": float(accuracy_score(Y_actual, Y_preds)),  # pyright: ignore[reportArgumentType]
            "token_macro_accuracy": float(balanced_accuracy_score(Y_actual, Y_preds)),  # pyright: ignore[reportArgumentType]
            # Below: `entity_*` keys hold CHUNK-span F1 for Q7 (vs entity-span F1 for Q1).
            "entity_micro_f1": float(seqeval_f1(y_true_tags, y_pred_tags)),  # pyright: ignore[reportArgumentType]
            "entity_macro_f1": float(seqeval_f1(y_true_tags, y_pred_tags, average="macro")),  # pyright: ignore[reportArgumentType]
        },
        "device": str(device),
        "git_commit": GIT_SHA,
    }
    out_path = RESULTS_DIR / f"seed_{seed}.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved metrics: {out_path}")
    PER_SEED_RESULTS.append(metrics)

    # ── Q3-style error analysis branch (chunking) ───────────────────────────
    # Reuses the per-sentence lists already built by EvaluateModel on the
    # test set — no second inference pass needed. Chunks are BIO so seqeval
    # F1 applies (same ranking as Q3 NER, not Q6 POS's token accuracy).
    if seed == Q3_ANALYSIS_SEED:
        print(f"\n--- Q3-style error analysis (seed={seed}) ---")
        assert len(y_true_tags) == len(test_sentences), (
            f"per-sentence count {len(y_true_tags)} != "
            f"test sentences {len(test_sentences)}"
        )

        def _per_sentence_chunk_f1(true: list[str], pred: list[str]) -> float:
            # seqeval F1 on a single sentence pair. Sentences with no true
            # chunk (all-O) are filtered out before this call.
            return float(seqeval_f1([true], [pred]))  # pyright: ignore[reportArgumentType]

        q3_candidates: list[tuple[int, list[str], list[str], list[str], float]] = []
        for cand_idx, (cand_sent, cand_true, cand_pred) in enumerate(
            zip(test_sentences, y_true_tags, y_pred_tags)
        ):
            cand_tokens = cand_sent["tokens"]
            if len(cand_tokens) < 10:
                continue
            if not any(t != "O" for t in cand_true):
                continue
            if not any(t != p for t, p in zip(cand_true, cand_pred)):
                continue
            cand_f1 = _per_sentence_chunk_f1(cand_true, cand_pred)
            q3_candidates.append((cand_idx, cand_tokens, cand_true, cand_pred, cand_f1))

        if not q3_candidates:
            print("  no candidate sentence found — skipping Q3 dump")
        else:
            q3_candidates.sort(key=lambda x: x[4])
            (
                worst_idx,
                worst_tokens,
                worst_true,
                worst_pred,
                worst_f1,
            ) = q3_candidates[0]
            print(
                f"  failing sentence: idx={worst_idx} "
                f"per-sentence-f1={worst_f1:.4f} "
                f"length={len(worst_tokens)} candidates_considered={len(q3_candidates)}"
            )
            for tok, t, p in zip(worst_tokens, worst_true, worst_pred):
                marker = "X" if t != p else " "
                print(f"    {marker} {tok:25s}  true={t:7s}  pred={p:7s}")

            # Wild-sentence inference (first-subword-wins, same alignment as training).
            print(f"\n  tagging wild sentence ({len(WILD_SENTENCE_TOKENS)} tokens)")
            wild_encoding = tokenizer(
                WILD_SENTENCE_TOKENS,
                truncation=True,
                padding="max_length",
                is_split_into_words=True,
                return_tensors="pt",
            )
            wild_inputs = {
                "input_ids": wild_encoding["input_ids"].to(device),
                "attention_mask": wild_encoding["attention_mask"].to(device),
            }
            with torch.no_grad():
                wild_logits = model(**wild_inputs).logits
            wild_preds_full = torch.argmax(wild_logits, dim=-1)[0].tolist()
            wild_word_ids = wild_encoding.word_ids()
            wild_pred_tags: list[str] = []
            prev_word_idx: int | None = None
            for word_idx, pred_id in zip(wild_word_ids, wild_preds_full):
                if word_idx is None or word_idx == prev_word_idx:
                    prev_word_idx = word_idx
                    continue
                wild_pred_tags.append(id2label[pred_id])
                prev_word_idx = word_idx
            assert len(wild_pred_tags) == len(WILD_SENTENCE_TOKENS)

            wild_f1 = _per_sentence_chunk_f1(WILD_SENTENCE_CHUNK_TAGS, wild_pred_tags)
            wild_errors = sum(
                t != p for t, p in zip(WILD_SENTENCE_CHUNK_TAGS, wild_pred_tags)
            )
            for tok, t, p in zip(WILD_SENTENCE_TOKENS, WILD_SENTENCE_CHUNK_TAGS, wild_pred_tags):
                marker = "X" if t != p else " "
                print(f"    {marker} {tok:25s}  true={t:7s}  pred={p:7s}")

            q3_summary = {
                "question": "Q3 (chunking)",
                "script": "src/q7_chunking.py",
                "model": bert_version,
                "task": "chunking",
                "seed": seed,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "training_seconds": training_seconds,
                "device": str(device),
                "git_commit": GIT_SHA,
                "test_entity_micro_f1": metrics["test"]["entity_micro_f1"],
                "n_candidate_sentences": len(q3_candidates),
                "failing_sentence": {
                    "test_idx": worst_idx,
                    "n_tokens": len(worst_tokens),
                    "n_errors": sum(t != p for t, p in zip(worst_true, worst_pred)),
                    "per_sentence_f1": worst_f1,
                    "tokens": worst_tokens,
                    "true_tags": worst_true,
                    "pred_tags": worst_pred,
                },
                "wild_sentence": {
                    "n_tokens": len(WILD_SENTENCE_TOKENS),
                    "n_errors": wild_errors,
                    "per_sentence_f1": wild_f1,
                    "tokens": WILD_SENTENCE_TOKENS,
                    "true_tags": WILD_SENTENCE_CHUNK_TAGS,
                    "pred_tags": wild_pred_tags,
                },
            }
            q3_out_path = RESULTS_DIR / f"q3_analysis_seed{seed}.json"
            q3_out_path.write_text(json.dumps(q3_summary, indent=2) + "\n", encoding="utf-8")
            print(f"\n  saved Q3 analysis: {q3_out_path}")

    # free memory between seeds (helps on MPS and on Colab T4)
    del model, optimizer, train_loader, valid_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# aggregate across seeds and print mean ± stdev — same 4 metrics as Q1
print(f"\n{'#' * 60}\n# Q7 — summary across {len(SEEDS)} seeds\n{'#' * 60}")
metric_keys = ["token_micro_accuracy", "token_macro_accuracy", "entity_micro_f1", "entity_macro_f1"]
print(f"{'metric':30s}  {'mean':>10s}  {'stdev':>10s}  values")
for k in metric_keys:
    vals = [r["test"][k] for r in PER_SEED_RESULTS]
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    vals_str = ", ".join(f"{v:.4f}" for v in vals)
    print(f"{k:30s}  {m:>10.4f}  {s:>10.4f}  [{vals_str}]")
times = [r["training_seconds"] for r in PER_SEED_RESULTS]
print(f"{'training_seconds':30s}  {statistics.mean(times):>10.1f}  "
      f"{statistics.stdev(times) if len(times) > 1 else 0.0:>10.1f}  "
      f"[{', '.join(f'{t:.1f}' for t in times)}]")
