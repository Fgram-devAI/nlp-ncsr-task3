#
# Q3 — Error analysis on the NER baseline.
#
# Self-contained: re-trains the Q1 BERT-base NER baseline with seed=42
# (identical hyperparameters to q1_baseline_3runs.py) and then surfaces:
#   (1) the worst-tagged test sentence with >= 10 tokens, at least one
#       true entity, and at least one wrong tag — selected by lowest
#       per-sentence seqeval entity-F1; and
#   (2) the model's predictions on a single invented "wild" sentence
#       defined as a constant below. The wild sentence stays shared so
#       a future POS / chunking error-analysis can reuse its tokens
#       with their own gold tags for like-for-like comparison.
#
# Inputs : CoNLL-2003 via kagglehub (same cache as Q1).
# Outputs: results/q3/summary_seed42.json
#          results/q3/predictions_seed42.json   (full test split)
#

# dependencies
import json
import random
import subprocess
import time
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.optim as optim
from seqeval.metrics import f1_score as seqeval_f1
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForTokenClassification

# hyperparameters identical to Q1 baseline so the analysis is on the same model
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-5
SEED = 42

# Invented "wild" news sentence used by Q3 (NER). The tokens are reused if a
# future POS / chunking error-analysis script lands, with their respective
# gold label sequences.
WILD_SENTENCE_TOKENS = [
    "Greek", "startup", "Helios", "bought", "Bavarian", "rival",
    "KronosAI", "in", "Berlin", "yesterday", "for", "2",
    "billion", "euros", ".",
]
WILD_SENTENCE_NER_TAGS = [
    "B-MISC", "O", "B-ORG", "O", "B-MISC", "O",
    "B-ORG", "O", "B-LOC", "O", "O", "O",
    "O", "O", "O",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "q3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── data loading (mirrors q1_baseline_3runs.py) ─────────────────────────────
print("downloading dataset (cached on second run)")
dataset_path = Path(kagglehub.dataset_download("alaakhaled/conll003-englishversion"))
train_file = next(dataset_path.rglob("train.txt"))
valid_file = next(dataset_path.rglob("valid.txt"))
test_file = next(dataset_path.rglob("test.txt"))

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)


def load_sentences(filepath):
    sentences = []
    tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if len(tokens) > 0:
                    sentences.append({
                        "tokens": tokens,
                        "pos_tags": pos_tags,
                        "chunk_tags": chunk_tags,
                        "ner_tags": ner_tags,
                    })
                    tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
            else:
                fields = line.strip().split(" ")
                if len(fields) >= 4:
                    tokens.append(fields[0])
                    pos_tags.append(fields[1])
                    chunk_tags.append(fields[2])
                    ner_tags.append(fields[3])
    if len(tokens) > 0:
        sentences.append({
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "ner_tags": ner_tags,
        })
    return sentences


print("loading data")
train_sentences = load_sentences(train_file)
valid_sentences = load_sentences(valid_file)
test_sentences = load_sentences(test_file)
print(f"train={len(train_sentences)}, valid={len(valid_sentences)}, test={len(test_sentences)}")

all_tags = sorted({tag for s in train_sentences for tag in s["ner_tags"]})
label2id = {tag: i for i, tag in enumerate(all_tags)}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(all_tags)

bert_version = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_version)


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


def encode(sentence):
    encodings = tokenizer(
        sentence["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
        return_tensors="pt",
    )
    labels = align_label(encodings, sentence["ner_tags"])
    return {
        "input_ids": encodings["input_ids"].squeeze(0),
        "attention_mask": encodings["attention_mask"].squeeze(0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


print("encoding data")
train_encoded = [encode(s) for s in train_sentences]
valid_encoded = [encode(s) for s in valid_sentences]
test_encoded = [encode(s) for s in test_sentences]


class InputDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = InputDataset(train_encoded)
valid_dataset = InputDataset(valid_encoded)
test_dataset = InputDataset(test_encoded)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


GIT_SHA = _git_sha()


# ── train one model with seed=42 (matches Q1 seed=42 exactly) ───────────────
set_seeds(SEED)
print(f"\n{'#' * 60}\n# Q3 — training NER baseline (seed={SEED})\n{'#' * 60}")

model = BertForTokenClassification.from_pretrained(
    bert_version,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
model = model.to(device)  # pyright: ignore[reportArgumentType]
optimizer = optim.AdamW(params=model.parameters(), lr=LR)

train_generator = torch.Generator().manual_seed(SEED)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator,
)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

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
training_seconds = time.perf_counter() - train_start
print(f"\ntraining done in {training_seconds:.1f}s")


# ── per-sentence inference on the test set ──────────────────────────────────
print("\nrunning per-sentence inference on the test set")
model.eval()
all_true_tags: list[list[str]] = []
all_pred_tags: list[list[str]] = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch_on_device)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        for idx in range(batch_on_device["labels"].size(0)):
            true_values_all = batch_on_device["labels"][idx]
            mask = true_values_all != -100
            true_values = true_values_all[mask].tolist()
            pred_values = preds[idx][mask].tolist()
            all_true_tags.append([id2label[i] for i in true_values])
            all_pred_tags.append([id2label[i] for i in pred_values])

assert len(all_true_tags) == len(test_sentences), (
    f"per-sentence prediction count {len(all_true_tags)} != test sentences {len(test_sentences)}"
)


# ── pick the worst-tagged sentence (>=10 tokens, has entities, has errors) ──
def per_sentence_entity_f1(true: list[str], pred: list[str]) -> float:
    # seqeval F1 on a single sentence pair. Sentences without true entities
    # are filtered out before this call so the F1 is meaningful.
    return float(seqeval_f1([true], [pred]))  # pyright: ignore[reportArgumentType]


candidates: list[tuple[int, list[str], list[str], list[str], float]] = []
for idx, (sent, true_tags, pred_tags) in enumerate(
    zip(test_sentences, all_true_tags, all_pred_tags)
):
    tokens = sent["tokens"]
    if len(tokens) < 10:
        continue
    if not any(t != "O" for t in true_tags):
        continue
    if not any(t != p for t, p in zip(true_tags, pred_tags)):
        continue
    f1 = per_sentence_entity_f1(true_tags, pred_tags)
    candidates.append((idx, tokens, true_tags, pred_tags, f1))

if not candidates:
    raise RuntimeError(
        "no test sentence found with >=10 tokens, true entities, and at least one error"
    )
candidates.sort(key=lambda x: x[4])
worst_idx, worst_tokens, worst_true, worst_pred, worst_f1 = candidates[0]

print(
    f"\nselected failing sentence — idx={worst_idx} "
    f"per-sentence-f1={worst_f1:.4f} length={len(worst_tokens)}"
)
print(f"  candidates considered: {len(candidates)}")
for tok, t, p in zip(worst_tokens, worst_true, worst_pred):
    marker = "X" if t != p else " "
    print(f"  {marker} {tok:25s}  true={t:7s}  pred={p:7s}")


# ── tag the wild sentence ───────────────────────────────────────────────────
print(f"\ntagging wild sentence ({len(WILD_SENTENCE_TOKENS)} tokens)")
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

# Map subword predictions back to one prediction per original word
# (first-subword-wins, same alignment Q1 uses for labels).
wild_pred_tags: list[str] = []
prev_word_idx: int | None = None
for word_idx, pred_id in zip(wild_word_ids, wild_preds_full):
    if word_idx is None or word_idx == prev_word_idx:
        prev_word_idx = word_idx
        continue
    wild_pred_tags.append(id2label[pred_id])
    prev_word_idx = word_idx

assert len(wild_pred_tags) == len(WILD_SENTENCE_TOKENS), (
    f"wild tagging length mismatch: {len(wild_pred_tags)} != {len(WILD_SENTENCE_TOKENS)}"
)
print("\nwild sentence predictions:")
for tok, t, p in zip(WILD_SENTENCE_TOKENS, WILD_SENTENCE_NER_TAGS, wild_pred_tags):
    marker = "X" if t != p else " "
    print(f"  {marker} {tok:25s}  true={t:7s}  pred={p:7s}")


# ── persist results ─────────────────────────────────────────────────────────
test_micro_f1 = float(seqeval_f1(all_true_tags, all_pred_tags))  # pyright: ignore[reportArgumentType]
wild_f1 = per_sentence_entity_f1(WILD_SENTENCE_NER_TAGS, wild_pred_tags)

summary = {
    "question": "Q3",
    "script": "src/q3_error_analysis.py",
    "model": bert_version,
    "task": "ner",
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "training_seconds": training_seconds,
    "device": str(device),
    "git_commit": GIT_SHA,
    "test_entity_micro_f1": test_micro_f1,
    "n_candidate_sentences": len(candidates),
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
        "n_errors": sum(t != p for t, p in zip(WILD_SENTENCE_NER_TAGS, wild_pred_tags)),
        "per_sentence_f1": wild_f1,
        "tokens": WILD_SENTENCE_TOKENS,
        "true_tags": WILD_SENTENCE_NER_TAGS,
        "pred_tags": wild_pred_tags,
    },
}
(RESULTS_DIR / f"summary_seed{SEED}.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(f"\nsaved summary: {RESULTS_DIR}/summary_seed{SEED}.json")

predictions = [
    {"idx": idx, "tokens": sent["tokens"], "true_tags": t, "pred_tags": p}
    for idx, (sent, t, p) in enumerate(zip(test_sentences, all_true_tags, all_pred_tags))
]
(RESULTS_DIR / f"predictions_seed{SEED}.json").write_text(
    json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
)
print(f"saved predictions: {RESULTS_DIR}/predictions_seed{SEED}.json")
