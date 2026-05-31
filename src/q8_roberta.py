#
# Q8 — NER with RoBERTa-base, 3 seeded runs.
#
# Modification of NER-BERT.py: identical pipeline, identical hyperparameters,
# identical 3-seed sweep as Q1 — but the pre-trained model is swapped from
# `bert-base-uncased` (BERT, WordPiece tokenizer) to `roberta-base`
# (RoBERTa, byte-level BPE tokenizer).
#
# Three concrete code changes vs Q1, ALL of them required for correctness:
#
#   1. Import `RobertaForTokenClassification` instead of
#      `BertForTokenClassification`.
#   2. `bert_version = 'roberta-base'` (kept the variable name to keep the
#      diff vs Q1 minimal — the value is what changed).
#   3. `AutoTokenizer.from_pretrained(bert_version, add_prefix_space=True)`.
#      This third change is the one that bites silently if omitted:
#      RoBERTa's BPE expects each token to be either at the start of the
#      sequence OR preceded by whitespace. With `is_split_into_words=True`
#      we hand the tokenizer pre-split words and it cannot tell which ones
#      should have an implicit leading space. Without `add_prefix_space=True`,
#      the first subword of every word after position 0 is tokenized
#      differently from how it would be in a contiguous sentence, and entity
#      F1 silently collapses by ~10 absolute points. See brainstorming spec
#      risk R5 — this was the one Q8 footgun the spec called out.
#
# Everything else is identical to Q1: NER task (`ner_tags`), all four metrics
# (token-level via sklearn, entity-level via seqeval), same hyperparameters,
# same seeds. Persists per-seed JSON to results/q8/.
#
# Diff anchors for the report:
#   git diff src/NER-BERT.py        src/q8_roberta.py  → full Q8 modification
#   git diff src/q1_baseline_3runs.py src/q8_roberta.py → concentrated Q8 delta
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
# ----- Q8 CHANGE: swap BertForTokenClassification → RobertaForTokenClassification
from transformers import AutoTokenizer, RobertaForTokenClassification
# ------------------------------------------------------------------------------
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
SEEDS = [42, 43, 44]                                         # 3-run sweep

# results location — separate q8 namespace
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "q8"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


# read the data files (unchanged from NER-BERT.py)
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

# build tag set and label mappings (unchanged from Q1 — still NER tags)
all_tags = sorted({tag for s in train_sentences for tag in s['ner_tags']})
label2id = {tag: i for i, tag in enumerate(all_tags)}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(all_tags)
print('Tagset size:', num_labels)
print('Tags:', all_tags)

# ----- Q8 CHANGE: model identifier + tokenizer ------------------------------
# `bert_version` kept as the variable name to keep the diff vs Q1 minimal.
# `add_prefix_space=True` is REQUIRED — see header docstring for the silent
# F1-collapse this prevents. Brainstorming spec risk R5.
bert_version = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(bert_version, add_prefix_space=True)
# ----------------------------------------------------------------------------


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


# unchanged from NER-BERT.py
def encode(sentence):
    encodings = tokenizer(
        sentence['tokens'],
        truncation=True,
        padding='max_length',
        is_split_into_words=True,
        return_tensors='pt'
    )
    labels = align_label(encodings, sentence['ner_tags'])
    return {
        'input_ids': encodings['input_ids'].squeeze(0),
        'attention_mask': encodings['attention_mask'].squeeze(0),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


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


# unchanged from NER-BERT.py
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


# unchanged from NER-BERT.py
def report_metrics(Y_actual, Y_preds, y_true_tags, y_pred_tags, split_name):
    print(f"\n=== {split_name} — Token-level metrics ===")
    print("Accuracy          : {:.3f}".format(accuracy_score(Y_actual, Y_preds)))
    print("Balanced accuracy : {:.3f}".format(balanced_accuracy_score(Y_actual, Y_preds)))
    print(f"\n=== {split_name} — Entity-level metrics (seqeval) ===")
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

# 3-seed sweep — identical to Q1's, with RobertaForTokenClassification as model
for run_index, seed in enumerate(SEEDS):
    print(f"\n{'#' * 60}\n# Q8 run {run_index + 1}/{len(SEEDS)} — seed={seed}\n{'#' * 60}")

    set_seeds(seed)

    # initialize the model afresh per seed (classifier head re-inits stochastically)
    print('initializing the model')
    # ----- Q8 CHANGE: RobertaForTokenClassification instead of BertForTokenClassification
    model = RobertaForTokenClassification.from_pretrained(
        bert_version,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    # ----------------------------------------------------------------------------
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
    print(f"=== Test (seed={seed}) — Entity-level classification report (seqeval) ===")
    print(seqeval_report(
        y_true_tags, y_pred_tags, digits=3,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    ))

    # persist this run's metrics JSON — same shape as Q1, with `model` set to roberta-base
    metrics = {
        "question": "Q8",
        "script": "src/q8_roberta.py",
        "model": bert_version,
        "task": "ner",
        "seed": seed,
        "run_index": run_index,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "training_seconds": training_seconds,
        # sklearn + seqeval stubs annotate scalar metrics as list[float] | float — at runtime
        # they return scalars. Per-line pyright ignores below are stub-quirk silencers, not bugs.
        "test": {
            "token_micro_accuracy": float(accuracy_score(Y_actual, Y_preds)),  # pyright: ignore[reportArgumentType]
            "token_macro_accuracy": float(balanced_accuracy_score(Y_actual, Y_preds)),  # pyright: ignore[reportArgumentType]
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

    # free memory between seeds (helps on MPS and on Colab T4)
    del model, optimizer, train_loader, valid_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# aggregate across seeds and print mean ± stdev
print(f"\n{'#' * 60}\n# Q8 — summary across {len(SEEDS)} seeds\n{'#' * 60}")
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
