#
# Q5 — NER-BERT with the pre-trained BERT encoder FROZEN, 3 seeded runs.
#
# Modification of NER-BERT.py: same model, same hyperparameters, same eval
# protocol, same 3-seed sweep as Q1 — but inside the seed loop the entire
# `model.bert` parameter tree (embeddings + 12 transformer layers; the pooler
# is `None` on `BertForTokenClassification`, so there is no pooler to freeze)
# is frozen via `requires_grad = False`, so only the classifier head trains.
# The optimizer only receives trainable parameters (no momentum buffers for
# frozen weights). Reports trainable / frozen / total parameter counts — the
# specific Q5 deliverable in the assignment preliminaries — and hard-asserts
# that the only remaining trainable tensors are `classifier.{weight, bias}`
# so an accidental under-freeze cannot pass silently.
#
# Diff anchor for the report: `git diff src/NER-BERT.py src/q5_frozen_bert.py`
# (also serves as the script-to-script diff vs `src/q1_baseline_3runs.py`).
#
# The notebook `notebooks/05_q5_frozen_bert.ipynb` is the actual Colab T4
# runtime; this script mirrors it for diff visibility. See CLAUDE.md §3 §5.
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

# hyper-parameters (identical to Q1 for a fair comparison).
# Note on LR: 1e-5 is held over from Q1 deliberately — the assignment asks to
# "repeat Q1" with frozen BERT, so changing the learning rate would muddy the
# comparison. With only ~7k trainable parameters, 3 epochs at LR=1e-5 likely
# under-trains the head; the report should note this trade-off when discussing
# the (expected) F1 gap vs Q1.
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-5
SEEDS = [42, 43, 44]                                         # 3-run sweep

# results location — separate q5 namespace
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "q5"
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

# build tag set and label mappings (unchanged from NER-BERT.py)
all_tags = sorted({tag for s in train_sentences for tag in s['ner_tags']})
label2id = {tag: i for i, tag in enumerate(all_tags)}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(all_tags)
print('Tagset size:', num_labels)
print('Tags:', all_tags)

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

# 3-seed sweep — identical structure to Q1, with the Q5 freeze step inserted
for run_index, seed in enumerate(SEEDS):
    print(f"\n{'#' * 60}\n# Q5 run {run_index + 1}/{len(SEEDS)} — seed={seed}\n{'#' * 60}")

    set_seeds(seed)

    # initialize the model afresh per seed (classifier head re-inits stochastically)
    print('initializing the model')
    model = BertForTokenClassification.from_pretrained(
        bert_version,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ----- Q5 CHANGE: freeze every BERT encoder parameter ----------------------
    # The assignment asks to freeze "weights related to the pre-trained BERT
    # language model" — in this `BertForTokenClassification` instance, that's
    # `model.bert` (the embeddings + 12 transformer layers). Note that
    # `model.bert.pooler` is NOT instantiated for the token-classification head
    # (it shows up as UNEXPECTED in the LOAD REPORT for the checkpoint and is
    # set to `None` on the model) — there is no pooler to freeze. Only the
    # `model.classifier` head still receives gradient. The classifier
    # (`Linear(768, num_labels)`) is newly initialised by `from_pretrained`
    # (MISSING keys in the LOAD REPORT) and is what training now learns to drive.
    for p in model.bert.parameters():
        p.requires_grad = False

    # Param count report — explicit Q5 deliverable, with hard cross-checks so
    # an accidental over- or under-freeze can't slip past.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    # Cross-check 1: total reconstructed from sums must equal the independent sum.
    independent_total = sum(p.numel() for p in model.parameters())
    assert total_params == independent_total, (
        f"param count mismatch: trainable+frozen={total_params} but sum={independent_total}"
    )
    # Cross-check 2: the pooler must literally not exist on this model class.
    assert model.bert.pooler is None, (
        f"unexpected pooler on BertForTokenClassification: {type(model.bert.pooler).__name__}"
    )
    # Cross-check 3: the ONLY trainable tensors should be the two classifier
    # head tensors. Any other name here means we under-froze.
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names == {"classifier.weight", "classifier.bias"}, (
        f"unexpected trainable parameter set: {trainable_names}"
    )
    # Cross-check 4: head-only count must equal num_labels * hidden_dim + num_labels.
    expected_trainable = num_labels * model.config.hidden_size + num_labels
    assert trainable_params == expected_trainable, (
        f"trainable_params={trainable_params}, expected {expected_trainable} "
        f"(= {num_labels} * {model.config.hidden_size} + {num_labels})"
    )

    print(f"Total params      : {total_params:>12,}")
    print(f"Frozen (BERT)     : {frozen_params:>12,}  ({frozen_params / total_params:.4%})")
    print(f"Trainable (head)  : {trainable_params:>12,}  ({trainable_params / total_params:.4%})")
    # --------------------------------------------------------------------------

    # `from_pretrained` returns a union of model classes; pyright loses narrowing on .to(device).
    model = model.to(device)  # pyright: ignore[reportArgumentType]
    # ----- Q5 CHANGE: optimizer only sees trainable params --------------------
    # AdamW would honour requires_grad=False at update-time, but still allocates
    # momentum / variance buffers for every param it sees. Filtering at construct
    # time saves ~110M × 2 × 4 bytes ≈ 850 MB of optimizer state vs Q1.
    optimizer = optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )
    # --------------------------------------------------------------------------

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

    # persist this run's metrics JSON — Q5 shape EXTENDS Q1 with frozen-encoder keys
    metrics = {
        "question": "Q5",
        "script": "src/q5_frozen_bert.py",
        "model": bert_version,
        "task": "ner",
        "frozen_encoder": True,
        "seed": seed,
        "run_index": run_index,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "training_seconds": training_seconds,
        "total_params": total_params,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
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


# aggregate across seeds and print mean ± stdev — same layout as Q1, plus the
# parameter-count summary (identical across seeds, printed once).
print(f"\n{'#' * 60}\n# Q5 — summary across {len(SEEDS)} seeds\n{'#' * 60}")
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

# parameter counts are identical across seeds — print once at the end
ref = PER_SEED_RESULTS[0]
print(f"\nParameter counts (same across all seeds):")
print(f"  total       = {ref['total_params']:>12,}")
print(f"  frozen      = {ref['frozen_params']:>12,}  ({ref['frozen_params'] / ref['total_params']:.4%})")
print(f"  trainable   = {ref['trainable_params']:>12,}  ({ref['trainable_params'] / ref['total_params']:.4%})")
