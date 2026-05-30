# nlp-ncsr-task3

NCSR Athens — Natural Language Processing course, **Assignment 3: Sequence Labeling with
Pretrained Language Models**.

The assignment fine-tunes a pretrained language model (`bert-base-uncased`, later
`roberta-base`) for **token-level sequence labeling** on **CoNLL-2003 English** —
Named-Entity Recognition (NER), Part-of-Speech tagging (POS), and text chunking — and
contrasts encoder-only fine-tuning against zero-shot decoder-only LLMs
(`Llama-3.1-8B`, `Llama-3.3-70B` via Groq).

The final deliverable is a single PDF report covering all 10 questions of the assignment
plus the modified code artifacts.

## Compute

Hybrid policy by question:

| Question(s) | Runtime                              | Why                                            |
| ----------- | ------------------------------------ | ---------------------------------------------- |
| Q1          | **Local Apple-Silicon MPS** (script) | Baseline; ~53 min/seed on M4 Pro Max.          |
| Q3          | Local, inference-only                | Loads Q1's seed=42 predictions, no retraining. |
| Q5–Q8       | **Google Colab T4** (notebooks)      | ~5–15 min/seed; saves ~10 h of MPS time.       |
| Q9, Q10     | Local HTTP to Groq                   | No GPU needed.                                 |
| Q2, Q4      | Written-only                         | —                                              |

The local `.venv` includes `torch`, `transformers`, `scikit-learn`, `seqeval`,
`tqdm`, and `kagglehub` so scripts (Q1, Q3, Q9, Q10) and IDE diagnostics work
end-to-end without surprises — see [pyproject.toml](pyproject.toml). Colab
notebooks bring their own dependency cell.

## Layout

```
.
├── src/
│   ├── NER-BERT.py                   # instructor-provided starter — UNCHANGED
│   ├── q1_baseline_3runs.py          # Q1 — 3-seed sweep, local MPS
│   └── q5_frozen_bert.py             # Q5 — diff anchor paired with the Q5 notebook
├── notebooks/
│   ├── 00_baseline_ner_bert.ipynb    # legacy Colab port of the starter
│   └── 05_q5_frozen_bert.ipynb       # Q5 — self-contained Colab T4 runtime
├── results/
│   └── q1/seed_{42,43,44}.json       # per-seed run metrics (committed)
├── reports/                          # PDF draft sources, figures
├── pyproject.toml                    # local runtime + dev deps
└── .gitignore
```

**For Q5–Q8 (training-heavy questions)** the deliverable is **two paired
artifacts** that are committed together:

1. `notebooks/0N_qN_*.ipynb` — the Colab T4 runtime. Self-contained: no
   `from src…` imports, no clone-then-run pattern. Open via the
   Open-in-Colab badge in cell 0.
2. `src/qN_*.py` — a Python script mirroring the notebook. Exists so
   `git diff src/NER-BERT.py src/qN_*.py` is the report's clean
   "what changed" diff. The notebook is authoritative if they ever
   disagree.

For Q1 (local MPS) only the script exists. Q3 / Q9 / Q10 are scripts
in `src/`. Q2 / Q4 are written-only sections of the report.

## Local setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv venv --python 3.11
uv sync
source .venv/bin/activate
```

To preview/edit notebooks locally:

```bash
uv run jupyter lab
```

## Running on Colab

1. Open the desired notebook from this repo on Colab: `File → Open notebook → GitHub` and
   pick the notebook.
2. Set Colab secrets (key icon in the left sidebar):
   - `KAGGLE_USERNAME`, `KAGGLE_KEY` — required to download CoNLL-2003 via `kagglehub`.
   - `GROQ_API_KEY` — required for Q9/Q10 only.
3. `Runtime → Change runtime type → GPU` (T4 is enough; P100/A100 if available).
4. Run all cells. The final cell saves a metrics JSON; download it and commit it under
   `results/`.

## Dataset

[CoNLL-2003 English](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)
via the Kaggle API. The raw data is **not** committed to this repo.

## Assignment questions at a glance

| #   | Task                                                                 |
| --- | -------------------------------------------------------------------- |
| 1   | Baseline NER-BERT, 3 runs, mean ± stdev of 4 metrics + training time |
| 2   | Discussion: token-level vs entity-level metrics for NER              |
| 3   | Error analysis on one failed test sentence + a novel news sentence   |
| 4   | Explain `align_label` and the `-100` label id                        |
| 5   | Freeze BERT, train only the classifier head — repeat Q1              |
| 6   | Switch task to POS tagging — repeat Q1 (no entity-level) + Q3        |
| 7   | Switch task to text chunking — repeat Q1 + Q3                        |
| 8   | Swap model to `roberta-base` — repeat Q1                             |
| 9   | Zero-shot NER on 200 test sentences via Groq `llama-3.1-8b-instant`  |
| 10  | Same as Q9 with `llama-3.3-70b-versatile`; cross-compare              |

## License

Coursework. Not for redistribution.
