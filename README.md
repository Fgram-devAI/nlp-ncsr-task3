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
│   ├── q1_baseline_3runs.py          # Q1 — 3-seed NER sweep, local MPS
│   ├── q5_frozen_bert.py             # Q5 — frozen BERT, diff anchor (paired w/ notebook)
│   ├── q6_pos_tagging.py             # Q6 — POS tagging, diff anchor (paired w/ notebook)
│   ├── q7_chunking.py                # Q7 — text chunking, diff anchor (paired w/ notebook)
│   └── q8_roberta.py                 # Q8 — RoBERTa swap, diff anchor (paired w/ notebook)
├── notebooks/
│   ├── 00_baseline_ner_bert.ipynb    # legacy Colab port of the starter
│   ├── 05_q5_frozen_bert.ipynb       # Q5 — self-contained Colab T4 runtime
│   ├── 06_q6_pos_tagging.ipynb       # Q6 — self-contained Colab T4 runtime
│   ├── 07_q7_chunking.ipynb          # Q7 — self-contained Colab T4 runtime
│   └── 08_q8_roberta.ipynb           # Q8 — self-contained Colab T4 runtime
├── results/
│   ├── q1/seed_{42,43,44}.json       # Q1 NER baseline metrics (committed)
│   ├── q5/seed_{42,43,44}.json       # Q5 frozen-BERT metrics (committed)
│   ├── q6/seed_{42,43,44}.json       # Q6 POS metrics (committed)
│   ├── q7/seed_{42,43,44}.json       # Q7 chunking metrics (committed)
│   └── q8/seed_{42,43,44}.json       # Q8 RoBERTa metrics (committed)
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

## Colab notebooks

One-click launchers for every runnable notebook. The badges resolve against `main`,
so a freshly cloned reader can open any of them directly.

| Question | Topic                              | Notebook                                                       | Launch                                                                                                                                                                            |
| -------- | ---------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1 (baseline) | NER fine-tune (legacy port)   | [`notebooks/00_baseline_ner_bert.ipynb`](notebooks/00_baseline_ner_bert.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/nlp-ncsr-task3/blob/main/notebooks/00_baseline_ner_bert.ipynb) |
| Q5       | Frozen BERT, head-only training    | [`notebooks/05_q5_frozen_bert.ipynb`](notebooks/05_q5_frozen_bert.ipynb)       | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/nlp-ncsr-task3/blob/main/notebooks/05_q5_frozen_bert.ipynb)         |
| Q6       | POS tagging (full fine-tune)       | [`notebooks/06_q6_pos_tagging.ipynb`](notebooks/06_q6_pos_tagging.ipynb)       | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/nlp-ncsr-task3/blob/main/notebooks/06_q6_pos_tagging.ipynb)         |
| Q7       | Text chunking (full fine-tune)     | [`notebooks/07_q7_chunking.ipynb`](notebooks/07_q7_chunking.ipynb)             | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/nlp-ncsr-task3/blob/main/notebooks/07_q7_chunking.ipynb)             |
| Q8       | NER with RoBERTa-base              | [`notebooks/08_q8_roberta.ipynb`](notebooks/08_q8_roberta.ipynb)               | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/nlp-ncsr-task3/blob/main/notebooks/08_q8_roberta.ipynb)               |

Q3, Q9, Q10 — added as they are implemented. Q1's authoritative
implementation is `src/q1_baseline_3runs.py` (runs locally on MPS); the
`00_baseline_ner_bert.ipynb` notebook is kept as a Colab fallback.

## Running on Colab

1. Open the notebook you want via the table above (or `File → Open notebook → GitHub`
   in Colab and pick from the `Fgram-devAI/nlp-ncsr-task3` repo).
2. Set Colab secrets (key icon in the left sidebar):
   - `KAGGLE_USERNAME`, `KAGGLE_KEY` — required to download CoNLL-2003 via `kagglehub`.
   - `GROQ_API_KEY` — required for Q9/Q10 only.
3. `Runtime → Change runtime type → GPU` (T4 is enough; P100/A100 if available).
4. **Optional but recommended for long sweeps** — run the "Persist results to
   Google Drive" cell (provided in each Q5+ notebook). It mounts Drive and
   redirects `RESULTS_DIR` so per-seed JSONs survive runtime kills (idle
   timeouts, browser close). Without it, results land in the ephemeral
   `/content/results/qN/` and must be downloaded before the runtime dies.
5. Run all cells. If you skipped the Drive cell, the final cell triggers
   browser downloads of the per-seed JSONs — drop them into the matching
   `results/qN/` folder locally and commit.

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
