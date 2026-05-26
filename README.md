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

All training and large-model inference runs on **Google Colab** (T4/P100/A100 GPU).
The local Apple-Silicon machine is used only for code editing, light data inspection,
notebook authoring, and writing the report. The local `.venv` deliberately omits
`torch` / `transformers` — see [pyproject.toml](pyproject.toml).

## Layout

```
.
├── src/
│   └── NER-BERT.py                   # instructor-provided starter
├── notebooks/
│   └── 00_baseline_ner_bert.ipynb    # Q1 baseline (faithful Colab port)
├── results/                          # per-run JSON metrics (committed)
├── reports/                          # PDF draft sources, figures
├── pyproject.toml                    # local-only dev deps (no torch)
└── .gitignore
```

Per-question notebooks (Q1 three-run wrapper, Q5 frozen-BERT, Q6 POS, Q7 chunking,
Q8 RoBERTa, Q9 Llama-3.1-8B zero-shot, Q10 Llama-3.3-70B zero-shot) are produced during
the planning/implementation phase.

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
