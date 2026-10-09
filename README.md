# Retrieval Benchmark

How much does retrieval help a language model answer questions about a technical
paper? This project compares BM25, dense and hybrid retrieval with a Gemini
baseline across 70 multiple-choice questions about *REFRAG: Rethinking RAG based
Decoding*.

Developed by **Shadi Fedriani Abdallah and Alejandro Domínguez López**. Finalist in
the 2025/26 *Modelización de Problemas de la Empresa* contest.

The [original report](docs/Report%20%26%20Results.pdf) describes approximate answer
accuracies of 73% for the baseline, 93% for BM25 and 96% for dense retrieval. Those
are the study's historical results; changes to chunking, answer parsing and model
availability can affect new runs. The project retrieves evidence from the REFRAG
paper; it does not implement its decoding architecture.

## Run locally

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or
`.venv\Scripts\Activate.ps1` in Windows PowerShell.

Preparation validates the questions and chunks the paper using only the standard
library. For local BM25 retrieval and tests, install the base dependencies:

```bash
python src/main.py
python -m pip install -r requirements.txt
python src/main.py --mode bm25
python -m unittest discover -s tests -v
```

For the full experiment, install the model dependencies, copy `.env.example` to
`.env`, and set `GEMINI_API_KEY` and `GEMINI_MODEL`:

```bash
python -m pip install -r requirements-full.txt
python src/main.py --mode full --limit 3
```

The study used `gemini-2.5-flash-lite`; use a model available to your account.
A different model means a different experiment. Full mode downloads MPNet and
MiniLM weights on first use and makes `4 × questions × rounds` Gemini requests
before retries: 12 in the example above, or 280 for one round of all 70 questions.
Requests use your account's quota and may incur charges. Questions, options and
retrieved passages are sent to Gemini; gold answers stay in the evaluation code.

`python src/main.py --help` lists the options, including `--rounds`, `--model`,
`--alpha`, `--threshold`, `--chunk-size`, `--overlap`, `--sleep` and `--output`.
The default pause between requests is five seconds.

## Method

The supplied paper is cleaned and split into chunks of at most 250 words, with
up to 30 words of overlap. The cleaning rule removes numeric-only lines and the
References–Appendix block. It is specific to this paper.

BM25 uses token overlap. Dense retrieval uses `all-mpnet-base-v2` embeddings in
Chroma with cosine distance. Hybrid retrieval adds separately min–max normalized
scores as `alpha * dense + (1 - alpha) * bm25`, with `alpha=0.5`. Missing candidates
contribute zero; tied scores normalize to one. Dense and hybrid answer contexts
are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

The experiment keeps the original method-specific prompts and retrieval budgets:

| Evaluation | BM25 | Dense | Hybrid |
| --- | --- | --- | --- |
| Answer accuracy | Top 3 | Top 15, rerank to 3 | Dense 50 + BM25 50, keep 50, rerank to 3 |
| Reference coverage | Top 5 | Top 20, rerank to 5 | Dense 20 + BM25 50, keep 15, rerank to 5 |
| Recall@K | Top K | Top K from dense 50 | Top K from dense 50 + BM25 50 |
| Context size | Top 3 | Top 15, rerank to 5 | Dense 15 + BM25 50, keep 15, rerank to 5 |

Recall@K does not use reranking. Answer accuracy counts abstentions and ambiguous
answers as incorrect. Reference coverage measures how many reference-word
occurrences appear in the retrieved text; the default hit threshold is 75%.
Recall@K is the proportion of questions with such a hit in the first K chunks.
Context size is estimated as characters divided by four.

These coverage measures are lexical proxies, not checks of factual correctness
or standard document-level recall. Repeated words and longer contexts can inflate
them. Prompts and context budgets differ across methods, and the single-paper
sample is small. Context size does not measure billed tokens or latency.

With the supplied inputs and defaults, preparation produces 69 chunks. BM25
retrieval gives 56/70 reference hits (80%) at top 3 and threshold 0.75.

## Saved results

Each run writes to a new `outputs/<UTC timestamp>/` folder. `--output PATH`
chooses another location; non-empty folders are rejected.

| Mode | Output |
| --- | --- |
| All | `run.json`, `paper_clean.txt`, `chunks.json` |
| BM25 | Also `bm25_retrieval.json` and `bm25_summary.json` |
| Full | Also `predictions.jsonl`, four metric CSVs and four charts |

`run.json` records settings, input hashes, package versions and run status.
`predictions.jsonl` saves each completed prompt, response, parsed answer and label
as it arrives. Completed records and metric tables survive a later failure;
runs do not resume automatically. API keys are not written to the output.

`src/core.py` contains text processing and scoring, `src/experiment.py` retrieval
and evaluation, and `src/main.py` the command-line workflow. Tests run offline
with fake model responses. GitHub Actions checks both supported Python versions
without downloading model weights.

## Source material

The paper is by Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali
Shrivastava and Vijai Mohan (arXiv:2509.01092v2). The supplied paper text, question
set and report are retained as research inputs. The original archive did not
include prediction logs or a dependency lockfile.

No repository-wide open-source licence has been assigned. Rights to the paper,
dataset and report are separate from the code; their redistribution terms are
not documented in the original archive.

*The report is written in Spanish.*
