# MPE 25/26 — RAG Benchmark for Technical Q&A

Compare **BM25, dense and hybrid retrieval** against a Gemini baseline without context. The dataset contains **70 multiple-choice questions** about *REFRAG: Rethinking RAG based Decoding*.

Developed for *Modelización de Problemas de la Empresa* (2025/26) by **Shadi Fedriani Abdallah and Alejandro Domínguez López**. The original [report and results](docs/Report%20%26%20Results.pdf) are in Spanish. This project evaluates retrieval over the paper; it does not implement REFRAG's decoding architecture.

## Setup

Use Python 3.11. Create a virtual environment from the repository root:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

All dependencies are listed in one file. Preparation also works without installing any packages.

## Run

```bash
# Validate the questions, clean the paper and create chunks. No API calls.
python src/main.py

# Retrieve evidence locally using BM25. No API calls.
python src/main.py --mode bm25

# Run all four methods on three questions, including Gemini calls.
python src/main.py --mode full --limit 3

# See all options.
python src/main.py --help
```

Before using `full`, copy `.env.example` to `.env` and set `GEMINI_API_KEY` and `GEMINI_MODEL`. The original study used `gemini-2.5-flash-lite`; choose a model available to your account. Changing the model creates a new experiment. `.env` is ignored by Git.

Full mode downloads `all-mpnet-base-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` on first use. It sends questions, options and retrieved context to Gemini at temperature zero. Gold answers and reference passages are used only for evaluation.

Each full run makes **4 × questions × rounds** Gemini requests before retries: 12 for the example above, or 280 for all 70 questions in one round. The default pause is five seconds after each response. Usage is subject to your account's charges and quotas.

Useful options include `--rounds`, `--alpha`, `--threshold`, `--chunk-size`, `--overlap`, `--sleep`, `--model`, `--data`, `--paper` and `--output`.

## Read the code

| File | What to look for |
| --- | --- |
| `src/main.py` | `main()` prepares the inputs, saves them and selects BM25 or the full experiment. Separate functions handle settings and each mode. |
| `src/core.py` | Validation, text cleaning, chunking, answer parsing and reference coverage. Uses only Python's standard library. |
| `src/experiment.py` | `Retriever` retrieves chunks, combines scores and reranks candidates. The evaluation functions calculate each metric; `plot_results()` draws the charts. |
| `tests/` | Checks for text processing, retrieval, evaluation and output files. |
| `data/ModelizaciónEmpresaUCMData.json` | The 70 questions, four options, correct answers and reference passages. |
| `docs/Paper.txt` | The paper text required by the experiment. |
| `docs/Report & Results.pdf` | The original academic report. |

The processing steps are:

1. Validate that every question has four options, a valid answer and a reference.
2. Clean extracted lines, remove numeric-only lines and remove the `References`–`Appendix` block. This cleaning rule is specific to the supplied paper.
3. Pack paragraphs and sentences into chunks of at most **250 words**, with up to **30 words of overlap**. Split long sentences by words when needed.
4. Retrieve relevant chunks using BM25, embeddings in Chroma, or a weighted combination of both. Dense and hybrid answer contexts are then reranked by a cross-encoder.
5. Evaluate answers and retrieved evidence, and save tables and plots.

Hybrid retrieval calculates `alpha * dense_score + (1 - alpha) * bm25_score`, with `alpha=0.5` by default. Dense similarity is `1 - cosine_distance`. Each candidate list is min–max normalized separately. Missing candidates contribute zero; equal scores normalize to all ones.

### Retrieval settings

These settings and the method-specific prompts are retained from the experiment:

| Evaluation | BM25 | Dense | Hybrid |
| --- | --- | --- | --- |
| Answer accuracy | Top 3 | Top 15, rerank to 3 | Combine dense 50 + BM25 50, keep 50, rerank to 3 |
| Reference coverage | Top 5 | Top 20, rerank to 5 | Combine dense 20 + BM25 50, keep 15, rerank to 5 |
| Recall@K | Top K | Top K from dense 50, no reranking | Top K after combining both top 50 lists, no reranking |
| Context size | Top 3 | Top 15, rerank to 5 | Combine dense 15 + BM25 50, keep 15, rerank to 5 |

### Metrics

- **Answer accuracy:** correct letters divided by questions. Abstentions and unparseable responses count as incorrect.
- **Reference coverage:** fraction of reference-word occurrences present in the retrieved context. A hit requires at least `--threshold` coverage (default 0.75).
- **Recall@K:** fraction of questions with a reference hit in the first K chunks. This is a lexical proxy, not standard recall over annotated document IDs.
- **Context size:** mean characters divided by four, a rough token estimate.

Word coverage does not verify factual correctness or generated citations. Repeated words and longer contexts can inflate it. Prompts and retrieval budgets differ by method, so the comparison is not controlled for those factors. The token estimate does not measure billed usage or latency.

## Outputs and tests

Each run creates a new `outputs/<UTC timestamp>/` directory. Use `--output PATH` to choose another location; non-empty directories are rejected.

| Mode | Files |
| --- | --- |
| All | `run.json`, `paper_clean.txt`, `chunks.json` |
| `bm25` | Also `bm25_retrieval.json`, `bm25_summary.json` |
| `full` | Also `accuracy.csv`, `source.csv`, `recall.csv`, `overhead.csv` and four PNG charts |

`run.json` records settings, input hashes, Python/package versions and run status. It does not include the API key. Failed API calls stop the run after retries; failed runs do not resume, and individual Gemini responses are not saved.

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests, preparation and a BM25 smoke check. Tests use fake Gemini, dense retrieval and reranking implementations, so they do not make API calls or download model weights.

The supplied inputs produce **69 chunks**. BM25 gives **56/70 reference hits (80%)** at top 3 and threshold 0.75. This measures retrieval coverage, not answer accuracy.

## Original results and credits

The original report describes approximate answer accuracies of **73% baseline, 93% BM25 and 96% dense retrieval**. These are historical results, not newly reproduced measurements. The supplied archive had no prediction log or dependency lockfile. Earlier fixes to chunking, answer parsing and SDK configuration may change the results. The small single-paper dataset also limits generalization.

The paper is by Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava and Vijai Mohan. The supplied text identifies arXiv:2509.01092v2, dated 12 October 2025. The dataset, paper text and report are preserved unchanged.

No repository-wide open-source licence has been assigned. The supplied archive did not document the dataset's authorship/licence or the paper extraction's redistribution terms; those rights are separate from the code.
