# MPE 25/26 — RAG Benchmark for Technical Q&A

An academic comparison of **BM25, dense and hybrid retrieval**, plus a Gemini baseline without retrieved context. The dataset contains **70 multiple-choice questions** about *REFRAG: Rethinking RAG based Decoding*.

Developed for *Modelización de Problemas de la Empresa* (2025/26) by **Shadi Fedriani Abdallah and Alejandro Domínguez López**. The original [report and results](docs/Report%20%26%20Results.pdf) are in Spanish. The dataset, paper and this README are in English.

This project evaluates retrieval over the REFRAG paper; it does not implement REFRAG's compressed decoding architecture.

## Start here

Clone the repository and prepare the dataset using only Python:

```bash
git clone https://github.com/alexdmngz/rag-mpe-benchmark.git
cd rag-mpe-benchmark
python src/main.py
```

The default `prepare` mode validates the questions, cleans `docs/Paper.txt` and creates chunks. **No API key, downloads or third-party packages are required.** The supplied archive did not contain `Paper.pdf`, so the code uses the included text extraction directly.

For tests and local BM25 retrieval, use Python 3.11 and a virtual environment:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-check.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src/main.py --mode bm25
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-check.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/main.py --mode bm25
```

BM25 runs locally after installation. It measures reference coverage in the top three chunks, not LLM answer accuracy.

## Full experiment with Gemini

Use your environment's Python executable for the commands below:

```bash
python -m pip install -r requirements.txt
python src/main.py --mode full --limit 3 --rounds 1
```

Before running, copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux). Set `GEMINI_API_KEY` and `GEMINI_MODEL`. Never commit your key; `.env` is ignored by Git.

The supplied study used `gemini-2.5-flash-lite`. Set a model available to your account or use `--model MODEL_ID`; [model availability can change](https://ai.google.dev/gemini-api/docs/deprecations). Changing models creates a new experiment. Generation uses temperature zero through the [Google Gen AI SDK](https://github.com/googleapis/python-genai).

Full mode downloads `all-mpnet-base-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` on first use, creates an in-memory Chroma index, and sends questions/options plus retrieved context to Gemini. Gold answers and reference passages are used for evaluation, not inserted into prompts or used to rank chunks.

There are **4 × questions × rounds** Gemini calls before retries: 12 for the small example, 280 for all questions in one round. The default five-second pause adds at least 23 minutes 20 seconds to a 280-call run, before generation and retrieval time. Charges and quotas depend on your account.

```bash
python src/main.py --mode full --rounds 2
python src/main.py --help
```

Options include `--limit`, `--alpha`, `--threshold`, `--chunk-size`, `--overlap`, `--sleep`, input paths and the output directory. Use `--limit` for smoke checks, not as a representative evaluation.

## Code map

| File | Responsibility |
| --- | --- |
| `src/main.py` | Command-line options, input/output paths, credentials and run metadata |
| `src/core.py` | Dataset validation, text cleaning, chunking, answer parsing and lexical metrics |
| `src/experiment.py` | One shared retriever, evaluation loops and plots |
| `notebooks/Code.ipynb` | Optional notebook using the same entry point |
| `data/ModelizaciónEmpresaUCMData.json` | 70 records with `question`, A–D `answers`, `correct_answer` and `paper_reference` |
| `docs/` | Original paper text and academic report |
| `tests/` | Local regression and integration checks |

`requirements-check.txt` contains the lightweight dependencies for tests and BM25. `requirements.txt` includes those plus the full model stack. Versions use compatibility ranges, not a historical lockfile. To use the notebook, install `jupyterlab` in the same environment and run `python -m jupyter lab notebooks/Code.ipynb`; API calls remain commented out until explicitly enabled.

## How the experiment works

1. **Validate** the dataset: four options, valid answer letters, non-empty references and no duplicate questions.
2. **Clean** the paper: join extracted lines, remove numeric-only lines, and remove the case-sensitive `References`–`Appendix` block. Without a closing marker, remove the rest of the reference section. This is a document-specific heuristic.
3. **Chunk** paragraphs and sentences into at most 250 **words**, with up to 30 words of overlap. Long spans fall back to word splitting; boundaries are not learned from embeddings.
4. **Retrieve** with BM25, sentence embeddings with cosine distance in Chroma, or their weighted combination. Dense/hybrid answer contexts are reranked by a cross-encoder.
5. **Evaluate** answer letters and retrieved evidence, then save tables and plots.

Hybrid fusion uses `alpha * normalized_dense + (1 - alpha) * normalized_bm25`, with `alpha=0.5` by default. Dense scores are `1 - cosine_distance`. Each candidate list is min–max normalized separately; missing candidates contribute zero and constant score lists normalize to all ones, matching the original convention.

### Retrieval budgets

The original settings are retained across the refactor. Different evaluation stages use different budgets:

| Stage | BM25 | Dense | Hybrid |
| --- | --- | --- | --- |
| Answer accuracy | Top 3 | Top 15, rerank to 3 | Fuse dense 50 + BM25 50, keep 50, rerank to 3 |
| Reference coverage | Top 5 | Top 20, rerank to 5 | Fuse dense 20 + BM25 50, keep 15, rerank to 5 |
| Recall@K proxy | Top K | First K of dense 50; no reranking | First K after fusing top 50 lists; no reranking |
| Context size | Top 3 | Top 15, rerank to 5 | Fuse dense 15 + BM25 50, keep 15, rerank to 5 |

Requests are capped at the number of chunks. Method-specific prompts are preserved, so this is not a controlled comparison with identical prompts and budgets.

### Metrics and limits

| Metric | Meaning |
| --- | --- |
| Answer accuracy | Correct letters / questions × 100; abstentions and unparseable responses are incorrect |
| `source_accuracy` | Percentage of questions whose retrieved context contains at least 75% of reference-word occurrences |
| `avg_overlap_score` | Mean reference-word coverage × 100; word order and punctuation are ignored |
| Recall@K | Question-level hit rate at the same coverage threshold within the concatenated top K chunks |
| Context overhead | Mean context characters / 4: a rough size estimate |

Coverage is a **lexical proxy**, not verification of a generated citation or factual correctness. Common/repeated words and longer contexts can inflate it. Recall@K is not standard recall over annotated document IDs. The overhead estimate is not a tokenizer count, billed usage, irrelevant-token count or latency measurement. Its five-chunk dense/hybrid budget differs from BM25's three-chunk budget.

## Outputs and checks

Each run creates `outputs/<UTC timestamp>/`, or a new directory supplied with `--output`. A non-empty destination is rejected. Generated runs are ignored by Git.

| Mode | Output files |
| --- | --- |
| All | `run.json`, `paper_clean.txt`, `chunks.json` |
| `bm25` | Also `bm25_retrieval.json`, `bm25_summary.json` |
| `full` | Also `accuracy.csv`, `source.csv`, `recall.csv`, `overhead.csv` and four PNG charts |

`run.json` records settings, Python/package versions, input hashes and run status, never the API key. API errors stop the run after retries rather than silently becoming wrong answers. Failed runs do not resume; per-question Gemini responses are not persisted. CSVs are the result tables, so a duplicate auto-generated conclusions document is no longer produced.

[GitHub Actions](https://github.com/alexdmngz/rag-mpe-benchmark/actions) runs tests, preparation and a BM25 smoke check on Python 3.11 without Gemini calls or neural model downloads. The supplied inputs produce 69 chunks; the local full-dataset BM25 check gives 56/70 reference hits (80%) at top 3 and threshold 0.75. This is retrieval coverage, not answer accuracy. Tests use doubles for Gemini, dense retrieval and reranking; live integration with those services/models still needs a configured full run.

## Historical results and attribution

The unmodified report describes approximate answer accuracies of **73% baseline, 93% BM25 and 96% dense retrieval**. These are historical claims, not newly reproduced results. The original ZIP had no prediction log, complete result tables or dependency lockfile. Earlier preparation fixes to chunking, answer parsing, threshold propagation and SDK configuration can change outcomes from that report. The small single-paper dataset and absence of uncertainty analysis also limit conclusions; high baseline accuracy alone cannot establish whether a model had seen the paper.

The project credits **both Shadi Fedriani Abdallah and Alejandro Domínguez López**. The paper is by Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava and Vijai Mohan; the supplied extraction identifies arXiv:2509.01092v2, dated 12 October 2025. The text, dataset and report retain their supplied contents.

No repository-wide open-source licence has been assigned. The dataset's original authorship/licence and the full paper extraction's redistribution terms were not documented in the supplied ZIP; their rights are separate from the project's code. Preserve those distinctions when reusing or licensing the material.
